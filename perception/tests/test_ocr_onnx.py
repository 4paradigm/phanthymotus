"""CPU contract tests; real weights are checked by the isolated inference probe."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from plugins.ocr import _adapter_options, _build_ocr_adapter
from plugins.ocr_runtime import _OnnxCPUModelSession, RapidOCRAdapter
from utils import model_downloader as md


def test_backend_selection_and_download(monkeypatch):
    calls = []
    monkeypatch.setattr(md, "ensure_ocr_model", lambda path: calls.append(("trt", path)))
    monkeypatch.setattr(md, "ensure_ocr_onnx_model", lambda path, cls: calls.append(("cpu", path, cls)))
    monkeypatch.setattr("plugins.ocr.RapidOCRAdapter", lambda **options: options)
    assert _build_ocr_adapter({})["backend"] == "tensorrt"
    assert calls[-1][0] == "trt"
    options = _build_ocr_adapter({"backend": "onnx-cpu", "use_angle_cls": False})
    assert calls[-1] == ("cpu", "/models/ocr/ppocrv6-small-onnx", False)
    assert options["backend"] == "onnx-cpu"
    with pytest.raises(ValueError, match="unsupported OCR backend"):
        _adapter_options({"backend": "auto"})
    with pytest.raises(FileNotFoundError, match="onnx-cpu"):
        RapidOCRAdapter("/nonexistent/ocr", backend="onnx-cpu")


def test_cpu_session_normalization_shapes_and_close(monkeypatch):
    class FakeSession:
        def __init__(self, path, *, sess_options, providers):
            assert providers == ["CPUExecutionProvider"]
            assert sess_options.intra_op_num_threads == 2
        def get_inputs(self):
            return [SimpleNamespace(name="x", type="tensor(float)", shape=[1, 3, 48, "width"])]
        def get_outputs(self):
            return [SimpleNamespace(name="y")]
        def run(self, outputs, feeds):
            return [feeds["x"]]

    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(
        InferenceSession=FakeSession, SessionOptions=SimpleNamespace))
    session = _OnnxCPUModelSession(Path("rec/inference.onnx"), device_id=0,
                                   mean=(127.5,) * 3, normal=(1 / 127.5,) * 3)
    assert session.optimization_shape == (1, 3, 48, 320)
    assert session.max_batch_size(48, 320) == 1
    image = np.zeros((48, 320, 3), dtype=np.uint8)
    image[:, :, 2] = 255
    output = session.run_uint8(image, (1, 3, 48, 320))
    assert output.dtype == np.float32 and output.flags.c_contiguous
    np.testing.assert_allclose(output[0, 0], -1)
    np.testing.assert_allclose(output[0, 2], 1)
    with pytest.raises(ValueError, match="does not match"):
        session.fit_input_image_shape(32, 320)
    with pytest.raises(ValueError, match="image/shape mismatch"):
        session.run_uint8(image, (1, 3, 48, 640))
    session.close()
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        session.run_uint8(image, (1, 3, 48, 320))


def test_cpu_downloader_checks_path_and_pins(monkeypatch):
    calls = []
    monkeypatch.setattr(md, "ensure_verified_bundle", lambda *args: calls.append(args))
    with pytest.raises(ValueError):
        md.ensure_ocr_onnx_model("/tmp/not-models")
    md.ensure_ocr_onnx_model("/models/ocr/cpu", use_angle_cls=False)
    assert [call[0] for call in calls] == ["ocr/onnx/det", "ocr/onnx/rec", "ocr/onnx/keys"]
    for _, _, _, files in calls:
        assert all(len(meta["sha256"]) == 64 and meta["size"] > 0 for meta in files.values())


@pytest.mark.parametrize("backend", ["tensorrt", "onnx-cpu"])
def test_pipeline_selects_sessions_and_preserves_trt_normalization(monkeypatch, backend):
    from plugins import ocr_runtime as runtime

    calls = []
    class FakeSession:
        def __init__(self, path, **options):
            calls.append((path, options))
        def close(self):
            pass

    for module, attributes in {
        "rapidocr.ch_ppocr_det.utils": {"DBPostProcess": lambda **kw: None},
        "rapidocr.ch_ppocr_rec.utils": {"CTCLabelDecode": lambda **kw: None},
        "rapidocr.utils.process_img": {"get_rotate_crop_image": lambda *a: None},
    }.items():
        monkeypatch.setitem(sys.modules, module, SimpleNamespace(**attributes))
    selected = "_OnnxCPUModelSession" if backend == "onnx-cpu" else "_TensorRTModelSession"
    monkeypatch.setattr(runtime, selected, FakeSession)
    pipeline = runtime._OCRPipeline(Path("/models/ocr"), device_id=0,
                                    max_side_len=1600, use_angle_cls=True, backend=backend)
    assert len(calls) == 3
    if backend == "tensorrt":
        assert [p.name for p, _ in calls] == ["det.engine", "rec.engine", "cls.engine"]
        assert calls[0][1]["mean"] == (127.5,) * 3
        assert calls[0][1]["normal"] == (1 / 127.5,) * 3
    else:
        assert [p.parent.name for p, _ in calls] == ["det", "rec", "cls"]
        np.testing.assert_allclose(calls[0][1]["mean"], np.array([0.485, 0.456, 0.406]) * 255)
    assert calls[1][1]["mean"] == calls[2][1]["mean"] == (127.5,) * 3
    pipeline.close()
