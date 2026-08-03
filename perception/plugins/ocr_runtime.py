from __future__ import annotations

import logging
import math
import tempfile
import threading
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from plugins.ocr_tiled_strategy import (
    AdaptiveTiledOCRStrategy,
    LargeImageStrategyConfig,
    decode_vips_overview,
    jpeg_dimensions,
)

from plugins.ocr_preprocess import preprocess_for_ocr

_log = logging.getLogger(__name__)


ORT_MODEL_FILES = ("det.onnx", "rec.onnx", "cls.onnx", "keys.txt")
MNN_MODEL_FILES = ("det.mnn", "rec.mnn", "keys.txt")
TENSORRT_MODEL_FILES = ("det.engine", "rec.engine", "keys.txt")
_OCR_MEAN = (127.5, 127.5, 127.5)
_OCR_NORMAL = (1 / 127.5, 1 / 127.5, 1 / 127.5)
DEFAULT_MAX_SIDE_LEN = 1600
DEFAULT_REC_MIN_SCORE = 0.9
DEFAULT_DET_THRESH = 0.3
DEFAULT_DET_BOX_THRESH = 0.5
DEFAULT_DET_UNCLIP_RATIO = 0.7


@dataclass(frozen=True)
class ImageHeader:
    format: str
    width: int
    height: int

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height


def probe_image_header(image_bytes: bytes) -> ImageHeader:
    jpeg_size = jpeg_dimensions(image_bytes)
    if jpeg_size is not None:
        return ImageHeader("JPEG", *jpeg_size)

    try:
        from PIL import Image

        with Image.open(BytesIO(image_bytes)) as image:
            width, height = image.size
            image_format = str(image.format or "unknown").upper()
    except Exception as exc:
        raise ValueError("invalid or unsupported compressed image header") from exc

    if width <= 0 or height <= 0:
        raise ValueError("compressed image has invalid dimensions")
    return ImageHeader(image_format, width, height)


def normalize_rapidocr_output(
    output,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    round_bbox: bool = True,
) -> list[dict]:
    if output is None or output.boxes is None:
        return []

    items = []
    for polygon, text, score in zip(output.boxes, output.txts, output.scores):
        xs = [float(point[0]) * scale_x for point in polygon]
        ys = [float(point[1]) * scale_y for point in polygon]
        bbox = [min(xs), min(ys), max(xs), max(ys)]
        if round_bbox:
            bbox = [
                math.floor(min(xs)),
                math.floor(min(ys)),
                math.ceil(max(xs)),
                math.ceil(max(ys)),
            ]
        items.append(
            {
                "text": str(text),
                "bbox": bbox,
                "score": float(score),
            }
        )
    return items


def scale_ocr_items(
    items: list[dict], scale_x: float, scale_y: float
) -> list[dict]:
    scaled_items = []
    for item in items:
        x1, y1, x2, y2 = item["bbox"]
        scaled_items.append(
            {
                **item,
                "bbox": [
                    math.floor(x1 * scale_x),
                    math.floor(y1 * scale_y),
                    math.ceil(x2 * scale_x),
                    math.ceil(y2 * scale_y),
                ],
            }
        )
    return scaled_items


def build_ocr_payload(results, timestamp, language, error=None) -> dict:
    payload = {
        "text": " ".join(item["text"] for item in results if item.get("text")),
        "items": results,
        "timestamp": timestamp,
        "language": language,
    }
    if error is not None:
        payload["error"] = str(error)
    return payload


def recognize_to_payload(
    adapter, image_bytes: bytes, language: str, timestamp: float
) -> dict:
    try:
        return build_ocr_payload(
            adapter.recognize(image_bytes, language), timestamp, language
        )
    except Exception as exc:
        return build_ocr_payload([], timestamp, language, error=exc)


class _MNNModelSession:
    def __init__(
        self,
        model_path: Path,
        *,
        num_threads: int,
        mean: tuple[float, float, float],
        normal: tuple[float, float, float],
    ):
        import MNN

        self._net = MNN.Interpreter(str(model_path))
        self._session = self._net.createSession(
            {
                "backend": "CPU",
                "numThread": int(num_threads),
                "precision": "low",
                "memory": 2,
                "power": 0,
            }
        )
        self._input = self._net.getSessionInput(self._session)
        self._process = MNN.CVImageProcess(
            {
                "sourceFormat": MNN.CV_ImageFormat_BGR,
                "destFormat": MNN.CV_ImageFormat_BGR,
                "filterType": MNN.CV_Filter_BILINEAL,
                "mean": (*mean, 0.0),
                "normal": (*normal, 1.0),
            }
        )

    def run_uint8(self, image, shape: tuple[int, ...]):
        import numpy as np

        image = np.ascontiguousarray(image, dtype=np.uint8)
        self._net.resizeTensor(self._input, shape)
        self._net.resizeSession(self._session)
        pointer = image.__array_interface__["data"][0]
        self._process.convert(
            pointer,
            image.shape[1],
            image.shape[0],
            image.strides[0],
            self._input,
        )
        self._net.runSession(self._session)
        output = self._net.getSessionOutput(self._session)
        import numpy as _np, MNN as _MNN
        out_shape = output.getShape()
        host = _MNN.Tensor(
            out_shape, _MNN.Halide_Type_Float, _MNN.Tensor_DimensionType_Caffe
        )
        output.copyToHostTensor(host)
        return _np.array(host.getData(), dtype=_np.float32).reshape(out_shape).copy()

    def close(self) -> None:
        release = getattr(self._net, "releaseSession", None)
        if callable(release):
            release(self._session)


class TensorRTShapeError(ValueError):
    """Raised when an OCR tensor is not covered by an engine profile."""


class _TensorRTModelSession:
    def __init__(
        self,
        engine_path: Path,
        *,
        device_id: int,
        mean: tuple[float, float, float],
        normal: tuple[float, float, float],
    ):
        import numpy as np
        import tensorrt as trt
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("OCR TensorRT backend requires CUDA")

        self._np = np
        self._torch = torch
        self._device_id = int(device_id)
        self._mean = np.asarray(mean, dtype=np.float32)
        self._normal = np.asarray(normal, dtype=np.float32)

        logger = trt.Logger(trt.Logger.WARNING)
        with torch.cuda.device(self._device_id):
            self._runtime = trt.Runtime(logger)
            self._engine = self._runtime.deserialize_cuda_engine(
                engine_path.read_bytes()
            )
            if self._engine is None:
                raise RuntimeError(
                    f"failed to deserialize TensorRT engine: {engine_path}"
                )
            self._context = self._engine.create_execution_context()
            if self._context is None:
                raise RuntimeError(
                    f"failed to create TensorRT context: {engine_path}"
                )
            self._stream = torch.cuda.Stream(device=self._device_id)

        inputs = []
        outputs = []
        for index in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(index)
            mode = self._engine.get_tensor_mode(name)
            (inputs if mode == trt.TensorIOMode.INPUT else outputs).append(name)
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(
                "OCR TensorRT engine must have exactly one input and output; "
                f"got inputs={inputs}, outputs={outputs}"
            )
        self._input_name = inputs[0]
        self._output_name = outputs[0]
        self._input_numpy_dtype = _trt_dtype_to_numpy(
            trt, self._engine.get_tensor_dtype(self._input_name)
        )
        self._output_torch_dtype = torch.from_numpy(
            np.empty(
                0,
                dtype=_trt_dtype_to_numpy(
                    trt, self._engine.get_tensor_dtype(self._output_name)
                ),
            )
        ).dtype
        self._profiles = self._read_profiles()
        self._active_profile = 0

    def _read_profiles(self):
        static_shape = tuple(
            int(value) for value in self._engine.get_tensor_shape(self._input_name)
        )
        if static_shape and all(value > 0 for value in static_shape):
            return [(static_shape, static_shape, static_shape)]

        profiles = []
        for index in range(self._engine.num_optimization_profiles):
            minimum, optimum, maximum = self._engine.get_tensor_profile_shape(
                self._input_name, index
            )
            profiles.append(
                tuple(tuple(int(value) for value in shape) for shape in (
                    minimum,
                    optimum,
                    maximum,
                ))
            )
        if not profiles:
            raise RuntimeError("OCR TensorRT engine has no optimization profile")
        return profiles

    @property
    def optimization_shape(self) -> tuple[int, ...]:
        return self._profiles[0][1]

    def _select_profile(self, shape: tuple[int, ...]) -> int:
        candidates = []
        for index, (minimum, _optimum, maximum) in enumerate(self._profiles):
            if len(shape) != len(minimum):
                continue
            if all(
                lower <= value <= upper
                for value, lower, upper in zip(shape, minimum, maximum)
            ):
                distance = sum(
                    abs(math.log(max(1, value) / max(1, optimum)))
                    for value, optimum in zip(shape, _optimum)
                )
                candidates.append((distance, index))
        if candidates:
            return min(candidates)[1]
        ranges = [
            {"min": minimum, "max": maximum}
            for minimum, _optimum, maximum in self._profiles
        ]
        raise TensorRTShapeError(
            f"OCR TensorRT input shape {shape} is outside profiles {ranges}"
        )

    def run_uint8(self, image, shape: tuple[int, ...]):
        image = self._np.ascontiguousarray(image, dtype=self._np.uint8)
        if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
            raise ValueError(f"invalid OCR TensorRT NCHW shape: {shape}")
        if image.shape[:2] != (shape[2], shape[3]):
            raise ValueError(
                f"OCR TensorRT image/shape mismatch: {image.shape} vs {shape}"
            )
        return self.run_uint8_batch(image[None], shape)

    def run_uint8_batch(self, images, shape: tuple[int, ...]):
        images = self._np.ascontiguousarray(images, dtype=self._np.uint8)
        if len(shape) != 4 or shape[1] != 3:
            raise ValueError(f"invalid OCR TensorRT NCHW shape: {shape}")
        if images.ndim != 4 or images.shape != (
            shape[0], shape[2], shape[3], shape[1]
        ):
            raise ValueError(
                f"OCR TensorRT image batch/shape mismatch: {images.shape} vs "
                f"{shape}"
            )
        value = (images.astype(self._np.float32) - self._mean) * self._normal
        array = self._np.ascontiguousarray(
            value.transpose(0, 3, 1, 2), dtype=self._input_numpy_dtype
        )
        return self._run(array)

    def max_batch_size(self, height: int, width: int) -> int:
        compatible = [
            maximum[0]
            for minimum, _optimum, maximum in self._profiles
            if len(minimum) == 4
            and minimum[1] <= 3 <= maximum[1]
            and minimum[2] <= height <= maximum[2]
            and minimum[3] <= width <= maximum[3]
            and minimum[0] <= 1 <= maximum[0]
        ]
        if not compatible:
            self._select_profile((1, 3, int(height), int(width)))
        return max(compatible)

    def _run(self, array):
        shape = tuple(int(value) for value in array.shape)
        profile = self._select_profile(shape)
        torch = self._torch
        with torch.cuda.device(self._device_id), torch.cuda.stream(self._stream):
            if profile != self._active_profile:
                if not self._context.set_optimization_profile_async(
                    profile, self._stream.cuda_stream
                ):
                    raise RuntimeError(
                        f"failed to select TensorRT profile {profile}"
                    )
                self._active_profile = profile
            if not self._context.set_input_shape(self._input_name, shape):
                raise RuntimeError(f"TensorRT rejected OCR input shape {shape}")
            output_shape = tuple(
                int(value)
                for value in self._context.get_tensor_shape(self._output_name)
            )
            if any(value < 0 for value in output_shape):
                raise RuntimeError(
                    f"TensorRT produced unresolved output shape {output_shape}"
                )
            input_tensor = torch.from_numpy(array).to(
                device=f"cuda:{self._device_id}", non_blocking=False
            )
            output_tensor = torch.empty(
                output_shape,
                dtype=self._output_torch_dtype,
                device=f"cuda:{self._device_id}",
            )
            self._context.set_tensor_address(
                self._input_name, input_tensor.data_ptr()
            )
            self._context.set_tensor_address(
                self._output_name, output_tensor.data_ptr()
            )
            if not self._context.execute_async_v3(self._stream.cuda_stream):
                raise RuntimeError("TensorRT OCR execution failed")
            self._stream.synchronize()
            return output_tensor.cpu().numpy()

    def close(self) -> None:
        self._context = None
        self._engine = None
        self._runtime = None
        self._stream = None


def _trt_dtype_to_numpy(trt_module, dtype):
    """Keep TensorRT's deprecated nptype warning isolated in one helper."""
    import numpy as np

    return np.dtype(trt_module.nptype(dtype))


class _MNNPipeline:
    def __init__(self, root: Path, *, num_threads: int, max_side_len: int,
                 rec_min_score: float = DEFAULT_REC_MIN_SCORE,
                 enable_preprocess: bool = True,
                 det_thresh: float = DEFAULT_DET_THRESH,
                 det_box_thresh: float = DEFAULT_DET_BOX_THRESH,
                 det_unclip_ratio: float = DEFAULT_DET_UNCLIP_RATIO):
        from rapidocr.ch_ppocr_det.utils import DBPostProcess
        from rapidocr.ch_ppocr_rec.utils import CTCLabelDecode
        from rapidocr.utils.process_img import get_rotate_crop_image

        self._det = _MNNModelSession(
            root / "det.mnn",
            num_threads=num_threads,
            mean=_OCR_MEAN,
            normal=_OCR_NORMAL,
        )
        try:
            self._rec = _MNNModelSession(
                root / "rec.mnn",
                num_threads=num_threads,
                mean=_OCR_MEAN,
                normal=_OCR_NORMAL,
            )
        except Exception:
            self._det.close()
            raise
        self._det_postprocess = DBPostProcess(
            thresh=float(det_thresh),
            box_thresh=float(det_box_thresh),
            max_candidates=1000,
            unclip_ratio=float(det_unclip_ratio),
            use_dilation=True,
            score_mode="fast",
        )
        self._rec_decode = CTCLabelDecode(character_path=root / "keys.txt")
        self._crop = get_rotate_crop_image
        self._rec_min_score = float(rec_min_score)
        self._enable_preprocess = bool(enable_preprocess)
        self._max_side_len = max(32, int(max_side_len))

    @staticmethod
    def _multiple_of_32(value: float) -> int:
        return max(32, int(round(value / 32)) * 32)

    def _detector_input(self, image):
        import cv2

        height, width = image.shape[:2]
        scale = min(1.0, self._max_side_len / max(height, width))
        target_height = self._multiple_of_32(height * scale)
        target_width = self._multiple_of_32(width * scale)
        if (target_width, target_height) == (width, height):
            return image
        return cv2.resize(
            image,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )

    def _detect(self, image):
        import numpy as np

        detector_input = self._detector_input(image)
        height, width = detector_input.shape[:2]
        prediction = self._det.run_uint8(
            detector_input, (1, 3, height, width)
        )
        boxes, scores = self._det_postprocess(prediction, image.shape[:2])
        if len(boxes) == 0:
            return np.empty((0, 4, 2), dtype=np.float32), []
        order = sorted(
            range(len(boxes)),
            key=lambda index: (boxes[index][0][1], boxes[index][0][0]),
        )
        return boxes[order], [scores[index] for index in order]

    @staticmethod
    def _prepare_recognition_crop(crop):
        import cv2
        import numpy as np

        height, width = crop.shape[:2]
        if height <= 0 or width <= 0:
            return None
        ratio = width / float(height)
        target_height = 48
        target_width = max(320, int(math.ceil(target_height * ratio)))
        target_width = max(32, int(math.ceil(target_width / 8)) * 8)
        resized_width = min(
            target_width, max(1, int(math.ceil(target_height * ratio)))
        )
        resized = cv2.resize(
            crop,
            (resized_width, target_height),
            interpolation=cv2.INTER_LINEAR,
        )
        padded = np.full(
            (target_height, target_width, 3), 128, dtype=np.uint8
        )
        padded[:, :resized_width] = resized
        return padded, ratio

    def _recognize_crop(self, crop):
        prepared = self._prepare_recognition_crop(crop)
        if prepared is None:
            return "", 0.0
        padded, ratio = prepared
        target_height, target_width = padded.shape[:2]
        prediction = self._rec.run_uint8(
            padded, (1, 3, target_height, target_width)
        )
        line_results, _ = self._rec_decode(
            prediction,
            False,
            wh_ratio_list=(ratio,),
            max_wh_ratio=target_width / target_height,
        )
        if not line_results:
            return "", 0.0
        text, score = line_results[0]
        return str(text), float(score)

    def infer(self, image) -> list[dict]:
        import copy
        import numpy as np

        det_image = image
        if self._enable_preprocess:
            try:
                det_image = preprocess_for_ocr(image)
            except Exception:
                _log.debug("preprocess failed, using original image", exc_info=True)

        boxes, _ = self._detect(det_image)
        items = []
        for box in boxes:
            # rec uses the ORIGINAL image crop (not the preprocessed one),
            # so bbox coordinates map back to the original without adjustment
            crop = self._crop(image, copy.deepcopy(box))
            text, score = self._recognize_crop(crop)
            if not text.strip() or score < self._rec_min_score:
                continue
            xs = box[:, 0]
            ys = box[:, 1]
            items.append(
                {
                    "text": text,
                    "bbox": [
                        float(np.min(xs)),
                        float(np.min(ys)),
                        float(np.max(xs)),
                        float(np.max(ys)),
                    ],
                    "score": score,
                }
            )
        return items

    def warm_up(self) -> None:
        import numpy as np

        self._det.run_uint8(
            np.zeros((64, 64, 3), dtype=np.uint8), (1, 3, 64, 64)
        )
        self._rec.run_uint8(
            np.zeros((48, 320, 3), dtype=np.uint8), (1, 3, 48, 320)
        )

    def close(self) -> None:
        self._det.close()
        self._rec.close()


class _TensorRTPipeline(_MNNPipeline):
    def __init__(
        self,
        root: Path,
        *,
        device_id: int,
        max_side_len: int,
        rec_min_score: float = DEFAULT_REC_MIN_SCORE,
        enable_preprocess: bool = True,
        det_thresh: float = DEFAULT_DET_THRESH,
        det_box_thresh: float = DEFAULT_DET_BOX_THRESH,
        det_unclip_ratio: float = DEFAULT_DET_UNCLIP_RATIO,
    ):
        from rapidocr.ch_ppocr_det.utils import DBPostProcess
        from rapidocr.ch_ppocr_rec.utils import CTCLabelDecode
        from rapidocr.utils.process_img import get_rotate_crop_image

        self._det = _TensorRTModelSession(
            root / "det.engine",
            device_id=device_id,
            mean=_OCR_MEAN,
            normal=_OCR_NORMAL,
        )
        try:
            self._rec = _TensorRTModelSession(
                root / "rec.engine",
                device_id=device_id,
                mean=_OCR_MEAN,
                normal=_OCR_NORMAL,
            )
        except Exception:
            self._det.close()
            raise
        self._det_postprocess = DBPostProcess(
            thresh=float(det_thresh),
            box_thresh=float(det_box_thresh),
            max_candidates=1000,
            unclip_ratio=float(det_unclip_ratio),
            use_dilation=True,
            score_mode="fast",
        )
        self._rec_decode = CTCLabelDecode(character_path=root / "keys.txt")
        self._crop = get_rotate_crop_image
        self._rec_min_score = float(rec_min_score)
        self._enable_preprocess = bool(enable_preprocess)
        self._max_side_len = max(32, int(max_side_len))

    def warm_up(self) -> None:
        import numpy as np

        det_shape = self._det.optimization_shape
        rec_shape = self._rec.optimization_shape
        self._det.run_uint8_batch(
            np.zeros(
                (det_shape[0], det_shape[2], det_shape[3], 3),
                dtype=np.uint8,
            ),
            det_shape,
        )
        self._rec.run_uint8_batch(
            np.zeros(
                (rec_shape[0], rec_shape[2], rec_shape[3], 3),
                dtype=np.uint8,
            ),
            rec_shape,
        )

    def infer(self, image) -> list[dict]:
        import copy
        from collections import defaultdict

        import numpy as np

        det_image = image
        if self._enable_preprocess:
            try:
                det_image = preprocess_for_ocr(image)
            except Exception:
                _log.debug("preprocess failed, using original image", exc_info=True)

        boxes, _ = self._detect(det_image)
        prepared_by_width = defaultdict(list)
        recognized = [("", 0.0)] * len(boxes)
        for index, box in enumerate(boxes):
            crop = self._crop(image, copy.deepcopy(box))
            prepared = self._prepare_recognition_crop(crop)
            if prepared is None:
                continue
            padded, ratio = prepared
            prepared_by_width[padded.shape[1]].append((index, padded, ratio))

        for target_width, group in prepared_by_width.items():
            max_batch = self._rec.max_batch_size(48, target_width)
            for offset in range(0, len(group), max_batch):
                chunk = group[offset:offset + max_batch]
                images = np.stack([entry[1] for entry in chunk])
                ratios = tuple(entry[2] for entry in chunk)
                prediction = self._rec.run_uint8_batch(
                    images,
                    (len(chunk), 3, 48, target_width),
                )
                line_results, _ = self._rec_decode(
                    prediction,
                    False,
                    wh_ratio_list=ratios,
                    max_wh_ratio=target_width / 48,
                )
                for entry, result in zip(chunk, line_results):
                    text, score = result
                    recognized[entry[0]] = str(text), float(score)

        items = []
        for box, (text, score) in zip(boxes, recognized):
            if not text.strip() or score < self._rec_min_score:
                continue
            xs = box[:, 0]
            ys = box[:, 1]
            items.append(
                {
                    "text": text,
                    "bbox": [
                        float(np.min(xs)),
                        float(np.min(ys)),
                        float(np.max(xs)),
                        float(np.max(ys)),
                    ],
                    "score": score,
                }
            )
        return items


class RapidOCRAdapter:
    def __init__(
        self,
        model_dir: str,
        backend: str = "onnxruntime",
        fallback_backend: str = "",
        fallback_model_dir: str = "",
        device: str = "cpu",
        device_id: int = 0,
        gpu_mem_mb: int = 512,
        use_angle_cls: bool = True,
        num_threads: int = 2,
        max_side_len: int = DEFAULT_MAX_SIDE_LEN,
        rec_min_score: float = DEFAULT_REC_MIN_SCORE,
        enable_preprocess: bool = True,
        det_thresh: float = DEFAULT_DET_THRESH,
        det_box_thresh: float = DEFAULT_DET_BOX_THRESH,
        det_unclip_ratio: float = DEFAULT_DET_UNCLIP_RATIO,
        large_image_strategy: dict | None = None,
    ):
        root = Path(model_dir)
        backend = str(backend).strip().lower()
        fallback_backend = str(fallback_backend).strip().lower()
        if backend not in ("mnn", "onnxruntime", "tensorrt"):
            raise ValueError(
                "OCR backend must be 'mnn', 'onnxruntime', or 'tensorrt'"
            )
        if fallback_backend not in ("", "mnn", "onnxruntime"):
            raise ValueError(
                "OCR fallback_backend must be empty, 'mnn', or 'onnxruntime'"
            )

        device = str(device).strip().lower()
        if device not in ("cpu", "cuda"):
            raise ValueError("OCR device must be 'cpu' or 'cuda'")
        use_cuda = device == "cuda"
        if backend == "tensorrt" and not use_cuda:
            raise ValueError("OCR TensorRT backend requires device='cuda'")
        if backend == "onnxruntime" and use_cuda:
            import onnxruntime as ort

            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" not in providers:
                raise RuntimeError(
                    "OCR device=cuda requires CUDAExecutionProvider; "
                    f"available providers: {providers}"
                )

        self._use_angle_cls = use_angle_cls
        self._max_side_len = max_side_len
        self._rec_min_score = float(rec_min_score)
        self._request_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        strategy_config = LargeImageStrategyConfig.from_mapping(
            large_image_strategy
        )
        self._large_image_strategy = (
            AdaptiveTiledOCRStrategy(strategy_config, max_side_len)
            if strategy_config.enabled
            else None
        )

        self._pipeline = None
        self._backend_name = backend
        self._runtime_fallback_pipeline = None
        self._runtime_fallback_loader = None

        def load_native_pipeline(selected_backend: str, selected_root: Path):
            if selected_backend == "mnn":
                required_files = MNN_MODEL_FILES
                pipeline_type = _MNNPipeline
                pipeline_kwargs = {"num_threads": num_threads}
                classifier_error = "OCR MNN backend does not load angle classifier"
            elif selected_backend == "tensorrt":
                required_files = TENSORRT_MODEL_FILES
                pipeline_type = _TensorRTPipeline
                pipeline_kwargs = {"device_id": device_id}
                classifier_error = (
                    "OCR TensorRT backend does not load angle classifier"
                )
            else:
                raise ValueError(
                    f"unsupported native OCR backend: {selected_backend}"
                )

            missing = [
                name for name in required_files
                if not (selected_root / name).is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    f"OCR {selected_backend} model files missing: "
                    f"{', '.join(missing)}"
                )
            if use_angle_cls:
                raise ValueError(classifier_error)
            return pipeline_type(
                selected_root,
                **pipeline_kwargs,
                max_side_len=max_side_len,
                rec_min_score=rec_min_score,
                enable_preprocess=enable_preprocess,
                det_thresh=det_thresh,
                det_box_thresh=det_box_thresh,
                det_unclip_ratio=det_unclip_ratio,
            )

        def start_native_pipeline(selected_backend: str, selected_root: Path):
            pipeline = load_native_pipeline(selected_backend, selected_root)
            try:
                pipeline.warm_up()
            except Exception:
                pipeline.close()
                raise
            return pipeline

        if backend in ("mnn", "tensorrt"):
            try:
                self._pipeline = start_native_pipeline(backend, root)
                self._engine = None
                if (
                    backend == "tensorrt"
                    and fallback_backend == "mnn"
                    and fallback_model_dir
                ):
                    fallback_root = Path(fallback_model_dir)
                    self._runtime_fallback_loader = lambda: (
                        start_native_pipeline("mnn", fallback_root)
                    )
                return
            except Exception:
                if self._pipeline is not None:
                    self._pipeline.close()
                    self._pipeline = None
                if not fallback_backend or not fallback_model_dir:
                    raise
                root = Path(fallback_model_dir)
                self._backend_name = fallback_backend
                if fallback_backend == "mnn":
                    self._pipeline = start_native_pipeline("mnn", root)
                    self._engine = None
                    return

        missing = [
            name for name in ORT_MODEL_FILES if not (root / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"OCR ONNX model files missing: {', '.join(missing)}"
            )

        # Load rapidocr's own default config, override model paths
        from rapidocr import RapidOCR
        from rapidocr.main import DEFAULT_CFG_PATH

        import yaml
        with open(DEFAULT_CFG_PATH) as f:
            cfg = yaml.safe_load(f)

        cfg["Det"].update(
            {
                "engine_type": "onnxruntime",
                "lang_type": "ch",
                "model_type": "tiny",
                "ocr_version": "PP-OCRv6",
                "model_path": str(root / "det.onnx"),
                # 必须显式转发：rapidocr 默认 unclip_ratio=1.6/thresh/box_thresh
                # 与 MNN 管线参数不一致会导致 ORT 结果偏离生产配置
                "thresh": float(det_thresh),
                "box_thresh": float(det_box_thresh),
                "unclip_ratio": float(det_unclip_ratio),
            }
        )
        cfg["Cls"].update(
            {
                "engine_type": "onnxruntime",
                "lang_type": "ch",
                "model_type": "mobile",
                "ocr_version": "PP-OCRv4",
                "model_path": str(root / "cls.onnx"),
            }
        )
        cfg["Rec"].update(
            {
                "engine_type": "onnxruntime",
                "lang_type": "ch",
                "model_type": "tiny",
                "ocr_version": "PP-OCRv6",
                "model_path": str(root / "rec.onnx"),
                "rec_keys_path": str(root / "keys.txt"),
            }
        )
        cfg["Global"]["use_cls"] = use_angle_cls
        # Engine-level cap must NOT shrink tiles: the single-pass path already
        # resizes to max_side_len before reaching the engine, while the tiled
        # strategy feeds full tile_size crops through the same engine.
        # (rapidocr crashes on max_side_len <= 0, so fall back to its default.)
        engine_global_cap = max_side_len if max_side_len > 0 else 2000
        if strategy_config.enabled:
            engine_global_cap = max(engine_global_cap, strategy_config.tile_size)
        cfg["Global"]["max_side_len"] = engine_global_cap

        engine_cfg = cfg.setdefault("EngineConfig", {}).setdefault(
            "onnxruntime", {}
        )
        engine_cfg["intra_op_num_threads"] = num_threads
        engine_cfg["inter_op_num_threads"] = 1
        engine_cfg["use_cuda"] = use_cuda
        if use_cuda:
            cuda_ep_cfg = engine_cfg.setdefault("cuda_ep_cfg", {})
            cuda_ep_cfg.update(
                {
                    "device_id": int(device_id),
                    "gpu_mem_limit": int(gpu_mem_mb) * 1024 * 1024,  # ORT CUDA EP: bytes
                }
            )

        with tempfile.TemporaryDirectory(prefix="rapidocr-config-") as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            with config_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, default_flow_style=False)
            self._engine = RapidOCR(config_path=str(config_path))

    @staticmethod
    def _jpeg_dimensions(image_bytes: bytes) -> tuple[int, int] | None:
        return jpeg_dimensions(image_bytes)

    @staticmethod
    def _probe_image_header(image_bytes: bytes) -> ImageHeader:
        return probe_image_header(image_bytes)

    def _infer_image(self, image) -> list[dict]:
        with self._inference_lock:
            pipeline = getattr(self, "_pipeline", None)
            if pipeline is not None:
                try:
                    return pipeline.infer(image)
                except TensorRTShapeError:
                    loader = getattr(self, "_runtime_fallback_loader", None)
                    if loader is None:
                        raise
                    fallback = getattr(
                        self, "_runtime_fallback_pipeline", None
                    )
                    if fallback is None:
                        _log.warning(
                            "OCR TensorRT profile does not cover this request; "
                            "loading the configured MNN fallback"
                        )
                        fallback = loader()
                        self._runtime_fallback_pipeline = fallback
                    return fallback.infer(image)
            output = self._engine(
                image,
                use_det=True,
                use_cls=self._use_angle_cls,
                use_rec=True,
            )
        items = normalize_rapidocr_output(output, round_bbox=False)
        # 与 _MNNPipeline.infer 对齐：ORT 路径同样应用 rec_min_score 过滤
        return [
            item for item in items
            if item["text"].strip() and item["score"] >= self._rec_min_score
        ]

    def _recognize_single_pass(self, image_bytes: bytes) -> list[dict]:
        header = self._probe_image_header(image_bytes)
        source_size = header.size
        image = decode_vips_overview(image_bytes, self._max_side_len)
        decoded_height, decoded_width = image.shape[:2]
        items = self._infer_image(image)
        source_width, source_height = source_size
        return scale_ocr_items(
            items,
            scale_x=source_width / decoded_width,
            scale_y=source_height / decoded_height,
        )

    def _recognize_request(self, image_bytes: bytes) -> list:
        strategy = getattr(self, "_large_image_strategy", None)
        if strategy is not None:
            header = self._probe_image_header(image_bytes)
            source_size = header.size
            if strategy.should_handle(source_size):
                return strategy.recognize(image_bytes, self._infer_image)
        return self._recognize_single_pass(image_bytes)

    def recognize(self, image_bytes: bytes, language: str = "zh") -> list:
        request_lock = getattr(self, "_request_lock", None)
        if request_lock is None:
            return self._recognize_request(image_bytes)
        with request_lock:
            return self._recognize_request(image_bytes)
