# OCR 与最新 Main 集成实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在 `feat/zengzhitao` 中合入最新 `origin/main`，保留 official Paraformer 离线 ASR 默认行为，并增加默认启用、模型总量小于 15 MiB 的本地 PP-OCRv6 tiny OCR 插件。

**架构：** 先合并主分支并只在 ASR 冲突处组合双方行为，再从 `origin/feat/wanglimin-test` 选择性迁移 OCR 的 MCP/ROS2 契约。OCR 推理由独立的 RapidOCR CPU 适配器完成，四个模型文件由专用下载器在 Jetson 镜像构建时从内网 HTTP 原子下载，Docker 层删除 RapidOCR wheel 自带 ONNX，Fast DDS 配置负责传输压缩图片。

**技术栈：** Python 3.8、ROS2 Humble、RapidOCR 3.9.1、ONNX Runtime CPU EP、PP-OCRv6 tiny、`unittest`、Docker Jetson、Fast DDS

---

## 文件职责

- 修改：`deploy/restart/entrypoint.sh`：通过合并 `origin/main` 获得 restart helper 修复。
- 修改：`perception/plugins/asr.py`：保留 feature 的离线 Paraformer、VAD、KWS、并发和指标逻辑，同时加入可选 SenseVoice adapter。
- 修改：`perception/utils/model_downloader.py`：加入主分支的 `asr_sensevoice` 模型注册。
- 创建：`perception/plugins/ocr.py`：OCR MCP 工具、ROS2 节点、云端可选 adapter 和插件生命周期。
- 创建：`perception/plugins/ocr_runtime.py`：RapidOCR CPU 初始化、结果归一化和错误结果构造。
- 修改：`perception/main.py`：按配置加载 `OCRPlugin`。
- 修改：`perception/config.yaml`：默认启用 `rapidocr`，不改变 ASR 默认配置。
- 创建：`perception/utils/ocr_model_downloader.py`：从内网 HTTP 下载四个规范化文件并执行非空和 15 MiB 校验。
- 创建：`tools/prepare_ocr_model_bundle.py`：从 RapidOCR 官方 ModelScope 路径下载并重命名四个模型文件，供上传 JuiceFS。
- 创建：`perception/config/fastdds_large_message.xml`：配置 ROS2 大消息 UDP 缓冲区。
- 修改：`perception/Dockerfile.jetson`：安装 RapidOCR 依赖、删除 wheel 内置 ONNX、下载 OCR 模型并启用 Fast DDS profile。
- 创建：`perception/tests/test_ocr_contract.py`：OCR MCP、topic、adapter、归一化和错误结果测试。
- 创建：`perception/tests/test_ocr_model_downloader.py`：下载完整性和模型大小限制测试。
- 创建：`perception/tests/test_ocr_packaging.py`：主程序、配置和 Docker 静态契约测试。
- 创建：`perception/tests/ocr_ros_smoke.py`：Jetson 容器内 MCP start、JPEG 发布和 OCR JSON 订阅验收脚本。
- 修改：`perception/tests/test_asr_contract.py`：增加 SenseVoice 选择和 offline 默认回归测试。

### 任务 1：合并最新 Main 并组合 ASR 行为

**文件：**
- 修改：`deploy/restart/entrypoint.sh`
- 修改：`perception/plugins/asr.py:68-306`
- 修改：`perception/utils/model_downloader.py:33-63`
- 测试：`perception/tests/test_asr_contract.py`

- [ ] **步骤 1：记录合并前基线并运行 ASR 测试**

运行：

```bash
git status --short
git fetch origin
python3 -m unittest \
  perception.tests.test_asr_contract \
  perception.tests.test_asr_runtime \
  perception.tests.test_asr_offline_sherpa_compat \
  perception.tests.test_asr_model_downloader \
  perception.tests.test_asr_packaging -v
```

预期：工作树为空；现有 ASR 测试全部 `OK`。

- [ ] **步骤 2：添加 SenseVoice 选择的失败测试**

在 `perception/tests/test_asr_contract.py` 加入：

```python
    def test_sensevoice_selection_precedes_offline_paraformer_mode(self):
        expected = object()
        adapter_factory = mock.Mock(return_value=expected)
        with mock.patch.dict(
            self.asr.ASR_MODELS["sensevoice-small"],
            {"adapter": adapter_factory},
        ):
            adapter = self.asr._build_asr_adapter(
                {
                    "mode": "offline",
                    "asr_model": "sensevoice-small",
                    "model_dir": "/models/sherpa-onnx/sensevoice",
                    "device": "cpu",
                    "num_threads": 2,
                }
            )

        self.assertIs(adapter, expected)
        adapter_factory.assert_called_once_with(
            "/models/sherpa-onnx/sensevoice", "cpu", 2
        )

    def test_offline_default_still_uses_official_paraformer(self):
        expected = object()
        with mock.patch(
            "plugins.asr_offline.OfflineASRAdapter.get_instance",
            return_value=expected,
        ) as get_instance:
            adapter = self.asr._build_asr_adapter({})

        self.assertIs(adapter, expected)
        get_instance.assert_called_once_with(
            model_path="/models/sherpa-onnx/asr-offline",
            config=None,
            num_threads=2,
            provider="cpu",
        )
```

- [ ] **步骤 3：运行新增测试验证失败**

运行：

```bash
python3 -m unittest perception.tests.test_asr_contract.ASRContractTest.test_sensevoice_selection_precedes_offline_paraformer_mode -v
```

预期：`ERROR`，原因是当前 feature 尚无 `SherpaOnnxSenseVoiceAdapter`。

- [ ] **步骤 4：合并主分支并显式解决 ASR 冲突**

运行：

```bash
git merge --no-ff origin/main
```

预期：`perception/plugins/asr.py` 产生冲突；restart helper 和 model downloader 自动合入。

在 `perception/plugins/asr.py` 保留 feature 版本的其余逻辑，并加入以下 SenseVoice adapter：

```python
class SherpaOnnxSenseVoiceAdapter(ASRAdapter):
    """Offline SenseVoice-Small recognizer exposed as an optional ASR model."""

    def __init__(self, model_dir: str, hw_provider: str = "cpu", num_threads: int = 2):
        from utils.model_downloader import ensure_model

        ensure_model("asr_sensevoice", model_dir)
        import sherpa_onnx

        model_path = os.path.join(model_dir, "model.int8.onnx")
        if not os.path.exists(model_path):
            model_path = os.path.join(model_dir, "model.onnx")
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=model_path,
            tokens=os.path.join(model_dir, "tokens.txt"),
            num_threads=num_threads,
            provider=hw_provider,
            use_itn=True,
            language="auto",
        )

    def transcribe(self, wav_bytes: bytes, language: str) -> str:
        import io
        import wave

        with wave.open(io.BytesIO(wav_bytes)) as wav_file:
            pcm = wav_file.readframes(wav_file.getnframes())
        samples = pcm16_to_float_samples(pcm)
        if hasattr(samples, "tolist"):
            samples = samples.tolist()
        stream = self._recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, samples)
        self._recognizer.decode_streams([stream])
        return stream.result.text.strip()
```

在 `ASR_MODELS` 加入：

```python
    "sensevoice-small": {
        "label": "SenseVoice Small (zh+en+ja+ko+yue)",
        "adapter": SherpaOnnxSenseVoiceAdapter,
        "default_model_dir": "/models/sherpa-onnx/sensevoice",
    },
```

把 `_build_asr_adapter` 的开头组合为：

```python
def _build_asr_adapter(cfg: dict) -> Optional[ASRAdapter]:
    model_name = cfg.get("asr_model", "paraformer-zh-en")
    provider = cfg.get("device") or cfg.get("hw_provider") or "cpu"
    num_threads = int(cfg.get("num_threads", 2))

    if model_name == "sensevoice-small":
        model_info = ASR_MODELS[model_name]
        model_dir = cfg.get("model_dir", model_info["default_model_dir"])
        other_defaults = {
            item["default_model_dir"]
            for name, item in ASR_MODELS.items()
            if name != model_name
        }
        if model_dir in other_defaults:
            model_dir = model_info["default_model_dir"]
        return model_info["adapter"](model_dir, provider, num_threads)

    mode = _resolve_asr_mode(cfg)
    if mode == "offline":
        from plugins.asr_offline import OfflineASRAdapter

        return OfflineASRAdapter.get_instance(
            model_path=cfg.get("model_path", "/models/sherpa-onnx/asr-offline"),
            config=cfg.get("sherpa_config"),
            num_threads=num_threads,
            provider=provider,
        )

    model_info = ASR_MODELS.get(model_name, ASR_MODELS["paraformer-zh-en"])
    model_dir = cfg.get("model_dir", model_info["default_model_dir"])
    other_defaults = {
        item["default_model_dir"]
        for name, item in ASR_MODELS.items()
        if name != model_name
    }
    if model_dir in other_defaults:
        model_dir = model_info["default_model_dir"]
    return model_info["adapter"](model_dir, provider, num_threads)
```

同时把 tool schema 的 `asr_model.enum` 扩为：

```python
["paraformer-zh-en", "zipformer-en", "sensevoice-small"]
```

确认 `perception/utils/model_downloader.py` 包含：

```python
    "asr_sensevoice": {
        "url": f"{COS_BASE}/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.zip",
        "check_file": "tokens.txt",
    },
```

然后运行：

```bash
git add deploy/restart/entrypoint.sh perception/plugins/asr.py perception/utils/model_downloader.py perception/tests/test_asr_contract.py
git commit --no-edit
```

预期：完成 merge commit，不再存在未合并文件。

- [ ] **步骤 5：运行 ASR 回归测试**

运行：

```bash
python3 -m unittest \
  perception.tests.test_asr_contract \
  perception.tests.test_asr_runtime \
  perception.tests.test_asr_offline_sherpa_compat \
  perception.tests.test_asr_model_downloader \
  perception.tests.test_asr_packaging -v
```

预期：全部 `OK`；默认 `{}` 仍构造 `/models/sherpa-onnx/asr-offline` 的 official Paraformer adapter。

### 任务 2：迁移 OCR MCP 与 ROS2 契约

**文件：**
- 创建：`perception/plugins/ocr.py`
- 修改：`perception/main.py:54-90`
- 修改：`perception/config.yaml:3-60`
- 创建：`perception/tests/test_ocr_contract.py`
- 创建：`perception/tests/test_ocr_packaging.py`

- [ ] **步骤 1：编写 OCR 注册和契约的失败测试**

创建 `perception/tests/test_ocr_contract.py`，安装与 ASR 测试一致的 ROS stub，并补充 `sensor_msgs.msg.CompressedImage`：

```python
import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PERCEPTION_ROOT))


def _install_ros_stubs():
    rclpy = types.ModuleType("rclpy")
    rclpy.node = types.ModuleType("rclpy.node")
    rclpy.node.Node = object
    rclpy.qos = types.ModuleType("rclpy.qos")
    rclpy.qos.QoSProfile = lambda **kwargs: kwargs
    rclpy.qos.ReliabilityPolicy = types.SimpleNamespace(RELIABLE="RELIABLE")
    rclpy.qos.HistoryPolicy = types.SimpleNamespace(KEEP_LAST="KEEP_LAST")
    rclpy.qos.DurabilityPolicy = types.SimpleNamespace(VOLATILE="VOLATILE")
    sensor_msgs = types.ModuleType("sensor_msgs")
    sensor_msgs.msg = types.ModuleType("sensor_msgs.msg")
    sensor_msgs.msg.CompressedImage = type("CompressedImage", (), {})
    std_msgs = types.ModuleType("std_msgs")
    std_msgs.msg = types.ModuleType("std_msgs.msg")
    std_msgs.msg.String = type("String", (), {})
    sys.modules.update({
        "rclpy": rclpy,
        "rclpy.node": rclpy.node,
        "rclpy.qos": rclpy.qos,
        "sensor_msgs": sensor_msgs,
        "sensor_msgs.msg": sensor_msgs.msg,
        "std_msgs": std_msgs,
        "std_msgs.msg": std_msgs.msg,
    })


class OCRContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_ros_stubs()
        cls.ocr = importlib.import_module("plugins.ocr")

    def test_tool_and_topic_contract(self):
        tool = self.ocr.TOOLS[0]
        self.assertEqual(tool["name"], "ocr")
        self.assertEqual(tool["inputSchema"]["properties"]["action"]["enum"],
                         ["start", "stop", "info", "config"])
        self.assertEqual(tool["topic_in"][0]["format"], "image/jpeg")
        self.assertEqual(tool["topic_out"][0]["format"], "data/json")
        self.assertEqual(self.ocr._ocr_output_topic("/robot/camera"),
                         "/robot/camera/ocr")

    def test_default_provider_builds_local_adapter(self):
        expected = object()
        with mock.patch("plugins.ocr.RapidOCRAdapter", return_value=expected) as adapter:
            result = self.ocr._build_ocr_adapter({
                "provider": "rapidocr",
                "model_dir": "/models/ocr/ppocrv6-tiny",
                "use_angle_cls": True,
                "num_threads": 2,
            })
        self.assertIs(result, expected)
        adapter.assert_called_once_with(
            "/models/ocr/ppocrv6-tiny", use_angle_cls=True, num_threads=2
        )


if __name__ == "__main__":
    unittest.main()
```

创建 `perception/tests/test_ocr_packaging.py` 的首批测试：

```python
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


class OCRPackagingTest(unittest.TestCase):
    def test_bundle_registers_ocr_plugin(self):
        source = (REPO_ROOT / "perception" / "main.py").read_text(encoding="utf-8")
        self.assertIn("from plugins.ocr import OCRPlugin", source)
        self.assertIn('plugins_cfg.get("ocr", {}).get("enabled", False)', source)

    def test_default_config_enables_local_ocr_without_changing_asr(self):
        config = (REPO_ROOT / "perception" / "config.yaml").read_text(encoding="utf-8")
        self.assertIn("  asr:\n    enabled: true\n    mode: offline", config)
        self.assertIn("  ocr:\n    enabled: true\n    provider: rapidocr", config)
        self.assertIn("model_dir: /models/ocr/ppocrv6-tiny", config)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
python3 -m unittest perception.tests.test_ocr_contract perception.tests.test_ocr_packaging -v
```

预期：`ERROR` 或 `FAIL`，因为 `plugins.ocr`、主程序注册和默认配置尚不存在。

- [ ] **步骤 3：选择性迁移 OCR 文件并接入 bundle**

以 `origin/feat/wanglimin-test:perception/plugins/ocr.py` 为唯一迁移源创建 `perception/plugins/ocr.py`：

```bash
git show origin/feat/wanglimin-test:perception/plugins/ocr.py > perception/plugins/ocr.py
```

保留其中 `OpenAIVisionAdapter`、`QwenVLAdapter`、`TesseractAdapter`、`_OCRNode`、`OCRPlugin`，不迁移 `asr_local.py`，然后执行以下确定性改动：

```python
from plugins.ocr_runtime import RapidOCRAdapter, build_ocr_payload


def _ocr_output_topic(input_topic: str) -> str:
    return f"{input_topic}/ocr"


def _build_ocr_adapter(cfg: dict) -> Optional[OCRAdapter]:
    provider = cfg.get("provider", "rapidocr")
    if provider == "rapidocr":
        return RapidOCRAdapter(
            cfg.get("model_dir", "/models/ocr/ppocrv6-tiny"),
            use_angle_cls=bool(cfg.get("use_angle_cls", True)),
            num_threads=int(cfg.get("num_threads", 2)),
        )
    if provider == "openai":
        key = cfg.get("key", "")
        return OpenAIVisionAdapter(cfg.get("url", ""), key, cfg.get("model", "")) if key else None
    if provider == "qwen":
        key = cfg.get("key", "")
        return QwenVLAdapter(cfg.get("url", ""), key, cfg.get("model", "")) if key else None
    if provider == "tesseract":
        return TesseractAdapter(cfg.get("language", "chi_sim+eng"))
    return None
```

把 tool schema 的 provider 改为：

```python
["rapidocr", "openai", "qwen", "tesseract"]
```

把 `_OCRNode` 的输出 topic 改为 `_ocr_output_topic(input_topic)`；worker 暂时继续使用迁移文件原有结果逻辑，任务 3 再统一为 `build_ocr_payload`。

在 `perception/main.py` 的 VOP 加载块后加入：

```python
        if plugins_cfg.get("ocr", {}).get("enabled", False):
            from plugins.ocr import OCRPlugin
            self._plugins.append(OCRPlugin(plugins_cfg["ocr"], executor))
            log.info("OCRPlugin loaded")
```

在 `perception/config.yaml` 末尾加入，并保持现有 ASR 段原样：

```yaml
  ocr:
    enabled: true
    provider: rapidocr
    model_dir: /models/ocr/ppocrv6-tiny
    language: zh
    use_angle_cls: true
    num_threads: 2
```

先创建一个可导入的 `perception/plugins/ocr_runtime.py`：

```python
from __future__ import annotations


class RapidOCRAdapter:
    def __init__(self, model_dir: str, use_angle_cls: bool = True, num_threads: int = 2):
        self.model_dir = model_dir
        self.use_angle_cls = use_angle_cls
        self.num_threads = num_threads

    def recognize(self, image_bytes: bytes, language: str = "zh") -> list:
        raise RuntimeError("RapidOCR runtime is not initialized")


def build_ocr_payload(results, timestamp, language, error=None):
    payload = {
        "text": " ".join(item["text"] for item in results if item.get("text")),
        "items": results,
        "timestamp": timestamp,
        "language": language,
    }
    if error:
        payload["error"] = error
    return payload
```

- [ ] **步骤 4：运行契约测试并提交**

运行：

```bash
python3 -m unittest perception.tests.test_ocr_contract perception.tests.test_ocr_packaging -v
```

预期：首批 OCR 契约测试全部 `OK`。

提交：

```bash
git add perception/plugins/ocr.py perception/plugins/ocr_runtime.py perception/main.py perception/config.yaml perception/tests/test_ocr_contract.py perception/tests/test_ocr_packaging.py
git commit -m "feat(ocr): add OCR plugin contract"
```

### 任务 3：实现 RapidOCR CPU 适配器和统一输出

**文件：**
- 修改：`perception/plugins/ocr_runtime.py`
- 修改：`perception/plugins/ocr.py:497-555`
- 修改：`perception/tests/test_ocr_contract.py`

- [ ] **步骤 1：编写归一化和错误输出的失败测试**

在 `OCRContractTest` 加入：

```python
    def test_rapidocr_output_normalizes_polygon_to_pixel_bbox(self):
        output = types.SimpleNamespace(
            boxes=[[[10.2, 20.8], [110.4, 19.9], [111.0, 50.1], [9.7, 51.2]]],
            txts=("你好 123",),
            scores=(0.9876,),
        )
        items = self.ocr.normalize_rapidocr_output(output)
        self.assertEqual(items, [{
            "text": "你好 123",
            "bbox": [10, 20, 111, 52],
            "score": 0.9876,
        }])

    def test_inference_error_becomes_publishable_empty_payload(self):
        adapter = mock.Mock()
        adapter.recognize.side_effect = ValueError("invalid image")
        payload = self.ocr.recognize_to_payload(
            adapter, b"not-an-image", "zh", 123.0
        )
        self.assertEqual(payload["text"], "")
        self.assertEqual(payload["items"], [])
        self.assertEqual(payload["timestamp"], 123.0)
        self.assertEqual(payload["language"], "zh")
        self.assertEqual(payload["error"], "invalid image")

    def test_rapidocr_adapter_uses_only_external_cpu_models(self):
        fake_engine = mock.Mock()
        rapidocr_module = types.ModuleType("rapidocr")
        rapidocr_module.RapidOCR = mock.Mock(return_value=fake_engine)
        with tempfile.TemporaryDirectory() as model_dir:
            for name in ("det.onnx", "rec.onnx", "cls.onnx", "keys.txt"):
                (Path(model_dir) / name).write_bytes(b"model")
            with mock.patch.dict(sys.modules, {"rapidocr": rapidocr_module}):
                self.ocr.RapidOCRAdapter(
                    model_dir, use_angle_cls=True, num_threads=2
                )
        params = rapidocr_module.RapidOCR.call_args.kwargs["params"]
        self.assertEqual(Path(params["Det.model_path"]).name, "det.onnx")
        self.assertEqual(Path(params["Cls.model_path"]).name, "cls.onnx")
        self.assertEqual(Path(params["Rec.model_path"]).name, "rec.onnx")
        self.assertEqual(Path(params["Rec.rec_keys_path"]).name, "keys.txt")
        self.assertFalse(params["EngineConfig.onnxruntime.use_cuda"])
```

- [ ] **步骤 2：运行新增测试验证失败**

运行：

```bash
python3 -m unittest perception.tests.test_ocr_contract -v
```

预期：`FAIL`，stub adapter 尚未构造 RapidOCR，且归一化和异常 helper 尚不存在。

- [ ] **步骤 3：实现本地 runtime**

用以下实现替换 `perception/plugins/ocr_runtime.py`：

```python
from __future__ import annotations

import math
from pathlib import Path


REQUIRED_MODEL_FILES = ("det.onnx", "rec.onnx", "cls.onnx", "keys.txt")


def normalize_rapidocr_output(output) -> list[dict]:
    if output is None or output.boxes is None:
        return []
    items = []
    for polygon, text, score in zip(output.boxes, output.txts, output.scores):
        xs = [float(point[0]) for point in polygon]
        ys = [float(point[1]) for point in polygon]
        items.append({
            "text": str(text),
            "bbox": [
                math.floor(min(xs)),
                math.floor(min(ys)),
                math.ceil(max(xs)),
                math.ceil(max(ys)),
            ],
            "score": float(score),
        })
    return items


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


def recognize_to_payload(adapter, image_bytes: bytes, language: str, timestamp: float) -> dict:
    try:
        return build_ocr_payload(
            adapter.recognize(image_bytes, language), timestamp, language
        )
    except Exception as exc:
        return build_ocr_payload([], timestamp, language, error=exc)


class RapidOCRAdapter:
    def __init__(self, model_dir: str, use_angle_cls: bool = True, num_threads: int = 2):
        root = Path(model_dir)
        missing = [name for name in REQUIRED_MODEL_FILES if not (root / name).is_file()]
        if missing:
            raise FileNotFoundError(f"OCR model files missing: {', '.join(missing)}")
        from rapidocr import RapidOCR

        self._use_angle_cls = use_angle_cls
        self._engine = RapidOCR(params={
            "Det.engine_type": "onnxruntime",
            "Det.model_path": str(root / "det.onnx"),
            "Cls.engine_type": "onnxruntime",
            "Cls.model_path": str(root / "cls.onnx"),
            "Rec.engine_type": "onnxruntime",
            "Rec.model_path": str(root / "rec.onnx"),
            "Rec.rec_keys_path": str(root / "keys.txt"),
            "Global.use_cls": use_angle_cls,
            "EngineConfig.onnxruntime.intra_op_num_threads": num_threads,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            "EngineConfig.onnxruntime.use_cuda": False,
        })

    def recognize(self, image_bytes: bytes, language: str = "zh") -> list:
        output = self._engine(
            image_bytes, use_det=True, use_cls=self._use_angle_cls, use_rec=True
        )
        return normalize_rapidocr_output(output)
```

在 `perception/plugins/ocr.py` 中导入并再导出测试目标：

```python
from plugins.ocr_runtime import (
    RapidOCRAdapter,
    build_ocr_payload,
    normalize_rapidocr_output,
    recognize_to_payload,
)
```

把 worker 的识别与消息构造统一为：

```python
            payload = recognize_to_payload(
                self._adapter, image_bytes, self._language, ts
            )
            msg = String()
            msg.data = json.dumps(payload, ensure_ascii=False)
            self._pub.publish(msg)
            if "error" in payload:
                log.error("[ocr] recognition error: %s", payload["error"])
```

- [ ] **步骤 4：运行测试并提交**

运行：

```bash
python3 -m unittest perception.tests.test_ocr_contract -v
```

预期：全部 `OK`。

提交：

```bash
git add perception/plugins/ocr.py perception/plugins/ocr_runtime.py perception/tests/test_ocr_contract.py
git commit -m "feat(ocr): add PP-OCRv6 tiny runtime"
```

### 任务 4：实现 OCR 模型下载和 JuiceFS 准备工具

**文件：**
- 创建：`perception/utils/ocr_model_downloader.py`
- 创建：`tools/prepare_ocr_model_bundle.py`
- 创建：`perception/tests/test_ocr_model_downloader.py`

- [ ] **步骤 1：编写下载器失败测试**

创建 `perception/tests/test_ocr_model_downloader.py`：

```python
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class OCRModelDownloaderTest(unittest.TestCase):
    def test_downloads_complete_bundle_without_checksum_pins(self):
        from utils.ocr_model_downloader import MODEL_FILES, download_model

        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as output_dir:
            source = Path(source_dir)
            for index, filename in enumerate(MODEL_FILES, start=1):
                (source / filename).write_bytes(bytes([index]) * index)
            download_model(source.as_uri(), output_dir)
            self.assertEqual(
                {path.name for path in Path(output_dir).iterdir()}, set(MODEL_FILES)
            )

    def test_rejects_empty_file_and_leaves_no_partial_bundle(self):
        from utils.ocr_model_downloader import download_model

        with tempfile.TemporaryDirectory() as output_dir:
            def empty_download(_url, destination):
                Path(destination).write_bytes(b"")
            with mock.patch(
                "utils.ocr_model_downloader.urlretrieve", side_effect=empty_download
            ):
                with self.assertRaisesRegex(ValueError, "empty"):
                    download_model("https://models.example.test", output_dir)
            self.assertEqual(list(Path(output_dir).iterdir()), [])

    def test_rejects_bundle_over_fifteen_mebibytes(self):
        from utils.ocr_model_downloader import download_model

        with tempfile.TemporaryDirectory() as output_dir:
            def oversized_download(_url, destination):
                Path(destination).write_bytes(b"x" * 4_000_000)
            with mock.patch(
                "utils.ocr_model_downloader.urlretrieve", side_effect=oversized_download
            ):
                with self.assertRaisesRegex(ValueError, "15 MiB"):
                    download_model("https://models.example.test", output_dir)
            self.assertEqual(list(Path(output_dir).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
python3 -m unittest perception.tests.test_ocr_model_downloader -v
```

预期：`ERROR`，模块 `utils.ocr_model_downloader` 尚不存在。

- [ ] **步骤 3：实现内网下载器**

创建 `perception/utils/ocr_model_downloader.py`：

```python
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

MODEL_FILES = ("det.onnx", "rec.onnx", "cls.onnx", "keys.txt")
MAX_BUNDLE_BYTES = 15 * 1024 * 1024


def download_model(base_url: str, output_dir: str,
                   filenames=MODEL_FILES, max_bundle_bytes=MAX_BUNDLE_BYTES) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ocr-model-", dir=output.parent) as staging_dir:
        staging = Path(staging_dir)
        for filename in filenames:
            staged_file = staging / filename
            urlretrieve(f"{base_url.rstrip('/')}/{filename}", staged_file)
            if staged_file.stat().st_size == 0:
                raise ValueError(f"Downloaded OCR model file is empty: {filename}")
        total = sum((staging / name).stat().st_size for name in filenames)
        if total > max_bundle_bytes:
            raise ValueError(
                f"OCR model bundle is {total} bytes, exceeds 15 MiB limit"
            )
        for filename in filenames:
            os.replace(staging / filename, output / filename)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    download_model(args.base_url, args.output_dir)


if __name__ == "__main__":
    main()
```

- [ ] **步骤 4：实现官方模型准备工具**

创建 `tools/prepare_ocr_model_bundle.py`，其中源路径与规范化文件名固定为：

```python
from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

BASE = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/master"
SOURCES = {
    "det.onnx": "onnx/PP-OCRv6/det/PP-OCRv6_det_tiny.onnx",
    "rec.onnx": "onnx/PP-OCRv6/rec/PP-OCRv6_rec_tiny.onnx",
    "cls.onnx": "onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx",
    "keys.txt": "paddle/PP-OCRv6/rec/PP-OCRv6_rec_tiny/ppocrv6_tiny_dict.txt",
}
MAX_BUNDLE_BYTES = 15 * 1024 * 1024


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for destination, source in SOURCES.items():
        path = output / destination
        urlretrieve(f"{BASE}/{source}", path)
        if path.stat().st_size == 0:
            raise ValueError(f"Downloaded file is empty: {destination}")
    total = sum((output / name).stat().st_size for name in SOURCES)
    if total > MAX_BUNDLE_BYTES:
        raise ValueError(f"OCR model bundle exceeds 15 MiB: {total} bytes")
    print(f"prepared {len(SOURCES)} files, total={total} bytes")


if __name__ == "__main__":
    main()
```

- [ ] **步骤 5：运行测试并提交**

运行：

```bash
python3 -m unittest perception.tests.test_ocr_model_downloader -v
python3 -m py_compile perception/utils/ocr_model_downloader.py tools/prepare_ocr_model_bundle.py
```

预期：测试全部 `OK`，编译命令退出码为 0。

提交：

```bash
git add perception/utils/ocr_model_downloader.py tools/prepare_ocr_model_bundle.py perception/tests/test_ocr_model_downloader.py
git commit -m "build(ocr): add bounded model downloader"
```

### 任务 5：完成 Jetson Docker 和 Fast DDS 打包

**文件：**
- 创建：`perception/config/fastdds_large_message.xml`
- 修改：`perception/Dockerfile.jetson:43-111`
- 修改：`perception/tests/test_ocr_packaging.py`

- [ ] **步骤 1：扩展 Docker 契约失败测试**

在 `OCRPackagingTest` 加入：

```python
    def test_jetson_image_pins_rapidocr_and_removes_bundled_models(self):
        dockerfile = (REPO_ROOT / "perception" / "Dockerfile.jetson").read_text(encoding="utf-8")
        self.assertIn("rapidocr==3.9.1", dockerfile)
        self.assertIn("--no-deps", dockerfile)
        self.assertIn("rapidocr.__file__", dockerfile)
        self.assertIn("-name '*.onnx' -delete", dockerfile)
        self.assertIn('python3 -c "import onnxruntime', dockerfile)

    def test_jetson_image_downloads_internal_ocr_bundle(self):
        dockerfile = (REPO_ROOT / "perception" / "Dockerfile.jetson").read_text(encoding="utf-8")
        self.assertIn("ocr_model_downloader.py", dockerfile)
        self.assertIn(
            "http://172.28.4.81:34567/zengzhitao/embodied-ai/ocr/ppocrv6-tiny",
            dockerfile,
        )
        self.assertIn("/models/ocr/ppocrv6-tiny", dockerfile)

    def test_fastdds_large_message_profile_is_activated(self):
        dockerfile = (REPO_ROOT / "perception" / "Dockerfile.jetson").read_text(encoding="utf-8")
        self.assertIn("fastdds_large_message.xml", dockerfile)
        self.assertIn("ENV FASTRTPS_DEFAULT_PROFILES_FILE=", dockerfile)
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
python3 -m unittest perception.tests.test_ocr_packaging -v
```

预期：新增三个测试均 `FAIL`。

- [ ] **步骤 3：创建 Fast DDS profile**

创建 `perception/config/fastdds_large_message.xml`：

```xml
<?xml version="1.0" encoding="UTF-8" ?>
<profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
    <transport_descriptors>
        <transport_descriptor>
            <transport_id>CustomUDPTransport</transport_id>
            <type>UDPv4</type>
            <maxMessageSize>65000</maxMessageSize>
            <sendBufferSize>8388608</sendBufferSize>
            <receiveBufferSize>8388608</receiveBufferSize>
        </transport_descriptor>
    </transport_descriptors>
    <participant profile_name="large_message_profile" is_default_profile="true">
        <rtps>
            <userTransports>
                <transport_id>CustomUDPTransport</transport_id>
            </userTransports>
            <useBuiltinTransports>false</useBuiltinTransports>
        </rtps>
    </participant>
</profiles>
```

- [ ] **步骤 4：修改 Dockerfile**

在 Python dependencies 后加入 RapidOCR CPU 依赖。分成一个 `RUN`，以保证 wheel 自带 ONNX 不留在历史层：

```dockerfile
# RapidOCR runtime; keep Jetson's system cv2 and preinstalled onnxruntime-gpu.
RUN pip3 install --no-cache-dir -i ${PYPI_MIRROR} \
      pyclipper six "Shapely!=2.0.4" Pillow "omegaconf!=2.2.1" colorlog && \
    pip3 install --no-cache-dir --no-deps -i ${PYPI_MIRROR} rapidocr==3.9.1 && \
    RAPIDOCR_MODELS="$(python3 -c 'from pathlib import Path; import rapidocr; print(Path(rapidocr.__file__).resolve().parent / "models")')" && \
    find "${RAPIDOCR_MODELS}" -type f -name '*.onnx' -delete && \
    python3 -c "import onnxruntime as ort; assert 'CPUExecutionProvider' in ort.get_available_providers()"
```

在 ASR offline model 下载后加入：

```dockerfile
# PP-OCRv6 tiny bundle, hosted outside Git.
ARG OCR_MODEL_BASE_URL=http://172.28.4.81:34567/zengzhitao/embodied-ai/ocr/ppocrv6-tiny
COPY perception/utils/ocr_model_downloader.py /tmp/ocr_model_downloader.py
RUN python3 /tmp/ocr_model_downloader.py \
    --base-url "${OCR_MODEL_BASE_URL}" \
    --output-dir /models/ocr/ppocrv6-tiny
```

在 application copy 区加入并激活 profile：

```dockerfile
COPY perception/config/fastdds_large_message.xml /opt/phanthy-motus/config/fastdds_large_message.xml
ENV FASTRTPS_DEFAULT_PROFILES_FILE=/opt/phanthy-motus/config/fastdds_large_message.xml
```

- [ ] **步骤 5：运行打包测试并提交**

运行：

```bash
python3 -m unittest perception.tests.test_ocr_packaging -v
```

预期：全部 `OK`。

提交：

```bash
git add perception/config/fastdds_large_message.xml perception/Dockerfile.jetson perception/tests/test_ocr_packaging.py
git commit -m "build(ocr): package Jetson OCR runtime"
```

### 任务 6：全量自动验证与大文件审计

**文件：**
- 创建：`perception/tests/ocr_ros_smoke.py`
- 修改：仅修改自动验证暴露出的本次任务文件

- [ ] **步骤 1：创建 Jetson OCR 端到端 smoke 脚本**

创建 `perception/tests/ocr_ros_smoke.py`：

```python
from __future__ import annotations

import json
import time
import urllib.request

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

MCP_URL = "http://127.0.0.1:15720/mcp"
INPUT_TOPIC = "/ocr_smoke/image"
OUTPUT_TOPIC = f"{INPUT_TOPIC}/ocr"


def call_tool(action: str) -> dict:
    body = json.dumps({
        "jsonrpc": "2.0",
        "id": int(time.time() * 1000),
        "method": "tools/call",
        "params": {
            "name": "ocr",
            "arguments": {
                "action": action,
                "instance_id": "ocr-smoke",
                "input_topic": INPUT_TOPIC,
            },
        },
    }).encode()
    request = urllib.request.Request(
        MCP_URL, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        payload = json.load(response)
    if "error" in payload:
        raise RuntimeError(payload["error"])
    return payload["result"]


def wait_for_server(timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            call_tool("info")
            return
        except Exception:
            time.sleep(0.5)
    raise TimeoutError("perception MCP server did not become ready")


class SmokeNode(Node):
    def __init__(self):
        super().__init__("ocr_smoke_client")
        self.result = None
        self.publisher = self.create_publisher(CompressedImage, INPUT_TOPIC, 10)
        self.subscription = self.create_subscription(
            String, OUTPUT_TOPIC, self._on_result, 10
        )

    def _on_result(self, message: String) -> None:
        self.result = json.loads(message.data)

    def publish_image(self) -> None:
        image = np.full((160, 640, 3), 255, dtype=np.uint8)
        cv2.putText(
            image, "HELLO 123", (20, 105), cv2.FONT_HERSHEY_SIMPLEX,
            2.0, (0, 0, 0), 4, cv2.LINE_AA,
        )
        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            raise RuntimeError("failed to encode smoke JPEG")
        message = CompressedImage()
        message.format = "jpeg"
        message.data = encoded.tobytes()
        self.publisher.publish(message)


def main() -> None:
    wait_for_server()
    call_tool("start")
    rclpy.init()
    node = SmokeNode()
    deadline = time.monotonic() + 30
    try:
        while node.result is None and time.monotonic() < deadline:
            node.publish_image()
            rclpy.spin_once(node, timeout_sec=0.5)
        if node.result is None:
            raise TimeoutError("no OCR result received")
        if node.result.get("error"):
            raise RuntimeError(node.result["error"])
        if not node.result.get("items"):
            raise AssertionError(f"OCR returned no text: {node.result}")
        print(json.dumps(node.result, ensure_ascii=False))
    finally:
        node.destroy_node()
        rclpy.shutdown()
        call_tool("stop")


if __name__ == "__main__":
    main()
```

- [ ] **步骤 2：运行 perception 全量测试**

运行：

```bash
python3 -m unittest discover -s perception/tests -p 'test_*.py' -v
```

预期：全部测试 `OK`，且原有 ASR 29 项测试无回归。

- [ ] **步骤 3：编译所有变更 Python 文件**

运行：

```bash
python3 -m py_compile \
  perception/main.py \
  perception/plugins/asr.py \
  perception/plugins/ocr.py \
  perception/plugins/ocr_runtime.py \
  perception/utils/ocr_model_downloader.py \
  perception/tests/ocr_ros_smoke.py \
  tools/prepare_ocr_model_bundle.py
```

预期：退出码 0，无输出。

- [ ] **步骤 4：执行 Git 大文件和误提交模型审计**

运行：

```bash
git ls-files -z | xargs -0 stat -f '%z %N' | awk '$1 > 1048576 {print}'
git ls-files | rg '\.(onnx|engine|pth|pt|bin)$' || true
```

预期：第一条命令仅可能列出现有五个且各自小于 1 MiB 的音频 fixture，因此实际应无输出；第二条命令无输出。

- [ ] **步骤 5：提交 smoke 脚本并检查差异和提交状态**

提交：

```bash
git add perception/tests/ocr_ros_smoke.py
git commit -m "test(ocr): add Jetson ROS smoke check"
```

运行：

```bash
git diff --check
git status --short
git log --oneline --decorate -8
```

预期：`git diff --check` 无输出，工作树为空，历史中能看到 main merge 和四个 OCR 实现提交。

### 任务 7：准备内网模型并在 Jetson 验收

**文件：**
- 不提交模型文件；模型存放于 `/mnt/data/zengzhitao/embodied-ai/ocr/ppocrv6-tiny/`

- [ ] **步骤 1：在能访问 ModelScope 和 JuiceFS 的服务器准备模型**

运行：

```bash
python3 tools/prepare_ocr_model_bundle.py \
  --output-dir /mnt/data/zengzhitao/embodied-ai/ocr/ppocrv6-tiny
find /mnt/data/zengzhitao/embodied-ai/ocr/ppocrv6-tiny \
  -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
```

预期文件及大小约为：

```text
cls.onnx 585532 bytes
det.onnx 1829618 bytes
keys.txt 27156 bytes
rec.onnx 4489813 bytes
```

- [ ] **步骤 2：从服务端本机和 Jetson 验证 HTTP 完整读取**

服务端运行：

```bash
curl --noproxy '*' -fL --max-time 30 \
  http://127.0.0.1:34567/zengzhitao/embodied-ai/ocr/ppocrv6-tiny/keys.txt \
  -o /tmp/ocr-keys.txt
wc -c /tmp/ocr-keys.txt
```

Jetson 运行：

```bash
curl --noproxy '*' -fL --max-time 30 \
  http://172.28.4.81:34567/zengzhitao/embodied-ai/ocr/ppocrv6-tiny/keys.txt \
  -o /tmp/ocr-keys.txt
wc -c /tmp/ocr-keys.txt
```

预期：两边均下载成功且输出 `27156` 字节；若出现 HTTP 200 后 0 字节，先修复 JuiceFS I/O，再进行 Docker build。

- [ ] **步骤 3：在 Jetson 做 no-cache 镜像构建**

在全新 clone 中运行：

```bash
git clone --branch feat/zengzhitao --single-branch \
  https://github.com/4paradigm/phanthymotus.git phanthymotus-ocr-check
cd phanthymotus-ocr-check
docker build --no-cache --network=host \
  -f perception/Dockerfile.jetson \
  -t phanthymotus-perception:ocr-check .
```

预期：RapidOCR 安装、四个模型下载、CPUExecutionProvider 检查和最终镜像构建全部成功。

- [ ] **步骤 4：检查最终镜像模型体积与 provider**

运行：

```bash
docker run --rm phanthymotus-perception:ocr-check \
  /bin/bash -lc "du -cb /models/ocr/ppocrv6-tiny/* | tail -1; \
  python3 -c 'from pathlib import Path; import rapidocr; root=Path(rapidocr.__file__).resolve().parent / \"models\"; print(list(root.glob(\"*.onnx\")))'; \
  python3 -c 'import onnxruntime as ort; print(ort.get_available_providers())'"
```

预期：模型总量小于 `15728640` 字节；RapidOCR wheel 模型列表输出 `[]`；provider 列表包含 `CPUExecutionProvider`。

- [ ] **步骤 5：运行容器并完成一张 JPEG 的 OCR 端到端验证**

运行：

```bash
docker run -d --rm --network host \
  --name phanthymotus-ocr-check \
  phanthymotus-perception:ocr-check
docker cp perception/tests/ocr_ros_smoke.py \
  phanthymotus-ocr-check:/tmp/ocr_ros_smoke.py
docker exec phanthymotus-ocr-check /bin/bash -lc \
  'source /opt/ros/humble/install/setup.bash && \
   source /ros_ws/install/setup.bash && \
   python3 /tmp/ocr_ros_smoke.py'
docker stop phanthymotus-ocr-check
```

预期：smoke 脚本退出码为 0，并输出一个 JSON 对象，结构包含：

```json
{
  "text": "HELLO 123",
  "items": [
    {"text": "HELLO 123", "bbox": [0, 0, 1, 1], "score": 0.5}
  ],
  "timestamp": 1720000000.0,
  "language": "zh"
}
```

示例值仅表示字段类型；实际文字、`bbox`、`score` 和时间戳由图片与推理结果决定。验收条件是 `items` 非空、bbox 为原图像素坐标、无 `error`、服务进程保持运行。若命令失败，先运行 `docker logs phanthymotus-ocr-check` 保存服务端异常，再停止容器。

- [ ] **步骤 6：记录性能准入数据**

在连续 OCR 请求期间运行：

```bash
tegrastats --interval 1000
```

同时记录首张及后续图片响应耗时。预期：模型目录小于 15 MiB、GPU 利用率低于 10%、请求失败率为 0；CPU ONNX Runtime 路径不应主动占用 GPU。

---

## 完成标准

- `origin/main` 的 restart 和 SenseVoice 更新已合入，但 ASR 默认仍是 internal official Paraformer offline。
- `ocr` 默认启用并保持 `start/stop/info/config`、`CompressedImage -> String JSON` 契约。
- OCR 输出包含全文、逐框文字、原图像素 bbox、score、timestamp 和 language。
- Git 中没有 ONNX 或超过 1 MiB 的新增文件。
- 最终镜像只保留约 6.61 MiB 的 PP-OCRv6 tiny bundle。
- 本地全量测试、Jetson no-cache build 和 JPEG 端到端验收均通过。
