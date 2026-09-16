# OCR 显式 ONNX CPU 后端

## 范围与原因

用户授权公共 Perception 修改并在验证后提 PR。现有 OCR 只能加载 JetPack 专用 TensorRT engine，阻塞 x86 仿真环境验证；不修改 Agent Core、ActuCore，也不将仿真专用代码混入本分支。

## 实施

1. 核对现有 PP-OCRv6 对应 ONNX 权重、版本、许可证和校验值。
2. 复用现有图像预处理、DB/CTC 后处理、旋转分类、裁剪增强及卡片生命周期，增加显式选择的 ONNX CPU 推理路径；默认仍为 TensorRT，禁止静默降级。
3. 复用模型目录约束与下载校验。CPU 模型使用独立目录，不覆盖 Jetson engine。
4. 补充后端选择、形状/归一化、异常和默认路径回归测试，更新 Perception 使用说明。

## 验证与交付

- 本地：现有 OCR 测试及新增 CPU 路径测试。
- 23 号机隔离进程：远端下载权重，真实 CPU 识别有字图、无字图和损坏输入，记录耗时与资源限制。
- 仿真集成：确认画布无其他使用者后测试相机→OCR 的 ROS 输出及 WebUI 可见结果，恢复原有画布状态。
- 验证后范围化提交并创建独立 PR；Jetson 未实测时明确说明，不以 CPU 验证代替 Jetson 硬件验收。

## 当前状态

方案与本分支范围一致。已实现显式 `onnx-cpu`、固定哈希模型下载及现有处理流程复用；README 和 config.yaml 已同步后端选择说明，其他模块文档无需修改。

2026-09-16 验证：语法检查、`git diff --check` 通过；在隔离 x86 容器内运行 `python3 -m pytest -p no:cacheprovider tests/test_ocr_plugin.py tests/test_ocr_onnx.py tests/test_vision_runtime.py -q`，66 项通过（5.30 秒）。本机未安装 pytest，因此未声称本机 pytest 通过。

真实模型已在远端下载并通过固定大小/哈希校验（`OCR_ONNX_WEIGHTS_VERIFIED`）。CPU 测试镜像的 pyvips-binary 下载被截断，哈希校验拦截后改为远端断点续传，不绕过校验。三个 OCR 业务文件的补丁已通过现有仿真 Perception 基线的 `git apply --check`，后续集成只应用该补丁，不顺带升级其他卡片。

隔离 CPU 模型链已对 MuJoCo 相机渲染图完成真实推理：`SIM ROOM`，score=0.99798，bbox=[107,92,215,112]；输出 `OCR_WORLD_SIGN_MODEL_PASS`（2 CPU、2 GiB、无网络）。此检查直接输入解码后的图像数组，不能替代完整图片入口验收。

后续依赖完整下载且 SHA-256 验证通过，最小补丁候选镜像 `phanthymotus-sim/ocr:onnx-compat1` 构建成功（image short ID `4eb60e1d684c`）。完整 `verify_ocr_cpu.py` 在旧基线加补丁、新主线 PR 源码两种组合均通过有字/空白/损坏图片检查；有字识别分别约 0.126/0.147 秒（非性能基准）。完整图片入口也识别出渲染图的 `SIM ROOM`，输出 `OCR_WORLD_SIGN_FULL_ENTRY_PASS`。

23 号机已部署最小补丁候选，真实 producer-first ROS 验收通过：`SIM ROOM`，score=0.99293；停止后 OCR 和相机均 idle。其余 ASR→TTS、VOP、空人脸联合 ROS 回归通过。Chrome 中七张卡片（两张 sensor、五张 processor）经页面开始按钮启动，OCR 数据流持续显示 `SIM ROOM`，截图已人工检查；随后页面停止并恢复原八张卡片、释放编辑锁。

运行源码核对 main/plugins/utils 共 43 个 Python 文件，只有本 PR 的 `ocr.py`、`ocr_runtime.py`、`model_downloader.py` 与部署基线不同，没有新增业务文件。Core/ActuCore 镜像未改；四服务 restart_count=0。Jetson 硬件回归未验证，尚未提交或创建 PR。
