# 具身智能-感知能力-OCR中英文

本榜单目标为对标手机端模型大小，快速将ocr的能力落到具身智能的perception层。此榜单更加关注通用场景下的文本框定位并文本识别能力。
资源限制与提交方式
- 运营硬件：NVIDIA  jetson orin-16G（约100 TOPS算力，Ampere架构，同3090）要求gpu占用10%以下，模型限定15M以下。
- 提交方法：
①代码准备：科学家本地开发后，在这里拉分支开发https://github.com/4paradigm/phanthymotus/tree/main
！请务必注意不要上传任何大于1mb的文件！模型放在juicefs上，然后镜像里通过http://172.28.4.81:34567/下载
【分支权限申请请联系xieaobo】

输入输出
输入：图片，JEPG/jpg等常见图片格式
输出：识别文字框的坐标、提取后的文本
评测标准
排序指标
F1均值：每个case的MAPcoco，F1_score，TEDS，BLEU均值，求和后除以case数
监测指标：
- case数
- 文本F1_score@NLCS
- 标注框识别准确率： 预测内容与标准内容逐字符完全一致的框数 / 标注框总数（排除 Table 类型）
- 字符集准确率：不区分顺序，预测字符只要在标准文本中存在就计数
- 平均响应时间(s)
- 请求失败率
- 生成速度（单词/分钟）： 英文单词数 + 中文字符数 + 数字个数 / 识别时间
$$F1 = 2 \frac{precision \times recall}{precision + recall} = \frac{2TP}{2TP + FP + FN}$$
计算 precision, recall 针对的是每个 bbox 的 prediction 与 ground truth 是否匹配。定义匹配为：
$$match(pred, gt) = NLCS(pred, gt) \gt \text{nlcs\_thresh}$$
其中 Normalized LCS 的定义如下，最长公共子串占整体长度。榜单阀值0.7
$$NLCS(pred, gt) = \frac{LCS(pred, gt)}{\max(|pred|, |gt|)}$$




描述
榜单介绍：https://my.feishu.cn/docx/ArRJdSaQPoZwUtxuS5Rch43xnwb

参考代码：https://github.com/4paradigm/phanthymotus/blob/feat/wanglimin-test/perception/plugins/ocr.py

重要：此榜单不要在平台上手动停止任务，如需停止任务请联系王利民处理

https://github.com/4paradigm/phanthymotus/blob/main/perception/main.py 中的第394~395行端口需修改为从环境变量中获取

mcp_port = int(os.environ.get("MCP_PORT") or cfg.get("mcp_port", 15720))
ws_port = int(os.environ.get("WS_PORT") or cfg.get("ws_port", 15721))

在第95行增加一下代码
if plugins_cfg.get("ocr", {}).get("enabled", False):
    from plugins.ocr import OCRPlugin
    self._plugins.append(OCRPlugin(plugins_cfg["ocr"], executor))
    log.info("OCRPlugin loaded")
将配置文件https://github.com/4paradigm/phanthymotus/blob/feat/wanglimin-test/perception/config/fastdds_large_message.xml copy到项目的perception/config/fastdds_large_message.xml

config.yaml中需要增加ocr配置

plugins:
  ocr:
    enabled: true
    provider: ""
    url: ""
    key: ""
    model: ""
    language: ""
提交方式(commit id)：

env:
  - name: PHANTHYMOTUS_COMMIT_ID
    value: "xxxxxx"

---

# 当前核心需求：离线中英文 OCR

## 目标

- OCR 必须能够在 NVIDIA Jetson Orin 16G 上离线运行，不依赖 OpenAI、Qwen、Google 等外部云端 API。
- 支持中文、英文、数字及常见中英文混排场景。
- 输入为 JPEG/JPG/PNG 等常见图片，输出每个文本区域的坐标和识别文字。
- 模型大小限定在 15M 以下，GPU 占用低于 10%。
- 模型文件不提交到 Git 仓库，通过 JuiceFS 或内部 HTTP 服务在部署时下载。
- 优先保证通用场景下的文本检测和文本识别能力，后续再根据榜单指标优化表格等特殊场景。

## 当前已完成的工程接入

- `perception/main.py` 已支持从 `MCP_PORT`、`WS_PORT` 环境变量读取端口。
- `perception/main.py` 已接入 `OCRPlugin`。
- `perception/config.yaml` 已增加 `plugins.ocr` 配置。
- `perception/config/fastdds_large_message.xml` 已加入仓库。
- Jetson Dockerfile 已将 FastDDS XML 复制到镜像。
- `perception/plugins/ocr.py` 已具备 ROS2 图片订阅、OCR 结果发布及 plugin 生命周期框架。

## 后续部署遗留排查清单

### P0：离线 OCR 方案

- [x] 确定最终离线 OCR 模型和推理框架：PP-OCRv6 Tiny ONNX +
  ONNX Runtime CPU。
- [ ] 确认 15M 限制指模型权重大小、参数量，还是完整 OCR pipeline 大小。
- [x] 同时核算文本检测模型和文本识别模型的总大小：官方 ONNX
  文件合计约 6MB。
- [ ] 确认是否允许纯 CPU 推理；若使用 GPU，验证平均及峰值 GPU 占用低于 10%。
- [ ] 确认模型在 JuiceFS/内部 HTTP 服务中的下载地址、保存目录、版本和校验值。
- [ ] 为 OCR 模型补充自动下载、文件完整性校验和下载失败处理。
- [ ] 确认 Jetson aarch64 环境下推理框架可用，例如 ONNX Runtime、TensorRT 或其他轻量运行时。
- [x] 默认 provider 已替换为 `local_ppocr`。
- [ ] 如果临时使用 Tesseract，需要在镜像中安装 `tesseract-ocr`、中文/英文语言包和 `pytesseract`，并验证模型大小与识别精度。

### P0：评测输入输出契约

- [ ] 确认评测通过 ROS2 topic、MCP、HTTP，还是离线图片文件调用 OCR。
- [ ] 确认输入图片支持的格式、最大文件大小和最大分辨率。
- [ ] 确认 bbox 使用像素坐标还是归一化坐标。
- [ ] 确认 bbox 格式是 `[x1, y1, x2, y2]` 还是四点多边形。
- [ ] 确认旋转文字框的表示方式。
- [ ] 确认文本框粒度：字符、单词、文本行或文本块。
- [ ] 确认输出 JSON schema、字段名称以及是否需要 request ID、时间戳和图片尺寸。
- [ ] 确认检测框与 GT 的匹配算法及 IoU 阈值。
- [ ] 确认 NLCS 阈值 `0.7` 的边界条件是 `>` 还是 `>=`。
- [ ] 确认 Table case 是否要求输出 HTML、单元格行列关系等结构化结果，以支持 TEDS。

### P1：FastDDS 和部署配置

- [x] 运行时已设置：
  `FASTRTPS_DEFAULT_PROFILES_FILE=/opt/phanthy-motus/config/fastdds_large_message.xml`。
- [x] 已移除 `FASTDDS_BUILTIN_TRANSPORTS=DEFAULT`，避免覆盖 XML 中的
  transport 配置。
- [ ] 验证大尺寸 `CompressedImage` 能否通过 ROS2 稳定传输。
- [ ] 确认榜单只构建 `perception/Dockerfile.jetson`，还是普通 `perception/Dockerfile` 也需要复制 FastDDS XML 和安装 OCR 依赖。
- [ ] 确认评测平台是否自动注入 `MCP_PORT`、`WS_PORT`。
- [ ] 如平台不注入，在部署配置中显式声明端口环境变量。
- [ ] 视需要在 `config.yaml` 中补充 `ws_port: 15721`。
- [ ] 确认模型目录是否需要以 volume 方式挂载，以及容器是否具备下载和写入权限。

### P1：现有 OCR plugin 问题

- [ ] 修正 OpenAI adapter 中无条件按 Gemini `[0,1000]` 坐标转换 bbox 的逻辑；即使最终使用离线方案，也应避免保留错误实现。
- [x] 已从配置 schema 中删除尚不可用的 `google` provider。
- [ ] OCR 默认启用但没有有效 adapter 时，应在启动或配置阶段给出明确错误。
- [x] 本地 provider 已支持实例级配置合并，并提供模型目录、线程数、
  检测和识别阈值等参数。
- [x] 已对 bbox 做类型检查、坐标排序和越界裁剪。
- [x] 已实现从上到下、从左到右的文本行排序。
- [x] 输出已保留识别置信度和检测置信度。

### P1：性能与稳定性

- [ ] 设置最大 OCR FPS 或最小处理间隔，避免摄像头每帧都执行 OCR。
- [ ] 明确队列满时的丢帧策略，优先处理最新帧。
- [ ] 设置单张图片处理超时，并在超时后返回明确错误。
- [ ] 记录每张图片的处理时间、成功/失败状态和字符数。
- [ ] 统计平均响应时间、请求失败率和生成速度。
- [ ] 在 Jetson Orin 16G 上实测 CPU、GPU、显存和内存占用。
- [ ] 使用不同分辨率、中英文比例和文本密度的图片进行压力测试。

### P1：测试与验收

- [x] 已增加离线 adapter 纯逻辑单元测试。
- [x] 已增加 bbox 后处理和坐标转换测试。
- [ ] 增加中文、英文、数字、中英文混排、空图片和无文本图片测试。
- [ ] 增加 ROS2 plugin 的 start、stop、info、config 生命周期测试。
- [ ] 增加模型下载成功、校验失败、断网和文件损坏场景测试。
- [ ] 准备固定小型验证集，离线计算 COCO mAP、NLCS F1、BLEU；如要求表格，再计算 TEDS。
- [ ] 在最终提交前执行类型检查、lint、测试和 Jetson 实机 smoke test。

## 推荐实施顺序

1. 确认评测调用协议、bbox 格式和 15M 计算口径。
2. 选定离线文本检测与中英文识别模型，并完成 Jetson 基准测试。
3. 实现本地 OCR adapter、模型自动下载和输出后处理。
4. 补齐 Dockerfile、FastDDS 和部署环境变量。
5. 增加测试与离线评测脚本。
6. 在 Jetson 实机验证精度、响应时间、失败率和资源占用。

## 2026-07-16 离线方案开发记录

- 已选择 `PP-OCRv6_tiny_det_onnx + PP-OCRv6_tiny_rec_onnx`。
- 官方 ONNX 文件合计约 6MB，满足当前“模型小于 15M”的文件大小要求。
- 使用 ONNX Runtime CPU provider，目标是避免占用 GPU。
- 已实现 `local_ppocr` provider，输出包含 `text`、`bbox`、
  `confidence` 和 `det_confidence`。
- 已实现模型按需下载，OCR 只允许使用 `model_base_url` 指定的内部
  HTTP 服务，不提供 Hugging Face 等公共模型源 fallback。
- 内部 HTTP 目录约定：
  - `det.onnx`
  - `rec.onnx`
  - `inference.yml`
- 本地待上传模型已保存到
  `perception/models/ppocr-v6-tiny/`，模型文件由 `.gitignore` 排除。
- 可使用 `perception/scripts/download_ocr_model.py` 验证服务器下载：

  ```bash
  python3 perception/scripts/download_ocr_model.py \
    --base-url http://172.28.4.81:34567/guohongjie/models/ppocr-v6-tiny \
    --output-dir /models/ppocr-v6-tiny
  ```
- 模型地址通过 Docker 构建参数 `OCR_MODEL_BASE_URL` 指定，并写入同名
  环境变量；容器运行时可再次覆盖。

  ```bash
  docker build \
    --build-arg OCR_MODEL_BASE_URL=http://172.28.4.81:34567/guohongjie/models/ppocr-v6-tiny \
    -f perception/Dockerfile.jetson \
    .
  ```
- 暂不启用方向分类、文档矫正和表格结构识别，以控制模型大小和资源占用。

### 部署后必须实测

- [ ] Jetson 镜像内 `onnxruntime` aarch64 wheel 是否可以正常安装和加载。
- [ ] 官方 ONNX 模型在 Jetson Python 3.8 环境下的算子兼容性。
- [ ] CPU 单线程/双线程响应时间和内存占用。
- [ ] 持续运行时 GPU 占用是否保持低于 10%。
- [ ] 不同图片分辨率下的检测召回率和 OCR FPS。
- [ ] 中文、英文、中英文混排和小字号场景的榜单指标。
- [ ] 内部模型地址可用性、下载速度和容器目录写权限。
