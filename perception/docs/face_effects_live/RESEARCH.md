# face_effects_live 技术调研与选型

核查日期：2026-11-16。优先核查官方文档、官方源码和作者论文。闭源 SDK 只描述公开接口及能力，不推断内部网络、训练数据或关联算法。下列“支持”是官方产品声明或工程可移植性，实际验证另列于验收报告；Android ARM64 不等于 Linux ARM64。

## 商业引擎

| 产品 / 公开证据 | 检测、关键点及跟踪 | 网格、配饰、遮挡与渲染 | Linux x86_64 / ARM64 / Jetson / Windows | 获取与部署结论 |
|---|---|---|---|---|
| 抖音 / 火山智能美化特效 | 官方列出人脸检测、多脸、106/280 点；闭源模型与 ID 关联未知 | 贴纸与图形特效；OpenGL 纹理输入输出，EGL 上下文示例 | Linux 文档默认 x86，位宽以交付包为准；其他架构需厂商适配；Jetson未确认；Windows有产品入口 | 商业鉴权与素材授权；未安装，不能视为默认可分发依赖 |
| 字节公开技术接口 | 106 点与扩展关键点配置、算法与资源句柄公开可核查 | 官方建议纹理输入降低 CPU/GPU 传输，YUV 可由 shader 转换；上下文有线程归属 | 复用上一行限制；公开接口不构成 ARM64 二进制保证 | 参考“检测/渲染分层、资源句柄、纹理管线”，不声称复现抖音内部算法 |
| Snapchat Lens Studio / Camera Kit | 人脸关键点、Head Binding、多脸 Face Index；稳定 ID 算法未公开 | Face/Head Mesh、Face Occluder、3D 配饰及材质；Web 性能依特效复杂度变化 | Camera Kit公开平台为iOS/Android/Web；Linux浏览器须另测；原生Linux/Jetson包未确认；Windows浏览器及编辑器分开评估 | 需账号/token、应用审核及条款；不直接引入 |
| TikTok Effect House | Face Binding 按检测顺序分配索引；Head Tracker | 3D Head、网格、Occluder 材质、资源面板 | 特效创作与 TikTok 播放生态；未找到可独立部署 Linux ARM64/Jetson 运行库的可靠保证 | 索引不是跨遮挡身份保证；不将创作工具视为可嵌入开源SDK |
| Banuba Face AR | C++ EffectPlayer 执行识别，Input/Player/Output 分层、异步帧消费 | 效果目录包含脚本和资源，支持多种输出 | 官方列表含iOS/Android/Web/macOS/Windows等；未确认原生Linux x86/ARM/Jetson交付 | 商业SDK；报价及资源授权单独核查；只借鉴输入/引擎/输出解耦 |
| DeepAR | 商业端到端 AR SDK；内部检测与跟踪未知 | Studio管理纹理、动画、物理与特效 | 官方系统要求列iOS/macOS、Android、Web；未列Linux原生、Jetson、Windows原生 | 商业license；第三方依赖开源许可证不等于SDK本身开源 |
| Luxand FaceSDK | 官方70点、跟踪接口；偏人脸分析 | 可作为定位组件，完整AR渲染需另建 | 官方下载文档列Linux 32/64、Linux ARM、Windows；Jetson还需具体系统适配 | 商业授权；作为ARM候选备选，未安装或性能验证 |

官方来源（对应上表，不含营销FPS比较）：

- 火山：[SDK概述](https://docs.volcengine.com/docs/intelligent-beautification-effects/sdk-overview?lang=en)、[Linux接入](https://docs.volcengine.com/docs/intelligent-beautification-effects/linux?lang=zh)、[关键点配置](https://docs.volcengine.com/docs/Intelligentbeautificationeffects/Mobileendv480AIeffectsplatformEboxcreativesconfigurationguide)、[数据预处理](https://docs.volcengine.com/docs/intelligent-beautification-effects/data-pre-processing?lang=en)、[鉴权/购买](https://docs.volcengine.com/docs/intelligent-beautification-effects/product-quotation?lang=en)。规格页的iPhone检测耗时不用于本机预算。
- Snap：[Camera Kit平台](https://developers.snap.com/camera-kit/getting-started/what-is-camera-kit)、[多脸](https://www.developers.snap.com/lens-studio/4.55.1/references/guides/lens-features/tracking/face/working-with-multiple-faces)、[头部配饰](https://www.developers.snap.com/lens-studio/4.55.1/references/guides/lens-features/tracking/face/face-effects/head-attached-3d-objects)、[Web限制](https://www.developers.snap.com/camera-kit/integrate-sdk/web/guides/web-considerations)、[发布审核](https://www.developers.snap.com/camera-kit/app-review/release-app)、[Camera Kit条款](https://www.snap.com/terms/snap-camera-kit)。条款应在采购时重新核查。
- TikTok：[Face Binding](https://effecthouse.tiktok.com/learn/guides/workspace/components/ar-capability/face-binding)、[3D Head](https://effecthouse.us.tiktok.com/learn/guides/workspace/objects/face-effects/3d-head)、[Occluder材质](https://effecthouse.tiktok.com/learn/guides/workspace/assets/material/material-overview)。未将 TikTok 内部算法归因于任何开源模型。
- Banuba：[架构](https://docs.banuba.com/far-sdk/)、[API](https://docs.banuba.com/far-sdk/tutorials/development/api_overview/)、[支持平台](https://www.banuba.com/faq/what-platforms-does-your-sdk-support)。
- DeepAR：[官方介绍](https://docs.deepar.ai/deepar-sdk/introduction/)、[系统要求](https://docs.deepar.ai/deepar-sdk/specification/system-requirements/)、[第三方许可证](https://docs.deepar.ai/deepar-sdk/useful-information/third-party-licenses/)。
- Luxand：[官方8.3文档](https://www.luxand.com/download/Luxand_FaceSDK_Documentation_83.pdf)。

## 按层比较开源与可部署技术

| 层 | 候选 | 本项目价值与限制 | 许可证 / 维护情况核查 |
|---|---|---|---|
| 检测+稀疏点 | OpenCV YuNet | 一次得到box/5点；便宜、多人可用；不提供表情网格；本轮CPU实际运行 | 模型目录MIT，OpenCV另有Apache-2.0；Zoo仍有维护，固定2023mar模型及SHA |
| 稠密定位 | OpenSeeFace MobileNetV3 ONNX | 66点、多种模型大小；可替换检测器；模型输出热图+offset；本轮model0/model3真实运行 | 作者明确代码和模型BSD-2-Clause；社区维护，非厂商SLA；无需训练 |
| 密集网格 | MediaPipe Face Landmarker / legacy Face Mesh | 官方478点、blendshape和变换矩阵；旧Face Mesh常见468点。多脸时不能假定内置单脸平滑仍有效 | 第二轮已锁定官方float16/1模型和0.10.21 Tasks，模型卡明确Apache-2.0；真实CPU全画面VIDEO和YuNet ROI IMAGE均运行，见ROUND2 |
| 传统定位 | OpenCV Facemark LBF / Kazemi / AAM；dlib | CPU可用，适合基线；侧脸与表情上限、贡献模块和权重来源增加部署工作 | OpenCV API有许可不代表任意LBF权重可商用；dlib 68点常见训练数据有商业限制，未引入 |
| 人脸分析 | InsightFace SCRFD / landmark组件 | 检测与定位可用；身份embedding对本Card非必要；不复用现有识别模型包 | 代码MIT，官方预训练模型限制非商业研究；未安装/使用，默认排除 |
| 其他稠密定位 | 3DDFA_V2、face-alignment (FAN)、PFLD、pytorch_face_landmark | 前者更适合3D拟合，FAN质量候选，PFLD轻量候选；需核实权重及3D模板/训练数据条款 | 作者仓库/论文可追溯；不把代码MIT直接延伸到BFM及任意权重；本轮仅研究 |
| 推理运行时 | ONNX Runtime | CPU/CUDA/TensorRT EP接口统一；真正创建session和检查实际provider；能力枚举不代表模型算子全在GPU | MIT；活跃；本机CPU 1.23.2已测，CUDA/TRT未测 |
| 推理加速 | TensorRT | NVIDIA目标优化；engine与驱动/硬件/JetPack有匹配成本；不应成为CPU运行前提 | 产品运行库/分发条款与开源组件分别核查；本轮通过ORT预留EP，非原生TRT engine实现 |
| 视频管线 | NVIDIA DeepStream / GStreamer | 多路解码、batch与元数据、设备内存路径；大型多流部署候选，V1单路依赖较重 | NVIDIA SDK有产品许可，不能整体称开源；开源GStreamer组件各自许可；未部署 |
| 视觉加速 | NVIDIA VPI | 图像转换、光流等异构CPU/CUDA/PVA/VIC/OFA后端；不是人脸模型，也不是完整AR引擎 | NVIDIA产品许可；依具体平台；未部署 |
| 图形接口 | OpenGL ES + EGL | ARM嵌入式常见；纹理/FBO/alpha/depth适合长期实现；GL context须线程归属 | Khronos API规范不等于驱动许可证；GLES运行路径仅设计 |
| 图形接口 | Desktop OpenGL + EGL | 本轮ModernGL实现相同精灵命令；纹理缓存、预乘alpha、显式关闭；仅llvmpipe软件实际验证 | ModernGL MIT；需要GL3.3和EGL；Jetson不保证此Python封装默认成功 |
| 图形接口 | Vulkan | 适合明确控制同步、内存和批处理；开发/驱动兼容成本高于V1需求 | Khronos标准；实现和驱动许可另核查；仅研究，不声称已实现 |
| 跟踪/稳定 | 几何关联、LK、One Euro、Kalman | 几何多目标关联+速度预测，LK双向误差检查，One Euro按速度改变平滑；Kalman为替换选项 | 本轮自有实现除OSF适配出处外无外部跟踪器代码；无外观身份识别保证 |

主要来源：[YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)、[OpenSeeFace及模型许可证](https://github.com/emilianavt/OpenSeeFace)、[MediaPipe功能](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)、[MediaPipe Python平台](https://developers.google.com/edge/mediapipe/solutions/setup_python)、[InsightFace许可](https://github.com/deepinsight/insightface#license)、[OpenCV face模块](https://docs.opencv.org/5.0/extra_modules/face.html)、[3DDFA_V2](https://github.com/cleardusk/3DDFA_V2)、[FAN](https://github.com/1adrianb/face-alignment)、[PFLD论文](https://arxiv.org/abs/1902.10859)、[其他定位](https://github.com/cunjian/pytorch_face_landmark)、[ORT EP](https://onnxruntime.ai/docs/execution-providers/)、[TensorRT矩阵](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html)、[DeepStream](https://docs.nvidia.com/deepstream/text/DS_Overview)、[VPI](https://docs.nvidia.com/vpi/index.html)、[EGL](https://registry.khronos.org/EGL/)、[GL/GLES](https://registry.khronos.org/OpenGL-Refpages/)、[Vulkan](https://docs.vulkan.org/samples/latest/samples/vulkan_basics.html)、[One Euro作者资料](https://gery.casiez.net/1euro/)。

## 三套可运行组合与最终选择

| 组合 | 推理 / 跟踪 / 渲染 | 视觉及部署权衡 | 结论 |
|---|---|---|---|
| A 轻量 | YuNet 5点 + 几何关联/One Euro + OpenCV ROI | 依赖最少；贴纸/2D眼镜/轨道动画可用；不支持稠密表情或PnP模板拟合 | 保留降级：显式不配置landmark_model，不因缺CUDA而失败 |
| B 质量候选 | YuNet + OSF model3 66点 + 相同跟踪/渲染 | 模型更大、定位耗时更高；模型作者定位为质量档，样例置信度更高不等同真实精度更高 | 已真实推理比较；真实侧脸与网格方案比较未完成 |
| C 平衡 | YuNet + OSF model0 66点 / ORT CPU + 几何关联/One Euro + CPU ROI | 许可明确、轻量、Linux易部署，保留PnP及插件扩展；无需专有GPU | **V1主方案**；选择是工程权衡，未证明全球精度最优 |

相同720p输入、同一机器、同一确定性照片变换、独立顺序进程，比较见 [实测报告](ACCEPTANCE.md) 与 JSON。稠密网格质量升级候选为MediaPipe/3DDFA；没有强行要求某框架获胜。B与C都不提供真实3D遮挡网格，V1眼镜是保守退场的2D配饰。

## 质量与兼容性对比矩阵

“待实测”指缺同输入标注/设备证据，不能用关键点数量、置信度或厂商宣传代替。

| 指标 | A | B | C | MediaPipe网格升级候选 |
|---|---|---|---|---|
| 点数 / 点误差 | 5 / 待实测 | 66 / 待实测 | 66 / 待实测 | 478 / 待实测 |
| 快速真实转头 / 侧脸 | 待实测 | 待实测 | 待实测 | 待实测 |
| 多脸 | 合成双脸真实推理 | 同左 | 同左 | API支持，多人跟踪待实测 |
| 两人交叉 / ID切换 | 合成几何关联测试；真实待测 | 同左 | 同左 | 需另加稳定ID，待实测 |
| 遮挡恢复 | 人工覆盖照片+几何轨迹测试，真实待测 | 同左 | 同左 | 待实测 |
| 时序抖动 / 跟随延迟 | 合成滤波探针；真实待测 | 同左 | 同左 | 多脸需自行稳定，待实测 |
| 处理延迟P50/P95/P99 | 本机有JSON实测 | 本机有JSON实测 | 本机有JSON实测 | 待实测 |
| 720p输入FPS / CPU / RAM | 本机有JSON实测 | 本机有JSON实测 | 本机有JSON及60s回放 | 待实测 |
| GPU使用率 / sensor-to-display | 待实测 | 待实测 | 待实测 | 待实测 |
| x86_64 Linux | WSL2 CPU实际通过 | 同左 | 同左 | 第二轮WSL2 CPU实际运行，非GPU |
| ARM64 Linux | OpenCV构建后待实测 | ORT+OpenCV构建待实测 | 同左 | 官方安装文档列aarch64，具体wheel待验证 |
| Jetson | CPU设计兼容 | CUDA/TRT EP待实测 | 同左 | GPU delegate/JetPack待实测 |
| Windows | OpenCV设计兼容 | ORT设计兼容 | 同左 | 官方支持，未实测 |
| 特效 | 2D贴纸/眼镜/动画 | 同左+稠密定位/PnP | 同左 | 可扩展网格/表情，渲染另建 |
| 成本 | 最低；模型约0.23MB | 约13.5MB定位模型，算力更高 | 约1.90MB定位模型，适中 | 新运行时及模型资源核查成本 |
| 维护 | Zoo/OpenCV | 社区OSF/ORT | 同左 | Google AI Edge |

## 优化取舍

- 先按阶段计时，再考虑分频。已实现并测逐帧 vs 每3帧检测+LK；后者平均成本有改善但尾延迟和恢复更差，不作为默认。
- One Euro 在归一化坐标工作。固定种子静止噪声及阶跃探针比较beta=1.5/4/8/12；选8，90%阶跃响应从100ms降至33.3ms，静止标准差仍约为原始48%。这不是自然视频质量证据。
- Kalman可联合位置/速度/协方差，但需调动态模型和遮挡门限；本轮没有实现Kalman或声称优于它。没有加入无证据的长时间预测。
- 帧槽深度1、主动丢旧、结果发布前再次检查年龄；不以累计后台FPS隐藏延迟。
- ROI alpha避免全帧逐像素Python循环；纹理缓存有界。GL读回成本使GPU未必在小贴纸上获益，当前无硬件证据，不默认启用。
- OSF模型动态batch维已核查；本轮每脸独立调用，未实现/宣称batch加速。零拷贝、深度遮挡、原生TensorRT、自适应频率均未实现。

## 与 privacy_view 的只读技术对照

只读查看了独立工作区中的 privacy_view、privacy_runtime、privacy_temporal 及V1.1报告。两者可共享输入协议、时间基准定义、有界队列和阶段计时思想；未导入、修改或直接复用隐私策略代码。privacy_view侧重发布门控、遮挡风险和马赛克覆盖，本Card可以在无法定位时输出无贴纸原图，因此隐私安全不等价。已有privacy基准包含模拟检测器，不能与本轮真实YuNet/OSF耗时相减得出性能胜负。未来需用同源真实视频、同设备、相同decode/encode边界分别报告特效质量和隐私覆盖。

## 调研限制

网页工具可访问多数官方资料；直接下载最初被沙箱阻止，经授权后成功。GitHub匿名API返回403，未伪造模型commit pin：模型以实际字节数和SHA256固定，下载地址虽跟随分支，内容变化会拒绝。错误历史快照链接返回404后改用官方模型路径。DeepAR部分旧Linux/报价URL不可用，以可访问系统要求为准。没有采购、安装或使用闭源SDK；没有上传任何本地图像或内部代码。

## 第二轮重新选型（2026-11-17）

上面的照片合成比较属于第一轮历史证据。第二轮按相同自然帧、相同CPU/渲染/线程和独立顺序进程，实际比较原OSF0、修正OSF0、OSF3、YuNet五点、官方MP全画面VIDEO及YuNet+MP大脸ROI/OSF小脸。输入与源码SHA、条件误差和漏失率一起报告，详见[ROUND2](ROUND2.md)。

新增依据：Google [Face Mesh V2模型卡](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf)明确模型Apache-2.0许可、478点定义及模型限制；[Tasks Python接口](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python)区分IMAGE/VIDEO/LIVE_STREAM及时间戳。用IMAGE方式逐ROI调用，避免把多个不同人的裁剪轮流送入隐藏的单脸VIDEO跟踪状态。[OpenSeeFace上游tracker](https://github.com/emilianavt/OpenSeeFace/blob/master/tracker.py)的66点mouth_corner特征使用58/62，不能套用68点的48/54。

结论仍保留C为默认。MP全画面对本组小脸/侧脸漏失明显，低耗时不能算等质胜出；ROI混合保留为可选质量档，尚未证明优于修正OSF0。OSF3在六脸片段关键点耗时约为OSF0两倍，人工标注草稿不足以支持质量收益；YuNet五点的近景眼线误差较大，但保留低依赖降级价值。以上是本机少量视频的工程选择，不能外推到所有人群或硬件，也不以置信度代替真实精度。未引入有非商业预训练限制的InsightFace权重或新增闭源SDK。

## 第四轮：独立公开标注与运行时约束

[ROUND4](ROUND4.md)记录了300-VW、WFLW、ChokePoint和PLUSFiaQ的官方许可/下载核查及COCO人工眼鼻参考的实际评测。取得327张许可筛选的COCO验证图像；其语义锚点代理误差、漏失与配饰眼点误差不等于完整68/98点基准，也不包含可信视频身份或三维姿态真值。未用模型生成的RAVDESS/OpenFace跟踪数据冒充人工标注；没有通过未知许可的数据镜像绕过作者限制。

同一COCO输入实际比较原始/当前OSF0、OSF3、MediaPipe和ROI混合，继续保留OSF0为默认，并提供非默认的有界上采样。扩展宿主回归还发现Card原来的独立ORT session违反已有隔离约定，已接入既有spawn ORT worker；热图解码在子进程执行以减少回复体积。算法输出一致性、实际性能开销和剩余缺口见本轮证据，未修改共享ORT worker或其他Card。
