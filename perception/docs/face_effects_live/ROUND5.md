# Face Effects Live ROUND5 算法优化与交付报告

本轮结论：**V1.0 NO-GO；候选算法有可复核的质量和性能收益，保留为本地候选版本 B。**

先完成全项目版本 A 的真实全量恢复，再创建分支 `feat/face-effects-live-v1-round5-20261118`。优化前可复现源码检查点 `76f3a08` 保留原有未提交/未跟踪实现；原分支及原始工作树快照均在 A 中。候选最终提交见 B 的 `VERSION_MAPPING.json`。没有合并其他 Card，也没有推送、创建 PR 或操作实体设备。

## 数据、许可及独立标注

本轮实际重新访问官方页面、作者仓库及机构来源。以下是逐项访问和许可结论；完整记录见 `face-effects-evidence/round5/dataset-investigation.json`。

|数据集|本轮访问/取得情况|许可与适用边界|
|---|---|---|
|[300-VW](https://ibug.doc.ic.ac.uk/resources/300-VW/)|official resource accessible; download still requires first/last name, email, affiliation|commercial permission not confirmed; no identity form submitted|
|[WFLW](https://github.com/wywu/LAB)|author repository read successfully via connected GitHub; project page HTTP404; author links Google Drive/Baidu images and annotation URL|code license is not image/annotation permission; commercial dataset grant not confirmed|
|[300W](https://ibug.doc.ic.ac.uk/resources/300-W/)|official terms inspected|research only, commercial use not allowed|
|[AFLW](https://www.tugraz.at/institute/icg/research/team-bischof/learning-recognition-surveillance/downloads/aflw/)|current official page accessible, request form available; older lrs URL HTTP404|explicit non-commercial research only; commercial derived data exploitation prohibited; no form submitted|
|[AFLW2000-3D](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)|author page timeout; official-maintainer 3DDFA/3DDFA_V2 GitHub README inspected|MIT code does not relicense AFLW-derived images/labels; AFLW prohibits commercial derived data exploitation|
|[BIWI Kinect Head Pose](https://vision.ee.ethz.ch/datsets.html)|ETH maintained listing accessible; old author page403/timeout|ETH listing says research only unless otherwise stated; commercial R&D permission not independently established; no third-party mirror used|
|[UPNA Head Pose](https://www.unavarra.es/gi4e/databases/hpdb)|tool denied by robots.txt; university-hosted thesis describes 120 videos/2D and 3D GT; no access restrictions bypassed|original dataset commercial license not obtained; thesis open-access license is not dataset permission|
|[MUCT](https://github.com/StephenMilborrow/muct)|author README.txt and landing/details read afresh; author warns against public face-image reproduction|no affirmative commercial grant verified; academic availability cannot be presumed commercial permission|
|[FGnet Talking Face](https://personalpages.manchester.ac.uk/staff/timothy.f.cootes/data/talking_face/talking_face.html)|author page timeout; institutional FGnet page502|no verified downloadable license/annotations; no unauthorized mirrors|
|[BioID / FGnet20](https://www.bioid.com/face-database/)|official downloads succeeded; original image, eye and20-point archives CRC checked; 1521 images/20 manual points and independent original .eye references retained|purpose-limited face-detection comparison research permission without further permission; not represented as a general commercial-product/data redistribution license|
|[White House 2014-06-25 Press Briefing](https://obamawhitehouse.archives.gov/photos-and-video/video/2014/06/25/press-briefing)|full260468410B transfer; first contiguous300s decoded to8991 native640x360 frames at source stride2|official source explicitly Public Domain; copyright policy confirms government-produced materials not copyright protected|
|[COCO2017 validation licensed subset](https://cocodataset.org/#download)|existing official raw annotations and327 licensed images preserved by versionA and actually inferred again|annotations CC BY4.0; selected photos CC BY2.0; per-image attribution/source retained|

BioID 从官方网站取得原始图像 ZIP、独立 `.eye` ZIP 和 FGnet20 ZIP，三个包全部通过 ZIP CRC，1521 张 384×286 PGM 全部解码、逐文件哈希。20 点为官方人工标注；映射固定为 0/1 双眼瞳孔、14 鼻、2/3 嘴角，只按画面左右排序成五锚点，未按模型结果改标。保留所有原始 20 点及多余 `bioid_1140a.pts`，使用精确同名的 1140 标注。`.eye` 与图像包内原始 `.eye` 逐一相同，单独评分。

许可是检测算法比较研究的用途许可，不冒充通用商业产品、训练或数据再分发许可。BioID 只用于本机受保护范围内的算法比较研究；本报告不重发人脸图像。OSF 眼廓均值与瞳孔并非相同语义，以下 NME 是五锚点 proxy，不能冒充标准 68/98 点榜单。原始坐标未经半像素/一像素拟合；官方网站未明确原点定义，相应系统偏差仍存在。

新取得官方 Public Domain 的 2014-06-25 新闻发布会完整 MP4（260468410B）。只将首连续 300s 按原 59.94Hz 的 stride2 转为 8991 帧 29.97Hz FFV1，保持原生 640×360，不放大冒充 720p。源、许可页面、原始包、转换工具、标签与 manifest 都保留。视频没有人工关键点、人物 ID 或三维头姿真值，只作为自然运动、连续性和性能诊断。另对 A 中原有 8 段自然视频共 1920 帧、327 张许可 COCO 图像重新推理；未沿用旧报告数字当作本轮结果。

## 基线复现与主要缺陷

同一机器、同一 Python/OpenCV/ORT 环境，OpenCV 两线程、同一输入 SHA256、检测尺寸 640、最多 8 脸、CPU 渲染；源模型及源码 SHA256 记录在每个 summary。静态图每张独立 Engine、两次相同观测只用于确认，不当作时序稳定性。动态视频逐帧计数，不丢弃困难帧。静态初始化/解码/写证据与处理耗时分列，不将吞吐容量当作直播 FPS。

真实 cProfile：1920 帧 40.416s，Engine 36.755s，关键点路径 19.986s，检测 10.429s，4654 次 session 调用。父进程 Queue 等待包含模型推理和调度，不能全算 IPC。原路径逐脸输入 602112B float32 张量，六脸每帧约 3.6MB，重复跨进程往返。

旧版实际复现 COCO：417/685 参考脸匹配，239/516 眼对渲染。小脸和大侧脸覆盖仍是主要缺口。原默认 OSF0 在 BioID 的五锚点 MAE 2.471px。PnP 一直是模板近似，没有独立三维角度真值，本轮不以重投影误差替代真实头姿误差。

## 实际实现与候选选择

1. **紧凑 ROI + 有界批量 RPC**：每帧最多 8 个原始 BGR ROI，一次 RPC 内依序调用原 batch=1 模型。小/中脸传 uint8 原始 ROI，大 ROI 仍在父进程处理以限制载荷。子进程严格保持先 float32、再 resize、再归一化的旧顺序，返回原有 66 点及置信度。ORT spawn 隔离、UUID session、独立卸载、崩溃重载均保留，没有引入宿主 ORT session。
2. **尺度和模型一致性守卫**：明确配置 `single_face_landmark_model=osf3.onnx` 后，先运行原 OSF0。仅恰好 1 个 proposal、快速结果有效、脸框短边≥96px且完整位于画面内时再运行 OSF3。只有五锚点两模型平均差≤快速模型眼距的0.04、单锚点最大差≤0.08时才采用精细结果；否则保留快速结果。96px扩展ROI约达到224输入的一半，避免过度放大小脸；分歧守卫防止不可靠模型替换。多脸仍全部保留。未删脸、未提高检测或评分阈值，也未跳帧。模型来源实际切换时保留轨迹ID并重置旧锚点滤波；仅决策原因变化时不误重置。
3. **按模型设置线程**：快速 OSF0 两线程并允许 spinning；OSF3 单线程、关闭 spinning，避免闲置大模型线程争抢。配置通过现有 worker 的 session options 下发。全局单线程和全局关闭 spinning 的候选均有尾部退化，未选为默认。
4. **有界单人像素跟踪桥接**：默认每帧检测。只有已确认的唯一轨迹、零检测 proposal 时才允许双向 LK + RANSAC，最多连续 2 帧且距最后模型帧 ≤100ms；失败立即停止。未使用静止旧贴纸填帧，不跨时间修复、分辨率变化、镜头切换或多脸状态。可设 `flow_bridge=false` 消融。

紧凑路径与旧路径在8段视频全部4654次人脸观测的box/五锚点/拓扑逐一完全相同；真实3ROI稠密66点测试也逐点相同。量化/整型resize未采用。在紧凑传输实验（尚未加入最终守卫）的六脸诊断中，请求平均约40799B，往返平均19.356ms，其中准备2.820ms、推理15.042ms、后处理0.808ms，剩余队列/复制/调度约0.685ms（P99 1.080ms）。该剩余项不是纯IPC。最终600s的前1200个实际输出也保留了同类诊断，见soak JSON，不能将局部样本P99当作全程P99。

保留失败实验：朴素 float32 批处理收益不稳定；眼角替代眼廓均值导致两套真值误差增加；半像素补偿使 BioID 变差且 COCO 眼镜 P99 增加；全局单线程造成六脸 P99 明显增加。最初不加守卫的单脸OSF3虽然把BioID MAE降至1.999px，却使COCO眼镜覆盖239→238、眼镜每眼P99 11.137→11.782px，因此拒绝作为全局候选。对0.025与0.04的模型一致性算法门限做过离线假设筛选，再对0.04方案实际全量推理；离线假设结果没有当作实际运行结果。源码、补丁及所有失败结果均保留，独立标注和评分门限未修改。

BioID/COCO在本轮参与算法开发和守卫选择，以下是公开可复查的开发集结果，不能视为未见测试集的无偏泛化证明；没有用样本ID或真值进行运行时选择。

## 独立静态准确率

最终候选以其实际调度配置重新跑完全部 1521 张 BioID；基线/候选都完整匹配和渲染，覆盖率为 1521/1521 与 1521/1521。

本报告 MAE 指每锚点欧氏定位距离的平均，再按每脸平均；不是分别对x/y取绝对差。proxy NME 为该距离均值除以独立真值眼距。COCO只对可用可见点评分，逐点汇总与逐脸汇总不混用。

|BioID 指标|原默认 OSF0|最终候选|
|---|---:|---:|
|五锚点每脸 MAE px|2.471|2.229|
|五锚点 MAE P95 px|4.435|4.011|
|五锚点 MAE P99 px|5.390|5.161|
|五锚点 proxy NME|0.045|0.041|
|眼镜每眼误差 px|1.777|1.621|
|眼镜每眼 P95 px|3.828|3.453|
|眼镜每眼 P99 px|5.191|4.841|
|2D roll MAE deg|0.834|0.720|
|2D roll P99 deg|2.918|2.771|

|分锚点 MAE px|原默认|候选|
|---|---:|---:|
|image_left_eye|1.626|1.475|
|image_right_eye|1.928|1.768|
|nose|3.035|2.811|
|image_left_mouth|3.182|2.728|
|image_right_mouth|2.584|2.365|

同图配对：702 张 MAE 降低，207 张增加，612 张相同。官方 `.eye` 最大双眼误差/人工眼距 ≤0.25 的全样本比例：1.000 → 1.000；该协议与 FGnet20 五锚点评分分开。

**新增失败不能隐瞒**：无守卫OSF3筛选时 BioID_0372（反光眼镜/胡须）proxy NME 从约0.0735升至0.1046，产生一个>0.1的新离群点。最终通用一致性守卫在该脸保留原模型，不使用样本ID特例；最终全量 NME>0.1 例数 0→0。原始标注、所有逐脸结果及本机专用叠图保留。总体改善不代表每张图改善。

COCO 同 327 张图：匹配 417 → 417/685；可渲染眼对 239 → 239/516；五锚点可用部分 MAE 2.672 → 2.667px；眼镜每眼均值 2.244 → 2.233px，P99 11.137 → 11.137px。COCO 只标部分语义锚点，未当作完整脸框或稠密真值。尺度分组、未匹配及每脸失败都在 quality JSON；无人工侧脸/遮挡标签的分组不编造。

### ROUND4 关键点 P99 退化专项复核

使用已封存预测与同一独立标签重新审计239个共同眼镜实例，没有把历史预测当作本轮新推理。其每脸眼误差均值2.682→2.244px、P99 9.014→9.487px，197例改善、42例变差，已逐例复算。239例的原始眼/鼻锚点全部相同，排除该次比较中新出现的神经眼定位差异；嘴角48/54→58/62的拓扑修正另外记录。新渲染眼中心与原始模型双眼的最大数值差仅5.7e-14px，旧通用sprite居中/缩放有额外偏移。修正为精确眼映射后，少数旧偏移恰好补偿过的模型错误反而变差，形成尾部退化；不能仅用均值宣称解决。

|仍保留的尾部实例|原通用sprite每脸眼误差px|精确映射px|
|---|---:|---:|
|coco_78170 / 460165|23.360|23.761|
|coco_19402 / 1734330|9.600|10.673|
|coco_173830 / 521392|8.994|9.855|

本机目视检查这些图像可见刘海遮挡、强光/过曝与护目镜等困难外观，属于诊断观察，不修改官方可见性标签。全部239例的坐标与差值在 `round4-tail-audit.json`，本机专用叠图在 `failure-review/round4-tail-local-only.png`。本轮守卫没有消除这些原有尾部，COCO眼镜P99仍保留基线值。9.487是**每脸**分布，11.137是**每眼**分布，不能混为同一口径的前后退化。

## 单人连续视频、快运动及特效诊断

新视频逐帧完整运行，未用合成运动冒充自然视频。包含人物低头、抬头及转头；最大的模型锚点变化案例保存在 temporal diagnostics。快转头候选案例由模型运动量筛选，不是独立人工快速转头标签，不能由此宣称快速转头准确率已过关。

|300s 自然视频|输入帧|有特效帧|内部无特效区段|光流桥接帧|单预测帧中不同模型 ID|
|---|---:|---:|---:|---:|---:|
|baseline|8991|8975|14|0|12|
|candidate|8991|8975|14|0|12|

无特效区段、返回时间、前后模型 ID、桥接帧号、锚点跳变、滤波前后步长、眼镜角度/尺度及滤波相对 raw 的相位对齐全量见 `briefing-guarded-temporal.json`。这些是模型输出诊断，区段可能包含真实离画、检测漏检、确认期和镜头切换，不能称为真实轨迹中断或错误身份恢复率。静止抖动、真实运动滞后、皇冠对真实头部漂移和错误贴附仍缺少独立时序真值，保持不可估计。已有形似 One Euro 平滑参数未调松；新代码只在模型来源改变时避免旧滤波状态污染。

单人像素桥接的已知平移 oracle 证明只取当前图像像素、最多两帧，并覆盖未确认目标/时间修复拒绝；这是合成单元测试，不能计为真实自然运动准确率。自然视频桥接的实际使用次数按上表报告，不能将未触发的功能声称为已改善真实恢复。

2D roll 可用人工双眼线衡量，已在 BioID 表中报告。3D yaw/pitch/roll 的真实 MAE/P95/P99 全为不可估计：本轮没有合法取得具有可用商业研发授权的独立三维真值。未将模型 yaw_proxy、通用 PnP 或重投影误差标成真值。

## 性能和长期状态

六脸720p同一源SHA256、30Hz、各600s串行运行。最终基线和候选使用完全相同harness SHA256（含阶段采集），另重跑匹配基线，避免将早期未采集阶段的记录器开销混入对照。记录全程输出延迟，不只报告最后1200帧。CPU100%表示一个逻辑核心，不是整机百分比；RSS求和会重复共享页，另列PSS。父进程含录像解码、计数和相同渲染检测探针开销。

|六脸持续回放指标|原默认|最终候选|
|---|---:|---:|
|输出 FPS|25.358|29.515|
|全程 CPU 核心百分比（父+子）|282.462|279.445|
|父+子 RSS 峰值 MiB|341.660|381.074|
|父+子 PSS 峰值 MiB|287.463|324.648|
|线程峰值|87.000|87.000|
|新帧替换旧帧数|2785.000|291.000|
|接收至 JPEG P50 ms|55.510|34.186|
|接收至 JPEG P95 ms|72.423|62.894|
|接收至 JPEG P99 ms|79.221|68.157|

基线/候选输入都为 18000 帧，实际输出 15215 / 17709；新鲜帧策略主动替换旧帧，不将替换隐瞒为全部处理。黑源帧 0 / 0，检测到实际特效像素变化的输出 15213 / 17708，不以黑帧充数。队列峰值 1 / 1，结束均归零；生产延迟历史固定 1200，不无限累积。RSS/PSS 采样、稳定期线性趋势、线程和资源清理检查在两份 soak JSON。600s 是有限持续试验，不证明数小时无泄漏。

测试记录器另存全程延迟和阶段数字数组，用于全程分位数，会随试验帧数增加约数MiB；这是有限测试采集器的开销，不能把相应正斜率直接归为生产泄漏。两版采集器源码相同，原始RSS/PSS包含它，不隐去。真正小时级运行仍需低开销采样与目标设备复测。

|单脸 720p、各60s 30Hz|FPS|接收至 JPEG P95 ms|P99 ms|CPU 核心百分比|PSS 峰值 MiB|
|---|---:|---:|---:|---:|---:|
|baseline|30.000|18.871|19.874|217.528|276.096|
|guarded|30.000|27.116|28.363|241.155|320.764|

另外完成120Hz输入过载对照；输入加速只做排队/丢帧诊断，不当作自然快速转头。详见 `overload-guarded-120hz.json` 和 `overload-baseline-matched-120hz.json`。以下8段视频为最终源码实际逐帧处理结果，全部1920帧计入；统计不包含JPEG。精细守卫要运行两模型，部分单脸片段会变慢，不能宣称所有输入均更快。

|自然片段 Engine 平均/P95/P99 ms|原默认|最终守卫候选|
|---|---:|---:|
|handshake|14.647/17.244/28.115|19.953/27.645/30.033|
|profiles|8.027/16.896/17.598|8.080/15.689/20.502|
|room_transition|15.174/17.210/18.672|15.302/20.081/22.323|
|close_speaker|13.768/17.214/18.082|22.837/26.297/28.182|
|close_listener|14.426/18.632/19.417|22.161/25.762/26.580|
|greeting_cross|35.763/40.646/43.132|31.854/42.580/46.554|
|greeting_cluster|23.059/29.849/33.110|20.476/26.606/28.424|
|outdoor_flash|23.140/29.413/31.361|19.648/25.274/27.317|

新8991帧视频处理均值 14.634→13.431ms，P95 20.585→21.831ms，P99 46.674→34.844ms。静态初始化、处理、IPC和端到端Service指标分列。历史ROUND4的28.59FPS/66.78ms与本轮持续试验和探针不同，未拿来充当本轮同条件基线。

另用同样真实Service各20s采集阶段分布作为短剖析，最终600s也已使用相同记录器，完整阶段数据在soak JSON。早期harness保存在 `harnesses/soak-600-used.py`；原始短试验与不加守卫的试验均保留，不代替最终表。复制/颜色转换包含在对应阶段，worker剩余项包含调度，未声称消除全部复制。

|输出帧阶段平均/P99 ms|原默认|最终守卫候选|
|---|---:|---:|
|detection_ms|5.419/6.638|5.432/6.605|
|landmark_ms|24.606/32.201|18.910/22.779|
|tracking_ms|0.792/1.418|0.778/1.372|
|pose_effect_ms|1.463/2.007|1.485/1.945|
|render_ms|3.351/4.180|3.319/4.170|
|jpeg_ms|1.639/2.004|1.633/2.066|

全程接收延迟减去同帧处理和JPEG后的剩余**均值**为 16.473→7.064ms，包含单槽等待与少量记账；不是纯IPC。候选Engine P99 37.820ms，而接收至JPEG P99 68.157ms，排队期间仍可能等待一帧当前处理。不能把两个不同分位数直接相减当作队列P99。小型240帧片段的P99也会受仅2–3个尾样本影响；片段尾部退化与18000帧持续结果均保留。

这都是当前 WSL2 x86_64 CPU、录像源和本机浏览器数据；无 Jetson/GPU、物理相机曝光、网络传输或显示扫描时延结论。ORT 仍共享串行 worker，其他耗时 Card 同时抢占的调度风险保留。

## 回归、平台与交付

|测试层|实际最终结果|日志（round5/final-tests）|
|---|---|---|
|shared-cards|通过 243|`shared-cards.log`|
|core|通过 35|`core.log`|
|frontend|通过|`frontend.log`|
|cpu|通过 320，跳过 3|`cpu-serialization-final.log`|
|optional|通过 132|`optional-serialization-final.log`|
|ros-adapter|通过|`ros-adapter-serialization-final.log`|
|platform|通过|`platform-shm.log`|

默认测试中ROS stubs只证明单元协议；真实模型、隔离worker、多实例生命周期与崩溃恢复另有实测。可选环境含MediaPipe和EGL软件渲染，软件GL不能当硬件GPU。真实ROS adapter通过两种sensor_msgs输入及恢复探针。

真实Driver公共运行时→DDS→Perception→完整Core→原生Chromium Canvas已运行通过，包含三特效、MCP发现、图连接、特效切换、停止、重启、浏览器暂停1秒后新鲜帧恢复及资源清理。本次通过使用独立域的FastDDS **SHM-only**，DDS发现和数据都仅在本机共享内存、内建网络传输全部关闭；Core源码通过相对符号链接读取，传输源码哈希与本仓库相同，未替换lifespan。Core/Driver/Perception三进程外部DDS UDP绑定均为空。原harness沿用loopback命名的字段解释在 `platform-shm/INTERPRETATION.md`。

失败也保留：新一致性诊断最初使用numpy.bool_，真实Service在JSON输出时报TypeError；已转原生bool，并新增序列化回归。首次DDS超时曾过早归因于localhost设置，更详细状态日志才定位此问题；更改localhost参数本身并未修复它。修复后同进程真实ROS通过，但UDP-only跨进程仍出现已发现2个订阅者却无图像数据；SHM-only成功不能冒充默认UDP路径通过，该底层问题仍需部署前定位。只在本任务进程配置ROS_LOCALHOST_ONLY=0及受保护的专属XML，未更改共享环境或任何服务。全部失败目录和日志保留。

版本A位置：`/home/utopus/fourth-paradigm-archives/A-20261118-original-59a31c4e`。版本B位置：`/home/utopus/fourth-paradigm-archives/B-20261118-face-effects-round5-f3f258ba`，保存本仓库、全部ROUND5证据及新数据，其他成果/模型/环境按A的sealed manifest关联；完整恢复命令与实际逐文件恢复证明在B/RESTORE_GUIDE.md和verification/。A不覆盖，不把多个Card强合并。

最简单的全项目候选恢复方式（目的目录必须尚不存在）：

```bash
python3 /home/utopus/fourth-paradigm-archives/B-20261118-face-effects-round5-f3f258ba/restore/restore_B.py restore --archive /home/utopus/fourth-paradigm-archives/B-20261118-face-effects-round5-f3f258ba --dest /tmp/phanthy-restored-B-NEW
```

该命令实际执行完整A恢复与回归，再恢复B全量payload并校验，保留`phanthymotus-face-effects-live.version-A`和独立B工作树；源码/原始数据文件逐一SHA验证，Git指针和环境路径只在新副本重关联。单独运行A脚本时推荐加 `PYTHONDONTWRITEBYTECODE=1`，避免向封存脚本目录新增缓存；B入口已自动设置。

候选profile需明确启用本地OSF3：`detector_model=yunet.onnx, landmark_model=osf0.onnx, single_face_landmark_model=osf3.onnx, max_faces=8, renderer=cpu`，通过上述尺度和一致性守卫才采用精细点。模型来源和SHA沿用受核验清单。未配置此字段时仍用OSF0，不能借用精细profile得分。精细模型默认单线程关闭spinning，快速模型两线程；无需修改共享环境依赖。无守卫OSF3实验的20分钟/单脸指标保留为实验记录，最终表只读取guarded目录及对应实际重跑。

复现命令使用各 receipt 的完整 argv 与环境、不可变包目录。新工作区恢复后将 argv 的旧工作根替换为恢复目录的 `workspace`；标签/图像相对路径和模型符号链接均可解析。`prepare_bioid.py`、`extract_natural_clip.py`、`dataset_benchmark.py`、`score_dataset.py`、`score_bioid_details.py`、`natural_benchmark.py`、`temporal_diagnostics.py`、`soak_benchmark.py` 都随代码保存。

## V1.0 剩余门槛

- 合法取得适用于本项目用途、独立人工关键点和可靠头姿真值的自然连续视频；完成静止、快速转头、大侧脸/部分遮挡的同一输入严格对照。
- 处理新增离群脸、困难小脸和不充分覆盖；用独立时序真值判断错误恢复/贴附、抖动和滞后，而非仅看连续输出。
- 在目标相机、目标 Jetson/主机上测试曝光至显示链路、热稳定、数小时内存、Sherpa/其他 Card 并发和异常恢复。实体机械臂与运动接口不属于本轮实测。

因此保留本地候选 B，**不合并正式开发基线，不声称完整 V1.0 质量验收通过**。
