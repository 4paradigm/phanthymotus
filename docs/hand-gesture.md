# 手部关键点与手势识别

给 `pose` 卡加手部 21 关键点和手势识别的方案。**这份文档里的数字是实测的，不是估算的** —— 每个数后面都注明在哪台机器、什么条件下量的，以及还有哪些仍然是假设。

关联：`perception/README.md` § `pose`（已实现部分的权威说明）、`perception/plugins/gesture_events.py`（身体层手势事件，已实现）。

---

## 1. 三档距离模型

**这是整个方案的骨架，也是最容易搞错的一件事。** 手部 21 关键点不是"手势识别"的同义词 —— 远距离召唤根本不该走手部模型。

| 档 | 条件 | 来源 | 能判什么 | 状态 |
|---|---|---|---|---|
| **body** | 人在画面里即可 | COCO-17 手腕/手臂几何 + ST-GCN++ | 举手、挥手、招手、粗指向、抱臂 | **已实现**（PR #297） |
| **hand-roi** | 前臂 ≥ 90 px（约 ≤ 3.5 m） | 手部 21 点，ROI 上采样 | 张掌/握拳/指向/V 字 —— 粗粒度手型 | 待实现 |
| **hand-roi 近** | 前臂 ≥ 200 px（约 ≤ 1.5 m） | 同上 | 加 OK / 捏合 / 数指等细分 | 待实现 |

**"三米招手叫机器人"这个场景由 body 档负责，不依赖任何新模型。** `pose_action.py` 的 `raising hand` 是单帧几何读数，`hand waving` 来自 ST-GCN++ 的 NTU A23，两者在人体可见的任何距离都有效。缺的从来不是识别能力，而是送达方式 —— 那是 `gesture_events.py` 解决的问题，见 README。

手部层的价值在近距精细交互（指哪个东西、OK/取消、捏合调节），不在远距召唤。把两者混为一谈会得出"要上手部模型才能支持手势"的错误结论，而那会让一个不需要新模型的场景等一个需要重训的模型。

**卡片必须报它在哪一档**（`gesture_tier`）以及被跳过的原因分类（`hand_skipped: {too_far, wrist_occluded, throttled}`）。否则四米外挥手没反应，没人分得清是太远、看不到手腕、还是 engine 挂了。

---

## 2. 模型选型

### 结论：微调过的 YOLO26s-pose，21 关键点，448 输入

基座是 `yolo26s-pose.pt`（和现在 pose 卡用的同一个架构、同一个 `one2one_*` NMS-free 头），在 [Ultralytics hand-keypoints 数据集](https://docs.ultralytics.com/datasets/pose/hand-keypoints/)（26,768 图，21 点，标注由 MediaPipe 生成）上微调。起点权重取自 [marceloeatworld/yolo26-training](https://github.com/marceloeatworld/yolo26-training)，训练配方公开（100 epoch，H200 约 1h45，约 $6），**所以"来源不明"不是阻塞 —— 我们能自己复现、也能自己按需重训**。

发布的 engine 必须是我们自己导的那一份，理由见 §6。

### 为什么解码零改动

ONNX 元数据（实测读出）：

```
task: pose   kpt_shape: [21, 3]   names: {0: 'hand'}
nms: False   end2end: True        opset 17
kpt_names: wrist, thumb_cmc/mcp/ip/tip, index_mcp/pip/dip/tip, middle_*, ring_*, pinky_*
```

输出 `(1, 300, 69)` = `6 + 3×21`，正是 `vision_runtime._pose_layout()` 的 offset-6 分支，而 `decode_poses(outputs, meta, conf, n_kpts=21)` **已经支持任意关键点数**。`vision_runtime.py` 一行不用改。

元数据里还有 `kpt_names`，所以应给 `VisionEngineSession` 加一个 `keypoint_names()`，和现有 `class_names()` 读 `names` 完全同构 —— 词表从 engine 内部取，不会和外挂文件漂移。

### 被否决的方案

**MediaPipe Hands。** 会给 perception 进程引入**第三套推理运行时**（tflite）。进程里已经有两份 ONNX Runtime 共用一个 CUDA provider 的坑（见 `plugins/kokoro_worker.py`，jp5.11 上直接 SIGSEGV），再加一套不是技术债而是定时炸弹。另外 Jetson aarch64 上装不干净，纯 CPU 跑会和 ASR/TTS 抢同一块 CPU。

**RTMPose-hand（两阶段 + SimCC 解码）。** 原本是首选，因为官方权重、有 benchmark。YOLO 路线验证通过后整条划掉：不需要写 SimCC 解码器，不需要第二个 bundle，不需要第二套 engine。

**整帧模式（不裁 ROI）。** 见 §3 —— 实测只能用到约 0.7 m，对机器人基本无用。

**320 输入。** 比 448 再省一半算力，NME 曲线也跟得上，但在三个尺寸上检出掉到 3/4（448 是 4/4）。失效方式是检不出，这比判错好，但对"举手要被看见"是致命的。不值。

---

## 3. ROI 几何：模型的工作区间是一条窄带

实测（整帧模式，手占 640 输入的像素数，n=5 张真实照片）：

| 手的像素 | 检出 | 形状误差 NME |
|---|---|---|
| 500 px | 2/5 | 0.158 |
| 320 px | 5/5 | 0.047 |
| **200 px** | **5/5** | **0.023** ← 最佳 |
| 160 px | 5/5 | 0.032 |
| 120 px | 5/5 | 0.041 |
| 100 px | 4/5 | 0.169 |
| 82 px | 4/5 | 0.154 |
| 55 px | 2/5 | 0.242 |
| 40 px | 0/5 | — |

**可用区间 120–320 px，目标 200 px。两头都掉** —— 手占到 424/640 时 conf 掉到 0.069。

所以：**`crop_side = 3.2 × 估计手框`**，让手落在目标占比上。"固定 320×320 窗口"是错的：在 1 米处会把手放到 500 px，而那一档 2/5 漏检。

### ROI 上采样确实救回了 3 米

从**原生分辨率**帧裁 3.2× 手框、INTER_CUBIC 拉到网络输入（JPEG q80 在裁之前施加，顺序和真实管线一致）：

| 原生手像素 | 约等距离 | 检出 | ROI 模式 NME | 整帧模式 NME |
|---|---|---|---|---|
| 240 px | 1.0 m | 5/5 | **0.016** | — |
| 160 px | 1.5 m | 5/5 | **0.026** | 0.032 |
| 123 px | 2.0 m | 4/5 | **0.038** | 0.041 |
| 100 px | 2.5 m | 5/5 | **0.061** | 0.169 |
| **82 px** | **3.0 m** | **5/5** | **0.059** | **0.154** |
| 68 px | 3.6 m | 4/5 | 0.054 | miss |
| 55 px | 4.5 m | 4/5 | 0.056 | 0.242 |
| 40 px | 6.2 m | 4/5 | 0.071 | miss |
| 30 px | 8.2 m | 3/5 | 0.114 | miss |

**3 米处误差降到三分之一。** 整帧模式要求手 ≥120 px of 640，1080p 下换算过来只能用到约 0.7 m —— 所以 **ROI 模式是默认，不是远场特例**，整帧检测这条路不要。

### 失效方式是「自信地判错」，所以闸门不能用置信度

100 px 那一行：**conf 0.92，NME 0.169**。NME 0.169 意味着平均关键点偏差为手宽的 17%，差不多一整个指节 —— 手指屈伸判定在这个误差下是噪声。

置信度一路保持 0.9，形状误差翻了三倍。这和 `pose_stgcn.py` 记的 NTU-60 教训同类：不是不确定，是确信地错。

**闸门用几何量**：手长 ≈ 0.9 × 前臂长，而前臂长在 COCO-17 里现成 —— 所以闸门是 `hand_min_forearm_px`，在推理之前就能判，零成本。手腕不可见则这只手不报，不猜（和现有"看不到髋/膝时姿态一律 unknown"同一个态度）。

> **0.9 是真机上改出来的，本文最初写的 0.45 是错的。** 0.45 是**手宽**，而模型的框罩的是张开手指的**手长**。错的方向正好会把自己藏起来：裁剪框只有应有尺寸的一半，手占到框的 70% 而不是 31%，而 70% 正是上面那张表里检出开始崩的那一端。Orin5 上第一张真实照片里，一只手直接 miss、另一只 0.71 但恰好卡在悬崖边；扫这个常量后两只手在 0.9 处都到 0.76/0.78。症状读起来像「手部模型不可靠」而不是「裁剪框小了一半」。

---

## 4. 算力预算

实测，Orin 5（jp5.11 / TRT 8.5.2.2，**停掉全部容器**后测的）：

| engine | engine p50 | e2e per ROI | 相对 640 |
|---|---|---|---|
| 640 | 15.09 ms | 17.43 ms | 1.00× |
| **448** | 10.21 ms | **11.26 ms** | **0.65×** |
| 320 | 6.85 ms | 7.95 ms | 0.46× |

缩放**次线性**：448 的 FLOPs 是 0.49× 而耗时 0.65×，320 是 0.25× / 0.46×。三点拟合出**每次推理约 4.1 ms 固定开销**（H2D/D2H + 同步）。

瓶颈是 engine 本身，不是 Python —— engine 10.21 对 e2e 11.26，裁剪+上采样+解码只占 1 ms。这和 vop 当年相反（那次是 ultralytics 的 Python 前后处理吃掉 30 ms），所以优化方向在输入尺寸，不在代码。

### 默认值怎么来的

12 fps 下帧预算 83 ms，现有 pose+stgcn 占 41 ms（README 记录的三人实测），剩 42 ms：

| 配置项 | 值 | 推导 |
|---|---|---|
| `hand_model` | 448 engine | 检出无损失、NME 在噪声内、耗时 0.65× |
| `hand_max_rois` | 2 | 2 个 ROI = 22.5 ms |
| `hand_interval_s` | **0.25**（4 Hz） | 22.5 × 4/12 = **7.5 ms/帧均摊** |
| `hands` | `off` | 开了才加载第二个 engine |
| `hand_min_forearm_px` | **50**（Orin5 实测后修正） | 几何闸门，不是模型置信度 |

每帧都跑也塞得下（41+22.5 = 63.5 / 83 ms），但那只剩 20 ms 给同一块 GPU 上的 vop/depth/ASR/TTS，所以节流到 4 Hz。

**批处理是后续优化，不进第一版。** 固定开销占 448 的 36%，2 个 ROI 批处理能省约 27%（22.5 → 约 16.3 ms），但需要 `dynamic=True` 重导。值得做，不救命。

### 采样率不同 ⇒ 迟滞不能按帧算

**手部通道的有效帧率是 4 Hz，不是 12 Hz。** 照抄身体层的 `label_hold=3` 在 4 Hz 下等于 **750 ms** 才换标签，对交互输入太慢（12 fps 下 3 帧只有 250 ms，所以那个默认值在身体层是对的）。

手部通道用 `label_hold=2`（500 ms）。所有 dwell 按**秒**定义，不按帧 —— 这也是 `gesture_events.py` 已经采用的做法，见该文件和 README。

---

## 5. 手势词表

判定一律用几何规则，**不上第二个学习模型**。`pose_stgcn.py` 的教训直接适用：分类器没有"这不是手势"这个类，静止的手会拿到一个自信的错答案（NTU-60 对真人躺地上给 "play with phone/tablet" 0.997）。21 个点的几何信息量足够，而且可解释、阈值可调、`evidence` 里能给出判据。

### body 档（任意距离）

`raising_hand` 举手 · `waving` 挥手 · `beckoning_arm` 大幅招手 · `pointing_arm` 手臂指向(+方向) · `arms_crossed` 抱臂 · `both_arms_up` 双手举起 · `arms_out` 双臂平举 · `hands_on_hips` 叉腰 · `salute` 敬礼式 · `t_pose` T 字（标定用）

前三个加指向已由 `gesture_events.py` 的默认白名单覆盖。

### hand 档，静态（◐ = 仅近距可靠）

`open_palm` 张掌 · `fist` 握拳 · `pointing` 食指指向(+方向) · `victory` V 字 · `thumbs_up` 点赞 · `thumbs_down` 倒赞 · `ok` ◐ · `pinch` 捏合(+连续捏合量) ◐ · `gun` 手枪指 · `rock` 摇滚手 · `three/four/five` 数指 · `palm_down` 掌心向下 · `palm_up` 掌心向上 · `fingers_spread` 五指张开 ◐

屈伸向量：每根手指由 MCP-PIP-TIP 夹角定 extended/curled，用 wrist→middle_MCP 向量做尺度和旋转归一化。

### hand 档，动态（窗口轨迹）

`waving_hand` 挥手 · `beckoning` 招手过来 · `dismissing` 摆手拒绝 · `swipe_left/right/up/down` · `circle_cw/ccw` 画圈 · `tap_air` 空中点按 · `pinch_drag` 捏住拖动 ◐ · `stop_push` 推掌(急停)

### 双手

`clap` 拍手 · `heart` 比心 · `timeout` T 字暂停 · `frame` 取景框 · `both_palms_out` 双掌外推

### 默认只开一小部分

词表越大，相邻手势的阈值互相挤压，边界抖动越多。默认 8 个（`open_palm` `fist` `pointing` `thumbs_up` `victory` `beckoning` `waving_hand` `stop_push`），其余在代码里但由 `gesture_whitelist` 打开。

### 和已有通道的关系

- 手部层的 `waving` / `pointing` **覆盖**身体层同名 activity（更准，且能说出是哪只手），并在 `evidence` 里记下被覆盖这件事。不同名的互不干扰。
- 现有 `point_direction` 从"肘→腕"升级为"食指 MCP→TIP"，取不到手时自动回退。这是"指那个东西"真正可用的前提。
- **左右手只由 ROI 来自哪只手腕决定**，不读模型自带的 handedness。`left` 指人的左手。

---

## 6. 两个必须显式处理的坑

### 非 end2end 的导出会被 `decode_poses` 当成合法布局读成垃圾

21 点模型的**原始检测头**输出是 68 通道，而 `_pose_layout` 的 offset-5 分支正好是 `5 + 3×21 = 68`。score 列过了 sigmoid 在 [0,1]、可见性列也在 [0,1]，**内容检查全部通过** —— 2100 个未经 NMS 的 anchor 被当成 2100 只手读进去，box 是 xywh 被当 xyxy 解。正如 `vision_runtime.py` 自己的注释所说，每一种错读都画出一个看着合理的骨架，没有任何报错。

**加载时必须显式守卫 end2end**：查 engine metadata 的 `end2end` 字段，或断言行数是 top-k 的 300 而不是 anchor 数（320→2100, 448→4116, 640→8400）。`decode_poses` 自己挡不住。

导出侧：**ultralytics 8.4.175 上 YOLO26 必须给 `nms=True` 才出 end2end 的 `(1,300,69)`**，不给就是原始头。8.4.33 不给也出 end2end —— 行为变过，所以照旧脚本抄会掉进来。

### 同一份权重、不同 ultralytics 版本，导出的数值不一样

8.4.175 对 8.4.33 实测：中位 NME **0.0145**、最大 0.0203、五指伸屈判定翻 **1/70**。

参照：fp16 对 fp32 只有 0.0005、0/140 翻转。也就是说**换一个 ultralytics 版本造成的差异，比 fp16 量化大 30 倍**，和「160 px 对 240 px」的退化同级。

所以 `HAND_MODEL_BUNDLES` 的注释里**必须记下导出用的 ultralytics 版本**，否则下次有人重导会得到一个差 1.5% 的 engine，而所有 size/SHA256 校验都过。

### fp16 本身是安全的

两条 JetPack 线、28 张逐字节相同的 640×640 输入（PNG 冻存，排除两边 cv2/JPEG 差异）：

| | Orin 6 (jp6.1 / TRT 10.4) | Orin 5 (jp5.11 / TRT 8.5) |
|---|---|---|
| 检出不一致 | 0 | 0 |
| 中位 / p95 NME | 0.0005 / 0.0008 | 0.0005 / 0.0007 |
| 单点最大偏差 | ≤0.6 px | ≤0.6 px |
| **五指判定翻转** | **0/140** | **0/140** |

一处需要修正的早期说法：这个 engine 只有一个输出张量（`output0`），所以"两条 JetPack 线输出顺序相反"那个坑**在这里不会触发**。真正被压到的是 offset-6 / 宽度-69 的布局识别，不是张量选择。

---

## 7. 数据面

### lean 话题 `{input}/poses` —— 只加语义，不加坐标

```json
{"id": 1, "...": "...", "gesture": "open_palm", "gesture_hand": "right", "gesture_score": 0.86}
```

约 45 B/人。手部坐标另给 `publish_hand_keypoints: off|compact|full`，默认 `off`，和现有 `publish_keypoints` 同构。

### skeleton 话题 —— 17 → 59 点

`keypoints` **追加**（绝不 interleave）手部 42 点：索引 0-16 身体、17-37 左手、38-58 右手。payload 里的 `keypoint_names` 和 `skeleton` 边表同步扩。

**前端零改动就能画**：`pose2d.js` 的 `parsePosePayload` 已经优先用 payload 自带的 `skeleton` / `keypoint_names`，`visibleBones` / `visibleJoints` 对关键点数完全泛型。旧消费者按下标取前 17 个也仍然正确 —— 这就是必须追加而不是插入的原因。

可选的前端微调：标签行加 `gesture`，索引 ≥17 的关节点半径减半（手部点密集，3 px 圆点会糊成一团）。不改也能画。

### gesture 话题 —— **已实现**

`{input}/poses/gesture`，稀疏，带 `priority`。完整说明见 `perception/README.md` § "The gesture topic is the one to wire into `decision_core`"。手部层接入时复用同一个 `GestureEventTracker`，只是喂给它的 activity 来自手部通道、采样率 4 Hz。

---

## 8. 实现清单

| 文件 | 改动 |
|---|---|
| `plugins/hand_runtime.py` | **新建**。前臂闸门、ROI 推导（肘→腕外推 + 上一帧 landmark 回归框）、裁剪边长 3.2×、左右手由手腕定、end2end 守卫、batch blob |
| `plugins/hand_gesture.py` | **新建**。纯 numpy：屈伸向量 → 静态手势；窗口轨迹 → 动态手势；复用 `LabelStabiliser`（`label_hold=2`） |
| `plugins/pose.py` | 第三通道接入、config、`info` 的 `gesture_tier` / `hand_skipped`、`list_actions` 扩词表 |
| `plugins/vision_runtime.py` | `HAND_KEYPOINTS` / `HAND_SKELETON` 常量 + `VisionEngineSession.keypoint_names()`。解码零改动 |
| `utils/model_downloader.py` | `HAND_MODEL_BUNDLES`（jp61/jp511）+ `ensure_hand_model`，注释记 ultralytics 版本 |
| `tools/export_vision_engines.py` | `--model hand`（`nms=True`、`imgsz=448`） |
| `web/js/renderers/pose2d.js` | 可选：标签加 gesture、手部关节点半径减半 |
| `perception/README.md` | 新增小节 |

### 落地顺序

1. **PR-A（已完成，#297）** —— 身体层稀疏手势事件。不依赖新模型，先交付远距召唤场景。
2. **PR-B（本文档）** —— 方案固化。
3. **PR-C** —— engine 发布 + `hand_runtime.py` + `hands: keypoints` 通道，**只画不判**。先把"手看不看得见"在真机上答掉。
4. **PR-D** —— `hand_gesture.py` 手势判定 + 词表。

跨层改动（`perception/` + `agent-core/`）必须在同一个分支同一个 PR —— 按层拆开两边都没法单独构建和验证。每个 PR 都是**先 commit 再拷到机器**。

### 测试策略

- `test_hand_gesture.py` —— 合成 21 点手，纯 numpy，不碰 engine 也不碰 cv2。每个手势一组正例、一组边界、一组逐帧抖动（验迟滞不是逐帧翻）。
- `test_hand_runtime.py` —— 前臂闸门、ROI 推导（手腕不可见必须返回 None）、裁剪边长、左右手归属、59 点追加后前 17 位不动、**end2end 守卫能挡住原始头张量**。
- `test_pose_plugin.py` 扩 —— `hands: off` 时**不加载第二个 engine**（最重要的一条）。
- `pose2d.test.mjs` —— 59 点 + 自带 skeleton 能画。
- 帧字节一律走 `vision_stubs.frame_bytes()`。写死 `b"WxH"` 字面量那次，29 个测试在笔记本上绿、在每个镜像里红。

---

## 9. 仍然是假设的部分

**说清楚边界，别把下面任何一条当成实测事实。**

1. **距离换算全部基于假设**：手宽 0.18 m、70° HFOV、1080p。所有"约 3.0 m"都是这么折算的。真机截图之前它不是实测。**所以三档边界必须是配置项**（`hand_min_forearm_px`），而不是代码常量 —— 这是对一个未验证常量该有的处理方式。
2. **样本量 4–5 张公开照片**。够用来选输入尺寸，不够定阈值。其中一张（点赞）在所有尺寸上 NME 恒定 0.13–0.14，是参考帧构图差异造成的 harness 伪影，把中位数整体抬高了一点；另一张（街头 V 字）在多个尺寸漏检，杂乱背景单独造成伤害，和尺寸无关。
3. **精度数字来自 CPU 上的 fp32 ONNX**，engine 的 fp16 等价性已在两条线上验证（§6），但精度曲线本身没有在 engine 上重跑。
4. **Orin 6（jp6.1）的延迟数全部不可用。** 448 在那台量到 24.43 ms，而更老的 Orin 5 是 10.21 —— 差 2.4 倍，方向还和 stgcn 的实测（jp6.1 快一倍）相反。那台一直有别人的并发实验在跑，且 448 对 640 只省 2.4%（算力是一半），这个比值本身就说明测的不是我们自己的计算。**jp6.1 线的 `hand_interval_s` 默认值目前没有依据**，只能先沿用 jp5.11 的。
5. **身体层手势事件（PR #297）没有在真机上跑过。** 全部是单测和接线测试。

### 真机实测时要补的

- 机器人自己相机、走 ROS 话题的帧（不能用手机拍 —— 问的是那些相机的分辨率/视场角/JPEG 质量下还剩多少有效像素），1/2/3 m × 张掌/指向/点赞。
- 相机的实际水平视场角。表里的 70° 是估的，差 20° 距离就差 30%。
- Orin 6 空闲时的延迟。
- 身体层手势事件端到端：真人举手 → 事件带 priority 到达主 agent → 阈值在真实帧率下的手感。
