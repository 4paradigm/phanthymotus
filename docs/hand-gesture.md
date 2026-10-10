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

### 结论：RTMPose-m (hand5)，21 关键点，256×256，top-down

> **这一节推翻了本文最初的结论。** 原方案选的是「YOLO26s-pose 在 Ultralytics
> hand-keypoints 上微调」，并把 RTMPose **整条划掉**，理由是「不需要写 SimCC
> 解码、不需要第二个 bundle、不需要第二套 engine」。那个判断是在**只比较了集成
> 成本、没有比较精度**的情况下下的，是错的。

YOLO 那个模型在真机上不够用：OK 手势的腕点落在掌心、张开的手回来是一团、运动
模糊直接把关节塌掉 —— 而**检测分数照样 0.8**，所以下游没有任何办法知道。拿同样
的裁剪框并排比，RTMPose 在它错的每一张图上都是对的。

而且更便宜，这是没预料到的：

| 每只手，走生产运行时 | Orin 6 @1020MHz | Orin 5 (jp5.11) |
|---|---|---|
| yolo26s-hand21 @448（原方案） | 7.99 ms | 8.19 ms |
| **rtmpose-m-hand5 @256** | **3.58 ms** | **4.33 ms** |
| rtmpose-m-hand5, batch 2 | 2.52 ms/只 | 3.14 ms/只 |

**不需要检测器。** 中途我差点保留 YOLO 做检测器、RTMPose 只做关键点（每只手
10.5 ms，比原来还贵）—— 起因是第一次对比时 RTMPose 三张输了一张。实际是**我喂
错了框**：YOLO 要手占输入 31%，RTMPose 是 top-down、要手填满框。换紧框后三张
全赢，而且实测它在 **0.30–0.65 这个 2.2 倍区间内都读得对**，比旧模型宽容得多。
所以手臂推出来的框就够。

hand5 只发布了 **-m**，没有 -t/-s（404）。要更小的得自己训。

### 被否决的方案

**MediaPipe Hands。** 会给 perception 进程引入**第三套推理运行时**（tflite）。
进程里已经有两份 ONNX Runtime 共用一个 CUDA provider 的坑（见
`plugins/kokoro_worker.py`，jp5.11 上直接 SIGSEGV），再加一套不是技术债而是
定时炸弹。

**YOLO26-pose 微调版。** 曾经是本方案的选择，已被实测淘汰，理由见上。

**320 输入。** 针对的是旧模型，随它一起作废。

---

## 3. ROI 几何与闸门

**框的目标占比 0.6**（旧模型是 0.31）—— top-down 模型要手填满框，检测器要
留出余地看周围。实测扫框：0.30–0.65 都对，旧模型要的 3.2 倍松框直接失败。

**距离闸门用几何量，不用置信度。** 手长 ≈ 0.9 × 前臂长（**不是 0.45，那是手宽**
—— 这个常量最初就取错了），前臂长在 COCO-17 里现成，推理之前就能判、零成本。

> 0.9 是真机上改出来的。0.45 是手宽，而模型的框罩的是张开手指的手长。错的方向
> 会把自己藏起来：裁剪框只有应有尺寸的一半，手占到框的 70%，正是检出开始崩的
> 那一端。Orin5 上一只手直接 miss、另一只恰好卡在悬崖边。

**置信度闸门 0.42**，标定自两个实测分布：

| | 置信度 |
|---|---|
| 框里**没有**手（灰板/衣服/背景/脸/牛仔裤/噪声） | 0.11 – 0.37 |
| 真的有手（含运动模糊帧） | 0.48 – 0.74 |

这个数**量不出模糊**（清晰 0.55/0.67 vs 模糊 0.55/0.65），它是定位峰值不是质量
分 —— 能挡「没有手」，挡不掉「手拟合坏了」。后者目前没有判据，换模型后也暂时
没再观察到。

### 两个会静默出错的点

**两个输出必须按名字取。** `simcc_x` 和 `simcc_y` 形状完全相同（N,21,512），
内容上分不开，而这个仓库记着两条 JetPack 线的输出顺序会相反。取反了会把每只手
转置，而且照样画出一只看着合理的手。

**归一化不能复用 `VisionEngineSession`。** 那个会 letterbox 并缩到 [0,1]，是
YOLO 的训练口径；RTMPose 要不加 padding 的方形裁剪 + ImageNet mean/std。喂错是
一只自信的、位置错误的手，没有任何东西会报错。

## 4. 算力预算

每只手一次推理（没有检测器、没有重试），Orin 6 @1020MHz **3.58 ms**、
Orin 5 **4.33 ms**。批处理两只手是 2.52 / 3.14 ms 每只（engine 建的是
dynamic batch 1..4），但第一版走串行。

### 默认值

| 配置项 | 值 | 推导 |
|---|---|---|
| `hands` | `off` | 开了才加载第二个 engine |
| `hand_max_rois` | 2 | 按手的像素大小取前 N 只 —— 大就是近 |
| `hand_interval_s` | 0.25 | 按 (track, side) 节流并按 track 错峰 |
| `hand_confidence` | **0.42** | 标定自两个实测分布，见 §3 |
| `hand_hold_s` | 0.5 | 一帧读坏不是手走了 |
| `hand_min_forearm_px` | 50 | 几何闸门；像素换米取决于视场角，真机上调 |

### GPU 频率：jp6.1 上比什么都重要

**在 jp6.1 上，同一个 plan 在我们的运行时里比 trtexec 慢 4.6 倍**，因为每次推理
后同步、中间夹主机侧工作，调频器把 GPU 压在 306 MHz 底频不放。这影响**所有**
TensorRT 卡（vop/depth/pose/hand + actucore 的 VLA），不是手部特有。

实测（Orin 6，空闲整板功耗 / 一次手部推理）：

| min_freq | 空闲功耗 | 推理 | 每瓦换来的毫秒 |
|---|---|---|---|
| 306（默认） | 5.68 W | 17.98 ms | — |
| 612 | 6.94 W | 11.69 ms | 5.0 |
| 816 | 7.88 W | 10.83 ms | 0.9 |
| **1020** | 8.19 W | **8.69 ms** | 6.6 |
| 1173 | 10.23 W | 7.88 ms | 0.4 |

1020 是拐点，顶格那一档 2 W 只换 0.8 ms。端到端 pose p50 从 115 降到 55 ms。

**这是主机层配置，不在任何卡片或容器里** —— 要 host root、影响所有 GPU 消费者、
有常驻功耗代价必须被看见。实现在 `deploy/host/`（systemd oneshot），按 L4T 版本
判断再动：**jp5.11 不需要**（1/10/100 次 enqueue 都是平的，和它自己的 trtexec
一致）。长期正确的修法在运行时：CUDA graph / 不每次同步 / 批处理。

### 采样率不同 ⇒ 迟滞不能按帧算

手部通道的有效帧率是 4 Hz 而不是 12 Hz，所以所有 dwell 按**秒**定义。这也是
`gesture_events.py` 已经采用的做法。

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
4. **jp6.1 的延迟不能按隔离基准定，原因不是我最初以为的那个。**

   最初记在这里的是"Orin 6 被同事的并发实验污染了"。那个归因**是错的**，已在真机上查清：

   | 同一个 448 plan，Orin 6 | 每次 |
   |---|---|
   | trtexec（背靠背 + spin wait） | 4.67 ms |
   | 我们的运行时（每次 execute 后 synchronize） | 21.71 ms |
   | 手工 100 次 enqueue 只同步一次 | 4.74 ms |

   采样 GPU 频率：我们的路径**全程 306 MHz**（空闲底频），trtexec 跑时 306→510→918→**1173**。
   每次调用之间夹着主机侧工作和一次阻塞同步，GPU 在间隙里空着，devfreq 看到低利用率就把
   频率压在底部 —— 频率低又让 kernel 更慢。拆解确认过不是拷贝（H2D 0.45 / D2H 0.10 ms）、
   不是预处理（letterbox 0.14 / to_blob 0.89）、也不是解码（0.31），全在 execute 里。

   **Orin 5（jp5.11，GPU 最高 765 MHz）不出现这个现象** —— 1/10/100 次都是平的 7.6 ms，
   和它自己的 trtexec（7.85）对得上。所以 §4 里基于 Orin 5 的那组数和由它推出的
   `hand_interval_s: 0.25` 是站得住的。

   两个后果：

   - **在空闲机器上量 jp6.1 的单模型延迟，得到的是悲观值，不是可达下限。** 生产时
     perception 同时跑多个模型，GPU 不会空，频率会上去。所以隔离基准不能直接当帧预算。
   - 这影响 jp6.1 上**所有** TensorRT 卡（vop / depth / pose / hand），不是手部通道特有，
     值得单独立一个问题去查，不属于本方案的范围。

   中途有一次"强制升频到 1173 仍然 20.5 ms"的测量，那次带着竞争负载、不干净，不要据此
   排除频率因素 —— 决定性的是上面那张 1/10/100 次 enqueue 的表。
5. **身体层手势事件（PR #297）没有在真机上跑过。** 全部是单测和接线测试。

### 真机实测时要补的

- 机器人自己相机、走 ROS 话题的帧（不能用手机拍 —— 问的是那些相机的分辨率/视场角/JPEG 质量下还剩多少有效像素），1/2/3 m × 张掌/指向/点赞。
- 相机的实际水平视场角。表里的 70° 是估的，差 20° 距离就差 30%。
- 身体层手势事件端到端：真人举手 → 事件带 priority 到达主 agent → 阈值在真实帧率下的手感。
