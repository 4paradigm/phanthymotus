# 动作空间管线：从模型输出到电机

> 状态：三步计划的第 1 步已开 PR，第 2、3 步未开始。
> 最后更新 2026-09-21。

这份文档记的是一件事：**模型吐出来的那串数字，和机器人关节之间，差着什么。**

写它是因为这条链路上的每一处错误都**不报错**——维度对得上、消息校验通过、延迟正常，
机械臂走到错误的地方。下面每一条设计决定后面都跟着「不这么做会怎样」，那才是理由。

## 一、问题：维度相同不代表空间相同

我们云上现在五个模型，动作空间有四种：

| 模型 | 动作空间 | 维度 | 要 IK 吗 |
|---|---|---|---|
| `smolvla-base` | `joint_position` 绝对关节角 | 6 | 不要 |
| **UnifoLM-WMA-0**（G1） | 关节角 2×7 + 2 夹爪 | 16 | 不要 |
| `pi05-droid` | `joint_velocity` 归一化关节速度 | 8 | 不要，但要积分 |
| `openvla-bridge` | 末端 **delta** 位姿 + 夹爪 | 7 | **要** |
| `unifolm-vla-g1` | `EE_R6_G1` 双臂**绝对**末端位姿 + 腰 | 23 | **要** |

`EE_R6_G1` 的布局是 `2 × [末端 xyz(3) + R6 旋转(6) + 夹爪(1)] + 腰 rpy(3) = 23`。

**这两件事是独立的，而从前的协商只比了其中一件。** 一个 23 维的末端位姿模型连到
一张 23 维 `joint_position` 卡片上，两个数字完全吻合，协商通过，然后位姿被当成关节角
发下去。这个缺口已经补上（见下），但它是整条管线的起点。

## 二、已经落地的

| 仓库 | 改动 | 状态 |
|---|---|---|
| cloud | `unifolm` runtime + 35.6 GB 权重清单 + golden 比对 | MR !14/!15 已合 |
| cloud | 反归一化精度对齐上游；golden 比对**逐位相同** | MR !16 已合 |
| cloud | `motus.vla/1` 的 capabilities 加 `control_mode`；四个模型声明填好 | MR !16 已合 |
| phanthymotus | `negotiate.check()` 比动作空间，**缺声明也拒** | PR #248 已合 |
| driver | G1 `servo` 卡（`rt/arm_sdk`，16 维关节空间） | PR #309 已合，**默认关闭** |
| driver | `Group.mode` —— 动作空间按段声明 | PR #311 **待合** |

线上：`unifolm-vla-g1` 在 kai 上 `ready 1/1` + `enabled`，NodePort 30108。
**它跑的镜像早于 `control_mode` 那个提交**，所以 `/capabilities` 里还没有这个字段——
装了新 actucore 的机器人连它会被拒，而那正是期望行为（这个 checkpoint 本来就没有
能接的下行通路）。

## 三、三步计划

```
云端规范化（模型知识）  →  标准空间  →  边端 IK（机器人知识）→ 关节角 → arm_sdk
```

这个切法把两件事按**知识归属**分开了：模型的古怪输出是模型的知识，归云端；
怎么落到这台机器人的关节是机器人的知识，归边端。

### 第 1 步 —— 扩 mode 词汇表（driver PR #311，待合）

`descriptor.mode` 从前是整份描述符一个，表达不了混合向量。现在 `Group` 可以带自己的
`mode`，不写就继承顶层。向后兼容是结构性的：现存 descriptor 没有按段 mode，继承顶层，
解析结果逐字段相同。

**这一步没有往 MODES 里加任何新 mode，是有意的。** `eef_r6_g1` 那种模型私有布局不进
协议词汇表——它该在云端被规范化。PR 里有一条测试就是拿它去试并要求被拒。

### 第 2 步 —— 云端规范化（未开始，cloud 仓库）

每个 runtime 把自己那个模型的怪癖抹平成标准空间：

```
openvla   末端 delta 位姿      →  绝对末端位姿（标准布局）
unifolm   R6 旋转 + 23 维复合  →  同上
pi05      归一化关节速度       →  joint_velocity
smolvla   已经是标准
WMA       已经是标准
```

要做的：

- 定下标准的绝对末端位姿布局（旋转用什么表示——建议四元数，别用 R6 或 rpy；
  它们长得都像一串浮点数，接反了不报错）。
- `control_mode` 报**规范化后**的空间，另加一个 `native_mode` 说原生是什么。
  **这个字段不是装饰**：没有它，「中间有个转换、它可能失败」对机器人完全不可见。
- **openvla 的 delta 转绝对需要当前末端位姿。** 要么云端按 `session_id` 自己累积
  （有状态，丢一帧永久偏移），要么机器人在请求里带上（更稳，小的协议增量）。
  **倾向后者**，但没定。

### 第 3 步 —— 边端 `servo_eef` + IK（未开始，driver 仓库）

一张新卡片，收标准的绝对末端位姿，解 IK，往 `rt/arm_sdk` 发关节角。

**用阻尼最小二乘（CLIK），不要 NLP。** 理由见下面的实测。

**解完必须检查残差，超阈值就拒绝并保持**（和看门狗同一个处置）。这是唯一让
「解不出来」变成响亮失败的办法——优化器不收敛时返回的是某个关节角，不是异常。

## 四、决定这些的实测

都在 Orin 6（JetPack 6.1，aarch64，Python 3.10）上量的，不是推断。

### IK 放边端可行，余量极大

阻尼最小二乘，纯 pinocchio，用本仓库的 `unitree/g1/resource/g1_model.urdf`，
锁掉腿和腰只留 14 个臂关节，warm start：

```
稳态    中位 0.239 ms   p95 0.259 ms   max 0.313 ms
迭代    中位 3 次        残差中位 6.5e-07
未收敛  0/280
30 Hz 预算 33.3 ms —— 占 0.8%
```

复现：Orin 6 上 `/tmp/iklib`（约 200 MB，`pip install --target`，没动系统包），
脚本 `/tmp/ikbench2.py`。

### 宇树的 `g1_arm_ik.py` 用 pip 装的 pinocchio 跑不起来

PyPI 的 `pin` wheel **不带 casadi 绑定**（要 `BUILD_WITH_CASADI_SUPPORT=ON` 编译），
而那份代码第一行就是 `from pinocchio import casadi as cpin`。这跟 aarch64 无关，
任何平台都一样。

而且 **NLP 对 30 Hz 控制回路本来就是错的工具**：宇树那套是给遥操作重定向用的
（容忍几十毫秒、把限位当硬约束）。控制回路只需要在上一解附近迭代几次雅可比，
快两个数量级，而且确定性。限位在这里是每步 clamp，不是硬约束——一个解不出来就
整步失败的求解器，在 30 Hz 上等于周期性丢指令。

### 动作块是 833 ms 开环，这是选边端的决定性理由

`actucore/plugins/vla/plugin.py` 的 `next_command()`：

```python
if self._chunk_index >= len(self._chunk):     # 只有块**用完了**才去取新的
    self._chunk = list(self._provider.infer(observation) or [])
```

**不是「新块到达就替换」，是「旧块跑完才去要新块」。** `chunk_size=25` @ 30 Hz
= 机器人执行完 25 步（833 ms）才重新看一眼世界。

这对 IK 放哪是决定性的：7 自由度手臂对同一个手部位姿有无穷多组解（肘部可上可下），
优化器靠「离种子最近」挑一个。云端解的话种子是 833 ms 前的关节角，过时的种子可能让
优化器**跳到另一个解支**——手的位姿是对的，整条手臂甩过去。边端在每步下发前才解，
种子永远是此刻的真实构型，把陈旧度从 833 ms 压到 33 ms。

顺带：这 833 ms 的开环和 IK 无关，是动作块机制本来就有的性质。`motus.vla/1` 里的
`supports_rtc`（实时分块）就是为它准备的，**四个 runtime 现在都报 false、没人实现**。

### G1 的下发通道：`rt/arm_sdk`，不是 `rt/lowcmd`

官方文档《底层运动开发》：

> 一旦 G1 开机，内置的运动控制程序会自动启动……如果您在这种状态下使用 SDK 进行
> 底层开发，可能会导致指令冲突，从而使 G1 出现抖动的情况。因此……请务必确保
> G1 已经进入调试模式（L2+A）。

`rt/lowcmd` 接管全部 29 个关节，要求人先把平衡控制器关掉。**宇树自己的
`unifolm-world-model-action/robot_devices/arm/g1_arm.py` 走的正是它**——那是给调试态
准备的，抄进一张 agent-core 随时能 start 的卡片里是危险的。

`rt/arm_sdk`（《手臂控制例程》）是高层运控服务提供的上肢接口，**无需进入调试模式**，
靠 `motor_cmd[kNotUsedJoint].q` 这个混合权重和内置控制器共存。`servo` 卡走的是它。

**交还必须渐出**：权重从 1 直接归零，手臂瞬间脱力。官方例程用 2 秒降下来。

## 五、明确没做的

- **VLA 那 23 维仍然驱动不了 G1。** 这正是第 2、3 步要解决的。
- **`servo` 卡一次真机都没跑过。** 两处待确认做成了启动时显式拒绝：
  `rt/arm_sdk` 要不要算 CRC（`send_crc` 默认 True）、Dex1 的消息类型（import 不到就
  拒绝启动）。真机验证按官方文档：**先把 G1 悬挂起来，进锁定站立**。
- **CLIK 基准只动了平移，姿态是固定的。** 旋转跟踪是 IK 变难的地方，没测。
  也没测奇异位形附近、没有自碰撞检查、腰那 3 维没算进来。
- **`pi05` 的速度→位置积分。** 它输出 `joint_velocity`，而 `servo` 卡只收
  `joint_position`。和 IK 无关的另一个缺口，规模小得多。
- **router 的载荷敏感问题。** 同一个请求只改图像大小：6 KB → 0.5 s，232 KB → 2.5-5.5 s，
  1.4 MB → 504。模型本身处理 1.4 MB 只要 155 ms，多出来的全在网络上；往 router 传 1 MB
  实测 72 KB/s。**根因没坐实**（拿别的目标做的上传基线被 307/405 提前拒绝，数据不可用）。
  不是 unifolm 引入的，pi05 同样中招。建议单独立项。

## 六、这条路上踩过的坑

留着是因为它们都**不报错**，而且下一个人大概率会以同样的方式踩。

- **靠扫 import 定依赖会漏掉字符串引用的类。** `rich` 是写在 logging 配置字典里的
  `"class": "rich.logging.RichHandler"`，整个仓库没有一条 `import rich`。漏了它之后
  报的是 `ValueError: Unable to configure handler 'console'`，一个字没提 rich。
  防线是构建期 import 完整推理链，不是补包。
- **`pip install` 装 PEP 420 命名空间包会静默丢掉子包。** `unifolm_vla` 的
  `model/` 下没有 `__init__.py`，`find_packages` 因此看不见整棵树。装完
  `import unifolm_vla` 成功、`pip show` 正常，第一次加载 checkpoint 才
  `ModuleNotFoundError: No module named 'unifolm_vla.model'`。
- **探测带宽要给够字节数。** 同一个 URL、同一台机器，2 MB 的 range 量到 536 KB/s，
  60 MB 量到 5.6 MB/s——十倍差，因为速率是拿整次传输（含 DNS、TLS、重定向）平均的。
  小的那个数让我们一度以为 35.6 GB 要下 18 小时，从而去规划了一条根本不需要的中转路线。
- **hf-mirror.com 对 `Python-urllib` 这个 UA 直接 403。** `curl -A "Python-urllib/3.11"`
  当场复现。表现是 `no usable source`，而同一个 URL 用 curl 拿是好的。
- **Orin 直连 PyPI 27 KB/s，清华源 1.27 MB/s。** 差 46 倍。在这些机器上，
  「慢得像卡住」几乎总是源的问题。
- **kubelet `/stats/summary` 的 imagefs 用量是缓存值。** 三次采样跨五分钟逐字节相同，
  据此会判成「镜像拉取停滞」。要 `du` containerd 的 ingest 目录。
- **kai 的 pod 网络和节点网络策略不同。** 镜像拉取走节点网络；用普通 pod 探仓库
  会得出「节点够不着」的错误结论。要 `hostNetwork: true`。

## 七、相关文件

| 位置 | 是什么 |
|---|---|
| `phanthymotus-cloud/runtimes/unifolm/policy.py` | UnifoLM-VLA runtime，模块文档记着 argv 常量那个坑 |
| `phanthymotus-cloud/runtimes/unifolm/golden_compare.py` | 和官方推理路径的逐位比对，换 checkpoint 后要重跑 |
| `phanthymotus-cloud/motus_vla/protocol.py` | `motus.vla/1` 线上类型，`control_mode` 在这 |
| `actucore/plugins/vla/negotiate.py` | 协商检查，动作空间那条在这 |
| `phanthymotus-driver/common/control/descriptor.py` | `motus.control/1` 描述符，`Group.mode` 在这 |
| `phanthymotus-driver/unitree/g1/servo.py` | G1 关节空间 servo 卡，arm_sdk 的权重渐变在这 |
| `phanthymotus-driver/x-humanoid/tianyi2.0/servo.py` | 写新 servo 卡时照这张抄 |
