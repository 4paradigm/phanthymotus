# 来源与修改

- `runtime.py`、`dispatch.py`、`protocol.py`、`capture.py`、`rtc.py`、`capture_server.py`、`descriptor.py` 选择性迁移自 `4paradigm/phanthymotus-driver` PR #152，提交 `0adb64cb63e358244f4e02c0a7823523824f17a8`，原路径 `unitree/g1/teleop/`。许可证 Apache-2.0，完整文本见本仓根 `LICENSE`。保留会话栅栏、最新帧队列、RTC/WSS 与测试用零输出适配器；改为 ActuCore 宿主、天轶能力、双握把/扳机校验和真实 Driver 状态。G1 SDK、权重混合和硬件停止仍留在 Driver，不放入 ActuCore。
- `../../openxr_capture_native/`（2026-09-21 从既有 pico-g1 工作树 V038 迁入，并增加天轶八点手臂及躯干显示）来自同仓 PR #151，提交 `a97bbfb56a97f1a4959120a470393a96e40fdbb8`，原路径 `generic/openxr_capture_native/`。许可证 Apache-2.0；其第三方依赖及各自许可证见该目录 `THIRD_PARTY_NOTICES.md`。原工作树新增扳机输入、中立重使能、PICO reconstruction 透视与透视失效停止；帧契约测试改接 ActuCore。
- 天轶双臂 IK 参考 PR #152 的 FK、实测状态初值和诊断思路；本次实现使用 Pinocchio + SciPy 有界最小二乘、天轶显式模型映射和保守运动范围检查，不依赖 CasADi 或 GPU。
- `models/tianyi2-official.urdf` 未修改复制自北京人形官方 `Open-X-Humanoid/TienKung_URDF`，提交 `5c221783fb92fcc4af891ef1dc0502963caf2266`，路径 `tianyi2_urdf/urdf/tianyi2.0_urdf_with_hands.urdf`。独立适用 OpenAtom Open Hardware License v1.0，完整文本见 `models/LICENSE`；不将其改标为本仓 Apache-2.0。官方设计模型不等于目标机器标定或安全验收。

官方透视实现参考：<https://github.com/picoxr/OpenXR_Demos/blob/main/app/src/main/cpp/openxr_program.cpp>。

自动发现复用框架同版本 `zeroconf 0.151.3`（LGPL-2.1-or-later）及 `ifaddr 0.2.0`（MIT），按独立 Python 包安装，不修改其源码；锁文件保留发行包哈希，许可证文本随安装包保留。Android 发现、界面、TLS 与 P-256 使用系统 API，没有新增 Android 网络或 UI 框架。

G1 `g1_ik.py` 和 `models/g1_body23.urdf` 迁移自上述 PR #152 的 `ik.py` 与 `resource/g1_body23.urdf`；后者字节未修改。原来源为 Unitree Robotics `xr_teleoperate`，revision `845b25a32f7febedf220e830952a7134897adb9d`，Copyright 2025 HangZhou YuShu TECHNOLOGY CO.,LTD，Apache-2.0。G1 模型不适用同目录仅为天轶模型保留的 OpenAtom LICENSE。

保留 G1_23 关节顺序、Pinocchio/CasADi 目标函数、滤波和 RNEA；新增实测锁定关节/手掌变换、求解期限、相对使能基准、位置残差和运动段包络检查。Driver 硬件代码单独保留来源说明。没有迁移 TeleVuer/Vuer、机器人视频或 PR 的独立 Driver 遥操宿主。

`models/g1_collision/*.STL` 七个碰撞网格来自 Unitree Robotics `xr_teleoperate`，固定提交 `817fb00c63cde15e5f24a0f8fa08e1e33ed89d3b` 的 `assets/g1/meshes/`，Apache-2.0。原始二进制不再纳入 Git；已镜像至项目 COS 的 `public/teleop/g1-collision/817fb00c63cde15e5f24a0f8fa08e1e33ed89d3b/`，由 `deploy/fetch_g1_collision.py` 在构建前下载并逐文件核验 SHA256，字节保持不变。同提交许可证保存在该目录 `LICENSE`，文件哈希见 `sha256.json`，二者也随 COS 资产分发；`ADOPTED_SOURCE.json` 保留原始迁移时的来源记录。2026-09-22 已重新读取厂商源、发布 COS，并匿名下载核对所有文件哈希。运行时从网格建立保守凸包，肩部圆柱体及其位姿直接读取上述已核验 G1 URDF；运行时不下载，缺失或篡改文件会阻止初始化，不跳过碰撞检查。

`g1_mapping.py` 的坐标矩阵与 G1ControllerPoseMapper 从上述 PR152 adapter.py 原样抽取（Apache-2.0），由 ActuCore 新映射版本使用。新版本恢复 0.02*q² 正则项；保留近零旋转数值修正和独立可达性/安全检查。
