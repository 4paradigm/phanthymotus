# D435 个人现场测试工具

本包为真实相机测试做准备，当前尚未连接 D435 或实验室设备。目标环境：Ubuntu 22.04、系统 Python 3.10、ROS2 Humble。包含 x86_64 和 aarch64 的离线 wheel；aarch64 仅完成 wheel 准备，未在该架构运行。

旧目录 `/home/agilex/goal-verifier-own-20261111-s`、旧端口 `18083` 均不使用。新目录默认 `/home/agilex/goal-verifier-roi-20261114-r1`，默认 loopback 端口 `18084`；占用时自动选择其后的空闲端口，不停止占用者。启动后以 `service.json` 的 port/url 为准。

## 1. 安装到全新个人目录

把完整现场 kit 复制到本人有权限且不存在的目录。传输包时同时传输外部校验文件，并先核对 SHA256。例如上传外层包后，在 Ubuntu 个人目录执行：

```bash
sha256sum --check FIELD-SHA256SUMS
KIT=/home/agilex/roi-kit-20261114-r1
[ ! -e "$KIT" ] && [ ! -L "$KIT" ] || { echo 'kit 目录已存在，请选择新目录'; exit 1; }
mkdir "$KIT"
tar -xzf tabletop-roi-d435-field.tar.gz -C "$KIT"
cd "$KIT"
bash install.sh /home/agilex/goal-verifier-roi-20261114-r1
```

安装前检查全部 SHA256，不覆盖已存在的部署目录，无 sudo，无网络安装。优先 `venv --without-pip`，从经过校验的 pip wheel 安装依赖，不要求 ensurepip。缺少 venv 时自动使用 `pip --target`。依赖、临时文件、缓存和日志都写入新部署目录。可显式选择：

```bash
ROI_INSTALL_METHOD=target ROI_PYTHON=/usr/bin/python3.10 bash install.sh /home/agilex/goal-verifier-roi-20261114-r1-target
```

安装失败不要覆盖旧目录重试；查看新目录 `logs/install.log` / `logs/venv.log`，选择另一个新目录。离线 wheel 只覆盖 Ubuntu 常见的 x86_64/aarch64 + CPython 3.10；不兼容的平台应另行准备，不修改系统 Python。

## 2. 只读核实相机，再填写配置

平台强制使用 Domain `42` 和官方 loopback FastDDS profile。安装包把未修改的官方 `dds-local.xml` 复制到本人的新目录；Runner/Doctor 只设置自身进程的环境，拒绝 Domain 222，不改共享节点。

**当前现场接入阻碍：** 相机此前确实在 Domain `222`，不能假定已在 42。新包不能直接读取该流。应先由有权限的维护者提供符合 Domain 42/loopback 约定的相机数据入口，然后再做以下测试。不得为过检修改相机 Domain、启动桥接、替换共享 ROS/Docker，或伪造相机已迁移。可审查的后续方案是维护者配置现有相机入口，或批准单独的数据接入适配；不在本次交付中执行。

历史记录：Domain `222`、topic `/camera_f/color/image_raw/compressed`、frame `camera_f_color_optical_frame`、期望型号 D435、序列号 `_262522073030`。配置模板中这些是待核实的历史期望，不能当作当前检测结果。

先运行 Doctor（此时服务未启动、配置记录留空会显示相应失败，ROS 图像检查仍会执行）：

```bash
ROOT=/home/agilex/goal-verifier-roi-20261114-r1
bash "$ROOT/bin/doctor.sh" > "$ROOT/evidence/doctor-before.json"
```

Doctor 只创建临时 ROS 订阅并读取图像/graph；不启动相机驱动，不发布控制话题，不改变参数。检查 domain、topic/type、解码、尺寸、frame_id、连续 header 时间和主机时间差。未运行服务时 HTTP 检查失败属预期；只有 222 上可见相机时，42 上无发布者也是当前真实结果，不能标成相机正常。

如无图像，检查现场已有相机服务，由操作者按实验室流程处理；本工具不修复或重启共享 ROS/Docker/驱动。可在 source 现有 ROS 环境后只读查看：

```bash
source /opt/ros/humble/setup.bash
export FASTRTPS_DEFAULT_PROFILES_FILE="$ROOT/dds-local.xml"
ROS_DOMAIN_ID=42 ros2 topic info /camera_f/color/image_raw/compressed --verbose
ROS_DOMAIN_ID=42 ros2 node list
# 以实际发现的 RealSense 节点名替换下面参数，不要猜测节点名：
ROS_DOMAIN_ID=42 ros2 param get /ACTUAL_REALSENSE_NODE serial_no
```

若驱动未公开实际序列号，结合只读设备枚举、设备标签和已有驱动日志人工核实。配置参数不一定等同实际连接设备，不能把 serial_no 设置值当作独立设备证明。序列号前导下划线常用于参数表达，应核对实际设备号与驱动值。

编辑新目录的 `config.json`：

- `camera_id`：本次人工核实的相机绑定标识，并在现场记录中保存实际型号、序列号与核实途径。
- `view_id`：本次固定拍摄视角标识。
- `fixed_view_record`：真实检查说明（何时、谁确认相机固定，ROI 外背景可见）；不要填虚构证明。
- `capture_time_record`：驱动 header stamp 代表采集时间的来源/检查说明；Doctor 仅验证时钟接近与时间递增，不能证明底层采集语义。
- topic/frame 必须与已核实的 42 域数据入口相符；domain_id 固定为 42。若只有 222，停止并报告接入阻碍。保持 `provenance=real, allow_test_data=false`。

不要在正式配置中使用模拟帧。默认模板的记录字段为空，未填写时启动会拒绝配置。

## 3. 启动、检查、关闭

```bash
bash "$ROOT/bin/start.sh"
cat "$ROOT/service.json"
bash "$ROOT/bin/doctor.sh" > "$ROOT/evidence/doctor-running.json"
# 需要自选端口时：ROI_PORT=18085 bash "$ROOT/bin/start.sh"
```

Doctor 的 `ros.code` 区分 `ROS_IMPORT_FAILED`（检查 Python/ROS 路径和 ABI）、`ROS_INIT_FAILED`、`ROS_EXECUTOR_FAILED`、`TOPIC_NO_PUBLISHER_DISCOVERED`、`MESSAGE_TYPE_MISMATCH`、`QOS_INCOMPATIBLE`、`TOPIC_NO_NEW_FRAMES`、`IMAGE_DECODE_FAILED`、`INVALID_OR_INSUFFICIENT_CAPTURE_TIMESTAMPS`、`CAMERA_FRAME_MISMATCH` 和 `CAMERA_STREAM_OK`。初始化失败保留 stage/type/完整 traceback；运行信息显示 Python、rclpy、NumPy 路径及 DDS profile。没有帧不能仅凭 topic 在 graph 中就证明发布正常；需要结合发布端 QoS 和相同域/profile 检查。退出清理只处理自建 Context/Executor/节点。

安装优先使用系统 `/usr/bin/python3.10` 建立 venv，再从已有 Humble setup 导入系统 ROS 包；不替换 rclpy。若 venv 路径/ABI 检查失败，可在另一个全新个人目录用 `ROI_INSTALL_METHOD=target ROI_PYTHON=/usr/bin/python3.10` 安装，由系统 Python 加载个人 deps。保留完整 Doctor 输出，不升级系统 ROS/Python。

服务日志：`$ROOT/logs/server.log`。启动会先检查配置和 rclpy，再验证 HTTP 与 Card 运行状态。Doctor 退出码 0 仅表示可操作的相机/页面/MCP 检查通过，`hardware_acceptance` 始终为 false，型号/序列号和固定视角仍需人工记录。

关闭只会给本目录记录且 PID/启动时间/命令均匹配的进程发 SIGTERM，不扫描/清理其他服务、不按端口杀进程：

```bash
bash "$ROOT/bin/stop.sh"
```

若记录 PID 已被复用，停止脚本会拒绝发信号。源程序正常退出清理 ROS context、线程和会话；重新启动后必须重新建立会话及基准。关闭浏览器不会自动停止个人服务。

## 4. Windows SSH 转发

在 Windows PowerShell 输入现场 SSH 地址，并使用 `service.json` 显示的远端端口：

```powershell
$LabHost = Read-Host 'Ubuntu SSH 主机名或 IP'
$RemotePort = 18084 # 如果 service.json 选择了其他端口，在这里改成实际值
$LocalPort = 18084  # Windows 端口被占用时可改 19084
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L "${LocalPort}:127.0.0.1:${RemotePort}" "agilex@${LabHost}"
```

保持此终端开启。浏览器 URL：`http://127.0.0.1:18084/`（如果 LocalPort 改为 19084，则用 19084）。无需开放防火墙、无需公共服务；Ubuntu 仅监听 `127.0.0.1`。关闭转发终端或按 Ctrl+C 只关闭隧道；Ubuntu 服务用 stop.sh 单独关闭。

## 5. 三态现场测试（依次 FAIL → PASS → INCONCLUSIVE）

先开始实时 RGB 预览，确认画面和采集时间持续更新。固定相机，ROI 外保留背景，在画面上拖框，选择标准预设。每次独立测试都新建会话，不复用上一次基准。

1. **FAIL**：目标区保持空白。创建会话 → 记录操作前 → 保持目标区不变 → 标记操作完成 → 获取新观测并验证。期望 FAIL / ROI_NO_CHANGE。
2. **PASS**：重新新建会话，目标区为空并记录基准。人工放入瓶子或明显物体，移开手，等待画面稳定 → 标记完成 → 验证。期望 PASS / ROI_CHANGE_MET。若物体覆盖 ROI 大部分面积，会保守返回 INCONCLUSIVE，应重新选适合的 ROI/更小物体并新建基准，不为达标修改证据。
3. **INCONCLUSIVE**：重新建会话并记录基准。验证时持续遮挡 ROI 大部分区域，或持续晃动画面，使算法无法可靠比较。期望 INCONCLUSIVE，实际错误码可能是 ROI_OCCLUDED_OR_REPLACED、ROI_NOT_STABLE 或 ROI_CAMERA_MOVED。若手只挡住小区域且保持不动，算法可能把它当成局部变化；应保存实际结果，不强行解释成通过。

四步按钮不会执行机器人操作。使用真实 ROS 图像做现场结果，不能把本地合成测试当成 D435 验收。

每次测试后选择本次预期、填写人工操作备注，点击 **保存现场证据到服务目录**。页面显示实际保存路径。结果与预期不同也保存，两者分别记录。也可以“导出证据 JSON”复制保存；不得只保留截图或一句 PASS。

证据结构：

```text
evidence/
  doctor-before.json
  doctor-running.json
  2026...-FAIL-随机标识/
    session.json       # ROI、参数、基准/标记/后帧时间、hash、source、结果历史
    result.json        # 实际三态、错误码、统计与比较证据
    before-roi.jpg
    after-roi.jpg      # 若缺少可靠后帧则可能没有此文件
    SHA256SUMS
```

`session.json` 是完整导出（去除 session token），包括历史与前后图像。复制保留 `logs/server.log`、Doctor 输出和人工相机/视角记录。图像 hash 是采集 BGR 像素的 SHA256，缩略图是重新编码的 JPEG，二者不应直接互算相等。JSON 中的 provenance 必须是 real，时间真实递增且晚于完成标记。

正式验收仍需人工核实相机身份、固定视角、三组真实结果及证据完整性。现场准备 Ready 与正式验收 Ready 是不同状态；当前交付只达到前者。


## 本版补充

断开/重连、长采集间断、时钟回退或 source epoch 变化后，新建会话重新记录基准。中性阴影/反光及纯灰度变化可能返回 ROI_PHOTOMETRIC_AMBIGUITY；选择有明显色彩/结构的目标做标准 PASS 场景，不为达标放宽阈值。证据失败不会发布半个测试目录；缺少可靠后帧的 INCONCLUSIVE 会明确标为证据不完整，不能冒充带前后图的 PASS。保留实际失败和日志。

stop.sh 使用 PID 固定句柄，仅发信号给本目录核实的进程；Linux x86_64/aarch64 的 Python 3.10 绑定缺失时使用同等内核接口，不退回可复用数字 PID。内核不支持或权限不足时给出诊断，不停止其他服务。
