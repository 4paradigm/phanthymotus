# Nav2 Perception 卡片规格

`nav2` 是单实例 `processor` 卡片，负责把机器人状态、LiDAR 点云和导航目标交给
ROS 2 Humble Nav2 运行时，提供建图、地图管理、位置标签和点到点导航能力。
卡片只发布带时效和任务标识的速度提案，不直接调用机器人 SDK，也不直接取得
物理控制权；最终执行、限幅、急停和停车确认由下游 Driver actuator 负责。

## 规格总览

| 项目 | 必填内容 |
| --- | --- |
| card id | `nav2` |
| display name | `Nav2` |
| card type | `processor` |
| 感知目标 | 消费机器人状态和 LiDAR 点云，完成建图、保存地图、定位、位置标签、全局/局部规划和导航状态管理；输出结构化速度提案 |
| input topic | 必需：`loco_state`、`lidar_cloud`；可选：`goal_pose`。首版只接受 G1 Driver 的 `/ubuntu/...` 精确绑定 |
| output topic | 只向 Canvas 暴露 `velocity_proposal`；Nav2 内部 `odom/scan/map/plan/status` topic 仅供运行和调试 |
| actions | 生命周期：`info/config/start/stop`；业务：14 个建图、地图、标签和导航 action |
| 配置 | robot namespace、地图持久化目录、输入新鲜度、速度上限、提案 TTL 和控制面超时；非法配置 fail closed |
| 模型/算法 | 无训练模型；基线为 ROS 2 Humble Nav2、SLAM Toolbox、AMCL、NavFn、DWB 和 velocity smoother |
| 部署 | 目标为 Jetson ARM64；Nav2 companion 纳入正式 Perception Compose 项目，与 Perception 一起启动和重启 |
| 测试 | 契约校验、纯逻辑单测、ROS 2 回放、Compose smoke、Canvas 闭环、故障注入和 owner 授权的 G1 真机验收 |

## 数据链路

```text
Driver.loco_state ──┐
                     ├──> Nav2 ── velocity_proposal.v1 ──> Driver.loco ──> Robot
Driver.lidar_cloud ─┘
goal_pose.v1（可选）─┘
```

导航卡片与 Driver 的职责边界：

- `nav2` 负责传感器适配、TF、地图、定位、规划、导航状态和速度提案。
- Agent Core 负责 Canvas 生命周期、工具调用，以及导航任务与 Driver lease 的关联。
- Driver `loco` 负责校验 `nav_id`、sequence、TTL、速度边界、主运控、急停和停车确认。
- 输入断流、TF/地图失效、Nav2 未 ready 或 Driver 未授权时必须拒绝导航，不能绕过门禁直连机器人 SDK。

## Input topics

topic 名以 Canvas 的端口绑定为准。首版 companion 与现有 G1 Driver 合同冻结为
`/ubuntu/...`；连接其他 namespace 前必须先完成 companion 参数化和相应回放验收，
当前 `config` 会 fail closed 拒绝其他 namespace。

| port | 当前 G1 示例 | ROS 2 type / format | QoS | 数据合同 |
| --- | --- | --- | --- | --- |
| `loco_state` | `/ubuntu/loco/state` | `std_msgs/msg/String` / `data/json` | `BEST_EFFORT + KEEP_LAST(depth=10) + VOLATILE` | 当前兼容 `unitree.g1.loco_state.legacy`，后续兼容带源时间戳和 frame 的 `phanthy.g1.loco_state.v2`；期望 10 Hz，最大 age 500 ms |
| `lidar_cloud` | `/ubuntu/lidar/cloud` | `std_msgs/msg/UInt8MultiArray` / `sensor/pointcloud` | `BEST_EFFORT + KEEP_LAST(depth=10) + VOLATILE` | 当前兼容 MID360 legacy envelope，后续兼容 `phanthy.sensor.pointcloud.v2`；期望 10 Hz，最大 age 500 ms |
| `goal_pose`（可选） | `/ubuntu/navigation/goal_pose` | `std_msgs/msg/String` / `data/json` | `RELIABLE + KEEP_LAST(depth=10) + VOLATILE` | `phanthy.navigation.goal.v1`；`map` frame；每条消息必须有唯一 `goal_id` |

legacy payload 没有源时间戳或 frame 时，adapter 必须明确标记接收时间和配置来源，
不得把接收时间伪装成设备源时间。LiDAR 使用 sensor-data QoS，以兼容现有
BEST_EFFORT 发布端。

`goal_pose` JSON 最小结构：

```json
{
  "schema": "phanthy.navigation.goal.v1",
  "goal_id": "room-a-001",
  "x": 1.2,
  "y": -0.8,
  "yaw": 0.0,
  "speed": 0.15
}
```

`x/y` 单位为米，`yaw` 单位为弧度，坐标系为 `map`。首版固定使用绕障模式，
不向 Canvas 暴露无可选值的 `mode` 参数。可选 topic 输入最终仍转换为普通
`navigate_to_pose` 调用，不能绕过 Agent Core 和 Driver lease。

当前官方 Agent Core 尚未实现本卡片声明的 `x-topic-actions` 消费逻辑，因此
`goal_pose` 端口目前只完成 schema/连线声明，不能作为已打通的执行入口。MCP action
仍可用于 shadow 规划；物理执行必须等待 Core 提供受信 `nav_id`/Driver lease 绑定。

## Output topic

| port | 推导规则 | ROS 2 type / format | QoS | 数据合同 |
| --- | --- | --- | --- | --- |
| `velocity_proposal` | `/<namespace>/navigation/nav2/velocity_proposal` | `std_msgs/msg/String` / `data/json` | `RELIABLE + KEEP_LAST(depth=10) + VOLATILE` | `phanthy.navigation.velocity_proposal.v1`，目标频率 20 Hz，`base_link` frame，TTL 不超过 250 ms |

提案至少包含 `nav_id`、单调递增 `sequence`、`issued_at_unix_ms`、`ttl_ms`、
`frame=base_link`、`nav_status` 和 `velocity{x,y,yaw}`。终态必须发布零速提案。
提案本身不携带执行权限，`shadow_only=true`、`physical_execution=false` 表示
卡片没有直接执行机器人动作；下游 Driver 是否执行由独立授权决定。

首版提案边界：前进不超过 `0.15 m/s`，后退不超过 `0.05 m/s`，禁止横移
（`y=0`），偏航角速度绝对值不超过 `0.35 rad/s`，平面合速度不超过
`0.18 m/s`。Driver 必须再次独立限幅。

## Actions

### 生命周期

| action | 请求 | 行为与返回 |
| --- | --- | --- |
| `info` | 无 | 只读返回 `state`、输入连线、输出 topic、runtime mode、active map、active goal、readiness 和 blocker；不得启动进程或改变配置 |
| `config` | 仅接受 `configSchema` 字段 | 幂等校验并保存配置；运行中拒绝更新，控制面 timeout 在下一次 `start` 生效 |
| `start` | Canvas 提供 `input_bindings` | 校验两路必需输入、Nav2 companion 和地图目录；ready 后进入 `idle/ready`，重复调用幂等 |
| `stop` | 无 | 取消活动导航并停止卡片会话，释放订阅和任务状态；重复调用幂等。物理停车确认仍由 Driver 返回 |

### 业务动作

| action | 参数 | 语义 |
| --- | --- | --- |
| `start_mapping` | `map_name:string` | 自动切换到 mapping runtime，开始新地图；已存在同名地图时拒绝覆盖 |
| `stop_mapping` | 无 | 停止建图，原子保存地图和 pose graph，然后切回 localization 并加载新地图 |
| `tag_place` | `name:string`、`description?:string` | 在当前 `map` 位姿记录语义位置标签 |
| `untag_place` | `name:string` | 删除当前地图中的标签；不存在时返回可判定错误 |
| `list_tags` | 无 | 返回当前地图的全部标签及其 pose |
| `list_maps` | 无 | 返回已保存地图、状态和元数据 |
| `delete_map` | `map_name:string` | 删除未加载、未建图中的地图；活动地图不得删除 |
| `load_map` | `map_name:string` | 加载地图并进入 localization；定位未 ready 时不得宣称成功 |
| `navigate_to_tag` | `tag_name:string`、`speed?:number` | 以固定绕障模式非阻塞创建导航任务并返回唯一 `nav_id` |
| `navigate_to_pose` | `x:number`、`y:number`、`yaw:number`、`speed?:number` | 以固定绕障模式非阻塞创建 `map` frame 导航任务并返回唯一 `nav_id` |
| `wait_navigation_done` | `stall_timeout?:number` | 等待当前任务到达、取消、超时或错误，返回 terminal receipt |
| `pause_nav` | 无 | 暂停当前导航并等待 Nav2 接受；无活动任务时幂等返回 |
| `resume_nav` | 无 | 恢复已暂停任务；状态不允许时明确拒绝 |
| `stop_nav` | 无 | 取消当前任务，发布终态零速并等待 Nav2 terminal；物理停车确认由 Driver 完成 |

`speed` 默认 `0.15 m/s`，范围 `0.05–0.15 m/s`；首版导航固定允许绕障。
参数错误、not-ready、timeout、cancelled 和内部错误必须具有不同的结构化
`error_code`，不能只返回 HTTP 200 或日志文本。

## 配置

卡片为单实例。Canvas `configSchema` 只暴露 `backend` 和三个控制面 timeout；其余字段
是 `perception/config.yaml` 与 companion 镜像之间的首版部署合同，用户不能在 Canvas
热改：

| 字段 | 默认值 | 范围/更新行为 |
| --- | --- | --- |
| `namespace` | `ubuntu` | 首版固定；其他 namespace 会被拒绝 |
| `map_storage_dir` | `/maps` | 首版固定；由正式 Compose 持久化挂载 |
| `input_max_age_ms` | `500` | 首版固定，超过后导航 fail closed |
| `max_forward_mps` | `0.15` | 首版固定安全上限 |
| `max_reverse_mps` | `0.05` | 首版固定安全上限 |
| `max_lateral_mps` | `0.0` | 首版禁止横移，非零提案 fail closed |
| `max_yaw_rps` | `0.35` | 首版固定安全上限 |
| `max_planar_mps` | `0.18` | 首版固定安全上限 |
| `proposal_ttl_ms` | `250` | 首版固定；Driver 仍独立校验 |
| `request_timeout_sec` | `30` | `1–120` s；控制面请求超时 |
| `runtime_switch_timeout_sec` | `120` | `10–300` s；mapping/localization 切换超时 |
| `discovery_timeout_sec` | `5` | `0.5–30` s；等待 companion command subscriber |

`shadow_only` 不是用户可关闭的配置：本卡片始终是 proposal-only。配置更新不能
自动扩大速度边界，不能静默重启活动任务。首版把数据面参数冻结为已验收值，避免
Canvas 配置与 companion 实际运行参数出现 silent drift。

## 算法与 TF

首版不包含可训练模型，使用：

- SLAM Toolbox：建图和 pose graph 持久化；
- Map Server + AMCL：已保存地图定位；
- NavFn：全局路径；
- Rotation Shim + DWB + velocity smoother：航向偏差较大时先原地转向，
  再以零横移的前向轨迹跟随路径并平滑限速。

导航默认不使用 G1 的横移能力：DWB 只采样 `x/yaw`，velocity
smoother 的 y 速度和加速度均为 0，提案协议也拒绝非零 y。这只约束
Nav2 速度方向；步态、躯干和手臂姿态仍由 Driver/Loco 主运控负责。

TF 链固定为：

```text
map -> odom -> base_link -> lidar_frame
```

传感器外参必须来自 Driver/机器人模型的可追溯配置，并在启动时校验；不能使用
零外参或视图层坐标变换代替真实 TF。依赖版本、镜像 digest 和许可证必须在实现
阶段写入 source lock/第三方清单，不能依赖浮动 latest。

## RViz 地图与导航可视化

companion 包内置 [nav2.rviz](companion/g1_nav2/rviz/nav2.rviz)，固定以 `map`
为坐标系，显示已保存/实时地图、全局路径、局部 costmap、LaserScan、
odom、机器人 footprint 和 TF。该配置只订阅标准 ROS 2 topic，不含
`SetGoal` / `InitialPose` 等会改变导航状态的 RViz 工具。

在与 G1 同一 DDS 网络的 Ubuntu 开发机上，从仓库根目录执行：

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4

ros2 topic info /map -v
ros2 topic info /ubuntu/navigation/nav2/scan -v
rviz2 -d perception/plugins/nav2/companion/g1_nav2/rviz/nav2.rviz
```

如开发机只安装 ROS 2 Jazzy，可将第一行替换为
`source /opt/ros/jazzy/setup.bash`，但必须先用上述 `topic info` 确认标准消息
可发现；这条路径只用于可视化，不用它发布目标或调用 Nav2 service。

在本地 colcon 安装过 `g1_nav2` 包后，也可从包的 share 目录打开：

```bash
source install/setup.bash
rviz2 -d "$(ros2 pkg prefix --share g1_nav2)/rviz/nav2.rviz"
```

主视图中 `/map` 只在 mapping runtime 正在建图，或 localization runtime
已成功加载地图时有数据。`Global Costmap` 默认关闭以避免遮住原图，
需要排查全局避障时可在 Displays 面板手动打开。本配置不要求 G1 安装
RViz，也不会从可视化机器上发送运动命令。

## 部署

- 目标运行环境：G1 Jetson ARM64、Ubuntu 20.04、ROS 2 Humble、DDS host network。
- `nav2` Plugin 运行在正式 Perception Bundle；Nav2 companion 作为同一正式
  Perception Compose 项目的受管 service 随 Perception 一起启动。
- 卡片进程不得调用 Docker、不得挂载 Docker socket，也不得自行创建容器。
- Compose 将宿主机 `/opt/phanthy-motus/data/nav2/maps` 挂载到容器 `/maps`；容器重建和机器人重启后地图仍可恢复。
- 不需要 CUDA、训练模型、云端服务或 Unitree SDK。
- x86 CPU 仅用于契约、回放和 Compose smoke；不能代替 ARM64 或真机验收。

正式 `perception/deploy/service.yml` 已加入 companion service；整体执行
`docker compose up -d` 时两者会一起启动和重启。Canvas 的 `start/stop` 只管理卡片
ROS 资源和导航任务，不负责创建或销毁基础容器。

当前 Agent Core 部署器仍按“一个 `service.yml` 只有一个 service”的旧假设工作：它会
合并两个 service，但仅对第一个执行 `docker compose up --no-deps`。因此从 Dashboard
首次部署 Perception 时，Nav2 companion 不会被自动拉起；还需要部署器新增 multi-service
启动和 sidecar 镜像解析能力。该修改超出本 PR 允许的 Perception 范围。

### G1 测试容器

在正式部署器支持 multi-service 前，可使用
`deploy/scripts/owner-start-g1-test-containers.sh` 在 G1 上显式启动两个临时测试容器：

- `embodied-perception-test`：使用默认 `perception/config.yaml` 的完整 Perception Bundle，不是 nav2-only 配置。
- `embodied-perception-nav2-test`：Nav2 companion，持久化地图目录与正式 Compose 一致。

脚本不修改 `/opt/phanthy-motus/docker-compose.yml`，不启动 Canvas，不调用 Nav2 action
或 Driver。两个容器的 restart policy 固定为 `no`，仅用于当次人工调试。
启动前必须保持 Core/Driver 运行且 Canvas project 已停止。若 Core 已启用
token 认证，脚本不会自动探测凭据：可由调用方显式提供 `CORE_ACCESS_TOKEN`，或在
现场确认 Canvas 已停止后设置 `I_CONFIRM_CANVAS_STOPPED=1`。两者都缺失时 fail closed。

从仓库根目录执行：

```bash
git switch feat/Nav2-card
git pull --ff-only origin feat/Nav2-card

BUILD_DATE="$(date +%y%m%d)"
COMMIT="$(git rev-parse --short=7 HEAD)"
export PERCEPTION_IMAGE="local/phanthy-motus/perception:release.${BUILD_DATE}.${COMMIT}-jetson"
export NAV2_IMAGE="phanthy-nav2:nav2-card-${COMMIT}"
export ROS_BASE_IMAGE="bj-warehouse.tencentcloudcr.com/phanthy-motus/ros-base@sha256:82d45949e7c3fd85e6baf4a2b24b384a3ec020a5e237c5f801bc2f2269ca649f"

./deploy/build_perception.sh --variant jetson --mirror tuna

(
  cd perception/plugins/nav2/companion
  DOCKER_BUILDKIT=0 NAV2_IMAGE="${NAV2_IMAGE}" ROS_BASE_IMAGE="${ROS_BASE_IMAGE}" \
    docker compose --env-file source-lock.env build nav2
)

I_CONFIRM_CANVAS_STOPPED=1 STAGE=preflight \
  bash perception/plugins/nav2/deploy/scripts/owner-start-g1-test-containers.sh

I_AM_G1_OWNER=1 I_CONFIRM_CANVAS_STOPPED=1 STAGE=start \
  bash perception/plugins/nav2/deploy/scripts/owner-start-g1-test-containers.sh
```

G1 构建显式使用已锁定的内部 ARM64 ROS Humble 基础镜像，并关闭 BuildKit 的
Docker Hub metadata 查询；因此即使 G1 无法访问 `registry-1.docker.io` 也能使用本地缓存构建。

`start` 会重复执行全部 preflight，等待 MCP `tools/list` 真实返回 `nav2`；任一容器
提前退出或超时时，它会输出两侧日志尾部，并且只清理本次创建、携带
`com.phanthymotus.test-owner=nav2-card` 标签的测试容器。

查看状态或停止：

```bash
STAGE=status \
  bash perception/plugins/nav2/deploy/scripts/owner-start-g1-test-containers.sh

I_AM_G1_OWNER=1 I_CONFIRM_CANVAS_STOPPED=1 STAGE=stop \
  bash perception/plugins/nav2/deploy/scripts/owner-start-g1-test-containers.sh
```

`stop` 同样要求 Canvas project 已停止，并拒绝操作缺少上述 owner 标签的同名容器。
上述流程每次都会先拉取当前分支，再根据新 `HEAD` 生成两个镜像 tag；Docker
层缓存会复用未变的构建步骤，但不会静默启动上一个 commit 的镜像。

范围外依赖：当前官方 Agent Core 没有消费 `x-execution-control` / `x-topic-actions`，
所以还不能把每个导航任务的受信 `nav_id` 自动绑定到 Driver `loco` lease。该能力应
由 Core 单独实现；本卡片不会通过直接调用 Driver 或伪造授权绕过这一门禁。

## 验收

实现完成后至少通过：

1. `validate_card.py`：tool id、type、生命周期、topic 和 schema 一致。
2. 单元测试：参数边界、状态机、重复 start/stop、地图原子保存、标签 CRUD、
   goal correlation、终态零速和结构化错误。
3. ROS 2 回放：合成/录制的 state 与点云能够产生新鲜 odom/scan、完整 TF、地图、
   非空路径和 bounded proposal；输入断流后在时限内停止提案。
4. Compose smoke：Perception 与 Nav2 companion 按依赖顺序启动，重启后地图恢复；
   companion 不可用时卡片显示 not-ready。
5. Canvas：两路必需输入和一个 proposal 输出可正确连线；固定绕障模式不显示
   `mode` 选项，缺输入和过期输入可见失败。`goal_pose` 执行要求 Core
   消费 `x-topic-actions`，在官方 Core 提供该能力前只验证 schema 和连线。
6. G1 真机：由 owner 明确授权并持有遥控器/急停，验证建图、保存/加载、标签导航、
   绕障重规划、到达、取消、Driver 停车确认和 lease 释放。

只有控制面注册、`tools/list`、进程存活、shadow proposal 或短距离直行都不构成
完整验收。真机运动必须单独授权，自动化测试和 CI 禁止下发物理动作。

## 当前阶段

阶段三到五的代码已实现。阶段六已在 amd64 Ubuntu 集成主机完成
镜像构建、Compose smoke、18 项测试、MCP 唯一注册和配置恢复、Canvas
`input_topics` 连线、建图/停图/原子存图、结构化 proposal、传感器断流
零速保护与恢复、10 轮 start/stop、非法输入隔离和 Perception 重启验证。

当前仍是 Draft，不能宣称完整可用：本轮使用的是严格匹配 G1 legacy
合同的确定性 fixture，没有真实 G1 录包，合成点云也不能证明非空房间
路径规划。此外，官方 Core 尚未消费 `x-execution-control` / `x-topic-actions`，
所以 Driver lease 和 `goal_pose` 执行闭环仍是外部 blocker。ARM64 生产镜像、真实
录包、Dashboard 人工界面确认以及 G1 真机执行属于后续验收。
