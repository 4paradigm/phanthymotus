# controlled_semantic_spatial 内部 Nav2 模块

该模块在统一 `controlled_semantic_spatial` 卡片内负责目标导航、路径规划、
局部避障和有时效的速度提案。建图、定位、点云去畸变、地图存储和
地图可视化由同卡片 FAST-LIVO2 模块负责。它不再单独注册 MCP 卡片，
不订阅 Driver 原始点云，也不再
启动 SLAM Toolbox、AMCL 或 Map Server。

## 责任边界

```text
Driver sensors
    |
    v
FAST-LIVO2 internal module
    |-- /ubuntu/navigation/odom                 nav_msgs/Odometry
    |-- /ubuntu/navigation/cloud_registered     sensor_msgs/PointCloud2
    |-- /ubuntu/navigation/static_map            nav_msgs/OccupancyGrid
    `-- TF: map -> base_link
              |
              v
      Nav2 internal module
              |-- planner + rolling costmaps + controller
              |-- /plan                                  nav_msgs/Path
              `-- /actucore/navi/cmd
                                      |
                                      v
                              兼容的下游消费者（需独立验证）
```

- FAST-LIVO2 负责把原始 `camera_init -> aft_mapped` 输出归一化为下表的
  `map/base_link` 合同。不能只改 topic 名来伪装坐标系。
- Nav2 只消费已去畸变、已对时、已归一化 frame 的数据，并在任一
  输入过期、frame 不符或 `map -> base_link` 不可用时 fail closed。
- Agent Core 负责 Canvas 生命周期和 MCP 请求转发；Nav2 planner
  在接受每个新目标时生成独立 `nav_id`。
- 物理执行、限幅、急停、TTL 和停车确认须由下游独立验证；旧 Driver `loco` 不承诺消费该格式。

## 输入合同

| port | topic | type / QoS | 必要条件 |
| --- | --- | --- | --- |
| `livo_odom` | `/ubuntu/navigation/odom` | `nav_msgs/msg/Odometry`; `BEST_EFFORT + KEEP_LAST(5)` | `header.frame_id=map`，`child_frame_id=base_link`，ROS system time，接收 age 最大 500 ms |
| `registered_cloud` | `/ubuntu/navigation/cloud_registered` | `sensor_msgs/msg/PointCloud2`; `BEST_EFFORT + KEEP_LAST(1)` | `header.frame_id=map`，已运动去畸变，ROS system time，接收 age 最大 500 ms |
| `static_map` | `/ubuntu/navigation/static_map` | `nav_msgs/msg/OccupancyGrid`; `RELIABLE + KEEP_LAST(1) + TRANSIENT_LOCAL` | `header.frame_id=map`，直接累计静态图，`-1/0/100` |

Odometry 和 registered cloud 的 source stamp 必须同时可用。当 FAST-LIVO2 还在
直接发布 `camera_init` / `aft_mapped` frame 时，Nav2 会返回
`fast_livo2_odom_frame_invalid` 或 `registered_cloud_frame_invalid`，而不会启动导航。
接收 freshness 仍严格按 500 ms 判定；source stamp 使用独立的 1.0 s 上限，
容纳 FAST-LIVO2 对点云解码、坐标变换和发布带来的有界处理延迟。接收断流超过
500 ms 或源时间戳超过 1.0 s 仍然 fail closed。上游 adapter 以 latest-only
QoS 消除内部积压，并且只在 odom/TF 历史已包围 cloud 源时间戳后发布
registered cloud。Nav2 readiness 直接检查该 cloud 时间点的
`map -> base_link` TF；“最新 odom 与最新 cloud”的时间差只作诊断，不再冒充
配对结果阻塞导航。静态图累计和 1 Hz Canvas/OccupancyGrid 编码不在该发布
快路径执行；它们使用独立 latest-only 后台任务和锁，因此不会周期性把
registered cloud 的接收 age 推过 500 ms。BT 和分段控制器均为 20 Hz，
可以用最新位姿在每个控制周期检查过点、转向和障碍；标准输出频率由下游接口
协商，停车零速不等待下一次周期发布。

Nav2 controller server 的 `odom` 通过启动 remap 接到上述 canonical odometry，
分段控制器使用 Nav2 传入的速度，不独立订阅或检查速度的有效性与新鲜度。
FAST-LIVO2 原始 twist 不可用，adapter 根据 `base_link` 位姿和源时间差分估计
体坐标速度；有效估计的六个 twist 分量必须有限且协方差对角线全部为正。
协方差为名义值，不是经过标定的精度保证；旧版全零协方差不是有效速度标记。
能够发布的无效估计用非有限 twist / 负对角线标识；上游坏帧被 adapter 丢弃
或发布者失联时可能没有无效消息。连续两个控制周期速度接近零不要求不同样本，
也不是六轴完全静止检查或物理停车证明。位姿差分不能识别定位冻结时机器人
仍在运动；速度估计依赖上游位姿及传感器外参真实可靠。
Nav2 的 `SimpleGoalChecker` 是独立位置/朝向判断，不提供反馈新鲜度或物理
停稳保证；action 成功不可作为停车确认。下游仍须有独立 TTL、停车及急停保护。

坐标目标通过 MCP `navigate_to_pose` 提交，语义目标由同卡片进程内 planner
接收；不再提供 `goal_pose` 外部 topic 输入。升级时需移除画布中该端口旧连线，
刷新卡片定义并保存。已准入调用仍由统一卡片的停止栅栏追踪和取消；未确认停止时须重试。
单元测试不能视为 ROS 端到端可用或物理停车保证。

## 输出合同

| port | topic | type / QoS | 语义 |
| --- | --- | --- | --- |
| `control_velocity` | `/actucore/navi/cmd` | `std_msgs/msg/String`; `RELIABLE + KEEP_LAST(1)` | `motus.control/1` 的六维 twist，消息格式与 Navi 相同 |
| `plan` | `/plan` | `nav_msgs/msg/Path`; `RELIABLE + KEEP_LAST(1)` | Nav2 原生 `map` 全局路径，Canvas 显示起点、终点、路径长度和折线 |
| `costmap` | `/global_costmap/costmap` | `nav_msgs/msg/OccupancyGrid`; `RELIABLE + KEEP_LAST(1) + TRANSIENT_LOCAL` | Nav2 实时二维全局代价地图，作为卡片默认预览，叠加路径、位姿、终点和膨胀障碍 |

桥接器内部缓存最新的 shadow 速度候选，超过 250 ms 或任务上下文不匹配时
不再发布其非零速度。终态立即发布一次标准零速，随后停止该任务的周期输出；
终态回执仍保留供查询和幂等重放。此 250 ms 是内部候选新鲜度上限，不是
Driver 的 TTL 保证。
卡片自身不直接操作底盘，但 `control_velocity` 可被兼容的下游控制卡片执行；
不能将它视为隔离的 shadow 输出，也不能仅凭消息发布或回执断言物理动作或停止。
状态及回执不再输出 `shadow_only` 和 `physical_execution`：两者不能表明下游底盘
是否实际执行。旧版内部配置中的 `shadow_only` 仅为兼容与校验保留，不是整条
控制链的物理执行状态；要确认是否运动必须结合执行侧和现场反馈。
非零控制消息的 `obs_stamp_ms` 取速度候选产生时快照的里程计与障碍点云中
较早的源时间；时间域或新鲜度检查不通过时该输出转为零速。这些就绪输入
时间并不能证明 Nav2 控制器实际消费了对应样本。桥接器主动生成的零速在
缺少可用源时间时使用零速生成时间，并以 `obs_stamp_kind` 明示，而非传感器采集时间。

速度限制：

- 导航请求 `speed` 范围为 `0.30–1.00 m/s`，默认 `0.50 m/s`；
- X 轴为仅前进合同：Nav2 controller、velocity smoother 或恢复树产生的
  负 X 均在 proposal 边界强制归零，不会向 Driver 发布倒退命令；
- `config` 动作可配置 `min_x_mps/max_x_mps`、`min_y_mps/max_y_mps`
  和单一 `rotate_speed_rps`；X/Y 的 `min_*` 是死区，yaw 则保留
  方向并固定为 `rotate_speed_rps` 的幅值；
- 默认 `X=0.30–1.00 m/s`、`Y=0–0 m/s`、`yaw=0.30 rad/s`；
  `max_y_mps=0` 保持当前禁止横移的行为；
- 每次导航的 `speed` 继续作为正向 X 上限；当该上限小于 X 死区时，
  它同时成为有效死区，不会把提案反向放大；
- 平移与转向强制互斥：先应用各轴死区，保留的 `|yaw| >= 0.20 rad/s` 时
  只原地转向（`x=y=0`），低于该阈值时只平移（`yaw=0`），不会向
  Driver 发布同时包含平移和转向的动作；
- 上述死区会把低于硬件有效阈值的减速提案收敛为零，不再把
  它们抬高成大动作；readiness blocker、暂停和终态零速仍保持严格零值；
- 分段控制器的终点容差为 `0.18 m / 0.10 rad`，位置容差严于 Nav2
  GoalChecker 的 `0.20 m / 0.10 rad`；位置到达后只进入最终原地转向，
  位置和朝向都达标后立即输出零速，最终 `arrived` 仍以 Nav2
  action result 为准；
- Nav2 Humble 固定创建 pose 与 through-poses 两个内部 navigator；卡片为二者
  显式加载无 BackUp 的恢复树，只使用清除代价地图、原地转向和等待。公开动作
  仍只有 `navigate_to_pose`；
- 下游是否提供二次限幅、TTL、急停和停车确认须独立验证。

## Actions

| action | 参数 | 语义 |
| --- | --- | --- |
| `config` | X/Y/yaw 的最小/最大速度绝对值 | 卡片停止时配置，下次 `start` 生效 |
| `navigate_to_pose` | `x`、`y`、`yaw`、`speed?` | 先校验全局代价地图目标格，通过后在 `map` frame 中非阻塞创建导航任务 |
| `wait_navigation_done` | `stall_timeout?` | 等待到达、取消、超时或错误 |
| `pause_nav` | 无 | 暂停当前导航 |
| `resume_nav` | 无 | 恢复已暂停导航 |
| `stop_nav` | 无 | 取消任务并发布终态零速 |

`start_mapping`、`stop_mapping`、地图 CRUD、tag CRUD 和 `navigate_to_tag` 已从
Nav2 公共合同移除。这些能力如果需要，应由 FAST-LIVO2/位置服务单独暴露，
不应再把 SLAM 运行时塞回 planning 模块。

为兼容重构前 Canvas 已保存的卡片实例，废弃字段
`runtime_switch_timeout_sec` 会在配置迁移时被忽略；Nav2 已不再切换
mapping/localization 运行模式。其他未知配置字段仍会拒绝。

## 规划与控制

- NavFn 生成全局路径。
- planner bridge 在创建 Nav2 action 前读取最新
  `/global_costmap/costmap` 目标格。代价 `>=99`（inscribed/lethal）直接返回
  `goal_in_collision`；地图缺失/过期、目标在当前 rolling window 外或未知格都有
  独立错误，不再进入长时间恢复流程。成功回执包含 `goal_cell`。Canvas 代价图
  只把同一拒绝阈值 `>=99` 的格子标红；`1..98` 的软 inflation 仍由 Nav2 用于
  路径评分，但不再渲染成类似不可通行区域。
- global/local costmap 都是 rolling window。global 插件顺序固定为
  `StaticLayer -> live ObstacleLayer -> InflationLayer`：StaticLayer 消费完整的
  `/ubuntu/navigation/static_map`，live layer 与 local costmap 一样消费实时
  `/ubuntu/navigation/cloud_registered` 并开启 marking/raytrace clearing。
  live layer 表达当前动态环境并依靠 raytrace clearing 更新。FAST-LIVO2
  adapter 的静态图则恢复为直接累计：同一扫描按 `0.10 m` 体素去重，有效
  高度带内的体素首次出现即写入 StaticLayer，不等待多帧确认、不做动态分量
  跟踪，也不通过后续自由射线删除。该模式保留稀疏墙面证据，但建图期间出现
  的人员或移动物体可能固化进地图，建图现场应尽量保持静止。
- 稀疏点云地图不会把尚未观测的格子当成硬障碍；global costmap 不跟踪
  unknown，NavFn 允许穿过未知格。已确认的静态/实时障碍、footprint 与目标格
  碰撞检查仍然 fail-closed，避免大场景仅因地图稀疏而完全无路可走。
- 两层都使用卡片的 `obstacle_min_height_m/obstacle_max_height_m` 预过滤；旧
  `/ubuntu/navigation/obstacle_map` 仅作兼容诊断，不再以 `clearing=false`
  驱动全局代价图。confirmed static PCD 加载时还要求保存的高度带与当前配置
  完全一致，不一致则拒绝加载并要求恢复配置或重新建图。
- 两张 costmap 保留 `inflation_radius=0.55 m`。当前矩形 footprint 的外接半径
  约 `0.425 m`，但控制器只把 `LETHAL_OBSTACLE` 及以上视为碰撞，不能将
  膨胀半径与足迹半径之差当作已经保证的硬性停车余量。
  heartbeat/status 的 `global_costmap` 字段现在分别给出 inflated、inscribed、
  lethal 数量及比例，用于独立评估静态图与实时层是否过度占用，不通过
  缩小安全边界掩盖问题。
- NavFn 仍生成全局路径，local/global costmap 仍以各自频率重规划和
  更新障碍。本地 DWB/Rotation Shim 替换为 20 Hz 分段控制器：
  从最新全局路径上选当前 local costmap 内最远的直线可通点，执行
  `STOP_CHECK -> ROTATE -> DRIVE -> FINAL_ROTATE -> ARRIVED`。任一
  转向或直线 footprint 检查失败时进入 `BLOCKED` 并立即给零。同一
  目标的周期重规划会保留最终停稳检查和 `FINAL_ROTATE`，不会因新路径
  重新进入平移；
  终点坐标或朝向真正改变时才重置状态。
- 控制器每 50 ms 使用最新 `map -> base_link` 位姿检查是否已经过
  线段终点、航向是否偏离和下一段是否可通。直行前不再按当前机器人朝向
  进行额外前向足迹预检，也不再用侧向、反向或角速度反馈拦截直行；原有按
  线段方向的足迹检查仍保留。机器人实际朝向、侧滑或残余旋转可能使运动
  偏离已检查路径；离散二维足迹检查也不保证连续扫掠或上身避障。
  切段前要求 Nav2 传入的速度在连续两个控制周期接近零；两个周期可以使用同一
  样本，控制器不独立检查该速度的新鲜度或有效性，因此不能据此证明物理停稳。
  上游 readiness 的 odom 门槛独立于此控制器。一旦位姿证明已经越过当前线段
  终点，无论横向误差是否还在容差内都先立即给零，停稳后再判断到达或重新选段。
- proposal bridge 直接消费 controller 的 raw 输出，不经过
  `OPEN_LOOP velocity_smoother`。执行器的可执行死区已在 proposal 出口处理；
  再叠加一个以“上次命令”作为反馈的平滑器会推迟首个可执行动作。
  构建时用校验过 hash 的最小上游补丁移除该节点，镜像也不编译对应包。
- 任一 readiness blocker 会抑制非零 shadow velocity 并输出标准零速。
- 直线段因实时代价地图进入 `BLOCKED` 后保持零速；下一条全局路径
  到来时丢弃旧线段并重新选择可通直线段，不会在障碍清除或规划绕行后
  永久停留在上一次 `segment_blocked`。
- 标准速度按下游接口协商的频率发布（默认 10 Hz）；切段、障碍、readiness
  blocker、协议错误和任务终态的首个零速直接发布，不等待周期定时器。
  topic 使用 `KEEP_LAST(1)`，限制队列深度，但不证明下游不会执行过期命令。
- Nav2 bringup 的 `/cmd_vel` remap 限定在 scoped launch group 内；
  proposal bridge 始终检查真正的根 `/cmd_vel`，发现外部发布者仍会拒绝导航。

上游 adapter 在发布 planning 输入前对单帧 live cloud 同时实施 200,000 点和
64 MiB 数据区上限；静态证据和 confirmed static map 各自最多 200,000 点，
超限均 fail closed。保存/加载会话最多包含 64 个 raw PCD，raw 与 static PCD
合计最多 512 MiB；manifest、PCD header 和 ASCII 单行都有 64 KiB 有界读取，
ASCII token 布局和数据行必须与声明精确一致。`load_map` 会在
停止旧定位前端前验证 manifest、全部 PCD、障碍高度带和静态点数，并准备
机器人周围的有界 OccupancyGrid 滚动窗口，因此 planning 不会在未验证新图上
切换 StaticLayer，也不会按全图包围盒分配稠密数组。

### 分段状态与并发保护

控制器把 `phase` / `segment_index` / `remaining_distance_m` /
`heading_error_rad` / `stop_reason` / `pose_stamp_ns` 发到内部
`/ubuntu/navigation/nav2/segment_status`，planner bridge 将它合并到现有
status 的 `execution`，不新增 Canvas 连线点。

shadow velocity 回调在计算后、写入 latest-only 缓存前会在同一互斥区内
二次校验 `nav_id/attempt/status`；标准速度发布定时器在真正发布前再次校验任务
上下文和 readiness。Nav2 action result、pause/resume 或新任务已使样本过期时，
该样本不发布；因此终态零速不会被旧回调用更高 `seq` 覆盖。当前自动证据覆盖纯函数和源码
合同；终点不徘徊及真机时序仍需目标本体验收。

## 统一卡片连线

odom、registered cloud 和 static map 由 `ControlledSemanticSpatialPlugin` 固定为同容器
内部 topic，不需要 Canvas 连线。刷新卡片定义后移除旧端口连线并连接
`control_velocity`。格式匹配本身不证明 Driver 已订阅或联调通过。该 topic 与 Navi 共用，
两张卡片不得同时向其发布；G1 `loco_servo` 默认 dry-run，实际停止与 watchdog
行为尚须独立验证。

`map_view` 为统一卡片的公开地图输出，`livo_odom` 只作为同容器内部定位/规划
数据，不生成 Canvas 公共端口。主线 Agent Core 不提供本 PR 原有的专用路径、
目标、位姿叠加及 2D/3D 地图渲染增强，不能假定这些数据会在 Dashboard 显示。

统一卡片不再公开 costmap 输出端口。规划器仍使用
`/global_costmap/costmap`，需要排障时可通过下述 RViz 流程检查。

主线 Agent Core 不支持本 PR 曾增加的 odom/path/costmap 专用监控转换；排障时应
使用支持 `nav_msgs/msg/Odometry`、`nav_msgs/msg/Path` 和
`nav_msgs/msg/OccupancyGrid` 的 ROS 工具。

## RViz 调试

[nav2.rviz](../runtime/nav2/rviz/nav2.rviz) 只读显示 FAST-LIVO2 registered
cloud、odom、TF、global path、rolling costmap 和 footprint，不包含
`SetGoal` 或 `InitialPose` 工具。

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=/opt/phanthy-motus/dds-local.xml

ros2 topic info /ubuntu/navigation/odom -v
ros2 topic info /ubuntu/navigation/cloud_registered -v
rviz2 -d actucore/plugins/controlled_semantic_spatial/runtime/nav2/rviz/nav2.rviz
```

## 构建与部署

JP5.11 的 Nav2 从锁定源码预编译在 ActuCore 基座中，日常镜像直接继承该基座并
编译仓库自有 ROS 包；不存在独立 Nav2 service 或运行时镜像。从仓库根目录执行
`./deploy/build_actucore.sh --mirror tuna` 生成 JP5.11 日常镜像。JP6.1 保留本地
VLA，但其新 navigation 默认关闭且尚未验证。
正式 `actucore/deploy/service.yml` 只有一个 service；现场容器编排不在卡片源码中维护。

Canvas 启动时 Nav2 只等待同容器 runtime 的 DDS 控制面，不等待 odom/cloud，避免
与 FAST-LIVO2 的 `start_mapping` action 形成生命周期环形等待。真正执行
`navigate_to_pose` 时仍会严格检查输入 freshness、frame、TF 和 lifecycle。
地图收口和加载的 backend 等待预算分别至少 360 s 与 900 s；可重试的
`stop_mapping` 会保留统一卡片 wiring 和 pending transaction，永久失败才释放
mapping 控制对象并继续整体停止。成功终态可对同名迟到请求幂等重放。统一
Runtime 对 launch 根进程及其独立 Linux 后代进程组执行有界信号阶梯回收，
不会因根进程提前退出而把算法子进程遗留在容器内。

## 验收边界

当前可以在不下发物理命令的情况下验证：合同、配置、MCP 注册、容器
启动、FAST-LIVO2 时间/frame readiness、路径/costmap 和 bounded proposal。

完整真机导航还必须同时满足：

1. 同卡片 FAST-LIVO2 模块已发布上述 canonical topic 和 `map -> base_link` TF；
2. planner 已为目标生成 `nav_id`，并有已验证的兼容下游消费对应格式；
3. 下游的执行、超时和物理停车机制经独立验收；
4. owner 持有遥控器/急停并显式授权真机运动。
