# Navigation 内部 Nav2 模块

该模块在统一 `ControlledSemanticSpatial` 卡片内负责目标导航、路径规划、
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
              `-- /ubuntu/navigation/nav2/velocity_proposal
                                      |
                                      v
                              Driver loco actuator
```

- FAST-LIVO2 负责把原始 `camera_init -> aft_mapped` 输出归一化为下表的
  `map/base_link` 合同。不能只改 topic 名来伪装坐标系。
- Nav2 只消费已去畸变、已对时、已归一化 frame 的数据，并在任一
  输入过期、frame 不符或 `map -> base_link` 不可用时 fail closed。
- Agent Core 负责 Canvas 生命周期和 MCP 请求转发；Nav2 planner
  在接受每个新目标时生成独立 `nav_id`。
- Driver `loco` 负责物理执行、二次限幅、急停、TTL 和停车确认。

## 输入合同

| port | topic | type / QoS | 必要条件 |
| --- | --- | --- | --- |
| `livo_odom` | `/ubuntu/navigation/odom` | `nav_msgs/msg/Odometry`; `BEST_EFFORT + KEEP_LAST(5)` | `header.frame_id=map`，`child_frame_id=base_link`，ROS system time，接收 age 最大 500 ms |
| `registered_cloud` | `/ubuntu/navigation/cloud_registered` | `sensor_msgs/msg/PointCloud2`; `BEST_EFFORT + KEEP_LAST(1)` | `header.frame_id=map`，已运动去畸变，ROS system time，接收 age 最大 500 ms |
| `static_map` | `/ubuntu/navigation/static_map` | `nav_msgs/msg/OccupancyGrid`; `RELIABLE + KEEP_LAST(1) + TRANSIENT_LOCAL` | `header.frame_id=map`，直接累计静态图，`-1/0/100` |
| `goal_pose`（可选） | `/ubuntu/navigation/goal_pose` | `std_msgs/msg/String`; `RELIABLE + KEEP_LAST(10)` | `phanthy.navigation.goal.v1`，`map` frame，每条唯一 `goal_id` |

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
registered cloud 的接收 age 推过 500 ms。BT 主循环为 20 Hz，仍快于 5 Hz
局部控制输出，但不会在 Jetson 上用 100 Hz 空转与定位、点云序列化争抢调度。

`goal_pose` 最小样例：

```json
{
  "schema": "phanthy.navigation.goal.v1",
  "goal_id": "room-a-001",
  "x": 1.2,
  "y": -0.8,
  "yaw": 0.0,
  "speed": 0.5
}
```

Agent Core 仅在 Canvas 项目处于运行状态、且上游 topic 实际连到
`goal_pose` 端口时激活 `x-topic-actions`。Core 会把通过校验的
`x/y/yaw/speed` 转为同一张卡片的 `navigate_to_pose` MCP 调用；
`schema` 不符、缺坐标、夹带未声明字段或重复 `goal_id` 的消息
都不会被执行。停止 Canvas 项目会先退订该 topic，再停止卡片。

## 输出合同

| port | topic | type / QoS | 语义 |
| --- | --- | --- | --- |
| `velocity_proposal` | `/ubuntu/navigation/nav2/velocity_proposal` | `std_msgs/msg/String`; `RELIABLE + KEEP_LAST(1)` | `phanthy.navigation.velocity_proposal.v1`，导航活动期间固定 5 Hz、只保留最新值，`base_link`，TTL 最大 250 ms |
| `plan` | `/plan` | `nav_msgs/msg/Path`; `RELIABLE + KEEP_LAST(1)` | Nav2 原生 `map` 全局路径，Canvas 显示起点、终点、路径长度和折线 |
| `costmap` | `/global_costmap/costmap` | `nav_msgs/msg/OccupancyGrid`; `RELIABLE + KEEP_LAST(1) + TRANSIENT_LOCAL` | Nav2 实时二维全局代价地图，作为卡片默认预览，叠加路径、位姿、终点和膨胀障碍 |

速度提案至少包含 `nav_id`、递增 `sequence`、`issued_at_unix_ms`、
`ttl_ms`、`nav_status` 和 `velocity{x,y,yaw}`。终态立即发布一次零速，随后停止该
`nav_id` 的周期 proposal；终态回执仍保留供查询和幂等重放。
卡片始终为 proposal-only；`shadow_only=true` 不代表 Driver 一定执行。

速度限制：

- 导航请求 `speed` 范围为 `0.30–1.00 m/s`，默认 `0.50 m/s`；
- X 轴为仅前进合同：Nav2 controller、velocity smoother 或恢复树产生的
  负 X 均在 proposal 边界强制归零，不会向 Driver 发布倒退命令；
- `config` 动作可配置 `min_x_mps/max_x_mps`、`min_y_mps/max_y_mps`
  和 `min_yaw_rps/max_yaw_rps`；这些值都是非零速度的绝对值，
  Nav2 原始方向符号保持不变；
- 默认 `X=0.30–1.00 m/s`、`Y=0–0 m/s`、`yaw=1.00–2.00 rad/s`；
  `max_y_mps=0` 保持当前禁止横移的行为；
- 每次导航的 `speed` 继续作为正向 X 上限，它比卡片配置的
  X 最小值优先，不会被速度下限反向放大；
- 平移与转向强制互斥：混合提案的原始 `|yaw| >= 0.20 rad/s` 时只原地
  转向（`x=y=0`），低于该阈值时只平移（`yaw=0`），不会向
  Driver 发布同时包含平移和转向的动作；
- 上述最小值只处理非零运动提案；readiness blocker、暂停和终态零速
  仍保持严格零值；
- 终点内层容差为 `0.18 m / 0.45 rad`：进入位置容差后不再发布平移，
  只保留 Nav2 的朝向校正；位置和朝向都进入容差后发布严格零速，避免
  速度下限把尾段微调放大成终点徘徊。位置容差首次满足后，当前
  `nav_id` 的终点阶段保持为“只转向”，不因定位边界抖动重新启用平移；
  新导航创建独立阶段，不会锁住下一次导航。该判断只在 fresh canonical
  odom 和当前 target 都存在时生效；
  `goal_tolerance_reached` 只是零速 proposal reason，最终 `arrived` 仍以
  Nav2 action result 为准；
- Nav2 Humble 固定创建 pose 与 through-poses 两个内部 navigator；卡片为二者
  显式加载无 BackUp 的恢复树，只使用清除代价地图、原地转向和等待。公开动作
  仍只有 `navigate_to_pose`；
- Driver 仍负责二次限幅、TTL、急停和停车确认。

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
- 两层都使用卡片的 `obstacle_min_height_m/obstacle_max_height_m` 预过滤；旧
  `/ubuntu/navigation/obstacle_map` 仅作兼容诊断，不再以 `clearing=false`
  驱动全局代价图。confirmed static PCD 加载时还要求保存的高度带与当前配置
  完全一致，不一致则拒绝加载并要求恢复配置或重新建图。
- 两张 costmap 保留 `inflation_radius=0.55 m`。G1 矩形 footprint 的外接半径
  约 `0.425 m`，该配置实际额外余量约 `0.125 m`，不是这次过度占用的首要根因。
  heartbeat/status 的 `global_costmap` 字段现在分别给出 inflated、inscribed、
  lethal 数量及比例，用于独立评估静态图与实时层是否过度占用，不通过
  缩小安全边界掩盖问题。
- Rotation Shim 在航向偏差大时先旋转；当前 DWB 仍只采样 `x/yaw`，
  proposal 出口把弧线速度离散成“只转”或“只走”，并对 X/Y/yaw
  三轴应用卡片配置的最小/最大绝对值。Y 默认上限为零。
- velocity smoother 使用 `/ubuntu/navigation/odom` 作为反馈。
- 任一 readiness blocker 会把非零 shadow velocity 改为带 reason 的零速提案。
- 局部 controller、velocity smoother 与 proposal bridge 统一为 5 Hz，匹配
  G1 实际执行能力；costmap 和传感器更新仍保持独立高频。安全 blocker、协议
  错误和任务终态的首个零速不等待下一个周期。proposal topic 使用
  `KEEP_LAST(1)`，避免 Driver 排队执行过期轨迹。
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

### 终点整形与并发保护

当前整形容差 `0.18 m / 0.45 rad` 严于 Nav2 GoalChecker 的
`0.20 m / 0.50 rad`。planner bridge 同时接收这组 GoalChecker 容差用于
启动校验；终点整形容差非正、非有限，或大于对应 GoalChecker
容差时直接拒绝启动。自定义 Nav2 params 时仍必须同步更新 bridge
中的 paired tolerance。

shadow velocity 回调在计算后、写入 latest-only 缓存前会在同一互斥区内
二次校验 `nav_id/attempt/status`；5 Hz 发布定时器在真正发布前再次校验任务
上下文和 readiness。Nav2 action result、pause/resume 或新任务已使样本过期时，
该样本不发布；因此终态零速不会被旧回调用更高 `sequence` 覆盖。当前自动证据覆盖纯函数和源码
合同；终点不徘徊及真机时序仍需 G1 验收。

## 统一卡片连线

odom、registered cloud 和 static map 由 `NavigationPlugin` 固定为同容器
内部 topic，不需要 Canvas 连线。Canvas 只需把公开 `velocity_proposal`
输出接到 Driver `loco.velocity_proposal`；可选 `goal_pose` 仍可作为外部输入。

地图和实时位姿从统一卡片的 `map_view` 查看；`livo_odom` 只作为同容器内部
定位/规划数据，不再生成 Canvas 公共端口。Agent Core 的地图 renderer 会额外
只读订阅 Nav2 `/plan`，
在同一个 `map` frame 中把绿色全局路径和橙色终点叠加到地图上；Nav2 仍不
复制 `map_view`，因此 Canvas 只有一张权威地图。`/plan` 独立数据流仍保留，
用于查看路径点数、长度和纯折线。地图右上角可切换 `2D/3D`：2D 是对
FAST-LIVO2 三维点云的正上方平面投影，与 Nav2 的二维规划坐标直接对齐，
不是新增的 occupancy grid。

统一卡片的 costmap“查看数据流”打开二维导航视图：底图是规划器
实际使用的 `/global_costmap/costmap`，红色是占据障碍，橙色是膨胀代价，
绿色是当前 `/plan`，绿色箭头是 `map -> base_link` 位姿，橙色圆点是目标。
该视图用来直接判断“目标在代价地图外”、“起点或终点落在膨胀区”和
“障碍将可通行区切断”等无有效路径原因。

Canvas 的 odom/path/costmap 监控依赖 Agent Core 按原生 ROS 2 类型订阅
`nav_msgs/msg/Odometry`、`nav_msgs/msg/Path` 和 `nav_msgs/msg/OccupancyGrid`；不得在同名 topic 上创建
`std_msgs/msg/String` 订阅，否则 DDS graph 会出现同 topic 多类型且数据流为空。
Core 镜像因此需要包含 `ros-humble-nav-msgs`；单独重建 ActuCore/Nav2
镜像不会更新 Canvas 的解码与 renderer。

## RViz 调试

[nav2.rviz](../runtime/g1_nav2/rviz/nav2.rviz) 只读显示 FAST-LIVO2 registered
cloud、odom、TF、global path、rolling costmap 和 footprint，不包含
`SetGoal` 或 `InitialPose` 工具。

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4

ros2 topic info /ubuntu/navigation/odom -v
ros2 topic info /ubuntu/navigation/cloud_registered -v
rviz2 -d actucore/plugins/navigation/runtime/g1_nav2/rviz/nav2.rviz
```

## 构建与部署

Nav2 ROS 包随统一 ActuCore 镜像从源码构建，不存在独立 Nav2 service 或镜像。
G1 临时验证从仓库根目录执行：

```bash
bash actucore/plugins/navigation/deploy/scripts/deploy-g1.sh
```

脚本只构建并启动一个 ActuCore 测试容器，不执行 Git 同步、Canvas action
或机器人动作。正式 `actucore/deploy/service.yml` 同样只有一个 service。

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
2. planner 已为目标生成 `nav_id`，Driver 在空闲时接纳该任务
   的首条新鲜、合法、非零 proposal；
3. Driver 返回提案执行和停车确认；
4. owner 持有遥控器/急停并显式授权真机运动。
