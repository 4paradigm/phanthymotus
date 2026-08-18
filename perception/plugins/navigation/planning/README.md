# Navigation 内部 Nav2 模块

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
    |-- /ubuntu/navigation/obstacle_map          sensor_msgs/PointCloud2
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
- Agent Core 负责 Canvas 生命周期和导航任务 lease。
- Driver `loco` 负责物理执行、二次限幅、急停、TTL 和停车确认。

## 输入合同

| port | topic | type / QoS | 必要条件 |
| --- | --- | --- | --- |
| `livo_odom` | `/ubuntu/navigation/odom` | `nav_msgs/msg/Odometry`; `BEST_EFFORT + KEEP_LAST(5)` | `header.frame_id=map`，`child_frame_id=base_link`，ROS system time，接收 age 最大 500 ms |
| `registered_cloud` | `/ubuntu/navigation/cloud_registered` | `sensor_msgs/msg/PointCloud2`; `BEST_EFFORT + KEEP_LAST(1)` | `header.frame_id=map`，已运动去畸变，ROS system time，接收 age 最大 500 ms |
| `obstacle_map` | `/ubuntu/navigation/obstacle_map` | `sensor_msgs/msg/PointCloud2`; `BEST_EFFORT + KEEP_LAST(1)` | `header.frame_id=map`，累计点云排除地板/天花板后投影到 `z=0` |
| `goal_pose`（可选） | `/ubuntu/navigation/goal_pose` | `std_msgs/msg/String`; `RELIABLE + KEEP_LAST(10)` | `phanthy.navigation.goal.v1`，`map` frame，每条唯一 `goal_id` |

Odometry 和 registered cloud 的 source stamp 必须同时可用。当 FAST-LIVO2 还在
直接发布 `camera_init` / `aft_mapped` frame 时，Nav2 会返回
`fast_livo2_odom_frame_invalid` 或 `registered_cloud_frame_invalid`，而不会启动导航。
接收 freshness 仍严格按 500 ms 判定；source stamp 额外允许 50 ms 有界调度抖动，
因此约 0.51 s 的处理边界帧不会被误拒，大于 0.55 s 仍 fail closed。

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
| `velocity_proposal` | `/ubuntu/navigation/nav2/velocity_proposal` | `std_msgs/msg/String`; `RELIABLE + KEEP_LAST(10)` | `phanthy.navigation.velocity_proposal.v1`，20 Hz，`base_link`，TTL 最大 250 ms |
| `plan` | `/plan` | `nav_msgs/msg/Path`; `RELIABLE + KEEP_LAST(1)` | Nav2 原生 `map` 全局路径，Canvas 显示起点、终点、路径长度和折线 |
| `costmap` | `/global_costmap/costmap` | `nav_msgs/msg/OccupancyGrid`; `RELIABLE + KEEP_LAST(1) + TRANSIENT_LOCAL` | Nav2 实时二维全局代价地图，作为卡片默认预览，叠加路径、位姿、终点和膨胀障碍 |

速度提案至少包含 `nav_id`、递增 `sequence`、`issued_at_unix_ms`、
`ttl_ms`、`nav_status` 和 `velocity{x,y,yaw}`。终态必须发布零速。
卡片始终为 proposal-only；`shadow_only=true` 不代表 Driver 一定执行。

速度限制：

- 导航请求 `speed` 范围为 `0.30–1.00 m/s`，默认 `0.50 m/s`；
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
  速度下限把尾段微调放大成终点徘徊。该判断无跨任务状态，不会锁住
  下一次导航。它只在 fresh canonical odom 和当前 target 都存在时生效；
  `goal_tolerance_reached` 只是零速 proposal reason，最终 `arrived` 仍以
  Nav2 action result 为准；
- BackUp 恢复动作固定为 `0.30 m/s`；
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
  独立错误，不再进入长时间恢复流程。成功回执包含 `goal_cell`。
- global/local costmap 都是 rolling window。global costmap 使用累计、去地面、
  去天花板并投影到二维的 `/ubuntu/navigation/obstacle_map`，避免已观察障碍
  因当前视角遮挡而消失；local costmap 继续使用实时
  `/ubuntu/navigation/cloud_registered`，高度带为 `-1.25…+0.30 m`。
- local 实时点云以 `base_link` 为 sensor origin 开启 raytrace clearing，
  不再把短时障碍轨迹永久留在局部窗口。global 输入本身是累计快照，
  继续 `clearing=false`，避免从错误的 map 原点向全图做射线清除。
  global 累计点的 range 也不再以 map 原点限制为 8 m，否则机器人走远后
  当前 rolling window 内的真实障碍会被误删。
- 两张 costmap 保留 `inflation_radius=0.55 m`。G1 矩形 footprint 的外接半径
  约 `0.425 m`，该配置实际额外余量约 `0.125 m`，不是这次过度占用的首要根因。
  heartbeat/status 的 `global_costmap` 字段现在分别给出 inflated、inscribed、
  lethal 数量及比例，用于独立评估累计障碍图是否过度占用，不通过
  缩小安全边界掩盖问题。
- Rotation Shim 在航向偏差大时先旋转；当前 DWB 仍只采样 `x/yaw`，
  proposal 出口把弧线速度离散成“只转”或“只走”，并对 X/Y/yaw
  三轴应用卡片配置的最小/最大绝对值。Y 默认上限为零。
- velocity smoother 使用 `/ubuntu/navigation/odom` 作为反馈。
- 任一 readiness blocker 会把非零 shadow velocity 改为带 reason 的零速提案。
- Nav2 bringup 的 `/cmd_vel` remap 限定在 scoped launch group 内；
  proposal bridge 始终检查真正的根 `/cmd_vel`，发现外部发布者仍会拒绝导航。

### 终点整形与并发保护

当前整形容差 `0.18 m / 0.45 rad` 严于 Nav2 GoalChecker 的
`0.20 m / 0.50 rad`。planner bridge 同时接收这组 GoalChecker 容差用于
启动校验；终点整形容差非正、非有限，或大于对应 GoalChecker
容差时直接拒绝启动。自定义 Nav2 params 时仍必须同步更新 bridge
中的 paired tolerance。

shadow velocity 回调在计算后、发布 proposal 前会在同一互斥区内
二次校验 `nav_id/attempt/status`。Nav2 action result 、pause/resume 或
新任务已使回调过期时，该回调不发布提案；因此终态零速不会被
旧回调用更高 `sequence` 覆盖。当前自动证据覆盖纯函数和源码
合同；终点不徘徊及真机时序仍需 G1 验收。

## 统一卡片连线

odom、registered cloud 和 obstacle map 由 `NavigationPlugin` 固定为同容器
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
Core 镜像因此需要包含 `ros-humble-nav-msgs`；单独重建 Perception/Nav2
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
rviz2 -d perception/plugins/navigation/runtime/g1_nav2/rviz/nav2.rviz
```

## 构建与部署

Nav2 ROS 包随统一 Perception 镜像构建，不存在独立 Nav2 service 或镜像。
G1 临时验证从仓库根目录执行：

```bash
bash perception/plugins/navigation/deploy/scripts/deploy-g1.sh
```

脚本只构建并启动一个 Perception 测试容器，不执行 Git 同步、Canvas action
或机器人动作。正式 `perception/deploy/service.yml` 同样只有一个 service。

Canvas 启动时 Nav2 只等待同容器 runtime 的 DDS 控制面，不等待 odom/cloud，避免
与 FAST-LIVO2 的 `start_mapping` action 形成生命周期环形等待。真正执行
`navigate_to_pose` 时仍会严格检查输入 freshness、frame、TF 和 lifecycle。

## 验收边界

当前可以在不下发物理命令的情况下验证：合同、配置、MCP 注册、容器
启动、FAST-LIVO2 时间/frame readiness、路径/costmap 和 bounded proposal。

完整真机导航还必须同时满足：

1. 同卡片 FAST-LIVO2 模块已发布上述 canonical topic 和 `map -> base_link` TF；
2. Agent Core 已把受信 `nav_id` 与 Driver lease 绑定；
3. Driver 返回提案执行和停车确认；
4. owner 持有遥控器/急停并显式授权真机运动。
