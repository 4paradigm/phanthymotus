# Nav2 Perception 卡片

`nav2` 是单实例 `processor` 卡片，现在只负责目标导航、路径规划、
局部避障和有时效的速度提案。建图、定位、点云去畸变、地图存储和
地图可视化由 FAST-LIVO2 卡片负责。Nav2 不订阅 Driver 原始点云，也不再
启动 SLAM Toolbox、AMCL 或 Map Server。

## 责任边界

```text
Driver sensors
    |
    v
FAST-LIVO2 card
    |-- /ubuntu/navigation/odom                 nav_msgs/Odometry
    |-- /ubuntu/navigation/cloud_registered     sensor_msgs/PointCloud2
    `-- TF: map -> base_link
              |
              v
          Nav2 card
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
| `livo_odom` | `/ubuntu/navigation/odom` | `nav_msgs/msg/Odometry`; `BEST_EFFORT + KEEP_LAST(5)` | `header.frame_id=map`，`child_frame_id=base_link`，ROS system time，最大 age 500 ms |
| `registered_cloud` | `/ubuntu/navigation/cloud_registered` | `sensor_msgs/msg/PointCloud2`; `BEST_EFFORT + KEEP_LAST(1)` | `header.frame_id=map`，已运动去畸变，ROS system time，最大 age 500 ms |
| `goal_pose`（可选） | `/ubuntu/navigation/goal_pose` | `std_msgs/msg/String`; `RELIABLE + KEEP_LAST(10)` | `phanthy.navigation.goal.v1`，`map` frame，每条唯一 `goal_id` |

Odometry 和 registered cloud 的 source stamp 必须同时可用。当 FAST-LIVO2 还在
直接发布 `camera_init` / `aft_mapped` frame 时，Nav2 会返回
`fast_livo2_odom_frame_invalid` 或 `registered_cloud_frame_invalid`，而不会启动导航。

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

## 输出合同

| port | topic | type / QoS | 语义 |
| --- | --- | --- | --- |
| `velocity_proposal` | `/ubuntu/navigation/nav2/velocity_proposal` | `std_msgs/msg/String`; `RELIABLE + KEEP_LAST(10)` | `phanthy.navigation.velocity_proposal.v1`，20 Hz，`base_link`，TTL 最大 250 ms |
| `plan` | `/plan` | `nav_msgs/msg/Path`; `RELIABLE + KEEP_LAST(1)` | Nav2 原生 `map` 全局路径，Canvas 显示起点、终点、路径长度和折线 |

速度提案至少包含 `nav_id`、递增 `sequence`、`issued_at_unix_ms`、
`ttl_ms`、`nav_status` 和 `velocity{x,y,yaw}`。终态必须发布零速。
卡片始终为 proposal-only；`shadow_only=true` 不代表 Driver 一定执行。

速度限制：

- 导航请求 `speed` 范围为 `0.10–1.00 m/s`，默认 `0.50 m/s`；
- 前进/后退提案的协议上限为 `1.00 m/s`，每次导航仍由请求的
  `speed` 再限幅；
- 禁止横移，`y=0`；
- 偏航角速度绝对值不超过 `0.35 rad/s`；
- BackUp 恢复动作仍固定为 `0.15 m/s`；
- Driver 仍负责二次限幅、TTL、急停和停车确认。

## Actions

| action | 参数 | 语义 |
| --- | --- | --- |
| `navigate_to_pose` | `x`、`y`、`yaw`、`speed?` | 在 `map` frame 中非阻塞创建导航任务 |
| `wait_navigation_done` | `stall_timeout?` | 等待到达、取消、超时或错误 |
| `pause_nav` | 无 | 暂停当前导航 |
| `resume_nav` | 无 | 恢复已暂停导航 |
| `stop_nav` | 无 | 取消任务并发布终态零速 |

`start_mapping`、`stop_mapping`、地图 CRUD、tag CRUD 和 `navigate_to_tag` 已从
Nav2 公共合同移除。这些能力如果需要，应由 FAST-LIVO2/位置服务单独暴露，
不应再把 SLAM 运行时塞回 Nav2 卡片。

为兼容重构前 Canvas 已保存的卡片实例，废弃字段
`runtime_switch_timeout_sec` 会在配置迁移时被忽略；Nav2 已不再切换
mapping/localization 运行模式。其他未知配置字段仍会拒绝。

## 规划与控制

- NavFn 生成全局路径。
- global/local costmap 都是 rolling window，直接使用
  `/ubuntu/navigation/cloud_registered` 的 `PointCloud2` 障碍物。
- Rotation Shim 在航向偏差大时先旋转，DWB 只采样 `x/yaw`，不采样横移。
- velocity smoother 使用 `/ubuntu/navigation/odom` 作为反馈。
- 任一 readiness blocker 会把非零 shadow velocity 改为带 reason 的零速提案。
- Nav2 bringup 的 `/cmd_vel` remap 限定在 scoped launch group 内；
  proposal bridge 始终检查真正的根 `/cmd_vel`，发现外部发布者仍会拒绝导航。

## Canvas 连线

Nav2 卡片需要两条必需输入：

1. FAST-LIVO2 `livo_odom` -> Nav2 `livo_odom`；
2. FAST-LIVO2 `registered_cloud` -> Nav2 `registered_cloud`；
3. Nav2 `velocity_proposal` -> Driver `loco.velocity_proposal`。

地图从 FAST-LIVO2 卡片的 `map_view` 查看，实时位姿从其
`livo_odom` 查看。Nav2 卡片输出 `/plan` 的 2D 路径视图，但不复制
`map_view`，避免两张“地图”在 Canvas 中同时成为权威源。

Canvas 的 odom/path 监控依赖 Agent Core 按原生 ROS 2 类型订阅
`nav_msgs/msg/Odometry` 和 `nav_msgs/msg/Path`；不得在同名 topic 上创建
`std_msgs/msg/String` 订阅，否则 DDS graph 会出现同 topic 多类型且数据流为空。
Core 镜像因此需要包含 `ros-humble-nav-msgs`；单独重建 Perception/Nav2
镜像不会更新 Canvas 的解码与 renderer。

## RViz 调试

[nav2.rviz](companion/g1_nav2/rviz/nav2.rviz) 只读显示 FAST-LIVO2 registered
cloud、odom、TF、global path、rolling costmap 和 footprint，不包含
`SetGoal` 或 `InitialPose` 工具。

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4

ros2 topic info /ubuntu/navigation/odom -v
ros2 topic info /ubuntu/navigation/cloud_registered -v
rviz2 -d perception/plugins/nav2/companion/g1_nav2/rviz/nav2.rviz
```

## 构建与部署

FAST-LIVO2 和 Nav2 companion 都已接入 `perception/deploy/service.yml`，与
Perception 一起启动。Nav2 容器无地图 volume、无 Driver SDK、无 Docker
socket，且以 read-only 模式运行。

G1 临时验证从仓库根目录执行一条命令；脚本按当前 commit 生成三个镜像标签，
依次构建 Perception、FAST-LIVO2 companion 和 Nav2 companion，再运行现有
`preflight` 与 `start`：

```bash
bash perception/plugins/nav2/deploy/scripts/build-and-start-g1.sh
```

运行前应停掉 Canvas 项目，并确保锁定的本地基础镜像
`phanthy-fast-livo2:g1-1fcd0d0-n3save1` 存在。脚本不执行 Git 同步、不删除旧
容器、不启动 Canvas，也不会发布建图或导航命令；已有测试容器需要先用
现有 `STAGE=stop` 入口移除。Core 启用认证时，运行前显式提供
`CORE_ACCESS_TOKEN`。`perception/deploy/service.yml` 是正式运行编排，不是
源码构建入口。

Canvas 启动时 Nav2 只等待 companion 的 DDS 控制面，不等待 odom/cloud，避免
与 FAST-LIVO2 的 `start_mapping` action 形成生命周期环形等待。真正执行
`navigate_to_pose` 时仍会严格检查输入 freshness、frame、TF 和 lifecycle。

## 验收边界

当前可以在不下发物理命令的情况下验证：合同、配置、MCP 注册、容器
启动、FAST-LIVO2 时间/frame readiness、路径/costmap 和 bounded proposal。

完整真机导航还必须同时满足：

1. FAST-LIVO2 卡片已发布上述 canonical topic 和 `map -> base_link` TF；
2. Agent Core 已把受信 `nav_id` 与 Driver lease 绑定；
3. Driver 返回提案执行和停车确认；
4. owner 持有遥控器/急停并显式授权真机运动。
