# ControlledSemanticSpatial ActuCore 卡片

`ControlledSemanticSpatial` 是 FAST-LIVO2、Nav2 和 VLN 的统一公开卡片，属于
**ActuCore（执行模型层）**：把意图/目标变成运动指令。Canvas 只看到这一张
`processor` 卡片，正式部署只运行一个 ActuCore 容器。

卡片的公开 action/topic 契约和单容器架构不绑定具体机器人。Driver 必须把
LiDAR/IMU 转为同一个 REP-103 `sensor_frame`，提供
`base_link -> sensor_frame` 静态 TF，并使用同一 ROS system time；当前真机证据
覆盖 G1，其他本体仍需分别验收。

## 运行边界

```text
Driver lidar + imu ──────────> ControlledSemanticSpatial card (ActuCore container)
optional RGB + depth frame ──┴─> semantic navigation / data collection
                            ├─ FAST-LIVO2 mapping/localization child process
                            ├─ Nav2 planner/controller child process
                            └─ semantic waypoint processor
                                      |
                                      `─ velocity_proposal -> Driver loco
```

- `NavigationPlugin` 是唯一 MCP/Canvas 生命周期所有者。
- FAST-LIVO2 和 Nav2 ROS launch 由 `NavigationRuntime` 作为同容器子进程组
  启停；运行时不调用 Docker，也不需要 Docker socket。
- odom、registered cloud 和 obstacle map 是卡片内部 ROS 边；
  `collection_status` 作为只读公共图像输出，供 Canvas 查看最新同步 RGB、
  当前采集帧号和 LiDAR 障碍物距离标注；停止后同一端口切换为离线导出
  进度与失败原因。机器诊断保留在内部 JSON topic。
- VLN 命中地点后直接调用同卡片 planner；无论是 Canvas 还是 VLN 入口，
  planner 都为每个新任务生成独立
  `nav_id`。到达或手动停止时先发布一次终态零速 proposal，再确认
  终态并允许下一个任务；已结束任务不会继续以 5 Hz 刷新，终态结果
  本身仍可查询和幂等重放。
- Nav2 仍只发布 `phanthy.navigation.velocity_proposal.v1`，Driver 继续负责
  物理执行、TTL、急停、二次限幅和停车确认。
- Nav2 保留 NavFn、rolling costmap 和重规划，局部执行改为 G1 分段
  `停稳 -> 原地转向 -> 直行 -> 停稳复查`。控制和位姿检查为
  20 Hz，非零 proposal 仍为 5 Hz，零速立即发布。
- 分段执行的转向和直行预检只在 footprint 命中 Nav2
  `LETHAL_OBSTACLE` 时拒绝；膨胀安全带仅用于代价与规划，不冒充实体碰撞。

## 外部输入

| port | topic | 必需 |
| --- | --- | --- |
| `lidar` | `/ubuntu/navigation/lidar` | 是 |
| `imu` | `/ubuntu/navigation/imu` | 是 |
| `rgb` | `/ubuntu/camera/rgb_frame` | 否；连接时启用语义导航，`collection_enabled=true` 时必需 |
| `depth_frame` | `/ubuntu/camera/depth_frame` | 平时否；`collection_enabled=true` 时必须连接，沿用 Driver `PSE1` 封装中的深度尺度、标定与源时间戳 |
| `goal_pose` | `/ubuntu/navigation/goal_pose` | 否；连线后由 Agent Core `x-topic-actions` 转换为 `navigate_to_pose`，不生成独立监控卡片 |

这是完整的外部输入集合；各端口的 `required` 都显式声明，正常启动只要
`lidar` 和 `imu`。`goal_pose` 是可选的外部 topic action，不是启动时内部连线。
Agent Core 会把消息里的 `goal_id` 作为私有任务 ID 传给 planner；同一消息在
投递结果未知后重试时会幂等重放活动或终态回执，不会创建第二个导航任务。
结果未知时 planner 保留活动 ID 而不再次下发；收到 runtime 的明确拒绝回执时
则释放 ID，允许修正后重试。同一 ID 携带不同目标会被拒绝。活动及终态幂等
回执最多保留 1024 个可信 ID，避免长时运行
无界增长。Nav2 goal response 超时只表示结果未知，runtime 保持 pending 并继续
接收迟到的 accepted/rejected 回调，不会先发布伪终态再让晚到 goal 独立运行。

LiDAR 每点 `timestamp` 必须是 `float64` 绝对纳秒，时间单调且一帧跨度位于
`(0, 200] ms`。FAST-LIVO2 mapper 从实际进入其 callback 和处理循环的数据中，
以 1 Hz 报告 LiDAR/IMU frame、源时间戳和实际扫描跨度；adapter 不再重复订阅
原始高频 LiDAR/IMU，也不再复制整帧点云做旁路校验。adapter 只消费 mapper
运行统计和 FAST-LIVO2 raw odom/cloud，负责 TF、坐标归一化与地图生成。状态会
显示 `sensor_frame`、TF、点时间跨度和 `odom_health`；
几何契约不满足时返回 `sensor_frame_mismatch`、`sensor_tf_unavailable` 或
`raw_odom_discontinuity`，不会继续生成伪正常地图；几何跳变会在当前
地图会话内锁存，需要重置或切换地图会话才能恢复。
每次 ActuCore 运行会话使用 rosbag2 snapshot 在内存中保留最近约 5 秒
的全频 LiDAR、IMU、FAST-LIVO2 raw odom 和 IMU 传播 odom；首次
`raw_odom_discontinuity` 后再采 5 秒，仅落盘一份 MCAP 到
`/opt/phanthy-motus/data/fast_livo2/recordings/faults/`。`status.fault_capture`
显示 `armed/post_trigger/saved/error`、触发原因、产物目录和
`diagnostic_summary.json`；摘要会对齐 Driver 发布、FAST-LIVO2 实际
callback/处理和 adapter 输出计数，因此旧录包中的低帧率不再被直接误判为
雷达源降频。没有故障时
停卡会丢弃空快照，不会将常态全频数据写入磁盘。录包进程不创建独立
进程组，会随 FAST-LIVO2 runtime 的 stop/restart 一起回收，避免多个全频
recorder 重复订阅传感器。FAST-LIVO2 输入使用 Reliable：LiDAR 默认只保留
最新 2 帧，IMU 默认保留 400 帧（按 200 Hz 约 2 秒）的积分历史，两个深度
均可在卡片配置中调整。上游 NodeOptions 会自动声明命令行覆盖，
因此运行补丁只读取这些值，不再二次 `declare_parameter`；基础镜像
构建会使用同样的参数执行一次真实启动检查。计算瞬时
积压时丢弃旧点云，但不丢掉两帧 LiDAR 之间不可替代的 IMU 运动量；
故障录包仍使用 Best Effort snapshot，不反压传感器链路。
mapper 运行统计超过 3 秒未更新时，状态会把 `mapper_runtime_stale` 放入
`sensor_contract_issues`；已确认的静态 TF 不会因此丢失。只要 FAST-LIVO2
自己的 odom/cloud 仍新鲜且 frame、TF 与 odom 连续性有效，就不会重复阻断输出。

## 公共输出

| port | topic | 用途 |
| --- | --- | --- |
| `map_view` | `/ubuntu/navigation/fast_livo2/map_view` | Canvas 地图与机器人位姿 |
| `status` | `/ubuntu/navigation/fast_livo2/status` | 定位、建图和运行状态 |
| `collection_status` | `/ubuntu/navigation/fast_livo2/collection_preview` | 采集中显示 RGB/帧号/距离，停止后显示导出进度 |
| `velocity_proposal` | `/ubuntu/navigation/nav2/velocity_proposal` | 连接 Driver `loco` 执行器 |
| `costmap` | `/global_costmap/costmap` | 实时全局代价地图 |

`livo_odom`、registered cloud、confirmed static map 和 obstacle map 仍只由
同容器内的定位、规划和语义逻辑消费，不生成 Canvas 右侧连线端口。
`/plan` 和 `/ubuntu/navigation/odom` 注册为隐藏辅助流，只供 `map_view` 与
`costmap` 叠加路径和位姿，不生成独立监控卡片。详细
frame、QoS、freshness、数采和速度约束见内部实现说明：

- [mapping/README.md](mapping/README.md)
- [planning/README.md](planning/README.md)
- [semantic/README.md](semantic/README.md)

RGB 和 depth 均为可选端口：只连 LiDAR + IMU 即可启动建图、定位与
Nav2 规划控制；连接 RGB 后才启动语义导航。Canvas 的静态卡片规格无法随
`collection_enabled` 动态改变端口外观，因此统一卡片在启动时执行条件
校验：启用数采时 RGB 和 depth 都必须连接，否则返回明确的
`invalid_canvas_wiring`，而不是启动一份不完整的数据集。

完整的静态插件配置样例见 [config.example.json](config.example.json)；其中
`semantic.vlm.api_key` 必须通过部署配置或环境注入真实值，不能提交凭据。
Canvas 的 `config` 动作还提供 `obstacle_min_height_m` 和
`obstacle_max_height_m`。它们控制实时与稳定静态二维障碍的 `map` frame
高度带，必须在卡片停止时修改；`map_view` 会把最新扫描的范围外点用
蓝色/粉色标记，方便现场根据地面和天花板分布调参。范围外表面以 `0.20 m`
体素单独累计或从已加载 raw PCD 恢复，只用于 Canvas；它们不会进入
registered cloud、静态障碍图或 Nav2 costmap。高度带默认值是
`-0.30...+0.30 m`；它采用 canonical `map` frame，不能按雷达物理安装高度
直接填写。直接累计不会清除已经由错误高度带写入的静态占用，修改上下界后
必须新建地图。
`map_view_enabled` 和 `fault_capture_enabled` 默认均为 `true`，用于真机
性能 A/B；只能停卡后配置，并在下次启动生效。关闭前者只停止 Canvas 点云
编码/发布，不改变 FAST-LIVO2 或 Nav2 输入；关闭后者只停止短时故障快照。

Canvas 的转向速度只暴露单一 `rotate_speed_rps`，默认
`0.3 rad/s`。非零 yaw proposal 保留方向并固定为该幅值；不再让
操作者分别配置易混淆的 yaw 最小值和最大值。该配置只能在
卡片停止时修改，下次启动生效。

建图恢复为直接累计模式：FAST-LIVO2 adapter 对每帧注册点云按 `0.10 m`
体素去重后，将导航高度带内且距离有效的体素首次出现即写入静态图，不等待
多帧确认、不跟踪动态分量，也不通过后续自由射线删除已累计点。这样可保留
墙面和稀疏结构的完整证据，但建图期间出现的人或其他移动物体也可能固化进
地图，需由操作者在相对静态的环境中完成建图。

实时避障与累计地图彼此独立：当前 registered cloud 继续进入 Nav2 开启
marking/raytrace clearing 的 live ObstacleLayer，所以导航期间新出现或移动的
物体仍参与即时避障。锁定版本的 Nav2 `ObservationBuffer::bufferCloud` 分别用
`sensor_frame` 计算射线原点、用 PointCloud2 `header.frame_id` 变换点坐标；因此
配置以动态 `base_link` 作为机身射线原点，而 registered cloud 的点仍按 `map`
坐标解释。原始 FAST-LIVO2 odom/cloud 使用 latest-only 订阅；cloud
只有在其源时间戳已被前后 odom/TF 包围且最近位姿差不超过 50 ms 时才对
Nav2 发布，避免 adapter 排队旧帧或让点云早于 TF。canonical registered cloud
发布是独立快路径：PointCloud2 的 XYZ 解码、`map` 坐标变换、高度过滤和
float32 打包均使用 NumPy 批量运算，不再逐点进入 Python 循环。静态地图累计与
Canvas 编码在 latest-only 后台任务中执行，
不会占用导航点云回调。后台来不及处理时覆盖旧建图样本；并发回调晚完成的
旧扫描也按每次建图会话的接收时间丢弃，两者都在诊断中累计
`mapping_work_dropped`，但不阻塞最新导航点云。Canvas `map_view` 显示累计静态点和新鲜的最新实时扫描，
并显示单独累计的高度带外表面。输出最多 80,000 点，并为低于、位于和高于
导航高度带的三组点分别保留显示预算，避免地面被大量障碍点截断；该采样只
影响监控，不改变规划输入。显示帧直接编码已经分别有界的静态、范围外和实时
点源，不再构造一次性全图体素副本，避免 Canvas 更新拖慢 Nav2 所依赖的 odom
与 registered cloud。点云主体按 1 Hz 重编码并缓存，编码完成后在同一次
回调中立即替换前 12 字节的机器人 `x/y/yaw` 并发布；DDS 只保留最新一帧，
因此慢显示不会堆积并挤占 odom/registered cloud 回调。
FAST-LIVO2 diagnostics 的 `latency_ms` / `latency_max_ms` 分别报告最近值和进程内
最大值，包含 `cloud_decode`、`cloud_pose_wait`、`cloud_transform_filter`、
`cloud_pack_publish`、`cloud_end_to_end`、`map_view_encode` 和
`map_view_pose_publish`；`map_view_cache_age_sec` 用于区分点云主体陈旧和位姿刷新
本身的开销。已保存的 confirmed static PCD 还会绑定建图时的障碍高度带；加载
时若当前上下界不同会拒绝使用，需恢复原配置或重新建图，避免静态证据语义
悄然变化。

### 断流与算力诊断

`status.pipeline_diagnostics` 以 60 秒窗口关联三层已有证据：Driver 隐藏
`_bridge_status`、FAST-LIVO2 mapper 1 Hz 运行统计和 adapter diagnostics。
输入丢帧只比较 Driver 发布与 mapper 实际 callback，adapter 仅代表 mapper
输出后的坐标归一化阶段。输出每层计数增量/频率、跨层比率和以下归因之一：
`driver_source_drop`、`dds_or_subscriber_drop`、
`fast_livo_processing_backlog`、`scan_match_degraded`、
`adapter_backlog`、`insufficient_evidence` 或 `healthy`。只有 Driver 自己的
接收/发布频率或 drop counter 异常时才认定源端降频。

ROS 2 topic 不做“定时清理”：DDS `KEEP_LAST` 会自动淘汰旧样本，清 daemon
也不会清订阅者队列。需要观察的是各订阅 callback 的生产/消费差、
FAST-LIVO2 内部 LiDAR/IMU buffer span、adapter latest-only 丢弃数和 rosbag
录制覆盖率；对应字段已经进入上述状态和故障摘要。

真机每个阶段运行 300 秒，由现场的只读诊断工具采集容器 CPU/内存、
Jetson `tegrastats`、进程 CPU 和完整 pipeline 状态。依次测量静止定位、
同一路线导航，并在停卡后分别配置：两项均关闭、只开
`map_view_enabled`、只开 `fault_capture_enabled`、两项均开启。每阶段记录
CPU 平均/P95、Driver LiDAR 发布率、mapper callback/处理率、最大帧间隔、
adapter 丢弃和 odom discontinuity 次数。先用跨层计数确认丢帧发生在哪一层，
再决定优化 Driver、DDS、FAST-LIVO2 或显示/录制；不能用“清 topic”掩盖
持续生产过快或消费过慢。

confirmed static map 以有点数上限的稀疏体素保存，不因覆盖范围变大而拒绝
建图。面向 Nav2 StaticLayer 的稠密 OccupancyGrid 则始终发布机器人周围
`static_grid_margin_m` 半径的滚动窗口；地图跨度增大或全局坐标远离原点时，
不会按整张地图包围盒分配巨型数组，也不会使 frame adapter 退出。

资源和地图事务均 fail closed：单帧 live PointCloud2 最多 200,000 点且数据区
最多 64 MiB；累计静态图最多 200,000 点，超限
时不静默抽样。PCD header 和 ASCII 单条记录均限制为 64 KiB，ASCII token
布局及实际非空数据行数必须与声明完全一致，解析还受 map-control deadline
约束。manifest 最大 64 KiB。一次
地图会话最多 64 个 raw PCD，raw 快照与 confirmed static PCD 合计最多
512 MiB。`load_map` 会先完成新旧 manifest/PCD、障碍高度带、点数限制及
滚动 OccupancyGrid 窗口验证，再停止旧定位前端；Adapter 先在旧状态之外准备图，
deadline 内只做原子状态切换，并在控制回执之后发布大栅格。

## Actions

- 生命周期：`info`、`config`、`start`、`stop`
- 建图/定位：`start_mapping`、`stop_mapping`、`load_map`、`relocalize`
- 规划控制：`navigate_to_pose`、`wait_navigation_done`、`pause_nav`、
  `resume_nav`、`stop_nav`
- 语义地点：`capture`、`navigate`

`navigate_to_pose` 保持非阻塞；需要在同一调用链内等待结果时仍使用
`wait_navigation_done`。Nav2 异步上报同一 `nav_id` 的到达、取消、停止、
超时或失败终态时，卡片会立即释放活动任务，下一个导航无需再手工
`stop_nav` 解锁；终态后迟到的 `wait_navigation_done` 会幂等返回已保存的
终态回执。不同 `nav_id` 的迟到消息不会解锁当前任务。

### ACP 完成事件

`navigate_to_pose` 与语义 `navigate` 都在 `inputSchema.x-completion` 中声明为
长时动作，接受目标后返回相同值的 `nav_id`/`action_id`。Nav2 上报终态时，ActuCore 通过 SSE 发布
匹配的 `action_complete`，Agent Core 因而不会把“已开始导航”误当成“已到达”。
ACP barrier 按原始工具名隔离，不阻塞 TTS 等其他卡片；本卡片的
`wait_navigation_done`、`pause_nav`、`resume_nav`、`stop_nav` 是明确的控制旁路，
系统打断则通过 `on_interrupt_navigation` 调用 `stop_nav`。

Canvas 手动执行 `navigate_to_pose` 时，Agent Core 只透明转发 MCP
请求。planner 接受目标后生成新 `nav_id`，并在整个任务的
velocity proposal 中保持该 ID。Nav2 上报匹配的终态后，卡片释放
活动任务；下一次点击因此会获得另一个 ID，无需重启
Canvas、Core 或 Driver。Driver 在空闲订阅状态接纳首条新鲜、合法、
非零且非终态 proposal 的 `nav_id`，活动任务期间拒绝 ID 切换，并在
终态零速后退役该 ID。Core 不另外维护 authorize/revoke 状态。

`start` 按 runtime → mapping → planning → semantic 顺序获取资源；任一步
失败会按相反顺序回滚。`stop_mapping` 和 `load_map` 的 backend 等待预算分别
至少为 360 s 和 900 s。未确认的地图收口失败会保留 Canvas wiring、运行时和
`finalizing` 事务，下一次 `stop`/`stop_mapping` 从原事务继续；只有明确终态
才会释放 mapping 控制对象并继续回收其他模块和运行时。已完成的同名
`stop_mapping` 终态可幂等重放原保存回执，避免迟到重试制造第二份成功结果。
数采目录收口遇到暂时性 I/O 错误时保留原 `.partial` 路径供下一次 stop 重试；
路径已丢失、目标目录冲突等不可恢复错误会保留路径证据并返回明确终态，避免
Canvas 永久卡在 retryable stop。
卡片级 `start`/`stop`/`config` 转换串行执行，不允许 Canvas 的迟到请求在
前一次启动中途关闭 backend。若 Nav2 command bridge 子进程仍存活，但
Fast DDS 在首个发现窗口内暂未报告 command subscriber，只重建一次
planning bridge 并重试发现，不重启 FAST-LIVO2 或 Nav2 子进程；第二次仍失败
才执行完整回滚。
`config` 在修改任一子组件前先校验 mapping、planning 和 semantic 的全部
候选参数；若子组件在应用阶段仍意外拒绝，已应用组件会恢复到请求前
配置，不会在返回 `invalid_config` 的同时留下半更新状态。

`stop` 先确认活动 Nav2 任务已经停下；未确认时保留所有子进程并返回可重试
错误；每个组件必须显式返回 `idle` / `stopped` / `disabled` 才算确认。
确认后才继续停止 mapping 和两个 launch 子进程组，并保留各模块回执，
避免部分停止冒充成功。Runtime 还跟踪 launch 进程派生的独立 Linux 进程组；
即使根进程快速退出，也会按有界的 `SIGINT -> SIGTERM -> SIGKILL` 阶梯回收，
并在首次发送信号前用进程启动时刻校验，避免 PID 复用误杀。
容器退出时 ActuCore 最多重试停卡 3 次；仍未确认时记录 critical 后
强制结束进程，交给容器管理器恢复，不伪装成安全停止。

## 构建与本地部署验收

导航基础镜像按 JetPack 版本选择精确的
`jetson-base:jp<JP_VERSION>-torch@sha256:<digest>`（Ubuntu 20.04 / Python
3.8，ROS Humble 是 `/opt/ros/humble/install` 下的源码 install-space）。基础镜像
按完整 SHA 拉取 Sophus、Vikit 和 FAST-LIVO2，校验并按非重叠顺序应用三份 G1 补丁，再编译
FAST-LIVO2 与 Nav2。日常 ActuCore 镜像通过 `@sha256` 固定该基础镜像，只重编
仓库自有的 `g1_fast_livo2`、`g1_nav2`、`g1_segmented_controller`。G1 输入使用
标准 PointCloud2；基础镜像只内置 FAST-LIVO2 编译所需的两条 Livox 消息定义，
不再编译未运行的 Livox SDK2/Driver。

**Nav2 也是源码编译**，不是 apt：base 是 Focal，而 `ros-humble-*` 的 Debian 包
只有 Jammy 版本，所以 `navigation2` 连同 base 里缺的
`behaviortree_cpp_v3` / `bond_core` / `diagnostic_updater` / `pcl_ros` /
`rosbag2_storage_mcap` 一起按锁定 SHA 自编。`navigation2` 钉在 **1.1.20**，与
迁移前的 `ros-humble-navigation2=1.1.20-1jammy` 是同一个上游 release，运行行为
不随打包形态变化。`mcap_vendor` 间接使用的 MCAP/LZ4 也按其上游
锁定 SHA 预先浅克隆，不在 CMake 构建中重复全量下载。运行时加载 planner/controller/
behavior/bt_navigator/waypoint_follower + navfn、costmap 三层，
以及卡片自带的 `g1_segmented_controller`；`nav2_bringup` 编译所需的轻量
`navigation2` 元数据包也保留。amcl、map_server、DWB、rotation shim、
smac、mppi、constrained_smoother、route、rviz_plugins 刻意不编；它们已被当前
链路取代，或会把 ompl、ceres、xtensor、Qt5 等无用依赖拖进镜像。

镜像里**没有** torch / CLIP / YOLO / ASR 依赖 —— 卡片自身只用标准库 + ROS
消息包，语义航点是 HTTP 调远端 VLM。那些模型依赖属于 perception。G1 实测的
上游 ActuCore 镜像为 `13,786,589,503` bytes，首次加入导航栈的镜像为
`14,665,479,002` bytes，增加 `878,889,499` bytes（约 `0.82 GiB` / `6.38%`）。
该数据由上海 G1 上的 `docker image inspect --format '{{.Size}}'` 实测；增加的是
磁盘、首次拉取和部署传输成本，不代表同等幅度的运行时 CPU/内存增长。稳定且昂贵的
第三方依赖因此预编译进可复用、digest-pinned 的 navigation base，日常 PR 构建
不再重复编译它们。发布该 base 并把精确 digest 写回构建脚本是正式发版前置条件；
只有 ActuCore 导航镜像继承它，Core、Perception 和不含导航的镜像不会承担这部分
体积。基础镜像使用 builder/runtime 双阶段；最终阶段只复制三个 ROS install-space
和包锁，不保留 `/opt/ros_deps_ws`、`/opt/fast_livo_ws`、`/opt/nav2_ws` 的源码、
build 与 log 目录。上述 `0.82 GiB` 是清理前的历史测量，新的最终体积必须在原生
ARM64 完整构建后重新记录。
该固定 base 同时提供 `colcon` / `empy` 和 PyYAML；日常镜像构建会实际执行
`colcon build` 并显式导入 `em` / `yaml`，避免把基础镜像的偶然环境冒充成依赖
契约，并从最终安装的 `nav2_params.yaml` 逐个动态加载全部 BT 插件库，缺库或
链接失败会直接终止镜像构建。VLM 客户端使用 Python 标准库 `urllib`，不额外安装未使用的
`requests`。ActuCore 主进程和三个 Python ROS 入口都安装同一份原子
标准输出 writer，避免并发输出破坏 Docker 日志记录。

`--mirror tuna` 只选择国内 APT/PyPI 源；Git 默认直连官方仓库，避免把公共代理
隐式固化进可复现构建。网络环境需要代理时由维护者显式传入
`GIT_MIRROR_PREFIX`；全局 Git 重写会保留到 FAST-LIVO2/Nav2 编译完成，使
FetchContent 也沿用该值，最终镜像再删除该重写。
构建会核对 fetch 后的完整 Git SHA 和本仓补丁 SHA256，但不会验证上游仓库的
签名提交或签名 tag；锁定 revision 可防漂移，不能替代上游身份签名。

Canvas 外部连线 topic 必须是完整绝对 ROS 名称：以 `/` 开头，每段只使用
字母、数字和下划线且不能以数字开头；空值、相对名称和重复 `/` 会在启动子进程
前返回 `invalid_canvas_wiring`。

```bash
./deploy/build_actucore.sh --mirror tuna
```

这是仓库默认的 ActuCore 构建入口，只有 Jetson 一个变体，无需 Navigation
专用 wrapper。FAST-LIVO2/Nav2 不在日常 PR 镜像中重编，避免 ARM64 QEMU 构建
超过 review 时限。当前正式构建只支持 JP 5.11，并使用已发布的
digest-pinned navigation base。JP 6.1 的对应基础镜像尚未发布；必须
先构建、发布并在仓库中固定匹配的精确 `@sha256` digest，才能恢复
JP 6.1 支持。构建脚本在此之前会拒绝普通 JP 6.1 构建，不会回退到
其他 JetPack 的基础镜像。
只有锁定源码、补丁或系统依赖变化时，维护者才执行：

```bash
GIT_MIRROR_PREFIX=https://ghfast.top/ \
  ./deploy/build_actucore.sh --base --mirror tuna
```

构建机内存较小时可设置 `BUILD_JOBS=2`，限制 navigation base 与日常
ActuCore 镜像的 C++ 并行编译数；默认值仍为 4。

基础镜像必须在原生 ARM64 构建；成功并推送后会输出可写回构建配置的精确
digest。它不是运行时 service，也不注册 Resource Center。日常构建成功后脚本输出
`ACTUCORE_BUILD_DURATION_SEC=<秒数>`，用于比较依赖精简前后的真机构建耗时。

临时验收脚本属于现场环境，不随卡片源码发布。日常镜像统一使用仓库默认的
`./deploy/build_actucore.sh --mirror tuna` 构建；现场脚本只应负责环境预检和替换
单个 ActuCore 容器，不得发送导航目标或速度指令。

正式 `actucore/deploy/service.yml` 也只有 `actucore` 一个 service。地图和
录制目录作为该容器的持久化 volume；不再定义 `fast_livo2` 或 `nav2` service。
首次部署前由现场操作者创建可写目录：

```bash
sudo install -d -o "$(id -u)" -g "$(id -g)" -m 0755 \
  /opt/phanthy-motus/data/fast_livo2/maps \
  /opt/phanthy-motus/data/fast_livo2/recordings
```

数采默认关闭。两类目录都不会自动清理：单次地图会话仍受 512 MiB 安全上限，
但总历史和录制数据会持续占用磁盘，操作者需在停卡后自行归档或删除旧目录。
导出时若磁盘重命名失败，保留原 `.partial` 目录和会话路径并返回可重试错误；
后续 `stop` 从同一会话继续导出，不会丢失未完成录制。
离线标注遇到临时异常会在 1 秒后自动重试一次；再次失败时保留原始会话和
错误状态，等待后续回执或进程恢复重新入队。

## 第三方与验收边界

统一镜像包含 FAST-LIVO2/Vikit GPL 组件和 Apache-2.0 Nav2/adapter，镜像标签
必须保留组合许可证和锁定 revision。版本与许可证见
[runtime/FAST_LIVO2_THIRD_PARTY.md](runtime/FAST_LIVO2_THIRD_PARTY.md) 和
[runtime/NAV2_THIRD_PARTY.md](runtime/NAV2_THIRD_PARTY.md)。

本地自动测试只能证明合同、生命周期、进程托管、失败回滚和无额外 service。
当前交付状态为 **待 G1 实机测试**：真实传感器、地图质量、重定位、路径和
物理运动仍需 owner 按上述 G1 脚本另行验收。
