# ControlledSemanticSpatial ActuCore 卡片

`ControlledSemanticSpatial` 是 FAST-LIVO2、Nav2 和 VLN 的统一公开卡片，属于
**ActuCore（执行模型层）**：把意图/目标变成运动指令。Canvas 只看到这一张
`processor` 卡片，正式部署只运行一个 ActuCore 容器。

## 运行边界

```text
Driver lidar + imu ─┐
camera rgb ─────────┼─> ControlledSemanticSpatial card (ActuCore container)
optional goal_pose ─┘      ├─ FAST-LIVO2 mapping/localization child process
                            ├─ Nav2 planner/controller child process
                            └─ semantic waypoint processor
                                      |
                                      `─ velocity_proposal -> Driver loco
```

- `NavigationPlugin` 是唯一 MCP/Canvas 生命周期所有者。
- FAST-LIVO2 和 Nav2 ROS launch 由 `NavigationRuntime` 作为同容器子进程组
  启停；运行时不调用 Docker，也不需要 Docker socket。
- odom、registered cloud、obstacle map 和 collection status 是卡片内部 ROS
  边，不再作为 Canvas 公共连线端口。
- VLN 命中地点后直接调用同卡片 planner；无论是 Canvas、
  `goal_pose` topic 还是 VLN 入口，planner 都为每个新任务生成独立
  `nav_id`。
- Nav2 仍只发布 `phanthy.navigation.velocity_proposal.v1`，Driver 继续负责
  物理执行、TTL、急停、二次限幅和停车确认。

## 外部输入

| port | topic | 必需 |
| --- | --- | --- |
| `lidar` | `/ubuntu/navigation/lidar` | 是 |
| `imu` | `/ubuntu/navigation/imu` | 是 |
| `rgb` | `/ubuntu/camera/rgb` | 是 |
| `goal_pose` | `/ubuntu/navigation/goal_pose` | 否 |

## 公共输出

| port | topic | 用途 |
| --- | --- | --- |
| `map_view` | `/ubuntu/navigation/fast_livo2/map_view` | Canvas 地图与机器人位姿 |
| `status` | `/ubuntu/navigation/fast_livo2/status` | 定位、建图和运行状态 |
| `velocity_proposal` | `/ubuntu/navigation/nav2/velocity_proposal` | 连接 Driver `loco` 执行器 |
| `plan` | `/plan` | 当前二维全局路径 |
| `costmap` | `/global_costmap/costmap` | 实时全局代价地图 |

`livo_odom`、registered cloud、confirmed static map、obstacle map 和
collection status topic 仍由
同容器内的定位、规划、语义和数据采集逻辑消费或发布，只是不再生成 Canvas
右侧连线端口。详细 frame、QoS、freshness 和速度约束见内部实现说明：

- [mapping/README.md](mapping/README.md)
- [planning/README.md](planning/README.md)
- [semantic/README.md](semantic/README.md)

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

建图恢复为直接累计模式：FAST-LIVO2 adapter 对每帧注册点云按 `0.10 m`
体素去重后，将导航高度带内且距离有效的体素首次出现即写入静态图，不等待
多帧确认、不跟踪动态分量，也不通过后续自由射线删除已累计点。这样可保留
墙面和稀疏结构的完整证据，但建图期间出现的人或其他移动物体也可能固化进
地图，需由操作者在相对静态的环境中完成建图。

实时避障与累计地图彼此独立：当前 registered cloud 继续进入 Nav2 开启
marking/raytrace clearing 的 live ObstacleLayer，所以导航期间新出现或移动的
物体仍参与即时避障。原始 FAST-LIVO2 odom/cloud 使用 latest-only 订阅；cloud
只有在其源时间戳已被前后 odom/TF 包围且最近位姿差不超过 50 ms 时才对
Nav2 发布，避免 adapter 排队旧帧或让点云早于 TF。canonical registered cloud
发布是独立快路径：PointCloud2 的 XYZ 解码、`map` 坐标变换、高度过滤和
float32 打包均使用 NumPy 批量运算，不再逐点进入 Python 循环。静态地图累计与
Canvas 编码在 latest-only 后台任务中执行，
不会占用导航点云回调。后台来不及处理时覆盖旧建图样本并在诊断中累计
`mapping_work_dropped`，但不阻塞最新导航点云。Canvas `map_view` 显示累计静态点和新鲜的最新实时扫描，
并显示单独累计的高度带外表面。输出最多 80,000 点，并为低于、位于和高于
导航高度带的三组点分别保留显示预算，避免地面被大量障碍点截断；该采样只
影响监控，不改变规划输入。显示帧直接编码已经分别有界的静态、范围外和实时
点源，不再构造一次性全图体素副本，避免 Canvas 更新拖慢 Nav2 所依赖的 odom
与 registered cloud。点云主体按 1 Hz 重编码并缓存；缓存帧只替换前 12 字节的
机器人 `x/y/yaw`，以 5 Hz 发布，因此位姿刷新不再重复打包最多 80,000 个地图点。
FAST-LIVO2 diagnostics 的 `latency_ms` / `latency_max_ms` 分别报告最近值和进程内
最大值，包含 `cloud_decode`、`cloud_pose_wait`、`cloud_transform_filter`、
`cloud_pack_publish`、`cloud_end_to_end`、`map_view_encode` 和
`map_view_pose_publish`；`map_view_cache_age_sec` 用于区分点云主体陈旧和位姿刷新
本身的开销。已保存的 confirmed static PCD 还会绑定建图时的障碍高度带；加载
时若当前上下界不同会拒绝使用，需恢复原配置或重新建图，避免静态证据语义
悄然变化。

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

### 为什么不声明 `x-completion`

ActuCore 契约允许长时动作用 `inputSchema.x-completion` 声明 ACP 完成回调，
但本卡片**刻意不声明**：`agent-core/src/mcp_client.py::await_pending()` 是
**全局 barrier**（"等所有 pending"），只有注册在 `on_interrupt_*` hook 里的
tool+action 免除。一次导航可能持续几分钟，声明后这段时间里任何
actuator/processor 调用都会被挡住 —— 包括 TTS 说话和本卡片自己的
`stop_nav`，等于让机器人在移动中失去"叫停"通路。

现有设计已经覆盖同一需求：调用立即返回 `nav_id`，终态通过 `status` topic
异步上报，需要阻塞语义时显式调 `wait_navigation_done`。若将来确实要
"导航到点并在到达时通知"，前置条件是 agent-core 先支持按资源作用域的
barrier（而不是全局），那时再加 `x-completion` 与 SSE 完成事件。

Canvas 手动执行 `navigate_to_pose` 时，Agent Core 只透明转发 MCP
请求。planner 接受目标后生成新 `nav_id`，并在整个任务的
velocity proposal 中保持该 ID。Nav2 上报匹配的终态后，卡片释放
活动任务；下一次点击因此会获得另一个 ID，无需重启
Canvas、Core 或 Driver。Driver 在空闲订阅状态接纳首条新鲜、合法、
非零且非终态 proposal 的 `nav_id`，活动任务期间拒绝 ID 切换，并在
终态零速后退役该 ID。Core 不另外维护 authorize/revoke 状态。

`start` 按 runtime → mapping → planning → semantic 顺序获取资源；任一步
失败会按相反顺序回滚。`stop_mapping` 和 `load_map` 的 backend 等待预算分别
至少为 360 s 和 900 s。可重试的地图收口失败会保留 Canvas wiring、运行时和
`finalizing` 事务，下一次 `stop`/`stop_mapping` 从原事务继续；永久失败会释放
mapping 控制对象，并继续回收其他模块和运行时。已完成的同名
`stop_mapping` 终态可幂等重放原保存回执，避免迟到重试制造第二份成功结果。

`stop` 始终尝试停止所有内部模块和两个 launch 子进程组，并保留各模块回执，
避免部分停止冒充成功。Runtime 还跟踪 launch 进程派生的独立 Linux 进程组；
即使根进程快速退出，也会按有界的 `SIGINT -> SIGTERM -> SIGKILL` 阶梯回收，
并在首次发送信号前用进程启动时刻校验，避免 PID 复用误杀。

## 构建与本地部署验收

统一镜像基于 `jetson-base:jp<JP_VERSION>-torch`（Ubuntu 20.04 / Python 3.8，
ROS Humble 是 `/opt/ros/humble/install` 下的**源码 install-space**）。构建时按
完整 SHA 拉取 Livox SDK2、Livox ROS Driver2、Sophus、Vikit 和 FAST-LIVO2，校验
并应用两份 G1 补丁，再编译 FAST-LIVO2、Nav2 与两套 ROS adapter。

**Nav2 也是源码编译**，不是 apt：base 是 Focal，而 `ros-humble-*` 的 Debian 包
只有 Jammy 版本，所以 `navigation2` 连同 base 里缺的
`behaviortree_cpp_v3` / `bond_core` / `diagnostic_updater` / `pcl_ros` /
`rosbag2_storage_mcap` 一起按锁定 SHA 自编。`navigation2` 钉在 **1.1.20**，与
迁移前的 `ros-humble-navigation2=1.1.20-1jammy` 是同一个上游 release，运行行为
不随打包形态变化。只编卡片真正加载的那一档（planner/controller/smoother/
behavior/bt_navigator/waypoint_follower/velocity_smoother + navfn、DWB、
rotation shim、costmap 三层）；amcl、smac、mppi、constrained_smoother、route、
rviz_plugins 刻意不编，它们会把 ompl、ceres、xtensor、Qt5 拖进镜像。

镜像里**没有** torch / CLIP / YOLO / ASR 依赖 —— 卡片自身只用标准库 + ROS
消息包，语义航点是 HTTP 调远端 VLM。那些模型依赖属于 perception。
APT、PyPI、Git 源码下载均走国内可达入口（`GIT_MIRROR_PREFIX`，mcap_vendor 的
FetchContent 也经同一镜像重写）。

```bash
./deploy/build_actucore.sh --mirror tuna
```

这是仓库默认的 ActuCore 构建入口，只有 Jetson 一个变体，无需 Navigation 专用
wrapper，也无需预构建 FAST-LIVO2 或 Nav2 镜像。首次构建约 1-3 小时（源码编译
Nav2 是主要开销），之后走 layer 缓存。

G1 临时验收只创建一个容器。将构建输出的精确镜像名传入：

```bash
export ACTUCORE_IMAGE=local/phanthy-motus/actucore:<exact-tag>
STAGE=preflight bash actucore/plugins/navigation/deploy/scripts/owner-start-g1-test-containers.sh
STAGE=start bash actucore/plugins/navigation/deploy/scripts/owner-start-g1-test-containers.sh
```

在北京 G1 上从当前 PR 分支的仓库根目录一次完成默认构建和测试容器替换：

```bash
bash actucore/plugins/navigation/deploy/scripts/deploy-g1.sh
```

`deploy-g1.sh` 不实现另一套构建逻辑；它只调用仓库默认的
`./deploy/build_actucore.sh --mirror tuna`，再调用上述容器生命周期脚本。
旧容器只有带 `com.phanthymotus.test-owner=navigation-card`，或迁移前已知值
`com.phanthymotus.test-owner=nav2-card` 时才会被替换；其他 owner 仍会安全拒绝。
脚本不发送导航目标或速度指令。

正式 `actucore/deploy/service.yml` 也只有 `actucore` 一个 service。地图和
录制目录作为该容器的持久化 volume；不再定义 `fast_livo2` 或 `nav2` service。

## 第三方与验收边界

统一镜像包含 FAST-LIVO2/Vikit GPL 组件和 Apache-2.0 Nav2/adapter，镜像标签
必须保留组合许可证和锁定 revision。版本与许可证见
[runtime/FAST_LIVO2_THIRD_PARTY.md](runtime/FAST_LIVO2_THIRD_PARTY.md) 和
[runtime/NAV2_THIRD_PARTY.md](runtime/NAV2_THIRD_PARTY.md)。

本地自动测试只能证明合同、生命周期、进程托管、失败回滚和无额外 service。
当前交付状态为 **待 G1 实机测试**：真实传感器、地图质量、重定位、路径和
物理运动仍需 owner 按上述 G1 脚本另行验收。
