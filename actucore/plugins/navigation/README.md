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
  `nav_id`。终态到达时只发布一次零速 proposal，不会继续以 5 Hz
  刷新已结束任务；终态结果本身仍可查询和幂等重放。
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
| `goal_pose` | `/ubuntu/navigation/goal_pose` | 否；连线后由 Agent Core `x-topic-actions` 转换为 `navigate_to_pose` |

LiDAR 每点 `timestamp` 必须是 `float64` 绝对纳秒，时间单调且一帧跨度位于
`(0, 200] ms`。状态会显示 `sensor_frame`、TF、点时间跨度和 `odom_health`；
契约不满足时返回 `sensor_frame_mismatch`、`sensor_tf_unavailable`、
`point_time_invalid` 或 `raw_odom_discontinuity`，不会继续生成伪正常地图。

## 公共输出

| port | topic | 用途 |
| --- | --- | --- |
| `map_view` | `/ubuntu/navigation/fast_livo2/map_view` | Canvas 地图与机器人位姿 |
| `status` | `/ubuntu/navigation/fast_livo2/status` | 定位、建图和运行状态 |
| `collection_status` | `/ubuntu/navigation/fast_livo2/collection_preview` | 采集中显示 RGB/帧号/距离，停止后显示导出进度 |
| `velocity_proposal` | `/ubuntu/navigation/nav2/velocity_proposal` | 连接 Driver `loco` 执行器 |
| `costmap` | `/global_costmap/costmap` | 实时全局代价地图 |

`livo_odom`、registered cloud、confirmed static map 和 obstacle map 仍只由
同容器内的定位、规划和语义逻辑消费，不生成 Canvas 右侧连线端口。详细
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
物体仍参与即时避障。原始 FAST-LIVO2 odom/cloud 使用 latest-only 订阅；cloud
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
与 registered cloud。点云主体按 1 Hz 重编码并缓存；缓存帧只替换前 12 字节的
机器人 `x/y/yaw`，以 1 Hz 发布；Canvas 编码/发布串行且 DDS 只保留最新一帧，
因此慢显示不会堆积并挤占 odom/registered cloud 回调。
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
卡片级 `start`/`stop`/`config` 转换串行执行，不允许 Canvas 的迟到请求在
前一次启动中途关闭 backend。若 Nav2 command bridge 子进程仍存活，但
Fast DDS 在首个发现窗口内暂未报告 command subscriber，只重建一次
planning bridge 并重试发现，不重启 FAST-LIVO2 或 Nav2 子进程；第二次仍失败
才执行完整回滚。

`stop` 始终尝试停止所有内部模块和两个 launch 子进程组，并保留各模块回执，
避免部分停止冒充成功。Runtime 还跟踪 launch 进程派生的独立 Linux 进程组；
即使根进程快速退出，也会按有界的 `SIGINT -> SIGTERM -> SIGKILL` 阶梯回收，
并在首次发送信号前用进程启动时刻校验，避免 PID 复用误杀。

## 构建与本地部署验收

导航基础镜像按 JetPack 版本选择精确的
`jetson-base:jp<JP_VERSION>-torch@sha256:<digest>`（Ubuntu 20.04 / Python
3.8，ROS Humble 是 `/opt/ros/humble/install` 下的源码 install-space）。基础镜像
按完整 SHA 拉取 Sophus、Vikit 和 FAST-LIVO2，校验并应用三份 G1 补丁，再编译
FAST-LIVO2 与 Nav2。日常 ActuCore 镜像通过 `@sha256` 固定该基础镜像，只重编
仓库自有的 `g1_fast_livo2`、`g1_nav2`、`g1_segmented_controller`。G1 输入使用
标准 PointCloud2；基础镜像只内置 FAST-LIVO2 编译所需的两条 Livox 消息定义，
不再编译未运行的 Livox SDK2/Driver。

**Nav2 也是源码编译**，不是 apt：base 是 Focal，而 `ros-humble-*` 的 Debian 包
只有 Jammy 版本，所以 `navigation2` 连同 base 里缺的
`behaviortree_cpp_v3` / `bond_core` / `diagnostic_updater` / `pcl_ros` /
`rosbag2_storage_mcap` 一起按锁定 SHA 自编。`navigation2` 钉在 **1.1.20**，与
迁移前的 `ros-humble-navigation2=1.1.20-1jammy` 是同一个上游 release，运行行为
不随打包形态变化。运行时加载 planner/controller/
behavior/bt_navigator/waypoint_follower + navfn、costmap 三层，
以及卡片自带的 `g1_segmented_controller`；`nav2_bringup` 编译所需的轻量
`navigation2` 元数据包也保留。amcl、map_server、DWB、rotation shim、
smac、mppi、constrained_smoother、route、rviz_plugins 刻意不编；它们已被当前
链路取代，或会把 ompl、ceres、xtensor、Qt5 等无用依赖拖进镜像。

镜像里**没有** torch / CLIP / YOLO / ASR 依赖 —— 卡片自身只用标准库 + ROS
消息包，语义航点是 HTTP 调远端 VLM。那些模型依赖属于 perception。G1 实测的
上游 ActuCore 镜像为 `13,786,589,503` bytes，加入导航栈后的镜像为
`14,674,555,445` bytes，增加 `887,965,942` bytes（约 `6.4%`）。稳定且昂贵的
第三方依赖因此预编译进可复用、digest-pinned 的 navigation base，日常 PR 构建
不再重复编译它们。

`--mirror tuna` 只选择国内 APT/PyPI 源；Git 默认直连官方仓库，避免把公共代理
隐式固化进可复现构建。网络环境需要代理时由维护者显式传入
`GIT_MIRROR_PREFIX`，mcap_vendor 的 FetchContent 也沿用该值。

```bash
./deploy/build_actucore.sh --mirror tuna
```

这是仓库默认的 ActuCore 构建入口，只有 Jetson 一个变体，无需 Navigation
专用 wrapper。FAST-LIVO2/Nav2 不在日常 PR 镜像中重编，避免 ARM64 QEMU 构建
超过 review 时限。默认构建入口会按 JetPack 选择基础镜像：JP 5.11
使用已发布的 digest-pinned navigation base；JP 6.1 在对应基础镜像
发布前直接拒绝构建，需维护者显式设置与 JP 6.1 匹配的
`ACTUCORE_NAVIGATION_BASE_IMAGE`。所有覆盖都必须使用精确 `@sha256`
引用，不会默认回退到其他 JetPack 的基础镜像。
只有锁定源码、补丁或系统依赖变化时，维护者才执行：

```bash
GIT_MIRROR_PREFIX=https://ghfast.top/ \
  ./deploy/build_actucore.sh --base --mirror tuna
```

基础镜像必须在原生 ARM64 构建；成功并推送后会输出可写回构建配置的精确
digest。它不是运行时 service，也不注册 Resource Center。日常构建成功后脚本输出
`ACTUCORE_BUILD_DURATION_SEC=<秒数>`，用于比较依赖精简前后的真机构建耗时。

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
`./deploy/build_actucore.sh --mirror tuna`，使用仓库锁定的默认基础镜像，再调用
上述容器生命周期脚本。
旧容器只有带 `com.phanthymotus.test-owner=navigation-card`，或迁移前已知值
`com.phanthymotus.test-owner=nav2-card` 时才会被替换；其他 owner 仍会安全拒绝。
脚本不发送导航目标或速度指令。

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

## 第三方与验收边界

统一镜像包含 FAST-LIVO2/Vikit GPL 组件和 Apache-2.0 Nav2/adapter，镜像标签
必须保留组合许可证和锁定 revision。版本与许可证见
[runtime/FAST_LIVO2_THIRD_PARTY.md](runtime/FAST_LIVO2_THIRD_PARTY.md) 和
[runtime/NAV2_THIRD_PARTY.md](runtime/NAV2_THIRD_PARTY.md)。

本地自动测试只能证明合同、生命周期、进程托管、失败回滚和无额外 service。
当前交付状态为 **待 G1 实机测试**：真实传感器、地图质量、重定位、路径和
物理运动仍需 owner 按上述 G1 脚本另行验收。
