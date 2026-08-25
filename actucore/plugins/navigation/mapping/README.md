# Navigation 内部 FAST-LIVO2 模块

该模块负责统一 `ControlledSemanticSpatial` 卡片内的会话级 LiDAR-Inertial 建图、里程计、
运动补偿点云、标准 TF 和 Canvas 地图可视化。它不再单独注册 MCP 卡片，
FAST-LIVO2 及 adapter 由同一个 ActuCore 容器内的 `NavigationRuntime`
托管；Driver 只提供已冻结的标准传感器输入。

## 边界

```text
Driver navigation_sensors
  /ubuntu/navigation/lidar            PointCloud2, 10 Hz
  /ubuntu/navigation/imu              Imu, 200 Hz
                    |
                    v
       FAST-LIVO2 internal module
              |-- /ubuntu/navigation/odom
              |-- /ubuntu/navigation/cloud_registered
              |-- /ubuntu/navigation/static_map
              |-- /ubuntu/navigation/obstacle_map
              |-- TF map -> base_link
              `-- /ubuntu/navigation/fast_livo2/map_view
                    |
                    v
              Nav2 planner/controller
```

- 使用已验证的 Driver `navigation_sensors` 接口，不读取 Canvas 旧
  `/ubuntu/lidar/cloud` envelope，也不在 ActuCore 内复制设备 SDK。
- FAST-LIVO2 原始输出是 `camera_init -> aft_mapped`，其中状态位姿是传感器
  位姿。adapter 使用实测 `base_link -> livox_frame` 外参计算
  `T_map_base = T_map_sensor * inverse(T_base_sensor)`，不能仅重命名 frame。
- 原始 `/tf`、debug cloud 和 marker 全部隔离在
  `/ubuntu/navigation/fast_livo2/raw/*`，只有 adapter 发布权威
  `map -> base_link`。
- 建图时 `map` 是本次 FAST-LIVO2 进程的会话原点。加载旧图后，
  adapter 只在有限搜索范围内匹配成功后建立旧图到当前会话的刚体变换；
  匹配前不发布伪造的 canonical odom/cloud 或 TF。

## 输入

| port | topic | type / QoS | 约束 |
| --- | --- | --- | --- |
| `lidar` | `/ubuntu/navigation/lidar` | `sensor_msgs/msg/PointCloud2`; `RELIABLE + KEEP_LAST(2)` | MID360 `livox_frame`，ROS system time，逐点 offset 保留 |
| `imu` | `/ubuntu/navigation/imu` | `sensor_msgs/msg/Imu`; `RELIABLE + KEEP_LAST(200)` | 与点云相同时钟域和安装旋转，约 200 Hz |

这两路输入来自既有 Driver `navigation_sensors` sensor cards。缺失、过期或
frame 不符时同容器 adapter 不发布伪造的 canonical odom/cloud。
adapter 对 FAST-LIVO2 原始 odom/cloud 使用
`BEST_EFFORT + KEEP_LAST(1)`，慢回调只保留最新样本，不把旧帧排成约
0.5 秒的内部积压。Nav2 readiness 容许最多 `0.8 s` 接收调度抖动，
并每次从最新 header stamp 重算源数据 age；源 age 超过 `1.0 s`
仍 fail closed，不会因为缓存的旧 age 伪装成新鲜数据。live
PointCloud2 的 XYZ 解码、刚体变换、高度过滤和 float32 输出打包走 NumPy
批处理；边界、字段、有限值、点数和字节上限仍在发布前 fail closed。

## 输出

| port                | topic                                             | frame              | 用途                            |
| ------------------- | ------------------------------------------------- | ------------------ | ----------------------------- |
| `livo_odom`         | `/ubuntu/navigation/odom`                         | `map -> base_link` | Nav2 位姿与速度反馈                  |
| `registered_cloud`  | `/ubuntu/navigation/cloud_registered`             | `map`              | Nav2 实时障碍层输入                        |
| `static_map`        | `/ubuntu/navigation/static_map`                   | `map`              | 直接累计体素生成的二维 OccupancyGrid，供 Nav2 StaticLayer |
| `obstacle_map`      | `/ubuntu/navigation/obstacle_map`                 | `map`              | 累计静态障碍的兼容诊断投影                |
| `map_view`          | `/ubuntu/navigation/fast_livo2/map_view`          | `map`              | Canvas 累计静态点、最新实时点和机器人位置            |
| `status`            | `/ubuntu/navigation/fast_livo2/status`            | JSON               | 算法进程、输入 freshness、frame 和产物状态 |
| `collection_status` | `/ubuntu/navigation/fast_livo2/collection_preview` | JPEG               | 最新同步 RGB、采集帧号和 LiDAR 障碍物距离标注     |

adapter 先按 `0.10 m` 体素把同一帧命中去重；导航高度带和有效距离内的体素
首次出现即写入累计静态图，不等待多帧确认、不跟踪动态分量，也不通过后续
自由射线删除。点云会先等待 odom/TF 历史从前后包围其源时间戳，再选择
时间差不超过 `50 ms` 的 canonical odom 位姿配对；配对成功前不会发布给
Nav2，也不会进入累计。不匹配帧进入 diagnostics 计数，不能拿最新位姿
拼接旧点。
直接累计可以恢复墙面和稀疏结构密度，但不提供人员语义分割或动态物体清除，
建图时经过的人可能固化进地图，现场应尽量保持环境静止。

高度阈值是 canonical `map` frame 的 Z，不是雷达安装高度。G1 默认从
`-0.30...+0.30 m` 开始；若把下界放到地面带（北京现场 `-1.0 m` 会纳入大量
`-0.8...-0.4 m` 点），这些点会立即成为静态占用。直接累计不会因之后调回
阈值而删除旧证据，因此修改高度带后必须开始一张新地图，不能继续使用已经
被错误高度带写入的地图。

当前 registered cloud 始终独立进入 Nav2 实时障碍层，因此导航期间移动物体
仍参与即时避障。Canvas `map_view` 在同一现有 XYZ 点数组中合并累计静态点和
新鲜的最新实时扫描。各来源已经分别受体素和点数约束，adapter 直接按高度组
分配显示预算，不再每秒把全部来源重新散列成临时体素图；这样监控编码不会
阻塞 canonical odom/cloud 的 freshness。高度带外表面另以 `0.20 m` 体素
累计；加载旧图时从 raw PCD 恢复。它们只用于 Canvas 阈值调试，不进入
registered cloud、累计静态图或 Nav2 costmap。`map_view` 最多编码 80,000 点，
对低于、位于和高于导航高度带的点分别保留预算，因此地面不会被先写入的
障碍点挤出前端上限。地图点主体每秒编码一次并缓存，机器人 `x/y/yaw` 只更新
固定 12 字节头部并以 5 Hz 发布；位姿刷新不再重复编码整张地图。现有渲染器
继续按高度显示范围外点为蓝色/粉色。
Agent Core 在显示层
把同为 `map` frame 的 Nav2 `/plan` 叠加为绿色路径和橙色终点，不改变
`map_view` 点数组语义，也不让 FAST-LIVO2 依赖 Nav2。旧图加载后在
重定位成功前不显示伪造的机器人位姿。地图卡片支持三维
浏览和正上方二维投影切换；二维模式只是同一三维点云的平面显示，不是另存
一份 occupancy grid。绿色机器人箭头沿 canonical `base_link` 的 `+X` 前向，
并保持 ROS `map` frame 的 yaw 正方向。它是监控视图，不是
可重定位地图格式。原始 PCD 分片保存在宿主机
`/opt/phanthy-motus/data/fast_livo2/maps`；`stop_mapping` 还会原子写入
`maps/static/<map>-*.static.pcd`，保存导航高度带内直接累计的静态点。

live PointCloud2 在复制数据区前同时检查 `width * height <= 200,000` 和
`data <= 64 MiB`；累计及加载后的 confirmed static map 都以
200,000 点为硬上限。live frame、静态证据或确认图超限都会 fail closed，
不会用静默下采样改变占用语义。用于有界重定位的 raw PCD 读取仍可在总预算内
抽样到最多 200,000 个参考点。

Nav2 的全局静态层使用 `/ubuntu/navigation/static_map`，实时动态层使用
`/ubuntu/navigation/cloud_registered`；`obstacle_map` 只保留为兼容诊断，不再
驱动 global costmap。卡片配置
`obstacle_min_height_m` 和 `obstacle_max_height_m` 定义 `map` frame 中参与
二维导航的 Z 高度带，默认 `-0.30…+0.30 m`，可在 Canvas 停止卡片后调整，
下次启动生效。adapter 用同一高度带过滤实时 Nav2 点云，并把稳定静态图输出
为 `-1/0/100` 的完整 OccupancyGrid；单帧超出 `8.5 m` 的点不参与静态证据，
栅格还受单边 `2,048` 格、总计 `2,000,000` 格的硬上限保护，异常外点或
无界轨迹会 fail closed，不会分配无限地图。监控中低于下界的最新点
显示为蓝色，高于上界的点显示为粉色，范围内点继续使用彩虹高度色，并显示
当前阈值图例。阈值是地图坐标，不直接等同于雷达物理安装高度；应结合现场
地面色带和低矮障碍调节，且必须满足 `-3.0 <= min < max <= 3.0`。

`/ubuntu/navigation/fast_livo2/diagnostics` 通过 `latency_ms` 给出最近一次
`cloud_decode`、`cloud_pose_wait`、`cloud_transform_filter`、
`cloud_pack_publish`、`cloud_end_to_end`、`map_view_encode` 和
`map_view_pose_publish` 分段耗时，`latency_max_ms` 保留本进程最大值；同时发布
`map_view_cache_age_sec`、`map_view_point_refresh_hz=1` 和
`map_view_pose_refresh_hz=2`。Canvas 帧最多携带 40,000 点，避免在 G1 上以
5 Hz 重复序列化近 1 MiB 帧并阻塞 odom/registered cloud；这些字段用于区分
传感器/TF 等待、点云计算、DDS 发布和 Canvas 编码延迟，不改变 500 ms
freshness 门禁。

## Actions

| action          | 参数                                                                                    | 语义                                                    |
| --------------- | ------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| `start_mapping` | `map_name`                                                                            | 清空 Canvas 会话图并启动一个新的 FAST-LIVO2 进程                    |
| `stop_mapping`  | 无                                                                                     | 先检查静态证据，通过 FAST-LIVO2 参数服务同步落盘 raw PCD，再 `SIGINT` 停止算法；随后保存 confirmed static PCD 并原子写 session manifest |
| `load_map`      | `map_name`                                                                            | 先校验新旧 manifest/PCD，再串行替换定位前端，失败时尝试回滚旧图 |
| `relocalize`    | `initial_x`, `initial_y`, `initial_z`, `initial_yaw`, `search_xy_m`, `search_yaw_rad` | 以操作者给定位姿为中心做有界二维 scan-to-map 匹配                       |

重定位只有在匹配率至少为 0.35、且最优候选未贴住 XY/yaw 搜索边界时才提交新的
`map -> base_link` 对齐；否则明确拒绝并要求扩大搜索范围或修正初始位姿。重复
重定位提交时会先丢弃旧对齐下生成的实时点和 Canvas 缓存，等待新对齐下的下一帧，
避免新箭头叠加旧扫描造成显示错位。该对齐同时供 Canvas、TF、costmap 和寻路使用，
因此拒绝低质量结果是导航安全边界，不只是显示策略。

`map_name` 只允许 `A-Z a-z 0-9 _ . -`，最长 64 字符。停止 Canvas 时若仍在
建图，卡片会先执行 `stop_mapping` 再释放 ROS backend。直接杀死容器或 ROS
进程不具备完成跨进程静态图事务的条件，只会受控终止算法并明确留下无成功
manifest 的产物；需要可加载地图时必须先获得 `stop_mapping.status=saved`。
FAST-LIVO2 在受控 `SIGINT` 收口期间可因上游 C++ 析构路径返回
`-SIGABRT (-6)`。supervisor 仅在自己已发出停止信号的路径将 `0/-SIGINT/-SIGABRT`
视为受控停止，并在回执保留原始 `algorithm_return_code`；运行中自行 `-6`
或其他退出码仍按 `algorithm_exited/algorithm_stop_failed` fail closed。
raw PCD 不再依赖上述退出路径触发析构：supervisor 先设置内部
`pcd_save.flush_sequence`，只有 FAST-LIVO2 同线程完成原子写盘且本会话出现新
PCD 后才发送停止信号。落盘失败或超时会返回可重试错误并保持建图进程运行，
不会先停算法再留下无法恢复的 `map_artifact_missing`。

重定位的操作顺序为：

1. 执行 `load_map(map_name)`；只接受该卡片 `stop_mapping` 生成的
   `phanthy.navigation.fast_livo2_map_session.v1` manifest，不猜测孤立 PCD 的归属。
2. 机器人保持静止，等待新前端产生新鲜 odom 和 registered cloud。
3. 在 Canvas 中给出机器人在旧图上的近似 `x/y/z/yaw`，执行
   `relocalize`。默认只搜索位置±1.0 m、航向±0.35 rad；允许范围分别是
   `0.1–3.0 m` 和 `0.05–π/2 rad`。
4. 返回 `status=relocalized` 后，卡片才发布旧图 frame 下的
   `map -> base_link`、registered cloud 和障碍图；此时再启动 Nav2 任务。
5. 切换地图时直接再次执行 `load_map`。supervisor 先校验
   目标图和当前图的 manifest/PCD，未通过时不停旧前端；切换后如
   adapter 加载或新算法启动失败，会尝试恢复旧图。回执中的
   `rollback_status` 为 `restored/failed`，`loaded_map/runtime_mode`
   是切换后真实状态；回滚也失败时必须 fail closed。卡片停止时会自动释放
   当前定位前端，不再对外提供 `unload_map`。

当前锁定的 FAST-LIVO2 本身仍然没有 PCD 加载、`/initialpose` 或全局回环。
这里的加载和重定位由同容器 ActuCore adapter 实现：它是依赖人工
初值的有界匹配，不是无初值的全局搜索，也不在定位成功后持续做全局
闭环校正。匹配分数不足、点数不足、数据过期、manifest/PCD 损坏时全部
fail closed。

当前有界匹配使用 `0.20 m` 二维体素，旧图和当前 scan 都至少需要
40 个有效障碍体素，scan 最多参与 1,200 点，最低匹配比为 `0.20`。
成功回执会返回 `match_ratio`、`matched_points` 和 `evaluated_points`；
`status=relocalized` 只证明本次有界匹配过关，不等同于长时间全局定位
可靠性已验收。

### 生命周期与大图读取

- supervisor 向 adapter 发送首次地图控制请求前，会同时等待 command subscriber
  和 response publisher 完成 DDS discovery，最长 5 秒。adapter 尚未就绪时返回
  可重试的 `fast_livo2_adapter_unavailable`，不会丢失一次性 VOLATILE 请求后
  占满普通 130 秒响应预算。
- Core 和 supervisor 都将 `start_mapping` / `stop_mapping` / `load_map` /
  `relocalize` 收口到单一地图生命周期锁；采集启停使用独立锁，
  不会与地图操作互相阻塞。并发生命周期请求立即返回 `runtime_busy`，不占满
  executor callback 线程。`stop_mapping` 的外层等待预算至少 360 s，覆盖算法
  最长 120 s 受控停止、adapter 最长 130 s 保存及有界快照/fsync；`load_map`
  的外层等待预算至少 900 s。两者不会被较小的普通请求超时配置缩短。
- `stop_mapping` 在发送停止信号前要求收到当前 session 的新鲜静态图诊断，
  且至少已有 40 个 confirmed 点；否则保留建图进程并返回可重试错误。算法
  已正常停止后进入 `finalizing`，静态 PCD 或 manifest 写失败不会清除会话，
  再次执行 `stop_mapping` 会继续同一事务；adapter 会复用已成功保存的静态
  PCD，避免重试生成重复文件。I/O 或响应超时属于可重试失败，mapping core、
  Canvas wiring 和 pending transaction 均保留；结构、路径或资源上限错误属于
  永久失败，会释放 mapping 控制对象并让统一卡片继续停止其他模块。若同名
  stop 请求在事务已成功后迟到，supervisor 会幂等重放原 `saved` 回执并标记
  `already_finalized=true`，不会重复产出地图。
- 地图替换是带 best-effort 旧图恢复的串行事务，不是两个前端
  同时运行的无损热切换。目标 manifest/PCD、障碍高度带、点数及完整静态
  OccupancyGrid 的尺寸/总格数会在停止旧前端之前全部验证；验证失败时旧图
  保持运行。验收必须读回 `status/error_code`、
  `rollback_status`、`loaded_map` 和 `runtime_mode`。
- binary PCD 按采样点偏移流式读取，不再将完整 payload 读入内存；PCD header
  和 ASCII 单条记录均限制为 64 KiB。ASCII token 布局、非空行数必须与声明
  完全一致，多 token、多行或少行均拒绝，解析过程持续检查 map-control
  deadline。每张图最多接受 64 个 raw PCD，manifest 最大 64 KiB，raw 会话快照与 confirmed static PCD
  合计最多 512 MiB；单个 PCD 另有 1 GiB 上限，但仍受更小的会话总量约束。
  raw 重定位参考点最多抽样 200,000 个，而 confirmed static map 超过
  200,000 点会直接拒绝，不做抽样。manifest
  的 `pcd_files` 继续保存 raw PCD 并仅用于有界重定位；
  `static_map_pcd` 保存导航高度带内的确认静态点并用于 Canvas/StaticLayer。
  新保存的 manifest 带 `static_map_format_version=2`，缺少 confirmed static
  PCD 会 fail closed；旧版 manifest 没有该版本字段时兼容回退到 raw PCD，并在加载回执标记
  `static_map_source=legacy_raw`；该回退不具备动态物体过滤保证。所有路径仍须
  位于当前 map root 的固定目录。confirmed static PCD 会同时记录建图时的
  `obstacle_height_range_m`；加载时必须与当前卡片上下界一致，否则 fail closed，
  操作者应恢复原上下界或重新建图，不能用另一套高度语义解释既有静态证据。
  raw 或 static 任一保存失败都不会发布新的成功 manifest；永久收口失败会
  best-effort 删除本次未提交快照，避免无 manifest 产物持续占用磁盘。

- `load_map` 先在当前状态之外完成 PCD 解析、静态体素化和完整 OccupancyGrid
  构建；deadline 内仅进行 O(1) 状态指针切换，控制回执发布后才序列化并发布
  大栅格。超时发生在 commit 前时旧图和定位状态保持不变。

统一 Runtime 会持续记录 launch 进程产生的独立 Linux 子进程组。停止或启动
回滚时，即使 launch 根进程已先退出，也会在校验进程组 leader 的启动时刻后，
按有界 `SIGINT` grace、`SIGTERM` 2 s、`SIGKILL` 2 s 阶梯回收；非 Linux
环境没有 `/proc` 时只执行常规 launch 进程组清理。

## 自动数据采集

数据采集属于统一 `ControlledSemanticSpatial` 卡片，不增加单独卡片，也不增加
`start_recording` / `stop_recording` 等公开 action。创建卡片时配置：

| 配置 | 默认值 | 语义 |
| --- | --- | --- |
| `collection_enabled` | `false` | Canvas 启动该卡片时自动开始采集，停止卡片时自动收口 |
| `collection_directory` | `/opt/phanthy-motus/data/fast_livo2/recordings` | ActuCore 容器内持久化根目录；只能配置为该挂载目录或其子目录 |

启用后，同容器 adapter 从以下 Driver/卡片输入中选择每秒一组时间戳对齐的
多模态快照，再通过内部 `collection/*` topic 交给 ROS 2 原生 rosbag2 MCAP
后端。FAST-LIVO2 和 Nav2 仍按原始频率消费输入；1 Hz 只限制数采旁路，不会
降低定位或避障频率。数采旁路对五路输入统一使用
`BEST_EFFORT + KEEP_LAST(4)`；CPU 拥堵时丢弃过时样本，不回放高频 IMU/LiDAR
积压。每条被选消息保留原始 CDR payload 和源时间戳：
每次以 RGB frame 为快照锚点，先选取最近的 Depth frame、LiDAR 和 Odom，
再以已选 LiDAR 的时间戳选取最近 IMU。因此 `20 ms` 门槛限制的是
LiDAR/IMU 运动对齐，不会错误要求高频 IMU 同时贴近 RGB 锚点。

| 数据 | topic | ROS type / QoS |
| --- | --- | --- |
| LiDAR | `/ubuntu/navigation/lidar` | `PointCloud2` |
| IMU | `/ubuntu/navigation/imu` | `Imu` |
| RGB frame | `/ubuntu/camera/rgb_frame` | `UInt8MultiArray`; `phanthy.sensor.camera_rgb_frame.v1` |
| Depth frame | `/ubuntu/camera/depth_frame` | `UInt8MultiArray`; `phanthy.sensor.camera_depth_frame.v1` |
| Odom | `/ubuntu/navigation/odom` | `Odometry` |

启用数采时 RGB frame 和 Depth frame 都是条件必需输入。Depth frame 的 PSE1 元数据已
包含 Z16 编码、`depth_scale_m`、深度/RGB 内参、`depth_to_rgb` 和
LiDAR-to-camera 外参，因此不再需要单独的 `CameraInfo` topic。任一 PSE1
封装、尺寸、尺度、标定或时间戳校验失败，该帧不会进入对齐快照。

RGB frame 直接沿用 Driver 已发布的
`PSE1 + uint32_le(metadata_size) + uint32_le(payload_size) + JSON + JPEG`
封装（`application/vnd.phanthy.sensor-envelope.v1`），不引入卡片私有
wire format。JSON 必须逐帧携带 `header.stamp_ns`、
`timing.source_stamp_ns`、`timing.driver_receive_stamp_ns`、尺寸、
`frame_id`、`calibration_id`、内参/畸变以及 Driver 标定的
LiDAR-to-camera 外参。该外参矩阵为 row-major 4x4 齐次变换，
按 `target_from_source` 把 LiDAR 点变换到相机 optical frame。

Driver 不需要另外发布 `T_base_camera`。ActuCore 使用与 G1 实时
adapter 相同的 `base_link -> livox_frame` 外参，与 Driver 的
`livox_frame -> camera` 标定组合出离线投影所需的
`T_base_camera`。
畸变模型接受 `none`、`plumb_bob` / `brown_conrady`、
`rational_polynomial` 和 Driver 实际发布的
`realsense_inverse_brown_conrady`；逆 Brown 模型按 RealSense 语义迭代求
投影坐标，不会忽略畸变继续投影。
因此不再依赖独立 `CameraInfo` topic，也不在 ActuCore 伪造或推断
标定。同一 `/ubuntu/camera/rgb_frame` PSE1 输入也由语义模块
解码 JPEG 使用，Canvas 不再需要第二路 RGB 连线。

MCAP 只保存通过对齐门槛的 1 Hz 快照。在线状态不会改写或伪造时间戳，而是
基于原始 source stamp 做有界的软件对齐诊断：检查每路时间戳覆盖率和单调性，
并计算 RGB/Depth、LiDAR/IMU、RGB/LiDAR、RGB/Odom、
Depth/LiDAR 的最近邻时间偏差。P95 门槛分别是 `150 / 20 / 60 / 120 /
60 ms`，对应 G1 当前 10 Hz RGB、5 Hz Depth、10 Hz LiDAR、高频 IMU 和约
5 Hz Odom 的异步采样上限；它们是数据健康门槛，不是硬件同步声明。超限、
时间戳缺失或倒序会令状态进入 `degraded`，但不会中断导航。这是 ROS
system-time 时钟域的软件对齐证据，不等同于 PTP、外部触发或硬件帧同步。

录制目录按 `ubuntu/YYYY-MM-DD/<session_id>` 分层。录制期间目录名带
`.partial`；Canvas 正常停止、rosbag2 完成 flush 且 receipt 写入后才原子改为
最终目录。异常退出会保留 `.partial`，不会冒充完整数据。

Canvas 公共 `/ubuntu/navigation/fast_livo2/collection_preview` 输出图像；它与旧的
`collection_status` String topic 使用不同名称，避免 ROS topic 在版本升级后
同名异型。该图像端口不再输出日志式 JSON，而是输出最新一帧已对齐的 RGB 预览：
顶部显示采集帧号，图中按现有离线
算法投影 LiDAR 可见最近点，并在对应像素标出障碍物距离。预览只消费采样器已经
生成的 1 Hz 录制快照，后台采用 latest-only 队列，不会提高录制频率或阻塞
FAST-LIVO2/Nav2 回调。

停止智能控制并完成 MCAP 原子收口后，同一 `collection_status` 图像端口自动
切换为本体离线导出进度卡；显示 session、阶段、已处理/总帧数、百分比、
RGB/JSON、Depth PNG、LiDAR PCD 产物数以及暂停或失败原因。`complete`、
`degraded`、`error` 终态采用 transient-local 保留，下一次采集产生新预览后
再被替换，不需要额外动作或新的 Canvas 连接点。

完整机器诊断继续保留在卡片原有
`/ubuntu/navigation/fast_livo2/status` 的 `collection` 字段，并额外发布到内部
`/ubuntu/navigation/fast_livo2/collection_status_json`，包含：

- 录制 `state`: `disabled | starting | recording | degraded | error`；
- 当前 session、落盘目录和 rosbag PID；
- 每路 topic 的原始到达 `count`、1 Hz 对齐快照 `sampled_count`、
  最近接收年龄、源时间戳和 publisher 数量；
- `missing_sources`、`stale_sources` 与 `failure_reason`；
- `sampling.emitted_count`、`sampling.rejections` 和
  `sampling.last_rejection_reason`，用于区分缺源与具体时间对齐失败；
- `time_alignment.alignment_ready`、每路时间戳覆盖率/单调性，以及五组
  `nearest_skew_ms` 的 P50、P95、最大值和阈值；
- 上一次停止时的 receipt 和最终目录；
- receipt 中每路 `sampled_count` / `recorded_count` /
  `recording_coverage`，用于直接暴露 rosbag 实际落盘数量与卡片发出的
  快照数量是否一致；即使各路采样数都为 0，空 MCAP 也会以
  `recording_empty` 明确标记为不健康；
- `postprocess.state/stage`: `queued` / `scanning` / `processing` / `paused` /
  `finalizing` / `complete` / `degraded` / `error`；
- `processed_images`、`total_images`、`generated_lidar_frames`、
  `generated_depth_frames`、`percent`、`paused_reason` 与 `failure_reason`。

Canvas `stop` 只有在算法和采集都确认停止后才返回
`state/status=idle`；任一收口失败时返回顶层
`error_code=canvas_stop_failed`并保留卡片控制对象以便重试。仍必须同时检查
`collection_stop_result.status`、`receipt.storage_complete` 和 `receipt.state`：
只有 `storage_complete=true` 才会将 `.partial` 改名为最终目录；
`state=degraded` 表示存储完整但存在缺失/过期源，仍应按降级验收。
已经原子落盘的 receipt 会立即交给后台处理器，即使同一次 Canvas `stop`
还因地图 manifest 等独立问题返回失败，也不会阻断数据集生成。
采集启停在内部串行；rosbag 启动早退时会保留 `.partial`
并写入 `state=failed/storage_complete=false` receipt，不会冒充完整 session。

完整标注依赖 Driver 发布上述 RGB frame 与 Depth frame topic。缺失或封装校验失败时，
原始 MCAP 仍按实际收到的数据收口，但状态明确显示
`missing_sources` 或解码失败；不会回退到无标定的旧 RGB/Depth。

停止 receipt 保存 `time_alignment`。MCAP 完成并原子收口后，ActuCore
父进程的后台 worker 直接在 G1 本体生成可交付的 `derived/`，用户不需要安装
MCAP/ROS 工具再手工解包：

- `rgb/frame-XXXXXXXX.jpg`：原始 JPEG；
- `depth/depth-<source_stamp_ns>.png`：与 RGB 最近邻匹配的 16-bit 灰度
  Z16 深度图；像素保留 Driver 原始无符号整数，米制距离为
  `pixel * frame.depth_parameters.depth_scale_m`，同一 Depth 帧按源
  时间戳去重；
- `lidar/lidar-<source_stamp_ns>.pcd`：与图片配对的 binary PCD，XYZ 为
  little-endian float32 米；同一雷达帧被多张图片复用时只保存一份；
- `frames/frame-XXXXXXXX.json`：逐图记录图片 ID/路径、Depth ID/路径与
  `depth_scale_m`、雷达 ID/路径、各路
  source timestamp、相机宽高与 `fx/fy/cx/cy`、像素单位等效焦距
  `sqrt(fx*fy)`、畸变参数、障碍物 ID、相机光学坐标系最近 LiDAR 三维点、
  LiDAR 距离真值与失败原因；
- `tracks.json`：session 内稳定的 `obs-XXXXXX` 轨迹 ID；
- `manifest.json`：总图片/Depth/雷达帧数、有效/无效帧数、产物格式与
  最终状态；
- `postprocess.json`：可读回的任务进度/错误日志。

这里的“等效焦距”明确使用像素单位；没有传感器物理尺寸时不会伪造毫米焦距。
“距离真值”来自配对 LiDAR 中投影到该图片且属于障碍物的最近可见三维点，
不是单目估计值。原始 MCAP 继续保留完整 ROS 消息，作为可追溯源数据。

流程使用 IMU 重力方向移除地面、LiDAR 点云做几何聚类，再通过
标定外参投影到每张 RGB；Depth frame 同时保留在原始 MCAP 和可直接查看的
16-bit PNG 产物中，当前尚不参与
障碍物最近点真值计算。卡片或
导航 runtime 活跃时 worker 自动暂停，卡片停止后继续，避免与定位/
规划抢占 G1 CPU。ActuCore 重启后会重新发现已完成但尚无
`derived/manifest.json` 的 session 并继续生成。最终目录只在 manifest
写入后由 `derived.partial` 原子改名。

需求依据见飞书文档
[机器人数据采集方案](https://my.feishu.cn/wiki/EWN2wVk5miId63kQ9zScxQMqnAg)。
本阶段不包含自动清理、容量配额、语义分类或 Canvas 数据浏览。

## 统一卡片连线与构建

Canvas 把 Driver `navigation_lidar`、`navigation_imu` 和 PSE1 相机 `rgb` 接到公开
`ControlledSemanticSpatial` 卡片；语义导航和数采共用这一路 RGB，启用完整数采时再连接可选的
`depth_frame`。`livo_odom`、registered cloud、obstacle map 都在同一容器内
交给 planning/semantic 模块。`collection_status` 是唯一新增的只读公共诊断
输出，用于直接查看数采是否正常及失败原因。

本模块随统一镜像构建，版本和许可证见
[`../runtime/FAST_LIVO2_THIRD_PARTY.md`](../runtime/FAST_LIVO2_THIRD_PARTY.md)。
统一构建和单容器部署命令见 [`../README.md`](../README.md)。
