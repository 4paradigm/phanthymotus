# controlled_semantic_spatial 内部 FAST-LIVO2 模块

该模块负责统一 `controlled_semantic_spatial` 卡片内的会话级 LiDAR-Inertial 建图、里程计、
运动补偿点云、标准 TF 和 Canvas 地图可视化。它不再单独注册 MCP 卡片，
FAST-LIVO2 及 adapter 由同一个 ActuCore 容器内的 `ControlledSemanticSpatialRuntime`
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
  位姿。adapter 从 Driver 的 `/tf_static` 解析 `base_link -> sensor_frame` 计算
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
| `lidar` | `/ubuntu/navigation/lidar` | `sensor_msgs/msg/PointCloud2`; `RELIABLE + KEEP_LAST(2)` | REP-103 `sensor_frame`，ROS system time，逐点 `timestamp=float64` 绝对纳秒 |
| `imu` | `/ubuntu/navigation/imu` | `sensor_msgs/msg/Imu`; `RELIABLE + KEEP_LAST(200)` | 与点云同 `sensor_frame`、同一源时钟域，约 200 Hz |

这两路输入来自既有 Driver `navigation_sensors` sensor cards。缺失、过期或
frame 不符时同容器 adapter 不发布伪造的 canonical odom/cloud。
FAST-LIVO2 内部订阅队列默认保留最新 `2` 帧 LiDAR 和 `400` 帧 IMU；可在
Canvas 停止卡片后通过 `lidar_qos_depth`（1..32）与 `imu_qos_depth`
（1..4000）调整，下次启动卡片生效。该参数在启动算法子进程前传入 ROS，
因此首次升级需要重建 navigation base，之后调值只需重启卡片，不再重建镜像。
FAST-LIVO2 使用与 LiDAR 同 frame 的 IMU 做重力对齐，因此整套传感器
相对机身的小角度倾斜不会直接把地图永久建斜；adapter 再用实测
`base_link -> sensor_frame` 静态 TF 恢复机器人机身位姿。若实际安装角度发生变化、
LiDAR 与 IMU frame 不一致，或外参仍沿用旧值，影响会同时出现在机器人箭头、
高度带过滤、重定位和 costmap，必须重新标定外参，不能靠放宽匹配阈值掩盖。
adapter 对 FAST-LIVO2 原始 odom/cloud 使用
`BEST_EFFORT + KEEP_LAST(1)`，慢回调只保留最新样本，不把旧帧排成约
0.5 秒的内部积压。Nav2 readiness 容许最多 `0.8 s` 接收调度抖动，
并每次从最新 header stamp 重算源数据 age；源 age 超过 `1.0 s`
仍 fail closed，不会因为缓存的旧 age 伪装成新鲜数据。live
PointCloud2 的 XYZ 解码、刚体变换、高度过滤和 float32 输出打包走 NumPy
批处理；边界、字段、有限值、点数和字节上限仍在发布前 fail closed。
输入方必须保证每帧 `timestamp` 单调、点时间位于 header 后 `200 ms` 内且具有
非零跨度。mapper 的运行诊断和 adapter 只检查实际处理的扫描跨度及 frame，
不会逐点验证时间戳的单调性或其相对 header 的范围；独立的
`absolute_point_time_span_ms` 辅助函数不在运行路径中，不能把它的单元测试当成
输入校验的运行证据。adapter 只从 TF2 获取 `base_link -> sensor_frame`，不再保存本体
专用六自由度外参。首个 raw odom 离会话原点超过 `2 m`、相邻线速度超过
`10 m/s` 或角速度超过 `4π rad/s` 时进入 `raw_odom_discontinuity`，并在当前
地图会话内锁存；仅 reset、新建/加载地图或成功重定位会解除。重复时间戳等
非几何异常仍可在连续三个健康样本后自动恢复；普通输入 freshness 断流也
仍在新有效帧到来后恢复。frame、TF、点时间或 odom 健康任一失败时，registered
cloud、静态累计和 Nav2 输入全部 fail-closed。

## 输出

| port                | topic                                             | frame              | 用途                            |
| ------------------- | ------------------------------------------------- | ------------------ | ----------------------------- |
| `livo_odom`         | `/ubuntu/navigation/odom`                         | `map -> base_link` | Nav2 位姿与速度反馈                  |
| `registered_cloud`  | `/ubuntu/navigation/cloud_registered`             | `map`              | Nav2 实时障碍层输入                        |
| `static_map`        | `/ubuntu/navigation/static_map`                   | `map`              | 直接累计体素生成的二维 OccupancyGrid，供 Nav2 StaticLayer |
| `obstacle_map`      | `/ubuntu/navigation/obstacle_map`                 | `map`              | 累计静态障碍的兼容诊断投影                |
| `map_view`          | `/ubuntu/navigation/fast_livo2/map_view`          | `map`              | Canvas 累计静态点、最新实时点和机器人位置            |
| `status`            | `/ubuntu/navigation/fast_livo2/status`            | JSON               | 算法进程、输入 freshness、frame 和产物状态 |

adapter 先按 `0.10 m` 体素把同一帧命中去重；导航高度带和有效距离内的体素
首次出现即写入累计静态图，不等待多帧确认、不跟踪动态分量，也不通过后续
自由射线删除。点云会先等待 odom/TF 历史从前后包围其源时间戳，再选择
时间差不超过 `50 ms` 的 canonical odom 位姿配对；配对成功前不会发布给
Nav2，也不会进入累计。不匹配帧进入 diagnostics 计数，不能拿最新位姿
拼接旧点。
直接累计可以恢复墙面和稀疏结构密度，但不提供人员语义分割或动态物体清除，
建图时经过的人可能固化进地图，现场应尽量保持环境静止。

高度阈值是 canonical `map` frame 的 Z，不是雷达安装高度。默认从
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
障碍点挤出前端上限。地图点主体每秒编码一次并缓存，编码完成后在同一定时
回调中更新机器人 `x/y/yaw` 的固定 12 字节头部并立即发布，不再用第二个过期
定时器争抢同一回调组；DDS 只保留最新一帧，避免大帧发布堆积后抢占
odom/registered cloud 回调。位姿刷新不再重复编码整张地图。主线 Agent Core 不包含本 PR 原有的高度着色、
路径叠加和二维/三维切换增强，不应假定这些视图可用。旧图加载后首次重定位失败时，
不会发布未确认候选对应的 TF、odom/cloud 或规划输入。`map_view` 是监控数据，
不是可重定位地图格式。原始 PCD 分片保存在宿主机
`/opt/phanthy-motus/data/navigation/fast_livo2/maps`；`stop_mapping` 还会原子写入
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
`map_view_pose_refresh_hz=1`。Canvas 帧最多携带 40,000 点，避免在机载计算平台上高频
重复序列化近 1 MiB 帧并阻塞 odom/registered cloud；这些字段用于区分
传感器/TF 等待、点云计算、DDS 发布和 Canvas 编码延迟，不改变 500 ms
freshness 门禁。

## Actions

| action          | 参数                                                                                    | 语义                                                    |
| --------------- | ------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| `start_mapping` | `map_name`                                                                            | 清空 Canvas 会话图并启动一个新的 FAST-LIVO2 进程                    |
| `stop_mapping`  | 无                                                                                     | 停止当前建图 session；静态证据足够时同步落盘 raw/static PCD 和 manifest，否则返回 `status=stopped, map_saved=false` 及 warning |
| `load_map`      | `map_name`                                                                            | 先校验新旧 manifest/PCD，再串行替换定位前端，失败时尝试回滚旧图 |
| `relocalize`    | `initial_x`, `initial_y`, `initial_z`, `initial_yaw`, `search_xy_m`, `search_yaw_rad` | 以操作者给定位姿为中心做有界二维 scan-to-map 匹配                       |

重定位只有在匹配率至少为 0.35、且最优候选未贴住 XY/yaw 搜索边界时才提交新的
`map -> base_link` 对齐；否则明确拒绝并要求扩大搜索范围或修正初始位姿。失败
回执仍返回 `candidate_pose`、匹配率和拒绝原因，Canvas `map_view` 显示该未确认
候选与当前点云，方便操作者修正初值；预览不会发布 TF、canonical odom/cloud
或启动 Nav2。重复重定位提交时会先丢弃旧对齐下生成的实时点和 Canvas 缓存，
等待新对齐下的下一帧，
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
不会先停算法再留下无法恢复的 `map_artifact_missing`。参数写入使用不经 daemon 的
直连 ROS 图发现并等待 5 秒，避免首次 stop 时 daemon 尚未发现 `/laserMapping`。

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

当前有界匹配以 confirmed static PCD 为参考，缓存最近 `2 s`、最多 20 帧，
每帧最多保留 2,000 点后合并；最终使用 `0.20 m` 二维体素去重并最多评分
1,200 点。评分按点到参考体素的距离连续衰减，并对偏离人工初值的候选施加
轻微先验代价。旧图和当前数据都至少需要 40 个有效障碍体素，最低匹配比为
`0.35`。`initial_z` 只决定成功后垂直对齐，二维 XY/yaw 评分不再用它平移
当前点云的 Z 高度带。成功回执会返回 `match_ratio`、`matched_points`、
`evaluated_points`、输入帧数和输入点数；
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
- `stop_mapping` 总会结束当前 session。当前 session 缺少新鲜静态图诊断，
  或不足 40 个 confirmed 点时，不生成可加载地图，返回
  `status=stopped, map_saved=false` 和明确 warning，随后可直接开始新的建图。算法
  已正常停止并开始持久化后进入 `finalizing`，静态 PCD 或 manifest 写失败不会清除会话，
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

## 运行诊断

FAST-LIVO2 adapter 在拒绝 LiDAR、IMU、odom 或 registered cloud 时发布
`/ubuntu/navigation/fast_livo2/sensor_rejection`。故障快照录制独立于
已移除的数据采集功能。

## 统一卡片连线与构建

Canvas 把 Driver `lidar_cloud` 的标准 PointCloud2、`lidar_imu` 和
`camera_rgb` 的 RGB PSE1 输出接到公开
`controlled_semantic_spatial` 卡片；RGB 用于语义导航。原机身 `imu` 不能替代 `lidar_imu`。
`livo_odom`、registered cloud、obstacle map 都在同一容器内
交给 planning/semantic 模块。机器诊断由内部 topic 提供，
统一卡片公开 `map_view`、旧格式 `motion_sequence` 和 Navi 格式 `control_velocity`。

本模块随统一镜像构建，版本和许可证见
[`../runtime/FAST_LIVO2_THIRD_PARTY.md`](../runtime/FAST_LIVO2_THIRD_PARTY.md)。
统一构建和单容器部署命令见 [`../README.md`](../README.md)。
