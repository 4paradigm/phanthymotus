# Navigation 内部 FAST-LIVO2 模块

该模块负责统一 `controlled_semantic_spatial` 卡片内的会话级 LiDAR-Inertial 建图、里程计、
运动补偿点云、标准 TF 和 Canvas 地图可视化。它不再单独注册 MCP 卡片，
FAST-LIVO2 及 adapter 由同一个 Perception 容器内的 `NavigationRuntime`
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
  `/ubuntu/lidar/cloud` envelope，也不在 Perception 内复制设备 SDK。
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
adapter 对接收 age 仍使用 500 ms 上限，对源时间戳另保留 50 ms 有界
调度抖动；约 0.51 s 的边界帧可接受，超过 0.55 s 仍拒绝。

## 输出

| port                | topic                                             | frame              | 用途                            |
| ------------------- | ------------------------------------------------- | ------------------ | ----------------------------- |
| `livo_odom`         | `/ubuntu/navigation/odom`                         | `map -> base_link` | Nav2 位姿与速度反馈                  |
| `registered_cloud`  | `/ubuntu/navigation/cloud_registered`             | `map`              | Nav2 实时障碍层输入                        |
| `static_map`        | `/ubuntu/navigation/static_map`                   | `map`              | 运动门控并经多帧确认的完整二维 OccupancyGrid，供 Nav2 StaticLayer |
| `obstacle_map`      | `/ubuntu/navigation/obstacle_map`                 | `map`              | 运动门控静态障碍的兼容诊断投影                |
| `map_view`          | `/ubuntu/navigation/fast_livo2/map_view`          | `map`              | Canvas 静态体素地图、最新阈值调试帧和机器人位置           |
| `status`            | `/ubuntu/navigation/fast_livo2/status`            | JSON               | 算法进程、输入 freshness、frame 和产物状态 |
| `collection_status` | `/ubuntu/navigation/fast_livo2/collection_status` | JSON               | 录制进程、每路计数、丢失/过期源与停止回执         |

adapter 先按 `0.10 m` 体素把同一帧命中去重；导航高度带内的同一体素默认
需要连续 8 个扫描支持才进入稳定静态图，候选点默认 1 秒过期。二维连通
分量会按不超过 `1.0 m` 的固定空间片分解后经过运动门。默认完整观察窗为
`max(0.8, 0.03 / 0.03, sqrt(2) * 0.10 / 0.03 + 0.40)`，即约
`5.114 s`；最后一项保证以阈值速度运动的返回即使从体素最不利位置进入，
也必须跨过完整体素对角线。轨迹和动态格历史按 20 Hz 时间桶保存紧凑二进制
摘要；完整观察窗最多 30 秒，所有轨迹、栅格、样本及近期动态键合计最多
1,000,000 个 history units。同一时间桶只更新几何，不滑动其时间戳；高频输入
和短命目标都不能扩大历史，预算饱和时保持隔离并 fail closed。连续两次成立
即视为动态，隔离当前空间片并撤销其近期
候选和已确认静态证据。动态分量停止 `1.5 s` 后才重新开始静态确认。稀疏到
单格的分量同样受运动门约束；长墙会被分片，而不是让与墙相连的移动物体
绕过运动门。判断同时比较重叠栅格内的原始点质心，因此墙面只改变可见子集
不会仅因整体质心偏移就被整片删除。

点云只和源时间戳相差不超过 `50 ms` 的 canonical odom 位姿配对；不匹配帧
跳过静态证据更新并进入 diagnostics 计数，不能拿最新位姿清除旧点。已确认点
不会因暂时离开视野而删除，但同一格连续 3 帧被射线观察为空闲后会移除。
上述门控不替代人员语义分割：静止人员最终仍可能进入静态图；人与墙相连时
空间分片可避免整块大分量直接绕过，但极端稀疏、遮挡或传感器噪声下仍不能
提供语义级人员识别保证。当前 registered cloud 始终独立进入 Nav2
实时障碍层，所以被静态门隔离的移动物体仍参与即时避障。Canvas 地图不再按
80,000 点截断；稳定部分显示多帧确认结果，高度带外仅叠加新鲜的最新扫描，
用于蓝色/粉色阈值调试而不进入累计静态图。Agent Core 在显示层
把同为 `map` frame 的 Nav2 `/plan` 叠加为绿色路径和橙色终点，不改变
`map_view` 点数组语义，也不让 FAST-LIVO2 依赖 Nav2。旧图加载后在
重定位成功前不显示伪造的机器人位姿。地图卡片支持三维
浏览和正上方二维投影切换；二维模式只是同一三维点云的平面显示，不是另存
一份 occupancy grid。绿色机器人箭头沿 canonical `base_link` 的 `+X` 前向，
并保持 ROS `map` frame 的 yaw 正方向。它是监控视图，不是
可重定位地图格式。原始 PCD 分片保存在宿主机
`/opt/phanthy-motus/data/fast_livo2/maps`；`stop_mapping` 还会原子写入
`maps/static/<map>-*.static.pcd`，保存导航高度带内经运动门控和多帧确认的静态点。

live PointCloud2 在复制数据区前同时检查 `width * height <= 200,000` 和
`data <= 64 MiB`；静态候选证据及加载后的 confirmed static map 各自也以
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

## Actions

| action          | 参数                                                                                    | 语义                                                    |
| --------------- | ------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| `start_mapping` | `map_name`                                                                            | 清空 Canvas 会话图并启动一个新的 FAST-LIVO2 进程                    |
| `stop_mapping`  | 无                                                                                     | 先检查静态证据，再 `SIGINT` 停止算法；分别保存 raw PCD 与 confirmed static PCD 后原子写 session manifest；保存失败可用同一 action 重试 |
| `load_map`      | `map_name`                                                                            | 先校验新旧 manifest/PCD，再串行替换定位前端，失败时尝试回滚旧图 |
| `relocalize`    | `initial_x`, `initial_y`, `initial_z`, `initial_yaw`, `search_xy_m`, `search_yaw_rad` | 以操作者给定位姿为中心做有界二维 scan-to-map 匹配                       |

`map_name` 只允许 `A-Z a-z 0-9 _ . -`，最长 64 字符。停止 Canvas 时若仍在
建图，卡片会先执行 `stop_mapping` 再释放 ROS backend。直接杀死容器或 ROS
进程不具备完成跨进程静态图事务的条件，只会受控终止算法并明确留下无成功
manifest 的产物；需要可加载地图时必须先获得 `stop_mapping.status=saved`。
FAST-LIVO2 在受控 `SIGINT` 收口期间可因上游 C++ 析构路径返回
`-SIGABRT (-6)`。supervisor 仅在自己已发出停止信号的路径将 `0/-SIGINT/-SIGABRT`
视为受控停止，并在回执保留原始 `algorithm_return_code`；运行中自行 `-6`
或其他退出码仍按 `algorithm_exited/algorithm_stop_failed` fail closed。

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
这里的加载和重定位由同容器 Perception adapter 实现：它是依赖人工
初值的有界匹配，不是无初值的全局搜索，也不在定位成功后持续做全局
闭环校正。匹配分数不足、点数不足、数据过期、manifest/PCD 损坏时全部
fail closed。

当前有界匹配使用 `0.20 m` 二维体素，旧图和当前 scan 都至少需要
40 个有效障碍体素，scan 最多参与 1,200 点，最低匹配比为 `0.20`。
成功回执会返回 `match_ratio`、`matched_points` 和 `evaluated_points`；
`status=relocalized` 只证明本次有界匹配过关，不等同于长时间全局定位
可靠性已验收。

### 生命周期与大图读取

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

数据采集属于统一 `controlled_semantic_spatial` 卡片，不增加单独卡片，也不增加
`start_recording` / `stop_recording` 等公开 action。创建卡片时配置：

| 配置 | 默认值 | 语义 |
| --- | --- | --- |
| `collection_enabled` | `false` | Canvas 启动该卡片时自动开始采集，停止卡片时自动收口 |
| `collection_directory` | `/opt/phanthy-motus/data/fast_livo2/recordings` | Perception 容器内持久化根目录；只能配置为该挂载目录或其子目录 |

启用后，同容器 adapter 使用 ROS 2 原生 rosbag2 MCAP 后端记录以下原始 topic，
保留消息自身的 CDR payload、`header.stamp`、`frame_id` 和 encoding：

| 数据 | topic | ROS type / QoS |
| --- | --- | --- |
| LiDAR | `/ubuntu/navigation/lidar` | `PointCloud2`; `RELIABLE + KEEP_LAST(2) + VOLATILE` |
| IMU | `/ubuntu/navigation/imu` | `Imu`; `RELIABLE + KEEP_LAST(200) + VOLATILE` |
| RGB | `/ubuntu/camera/rgb` | `CompressedImage`; `BEST_EFFORT + KEEP_LAST(4) + VOLATILE` |
| Depth | `/ubuntu/camera/depth` | `Image`; `BEST_EFFORT + KEEP_LAST(4) + VOLATILE` |
| CameraInfo | `/ubuntu/camera/camera_info` | `CameraInfo`; `BEST_EFFORT + KEEP_LAST(4) + VOLATILE` |

MCAP 保留实际收到消息的 CDR payload，但 `BEST_EFFORT` 源在网络或系统
负载下允许丢帧。`collection_status.sources[*].count` 和 stale 状态是
采集健康证据，不是跨传感器帧同步或数据完备性证明。

录制目录按 `ubuntu/YYYY-MM-DD/<session_id>` 分层。录制期间目录名带
`.partial`；Canvas 正常停止、rosbag2 完成 flush 且 receipt 写入后才原子改为
最终目录。异常退出会保留 `.partial`，不会冒充完整数据。

卡片原有 `/ubuntu/navigation/fast_livo2/status` 的 `collection` 字段，以及
独立的 `/ubuntu/navigation/fast_livo2/collection_status` 数据流，都会显示：

- `state`: `disabled | starting | recording | degraded | error`；
- 当前 session、落盘目录和 rosbag PID；
- 每路 topic 的消息计数、最近接收年龄、源时间戳和 publisher 数量；
- `missing_sources`、`stale_sources` 与 `failure_reason`；
- 上一次停止时的 receipt 和最终目录。

Canvas `stop` 只有在算法和采集都确认停止后才返回
`state/status=idle`；任一收口失败时返回顶层
`error_code=canvas_stop_failed`并保留卡片控制对象以便重试。仍必须同时检查
`collection_stop_result.status`、`receipt.storage_complete` 和 `receipt.state`：
只有 `storage_complete=true` 才会将 `.partial` 改名为最终目录；
`state=degraded` 表示存储完整但存在缺失/过期源，仍应按降级验收。
采集启停在内部串行；rosbag 启动早退时会保留 `.partial`
并写入 `state=failed/storage_complete=false` receipt，不会冒充完整 session。

当前 G1 Driver 代码已经发布 RGB 与 depth，但尚未发布 ROS
`sensor_msgs/msg/CameraInfo`。因此在 Driver 补齐该真实 producer 前，启用采集
会继续保存其余四路数据，同时状态明确显示
`missing_sources:camera_info`，不会生成或伪造内参。此能力只记录数据，
不会启用 FAST-LIVO2 图像处理，也不会发送 Driver、Nav2 或机器人运动命令。

本阶段不包含离线回放、派生标注、自动清理、容量配额或 Canvas 数据浏览；
这些是后续独立子任务。

## 统一卡片连线与构建

Canvas 只把 Driver `navigation_lidar`、`navigation_imu` 接到公开
`controlled_semantic_spatial` 卡片。`livo_odom`、registered cloud、obstacle map
都在同一容器内交给 planning/semantic 模块；collection status 也只保留为内部
诊断 topic。它们均不再暴露成 Canvas 公共连线端口，公共定位状态统一通过
`status` 输出查看。

本模块随统一镜像构建，版本和许可证见
[`../runtime/FAST_LIVO2_THIRD_PARTY.md`](../runtime/FAST_LIVO2_THIRD_PARTY.md)。
统一构建和单容器部署命令见 [`../README.md`](../README.md)。
