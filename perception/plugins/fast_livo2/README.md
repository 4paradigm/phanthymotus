# FAST-LIVO2 Perception 卡片

`fast_livo2` 是单实例 `processor` 卡片，负责会话级 LiDAR-Inertial
建图、里程计、运动补偿点云、标准 TF 和 Canvas 地图可视化。算法运行在
独立 companion 容器中，但卡片、生命周期、配置和对外合同都归
Perception Bundle；Driver 只提供已冻结的标准传感器输入。

## 边界

```text
Driver navigation_sensors
  /ubuntu/navigation/lidar_fast_livo  PointCloud2, 10 Hz
  /ubuntu/navigation/imu              Imu, 200 Hz
                    |
                    v
            FAST-LIVO2 card
              |-- /ubuntu/navigation/odom
              |-- /ubuntu/navigation/cloud_registered
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
| `lidar` | `/ubuntu/navigation/lidar_fast_livo` | `sensor_msgs/msg/PointCloud2`; `RELIABLE + KEEP_LAST(2)` | MID360 `livox_frame`，ROS system time，逐点 offset 保留 |
| `imu` | `/ubuntu/navigation/imu` | `sensor_msgs/msg/Imu`; `RELIABLE + KEEP_LAST(200)` | 与点云相同时钟域和安装旋转，约 200 Hz |

这两路输入来自既有 Driver `navigation_sensors` sensor cards。缺失、过期或
frame 不符时 companion 不发布伪造的 canonical odom/cloud。

## 输出

| port | topic | frame | 用途 |
| --- | --- | --- | --- |
| `livo_odom` | `/ubuntu/navigation/odom` | `map -> base_link` | Nav2 位姿与速度反馈 |
| `registered_cloud` | `/ubuntu/navigation/cloud_registered` | `map` | Nav2 rolling costmap 障碍输入 |
| `obstacle_map` | `/ubuntu/navigation/obstacle_map` | `map` | 去除地板/天花板后累计投影的 Nav2 全局二维障碍输入 |
| `map_view` | `/ubuntu/navigation/fast_livo2/map_view` | `map` | Canvas 体素化累计地图和机器人位置 |
| `status` | `/ubuntu/navigation/fast_livo2/status` | JSON | 算法进程、输入 freshness、frame 和产物状态 |

Canvas 地图最多保留 80,000 个 `0.10 m` 体素占用点；Agent Core 在显示层
把同为 `map` frame 的 Nav2 `/plan` 叠加为绿色路径和橙色终点，不改变
`map_view` wire payload，也不让 FAST-LIVO2 依赖 Nav2。旧图加载后在
重定位成功前不显示伪造的机器人位姿。地图卡片支持三维
浏览和正上方二维投影切换；二维模式只是同一三维点云的平面显示，不是另存
一份 occupancy grid。绿色机器人箭头沿 canonical `base_link` 的 `+X` 前向，
并保持 ROS `map` frame 的 yaw 正方向。它是监控视图，不是
可重定位地图格式。原始 PCD 分片保存在宿主机
`/opt/phanthy-motus/data/fast_livo2/maps`。

Nav2 使用的 `obstacle_map` 不等同于 Canvas 三维渲染数据。adapter 按当前
上海 G1 室内点云分布保留相对雷达原点 `z=-1.15…+0.80 m` 的高度带，排除
约 `z=-1.3 m` 的地板和约 `z=+1.7 m` 的天花板，再按 XY 体素去重投影到
`z=0`。由于 MID360 位于机器人上部，这个范围向下覆盖更多、向上覆盖更少，
用于保留桌腿、椅子、箱体、人体和墙面等二维碰撞障碍。

## Actions

| action          | 参数                                                                                    | 语义                                                    |
| --------------- | ------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| `start_mapping` | `map_name`                                                                            | 清空 Canvas 会话图并启动一个新的 FAST-LIVO2 进程                    |
| `stop_mapping`  | 无                                                                                     | `SIGINT` 停止算法，等待尾段保存并写 session manifest               |
| `load_map`      | `map_name`                                                                            | 读取 `stop_mapping` 生成的 manifest 和 PCD，启动不再写 PCD 的新定位前端 |
| `relocalize`    | `initial_x`, `initial_y`, `initial_z`, `initial_yaw`, `search_xy_m`, `search_yaw_rad` | 以操作者给定位姿为中心做有界二维 scan-to-map 匹配                       |
| `unload_map`    | 无                                                                                     | 停止定位前端并卸载旧图                                           |

`map_name` 只允许 `A-Z a-z 0-9 _ . -`，最长 64 字符。停止 Canvas 时若仍在
建图，卡片会先执行 `stop_mapping` 再释放 ROS backend。

重定位的操作顺序为：

1. 执行 `load_map(map_name)`；只接受该卡片 `stop_mapping` 生成的
   `phanthy.navigation.fast_livo2_map_session.v1` manifest，不猜测孤立 PCD 的归属。
2. 机器人保持静止，等待新前端产生新鲜 odom 和 registered cloud。
3. 在 Canvas 中给出机器人在旧图上的近似 `x/y/z/yaw`，执行
   `relocalize`。默认只搜索位置±1.0 m、航向±0.35 rad；允许范围分别是
   `0.1–3.0 m` 和 `0.05–π/2 rad`。
4. 返回 `status=relocalized` 后，卡片才发布旧图 frame 下的
   `map -> base_link`、registered cloud 和障碍图；此时再启动 Nav2 任务。
5. 完成后执行 `unload_map`。

当前锁定的 FAST-LIVO2 本身仍然没有 PCD 加载、`/initialpose` 或全局回环。
这里的加载和重定位由 Perception companion adapter 实现：它是依赖人工
初值的有界匹配，不是无初值的全局搜索，也不在定位成功后持续做全局
闭环校正。匹配分数不足、点数不足、数据过期、manifest/PCD 损坏时全部
fail closed。

## 自动数据采集

数据采集属于同一张 `fast_livo2` 卡片，不增加单独卡片，也不增加
`start_recording` / `stop_recording` 等公开 action。创建卡片时配置：

| 配置 | 默认值 | 语义 |
| --- | --- | --- |
| `collection_enabled` | `false` | Canvas 启动该卡片时自动开始采集，停止卡片时自动收口 |
| `collection_directory` | `/opt/phanthy-motus/data/fast_livo2/recordings` | companion 内持久化根目录；只能配置为该挂载目录或其子目录 |

启用后，companion 使用 ROS 2 原生 rosbag2 MCAP 后端记录以下原始 topic，
保留消息自身的 CDR payload、`header.stamp`、`frame_id` 和 encoding：

| 数据 | topic | ROS type / QoS |
| --- | --- | --- |
| LiDAR | `/ubuntu/navigation/lidar_fast_livo` | `PointCloud2`; RELIABLE |
| IMU | `/ubuntu/navigation/imu` | `Imu`; RELIABLE |
| RGB | `/ubuntu/camera/rgb` | `CompressedImage`; BEST_EFFORT |
| Depth | `/ubuntu/camera/depth` | `Image`; BEST_EFFORT |
| CameraInfo | `/ubuntu/camera/camera_info` | `CameraInfo`; BEST_EFFORT |

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

当前 G1 Driver 代码已经发布 RGB 与 depth，但尚未发布 ROS
`sensor_msgs/msg/CameraInfo`。因此在 Driver 补齐该真实 producer 前，启用采集
会继续保存其余四路数据，同时状态明确显示
`missing_sources:camera_info`，不会生成或伪造内参。此能力只记录数据，
不会启用 FAST-LIVO2 图像处理，也不会发送 Driver、Nav2 或机器人运动命令。

本阶段不包含离线回放、派生标注、自动清理、容量配额或 Canvas 数据浏览；
这些是后续独立子任务。

## Canvas 连线

1. Driver `navigation_lidar_fast_livo` -> FAST-LIVO2 `lidar`；
2. Driver `navigation_imu` -> FAST-LIVO2 `imu`；
3. FAST-LIVO2 `livo_odom` -> Nav2 `livo_odom`；
4. FAST-LIVO2 `registered_cloud` -> Nav2 `registered_cloud`；
5. FAST-LIVO2 `obstacle_map` -> Nav2 `obstacle_map`；
6. Nav2 `velocity_proposal` -> Driver `loco.velocity_proposal`。

Canvas 启动后，Nav2 可以先进入 wired 状态，但在 FAST-LIVO2 尚未开始产出
odom/cloud 时，`navigate_to_pose` 会返回 readiness blocker。新图流程先执行
`start_mapping`；旧图流程先执行 `load_map` 和 `relocalize`。等地图视图、odom
和 diagnostics `ready=true` 后再导航。

## 构建

companion 从已验证的本地
`phanthy-fast-livo2:g1-1fcd0d0-n3save1` 镜像派生。统一构建脚本会先检查
这个 tag 的 arm64 架构、上游 revision 和两个补丁 SHA256 与
`source-lock.env` 完全一致，再调用 Compose；同名镜像内容变化会直接失败：

```bash
cd perception/plugins/fast_livo2/companion

./build-companion.sh
```

FAST-LIVO2 和 Vikit 保持在独立 GPL companion 镜像中；主 Perception 镜像
没有复制其源码。详细版本见 [THIRD_PARTY.md](companion/THIRD_PARTY.md)。

## 部署与只读验收

完整 G1 测试栈从仓库根目录用同一个入口构建并启动：

```bash
bash perception/plugins/nav2/deploy/scripts/build-and-start-g1.sh
```

脚本只负责准备地图目录、构建三个 arm64 镜像并调用现有测试容器
`preflight/start`。它不会执行 Git 同步、清理旧容器、启动 Canvas、
`start_mapping` 或导航。运行前停止 Canvas；Core 启用认证时提供真实
`CORE_ACCESS_TOKEN`。
