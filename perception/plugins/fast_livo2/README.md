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
- `map` 是本次 FAST-LIVO2 进程的会话原点，不宣称跨重启全局定位。

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
| `map_view` | `/ubuntu/navigation/fast_livo2/map_view` | `map` | Canvas 体素化累计地图和机器人位置 |
| `status` | `/ubuntu/navigation/fast_livo2/status` | JSON | 算法进程、输入 freshness、frame 和产物状态 |

Canvas 地图最多保留 80,000 个 `0.10 m` 体素占用点；它是监控视图，不是
可重定位地图格式。原始 PCD 分片保存在宿主机
`/opt/phanthy-motus/data/fast_livo2/maps`。

## Actions

| action | 参数 | 语义 |
| --- | --- | --- |
| `start_mapping` | `map_name` | 清空 Canvas 会话图并启动一个新的 FAST-LIVO2 进程 |
| `stop_mapping` | 无 | `SIGINT` 停止算法，等待尾段保存并写 session manifest |

`map_name` 只允许 `A-Z a-z 0-9 _ . -`，最长 64 字符。停止 Canvas 时若仍在
建图，卡片会先执行 `stop_mapping` 再释放 ROS backend。

当前锁定算法只有 PCD 保存，没有 PCD 加载、`/initialpose` 或 scan-to-map
全局重定位。因此首版只支持同一进程会话内“边建图边导航”；重启 companion
后旧 PCD 仍在，但不能把它冒充为已加载定位地图。

## Canvas 连线

1. Driver `navigation_lidar_fast_livo` -> FAST-LIVO2 `lidar`；
2. Driver `navigation_imu` -> FAST-LIVO2 `imu`；
3. FAST-LIVO2 `livo_odom` -> Nav2 `livo_odom`；
4. FAST-LIVO2 `registered_cloud` -> Nav2 `registered_cloud`；
5. Nav2 `velocity_proposal` -> Driver `loco.velocity_proposal`。

Canvas 启动后，Nav2 可以先进入 wired 状态，但在 FAST-LIVO2 尚未开始产出
odom/cloud 时，`navigate_to_pose` 会返回 readiness blocker。先在
FAST-LIVO2 卡片执行 `start_mapping`，等地图视图和 odom 出现后再导航。

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
