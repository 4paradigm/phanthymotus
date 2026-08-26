# 第三方依赖锁

统一 ActuCore 镜像里的 Nav2 **从源码编译**，版本按 `nav2-source-lock.env` 中的
完整 Git SHA 锁定，构建时在 `actucore/Dockerfile.jetson` 里逐个校验为 40 位 SHA。
镜像内同时生成 `/opt/actucore-ros-package-lock.txt`（`ros2 pkg list` 快照）供
部署审计。Nav2 作为同容器子进程运行，不存在独立 Nav2 service 或 companion image。

## 为什么不是 apt

ActuCore 的 base 是 `jetson-base:jp*-torch`（Ubuntu 20.04 Focal，ROS Humble 为
`/opt/ros/humble/install` 下的源码 install-space）。`ros-humble-*` 的 Debian 包
只为 Jammy 发布，Focal 上不存在，因此原先的
`ros-humble-navigation2=1.1.20-1jammy...` 这套 apt pin 无法沿用。`navigation2`
钉在同一个上游 release **1.1.20**，保证换打包形态不换运行行为。

## 锁定的源码

| 包 | 锁定 revision | 上游 | 许可证 |
|---|---|---|---|
| `navigation2`（子集） | `a097086` = tag `1.1.20` | <https://github.com/ros-navigation/navigation2> | Apache-2.0 |
| `behaviortree_cpp_v3` | `9d624dc` = tag `3.8.8` | <https://github.com/BehaviorTree/BehaviorTree.CPP> | MIT |
| `bond_core`（bond / bondcpp / smclib） | `1e58909` = branch `humble` | <https://github.com/ros/bond_core> | BSD-3-Clause |
| `diagnostic_updater` | `de779cf` = branch `ros2-humble` | <https://github.com/ros/diagnostics> | BSD-3-Clause |
| `perception_pcl`（pcl_ros） | `67a5c2b` = branch `humble` | <https://github.com/ros-perception/perception_pcl> | BSD-3-Clause |
| `rosbag2_storage_mcap` | `c1c2159` = branch `main` | <https://github.com/ros-tooling/rosbag2_storage_mcap> | Apache-2.0 |

`angles`、`map_msgs`、`laser_geometry`、`pcl_msgs`、`cv_bridge`、`image_transport`
等已在 base 里，构建时用 `ros2 pkg prefix` 逐个探测，只编真正缺的那些（日志里
会打印 `[ros-deps] already in base:` / `will build:`），换 JetPack 版本时自动适配。

## 只编卡片会加载的子集

`navigation_launch.py` 起的 controller / planner / smoother / behavior /
bt_navigator / waypoint_follower / velocity_smoother / lifecycle_manager，加上
`nav2_params.yaml` 点名的 navfn 和 costmap static/obstacle/inflation 三层。
DWB 与 rotation shim 包仅作为上游 navigation2 组合构建兼容项保留，
运行配置已由卡片内的 `g1_segmented_controller` 取代。刻意不编 `nav2_amcl`、`nav2_smac_planner`、
`nav2_mppi_controller`、`nav2_constrained_smoother`、`nav2_route`、
`nav2_rviz_plugins`、`nav2_collision_monitor`、`nav2_system_tests` —— 它们会把
ompl、ceres、xtensor/xsimd、nlohmann-json、Qt5 拖进镜像，而这张卡片一个都不用。

## 已知的源码 base 差异

tf2 在 Humble 生命周期里把 include 从 `.h` 改名为 `.hpp`。base 的 ROS 快照较早
（`tf2/LinearMath/*.h`、`tf2/utils.h`），FAST-LIVO2 与 Nav2 1.1.20 都按新名字
include。镜像因此在编译前给 tf2 家族每个 `.h` 生成同名 `.hpp` 转发头（纯别名）。
这是从 Jammy apt 包迁到 Focal 源码 base 时唯一需要的头文件层适配。
