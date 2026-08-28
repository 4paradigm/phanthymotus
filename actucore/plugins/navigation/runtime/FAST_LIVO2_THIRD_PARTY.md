# FAST-LIVO2 第三方依赖锁

统一 ActuCore 镜像基于按 JetPack 版本锁定 digest 的 `jetson-base`，不再继承或拉取
`phanthy-fast-livo2`。`fast_livo2-source.lock` 锁定 3 个上游仓库 revision
和三个 G1 补丁哈希；`actucore/Dockerfile.navigation-base` 在同一构建中
校验、应用补丁并从源码编译完整 FAST-LIVO2 工作区。

- FAST-LIVO2 ROS 2/MID360 fork:
  `Rhymer-Lcy/FAST-LIVO2-ROS2-MID360-Fisheye@1fcd0d05cadaeb25ca59fd87cda95aaaee41e3ea`, GPL-2.0-only.
- rpg_vikit ROS 2 fork:
  `Rhymer-Lcy/rpg_vikit_ros2_fisheye@0b5548d9adab58128f3a59a507e96a83acfa8fdf`, GPL-3.0-only.
- `runtime/livox_ros_driver2_msgs/` 只保留 Livox ROS Driver2 的
  `CustomMsg` / `CustomPoint` 消息定义，来源 revision
  `Livox-SDK/livox_ros_driver2@13eb05e4e6dd7a765b934d0c5fd6236676a57b49`，MIT。
  G1 运行配置使用标准 `sensor_msgs/PointCloud2`，不启动 Livox Driver，也不再
  拉取或编译 Livox SDK2。
- Sophus:
  `strasdat/Sophus@de0f8d3d92bf776271e16de56d1803940ebccab9`，MIT。
- `runtime/g1_fast_livo2/` 下的 ActuCore adapter 使用 Apache-2.0。

三份补丁存放在 `runtime/patches/`，其 SHA256 与版本锁一致；分别保留已经
验收过的传感器队列/IMU gap 恢复、PCD checkpoint 和 finalization 行为。

FAST-LIVO2/Vikit 与 Nav2 位于同一个 ActuCore 镜像，不再通过独立基础镜像或
companion image 隔离。分发该组合镜像时必须同时满足 GPL-2.0-only、
GPL-3.0-only 和 Apache-2.0 的适用义务；镜像标签也必须保留组合许可证和锁定
revision。运行时只启动同容器子进程，不依赖 Docker socket 或额外容器。
