# 第三方依赖锁

统一 Navigation Perception 镜像中的 Nav2 apt 包必须按
`nav2-source-lock.env` 与 `perception/Dockerfile.navigation` 中的精确版本
构建。下表记录本卡片直接依赖；镜像内同时生成
`/opt/g1-nav2-package-lock.txt` 供部署审计。Nav2 作为同容器子进程运行，
不再存在独立 Nav2 service 或 companion image。

| 包 | amd64 锁定版本 | arm64 锁定版本 | 上游 | 许可证 |
|---|---|---|---|---|
| `python3-colcon-common-extensions` | `0.3.0-100` | `0.3.0-100` | <https://github.com/colcon/colcon-common-extensions> | Apache-2.0 |
| `python3-pytest` | `6.2.5-1ubuntu2` | `6.2.5-1ubuntu2` | <https://github.com/pytest-dev/pytest> | MIT / Expat |
| `ros-humble-nav2-bringup` | `1.1.20-1jammy.20260804.225407` | `1.1.20-1jammy.20260805.020157` | <https://github.com/ros-navigation/navigation2> | Apache-2.0 |
| `ros-humble-navigation2` | `1.1.20-1jammy.20260804.223401` | `1.1.20-1jammy.20260805.013510` | <https://github.com/ros-navigation/navigation2> | Apache-2.0 |
| `ros-humble-rmw-fastrtps-cpp` | `6.2.10-1jammy.20260724.002510` | `6.2.10-1jammy.20260725.133410` | <https://github.com/ros2/rmw_fastrtps> | Apache-2.0 |

版本 pin 由 2026-08-11 的 ROS 2 Humble 软件源实际可用版本生成；ROS buildfarm 的 amd64 与 arm64 构建时间戳不同，因此必须分架构精确锁定。许可证来自镜像内 Debian copyright 或 ROS `package.xml`。镜像构建若无法解析任一精确版本或遇到未支持架构必须失败，不允许自动升级到仓库最新版本。

本机 Docker Desktop 的 arm64 模拟环境中，apt 的 `_apt` sandbox 会让 detached signature 校验异常退出，而同一响应的 `gpgv` 验签正常；Dockerfile 因此仅在构建阶段设置 `APT::Sandbox::User=root`。仓库签名校验仍然开启，未使用 `trusted=yes`、`AllowUnauthenticated` 或 `--allow-unauthenticated`。
