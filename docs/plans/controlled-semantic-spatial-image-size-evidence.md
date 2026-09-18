# ControlledSemanticSpatial 镜像体积证据

2026-09-18 读取 `bj-warehouse.tencentcloudcr.com/phanthy-motus` 的 registry v2
manifest；以各 `layers[].size` 求和，测量压缩层传输字节。它不是 Docker
`image inspect .Size` 的解压后磁盘大小，不与历史 0.82 GiB 增量混用。
导航基线的 index 必须先解析到 linux/arm64 子 manifest，不能把 index 大小当镜像大小。

| 镜像 | manifest digest | 压缩层 bytes | 相对固定 JP5.11 平台父镜像 |
| --- | --- | ---: | ---: |
| jetson-base JP5.11 | `sha256:92c4c12a1dc5d4a4e8cb479a69164260578b4c3b022ef3b94c6f0fc20f2462d6` | 6,497,999,697 | 基线 |
| actucore-navigation-base ARM64 | `sha256:1064b34c49d8d1c9124e721b89b16f3af955a47e09ff97bbac5d1711d7a2aee2` | 6,513,793,109 | +15,793,412（15.06 MiB） |
| actucore `release.260918.d6d72a5-jetson-jp5.11` | `sha256:4af31256fab0cf10ab909ad7608a48d74a5357628df999075a000bc17fc19f93` | 6,513,347,191 | +15,347,494（14.64 MiB） |
| actucore `release.260918.d6d72a5-jetson-jp6.1` | `sha256:9f3e91c2d94aa016cf6ae5927f3e7ba4d7285f3805bc07dc4c11139b474bfccb` | 9,124,826,210 | 不同平台，不作差 |

导航基线固定 index 为 `sha256:14550b74bfce5c0ede5908e6081da375148b67695a134f102f025c535f702e4b`。
按 digest 和 size 对照，导航基线及 JP5.11 最终镜像的前 47 层均与平台父镜像相同；
上表增量是其余层之和，不是假定依赖包全部新装后的估算。此次未改导航基线，也未
重新发布它；日常构建复用固定 digest。表中业务镜像仅代表写明的已构建版本。

## 编译依赖与可缩减范围

基线中 APT 安装层的压缩大小为 1,150,381 bytes。只读检查该层 tar 的 dpkg
元数据：新增/修改的 `var/lib/dpkg/info/*.list` 只有 `libfmt-dev.list`；
dpkg status 确认 build-essential、CMake、Git、pkg-config、Boost、Eigen、PCL、fmt
已安装。大多数开发依赖已由父镜像提供，并非这个基线重新加入完整编译栈。
显式 apt 声明保留可复现的源码构建契约；已经安装的包由 apt 复用。

fmt 开发头文件用于 FAST-LIVO2 源码编译；Boost/PCL/Eigen 是其编译接口，Nav2
使用 Boost program-options。这里只保留 `--no-install-recommends` 的构建依赖；
OpenCV 仅在父镜像没有 opencv4 pkg-config 时安装。当前不为减少声明行数而卸载
父镜像依赖，也不假定未经完整源码构建验证的更窄包集合可替代它们。

最终服务镜像从平台父镜像重新开始，额外的 APT 构建层不跨阶段复制。新增运行层是
裁剪后的 ROS install spaces 与应用文件；未编译的 Nav2 组件清单见导航 README。
进一步缩小 install space 必须用实际动态加载/运行路径验证，不能随意删除插件。
最终阶段检查 FAST-LIVO2、Nav2 节点、segmented_controller 的 ldd 和全部配置的
BT 插件 dlopen，已在 `d6d72a5` BOT 构建通过。它证明加载依赖完整，不代表导航真机验收。

## 复现测量

通过 registry 的匿名 pull token 调用 `/v2/phanthy-motus/<repository>/manifests/<digest>`，
Accept 同时包含 Docker v2 manifest、OCI manifest 与 OCI index；若返回 index，
选 `platform.os=linux`、`architecture=arm64` 的子 manifest，再对 `layers[].size` 求和。
对层 digest/size 前缀作比较得到共享父层与实际新增传输量。未下载整个镜像，也未
将压缩体积标成磁盘占用；部署空间规划仍需目标机器的解压后大小与 Docker 共享层情况。
