# controlled_semantic_spatial Perception 卡片

`controlled_semantic_spatial` 是 PR #99 中 FAST-LIVO2、Nav2 和 VLN 的统一
公开卡片。Canvas
只看到这一张 `processor` 卡片，正式部署只运行一个 Perception 容器。

## 运行边界

```text
Driver lidar + imu ─┐
camera rgb ─────────┼─> controlled_semantic_spatial card (Perception container)
optional goal_pose ─┘      ├─ FAST-LIVO2 mapping/localization child process
                            ├─ Nav2 planner/controller child process
                            └─ semantic waypoint processor
                                      |
                                      `─ velocity_proposal -> Driver loco
```

- `NavigationPlugin` 是唯一 MCP/Canvas 生命周期所有者。
- FAST-LIVO2 和 Nav2 ROS launch 由 `NavigationRuntime` 作为同容器子进程组
  启停；运行时不调用 Docker，也不需要 Docker socket。
- odom、registered cloud、obstacle map 和 collection status 是卡片内部 ROS
  边，不再作为 Canvas 公共连线端口。
- VLN 命中地点后直接调用同卡片 planner，并透传 Agent Core 的
  `_control_nav_id`；`goal_pose` topic 只保留为可选外部控制入口。
- Nav2 仍只发布 `phanthy.navigation.velocity_proposal.v1`，Driver 继续负责
  物理执行、TTL、急停、二次限幅和停车确认。

## 外部输入

| port | topic | 必需 |
| --- | --- | --- |
| `lidar` | `/ubuntu/navigation/lidar` | 是 |
| `imu` | `/ubuntu/navigation/imu` | 是 |
| `rgb` | `/ubuntu/camera/rgb` | 是 |
| `goal_pose` | `/ubuntu/navigation/goal_pose` | 否 |

## 公共输出

| port | topic | 用途 |
| --- | --- | --- |
| `map_view` | `/ubuntu/navigation/fast_livo2/map_view` | Canvas 地图与机器人位姿 |
| `status` | `/ubuntu/navigation/fast_livo2/status` | 定位、建图和运行状态 |
| `velocity_proposal` | `/ubuntu/navigation/nav2/velocity_proposal` | 连接 Driver `loco` 执行器 |
| `plan` | `/plan` | 当前二维全局路径 |
| `costmap` | `/global_costmap/costmap` | 实时全局代价地图 |

`livo_odom`、registered cloud、confirmed static map、obstacle map 和
collection status topic 仍由
同容器内的定位、规划、语义和数据采集逻辑消费或发布，只是不再生成 Canvas
右侧连线端口。详细 frame、QoS、freshness 和速度约束见内部实现说明：

- [mapping/README.md](mapping/README.md)
- [planning/README.md](planning/README.md)
- [semantic/README.md](semantic/README.md)

完整的静态插件配置样例见 [config.example.json](config.example.json)；其中
`semantic.vlm.api_key` 必须通过部署配置或环境注入真实值，不能提交凭据。
Canvas 的 `config` 动作还提供 `obstacle_min_height_m` 和
`obstacle_max_height_m`。它们控制实时与稳定静态二维障碍的 `map` frame
高度带，必须在卡片停止时修改；`map_view` 会把最新扫描的范围外点用
蓝色/粉色标记，方便现场根据地面和天花板分布调参，但不会累计这些点。

动态物体不会直接写入永久全局图：当前 registered cloud 始终独立进入 Nav2
开启 marking/raytrace clearing 的 live ObstacleLayer；FAST-LIVO2 adapter 只有在
运动门和多帧证据均通过后，才把 confirmed static OccupancyGrid 交给
StaticLayer。静态层在连续 8 帧体素确认前，还会把二维连通分量拆成不超过
`1.0 m` 的空间片执行运动跟踪。默认 `0.10 m` 体素、`0.03 m/s` 速度阈值和
`0.40 s` 比较窗共同给出
`sqrt(2) * 0.10 / 0.03 + 0.40 ~= 5.114 s` 的完整体素驻留观察窗；这比仅看
`0.03 m` 位移所需的 `1.0 s` 更保守。历史按 20 Hz 时间桶保存紧凑二进制摘要，
完整观察窗最多 30 秒；所有轨迹、栅格、样本和近期动态键合计最多
1,000,000 个 history units，输入频率或短命目标增多都不会让历史无界增长。
预算饱和时对应分量保持隔离并 fail closed。满足动态条件的局部
格会被隔离，并撤销其近期的候选和已确认静态格；
停止运动 `1.5 s` 后才重新开始静态确认。其他已确认静态格只有被连续 3 帧
自由射线穿过才会清除。Canvas `map_view` 的稳定部分读取同一静态结果；高度
带外仅叠加最新一帧用于阈值调试，不累计成拖尾。

这是一套只依赖几何和时间的保守过滤，不是人员语义分割：已经静止足够久的
人会按静态物体处理；空间分片会处理人与墙相连的常见情况，但极端稀疏、
遮挡和噪声仍不具备语义分割保证。
无论是否写入静态层，当前帧都继续进入实时障碍层，因此移动人员仍会触发
即时避障。已保存的 confirmed static PCD 还会绑定建图时的障碍高度带；加载
时若当前上下界不同会拒绝使用，需恢复原配置或重新建图，避免静态证据语义
悄然变化。

资源和地图事务均 fail closed：单帧 live PointCloud2 最多 200,000 点且数据区
最多 64 MiB；静态候选证据和 confirmed static map 各最多 200,000 点，超限
时不静默抽样。PCD header 和 ASCII 单条记录均限制为 64 KiB，ASCII token
布局及实际非空数据行数必须与声明完全一致，解析还受 map-control deadline
约束。manifest 最大 64 KiB。一次
地图会话最多 64 个 raw PCD，raw 快照与 confirmed static PCD 合计最多
512 MiB。`load_map` 会先完成新旧 manifest/PCD、障碍高度带及完整静态
OccupancyGrid 边界验证，再停止旧定位前端；Adapter 先在旧状态之外准备图，
deadline 内只做原子状态切换，并在控制回执之后发布大栅格。

## Actions

- 生命周期：`info`、`config`、`start`、`stop`
- 建图/定位：`start_mapping`、`stop_mapping`、`load_map`、`relocalize`
- 规划控制：`navigate_to_pose`、`wait_navigation_done`、`pause_nav`、
  `resume_nav`、`stop_nav`
- 语义地点：`capture`、`navigate`

`start` 按 runtime → mapping → planning → semantic 顺序获取资源；任一步
失败会按相反顺序回滚。`stop_mapping` 和 `load_map` 的 backend 等待预算分别
至少为 360 s 和 900 s。可重试的地图收口失败会保留 Canvas wiring、运行时和
`finalizing` 事务，下一次 `stop`/`stop_mapping` 从原事务继续；永久失败会释放
mapping 控制对象，并继续回收其他模块和运行时。已完成的同名
`stop_mapping` 终态可幂等重放原保存回执，避免迟到重试制造第二份成功结果。

`stop` 始终尝试停止所有内部模块和两个 launch 子进程组，并保留各模块回执，
避免部分停止冒充成功。Runtime 还跟踪 launch 进程派生的独立 Linux 进程组；
即使根进程快速退出，也会按有界的 `SIGINT -> SIGTERM -> SIGKILL` 阶梯回收，
并在首次发送信号前用进程启动时刻校验，避免 PID 复用误杀。

## 构建与本地部署验收

统一镜像只基于锁定 digest 的 ROS Humble/Jammy 基础镜像。构建时按完整 SHA
拉取 Livox SDK2、Livox ROS Driver2、Sophus、Vikit 和 FAST-LIVO2，校验并应用
两份 G1 补丁，再编译 FAST-LIVO2、精确版本 Nav2、两套 ROS adapter 和
Perception Bundle。它不再要求本机预先存在 `phanthy-fast-livo2` 镜像。

该镜像继续安装 ASR、TTS、HTMSG、VOP 所需的 Perception 依赖和 VOP CLIP
权重，但 Jammy 统一镜像通过选定的国内 PyPI 镜像安装锁定的 ARM64 CPU
PyTorch 2.2.2、Torchaudio 2.2.2、Torchvision 0.17.2、NumPy 1.26.4 和
OpenCV Python 4.11，不让 `silero-vad` 或 `ultralytics` 将核心运行时升级到
不兼容的新版本，也不把旧 JetPack/Focal 的 CUDA Python wheel 跨发行版复制进来。
CLIP 的 PEP 517 构建工具也锁定版本并从同一 PyPI 镜像安装，避免旧 Jammy
`packaging` 将合法的 `clip-1.0` 错装为不可导入的 `UNKNOWN-0.0.0`。
APT、ROS、PyPI、Git 源码和权重下载均使用国内可达入口。因此单容器边界与
其他插件的功能依赖在本地合同中保留，Jetson 上的推理吞吐仍属于实机验收项，
不能由单元测试代替。

```bash
./deploy/build_perception.sh --mirror tuna
```

这是仓库默认的 Perception 构建入口；无需 Navigation 专用构建 wrapper，
也无需预构建 FAST-LIVO2 或 Nav2 镜像。旧 CPU/Jetson 构建仍可通过显式
`--variant cpu` 或 `--variant jetson` 选择。

G1 临时验收只创建一个容器。将构建输出的精确镜像名传入：

```bash
export PERCEPTION_IMAGE=local/phanthy-motus/perception:<exact-navigation-tag>
STAGE=preflight bash perception/plugins/navigation/deploy/scripts/owner-start-g1-test-containers.sh
STAGE=start bash perception/plugins/navigation/deploy/scripts/owner-start-g1-test-containers.sh
```

在北京 G1 上从当前 PR 分支的仓库根目录一次完成默认构建和测试容器替换：

```bash
bash perception/plugins/navigation/deploy/scripts/deploy-g1.sh
```

`deploy-g1.sh` 不实现另一套构建逻辑；它只调用仓库默认的
`./deploy/build_perception.sh --mirror tuna`，再调用上述容器生命周期脚本。
旧容器只有带 `com.phanthymotus.test-owner=navigation-card`，或迁移前已知值
`com.phanthymotus.test-owner=nav2-card` 时才会被替换；其他 owner 仍会安全拒绝。
脚本不发送导航目标或速度指令。

正式 `perception/deploy/service.yml` 也只有 `perception` 一个 service。地图和
录制目录作为该容器的持久化 volume；不再定义 `fast_livo2` 或 `nav2` service。

## 第三方与验收边界

统一镜像包含 FAST-LIVO2/Vikit GPL 组件和 Apache-2.0 Nav2/adapter，镜像标签
必须保留组合许可证和锁定 revision。版本与许可证见
[runtime/FAST_LIVO2_THIRD_PARTY.md](runtime/FAST_LIVO2_THIRD_PARTY.md) 和
[runtime/NAV2_THIRD_PARTY.md](runtime/NAV2_THIRD_PARTY.md)。

本地自动测试只能证明合同、生命周期、进程托管、失败回滚和无额外 service。
当前交付状态为 **待 G1 实机测试**：真实传感器、地图质量、重定位、路径和
物理运动仍需 owner 按上述 G1 脚本另行验收。
