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
蓝色/粉色标记，方便现场根据地面和天花板分布调参。范围外表面以 `0.20 m`
体素单独累计或从已加载 raw PCD 恢复，只用于 Canvas；它们不会进入
registered cloud、静态障碍图或 Nav2 costmap。高度带默认值是
`-0.30...+0.30 m`；它采用 canonical `map` frame，不能按雷达物理安装高度
直接填写。直接累计不会清除已经由错误高度带写入的静态占用，修改上下界后
必须新建地图。

建图恢复为直接累计模式：FAST-LIVO2 adapter 对每帧注册点云按 `0.10 m`
体素去重后，将导航高度带内且距离有效的体素首次出现即写入静态图，不等待
多帧确认、不跟踪动态分量，也不通过后续自由射线删除已累计点。这样可保留
墙面和稀疏结构的完整证据，但建图期间出现的人或其他移动物体也可能固化进
地图，需由操作者在相对静态的环境中完成建图。

实时避障与累计地图彼此独立：当前 registered cloud 继续进入 Nav2 开启
marking/raytrace clearing 的 live ObstacleLayer，所以导航期间新出现或移动的
物体仍参与即时避障。原始 FAST-LIVO2 odom/cloud 使用 latest-only 订阅；cloud
只有在其源时间戳已被前后 odom/TF 包围且最近位姿差不超过 50 ms 时才对
Nav2 发布，避免 adapter 排队旧帧或让点云早于 TF。Canvas `map_view` 显示累计静态点和新鲜的最新实时扫描，
并显示单独累计的高度带外表面。输出最多 80,000 点，并为低于、位于和高于
导航高度带的三组点分别保留显示预算，避免地面被大量障碍点截断；该采样只
影响监控，不改变规划输入。显示帧直接编码已经分别有界的静态、范围外和实时
点源，不再构造一次性全图体素副本，避免 Canvas 更新拖慢 Nav2 所依赖的 odom
与 registered cloud。已保存的 confirmed static PCD 还会绑定建图时的障碍高度带；加载
时若当前上下界不同会拒绝使用，需恢复原配置或重新建图，避免静态证据语义
悄然变化。

confirmed static map 以有点数上限的稀疏体素保存，不因覆盖范围变大而拒绝
建图。面向 Nav2 StaticLayer 的稠密 OccupancyGrid 则始终发布机器人周围
`static_grid_margin_m` 半径的滚动窗口；地图跨度增大或全局坐标远离原点时，
不会按整张地图包围盒分配巨型数组，也不会使 frame adapter 退出。

资源和地图事务均 fail closed：单帧 live PointCloud2 最多 200,000 点且数据区
最多 64 MiB；累计静态图最多 200,000 点，超限
时不静默抽样。PCD header 和 ASCII 单条记录均限制为 64 KiB，ASCII token
布局及实际非空数据行数必须与声明完全一致，解析还受 map-control deadline
约束。manifest 最大 64 KiB。一次
地图会话最多 64 个 raw PCD，raw 快照与 confirmed static PCD 合计最多
512 MiB。`load_map` 会先完成新旧 manifest/PCD、障碍高度带、点数限制及
滚动 OccupancyGrid 窗口验证，再停止旧定位前端；Adapter 先在旧状态之外准备图，
deadline 内只做原子状态切换，并在控制回执之后发布大栅格。

## Actions

- 生命周期：`info`、`config`、`start`、`stop`
- 建图/定位：`start_mapping`、`stop_mapping`、`load_map`、`relocalize`
- 规划控制：`navigate_to_pose`、`wait_navigation_done`、`pause_nav`、
  `resume_nav`、`stop_nav`
- 语义地点：`capture`、`navigate`

`navigate_to_pose` 保持非阻塞；需要在同一调用链内等待结果时仍使用
`wait_navigation_done`。Nav2 异步上报同一 `nav_id` 的到达、取消、停止、
超时或失败终态时，卡片会立即释放活动任务，下一个导航无需再手工
`stop_nav` 解锁；终态后迟到的 `wait_navigation_done` 会幂等返回已保存的
终态回执。不同 `nav_id` 的迟到消息不会解锁当前任务。

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
