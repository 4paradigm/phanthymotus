# Nav2 阶段六集成环境

该目录用于在 amd64 Ubuntu 上验证 `Core -> Perception MCP -> ROS 2 -> Nav2`
闭环，不是生产部署入口。`perception` 服务复用同一 Nav2 镜像中的 ROS
Humble 运行环境，并只加载 `nav2` 插件，避免阶段六下载与本卡无关的
ASR/VLM 模型依赖。

```bash
NAV2_IMAGE=phanthy-nav2:nav2-card-stage6-amd64 \
  docker compose -f compose.stage6.yml up -d nav2 perception

NAV2_IMAGE=phanthy-nav2:nav2-card-stage6-amd64 \
  docker compose -f compose.stage6.yml --profile fixture run --rm fixture
```

fixture 发布与已发布 G1 Driver 一致的 legacy `loco_state` 和 point-cloud
envelope，同时完整记录 status/proposal。向
`/nav2_stage6/fixture_control` 发布 `stop_inputs` 可在保留订阅观测的同时中断
传感器输入；发布 `resume_inputs` 恢复。fixture 是可重复的合同和故障注入
样本。为了在导航终态前精确触发 freshness 门限，也可向 fixture
容器发送 `SIGUSR1` 停止输入、`SIGUSR2` 恢复输入。这只是阶段六的
确定性合同测试，不代替 G1 真实录包和真机物理执行验收。

默认 `NAV2_FIXTURE_SENSOR_SCHEMA=legacy` 保留发布版回归。验证 Driver
v2 同时钟合同时使用：

```bash
NAV2_FIXTURE_SENSOR_SCHEMA=v2 \
NAV2_IMAGE=phanthy-nav2:nav2-card-stage6-amd64 \
  docker compose -f compose.stage6.yml --profile fixture run --rm fixture
```

v2 fixture 使用同一 ROS system clock 时间生成 `loco_state.v2` 和
`PCV2 flags=0x0001` MID360 XYZIRT 帧，用于验证字段恢复、freshness
以及 odom/scan 源时间差门禁。
