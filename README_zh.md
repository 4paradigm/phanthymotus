# Phanthy Motus

[English](README.md) | [官网](https://motus.phanthy.com)

**赋予具身智能真正的灵魂。** PhanthyMotus 是新一代开源具身智能 Agent 框架与平台。基于稳健的 ROS2 内核，无缝连接多模态传感器与机器人执行层，灵活集成 World Model、LLM 和 VLM，将传统硬件转化为能够自主感知、思考并行动的智能助手。

## 快速开始

一行命令安装并运行：

```bash
curl -fsSL https://motus.phanthy.com/install.sh | sudo bash
```

或指定版本：

```bash
curl -fsSL https://motus.phanthy.com/install.sh | sudo bash -s <tag>
```

安装脚本会自动安装 Docker（如未安装）、拉取最新 Agent Core 镜像并启动服务。

打开 `http://<设备IP>:15678` 进入 Web Dashboard。

在 [Resource Center](https://motus.phanthy.com) 浏览可用版本和镜像。

### 连接硬件

从 **[phanthymotus-driver](https://github.com/4paradigm/phanthymotus-driver)** 部署硬件驱动。驱动启动后会自动注册到 Agent Core，无需手动配置。

### 从源码构建

参见 [CONTRIBUTING.md](CONTRIBUTING.md) 了解如何从源码构建和运行。

## 特性

- **可视化编排** — 拖拽式 Web Dashboard，在画布上连接设备、传感器和 AI 模型
- **MCP 数据总线** — 统一的 [Model Context Protocol](https://modelcontextprotocol.io) 硬件接口
- **事件驱动 Agent Loop** — LLM 驱动的推理引擎，支持多轮工具调用，由实时传感器事件触发
- **ROS2 集成** — 原生 DDS Bridge，无缝中继和监控 ROS2 Topic
- **可插拔感知栈** — 模块化 ASR/TTS，支持本地推理（Jetson）
- **Web Dashboard** — 浏览器内实时监控设备、查看 Agent 活动流、管理配置

## 架构

```
+------------------+                   +--------------------+
|                  |    SDK / DDS      |                    |
|  Hardware(机器人) | <===============> |  Hardware Drivers  |
|                  |                   |   (底层驱动/传感器) |
+------------------+                   +--------------------+
                                            ||          ||
              [原始数据: PCM/图像]            ||          || [无需感知的传感器数据]
              (DDS / ROS2 Topic)            ||          || (DDS 直连)
                                            \/          |
+------------------+                   +------------+   |
|                  |                   |            |   |
|  Web Dashboard   | <===============> | Perception |   |
|   (前端可视化)    |     WebSocket     |  (15720)   |   |
|                  |                   |  (语义理解) |   |
+------------------+                   +------------+   |
         ^                                  ||          |
         |                                  || [高维语义]
         | WebSocket (控制/状态)             \/          \/
         |                             +--------------------+
         |                             |                    |
         +===========================> |    Agent Core      |
                                       |     (15678)        |
                                       |   (LLM/AI决策核心) |
                                       +--------------------+
                                                ||
                                                || [MCP 执行指令]
                                                \/
                                       (回流到 Drivers 执行动作)
```

硬件驱动在独立仓库维护：**[phanthymotus-driver](https://github.com/4paradigm/phanthymotus-driver)**。

## Web Dashboard

Dashboard（`http://<设备IP>:15678`）提供：

### Canvas — 可视化编排

将所需的传感器与执行器放入画布，连接到核心 Agent Loop，框架自动完成数据流转与动作执行。像搭积木一样搭建你的具身智能体。

![Canvas](docs/images/home.png)

### 实时监控

传感器数据实时可视化 — 音频波形、电池状态、3D 骨骼/点云等。

![监控面板](docs/images/dashboard.png)

### 智能体定义

在 UI 中直接定义 Agent 的身份、系统提示词和长期记忆。

![智能体定义](docs/images/agent-definition.png)

### 历史日志

浏览历史 Agent 会话，查看完整事件轨迹和工具调用结果。

![历史日志](docs/images/history.png)

### 技能管理

社区驱动的技能广场，汇聚用户提交的技能。浏览并一键安装他人分享的技能，也可以用自然语言教会机器人新的特殊技能，无需编程。

![技能](docs/images/skills.png)

### 服务部署

从 Dashboard 部署和管理 Agent Core 及硬件驱动容器。

![部署](docs/images/deploy.png)

## 端口

| 服务 | 端口 |
|------|------|
| Agent Core | 15678 |
| Perception MCP | 15720 |
| Perception WebSocket | 15721 |

硬件驱动端口请参见 [phanthymotus-driver](https://github.com/4paradigm/phanthymotus-driver)。

## Resource Center（可选）

平台可选连接 [Resource Center](https://motus.phanthy.com) 获取：
- 预构建的驱动/感知镜像浏览和部署
- 技能和扩展管理
- OTA 更新

通过 `RESOURCE_CENTER_URL` 环境变量配置。

## 贡献

参见 [CONTRIBUTING.md](CONTRIBUTING.md) 了解开发环境搭建、架构细节和贡献指南。

## 许可证

[Apache License 2.0](LICENSE)
