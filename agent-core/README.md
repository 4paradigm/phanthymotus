# Agent Core 遥操卡片联动

本页说明 Canvas 的遥操配置、安装入口和项目生命周期。采集、协议与求解实现见 [ActuCore 手册](../actucore/plugins/teleop/README.md)及[三段架构计划](../docs/plans/teleop-end-effector-architecture.md)。本轮新链路只接入天轶双臂；不启用手、腿或 G1 新版运动控制。

## 使用卡片

1. 将普通 ActuCore 服务中的 `teleop` 卡拖入 Canvas。打开“连接机器人卡片”，选择已注册的天轶 Driver，点击“建立三段连线”；也可以手工添加 `motion_control` 和 `arm` 并连接。
2. 正向连线是 `teleop(control/eef) → motion_control → arm(control/joint)`。Canvas 自动增加从运动控制卡返回遥操卡的虚线，用于求解、显示及执行反馈。无需输入机器人 IP 或复制后端脚本；虚线随命令连线生成，不能单独删除。
3. 在各卡片的配置中设置本卡负责的参数。遥操卡只配置输入与映射；模型、TCP、IK 和碰撞配置属于运动控制卡；执行范围及硬件限制属于 Driver。保存配置不会开始机器人运动。
4. 打开遥操卡的“安装 PICO 与一键连接”，生成安装短链接，在 PICO 浏览器打开并下载应用。如果镜像没有安装包，卡片会明确提示，不能把服务启动当作 APK 已可下载。
5. 点击“生成一次性连接邀请”。新链接和二维码包含会自动预填的机器人资料；安装应用后返回该页面，点击“打开应用并连接此机器人”。邀请十五分钟有效、仅可兑换一次；新邀请会替换旧邀请。也可继续使用“允许新设备配对”及人工核对指纹的原有流程。
6. 确认模型、标定和执行条件就绪后，开启智能控制。此时只进入“等待 PICO 开始”，不会占有执行权或自动运动。在 PICO 点击开始，之后按住双握把输入；松开任一握把保持，重新握住按实测双臂建立新基准。
7. 点击 PICO 的“结束并收臂”，或关闭智能控制。Core 必须等收臂完成且控制权释放后，才能停止下游卡片。失败会保留当前图和 `stop_failed` 状态，显示原因并允许重试；不会把“请求已接收”显示成完成。“立即停止”是独立入口，不承诺完成自然下垂轨迹。

状态显示区分 PICO 在线、求解状态和硬件执行反馈。`Shadow` 只预览，不请求硬件执行权；连接成功、黄色模型更新或项目“运行中”均不等于机器人正在执行。

旧 `teleop → teleop_executor` 连线仍可使用。G1 仅保留它已声明的独立卡片能力；没有项目托管或收臂能力时，不显示这些功能为可用。

## Core 的接口边界

- 连接身份来自已保存的卡片、端口和当前 MCP 注册表。启动前分别核对运动控制卡和执行卡的实时能力；不接受图中缓存 URL 或话题作为授权依据。
- `x-teleop-target` 的版本 2 配合 `x-motion-control` 描述中间卡；`x-control-target` 描述执行卡。当前要求同机同 Driver、双臂完整资源组。普通版本 1 控制卡仍沿用原单端口协商与优先级语义。
- 新的多输出控制卡通过 `control_interfaces` 按输出 `port_id`（兼容 `id`，无 ID 时为端口索引）接收下游描述。不同输出端口不再合并为一个动作空间。同一输出端口接到不一致的版本 2 坐标系、分组或模型时拒绝启动。
- 反馈线从注册能力自动重建，不参与启动 DAG，也不作为普通 `input_topic` 注入遥操卡。实际反馈订阅由 `project_start.driver_binding.feedback_topic` 绑定；Core 不转发每帧运动命令。
- `motion_control.start` 收到 arm 的动作空间 `control_interface` 及专门的 `execution_binding`；两者不混用。所有普通卡片准备成功后，Core 才调用 `teleop.project_start`。
- 停止调用 `teleop.project_stop` 并等待 `return_completed/return_required` 与 `authority_released` 明确回执。只有回执成功后才停止其他卡片；Graph 编辑和替换同样遵守该顺序。

## 安装入口与权限

安装服务仍在现有 Core 与 ActuCore 内，不增加独立部署服务：

| 接口 | 用途与权限 |
|---|---|
| `POST /api/teleop-install/{mcp_id}` | 经过 Dashboard 鉴权，从已注册遥操服务取得安装信息并生成十五分钟下载链接 |
| `DELETE /api/teleop-install/{mcp_id}/{ticket}` | 经过 Dashboard 鉴权撤销本服务下载链接；重新生成也会撤销旧链接 |
| `POST /api/teleop-install/{mcp_id}/invitation/{ticket}` | 经过 Dashboard 鉴权和遥操管理凭据调用一次性邀请；不授予运动权限 |
| `GET /pico/{ticket}` | 有效下载链接的安装页；不提供控制 API |
| `GET /pico/{ticket}/apk` | 固定 APK 路径代理，核验 TLS 证书指纹、大小和 SHA256 后发送；最多三次（含重试），间隔至少两秒，单链接不并行下载 |

连接邀请位于 URL fragment，浏览器不会将它发送给安装页的 HTTP 服务。Core 不把邀请写入下载票据、配置、日志或长期存储。下载票据在现有 Core SQLite 独立表中仅保存 token 的 SHA256、公开安装信息和绝对期限，重启不重置十五分钟有效期；多 worker/容器须共享同一个 DB 卷，最多保留 128 条有效记录。邀请本身仍由 ActuCore 单次消费，随其进程重启失效。公开短链接只提供有限次 APK 下载，不代表配对成功，更不代表开始遥操。下载中到期或撤销会终止后续传输；进程崩溃的下载占位最多保留 240 秒，随后可在剩余期限和次数内重试。

ActuCore 的 MCP `installation_info.package` 是摘要；Core 根据同时返回的固定路径和证书指纹，另取 HTTPS `/onboarding/package` 的扁平元数据并校验。不能把 MCP 外层对象直接当作包元数据。跨服务契约测试使用生产 Plugin dispatch、MCP handler 和真实 TLS package handler 覆盖此边界。

二维码由锁定的 `qrcode==8.2` 在本机生成 SVG，不依赖外部二维码网站或 Pillow。依赖许可证为 [BSD-3-Clause，含继承的 MIT 说明](https://github.com/lincolnloop/python-qrcode/blob/v8.2/LICENSE)。生产依赖见 `pyproject.toml` 和 `uv.lock`。

## 离线验证

从 `agent-core/` 运行：

```bash
python -m pytest tests/test_motion_project.py tests/test_teleop_project_lifecycle.py \
  tests/test_start_project_control_plane.py tests/test_teleop_install.py \
  tests/test_teleop_panel_metadata.py
NODE_PATH=/path/to/playwright/node_modules node tests/teleop-panel.browser.cjs
```

浏览器测试可通过 `CHROME_PATH` 指定已有 Chrome。二维码解码检查使用测试环境的 CairoSVG、Pillow 和 zxing-cpp；这些不是生产依赖。测试使用本地 TLS 服务和受控 MCP 替身，验证连线、协议、安装校验及状态流程，不代表部署或真机验收。
