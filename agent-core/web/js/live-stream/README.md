# Canvas `live_stream` 引导模块

在原有 Canvas Card 内完成小红书、虎牙的账户准备指导、RTMP 凭据导入、人工确认、开播、状态查看和停播。不引入独立管理页面或 OBS 中转。

## 模块

- `platforms.js`：两平台的官方入口、账号状态分支、步骤及地址白名单。小红书分开填写服务器地址和推流码，虎牙填写完整 RTMP URL。
- `card-ui.js`：卡片内引导状态机、单步帮助、用户确认、错误恢复与停播操作。
- `api.js`：调用认证的 Core API 保存配置、重新授权及发起公开直播；状态与停播沿用 MCP 工具调用。
- `attach.js`：仅挂载 `live_stream` 卡片 UI，不修改通用 Card 渲染逻辑。
- `web/css/live-stream-card.css`：直播卡片专用样式。

## 权限与安全

- 平台的注册、实名认证、OBS 权限申请和获取推流信息全部发生在官方页面。
- 浏览器只持久化非敏感步骤进度，不在 localStorage 保存推流码。
- 直播地址经认证的 `/api/canvas/tool-config` 保存，只有 Driver 返回配置成功才在 Core 配置中持久化。
- `authorize-start` 和 `go-live` 由 Core 对操作员请求签发一次性、短时有效的签名；Driver 对直接发来的未签名配置或开播请求拒绝执行。
- `live_stream.go_live` 在 Core 的通用 MCP 转发中跳过自动重放旧配置，防止被未签名的配置请求阻断。
- Solution 导出时无论 Driver 元数据是否在线，都必须清空 `live_stream.rtmp_url`；这不等同于数据库静态加密。
- Core 与 Driver **必须使用配套版本**，不支持单端升级后直接开播。

## 流程

小红书：手机端首次开播与实名认证 → 电脑 OBS 权限申请 → 下载小红书直播助手 → 三方推流 → 保存本场封面和标题 → 复制服务器与推流码 → Bumi 确认开播。

虎牙：登录 → 主播认证 → 主播设置 → 普通/VR 远程推流 → 直播分类和标题 → 获取完整 RTMP URL → Bumi 确认开播。

两平台均须在观众端确认实际视频、声音和停播效果。Driver 返回 `streaming` 仅代表推流进程工作，不能替代平台侧验收。

## 测试

在仓库对应目录执行：

```bash
cd agent-core
python -m pytest tests/test_live_stream_canvas.py tests/test_tool_response.py -q
cd web/js
node --test live-stream/live-stream.test.mjs
```

Linux 下另需运行 Core↔Driver 签名互通测试；真实平台推流只在操作员明确授权、准备好停播措施后进行，不进入默认 CI。不要将真实推流密钥写入测试用例、日志或 Git。
