# Bumi live_stream 卡片内新手引导（小红书 / 虎牙）

日期：2026-11-20。分支：`feat/live-stream-canvas-fix`。

## 目标

只在现有 Canvas `live_stream` 卡片内提供文字引导；无需打开额外网页管理端（除官方注册、认证和获取推流信息步骤外），不开发语音引导。不更换 Driver 推流 Worker，不修改 QQ 或 Watch7。

## 当前组成

- `platforms.js`：平台注册表、账号资格分支、官方操作链接、分步提示、异常处理、推流信息白名单校验。
- `card-ui.js`：卡片交互状态；从平台选择、已有账号与权限判断、获取推流信息到保存、明确确认开播、实时状态、停播。
- `api.js`：只向 MCP 转发非敏感动作；敏感推流信息通过 `/api/canvas/tool-config/...`，必须确认 Driver 应答。
- `attach.js`：基于已有 `cardsForTool` 查找 live_stream 卡片，不修改庞大的 `canvas.js`，只替换该卡片原有动作区。
- `web/css/live-stream-card.css`：卡片专用样式，其他卡片不受影响。
- `app.js`、`index.html`：加载挂载模块与样式。
- `src/api/canvas.py`：直播配置仅在 Driver 业务应答 `ok=true`、`configured=true` 后接受并保存。

## 用户操作

1. 添加 `live_stream` 卡片（开发/QA 画布中已有）；选择小红书或虎牙。
2. 小红书先按真实状态选择：“未完成首次手机直播 / 手机能直播但没有三方推流 / 有三方推流但未拿到推流码 / 已有本场推流信息”。虎牙保持原有四种状态。
3. 小红书普通直播：手机 App 创作者中心 → 主播中心 → 去开播 → 准备封面、标题 → 实名认证 → 同意协议 → 完成一次真正的直播 → 正常结束。
4. 小红书 OBS 权限：电脑同账号登录 `https://www.xiaohongshu.com/zhibo/obs`，如出现「暂无 OBS 开播权限」点击「申请」。申请成功后官方页面直接进入直播助手下载流程，引导不再设置额外的「确认权限已开放」步骤；若申请失败或仍提示无权限，则停止并查看平台提示。
5. 小红书 RTMP 获取：下载安装官方电脑直播助手 → 同账号扫码登录 → 选择「三方推流」→ 「去设置」保存本场标题与封面 → 获取服务器和推流码。**无需安装 OBS**。
6. 小红书配置页与官方格式保持一致：**服务器地址**、**直播推流码**两个独立输入框。Bumi 在点击「检查并保存」时本地合成 RTMP 完整地址并执行域名白名单校验，兼容官方实测 `live-push-reg-tx.xhscdn.com` 和历史 `live-push.xhscdn.com`，不再提供切换完整地址或额外说明小字。虎牙继续使用完整推流地址。进度仅保存非敏感信息。
7. 机器人确认成功后，再明确确认公开摄像头与麦克风，调用 `go_live`；如平台另需点击开始直播，按官方提示完成，观众端检查音视频。
8. “结束直播”调用 `end_live`；须同时在小红书平台确认直播间确实结束。

本地浏览器只保存非敏感引导进度，不保存推流密钥；秘密仍会经现有 Core 配置 API 发送到 Driver。

## 虎牙：每一步直达链接（2026-11-20）

- 登录虎牙：https://www.huya.com/
- 主播实名认证：https://i.huya.com/index.php?m=ProfileAuth（官方指导：https://help.huya.com/165）
- 主播设置与开通远程推流：https://i.huya.com/index.php?m=ProfileSetting
- 选择直播品类、普通直播、填写标题、点击「获取推流地址并开播」：继续使用上述主播设置页（官网未提供可验证的更深层直链）
- 远程推流官方图文教程：https://help.huya.com/355
- 个人中心兜底：https://i.huya.com/，找「我是主播 → 主播设置」

每个引导步骤都有自己的直达官方链接和明确的按钮文字。若找不到按钮，显示针对性的提示，并在有对应帮助资料时提供官方文档链接。「获取推流地址并开播」可能启动虎牙侧直播状态，提醒用户先准备好 Bumi。
本次只改虎牙的引导元数据和卡片显示，不改 Driver 的推流路径、虎牙 URL 白名单或小红书流程。

## 验证记录

- `node --test agent-core/web/js/live-stream/live-stream.test.mjs`：10 passed。
- `node --test agent-core/web/js/qr-field.test.mjs agent-core/web/js/browser-source.test.mjs`：43 passed。
- `python -X utf8 -m pytest noetix/bumi/live_stream/tests -q --basetemp=.pytest-live-stream-temp`（Driver 仓库根）：45 passed、7 skipped。
- 全部旧前端套件未通过：`config-topic-refresh.test.mjs`、`startup-progress.test.mjs` 的 VM 用例引用已不存在的内部函数；未修改相应 Canvas 主文件。
- QA：`https://localhost:15684/`，`phanthy-qq-minimal-dashboard`（非正式容器）。已热加载前端并替换测试容器的 `canvas.py`，测试 Core 已重启并恢复。静态资源 HTTP 200、MCP `live_stream` 已注册。
- QA 合成地址配置结果：`code=200, applied=true, ready_to_go_live=true`。模拟 Driver 状态为 `idle`，使用固定本机 RTMP 目的地。
- QA 的 `go_live` 测试执行被工具安全检查拦截，**未完成**浏览器按钮全流程和平台观众端验收，不能声称真实直播通过。
- 正式 `15683` 与 Watch7 驱动未改动。

## 15684 空白页故障修复（2026-11-20）

- 现象：HTTPS、HTML、CSS 与全部 JS 资源返回 HTTP 200，但整个 Dashboard 空白。
- 浏览器真实报错：`The requested module './api.js' does not provide an export named 'authorizeStoredLiveStream'`。
- 根因：容器中的 `card-ui.js` 已升级，但 `api.js` 仍是旧版，ES Module 静态导入失败，应用入口没有执行；并非虎牙登录或网络问题。
- 处理：更新 `api.js` 后，将 `web/js/live-stream/` **整个目录**一次性同步至 15684 QA，而不是逐文件热更新。
- 预防：`live-stream.test.mjs` 显式导入 `card-ui.js`，保证 API 导出契约在测试期校验；更新测试环境后必须用浏览器自动化验证 `#app` 可见、`live-stream-card` 已挂载、虎牙直达链接正确。仅凭 HTTP 200 或 Node 逻辑测试通过不得宣称页面正常。
- QA 实测：登录后 `#app` 可见、2 张 Canvas 卡片、1 张直播卡片；虎牙「查找直播信息」步骤链接指向 `https://i.huya.com/index.php?m=ProfileSetting`。
- 正式 15683 与 Watch7 容器没有修改。

## 上线前阻塞

1. 必须浏览器手动验收卡片页面、进度恢复、剪贴板权限拒绝、双平台完整导入、明确确认开播与停播。
2. 现有 Core 配置数据库可能存储完整 `rtmp_url`；尚未改为秘密引用存储或完成历史秘密清理。不能宣称生产秘密安全。
3. Driver 目前以“保存本场不同地址 + 五分钟许可”作为开播授权，尚未与操作员身份强绑定，正式公开使用需要补齐。
4. 小红书部分按钮名称来自第三方教程，真实账号 UI、权限入口及平台观众端需逐步实测。
5. QA 测试容器静态文件和 `canvas.py` 是容器内热更新，重建容器会丢失；以本地源码作为可持续版本。备份位于 QA 容器 `/tmp/qa-live-card-static-backup`。

## 不要做的事

不要在 `.ai-bridge` 目录写交接，不要把任何真实推流码、验证码或 AppSecret 写进仓库或日志；不要直接动正式 15683、Watch7 或真实 Bumi 推流配置。
