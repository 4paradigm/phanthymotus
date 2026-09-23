# PICO 浏览器遥操

此入口在现有 ActuCore Capture TLS 服务提供 `/webxr/` 页面，通过 WebXR 获取头显与双手柄位姿、握把和扳机，通过现有 WSS 配对、操作命令及 WebRTC 双 DataChannel 连接后端。无需安装 APK、浏览器插件或加载外部 CDN。页面可通过 PICO 浏览器添加到应用库；是否出现该入口取决于头显浏览器版本，不保证普通桌面浏览器可安装。

## 用户流程

1. 在 Canvas 遥操卡片展开“浏览器遥操 · 试验版”，显示浏览器入口。在 **PICO 头显浏览器**打开该地址；手机扫码只用于查看地址。
2. 电脑端打开“允许新设备配对”。网页点击“首次使用：申请配对”，核对两端配对码，一致后分别确认。浏览器保存该站点的配对凭据。
3. 点击“进入透视遥操”，完成浏览器权限确认，拿起双手柄。在透视面板用手柄指向按钮，按扳机选择“开始遥操”。天轶仍需先在 Canvas 开启智能控制。
4. 松开扳机、松开至少一个握把，再握紧双握把跟随。松开任一握把保持；面板“结束并收臂”与“停止输入”请求已有后端操作，不把请求已接收当作物理动作完成。未声明头显操作能力的后端仍在 Canvas 管理会话。
5. 退出透视、切换页面、跟踪丢失、输入源变化或网络中断会关闭本次输入连接。恢复后点击“连接已配对机器人”，重新进入透视并明确开始，不重发旧命令。更换机器人需撤销旧配对；清浏览器数据会丢失本机凭据。

原生 App 和 WebXR 使用不同 `client_kind`，凭据不能混用。服务仍只允许一个配对采集端；从原生 App 切换到浏览器时需要在 Canvas 撤销旧配对。旧版本服务不识别 WebXR 类型；回滚服务前也需先撤销浏览器配对，不能静默删除状态文件。

## 部署前提

- 使用 PICO 4/4 Ultra 支持 WebXR 的系统浏览器；必须支持 `immersive-ar`、`local-floor`、非 opaque 透视和左右 `xr-standard` 手柄。不支持时明确禁用，不回退为普通网页键鼠输入或无透视 VR。
- `capture.public_wss_url` 配置为头显能访问的 `wss://<域名>:<端口>/ws/teleop-capture`，浏览器入口为同一地址的 `https://<域名>:<端口>/webxr/`。站点维护者配置浏览器信任、包含匹配 SAN 的证书与完整证书链；PICO 浏览器必须正常验证证书。现有原生 App 固定信任的自签名证书不会自动成为浏览器信任的证书。
- 继续使用独立 Capture TLS 文件，不复用 Agent Core 私钥。页面、配对 HTTP 与 WSS 同源；不开放 CORS，不代理用户指定地址，不提供忽略 TLS 错误的选项。Canvas 的入口 API 仍要求管理认证，打开网页本身不授予配对或运动权限。
- 局域网直连，现有通道不配置 STUN/TURN。跨 NAT、跨公网、浏览器本地网络权限及站点 DNS 需要另行部署验证。
- WebXR 静态文件随 teleop 插件被现有 ActuCore 镜像复制；没有独立服务、Node 构建步骤或新 Python 依赖。原生 APK 安装入口保持独立，浏览器入口不要求 APK 制品可用。

## 输入和操作边界

浏览器只产生 `motus.teleop.rtc-frame.v1` 公共输入；映射、IK、边界处理、平滑、录制和机器人执行继续由现有后端负责。首版接受双臂、手部和 `end_effectors/eef_pose` 任务；拒绝底盘或未知执行器能力，不产生移动命令。裸手追踪、机器人三维模型显示和摄像头视频不在本版范围。

采集使用 `gripSpace`，不使用瞄准射线代替手柄位姿。无效/模拟位姿、未知手柄映射、重复手别及非有限数值不作为有效输入。首次开始、任务重绑或恢复均需要松握把与扳机中立；双握把重新握持才使能。`local-floor` 重置关闭连接。

顺序可靠 `teleop-control` 与无序零重传 `teleop-pose` 沿用现有契约。姿态发送积压、XR 帧中断超过 250 ms、控制应答/存在租约超时关闭连接并禁止旧异步 SDP 重新生效。服务端与 Driver 原有 watchdog 继续独立处理保持；网页不能替代实体急停，状态未知时不宣称机器人已停止。

凭据只存同源浏览器 localStorage；邀请/配对码不放 URL 或日志。静态资产 `no-store`，无 Service Worker 缓存旧控制代码，严格 CSP 且无外部脚本。配对端点及 WebSocket 拒绝其他浏览器 Origin；原生端不带 Origin 的既有协议保持兼容。

## 验证

```sh
node --test actucore/tests/webxr.test.mjs actucore/tests/webxr-lifecycle.test.mjs
python -m pytest actucore/tests/test_webxr.py actucore/tests/test_teleop_transport.py -q
RUN_WEBXR_BROWSER=1 CHROME_PATH=/path/to/chrome python -m pytest actucore/tests/test_webxr.py -q
python -m pytest agent-core/tests/test_teleop_install.py agent-core/tests/test_teleop_install_contract.py -q
node agent-core/tests/teleop-panel.browser.cjs
```

Node 使用可控时钟检查输入门控、失焦、帧中断、积压、任务失效及迟到 SDP；Python 检查实际 HTTPS/WSS、同源限制和跨客户端凭据隔离。可选 Chromium 检查使用真实页面、浏览器加密配对、WebSocket、WebRTC 和 RecordingAdapter；设备能力探测、位姿/握把及输入时钟是测试输入，避免把桌面定时器调度当作头显帧率证据。TLS 忽略只存在于临时 localhost 测试上下文，生产客户端没有该选项。HUD 使用真实 WebGL 编译和绘制，但 XR framebuffer/视图为夹具。

2026-09-23，PICO 4 Ultra（OS 5.15.7、浏览器 4.0.38）通过 USB localhost 诊断页，以本版 `frame.mjs` 和 `view.mjs` 完成两轮 30 秒真实输入测试：2,142 / 2,154 帧，约 71.4 / 71.8 FPS，头显和左右手柄每帧有效，没有超过 250 ms 的帧间隔；记录到双握使能和松握解除。第二轮确认左右手柄均能射线点击“开始”按钮，其余按钮与多轮重握仍待补测。诊断页不连接机器人，也未运行完整配对 / WSS / RTC 链路，USB localhost 不构成生产 HTTPS 部署证据。

该设备进入透视时实际产生短暂 `hidden → visible-blurred → visible` 事件，随后开始有效跟踪。客户端在首次有效帧前等待这段切换，首次采集后任何失焦仍关闭输入。生命周期回归直接加载 `app.mjs`，覆盖初始化异步等待、未聚焦帧、首次跟踪和后续失焦；旧实现会在初始化阶段提前断开。

提交前需明确区分上述证据与真机验收：PICO 4 与 4 Ultra 的实际透视、控制器映射、HUD 射线操作、15 分钟帧率/延迟、至少 10 轮失焦/断网/重定位恢复、TLS 部署、应用库入口及机器人动作仍需现场验证。首次现场验证先使用 Shadow，随后按现有机器人验收流程验证 Live。

官方接口参考：[PICO WebXR/PWA 能力](https://developer.picoxr.com/pico4-ultra/)、[WebXR Device API](https://www.w3.org/TR/webxr/)、[WebXR Gamepads Module](https://www.w3.org/TR/webxr-gamepads-module-1/)。
