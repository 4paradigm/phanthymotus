# 当前验证记录（2026-11-13）

基线为上游 main `541d53461818f220a56f902e5c7093be61e155d8`。以下均为本次实际运行；没有沿用原 ROI 报告中的 87/2325 数字。

| 检查 | 结果 |
|---|---|
| Python 3.10 ROI 核心 / MCP HTTP | 35 passed |
| 全新解包独立包自测 | 35 passed |
| Chromium 独立页面 | FAIL / PASS / INCONCLUSIVE 三场景通过 |
| Chromium 未修改上游 Workbench.js | FAIL / PASS / INCONCLUSIVE 三场景通过，page_errors=[] |
| Core MCP / registry / skill execution 回归 | 37 passed |
| Perception 全量（允许回环 socket） | 2289 passed, 1 failed, 40 skipped, 69 subtests passed |
| 原样上游全量对照 | 2253 passed, 2 failed, 40 skipped, 69 subtests passed |
| Ruff（新增插件、测试、工具） / git diff --check | 通过 |
| 真实 D435 / 机器人运动 | 未执行 |

全量 ROI 分支失败：
`test_phanthy_music.py::MCPTests::test_real_http_tools_list_dispatch_and_log_redaction`（HTTP 超时）。原样上游也失败；上游另有 `test_music.py::PluginTests::test_playing_a_specific_track_still_works_for_stored_callers` 失败。两项在 ROI 分支单独运行时也失败；music 的整套与单独运行存在差异，未修改无关代码或隐藏失败。

早期运行还遇到沙箱禁止 socket、环境代理导致回环 HTTP 502、Chromium 本地库缺失。通过仅对本机测试解除 socket 限制、在测试 HTTP 客户端禁用代理、使用既有个人 Chromium 库后完成对应检查。浏览器测试适配器的异步等待和 Core content-list 响应格式已修正。详细失败与最终通过日志都在交付 verification 目录。

覆盖：变化/不变/不可靠图像、亮度补偿、相机位移、时间前后与重放、新帧稳定性、来源变化、VOP 独立、四步会话、MCP tools/list/call、真实指针拖框、三个预设、前后 ROI 图与导出、并发幂等、重置/关闭重试、请求冲突、过期与时钟回退、来源异常恢复、stop/start 退役。

当前是 Code Ready / Draft PR Ready，现场工具随后单独准备；正式验收需真实 D435 证据。工作台宿主测试使用官方未改 JS 与等价的 Core MCP HTTP 响应适配器，并未启动完整 Core 服务或共享 ROS 环境。
