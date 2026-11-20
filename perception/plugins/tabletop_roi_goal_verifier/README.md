# Tabletop ROI Goal Verifier

独立的操作结果视觉验证 Card，MCP 工具名为 `tabletop_roi_goal_verifier`。固定相机拍摄桌面，框选一个目标 ROI，记录操作前图像，再比较操作完成标记之后的新图像。无需 VOP、YOLOE、类别、检测框、深度或机器人驱动。

- **PASS**：ROI 有足够明显、连续新帧中稳定的局部视觉变化。
- **FAIL**：图像有效且可比较，ROI 没有达到变化条件。
- **INCONCLUSIVE**：遮挡、模糊、相机/视角变化、时钟不匹配、旧帧、缺少帧或边界变化等，无法可靠判定。

PASS 不证明物体身份相同，不证明机器人真实运动，也不提供物理位置或毫米精度。固定的手或局部遮挡可能与物体放入无法区分，必须移开手并现场复核。

## 启动独立服务

使用 Python 3.10+（现场为 Ubuntu 22.04 / ROS2 Humble）。在个人隔离环境安装 `requirements.txt`，source 现场已有 ROS setup，然后填写 `deploy/config.template.json` 的空项。不得用占位文字伪装已核实的相机身份、固定视角或时间语义。

```bash
export ROS_DOMAIN_ID=42
export FASTRTPS_DEFAULT_PROFILES_FILE="$PWD/agent-core/deploy/dds-local.xml"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export PYTHONPATH="$PWD/perception${PYTHONPATH:+:$PYTHONPATH}"
python3 -m plugins.tabletop_roi_goal_verifier.server --config /path/to/personal-config.json --port 18084
```

浏览器打开 `http://127.0.0.1:18084/`；MCP 地址为 `http://127.0.0.1:18084/mcp`。仅绑定 loopback。页面具备轮询实时 RGB、拖框、三个预设、四步流程、变化统计、前后 ROI 图、错误码和 JSON 证据导出。图像始终带采集时间和 provenance；预览帧不直接作为验证结果。

## Perception / Core

注册入口在 `perception/main.py`，默认关闭，不修改共享 `config.yaml`。在自己的配置 `plugins` 中设置以下内容，并填写与独立配置相同的 `sources`：

```yaml
plugins:
  tabletop_roi_goal_verifier:
    enabled: true
    verifier_id: personal-tabletop
    max_age_s: 2
    session_ttl_s: 600
    sources: {} # 必须填写已核实的 ROS 图像绑定
```

启动 Bundle 只公布工具；`start` 动作才创建订阅。上游已有 `motus.workbench/1` 页面宿主。将 Card 的 action 设为 `ui` 可打开四步工作台。无需 #374 的侧边栏入口；也可直接用独立服务页面，无需 Core。

## 算法与边界

图像最多缩到 640×480，ROI 外上下文补偿全局亮度，检查上下文改变与可测背景位移；ROI 内做平滑差分、连通区域面积和新帧间稳定性检测。三种预设为宽松 `(15, 0.02, 2)`、标准 `(25, 0.04, 2)`、严格 `(40, 0.08, 3)`，依次为像素阈值、最小变化比例、帧数。至少两帧；不应把有限时间窗稳定解释为长期稳定。ROI 每边至少 8 个处理像素，外部上下文至少 64 像素。大于 80% 的 ROI 变化保守返回 INCONCLUSIVE，过大目标也可能触发此条件。

相机必须固定，目标 ROI 外应保留可见背景。无纹理桌面无法可靠恢复相机位姿。切换分辨率、source epoch、frame/view/provenance 都要求重建基准。固定视角记录与采集时间记录是操作者配置声明，并非算法独立证明。相机序列号不能从 `header.frame_id` 推导。

默认仅接受真实 ROS 图像。离线接口仅在 `kind=buffered_image, allow_test_data=true, provenance=synthetic|replay` 时可用；没有公共像素提交 MCP 动作。ROS 测试数据也必须显式标记和启用。采集时间直接来自 ROS header，不能用收到消息的时间替代。

## 会话与测试

最多 32 个会话，默认有效期 600 秒；请求回执上限 256，历史结果上限 50。操作完成标记后 120 秒内验证，默认帧年龄上限 2 秒；新请求不能复用旧后观测。相同 `request_id` 与同参数返回原回执，冲突参数拒绝，重置/关闭回执可重试；回执幂等只在本进程有限缓存内保证。停止服务清理会话和图像，重启必须新建会话。

```bash
ROI_TEST_PYTHON=python3 bash perception/plugins/tabletop_roi_goal_verifier/deploy/run-tests.sh
python3 perception/tools/tabletop_roi_browser.py --output /tmp/roi-browser --chromium /path/to/chrome
```

测试依赖：pytest；浏览器测试另需 playwright 和 Chromium。浏览器测试只使用合成图像，并测试独立页面与未修改的上游 Workbench 宿主。测试日志、PR 正文和独立性检查在本次交付目录；硬件结果需另行采集。

## 与 Object Goal Verifier / region_presence 的区别

不依赖 `plugins/goal_verifier`，可先于 #374 合并。Object Card 检查被选择物体的目标位置；本 Card 判断区域前后视觉状态变化。它不做单帧物体在场检测，也不包装 VOP。当前上游代码搜索没有发现 `region_presence` 标识；如未来加入相似能力，应按输入证据与跨操作会话语义比较。

若将来决定整合到 Object Card，可以迁移 `roi.py` 与独立 Sources/Engine 的 ROI 调用；本次不执行备用整合，也不承诺两个接口可直接互换。

## D435 现场 kit

一键安装、启动/停止、只读 Doctor、Windows SSH 转发和三态采集步骤见 [deploy/field/ONSITE.md](deploy/field/ONSITE.md)。现场模板的序列号/topic 是历史期望；相机身份、固定视角和采集语义必须现场核实，记录字段留空时拒绝启动。

kit 包含 CPython 3.10 的 x86_64/aarch64 离线 wheel 和 pip bootstrap，无需系统 pip、ensurepip、sudo 或公共服务。默认新目录、端口 18084，冲突时选择后续空闲端口。独立页面可将本次结果、ROI 图、参数、时间、session 和页面日志保存到个人 evidence/，去除 session token。只会停止本目录记录且进程身份匹配的服务。

从预先下载的 wheel 目录重建交付：

```bash
python3 perception/tools/package_tabletop_roi_field.py --wheels-dir /path/to/wheel-input --output /path/to/new-artifacts/d435-field
```

现场软件工具已准备；现场 Domain 222 相机尚无合规的 Domain 42 数据入口，不能直接连接或宣称现场运行 Ready。未完成真实 D435 验收；aarch64 只准备了匹配 wheel，尚无该架构运行记录。

## 2026-11-13 审查收尾

专项审查与当前测试结论见 REVIEW.md / TESTING.md；旧交付数字不代表本次结果。相机错误恢复、采集间断或壁钟回退后需新建会话，停止可抢占等待图像的请求。图片编码失败不会留下半个基准，失败的后观测不会借用旧截图；证据写入使用临时目录和原子发布。

新增 `ROI_PHOTOMETRIC_AMBIGUITY`：纯局部阴影、反光等中性亮度变化保守返回 INCONCLUSIVE。纯灰度物体也可能触发此规则；不要放宽阈值来强求 PASS。已识别的变化不等于物体身份或机器人成功。Core 的 `_trace_id` 仅用于诊断，不参与幂等身份；session token 不写入 Bundle 请求/结果日志。

新一版现场安装建议使用 `/home/agilex/goal-verifier-roi-20261114-r1`。旧目录和旧服务不覆盖。本地修复可供现有 PR #393 审查；远程状态未变，真实 D435 验收及 Domain 42 数据接入尚未完成。

## PR #393 兼容修复（2026-11-14）

官方 JetPack 5 镜像使用 Python 3.8；算法及 Doctor 已移除 Python 3.10 的 pairwise 依赖，独立现场 wheel 仍专供 Python 3.10/Humble。Doctor 显式用同一个私有 Context 和 Executor，避免默认未初始化 Context 的 `__enter__` 异常。Engine 在完成取帧后的校验入口重新读取主机时间，不使用请求时间或帧自身时间作为 now，原有重放/marker/过期/回退保护保留。

配置、Runner 和 Doctor 强制 Domain 42。现场相机在 222 的历史事实保留为接入阻碍；没有自动桥接或 Domain 222 例外。详见 TESTING.md、REVIEW.md 和现场说明。
