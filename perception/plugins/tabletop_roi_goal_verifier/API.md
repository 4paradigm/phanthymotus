# MCP API

唯一工具：`tabletop_roi_goal_verifier`。所有调用通过 `tools/call` 的 `arguments.action`，精确 Schema 由 `tools/list` 返回。独立服务还提供 `/health` 和 `/` 页面。

| 动作 | 必需字段 | 说明 |
|---|---|---|
| info / ui / start / stop | 无 | 状态 / Workbench / 开始图像订阅 / 关闭并退役会话 |
| observe | source_id | 新近的相机观测 metadata，不运行检测 |
| get_preview | source_id, observation_id | 与观测绑定的历史缩略图，最多缓存 64 帧 |
| start_session | request_id, task_id, source_id, target_roi | ROI 为归一化 `[x1,y1,x2,y2]`；可选 roi_options |
| capture_before | request_id, session_id, session_token | 记录真实图像基准 |
| mark_operation_complete | request_id, session_id, session_token | 设置采集时间下界；不执行运动 |
| verify_after | request_id, session_id, session_token, marker_id | 等待至少两帧不同且晚于标记的新图像 |
| get_result / export_session | session_id, session_token | 当前结果 / 证据（导出剔除 token） |
| reset_session / close_session | request_id, session_id, session_token | 重置换 token / 关闭会话 |

`roi_options`：change_threshold `[8,80]`，min_change_fraction `[.005,.5]`，stable_frames `[2,5]`，stability_threshold `[2,40]`。默认 `25,.04,2,8`。没有 object/selector/detection/goal/observation/VOP 参数。

结果 Schema：`motus.tabletop-roi-goal-result/1`。包含 result、code、data_valid、criterion、verified_at、provenance、target_roi、统计、ROI JPEG 和 evidence。evidence 保存实际 before/after metadata（SHA256、采集时间、source epoch、sequence、身份绑定）、marker 和选定参数。异常恢复后的新请求可重试，失败不会留下旧 PASS 作为最新结果。

典型错误：SOURCE_TIMEOUT、SOURCE_INPUT_ERROR、STALE_OR_FUTURE、PRE_OPERATION_OBSERVATION、REPLAYED_AFTER_OBSERVATION、SOURCE_GEOMETRY_CHANGED、ROI_NOT_STABLE、ROI_CAMERA_MOVED、ROI_OCCLUDED_OR_REPLACED、REQUEST_ID_CONFLICT、SESSION_EXPIRED。错误不映射成任务 FAIL，可靠数据中的 ROI_NO_CHANGE 才是 FAIL。

会话 token 用于隔离本进程会话；服务仅供同用户 loopback / SSH 隧道使用。Bundle 请求/结果日志会递归隐去 session_token；独立服务不记录请求正文。图像、会话和诊断仍按个人证据管理。


审查新增拒绝码：OPERATION_CANCELLED、START_CANCELLED、SESSION_CLOCK_INVALID、IMAGE_SATURATED_OR_EMPTY、ROI_PHOTOMETRIC_AMBIGUITY、EVIDENCE_INCOMPLETE、EVIDENCE_IMAGE_INVALID。来源恢复后 source_epoch 变化会触发 SOURCE_GEOMETRY_CHANGED。`_trace_id` 接受 1–256 字符串，仅作诊断，重试时改变它不会改变逻辑请求身份。证据缺失或写入失败不表示验证成功；失败的新后观测不导出上一轮 after 图。
