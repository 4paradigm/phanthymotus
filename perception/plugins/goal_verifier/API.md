# API v1

单一 Tool `piper_object_goal_verifier`，`type: processor`。标准 HTTP MCP JSON-RPC 2.0：`initialize`、`tools/list`、`tools/call`、`resources/list`、`ping`。Card 内部不提供运动、使能、模式切换或机器人连接接口。

## Actions

| action | 必需参数 | 行为 |
|---|---|---|
| info / ui | 无 | 状态/来源诊断，或原生工作台HTML；info不泄漏会话token |
| start / stop | 无 | 启动自己的视觉订阅 / 释放自己的节点并退役所有本Card会话；不启动PiPER owner |
| observe | source_id | 读取新鲜视觉数据及当前多个检测；不自动开始任务 |
| get_preview | source_id, observation_id | 返回此精确观测的历史JPEG缩略图及原观测，缓存缺失返回null；不替换为其他帧，不参与目标计算 |
| submit_observation | observation | 默认禁止外部造入；配置可信规范输入或引用本Card已接收原样观测时可用 |
| start_session | request_id, task_id, source_id, selector, goal | 返回独立 session_id、随机 session_token、期限；可选 min_confidence `[0.1,1]`、max_displacement_norm `(0,1.4143]` |
| set_goal | 会话对, request_id, goal | 仅允许在操作前基准尚未设置时改目标 |
| capture_before | 会话对, request_id | 获取来源的新鲜基准；可选 observation，适用外部可信生产者/已接收数据引用 |
| mark_operation_complete | 会话对, request_id | 返回 marker_id / not_before；调用方先完成外部操作，再发此请求 |
| verify_after | 会话对, request_id, marker_id | 等待屏障后帧并计算；可选 observation 同上 |
| get_result | 会话对 | 最近一次判断，明确 verified_at；不是持续状态承诺 |
| export_session | 会话对 | 目标、基准、参与验证的前后观测、判据、历史和散列 |
| reset_session | 会话对, request_id | 原session_id保留，新token；旧请求/marker退役，清除基准和结果 |
| close_session | 会话对, request_id | 只终止该会话，其他会话不变 |

会话对指 session_id + session_token。task_id 是外部任务标签，不是可恢复权限。默认会话600s，最多32个，历史最多50次，已知观测/幂等缓存有界。服务重启不会恢复旧会话。

session_token在输入Schema标记为password/x-sensitive；这是表单/调用元数据，不能代替Core访问控制或保证平台所有请求日志已脱敏。凭据和私有生产者配置保存在owner文件中，不作为公开configSchema字段或共享Solution配置。

请求ID在会话generation内标识一次请求；成功重试得到相同历史结果。重置后的旧token即使带旧request_id也拒绝，不能取得旧PASS冒充新任务。reset回复丢失时可重新创建独立会话；不要复用旧token重试其他操作。

## Goal 与选择

```json
{"units":"normalized","frame_id":"实际光学坐标系","shape":"circle","center":[0.75,0.5],"radius":0.08,"tolerance":0}
```

上例只有软件图像区域，无设备/物理标定含义。rectangle 用 bounds `[xmin,ymin,xmax,ymax]` 替代center/radius。pixels使用实际图像尺寸；normalized使用 `[0,1]`，原点左上，x向右y向下。容差是圆的半径增量、矩形外部到矩形的欧氏距离；边界包含在内。目标条件为可见 bbox 中心，非整个物体包络、姿态或力状态。

selector 至少 `class_name`，可选 color。`detection_id` 必须同时带 `observation_id`，基准必须引用该精确帧；不能把另一帧中重用的检测索引当作同一物体。`track_id` 仅在owner确认来源的持续身份语义时允许。GUI对唯一类别/颜色选择自动用语义选择，对同外观多目标会保留精确帧ID；若用户填写参数耗时使选择过期，应重新观察。

plane_mm 使用实测配置中的 frame_id。不同view可选不同calibration_id；同相机/来源/epoch/畸变及实际标定绑定必须成立，跨view需要已验证持续身份。没有可靠标定时不能给出毫米偏差/PASS，而应改用可比较的图像目标或INCONCLUSIVE。

## Observation v1

字段必须与 advertised JSON Schema 完全匹配，不接受未知字段、NaN/Inf、布尔数字、缺字段、重复JSON键或非法bbox。

| 字段 | 语义 |
|---|---|
| schema | `motus.piper-object-observation/1` |
| observation_id | 生产者唯一观测标识，不是调用时重新生成的缓存ID |
| source_id / source_epoch / sequence | owner配置来源、生产者生命周期、该生命周期严格递增序列 |
| camera_id / frame_id | 已核对相机身份/光学坐标系；Topic名不是相机身份 |
| view_id | 拍摄几何/位姿注册标识；相机移动/裁剪不能沿用旧view |
| image_size | 实际像素 `[width,height]`，正整数，每边≤8192 |
| distortion | raw / rectified；两次须一致，不能改标签假装去畸变 |
| captured_at / timestamp_basis | 原采集Unix秒，basis固定capture；允许年龄默认2s、可配置0.05..5s，不接未来值 |
| provenance | real / synthetic / replay；真实配置默认拒绝后两者 |
| objects | ≤100项，每项 detection_id、class_name、confidence、像素bbox `[x1,y1,x2,y2]`；可选color、track_id、confidence_kind |
| track_epoch | 可信持续跟踪器的生命周期；stock VOP不能自己补这个字段 |
| calibration_id / image_sha256 | 可选真实标定引用/实际像素散列；散列不是身份或标定的替代 |

重放验证同时检查观测ID、时间、序列和来源epoch。不同相机/坐标系/畸变拒绝。图像模式下尺寸或view改变返回INCONCLUSIVE。过期、最新接收错误或无采集时间时不能退回旧缓存。若输入为历史录制数据，CLI replay 采用录制时间轴，保留时间戳并标明OFFLINE_RECORDED_TIMELINE，不向真实Card刷新时间。

## Result v1

`schema: motus.piper-object-goal-result/1`，主要字段：result、code、message、data_valid、session_id/task_id/piper_id、verified_at、before/after_observation_id、units、before/after_position、displacement、displacement_length、target_error、target_error_length、signed_region_error、inside_region、within_tolerance、criterion、association_method、evidence。

signed_region_error 为到圆/矩形边界的有符号距离，内负外正；target_error 是相对目标中心的二维向量。单位严格与goal相同。INCONCLUSIVE的不可证明数值保持null。HTTP/MCP参数错误返回 state:error/result:INCONCLUSIVE；有效输入但无法判定为 state:verified/result:INCONCLUSIVE。后一次采集/校验失败会覆盖“最近一次判断”，不会留下前一次PASS。

典型code：TARGET_NOT_OBSERVED、IDENTITY_AMBIGUOUS、IDENTITY_LOST、DISPLACEMENT_AMBIGUOUS、SELECTION_FRAME_CHANGED、STALE_OR_FUTURE、REPLAYED_OBSERVATION、PRE_OPERATION_OBSERVATION、CLOCK_ROLLBACK、VIEW_OR_RESOLUTION_CHANGED、SOURCE_GEOMETRY_CHANGED、CALIBRATION_REQUIRED、UNCERTAIN_BOUNDARY、SOURCE_TIMEOUT、SOURCE_INPUT_ERROR、RETIRED_SESSION、REQUEST_ID_CONFLICT、EXTERNAL_OBSERVATION_DISABLED。没有把“没检测到”标成FAIL。

所有可信性依赖已配置视觉生产者与测量记录。它不提供加密相机认证、长期重识别或生产级防篡改传输；默认回环监听，沿用Core访问控制，不将MCP标识字符串作为机械臂控制权限。

P0补强：不同request_id的后续验证必须消费比已验证后观测更新的ID/sequence/captured_at，旧后帧返回REPLAYED_AFTER_OBSERVATION；原request_id精确重试仍返回原结果，冲突请求不能覆盖该回执。新建会话时回收已到期会话；ROS直入与MCP对可选字符串/图像SHA格式保持相同校验。ROS合成/回放测试必须在来源配置显式声明provenance并启用allow_test_data；生产默认real并继续要求真实采集证据。

图像副本除每边≤8192外，还限制像素缓冲≤32MiB。默认类别/颜色关联依赖可区分且有限移动的场景，未处理帧之间的替换/遮挡不能保证检测；需要更强身份时须使用已验证的持续track来源。

export_session导出不包含session_token；start_session/reset_session仍向授权调用方返回会话能力。info/常规Perception日志也不输出该token。实际token应单独受限保存，不加入PR或公开截图；它不授予机械臂运动权限。
