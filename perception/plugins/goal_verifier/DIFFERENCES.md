# 官方功能审查与新增价值

审查基线：官方 `4paradigm/phanthymotus` main，`990257f881af540a1949237678b71dceaab5ab1c`，2026-11-07可访问版本。本Card复用已合并的官方模块，不依赖其他未合并Card。

| 官方能力 | 已有内容 | 本Card的差异 |
|---|---|---|
| perception/plugins/vop.py | 检测类别、像素框、相对位置和颜色；实时流/单图/URL推理 | 建立操作前基准、操作结束屏障、选择目标、跨操作保守关联、目标条件和三态结果 |
| perception/plugins/visual_depth.py | 单目深度/深度发布 | 本Card图像模式不依赖深度；不把未标定深度当毫米目标验证 |
| perception/plugins/measure | 可见尺寸/距离、短窗口中值稳定化、会话摘要和有界历史 | 原Stabilizer明确不是身份tracker。复用bbox规范化；不复用其短窗口key作为操作前后身份保证 |
| pick_place/操作完成与历史观测 | 本基线没有检索到完整的物体目标达成Card；运动完成、尺寸查询不是目标到位证据 | 本Card没有执行器，适用于PiPER推/放/其他位置调整后的独立条件验证 |
| Core canvas/workbench/MCP | 工具发现、参数、原生sandbox工作台 | 直接复用；新增只读processor入口避免开启整个项目，没有改运动授权或控制Owner |

相关官方PR：[measure #349](https://github.com/4paradigm/phanthymotus/pull/349)、[height #344](https://github.com/4paradigm/phanthymotus/pull/344)。前者介绍尺寸测量和短窗口历史，后者测高度；均不提供本Card完整的操作前后目标验证协议。可访问的PR搜索和本基线源码未发现相同完整功能；这不是对未来/不可见分支的保证。

新能力独立成立：同一PiPER上的不同物体操作都可以调用；不要求Visual Push、URDF、TCP、CAN或真实运动才能验收。范围专门服务PiPER，不添加跨机器人控制层或泛化硬件抽象。

VOP兼容性依据来自真实源码：`recognize_by_url`接受url，`image_input.py`不接受base64；返回source/image_size/objects。实时流timestamp为发布时间，无法直接证明采集新鲜度。新适配器使用有原始采集时间的图像及临时URL，或明确的标准观测生产者，不修改VOP字段去掩盖信息不足。
