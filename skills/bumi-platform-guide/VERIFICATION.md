# 验证记录

日期：2026-10-22。平台仓库 `4paradigm/phanthymotus`，分支 `feat/bumi-platform-guide`，基线 `6df601f612818716bb0dc7b180a9467e2c43172c`，与任务书一致。本次增改仅在 `skills/bumi-platform-guide/`。

Driver 实际 HEAD `10fa798de1553c266ec989e847fa4f0583e0791c`，不同于任务书检查点；另有原先的 Dockerfile、exercise_routine.py、说明和 calibration 未提交变更。本次只读核对媒体工具，保留这些变更，没有继续旧健身任务。

## 自动化本地测试：已执行

命令：`python3 -B -m unittest discover -s skills/bumi-platform-guide/tests -v`。

提交PR前复核18项全部通过（0.469秒），包括真实命令行渲染/安装以及更新后的定向恢复。记录见同目录 `test-results.txt`。

覆盖：空配置和未确认方向可通用服务、已确认5/12/17车及覆盖缺口、编组/车次/锚点等变更后拒绝旧确认、非法配置、周期/视觉开关拒绝；安装幂等、冲突显式更新、更新保留安装时间/可见性/未知字段、卸载保留其他技能、缺 skills 行、错误路径/JSON/表结构/损坏DB、事务失败与触发器副作用回滚、写锁冲突、SQLite WAL 完整备份、定向恢复保留后来其他技能和目标变更保护。

平台兼容测试执行当前源码的 validate_local_skill（AST 抽取，HTTPException 为测试替身），实际加载 event/skills.py 和 prompt.py（仅配置/事件总线替身）：验证生成定义合法，完整DB对象不适合import，active只可见，运行时激活后完整instruction注入，停用后移除。**这不是运行完整 Agent Core HTTP/ROS 服务的端到端验证**。

## LLM行为样例：预期样例已编写，平台执行待测

`acceptance_cases.json` 每项包含场景、输入、预期及 status。是供真实平台复测的测试向量，不是模型调用日志。例：已确认模拟16车映射时，5车→A标识、12车→B标识、17车→越界解释；未确认则不指路；person检测不作为越线证据；结束后不自动恢复。

当前环境没有可用的完整 Core 运行依赖和已核实的目标模型会话。本次没有读取凭据或调用收费模型；不能以字符串测试证明平台模型一定遵循指令。

## 真机语音：未执行

源码已核对 perception/plugins/tts.py 的 speak/text、interrupt与stop区别、queued/action_id、audio/pcm-16k 输出；Driver noetix/bumi/device.py 的 speaker 接收PCM16 16k单声道。TTS speak复用第一个running节点的限制已写入指令和部署说明。

本机Docker socket无访问权限；未连接机器人，未知实际服务版本、DB_PATH、画布话题、模型ready状态、音量、实际播出与打断结果。按 README 第3至6节部署核对，并保存现场听到的证据后才算通过。测试未访问任何生产数据库。

## 真实感知：未执行且功能未启用

vop/pose源码有检测和关键点能力，但没有已标定站台区域与新鲜时序越线判断链路。本交付配置拒绝启用视觉越线；不把检测到person或旧图当成越线。周期同样未启用，不创建任务或订阅。


## 2026-10-23：101/104室内演示流程（新增）

新增 `tools/station_flow.py`、`tools/run_flow.py`、`tools/run_flow.sh`、`flow.example.json` 与 `FLOW.md`。原手动问答定义和数据库安装工具未改变；附加流程须独立显式启动。只读发现现有MCP与话题，复用101挥手、104指向A。用户报告已录制104，助手未验证机器人里的动作内容。

本次实际执行 `python3 -B -m unittest discover -s skills/bumi-platform-guide/tests -v`：**43 tests passed，1.370秒**（原18项＋新增25项）。新测试覆盖人物持续/消失/断流去重，旧帧/回声/重复ASR，101→欢迎语→104与A方向语音→三遍提示，B方向不调用104，17车越界，场景变更后禁用方向，插话取消剩余提醒，结束指令，MCP错误，空音频EOF，queued不等于完成，动作拒绝/停止和并行语音清理，speaker实际playing契约，多TTS拒绝与只读预检。

`bash -n tools/run_flow.sh` 和 `git diff --check` 通过。`flow-simulation.json` 由真实控制器处理模拟人物帧与ASR生成，明确标注未连真机；其模拟窗口不能用于现场配置。

此前用户报告真机语音和人工询问视觉结果成功；本次新流程没有连接机器人、没有播放动作、没有改机器人数据库或镜像，没有推送PR。101/104实测窗口、真实ROS订阅、ASR/AEC、动作朝向、扬声器出声与整套联排均待现场验收。未填猜测时长；没有实现黄线检测。
