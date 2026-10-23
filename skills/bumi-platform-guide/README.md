# Bumi站台引导员 1.0.0

新增：用户指定的“检测到人 → 101挥手 → 迎宾 → 车厢问路 → 104指向A并播报 → 安全提示3遍”已有独立、显式启动的室内演示控制程序，见 [FLOW.md](FLOW.md)。下文1.0.0指手动问答Skill；仅更新安装指令不会启用附加流程。附加流程尚未真机验收。

Phanthy Motus 平台 Skill，slug `bumi-platform-guide`，存入 Agent Core SQLite `config.skills.installed`。固定位置提供中文安全提醒和车厢问答。不是 Codex 个人技能，也不需要构建 Driver 镜像。

## 当前交付状态

- **基础版本地实现完成**：完整中文指令；铁路/地铁配置；确认指纹；便携导入定义；离线数据库检查、安装、更新、卸载、定向恢复；本地测试。
- **语音接线已按源码核对，真机待验收**：依赖 perception `tts`，音频接 Bumi `speaker`。未连接当前机器人、未发现现场工具实例、未实际播放，不宣称已经播出。
- **值守意图是会话规则**：开始、暂停、恢复、结束、单次播报和状态查询由模型按 instruction 处理；没有独立状态工具或持久化状态机。新会话/重启默认未开始；模型执行效果待平台验收。
- **周期播报未启用**：配置为 true 会被拒绝。没有创建任务、cron 或监听器；结束后不会有本技能创建的残留广播。
- **视觉越线检测未启用**：配置为 true 会被拒绝。人工事件只在来源可信时使用，无法确认则给通用提醒。

普通安全提醒在空配置时也可使用。默认样例是未经确认的模拟站台，导入后不能直接回答具体车厢方向。

## 1. 在电脑验证

已有本机仓库可直接使用。首次下载平台时：

```bash
git clone https://github.com/4paradigm/phanthymotus.git
cd phanthymotus
```

测试尚未合并的版本时，使用 PR 的源分支；如果当前检出版本没有本目录，可将交付包中的 `bumi-platform-guide` 目录复制到 `skills/`。

以下命令从平台仓库根目录开始。Python 3.10+，本交付工具仅用标准库，不依赖 ROS、FastAPI 或额外 pip 包。

```bash
python3 -B -m unittest discover -s skills/bumi-platform-guide/tests -v
cd skills/bumi-platform-guide
python3 -B tools/render_skill.py --config site_config.example.json
```

省略 `--output` 仅检查，输出包含规范化配置、完整指纹、有效方向表和能力状态。默认不写 DB。`skill.json` 是铁路模拟样例生成的便携定义，包含完整指令；不是带 active/installedAt 的数据库对象。

## 2. 先配置，再生成

```bash
cp site_config.example.json site_config.local.json
```

编辑本地配置：站名、站台、车次、编组、固定锚点、区间、`scene_id`、`config_version`。地铁请从 `site_config.metro.example.json` 开始，不能沿用铁路16车映射。

确认具体方向分三步：

1. 先保留 `staff_confirmed=false`、`confirmed_scene_id=null`、`confirmed_config_sha256=null`；运行上面的配置检查命令，把文件名换成 `site_config.local.json`。
2. 操作员核对固定锚点和每个区间后，将 `staff_confirmed` 改为 true，`confirmed_scene_id` 填当前 `scene_id`，`confirmed_config_sha256` 填检查结果中的 `confirmation_sha256`。
3. 再生成：

```bash
python3 -B tools/render_skill.py --config site_config.local.json --output skill.local.json
```

模拟确认仍只用于演示，回答应带“模拟场景”。现场需 `mode=live`，并填写站名、站台、车次后重新确认。确认指纹用于发现配置变更，**不提供人员身份认证**。换场景/车次/编组/锚点/映射时旧指纹会失败；应先取消确认、递增版本、重新检查和确认。普通乘客对话不能更新配置。

覆盖不全允许生成，但未覆盖车厢不会进入有效方向表。重叠、越界、布尔车厢号、重复 JSON 字段等会拒绝。周期/视觉开关必须 false，`routine_interval_s` 是未启用的预留数据，不会创建定时任务。

模型只读取已安装的渲染结果，不会自动重读电脑上的配置文件；更新配置后必须重新生成、更新安装、重启 Core 并开始新会话。

## 3. 核实真正使用的数据库

不要把源码目录当作真机 DB，不要创建一个空文件冒充部署数据库。先在实际部署宿主机检查：

```bash
docker ps --format '{{.Names}}\t{{.Image}}'
```

手工将 `CORE_CONTAINER` 设为上一步确认的 Agent Core 容器名，再执行以下只读检查：

```bash
CORE_CONTAINER='填写实际容器名'
docker inspect "$CORE_CONTAINER" --format '{{.Config.WorkingDir}}'
docker inspect "$CORE_CONTAINER" --format '{{range .Config.Env}}{{println .}}{{end}}' | rg '^DB_PATH='
docker inspect "$CORE_CONTAINER" --format '{{json .Mounts}}'
```

只记录 DB_PATH、工作目录、对应挂载和镜像版本；不要分享整个环境变量输出。若没有 DB_PATH，源码默认是相对工作目录的 `./resource/data.db`。按容器路径与挂载映射确定宿主文件。

当前源码部署模板给出 `DB_PATH=/work/resource/data.db`，挂载 `/opt/phanthy-motus/data:/work/resource`；这是**模板证据，不是当前真机检查结果**。本次未连接机器人。本机 Docker socket 无访问权限，未取得运行容器信息；不据此推断机器人部署状态。

将整个 Skill 目录复制到数据库所在宿主机，确保有 Python 和 DB 目录写权限。使用实际宿主路径：

```bash
GUIDE_DB='/填写实际宿主机路径/data.db'
python3 -B tools/install_skill.py --db "$GUIDE_DB"
python3 -B tools/install_skill.py install --db "$GUIDE_DB" --skill skill.local.json --dry-run
```

`--db` 必填，默认 action 是 check；不存在的路径不会隐式创建。dry-run 检查定义和事务合并，不改库、不备份；也不证明部署已停止或语音已就绪。

## 4. 离线安装、更新和卸载

先停止**已核实的实际 Agent Core**。关闭浏览器页面不等于停止服务；`--core-stopped` 只是操作员确认标记，不会替你停止容器。即使此版本 ConfigDB 按次读取 DB，运行时还有缓存的 prompt、会话与并发配置写入，仍须停服修改并重启。

```bash
docker stop "$CORE_CONTAINER"
python3 -B tools/install_skill.py install --db "$GUIDE_DB" --skill skill.local.json --core-stopped
python3 -B tools/install_skill.py check --db "$GUIDE_DB"
docker start "$CORE_CONTAINER"
```

操作会先验证表结构和 JSON，再在 `BEGIN IMMEDIATE` 中按准确 slug 合并。首次安装产生 UTC `installedAt` 和 `active=true`，后者只表示可见。相同定义重复 install 不修改；冲突必须显式 update。每次真正变化前用 SQLite backup API 备份，包含 WAL 中的数据；备份在 DB 同目录、权限0600。备份含整个库的敏感配置，不要打包或上传。

更新时同样先停 Core；目标 `installedAt`、`active`、未知元数据及其他技能、同级未知字段保持。示例：

```bash
python3 -B tools/install_skill.py update --db "$GUIDE_DB" --skill skill.local.json --dry-run
python3 -B tools/install_skill.py update --db "$GUIDE_DB" --skill skill.local.json --core-stopped
```

卸载也先停 Core：

```bash
python3 -B tools/install_skill.py uninstall --db "$GUIDE_DB" --dry-run
python3 -B tools/install_skill.py uninstall --db "$GUIDE_DB" --core-stopped
```

本版配置嵌在目标 instruction 中，没有本技能创建的任务、文件传输或其他 DB 键；卸载精确删除该项即可。它不清理别人手工添加的同名 cron 或其他旧版本的附件。发现同 slug 多条记录时拒绝所有操作，需先人工核对。

写后会重读验证目标及其他配置未变化；事务内失败回滚。无变化不产生备份。脚本输出只含状态、指纹和备份路径，不打印其他技能内容。记录成功输出中的 `backup` 和 `after_sha256`。

## 5. 只恢复本次目标技能

停止实际 Core 后，用上次输出的备份和 after_sha256：

```bash
python3 -B tools/install_skill.py restore --db "$GUIDE_DB" --backup '/实际备份路径.bak' --expect-current '上次after_sha256' --dry-run
python3 -B tools/install_skill.py restore --db "$GUIDE_DB" --backup '/实际备份路径.bak' --expect-current '上次after_sha256' --core-stopped
```

只恢复备份中的目标技能，保留之后新增的其他技能和配置；目标有后续修改则拒绝覆盖。原先没有目标技能时恢复等于移除它，原先缺 skills 行时可留下空 installed 行。不会把旧整库备份覆盖当前生产库。恢复本身也有备份。完成后查询并启动 Core，重新检查可见性及运行时激活。

## 6. 画布与真正的语音

安装后，在“设计/设置 → 技能”核对可见。系统实际工具 `activate_skill(slug="bumi-platform-guide")` 调用后才注入完整指令；界面勾选不是开始值守。开始前核对本版本平台工具和完整 schema，可在已登录页面开发者工具中查看 `GET /api/mcp` 及相应服务 `GET /api/mcp/{id}/tools`，不要假定实例 ID/话题固定。

搭好实际链路：

```text
remote_message → Agent Core （先行调试）
Bumi mic → ASR → Agent Core （乘客语音）
Agent Core → perception TTS speak → audio/pcm-16k → Bumi speaker
```

- 依赖声明 `requiredTools=["tts"]` 来自源码插件名；Core 可将其拆成 speak/interrupt 并加前缀。按发现结果使用，不照抄虚构的完整 MCP 名。
- 原始调用是 `tts(action="speak", text="请在黄色安全线的安全侧候车。")`；实际拆分后的工具可能只接受 text。`interrupt` 停当前音频，`stop` 关插件节点，不能混用。
- 检查 TTS 模型 ready、输出实例和实际 `topic_out`；Bumi speaker 的 `input_topic` 必须接同一 `audio/pcm-16k` 流。speaker 接 PCM，不接 text。
- 当前源码 TTS speak 会复用服务中第一个 running 节点；应保证该 TTS 服务仅有一个有效节点接目标 speaker，不能以 instance_id 作为隔离保证。
- 画布只向 Agent Core 绑定所需语音能力；核对模型可见工具没有移动、起身、录制、健身编排、舞蹈等执行器。requiredTools **不是权限沙箱**。共享系统的全局打断和自动进展播报由部署人员核对，不由本 Skill 擅自切换。
- `queued/action_id` 只是入队。核对 mouth 完成回执并现场听见才算播出；不能只截终端文本。中断只在可核实当前输出属于本技能时使用，不停止共享节点。

推荐依次发送：

1. “开始站台引导。播报一句候车提示。”现场听见短句。
2. “5车怎么走？”默认未确认样例应拒绝猜方向；安装已确认模拟映射后应答“模拟场景，5号车厢请沿A标识方向前往”，并说明参考方向。
3. “12车呢？”应按 B 标识；“17车呢？”应说明超出16车配置并建议看屏或问工作人员。
4. “模拟操作员指令：切换上下车提醒，播报一遍。”应有先下后上和注意间隙，不宣布开门。
5. “暂停例行播报”→“恢复”→“结束值守”→“恢复”。明确本版无后台周期；结束后不自动恢复，仍可回答被明确询问的问题。
6. “去黄线边巡逻”。应保持固定位置，无运动工具调用。

TTS 未就绪/断开 speaker、插话、重复开始、人员报告和无视觉时的行为见 `acceptance_cases.json`。保存工具结果、时间、实际听感和必要截图，避免保存乘客图像或敏感配置。未通过语音链路不能宣布基础真机验收完成。

## 7. 周期与视觉的准确缺口

当前 `scheduler.py` 支持五字段分钟 cron、动态 add/remove；但没有本技能可调用并核验的值守控制、任务归属、触发时状态门控、过期丢弃、轮换与取消闭环。`task_create` 只追踪任务，没有 check_cron 入参。单纯写静态 cron 文本不能保证结束后无广播，因此本版明确不接入。

`vop` 提供对象类别/位置；`pose` 提供 COCO 关键点及动作。没有经过确认的站台地面危险区域、落脚点映射、时序越界判断、标定失效检测、新鲜度和冷却链路；person 不代表越线。本版不添加视觉依赖，保留可信人工报告及通用提醒。

## 8. 交付与证据

- `instruction.md`：完整中文角色、规则、话术及语音边界。
- `skill.json`：可导入便携定义；现场生成用 `skill.local.json`。
- `tools/render_skill.py`、`tools/guide.py`：配置校验/确认指纹/渲染。
- `tools/install_skill.py`：安全离线数据库操作与选择性恢复。
- `tests/test_guide.py`：配置、真实 SQLite/WAL、失败回滚、平台导入及激活注入验证。
- `acceptance_cases.json`：平台 LLM 与真机复测用例，附预期，不伪装执行结果。
- `VERIFICATION.md`：本次实测与待测分开记录。
- `marketplace-draft.md`：技能市场填写草稿，未提交；作者和真机证据待补。

本交付通过代码 PR 审阅；技能市场发布和真机安装需要分别执行，无需沿用 Card 的镜像流程。最终验收还需当前 Core 地址/版本、DB 映射、实际工具/画布接线、现场配置和可听见的真机语音证据。
