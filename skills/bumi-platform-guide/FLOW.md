# 室内站台演示流程

此附加控制程序与手动问答 Skill 共用车厢配置，通过现有 MCP 与 ROS 接口运行。
**更新 SQLite 中的 Skill 不会启动此程序。** 不需要改 Driver 或 Core 镜像，但需把整个技能目录放到现有 Core 容器可读的位置，并显式启动。

流程：人物稳定出现1秒 → 录制101挥手 → 退出回放 → 欢迎语 → 等待ASR提问 → 查询已确认车厢表 → A方向播放104并配合语音 → 退出回放 → 安全提示完整播3遍（每遍间隔2秒）→ 等待下次提问。

- 101=挥手、104=指向A，来自用户确认；真机动作内容和朝向仍需现场验收。
- 画面连续无人5秒后重新允许迎宾。按“画面从无人变有人”去重，无人脸识别，不区分同一画面里的每位乘客。
- B方向只语音回答，不复用104。未知/越界车厢不做指向，不接三遍提示。
- 播报期间收到新提问会中断本轮剩余提醒，先回答；在迎宾/指向期间收到的问题依次处理，队列最多2条，不并行启动动作。
- “结束站台引导”“停止站台引导”“结束值守”、Ctrl+C或SIGTERM终止本流程；不自动恢复。不要用 `docker exec -d` 后失去停止入口。
- 只支持已确认的16车室内demo；不判断真实列车到站、开门，不做黄线检测。

## 真机准备

先保留 camera、vop、mic、ASR、tts、speaker 节点运行，**停止智能控制**。Core容器保持启动。演示期间独占动作和语音，不允许另一个画布、模型、录制工具或定时任务同时输出；程序发现Core恢复订阅会停止，但不能检测其他直接调用者。

把整个 `bumi-platform-guide` 目录复制到宿主挂载目录，例如 `/opt/phanthy-motus/data/bumi-platform-guide`，容器中对应 `/work/resource/bumi-platform-guide`。这不是复制数据库备份。保留 `site_config.local.json`，其方向必须已按README确认。以下命令在机器人宿主执行：

```bash
docker exec -it -w /work/resource/bumi-platform-guide phanthy-motus-agent-core-1 bash tools/run_flow.sh prepare --db /work/resource/data.db
```

prepare只读取DB与工具info，自动发现唯一的MCP地址和话题，生成 `flow.local.json`；不会播放动作、不会写数据库、不会覆盖已有流程配置。多实例时拒绝猜选。

若动作库已有当前机器人/编号匹配的人工测时证据，自动导入。否则输出 `missing_windows`。先现场完整回放101、104测时，再填写：

```json
"wave": {"recording_id": 101, "window_s": "填实测数字，不要加引号", "measured": true},
"point_a": {"recording_id": 104, "window_s": "填实测数字，不要加引号", "measured": true}
```

上面的片段是填写说明，不能原样运行。窗口从确认进入回放模式后计时，应覆盖整个录制。SDK没有可靠物理完成信号；窗口到期会请求退出并确认返回walking模式，**窗口完成不等于物理动作完成**。覆盖录制内容后必须重新测时。不要为了通过预检猜填秒数。

然后只做预检查：

```bash
docker exec -it -w /work/resource/bumi-platform-guide phanthy-motus-agent-core-1 bash tools/run_flow.sh check --db /work/resource/data.db
```

check核对车厢配置指纹、101/104实测窗口、智能控制已退订、感知运行状态、唯一TTS和speaker话题；不执行动作。它不能从SDK验证编号背后的动作内容或真实姿态。

准备好稳定站立和手臂活动空间，确认其他输出停止后，前台运行：

```bash
docker exec -it -w /work/resource/bumi-platform-guide phanthy-motus-agent-core-1 bash tools/run_flow.sh run --db /work/resource/data.db --exclusive-output
```

运行在Core现有Python/ROS环境，不在宿主额外安装依赖。进程退出会尝试退出已确认属于本流程的回放；请求超时而未收到启动确认时，不停止未知归属动作，应现场核对。SIGKILL/断电无法保证清理，重新启动前应检查实际状态。`--exclusive-output` 是操作员确认，不是硬件权限锁；进程锁仅防同一容器重复运行。

语音按实际收到非空PCM及EOF后推进，额外等待0.5秒缓冲；只说明音频流到达终点，**不能证明扬声器实际出声**。无音频、超时、MCP错误、动作启动/退出未确认或视觉数据中断时停止，不自动重试。使用ASR主输出（带audio_end_ts），丢弃旧帧/重复帧/本轮自身播报的回声。需要现场测试AEC；规则不能保证识别所有回声或任意自然语言。

## 验收顺序

1. 空画面时保持安静；仅检测到物品不迎宾。
2. 人进入画面：101一次，结束后欢迎语；人停留不反复挥手。
3. 说“请问五号车厢在哪边？”：104与A方向语音配合，退出回放后整段提示3遍；现场核对动作、听见语音。
4. 说“十二号车厢在哪边？”：B方向语音，不播放104；说“十七号车厢呢？”：越界说明。
5. 三遍播报时插入问题，取消剩余提醒；说“结束站台引导”，后续不再触发。
6. 连续无人5秒，再有人进入，可重新迎宾；摄像头断流不算无人，不自行重新触发。

本地模拟测试只证明逻辑和异常处理，以上真机验收尚未执行。
