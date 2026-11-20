# U1 单人近似方向实验

沿用 Bumi 两个输入：/vision/mic/audio → ASR audio/pcm-16k；/vision/sound_direction → ASR sensor/sound-direction；ASR 主输出 /vision/mic/audio/asr → Agent Core decision_core 输入。使用 asr_kws、小范小范，测试完整句“小范小范，你知道我在哪吗？”。B 版方向门控仍 -65 dBFS、margin 2.5dB。

默认严格采集窗口路径不变。U1 尚未验证共享采集窗口，因此明确取得用户许可后，只在实验 perception 进程设置 U1_ASR_RECEIPT_DIRECTION_EXPERIMENT=1。来源必须声明 native / receipt_time_only / association_state unknown / 零采集窗口；消息时间戳应在接收时1秒以内，心跳按原事件时间去重。

只支持一个人、句首唤醒词。按可信音频起始时间加真实 pre-roll 时长确定发声起点；在起点后前2秒且不超出本句结束范围内至少有两个独立新方位事件，全部与圆周均值相差不超过25度才输出 speaker_direction_state=estimated，speaker_direction 明确带“估计”，basis/说明保留单人和非采集对齐限制。没有新事件、方向冲突、音频时间不可信、唤醒词前有其他词均 unknown。不伪造音频窗口、不返回 valid、不以ASR完成时最新角度作匹配。Agent Core 现有 collector 会保留这些语义字段；无需额外实现ASR或改Agent Core推理。

实验仍可能因未知厂商延迟或方向相近的多人而误关联，只能作单人相对方向测试，不提供距离/房间坐标/身份。Agent 应回答“大概在我左/右/前/后方”，未知时明确说不能确定；禁止据此自动转头或动作。

现场补丁只替换已确认与本地一致的 /work/plugins/asr.py，备份源文件和compose，实验环境开关和只读源挂载均可撤回。ASR空闲后才更新，重启 perception 会使该容器内其他感知卡片重连，需重新启动画布卡片。不要把它称为已完成多人或精确时间关联。
