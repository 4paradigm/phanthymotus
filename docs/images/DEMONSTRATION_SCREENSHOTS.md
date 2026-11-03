# 示范工作台截图来源

这些 PNG 是浏览器直接截图，没有经过生成式重绘或修改 UI。截图日期：2026-11-03。

- 实际运行原 Canvas、内嵌 iframe、Core API、独立 Perception HTTP 进程和视频编码。
- 视频由 FFmpeg `testsrc2` 合成；轨迹、相机和状态消息由测试产生。
- 模型目录、分析、技能提炼和语义比较使用明确的测试响应，不证明真实模型效果。
- 未连接实体机器人；截图不包含真实 API Key 或第三方视频帧。
- 测试等待短暂通知自然消失后截图，不隐藏页面元素。

## 复现

环境准备见 [使用指南](../DEMONSTRATION_WORKBENCH_ZH.md#9-开发者验证与截图复现)。从仓库根目录运行：

```sh
VIDEO_WORKBENCH_ARTIFACT_DIR=/tmp/motus-demo-screens python -m pytest agent-core/tests/test_video_workbench.py -k 'capture_card_in_original_canvas or enters_workbench_from_real_canvas' -q
```

来源：[test_video_workbench.py](../../agent-core/tests/test_video_workbench.py) 中两个实际 Canvas 测试及其技能流程。测试会生成下面的原始文件名；本文使用稳定的文档文件名。

| 文档图片 | 测试输出文件 | 页面内容 |
| --- | --- | --- |
| [demo-canvas.png](demo-canvas.png) | `demo-canvas.png` | 原画布中的两张卡片和测试连线 |
| [demo-video-entry.png](demo-video-entry.png) | `demo-video-entry.png` | 导入起点 |
| [demo-model-settings.png](demo-model-settings.png) | `demo-model-settings.png` | 模型设置，空密钥，测试目录 |
| [demo-multiview.png](demo-multiview.png) | `demo-multiview.png` | 辅助视角时间同步 |
| [demo-keypoints.png](demo-keypoints.png) | `v3-data.png` | 人工标记点 |
| [demo-motion.png](demo-motion.png) | `v3-trajectory.png` | 合成 UMI 数值轨迹 |
| [demo-steps.png](demo-steps.png) | `v3-task.png` | 步骤回看和编辑 |
| [demo-skill.png](demo-skill.png) | `v4-skill-extraction.png` | 固定模型响应生成技能候选 |
| [demo-skill-comparison.png](demo-skill-comparison.png) | `v4-skill-comparison.png` | 固定模型响应比较已有技能 |
| [demo-skill-task.png](demo-skill-task.png) | `v4-skill-new-task.png` | 新任务缺现场位置与绑定提示 |
| [demo-device.png](demo-device.png) | `v3-device.png` | 读取已连接工具和能力缺口 |
| [demo-capture-recording.png](demo-capture-recording.png) | `v5-capture-recording.png` | 合成 JPEG 和状态消息录制 |
| [demo-capture-saved.png](demo-capture-saved.png) | `v5-capture-saved.png` | 录制整理后保存到示范列表 |
| [demo-captured-motion.png](demo-captured-motion.png) | `v5-captured-motion.png` | 采集质量和动作状态 |
| [demo-review.png](demo-review.png) | `v3-review.png` | 参考与执行录像复盘，两者均合成 |
| [demo-skill-feedback.png](demo-skill-feedback.png) | `v4-skill-feedback-candidate.png` | 由未知结果反馈产生的新候选 |
| [demo-small.png](demo-small.png) | `v4-skill-small-canvas.png` | 窄屏内嵌技能页 |
