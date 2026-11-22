# 提取前检查

工作区上层存放多个仓库和工作树。主目录的 PhanthyMotus 检出在 `fix/piper-card-count`（`73b7a9f`，干净）。旧马赛克实现位于 `feat/pr382-mosaic-update`（`c8b6b15`，干净），GitHub PR #382 的 head 与该提交一致。PR 标题仍是 Privacy View，但其实际差异包括 `969189d` 的 YuNet + LK demo、`335bf4a` 模型预配、`53043c8` ORT 兼容和加载进度，以及一项 VITS2 测试提交。独立 demo 分支 `6fd096a` 基于本地已同步的上游 `a642f8a`，缺少后两项模型改进。旧 Privacy V2 工作区有进行中的修改。新工作树 `feat/yunet-lk-mosaic-final` 从 `a642f8a` 开始，只提取本组 YuNet + LK 路径。 [PR #382](https://github.com/4paradigm/phanthymotus/pull/382)

| 功能 | 已有实现 | 当前问题 | 是否需要修改 | 复用位置 |
| --- | --- | --- | --- | --- |
| YuNet 检测 | `process → _detect → FaceDetectorYN`，旧 OpenCV 可走 ORT CPU 子进程 | PR 分支依赖其改过的 ORT API | 提取旧分支的 worker 内会话工厂扩展，保留 OpenCV 优先 | `mosaic_smooth_runtime.py`、`yunet_ort.py`、`utils/model_downloader.py` |
| LK 光流 | YuNet 确认后在框内选角点，间隔帧用 LK 位移中位数更新位置 | 多张脸共用一个位移。检测帧漏检时角点停在上一帧，灰图却前进 | 按人脸分组，同批 LK，保持角点和灰图同步 | `mosaic_smooth_runtime.py:SmoothMosaic.process` |
| 全局马赛克 | 有可靠人脸时背景做 2px 整帧马赛克，其他情况做 40px 整帧马赛克 | 正常无脸也改变背景。模型故障通过异常走状态而非正常检测 | 移除两条全局路径。延用异常不发布图像 | `_whole_mosaic` 和 `_DemoNode._run` |
| 马赛克扩框 | 局部框每边扩大 35%，裁剪后以 24px 像素块处理 | 跟踪不稳与识别正常始终同一扩框比例 | 复用局部渲染，按可靠性切换扩框比例 | `_mosaic_face` |
| 漏检补偿 | 复用历史框，在 550ms 内维持光流跟踪 | 仅检查整组点，也无法识别单张脸的漏检 | 按脸保留短时目标，光流失效后只短暂保留最后可靠位置 | `COAST_LIMIT_S` 和 `SmoothMosaic.process` |
| 延迟发布 | 单槽最新帧，处理后立即发布 | 无限时待发布队列，也没有历史位置修正 | 新增有界发布缓存。只用可靠的前后检测与连续跟踪做回溯修正 | 复用 ROS 帧头和源时间戳 |
| 自适应检测 | `(number - 1) % detect_every_frames == 0`，默认 4 | 无运动或质量触发，也没有检测空画面的专门策略 | 保留固定模式。自适应复用当前 LK 结果并限制最长间隔 | `SmoothMosaic.process` |
| 光流质量检查 | LK `status`、最少 7 点、位移各轴小于 40px、550ms 上限 | 无误差、有效比例和运动一致性检查，全组失败 | 复用 LK 参数，加每脸质量过滤 | `goodFeaturesToTrack`、`calcOpticalFlowPyrLK` |
| 检测置信度 | 默认 0.28，可配置 0.1–0.9。OpenCV 内部阈值固定 0.22，再过滤结果 | 配置低于 0.22 时部分区间实际无效 | 使参数直接驱动两个后端，保留 0.28 基线 | `DemoConfig`、`YuNetOrtService` |
| 过期帧丢弃 | ROS 回调拒收旧帧和乱序帧。最新帧覆盖未处理输入，处理前后都有年龄检查 | 覆盖和过期丢弃未分开统计，无延迟缓存 | 复用上游 `LatestFrame`，新增时间、帧数和字节上限及计数 | `_DemoNode._on_frame/_run`、`utils/latest_frame.py` |

Bumi V3 独立脚本也使用全组位移、固定 4 帧、35% 扩框、550ms 补偿和两种全帧马赛克。它额外含有独立 HTTP 演示界面和全局 OpenCV 线程设置，这些不属于新 Card。旧日志记录了 Bumi 实际摄像头 21.33–29.98 FPS，主要输出整帧粗马赛克。另有 x86 160 帧离线重复输入、旧卡合成 ROS2 DDS 测试和已校验的模型文件。历史数据的材料、统计口径与本轮不同，仅用于参考。本轮在写 Card 代码前重新测旧版。

注册沿用 Perception `main.py` 的 opt-in processor 加载、`get_tools/dispatch/shutdown` 和 YAML 配置。MCP 使用 `start/stop/info/config`，输入是项目摄像头的 `sensor_msgs/CompressedImage` JPEG，输出 JPEG 与 `std_msgs/String` JSON。用现有 `CAMERA_QOS`、`dispose_node`、`LatestFrame`、`ensure_verified_bundle` 和共享 ORT worker。Perception 镜像已复制整个 `plugins/` 并提供 OpenCV、NumPy、ROS2 和 ORT。
