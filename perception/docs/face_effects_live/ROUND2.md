# 第二轮：自然视频质量及本地平台验证

日期：2026-11-17。开发仍在独立`feat/face-effects-live-v1`分支，无提交、推送、PR或发布。原型已推进到自然视频和真实本地平台链路验证；**仍不能宣布完整V1.0离线质量验收通过**。所有结果属于WSL2 Linux x86_64 CPU环境，未操作机器人。

## 输入、许可和标注边界

真实视频来自[White House采访的Commons来源记录](https://commons.wikimedia.org/wiki/File:The_YouTube_Interview_with_President_Obama.webm)和[Obama Presidential Library录像来源记录](https://commons.wikimedia.org/wiki/File:President_Obama_welcomes_President-Elect_Trump_to_the_White_House_(25-12322-F).webm)。来源页声明美国联邦政府作品公有领域；本轮只选原始政府拍摄画面，采访74s之后，避开第三方片头。没有上传摄像头素材或下载受非商业模型限制的权重。完整文件、来源HTML及模型卡在本机证据目录，哈希见`evidence/round2/sources.json`。

8段各240帧，共1,920帧，约64秒：

| 片段 | 源起点s | 目视覆盖与边界 |
|---|---:|---|
| handshake | 74 | 两人、握手、转头、侧脸；不是标定的快速转头 |
| profiles | 120 | 开头侧脸，之后切到无脸幻灯片；负样本保留 |
| room_transition | 178 | 走动、镜头切换、部分遮挡 |
| close_speaker | 246 | 近景表情、转头、多次人物硬切、手部遮挡 |
| close_listener | 306 | 近景头部运动、人物反切；另标30帧鼻部轨迹 |
| greeting_cross | 439 | 多人同框、转头、挥手；非受控两人完全交叉实验 |
| greeting_cluster | 449 | 多人转身、拥抱、短时遮挡/重叠；极侧脸/背头不可渲染 |
| outdoor_flash | 176 | 远景小脸、自然亮度变化；不能称强光闪烁/低照全覆盖 |

采访下载版本为480p，测试时空间缩放到1280×720；另一录像原生1280×720、约59.94Hz，每2帧取1帧为约29.97Hz。FFV1只用于无损测试容器，不插值、不合成头动。输出尺寸720p不意味着所有源细节原生720p。固定片段哈希及解码像素哈希在`evidence/round2/natural-manifest.json`；重新封装可能改变Matroska文件UID，应同时核对像素哈希。

`natural-annotations-reviewed.json`含32个源帧快照、67个脸实例，能确定可见点才标五点；部分片段只有框。另有1秒30帧鼻轨迹。标签由**助手目视源帧/放大网格整理，未经独立人类或专家复核**，不能称已获得可靠人工真值。框只是关联门限，远景背景脸没有完整穷举，不能计算完整FP/FN或MOT指标。初稿发现坐标偏移后保留为`natural-annotations-v1.json`历史草稿，评分只用reviewed文件；鼻轨迹在初次预测检查后重审，因此不声称盲标。近景点估计不确定性约5px，小脸约6px，鼻轨迹6px；这些是标注者估计范围，不是校准置信区间。

评分按框中心、脸宽尺度门限逐帧一对一关联，独立报告原始观测漏失和已确认效果轨迹漏失；不插值真值。这里原始观测是检测＋关键点输出，不是YuNet单独召回率。首次两帧确认会增加效果漏失。ID统计仅是离散标注帧观察到的错误复用/变化，不能冒充逐帧MOT IDSW。三维yaw/pitch真值缺失；PnP重投影误差不替代姿态误差。`review_annotations.py`核对了全部32张源图的SHA并生成不读预测结果的叠图，供独立人工审阅；哈希正确不代表标签正确。

## 同条件方案比较

冻结原版源码位于外部证据`round2/baseline/face_effects_live`。所有正式比较使用同一新隔离Python3.11环境：NumPy1.26.4、OpenCV4.11、ORT1.23.2、CPU、OpenCV线程2、ORT intra2/inter1、检测长边640、最多8脸、相同三特效/CPU渲染。每方案独立顺序进程，每段240帧，含推理首帧，不含Engine构造、解码、JPEG或浏览器。非独占物理机器，没有把旧Python3.10/OpenCV5预试验混入公平对照。

| 方案 | 八段平均处理ms | 六脸片段平均 / P95 ms | 已标脸原始漏失 /67 | 效果轨迹漏失 /67 | 条件五点MAE px |
|---|---:|---:|---:|---:|---:|
| 冻结原OSF0 | 14.30 | 26.53 / 29.69 | 15 | 32 | 13.62 |
| 修正OSF0，默认 | 14.47 | 26.83 / 30.25 | 15 | 32 | 12.20 |
| OSF3 | 21.93 | 43.41 / 49.45 | 14 | 31 | 12.33 |
| MP全画面VIDEO | 5.18 | 2.46 / 6.50 | 56 | 59 | 9.48 |
| YuNet＋MP大脸ROI/OSF小脸 | 15.89 | 27.09 / 30.68 | 15 | 32 | 12.37 |
| YuNet五点 | 7.24 | 9.33 / 10.31 | 14 | 31 | 12.90 |

条件误差对应点数不同，尤其MP只保留容易的大脸；**不能据此认定MP最准确或最快完成等质任务**。15/67等数字仅是本稀疏标签集的观测漏失率，含极侧脸/背头、小脸和不确定框，并非总体人脸检测召回率。全部分场景P50/P95/P99、CPU、RSS和源码SHA见`evidence/round2/comparison.json`，质量分布见`quality.json`。不报告硬件GPU利用率，因为未使用硬件GPU。

OSF0默认改动平均约增加0.17ms（1.2%），其中六脸关键点仍约16.92ms，检测5.01ms、渲染2.55ms、外观0.20ms、关联0.69ms。关键点逐脸推理是主要瓶颈，未据此盲目转GPU或batch。OSF3同片段关键点约33.37ms，尚无足够质量证据支持默认升级。OSF3远景P99约158ms长尾保留在报告中。

## 正确性与视觉变化

- 修复66点嘴角索引48/54→58/62，影响嘴部锚点和PnP；与OSF上游mouth_corner定义核对。没有用68点拓扑猜索引。
- 硬切后重置轨迹/MP VIDEO内部状态，避免镜头切换把旧ID交给另一人。离散样本观察到的错误复用从1次降为0次；这不证明所有交叉遮挡无串脸。代价是切镜后重新确认，以及同一人跨镜头得到新ID。
- 短时HSV颜色/上身外观辅助几何和速度关联，歧义隐藏，特征不持久化；相似服装、交叉全遮挡仍是限制。
- 相似变换One Euro按中心、角度、尺度平滑，恢复/长间隔重置；不通过长时间外推伪装检出。1秒鼻轨迹最佳对齐从原版2帧约67ms到1帧约33ms；标签扰动敏感性与完整曲线在`motion.json`。它只测局部轨迹相位，30Hz分辨率粗，且标签非独立真值，**不能称已验证真实快速转头延迟**。一阶残差RMS约4.321→4.324px，几乎相同，落在标签噪声尺度内，不宣称抖动显著改善。
- 眼镜纹理眼中心精确映射双眼，加入受限鼻部仿射、极侧脸退场；近景speaker眼镜眼中心条件MAE约11.65→8.26px（6个眼点），listener约5.09→4.21px（6个眼点）；小脸crowd约7.80→8.13px，未改善。样本少且误差差值接近标注不确定性，不作统计显著性结论。原版和修正OSF0眼镜覆盖均为21/30个双眼可见标签对，MP为7/30，不能靠隐藏困难帧声称变准。全部可匹配稳定锚点眼线roll平均绝对误差约3.54→3.57度，未显示改善；姿态接口实际报告的原始2D roll误差为3.32→3.32度（22个标签对），仍无改善。这些均不是3D姿态精度。
- 皇冠/轨道使用有界脸宽，避免侧脸眼距收缩导致装饰塌缩。仍为2D特效，无真3D眼镜、面部遮挡网格或深度测试。2D眼线roll已评分，3D角误差仍待真值。

## 真实本地平台链路

`platform_acceptance.py`实际调用未修改的Driver `common.vendor_runtime.DriverBundle/make_handler`，插件发布公开自然视频；另进程运行实际PerceptionBundle与原HTTP Handler。使用真实sensor_msgs、rclpy executor、FastDDS，Core使用未替换的完整lifespan和真实SQLite/HTTP/MCP/inspection bridge。原生Canvas查看按钮打开CameraRenderer，真正解码从WebSocket收到的JPEG。仅视频源是回放，未模拟模型、Core路由或DDS。

Core/数据目录新建，绑定127.0.0.1，随机可用DDS domain与本仓库loopback profile。LLM执行、peer发现和外部渠道由配置关闭；Docker连接指向本轮不存在的socket，防止Core启动清理触及共享容器。只在新Canvas放录像sensor和本Card，无运动/执行器工具。CDN仅复用本机已有三维前端JS文件，浏览器外部请求被阻断。

已通过：真实MCP发现、原生项目按接线启动、input_topic解析、ROS结果发布、Canvas解码、特效热切换、项目停止和重启、Core正常关闭。额外修复宿主`/mcp/sse`兼容路径，保留`/sse`，实际观察到Core订阅Perception SSE；没有发送假completion事件冒充Card长任务。`platform.json`记录原始浏览器帧标签及断言。

最终默认CPU平台环境（pip check通过），录像解码返回→ROS编码发布→Card→Core DDS bridge→WebSocket→浏览器图像load：P50 **76.41ms**、P95 **91.47ms**、P99 **93.60ms**；到下一rAF为78.89/95.65/95.98ms，共307帧，实际浏览器load约27.59fps。使用Driver单调时钟标签和导航后浏览器performance时钟的往返校准，起止偏差漂移0.37ms，校准不确定性估计约2.14ms；它不是传感器曝光或屏幕扫描延迟。不能用处理容量或截图FPS标签宣称整个链路稳定30fps。原始Unix时钟指标同时保存，仅作对照。

额外60Hz回放＋真实浏览器主线程暂停2s：单调校准漂移2.13ms，不确定性约5.56ms；整体load P50/P95/P99为72.34/97.46/1466.93ms。恢复后最大加载帧龄**2068.69ms**，约**103.6ms**后首次恢复到250ms以内新鲜帧。功能/控制断言通过，250ms显示新鲜度门槛**失败**，见`platform-load.json`的独立derived_performance_gates。Card处理队列有界不等于Core/TCP/浏览器队列均保证新鲜度；此问题保留为正式质量验收阻塞，没有改动共享Core的队列策略。该负载探针使用前一隔离实验环境，未混入算法速度对照；最终默认CPU环境正常负载全链路另行通过。

失败历史保留：第一次Core在同秒注册两服务产生相同ID，真实工具路由失败；不修改共享Core，以串行间隔注册规避。第二次浏览器版本路径不匹配，改用已有匹配浏览器后通过。单调时钟探针最初在about:blank校准、导航后计时起点重置，稳定性门槛判失败；改为实际页面导航后校准再复测，未沿用失败探针的延迟结论。又发现Core访问`/mcp/sse`而宿主只接`/sse`，在当前独立分支的Perception入口加兼容路径后验证真实订阅。真实DDS适配器还发现实例输出哈希首字符为数字时话题非法，增加i_前缀并复验两类ROS消息及原生Canvas。正式平台建议修正Core秒级ID碰撞，但不属于本轮Card算法改动。真实机器人SDK/摄像头、跨机时钟、多设备DDS、实际网络RTSP/USB仍未验收。完整原生ACP长任务交互/认证部署不能用该回环订阅测试代替。

## 有界缓存、背压和长期运行

实际Service，同一`greeting_cross.mkv`重复播放，默认三类特效、最多8脸、CPU渲染，原版和修正版各新进程顺序运行。提速只改变负载，不作为自然转头质量实验。输入端从解码返回建立单调接收时间；不含DDS/Core/浏览器。

| 输入负载 | 原版输出/输入 | 修正版输出/输入 | 原版 / 修正版覆盖旧帧 | 原版 / 修正版P95接收至JPEG ms |
|---|---:|---:|---:|---:|
| 30Hz ×15s | 450/450 | 450/450 | 0 / 0 | 38.11 / 43.80 |
| 60Hz ×15s | 507/900 | 493/900 | 393 / 407 | 46.30 / 46.96 |
| 120Hz ×15s | 473/1800 | 475/1800 | 1327 / 1325 | 41.71 / 41.23 |
| 30Hz ×300s | 8995/9000 | 9000/9000 | 5 / 0 | 32.39 / 40.44 |

5分钟行的延迟分位数只含最后1200个输出样本，不能冒充全程精确分位数。两版全部8个负载进程均无过期输出、无错误，结束pending为0，工作线程退出；Service历史上限1200，输入pending最多1，旧帧覆盖而非无限排队。原版长测P50/P95/P99为27.54/32.39/45.17ms，修正版29.26/40.44/49.29ms。修正版尾延迟有所增加，未宣称性能全面改善；原版偶发丢5帧与修正版0帧不能证明后者更快。

原版/修正版CPU约284.46%/286.99%（单核100%）；峰值RSS221.39/222.63MiB；30秒后RSS219.09/220.79MiB，结束219.79/217.92MiB；稳态线性拟合斜率+0.67/-0.40MiB/min。数据含分配器/解码器缓存，不足以证明数小时无泄漏。此轮是前后各5分钟，未把10分钟相加称为单实例连续10分钟，也未把服务组件长测称为整套Core长测。

完整数据及逐秒RSS在`load.json`，每次命令/退出状态在`load-commands.json`。Core WS/浏览器消费者的缓冲与源设备内部缓存不由Service单槽策略保证，平台负载与暂停探针单独报告；未宣称端到端零缓存或零拷贝。

## 复现

模型、录像、冻结源码和大体积逐帧JSONL在`/home/utopus/phanthy-work/face-effects-evidence/round2`及其父目录。源码库只保留紧凑JSON、标签、工具和报告。不要覆盖原证据；新一次运行选择新输出目录。

```bash
# 提取前先按sources.json获取并核对合法来源视频。
python perception/tools/face_effects_live/natural_dataset.py --evidence /your/round2
python perception/tools/face_effects_live/natural_benchmark.py --evidence /your/round2 \
  --package /your/frozen-or-current/face_effects_live --name run-name --model osf0 --output /your/new-results
python perception/tools/face_effects_live/score_natural.py --predictions /your/new-results \
  --annotations perception/docs/face_effects_live/evidence/round2/natural-annotations-reviewed.json --output /your/new-quality.json
python perception/tools/face_effects_live/score_motion.py --predictions /your/new-results \
  --annotations perception/docs/face_effects_live/evidence/round2/natural-annotations-reviewed.json --output /your/new-motion.json
python perception/tools/face_effects_live/load_acceptance.py --package /your/face_effects_live \
  --video /your/round2/natural/greeting_cross.mkv --models /your/models --fps 120 --seconds 15 --output /your/load.json
python perception/tools/face_effects_live/platform_acceptance.py --output /your/NEW-runtime \
  --models /your/models --video /your/round2/natural/greeting_cross.mkv --driver-root /your/phanthymotus-driver \
  --assets /your/browser-assets
```

平台脚本需要真实ROS2/完整Core依赖、Playwright与兼容浏览器及系统库；如不在标准位置，用`PLAYWRIGHT_BROWSERS_PATH`或`--browser`指定已有可执行文件。它不会自动安装、连接机器人或改动Driver。长时负载和最终回归结论在[ACCEPTANCE](ACCEPTANCE.md)汇总。
