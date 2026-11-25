# 后续研发：新鲜帧通道与源帧人工复核工具

集成验收已按用户最新要求**暂缓**，后续真机验收由用户进行。不再追加Core/Driver联调运行。已有成功、失败和未完成状态保留，不能把暂停解释为验收通过。

## 旧帧显示的可回退原型

此前2秒浏览器停顿后曾加载约2069ms旧帧。实现了仅由FEL1标签JPEG启用的新路径：

- 首次识别标签后，以`fresh=1`重订阅；Core跳过历史快照，该连接队列容量为1，满时替换旧帧。
- 消费者事件循环停顿超过500ms后清空显示、重连并忽略旧连接代次的消息；发布停止时也清除旧图。
- 新订阅需服务器明确确认`fresh:true`；无确认时不把可能过期快照当作新画面。
- 详情面板和原生监控视图接入同一独立模块。未标记JPEG、其他消息和默认队列沿用原路径；即使带fresh查询，JSON也保留原缓存/队列语义。未修改privacy_view算法或其帧内容。

涉及本独立工作区的Core `api/inspection.py`、`detail-panel.js`、`monitor-dashboard.js`、`renderers/camera.js`和新增`fresh-bus-socket.js`；未修改原Core工作区。属于待进一步集成审阅的可分离改动，不是已发布能力，不宣称零拷贝或全链路固定延迟上界。

首轮真实负载：60Hz请求、相同自然录像、2秒浏览器停顿，新连接队列实测为1；恢复后最大帧龄94.90ms，44.8ms后首次恢复到250ms内；load P50/P95/P99约70.58/85.00/96.46ms，时钟校准不确定性约5.82ms。[完整首轮数据](evidence/round3/fresh-first.json)。这是一个成功样本，**未完成严格同环境修改前后验收，不能当作正式性能改进结论**。

冻结Core源码后复跑，出现录像源已发布、DDS发现2个订阅者，但Card实际处理0帧、Core缓存也无帧的情况。更换低域、LOCALHOST_ONLY设置及已安装RMW后仍有失败，根因未确认；不能把失败算成算法耗时或以此证明某配置修复了问题。测试域选择已修正为避开本机Linux临时端口区，并核查整段端口；[ROS官方文档源码](https://github.com/ros2/ros2_documentation/blob/jazzy/source/Concepts/Intermediate/About-Domain-ID.rst)说明了域与临时端口的约束。记录见[integration-attempts.json](evidence/round3/integration-attempts.json)。全部自有测试均已关闭，未停止共享服务。

协议状态机[Node测试7项通过](evidence/round3/node-protocol-tests.txt)；Core队列、快照、JSON保持原语义和既有相关测试共[16项通过](evidence/round3/core-unit-tests.txt)。它们是软件/协议测试，不能冒充DDS、真机或完整多视图验收。旧的单核FPS、自然视频模型比较仍引用ROUND2，未混入这些失败样本。

## 本地源帧标注复核

为质量决策补齐可操作的人工复核入口，新增`annotation_workbench.py/html`。它不运行检测、关键点或跟踪模型，不读取预测JSONL，不上传图像，服务器只读源标签；修改由浏览器下载为新JSON，原草稿不覆盖。

- 准备并校验61张唯一源PNG，对应32个五点/框快照和30个鼻轨迹项，共62个编辑项。
- 支持关键点点击、可见性null、框、逐帧人脸ID、鼻轨迹ID、缩放、隐藏已有标签及修改记录。
- 鼻尖只可同步到同一源帧、同一ID的重复标签，不跨时间自动补点或插值。
- 源图载入完成前禁止编辑，避免快速切帧时把旧图点击写入新帧；源尺寸和哈希均检查。
- 部分复核导出始终`independent_human_review=false`。全部项确认加操作者声明才记录完成；该声明不是程序自动认证的真值。导出保留来源哈希和编辑历史。

细致标注建议100%或200%缩放，降低CSS缩放下的像素取整误差。人脸ID需要人工跨帧核对，不能把相似外观或自动关联当作身份真值。原始助手草稿仍未获得独立人工复核。

本机已准备：`/home/utopus/phanthy-work/face-effects-evidence/round3/annotation-sources`。从仓库根目录运行：

```bash
/home/utopus/phanthy-work/face-effects-platform-env/bin/python -B \
  perception/tools/face_effects_live/annotation_workbench.py \
  --labels perception/docs/face_effects_live/evidence/round2/natural-annotations-reviewed.json \
  --frames /home/utopus/phanthy-work/face-effects-evidence/round3/annotation-sources
```

工具打印随机回环地址。其他环境可先准备一个新目录：

```bash
python perception/tools/face_effects_live/annotation_workbench.py \
  --labels /your/draft.json --frames /your/original-annotation-pngs \
  --clips /your/natural-clips --prepare /your/NEW-source-pngs
```

导出后使用新文件评分；正式评估可显式要求已完成人工声明：

```bash
python perception/tools/face_effects_live/score_natural.py \
  --predictions /your/predictions --annotations /your/reviewed-labels.json \
  --require-reviewed --output /your/quality.json
```

`score_motion.py`现在正确处理不可见鼻尖和不连续帧：排除缺失，数据不足时返回不可估计，不用零点/插值编造轨迹或输出NaN。原30帧有效样本重新计算仍为1帧最佳对齐、残差约4.3235px；没有因工具更新宣称精度提升。

## 已完成的离线验证

真实Chromium对标注工具完成坐标点击、不可见点、同帧同步、部分导出、原文件未变和无页面错误验证。[UI结果](evidence/round3/annotation-ui.json)。自动化人为改点只用于界面测试，导出明确为`UI_TEST_ONLY_DO_NOT_SCORE`且未完成人工复核；没有放进质量评分数据集。

初次QA被CSP阻止了自动化字符串求值，随后改用真实DOM状态和下载文件断言，未放宽产品CSP。再次发现75%缩放会产生1px级点击取整差，精确坐标QA改为100%缩放后通过。

默认CPU环境最终相关离线回归：**155 passed，3 skipped**（可选MP两例与GL）；新增五项合成单元测试只检查评分正确性，不算自然视频质量证据，覆盖不可见点、不连续时间、预测不能重复匹配两脸、非法框及帧ID类型。正式评分还会拒绝未完成的UI测试导出标签，[守卫结果](evidence/round3/review-guard.json)。[回归日志](evidence/round3/pytest-offline.txt)。当前仍未达到完整V1.0离线质量验收标准，核心缺口是可信标注和困难自然场景质量；平台/真机验收按用户安排暂缓。
