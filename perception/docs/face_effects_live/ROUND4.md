# 独立公开标注数据验收

本轮结论为 **NO-GO：完整 V1.0 离线质量验收未通过**。取得了独立人工标注的静态定位证据，补齐格式转换、可重复评分及运行时隔离；没有取得许可和下载条件均满足的独立视频关键点/多人脸时序身份测试集。未把静态重复观测、助手标签或合成单元测试当作这些缺失的真值。集成/真机验收继续按用户要求暂缓。

## 数据来源和可用边界

| 数据 | 本机结果 | 条款与适用性 | 可测范围 |
|---|---|---|---|
| 300-VW | 官方下载入口要求姓名、邮箱、单位；没有代填或绕过表单，未取得原始包 | 下载表单前未确认到完整数据使用许可；不能推断商业许可 | 已实现 PTS68/视频帧适配；真实视频验收待取得数据与许可 |
| WFLW | 官方项目页及作者仓库指向的标注包均实测 HTTP 404 | 作者代码许可不能代替图像/标注许可；当前未确认独立数据授权 | 已实现 98 点/框/六属性适配；未声称已运行 WFLW 真实评测 |
| ChokePoint | 核查官方站点及作者 Zenodo 记录；未下载受限图像 | NICTA 明确限非商业研究，当前平台研发用途的授权未确认 | 普通单人序列不能证明多人交叉；拥挤序列还需要可信完整脸框身份标注 |
| PLUSFiaQ / ChokePoint 拥挤序列新标注 | 作者页面要求签署许可并发送后获得链接；未代签或发信 | 需操作者取得授权；网页说明 visibility 尚未完整校正 | 已有带显式脸框来源声明的 MOT 格式入口，但原生 CSV 未取得，未宣称直接支持其未知字段 |
| COCO 2017 validation keypoints | 官方标注包与 327 张合格图像实际下载、校验和推理 | COCO Consortium 标注 CC BY 4.0；仅选图像元数据中 CC BY 2.0，保留原 Flickr 图像及照片页链接 | 独立眼/鼻语义定位参考和眼镜眼部锚点；没有嘴角、密集脸网格、完整脸框或视频身份真值 |
| COCO-WholeBody 密集脸标注 | 核查作者 README，未下载 | 明确 ONLY research/non-commercial；链接实际指向 CC BY-NC 4.0，不能被链接文字的 Attribution 4.0 误导，也不能套用原 COCO 标注许可 | 有人工校正的68点潜力，当前用途授权未确认，未作为已取得密集真值 |

原始包、图像、完整标签和预测均位于仓库外：`/home/utopus/phanthy-work/face-effects-datasets/round4` 与 `/home/utopus/phanthy-work/face-effects-evidence/round4`。源码仅保留工具和小型统计/哈希/来源记录，不包含原始人脸素材。没有上传图像、填写个人身份或接受机构协议。

官方依据：[300-VW 资源](https://ibug.doc.ic.ac.uk/resources/300-VW/)、[下载表单](https://ibug.doc.ic.ac.uk/download/300VW_Dataset_2015_12_14.zip/)、[WFLW 作者仓库](https://github.com/wywu/LAB)、[作者下载脚本](https://github.com/wywu/LAB/blob/master/scripts/download/download_wflw_annotation.sh)、[ChokePoint 条款](https://arma.sourceforge.net/chokepoint/)、[作者数据记录](https://zenodo.org/records/815657)、[PLUSFiaQ](https://www.wavelab.at/sources/plusfiaq/)、[COCO 条款](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm)、[COCO 格式](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/format-data.htm)。未把第三方数据镜像的 license 标签当作作者授权。

## 数据完整性与转换

COCO 官方 `annotations_trainval2017.zip` 大小 252,907,541 B、MD5/官方 S3 ETag `f4bbac642086de4f52a3fdda2de5fa2c`；提取的 `person_keypoints_val2017.json` SHA256 为 `788e2dae83c86bd547be7fab269d6399df5671063d29a61360cdb2cc370d2b14`。没有关闭 TLS 校验：原图域名证书主机名不匹配，改用同一官方存储桶的有效 HTTPS 路径 `s3.amazonaws.com/images.cocodataset.org`。逐张原 JPEG 哈希、来源 ID、尺寸和标签来源均保留。

在推理前按原许可证及可用人工头部点选择：5000 张 validation 图像中 857 张为 CC BY 2.0，327 张满足非退化头部参考区域及至少可评估眼对/鼻点条件，全部使用，未按模型结果删样本。共 685 个可评估脸实例、516 个可见眼对；171 个头部点部分不可见实例与前者有 2 个重叠，不能相加当作互斥场景。304/134/59/19 个眼对分别位于眼距 <12/12–24/24–48/≥48 px，说明该子集小脸偏多。7 张含超过 8 个可评估脸，仍保留；Card 配置上限为 8。没有独立姿态角、模糊或光照属性标签，不自动把点不可见等同于侧脸/遮挡原因。

`dataset_adapters.py` 输出扩展的 `clips/frame_id/faces/points` 格式，与既有五点评分格式兼容，额外保留 dense_points/topology/visibility/原路径/属性/时间来源：

- 300-VW 要求显式零/一基坐标选择，保留原 PTS 帧号、decode_index 和源 FPS 时间；缺帧不紧凑重编号，不插值。68 点按眼轮廓均值、鼻尖 30、嘴角 48/54 转换。OSF66 嘴角为 58/62，绝不套用其编号去改真值。
- WFLW 保留全部 98 点与六个原始属性；眼轮廓 60–67/68–75、鼻尖 54、嘴角 76/82，不误用 96/97 的瞳孔点。occlusion 是脸属性，没有凭空生成逐点可见性。非有限坐标显式 null，不夹紧到图内或根据预测修补。
- COCO 原点是零基，v=0/1/2 原样保留。主评测只用可见点，嘴角为空；孤立可见眼不猜测 Card 的屏幕左右槽，仍保留在原始 dense_points 中。其眼点与轮廓中心只有语义对应，结果称为“眼鼻/五锚点代理误差”，不称标准 68/98 点 NME。
- COCO 人体框绝不当作人脸框。固定头部参考区域只用于一对一匹配（IoU≥0.2），不能计算标准脸框 AP/完整检测 FP。未匹配预测保留数量，但 FP 明确为 null；不能把未标注的背景脸当误检。
- MOT 入口拒绝普通行人框，要求外部来源说明明确是 face、时序身份与穷尽脸框、坐标原点；保留空帧/ignore 区域，拒绝缺失源帧和重复身份。只有满足这些条件才计算 IoU≥0.5、保留上一有效匹配的 CLEAR 风格 IDSW/MOTA，未实现 HOTA/IDF1。

转换器不会认证任意输入就是人工真值。公开数据保留 `annotation_origin=external_dataset_not_local_workbench_review`，不冒用本地工作台的独立复核声明。首次导出该元字段的错误已留存并仅修正元信息；所有坐标、可见性、选择和原始标注均逐项确认未变。格式/评分的合成 oracle 测试不计入真实数据帧数。

## 真实定位结果与可选优化

完整同输入比较、分组、覆盖率、P50/P95/P99、CPU/RSS 和成对误差见 [comparison.json](evidence/round4/comparison.json)。每个模型均实际推理所有 327 张图像；静态图片重复两次只为满足 Card 的两次确认规则，不产生视频跟踪证据。每张图像重新初始化流水线及 MP VIDEO 状态。初始化/解码另计，处理耗时不含预测序列化；OSF 两线程、OpenCV 两线程，CPU EP，不含 CUDA 或物理相机。

| 同327张图像，含首次确认的654次观测 | 匹配脸/685 | 条件眼鼻 NME | 眼镜眼对/516 | 处理均值/P95/P99 ms | 父子RSS峰值 MiB |
|---|---:|---:|---:|---:|---:|
| 原始OSF0，进程内 | 417 | 0.1672 | 240 | 12.05/26.66/31.11 | 155.5 |
| 本轮起点OSF0，进程内 | 417 | 0.1672 | 239 | 11.57/25.71/30.17 | 158.9 |
| 本轮最终OSF0，隔离＋紧凑回复 | 417 | 0.1672 | 239 | 14.76/33.94/41.00 | 271.0 |
| OSF3，隔离＋紧凑回复 | 424 | 0.1665 | 245 | 19.88/55.39/64.60 | 364.3 |
| YuNet＋MP/OSF混合 | 417 | 0.1680 | 239 | 13.92/33.81/38.40 | 976.9 |
| MP全画面 | 57 | 0.1250 | 52 | 3.29/8.32/11.77 | 699.6 |
| YuNet五点 | 437 | 0.2886 | 239 | 7.51/10.38/12.04 | 123.0 |
| OSF0＋检测960上采样 | 477 | 0.1858 | 246 | 26.75/46.28/52.48 | 318.6 |

条件误差来自不同匹配集合，不能据此把MP认定为最准确；共同样本对照另存JSON。静态观测包含首次未显示特效的帧，不应换算成持续视频输出FPS。CPU含初始化/证据记录，内存合计含共享页，不与下文720p常驻Service直接混算。

全部算法/回放比较在同一专用MP完整依赖venv中使用CPU：Python3.11、NumPy1.26.4、OpenCV-contrib4.11、ORT1.23.2、MP0.10.21；主机i9-13900HX、32逻辑CPU、Linux x86_64 WSL2。默认平台venv用于回归；未把不同环境的FPS混作A/B。[环境与宿主哈希](evidence/round4/environment-check.json)。

固定匹配后，当前 OSF0 的原始眼鼻代理 MAE 约 2.672 px、条件眼距归一化误差 0.1672，但只匹配 417/685 个参考脸，不能只报告条件误差。可见眼对 354/516 获得关键点匹配，眼镜实际渲染 239/516。原始版本与当前版本相同的 239 个眼镜实例，独立参考眼点误差 2.682→2.244 px；原始眼鼻模型输出相同。这验证了既有配饰变换改善，没有验证连续跟随或三维配饰。

同239例中197例改善、42例变差；眼镜每脸误差P95由6.470降到5.367px，P99由9.014升到9.487px。眼镜覆盖从原始240例降为239例，未隐藏这一代价。新增上采样的共同338个有眼距实例，原始代理NME均值0.1638→0.1599；整个匹配集合NME反而升高，两种口径均保留。

OSF3 的额外覆盖和条件精度收益有限、耗时更高；全画面 MediaPipe 匹配 57/685，低耗时主要源于少检。ROI 混合未表现出明确总体优势；继续默认 YuNet＋OSF0，保留可替换后端及显式 YuNet 五点降级。

新增 `detect_upscale=true, detect_side=960` 可选质量配置：名义上采样最多 2×（另有 YuNet 的32像素尺寸对齐），输出仍映射到原帧坐标，不改变相机输入或默认配置。参考脸匹配 417→477/685，但加入更困难的小脸后条件 NME 0.1672→0.1858、2D roll 误差也增加；眼镜覆盖只 239→246/516。CPU 成本明显增加，未设为默认，也没有通过调阈值或删难例提升分数。这是同一数据集上的探索性配置选择，没有声称新盲测集上的泛化提升。

误差最大的案例及缺失样本按原图/标注 ID 保留在外部各运行的 `quality-final.json`，其排序只用于诊断，不改变评测集合。眼鼻代理定义及 COCO 人工点误差本身限制了精度解释；未核实预训练模型是否见过相同来源图片，不能声称训练数据完全隔离。

例如 image 78170 / annotation 460165 的眼距17.03px、原始代理MAE22.80px、NME1.339；image408696 / annotation1245345 眼距6.71px、MAE8.01px、NME1.194。它们没有被删除或重标。[failure-cases.json](evidence/round4/failure-cases.json)还保留全部268个未匹配参考脸；这里未做独立目视裁决，不把所有异常武断归因于模型，也不以“小脸”解释掉所有错误。

## 运行时正确性和性能

扩展到仓库 `test_ort_worker.py` 的回归首先失败：本 Card 直接创建 ORT session，违反宿主已建立的 sherpa-onnx/ORT 进程隔离约定。保留失败日志，没有修改已有守卫测试或共享 worker。现改用原有 `plugins.ort_worker` 的 spawn 宿主、每实例独立 session key，只卸载自身 session，停止 Card 不终止其他 Card 使用的宿主；没有机器人 SDK 依赖。

初版隔离后，720p 六脸片段平均处理 28.7→40.3 ms，关键点阶段约 29.2 ms。依据此 profiling，把完全相同的热图/offset 解码移到子进程，198×28×28 float32 的 620,928 B 输出变为 66×2 坐标及 66 置信度，数组负载约 1,320 B。输入张量仍经 Queue 复制，不宣称零拷贝。真实数据逐项比对用于验证数值保持；进程通信开销按最终实测报告，不用数组压缩比代替延迟收益。

紧凑回复后六脸均值/P95/P99为33.80/38.86/40.18ms，关键点23.31ms；相对完整热图回复40.28/45.72/50.62ms改善，但仍慢于进程内28.72/31.89/33.44ms。八段平均15.82→20.41（完整回复）→17.84ms；上采样档为26.41ms。327张静态图的原始框、66点、语义锚点、眼镜及姿态几何在隔离前后逐项数值完全一致，见[pipeline-invariance.json](evidence/round4/pipeline-invariance.json)。这验证传输/解码改动保持输出，不等于精度通过。

现有共享宿主在崩溃重载时不保留可选 postprocessor。本 Card 能识别该首个原始热图响应，用同一函数解码并重新挂接紧凑返回；未为此修改共享基础设施。真实子进程测试覆盖多个 session、关闭一个不影响另一个、杀死本测试自有 worker 后恢复且关键点一致；不能据此宣布实际 Sherpa 并发、多 Card 调度或 Jetson 通过。

720p 的 8 段真实录像（每次 1920 帧）与同源重复负载分别保存于 [natural.json](evidence/round4/natural.json)、[load.json](evidence/round4/load.json)。隔离前后各 30Hz/300秒，以及 60/120Hz 过载；仅有界 Service 回放，不启动 Core/Driver/DDS。延迟从解码返回后的接收到 JPEG 编码，历史保留最后 1200 项，不包含曝光/网络/Canvas。父子进程 CPU、父 RSS 与父子 RSS 合计均记录；RSS 合计包含共享页重复计数，不等于 PSS 或精确物理内存。

| 同一720p六脸录像，30Hz各300秒 | 隔离前 | 最终隔离＋紧凑回复 |
|---|---:|---:|
| 输入/输出 | 9000/8964 | 9000/8576 |
| 按源时长输出FPS | 29.88 | 28.59 |
| 旧帧覆盖/过期结果 | 36/0 | 424/0 |
| 最后1200项接收至JPEG P50/P95/P99 ms | 30.17/52.79/60.62 | 48.62/66.78/70.08 |
| 父子CPU，单核100% | 286.68% | 293.59% |
| 父RSS峰值/父子RSS合计峰值 MiB | 235.50/235.50 | 225.18/338.27 |
| 父RSS30秒后线性斜率 MiB/min | +0.84 | +0.26 |
| pending结束/历史上限/线程退出 | 0/1200/是 | 0/1200/是 |

60Hz/15秒：前后489/900→423/900帧、覆盖411→477；120Hz/15秒：482/1800→423/1800帧、覆盖1318→1377，均没有无界积压或资源退出失败。上采样档30Hz/60秒1338/1800帧、约22.3FPS、P95 77.25ms、父子RSS峰值359.32MiB，不能称全场景实时30FPS。短时RSS斜率不是无泄漏证明，未用15秒样本声称长期内存改善。

## 回归与三个跳过项

- 最终默认CPU环境：**309 passed / 3 skipped**，覆盖全部Card与相关Bundle/measure/image/vision/provider/ORT worker回归。新数据格式/评分与来源守卫oracle17项、既有评分5项均包含其中。
- 独立MP＋ModernGL环境：**全Card 106 passed / 0 skipped**；这是重叠用例的另一环境，不能与309相加当作415个独立用例。前述3项是默认环境未安装MP的hybrid/mediapipe两例、未安装ModernGL的一例；现已全部在可选环境实际执行。EGL实际renderer为Mesa llvmpipe，属于软件GL，不是硬件GPU证据。
- 最初扩展回归有ORT隔离守卫失败（305 passed/1 failed/3 skipped），通过修正实现解决；未改守卫或放宽测试。可选环境初次全Card104 passed/1 failed，原因是宿主入口测试缺PyYAML；仅在专用venv补齐6.0.3后105通过；再新增冻结来源守卫后最终106通过。所有失败日志保留。
- Core相关软件回归16 passed、前端协议7 passed；它们不是DDS/完整Canvas链路。标注工作台真实Chromium QA通过：61张源图、62项、点击/不可见点/同帧同步/部分导出/原文件未变/无JS错误；自动编辑导出仍明确不是GT。
- 两个独立环境pip check通过；未覆盖共享Core/Driver依赖。模型、原始数据、预测与冻结源码位于外部证据目录；[完整测试日志](evidence/round4/pytest-cpu-final.txt)和[可选环境](evidence/round4/pytest-optional-final.txt)可核对。

## 缺失的真实视频指标

原有 8 段 White House/Obama Presidential Library 公有领域录像实际重跑，但其助手草稿标签仍不是独立真值。因此本轮 **独立标注视频有效帧数为 0，真实多人交叉 IDSW、视频锚点抖动、快速运动相位/跟随延迟、三维姿态角误差均不可估计**。没有把两次相同静态观测的零位移当作稳定性通过，也未用单人 landmark 序列证明多人交叉。

新时序评分对连续标注的鼻参考与输出计算残差一阶 RMS、同一有效窗口的 lag 曲线，缺帧不跨越、运动不足/样本不足时返回 null；它只是相位代理，不是物理曝光到显示的延迟。测试中的已知两帧延迟和两个 ID switch 来自合成 oracle，明确不进入自然视频结果。

## 重复执行

从仓库根目录，以独立环境运行；将路径替换为自己获准取得的数据：

```bash
python perception/tools/face_effects_live/dataset_adapters.py --kind 300vw \
  --input /your/300VW --sequence 001 --fps 30 --coordinate-origin one \
  --license-note 'your actual granted terms' --output /your/NEW-300vw-labels.json
python perception/tools/face_effects_live/dataset_adapters.py --kind wflw \
  --input /your/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt \
  --license-note 'your actual granted terms' --output /your/NEW-wflw-labels.json
python perception/tools/face_effects_live/fetch_coco_subset.py \
  --annotations /your/person_keypoints_val2017.json --output /your/NEW-coco-subset --limit 400
python perception/tools/face_effects_live/dataset_benchmark.py \
  --labels /your/NEW-coco-subset/labels.json --images /your/NEW-coco-subset/images \
  --package perception/plugins/face_effects_live --models /your/models --model osf0 \
  --output /your/NEW-predictions
python perception/tools/face_effects_live/score_dataset.py \
  --labels /your/NEW-coco-subset/labels.json --predictions /your/NEW-predictions \
  --output /your/NEW-quality.json
```

300-VW 的 FPS 必须来自对应原始视频，coordinate-origin 必须按实际获准包确认；示例数值不是对未下载文件的自动判断。COCO 的官方标注下载链接见其作者下载页面；HTTPS 官方桶替代地址、验证值和本轮全部模型/自然视频/负载命令另存于证据目录。输出工具通常拒绝覆盖已有结果，应使用新目录。冻结版本若包含与当前worker不同的OSF解码源码，比较工具会明确拒绝，必须使用匹配工作区，避免混用实现。

## 未完成与人工操作

需要操作者取得 300-VW/WFLW 及可信多人脸时序数据的实际授权、完整下载与标注约定；或提供已获准、独立逐帧标注的自然视频。届时应先固定完整测试集和质量门槛，再做盲测/回归，测量侧脸、快速转头、真实交叉、遮挡、模糊、光照等场景。当前没有这些场景的可靠数字，不以 Mock 或静态代理补齐。

正式 PR 仍需上述质量证据、明确质量门槛、共享 ORT 宿主并发/干净镜像验证，以及审阅先前新鲜帧原型和既有 MCP ID 碰撞问题。平台/真实机器人验收已暂缓；ARM64、Jetson、原生 GPU、Windows、物理摄像头/RTSP 与长时真实输入均未新增验证。本轮只验证 Linux x86_64 WSL2 CPU 与软件 EGL，不修改 privacy_view、不推送/创建 PR/发布或操作机器人。
