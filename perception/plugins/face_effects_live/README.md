# face_effects_live

机器人无关的人脸实时特效原型。推荐YuNet + OpenSeeFace model0 + ORT CPU + ROI预乘alpha渲染；不依赖CUDA或机器人SDK。提供透明PNG皇冠、2D眼镜、随源时间变化的轨道星光。默认不访问摄像头，不下载模型，不启动真实机器人，也不修改privacy_view。

研发报告：[调研与候选对比](../../docs/face_effects_live/RESEARCH.md)、[系统架构](../../docs/face_effects_live/ARCHITECTURE.md)、[实测与验收边界](../../docs/face_effects_live/ACCEPTANCE.md)、[第二轮自然视频及真实平台验证](../../docs/face_effects_live/ROUND2.md)。已实际运行自然视频和本地完整Core/Driver公共运行时/ROS2/原生Canvas链路；自然视频标注仍待独立人工复核，尚未达到完整V1.0质量验收标准。

后续按用户要求暂缓集成验收，真机由用户进行。[ROUND3](../../docs/face_effects_live/ROUND3.md)包含新鲜帧原型的成功/失败边界，以及可直接运行的本地人工标注复核工具和命令。标注工具不访问模型或第三方服务，原草稿只读，导出新JSON；未完成人工复核不会自动标成可靠真值。

## 独立环境

最新独立公开标注验收见 [ROUND4](../../docs/face_effects_live/ROUND4.md)：327张许可筛选的COCO静态图像已真实推理，300-VW/WFLW及可信多人时序数据仍受下载/许可限制，完整离线结论为NO-GO。格式转换与评分工具见下方“独立数据集评分”，没有将静态重复观测当作视频跟踪通过。

建议Python3.10+。使用独立venv或新容器；不要覆盖Core/Driver共享环境中的OpenCV、NumPy、ORT或ROS包。

```bash
python3 -m venv /your/isolated/face-effects-env
/your/isolated/face-effects-env/bin/python -m pip install -r perception/plugins/face_effects_live/requirements-cpu.txt
/your/isolated/face-effects-env/bin/python perception/tools/face_effects_live/fetch_models.py --output /your/models --models yunet osf0
```

下载只在显式运行fetch工具时发生；哈希、字节数和许可证见`models.json`、`THIRD_PARTY.md`及`licenses/`。已有模型不一致会拒绝覆盖。模型主分支更新也会因SHA变化拒绝；需要人工核查后更新清单，不能跳过哈希。

运行时需保留本仓库通用 `perception/plugins/ort_worker.py`（仅Python/NumPy/ORT，无机器人SDK），并设置示例中的 `PYTHONPATH=perception`。OSF在已有spawn宿主中加载每实例独立session，避免与Perception的sherpa-onnx运行时冲突；关闭实例只释放自身模型，不关闭共享宿主。不要仅复制Card文件夹而漏掉此通用依赖。

复制`config.example.json`到自己的配置文件，填写实际本地模型路径。不要将占位路径当作设备发现。`landmark_model`删掉可运行轻量A方案；它只有5点，不提供稠密点PnP。`provider=cpu`在没有CUDA时正常工作。`allow_cpu_fallback=false`可严格拒绝加速失效；provider列表不是GPU算子驻留证明。

```bash
PYTHONPATH=perception /your/isolated/face-effects-env/bin/python perception/tools/face_effects_live/doctor.py --config /your/config.json --probe
PYTHONPATH=perception /your/isolated/face-effects-env/bin/python -m plugins.face_effects_live --config /your/config.json --video /your/replay.mp4 --metadata /your/frames.jsonl
```

`--output /your/demo.mp4`是离线验收导出，可检查解码/处理/编码整条链；没有另建或实现`face_effects_video` Card。无节流回放输出的throughput不能当作实时输入FPS。输入每帧已解码后开始测量延迟，不含传感器/传输/解码和浏览器显示。

## 本机 Canvas 预览

```bash
PYTHONPATH=perception /your/isolated/face-effects-env/bin/python -m plugins.face_effects_live --config /your/config.json --serve
```

终端打印随机回环端口；在本机打开该地址，填本机可读视频路径，启动、切换特效或停止。页面使用真实Canvas绘制服务端已处理JPEG。这个入口用于离线验收，与完整Core ROS2 Canvas集成分开标记。页面不上传图像、不加载CDN、不申请浏览器摄像头。摄像头和RTSP请通过MCP/配置显式选择；不自动扫描或开启设备。

## 原生 Perception Card

现有Perception `config.yaml`中增加以下可选配置；默认仓库配置未启用。模型和代码需在同一容器可读。

```yaml
plugins:
  face_effects_live:
    enabled: true
    detector_model: /your/models/yunet.onnx
    landmark_model: /your/models/osf0.onnx
    provider: cpu
    renderer: cpu
    max_faces: 4
    detect_interval: 1
    max_age_ms: 250
    source_kind: ros_compressed
```

将实际上游相机接到Card，或给`start`传`input_topic`。底层使用BEST_EFFORT、VOLATILE、depth1。ROS原始图像用`source_kind=ros_image`，目前支持bgr8/rgb8/bgra8/rgba8/mono8，支持padding stride；拒绝深度和未知编码。原生Canvas接线已验证CompressedImage `image/jpeg`。原始Image输入由适配器直接订阅，`info`如实声明`image/raw`；当前Core没有该格式的预览解码器，不能假设原始Image也能在上游监控窗显示。特效输出始终为标准CompressedImage JPEG与JSON元数据，可由标准Canvas渲染。后续新鲜帧原型涉及本分支Core详情/监控和JPEG显示路径，集成验收暂缓；未改Driver或添加运动权限。

MCP工具`face_effects_live`的`action`：

| 动作 | 参数及行为 |
|---|---|
| info | instance_id可选；返回状态、错误、能力、输出话题、统计；无实例时不打开模型或设备 |
| config | 停止实例后修改source_kind/source_uri/模型/后端/effect；非法配置不写入 |
| start | instance_id、input_topic和配置项；初始化模型成功后采集；同实例重复启动幂等 |
| stop | 指定实例停止；省略instance_id停止所有实例；有线程未退出则拒绝该实例重启 |
| list_effects | 列出manifest中的效果 |
| select_effects | effects字符串数组；`[]`关闭特效，运行中下一帧切换 |
| snapshot | 当前新鲜结果JPEG base64及元数据；过期/停止无图像 |

`source_kind=file`与`source_uri`选择本地视频；`usb`与显式设备路径/序号选择USB；`rtsp`从服务端`rtsp_url`读取地址，避免凭据通过UI和MCP参数传递。没有默认设备序号或ROS_DOMAIN_ID。Core日志已为该Card省略MCP参数及结果内容。

非ROS Driver可通过`PushSource.submit(image, timestamp, metadata=...)`接入Service，调用方负责将专有图像转换为BGR；核心从不导入该Driver SDK。原始源时间错误有修复计数，直接Engine输入乱序会拒绝。分辨率改变会重置跟踪，旧帧会被丢弃。

## 特效扩展与后端

`assets/effects.json`定义id、plugin、PNG、width/offset。已有sticker/glasses/orbit可增添配置和素材；新行为继承Effect并在PLUGINS注册，返回Sprite(texture,matrix,opacity)。素材路径必须在manifest目录内，RGBA PNG不大于2048px。运行时选择控制active集合，停机关闭资源。

CPU使用NumPy/OpenCV ROI合成。可选`moderngl==5.12.0`提供EGL+OpenGL3.3 shader路径；`auto`探测失败或软件GL回退CPU并返回原因，`gl`严格失败。真正硬件GPU仍待测；这不是GLES原生实现，更不是零拷贝。Jetson需要匹配JetPack的OpenCV/ORT/CUDA/EGL构建，不能直接安装x86测试包宣称兼容。

眼镜用双眼精确锚定及受限鼻部仿射，极端侧脸退场；没有头部深度遮挡。皇冠/星光使用有界脸宽，减轻侧脸眼距收缩。OpenSeeFace使用其66点拓扑的58/62嘴角。头姿是通用模板近似PnP，即使提供K也标记`approximate=true`，不能用于度量控制。默认相似变换One Euro平滑中心/角度/尺度；`stabilization=landmarks`可回到逐点滤波进行消融。短时HSV外观辅助几何关联，镜头硬切重置轨迹；外观特征不写日志、不持久化，也不提供身份识别或完全遮挡后的ReID保证。

可选安装`requirements-mediapipe.txt`至另一个**纯venv**，再显式获取`fetch_models.py --models mediapipe`。已验证的Linux x86_64/Python3.11完整版本锁在`requirements-mediapipe-linux-x86_64.lock.txt`。MP0.10.21要求protobuf<5，与当前宿主部分包要求protobuf>=6冲突，不能叠加安装到默认Core/ROS环境；两套新隔离环境分别通过pip check，MP纯环境完成真实源帧推理。该可选后端的ROS发布镜像/进程隔离尚待验收，主方案不要求MP。

全画面选择`detector_backend=mediapipe`且`provider=cpu`；保留YuNet检测并在大脸ROI使用网格选择`landmarker_backend=hybrid`。两者需`face_landmarker_model=/your/models/face_landmarker.task`，支持`face_landmarker_sha256`；混合模式必须配置OSF降级模型。ROI小脸/无网格时使用OSF，默认阈值120px；全画面MP不自动切换检测器，CPU-only MP拒绝CUDA/TRT请求。MP Tasks的presence值不作为跨模型精度分数。ARM64尚未安装验证。

ROS JPEG含小于4KB的APP15帧来源标签，普通JPEG解码忽略它；不改变像素。用于在Core去掉ROS消息头后对齐回放帧与浏览器事件。标签保留来源frame_id/时间，不会自行认定来源时钟有效；只有确认同一时间基准后才能计算延迟。此功能不实现零拷贝。

## 测试、基准及平台状态

普通离线测试不访问摄像头；真实模型测试需显式证据目录，其models含yunet/osf0/osf3，fixtures含公开grace_hopper.jpg。不同性质的测试在验收报告中分开列出。

```bash
PYTHONPATH=perception python -B -m pytest -q -p no:cacheprovider perception/tests/face_effects_live
PYTHONPATH=perception FACE_EFFECTS_EVIDENCE=/your/evidence python -B -m pytest -q -p no:cacheprovider perception/tests/face_effects_live
PYTHONPATH=perception python perception/tools/face_effects_live/benchmark.py --evidence /your/evidence --output /your/results --frames 60
PYTHONPATH=perception python perception/tools/face_effects_live/acceptance.py --evidence /your/evidence --seconds 60
```

`test_real.py`的HTTP用例要求本机socket权限，所有服务器绑定127.0.0.1随机端口；不接入共享服务。旧`browser_acceptance.py`只代表独立页面；新`platform_acceptance.py`运行未替换lifespan的真实Core、真实Driver公共运行时、实际PerceptionBundle与DDS和原生Canvas，全部配置/数据库在新目录，源为公开自然视频。`ros_acceptance.py`为真实DDS适配器恢复测试。二者不会调用机器人SDK。

第二轮工具：`natural_dataset.py`提取固定自然窗口；`natural_benchmark.py`加载指定冻结源码路径，逐帧记录原始观测与效果锚点；`score_natural.py`同时报告条件误差和漏失，`score_motion.py`给出短序列抖动/延迟探索值与标签不确定性；`load_acceptance.py`以实际Service做有界缓存与长期回放。复现命令和来源哈希见ROUND2。旧照片合成测试保留为回归证据，不替代自然视频测试。

## 独立数据集评分

`dataset_adapters.py`支持300-VW PTS68、WFLW98、COCO17和有可信脸框时序声明的MOT格式，保留原始编号/可见性/时间与拓扑。它不下载受限数据或把模型输出变成真值。`fetch_coco_subset.py`只选官方元数据中CC BY 2.0图像，来源/照片页/逐文件SHA保存在独立输出目录。

```bash
python perception/tools/face_effects_live/dataset_benchmark.py \
  --labels /your/dataset/labels.json --images /your/dataset/images \
  --package perception/plugins/face_effects_live --models /your/models --model osf0 \
  --output /your/NEW-predictions
python perception/tools/face_effects_live/score_dataset.py \
  --labels /your/dataset/labels.json --predictions /your/NEW-predictions \
  --output /your/NEW-quality.json
```

COCO只评独立眼鼻语义代理和眼镜参考眼点，未穷尽脸框时FP为null，不报告标准人脸AP或完整68/98点NME；单张图片不能提供抖动、延迟和IDSW。真实视频用原始帧时间，只有可信逐帧身份/穷尽脸框才启用MOT指标。`compare_dataset_runs.py`检查输入哈希相同，再同时输出全体覆盖和共同匹配样本的前后误差。具体许可、关键点定义、命令及未取得的数据见ROUND4。

小脸可显式配置`detect_side=960, detect_upscale=true`；检测输入名义上最多放大2倍（另有32像素对齐），效果坐标仍为源分辨率。它增加小脸覆盖及CPU耗时，没有改善全部指标；默认640、不上采样不变。

| 平台 | 当前状态 |
|---|---|
| WSL2 Linux x86_64 CPU | 本轮独立静态参考、自然视频、三类特效与离线负载实际执行；完整平台为ROUND2历史成功，新ORT路径集成暂缓 |
| EGL软件GL | llvmpipe shader合成测试通过；无硬件加速结论 |
| 原生Linux硬件GPU / CUDA / TRT | 接口及降级已实现，真实加速未测 |
| Linux ARM64 / Jetson | 架构与依赖路径已规划；未在这些平台实际运行 |
| Windows | CPU代码设计兼容，未测；ROS/EGL需另验证 |
| ROS2 CompressedImage | ROUND2真实DDS及本地Canvas有历史成功；输入为自然视频回放，新ORT路径整链路暂缓 |
| ROS2 Image | 真实DDS恢复探针通过；Core尚无image/raw上游预览解码器 |
| USB / RTSP | 适配器已实现；真实设备、真实网络断流恢复待测 |

完整未决项、PR前门槛及真机清单见验收报告。不要把模拟断流、公开照片运动、ROS框架stub或导入成功视作完整平台验收。
