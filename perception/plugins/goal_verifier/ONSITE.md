# 现场交付：独立 PiPER 视觉验证，不启动运动 Owner

本说明不表示现场已经部署或验收。主路径是**自己拥有的只读MCP进程**，接已经运行的相机Topic，注册到获准使用的Core。无需重启/修改 PiPER owner、CAN、公共容器或运动许可。首次验收只需人工移动物体。

## 1. 包与隔离目录

开发仓库中生成新包（输出目录必须不存在）：

```bash
python perception/tools/goal_verifier_cli.py package --output /absolute/own/package-UNIQUE
```

包包含Card、官方复用的measure/VOP格式辅助源码、MCP入口、CLI、配置/依赖、测试及文档，含逐文件MANIFEST和archive SHA。没有旧Visual Push变更、现场快照、授权、真实身份/标定/图像、个人日志、模型权重或Core数据库。

获准复制到实验室自己的目录后，解压到全新个人目录，使用包内CLI做hash校验和stage（这两步及doctor可在未安装视觉依赖时运行；缺依赖返回明确诊断）：

```bash
export GOAL_PAYLOAD=/absolute/own/unpacked/piper-goal-verifier
export GOAL_CODE=/absolute/own/piper-goal-run-UNIQUE/code
export GOAL_DATA=/absolute/own/piper-goal-run-UNIQUE/data
export PYTHONPATH="$GOAL_PAYLOAD/perception"
python "$GOAL_PAYLOAD/perception/tools/goal_verifier_cli.py" stage \
  --payload "$GOAL_PAYLOAD" --output "$GOAL_CODE"
mkdir -m 700 "$GOAL_DATA"
```

记下run_id。不要覆盖既有目录。不要把包解压到公共`/work`或旧Visual Push仓库。

## 2. Python与视觉环境

独立venv的固定依赖面向Python3.10及以上；Jetson镜像使用镜像自身匹配的Python/NumPy/OpenCV，不使用该固定依赖覆盖NVIDIA栈。先source**现有**ROS Humble环境，再建立自己的venv，保留rclpy可见性：

`deploy/requirements.txt` 仅供独立环境使用，官方 Perception Docker 构建不会安装它，因此不会增加官方镜像的依赖体积。安装这些依赖会增加独立环境体积，尤其是 `opencv-python-headless`。

```bash
source /opt/ros/humble/setup.bash
python3 -m venv --system-site-packages /absolute/own/piper-goal-run-UNIQUE/venv
export GOAL_PYTHON=/absolute/own/piper-goal-run-UNIQUE/venv/bin/python
"$GOAL_PYTHON" -m pip install -r "$GOAL_CODE/perception/plugins/goal_verifier/deploy/requirements.txt"
export PYTHONPATH="$GOAL_CODE/perception"
export PYTHONDONTWRITEBYTECODE=1
```

所有安装只进新venv。不要升级全局NumPy/OpenCV/ROS，不改旧项目环境。若使用现有Perception镜像环境且依赖已满足，可直接用其Python读取私人代码目录，无需安装或替换公共镜像；进程和端口仍属于自己。只用MCP规范输入时不需要rclpy；ROS数据路径必须有真实rclpy/sensor_msgs/std_msgs，测试stub不能算安装成功。

## 3. 生成并填写配置

```bash
"$GOAL_PYTHON" "$GOAL_CODE/perception/tools/goal_verifier_cli.py" template \
  --mode color --output "$GOAL_DATA/config.json"
```

模式可为color、vop、normalized；空模板故意不可用。填写：

- piper_id：自己确认的PiPER安装标识，不伪装SDK序列号。
- source_id：软件来源键；camera_id：已核对的物理相机身份；frame_id：实际消息header；distortion：实际raw/rectified。
- domain_id：平台 ROS Domain 固定为 42，所有 ROS 数据源须显式填写 JSON 整数 `42`。其他数字、`"42"`、布尔值均拒绝，不会自动纠正；外部 executor 的 Context 也必须是域 42。
- topic、message_type：从实际ROS图确认。compressed只接受JPEG，raw为bgr8/rgb8。不可凭Topic名字认定类型。
- stamp_semantics=capture、capture_time_record：实际核实header stamp来自采集时刻的依据。ROS消息有header不等于其语义自动可信；静态图片发布器的发送时间不能作为采集时间。优先使用经核对的原始相机采集流或有真实采集元数据的标准生产者，不能把该静态照片时间改标签。
- 图像源view_id、view_mode=fixed、fixed_view_record：确认本次图像目标比较期间拍摄几何不变的记录；相机移动、缩放/裁剪后不可沿用。移动相机用逐帧注册的规范观测及对应测量标定，不能贴一个固定view标签。
- color模式classes：自己选择的类别名、颜色名、HSV范围、最小面积。它是显式颜色类，不冒充YOLO语义识别。多个同类同颜色物体会产生身份歧义，首次验收可选视觉上明显且可区分的目标。
- VOP模式：真实vop_mcp_url及工具名。VOP需支持recognize_by_url，能回连本Card的短时图像URL，模型已就绪且推理未超出采集年龄上限。容器网络/代理必须允许这条路径，localhost不自动等于另一个容器；host网络/同主机是默认可达条件。不要修改别人的代理配置。不可达时用CPU路径或记录阻塞。
- 默认allow_test_data=false、verified_persistent_tracks=false、accept_external_observations=false。只有实际核实生产者的持续身份/观测来源，才填写identity_record/external_observation_record并显式开启。stock VOP/measure不提供持续track保证。

normalized用于已经符合Observation v1的生产者。ROS topic中的原VOP JSON缺少采集时间/身份不能直接接；CLI/接口会拒绝。真实驱动若只输出无header的JPEG字节流，也不能把接收时间假装采集时间。

Topic和消息类型应从本次实际视觉环境确认，不预设相机或安装场景。核实来源遵守平台 `ROS_DOMAIN_ID=42` 约定；不符合时记录阻塞，不修改 PiPER 控制 Owner 或公共服务的域，也不连接 CAN。

## 4. 一键只读预检

```bash
"$GOAL_PYTHON" "$GOAL_CODE/perception/tools/goal_verifier_cli.py" doctor \
  --config "$GOAL_DATA/config.json" --ros --sample --output "$GOAL_DATA/visual-preflight.json"
```

检查依赖、配置、ROS域、Topic真实类型、采集时间、相机/frame/尺寸、实际数据结构和可解码检测。只创建并关闭自己的视觉订阅；VOP采样可临时提供自己的内存JPEG端点，不启动任何机械臂对象。不填--ros/--sample时只检查配置，不能称为真实数据通过。

域校验使用 [ROS Humble Context.get_domain_id](https://github.com/ros2/rclpy/blob/humble/rclpy/rclpy/context.py)，现场仍需确认实际安装接口。已有VOP记录可用`inspect-vop --input ... --frame-record ... --config ... --source-id ... --output ...`检查；缺少原始采集记录会输出CAPTURE_TIME_UNPROVEN，不自动修复。

## 5. 启动与显式注册

先确认自己选择的空闲端口，不停用占用者。以自己的会话/进程管理器启动：

```bash
export GOAL_CONFIG="$GOAL_DATA/config.json"
export GOAL_PORT=15785
bash "$GOAL_CODE/perception/plugins/goal_verifier/deploy/start.sh"
```

默认127.0.0.1，只适用于同主机Core（包括host网络容器）。跨主机应由维护者配置受保护的可达方式，本工具不擅自开放LAN或改公共代理。

维护者确认实际Core协议和URL，不能据端口号猜测服务身份。获准注册后运行显式命令：

```bash
export GOAL_CORE_URL=https://OWNER-CONFIRMED-CORE
export GOAL_MCP_URL=http://127.0.0.1:15785/mcp
"$GOAL_PYTHON" "$GOAL_CODE/perception/tools/goal_verifier_cli.py" register \
  --core-url "$GOAL_CORE_URL" --mcp-url "$GOAL_MCP_URL" \
  --name 'PiPER Goal Verifier / own-UNIQUE' --output "$GOAL_DATA/registration.json"
"$GOAL_PYTHON" "$GOAL_CODE/perception/tools/goal_verifier_cli.py" doctor \
  --config "$GOAL_DATA/config.json" --core-url "$GOAL_CORE_URL" --mcp-url "$GOAL_MCP_URL" \
  --output "$GOAL_DATA/core-preflight.json"
```

Core需要时增加`--ca /own/ca.pem --token-file /own/core-token`，不关闭TLS验证。CLI不会覆盖已有同名/同URLMCP，注册分类固定perception；正常doctor不注册/更改配置。Core原注册API不支持跨客户端事务，现场须避免并发注册并核对receipt；没有为此改公共注册机制。

## 6. Dashboard与最简验收

使用兼容`motus.workbench/1`的官方Dashboard。本分支的只读入口在侧栏该processor上显示“打开工作台”，无需开启项目，不触发其他Card。前端最小patch与目标sidebar文件在包的Core集成目录，**仅供维护者授权评审**，没有自动应用脚本。不得自行覆盖公共Core。若现场版本没有此入口，但已有官方openToolConsole，可由获准操作员在同源Dashboard开发者控制台调用官方宿主：

```javascript
const list = await (await fetch('/api/mcp')).json();
const owner = list.data.find(m => m.url === '实际注册的MCP完整URL');
const card = owner.tools.find(t => t.name === 'piper_object_goal_verifier');
(await import('/js/workbench.js')).openToolConsole(owner.id, card);
```

这是调用现有工作台，不改服务器文件或执行项目。使用真实MCP URL替换一次即可。若原Core连该宿主都不支持，预检BLOCKED，需维护者批准兼容版本，不能冒充UI验收完成。不要为了打开页面启动包含他人运动Card的整个项目。

正常成功：启动本Card视觉→读取多个检测并选择物体→填区域/类别/颜色/坐标系→创建会话→记录操作前状态→人工移动进入目标→标记操作已完成→获取新观测并验证→PASS。物体保持区外应FAIL；遮挡/移走/同外观歧义应INCONCLUSIVE。

先填写目标参数再读取并选物体，减少基准过期。每次判断是该帧中心条件，不保证之后持续在位。读取/颜色/视角发生变化时新建会话，不修改旧成功记录冒充新结果。外部PiPER操作也使用同样API，不与推动动作绑定。

## 7. 标定、录制、证据

第一轮用pixels/normalized，无需真实毫米标定。需要plane_mm时准备每个view独立实测记录：piper_id/camera_id/frame_id/view_id、实际image_size/distortion、created_at/expires_at、measurement_record/object_plane_record/class_name、valid_pixel_bounds、≥4组非共线fit_pixels/fit_mm、≥3组独立check_pixels/check_mm、max_reprojection_px/max_check_mm、measurement_uncertainty_mm。运行`fit-plane --measurements ... --output ...`，将结果放入自己config.calibrations对应ID，重启自己的只读服务；不修改原运动标定。

工作台“导出证据”提供可全选复制保存JSON（宿主sandbox禁直接下载）。CLI可直接写0600文件：

```bash
"$GOAL_PYTHON" "$GOAL_CODE/perception/tools/goal_verifier_cli.py" export \
  --mcp-url "$GOAL_MCP_URL" --session-id '实际会话ID' \
  --session-token-file "$GOAL_DATA/session-token" --output "$GOAL_DATA/session-evidence.json"
```

保存预检、配置SHA、前后观测/图像、PASS/FAIL/INCONCLUSIVE各一份、错误/重放/重置结果、Dashboard截图与必要视频、安装包/源码SHA、操作者和实际消息类型/时钟证据。会话停止/重启会退役，先导出。照片/录像通过已授权相机录制方式另存；本Card不秘密录像或把图像上传公共服务。

历史样本可用`replay --record ... --output ...`：record包含schema=motus.piper-goal-replay/1、config、selector、goal、before、after、operation_completed_at。采用录制时间轴，原时间不变、输出明确REPLAY；不能把它提交到真实默认Card制造新观测。

## 8. 回退与排错

1. 只停用本Card/本次拥有的进程，不操作PiPER owner或公共服务。
2. 使用自己的registration receipt执行`unregister --receipt ... --output ...`；工具核对UID、URL、name和id，变化则保留并拒绝。不要删除他人MCP或公共画布节点。
3. 保留GOAL_DATA、截图、日志和配置。卸载代码先`rollback --stage "$GOAL_CODE" --run-id '...'`干检；确认无人引用后加`--remove-owned-stage`。有新增/修改/符号链接就拒删，代码目录与venv/数据分开放。
4. 未授权安装过Core补丁则无需回退Core；若维护者另行批准了Core更新，回到其记录的版本，由维护者操作，工具不重启/改写公共Core。

常见错误：SOURCE_TIMEOUT查Topic/域/时间；CAPTURE_TIME_UNPROVEN查原时间来源，不能填接收时间；CAMERA/VIEW错误查实际身份/几何；IDENTITY_AMBIGUOUS/LOST重新选择可区分物体或建立新基准，不能只放宽阈值；VOP_FAILED查URL回连、代理、模型可用和推理延迟；CALIBRATION_REQUIRED退回明确图像目标或完成测量；RETIRED_SESSION使用新token。任何预检的READY FOR ONSITE TEST都不表示真实视觉已验收。

证据导出已去掉session_token；CLI导出所需token由授权调用方单独保管。发布真实现场截图/录像前遮盖界面会话能力及敏感身份，不把运行日志/真实配置加入PR。
