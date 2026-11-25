# SuperMimic 训练与仿真服务

本服务在 GPU 服务器独立运行。ActuCore 的 `motion` 卡片和 Driver 仓库的
`simulator/mujoco` 卡片通过同一 HTTP API 使用它。训练、仿真和安装逻辑独立于 Agent Core；
内嵌连接设置需要本 PR 提供的通用工作台 `config` 桥，复用主页面已有的配置存储。

首次使用请先阅读 [完整操作与服务器连接指南](../../docs/supermimic/USER_GUIDE_ZH.md)；
团队审阅入口见 [实现范围与验证说明](../../docs/supermimic/REVIEW_GUIDE_ZH.md)。

- `motion`：统一动作工作台，包含参考动作、训练、MuJoCo 试跑和环境准备。
- `simulation`：原生 MuJoCo 交互会话、真实机器人网格画面、策略执行、扰动、录制回放。
- 当前不包含真实机器人全身接管或已验证的真机策略部署。

训练与正式评测使用官方 mjlab/MuJoCo Warp；交互调试使用原生 MuJoCo。
页面会显示实际控制模式。两者共享经过校验的机器人和策略资产，但不同仿真后端
的结果不能相互替代。控制窗口每5秒续约；关闭或断线后不再续约，服务端最迟在
20秒租约到期后的0.5秒检查周期内暂停会话。观看窗口关闭不影响控制者，训练任务和环境安装任务继续运行。
主动“释放控制”立即暂停并释放控制权，保留会话供重新打开。

## 主页面入口

在 Phanthymotus 主页面画布上打开 `motion` 卡片，选择 `ui` 并执行，即可弹出
“动作工作台”；`simulation` 卡片仍可单独打开高级仿真环境。
动作工作台提供三个主页面：

1. **动作**：选动作后显示真实 G1 参考姿态，可以播放、拖动进度和旋转视角。
   此预览只做 MuJoCo 前向运动学，不执行策略，不推进物理。模型历史默认折叠。
2. **训练**：自动带入所选动作和机器人。预设、实例总数、迭代数放在高级设置。
3. **仿真**：动作已有匹配模型或训练完成后，点击“在仿真里试跑”。工作台自动
   创建或恢复匹配模型的 MuJoCo 会话，停在初始状态。已有会话先点“取得控制”，
   再点“开始运行”才推进物理；只切换页面不会开始试跑。

模型绑定其训练时的动作、机器人配置和文件哈希，不能任意换一个动作执行。
“检查模型”和“导出模型”在模型详情中；检查完成不代表动作质量达标。

默认自动调度，由服务决定显卡数量和编号，页面显示预计分配；任务启动时按最新
显存和负载重新分配并显示实际结果。临时资源不足会排队，资源可用后自动启动，
等待期间可取消。在“高级选项”切到手动选择后，可逐卡勾选或一键选择1、2、4、8卡。
资源首次检测在后台完成，检测期间可先查看显存和负载，就绪后启用训练。
提交后训练页直接显示最多8个真实实例的采样画面、进度及停止按钮；支持放大
单个实例与最近采样回放。完成后保留最终画面，可直接试跑，或检查和导出模型。

右上角“环境”可查看已登记服务器、检查结果、安装步骤与真实日志。安装任务由
独立的 `motion_environment` 后台执行，目标 Motion 服务未安装时也能工作。
支持复用已有环境，或用固定配方创建独立 Docker 环境；关闭页面后任务继续，
重新打开会恢复状态。基础驱动、Docker 与 NVIDIA runtime 必须已可用，安装器
不会替换主机驱动或重启其他服务。目标服务器及凭证由管理员登记，浏览器不接收私钥。

ActuCore 设置 `MOTION_ENVIRONMENT_URL` / `MOTION_ENVIRONMENT_TOKEN` 接入部署后台。
连接成功的环境以非敏感 `environment_job` 标识保存在主页面配置中；实际服务凭证
只在后台解析和存储。手动切回服务地址时，工作台会清除该标识。
`MOTION_WORKBENCH_STATE_DIR` 可指定受保护的连接与会话存储目录，默认
`~/.local/state/phanthymotus/motion`。仿真调试卡可观看或显式接管动作学习的试跑；
动作学习接入已有会话时必须与当前模型和机器人精确匹配。接管暂停当前会话，旧窗口的迟到操作不能继续控制。

## 本机开发工作台

没有 ROS 的开发电脑可以直接打开现有两套工作台。在本仓库根目录运行：

```bash
python actucore/tools/motion_workbench.py \
  --driver-root /absolute/path/phanthymotus-driver \
  --endpoint http://127.0.0.1:15732 \
  --token-file /absolute/path/motion_api_token
```

凭证文件权限必须为 `0600`；也可省略 `--token-file`，从 `MOTION_API_TOKEN` 读取。
浏览器打开 `http://127.0.0.1:15733`。服务器在远端时先建立到 15732 的 SSH 隧道。
此入口只加载真实插件和原有工作台页面，不依赖 ROS，也不启动训练、仿真或机器人。
它是本机开发入口，不包含完整 Agent Core；页面内修改连接只保留到启动器退出，
不会改写凭证文件。凭证始终保留在 Python 进程内，不回传工作台 iframe。

## GPU 运行环境

兼容基线：SuperMimic `abc124c1bd5976341034668af3d12c087f2f1f55`，
mjlab_research `dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd`，Python 3.13。
保持它们在独立环境安装；不要装进机器人 ROS 环境。

镜像在本仓库根目录构建：

```bash
docker build -f actucore/motion_service/Dockerfile -t motus-motion:local .
```

也可以在已经安装上游依赖的 Python 3.13 环境中运行。将本仓库 `actucore`
加入 `PYTHONPATH`，使用该环境的 Python 安装本目录 `requirements.txt`，补齐 ONNX Runtime、
Pillow 等依赖，并核对 MuJoCo 与机器人资产版本保持一致。默认使用本地 W&B 离线日志，
本地检查点和评测读取不需要 W&B 账号；在线 W&B 的配置只留在服务器环境中。

## 创建准确的机器人配置

使用上游已能读取的一份动作文件，在安装了上游的环境执行：

```bash
cd actucore
python -m motion_service.prepare_profile \
  --motion-file /absolute/path/reference.npz \
  --output /absolute/path/g1-profile
```

这会从真实任务提取关节顺序、控制频率、物理步长、PD 参数、动作缩放和限位，
生成 `profiles.json` 及包含网格资源的 `scene/`。配置使用 `scene/scene.mjb`，
保留训练环境实际生效的物理步长、求解器及执行器参数；同目录 XML 用于查看与编辑。
MJB 必须使用相同 MuJoCo 版本加载。输出目录必须不存在。
它描述官方仿真机器人，不证明现场机器人具有相同自由度。

服务启动时固定配置，已有数据与模型绑定 `profile_hash`。绑定包含完整编译模型
（含引用的网格与 include 内容）的指纹及 MuJoCo 版本。正式训练、评测和交互会话
都会核对实际加载的基础模型；训练中的逐环境随机化仍由上游任务控制。
修改配置后旧资产不会被静默重新解释。模型路径只由服务器管理员配置。

## 启动与连接

```bash
export MOTION_API_TOKEN='<至少16字符的服务凭证>'
export MOTION_GPU_IDS='0,1,2,3,4,5,6,7'
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=2
python -m motion_service.server --host 127.0.0.1 --port 15732 \
  --profiles /absolute/path/g1-profile/profiles.json \
  --data-dir /absolute/path/motion-data
```

远程部署可在受控内网地址监听，或在前面配置 HTTPS 反向代理。服务本身默认
只监听回环地址。`MOTION_GPU_IDS` 是此训练服务可调度的物理显卡编号；未设置时
遵循 `CUDA_VISIBLE_DEVICES`，两者均未设置则使用容器可见显卡。`compose.yaml`
暴露可用 NVIDIA 显卡，并默认仅分配 GPU 0 给训练服务；通过 `MOTION_GPU_IDS`
设定实际分配范围，`MOTION_RENDER_GPU_ID` 指定 Compose 中的渲染显卡。
仅暴露显卡或读取资源列表不会启动训练。

工作台显示实际显存和利用率，支持在已分配显卡间共享调度。已有显存占用不会
自动排除一张卡；剩余显存低于4 GiB时禁止新任务使用它。此门槛用于启动前检查，
不是训练显存上限估计；实际需求随模型、数据和环境数量变化。提交和真正启动前
各检查一次，显存不足会给出减少环境数、改用小规模预设或增加显卡的操作建议。
任务不终止其他使用者的进程，CUDA 子进程按 GPU UUID 绑定选中的物理显卡。

无显示器的 NVIDIA 服务器使用 `MUJOCO_GL=egl`。EGL 设备选择由
`MUJOCO_EGL_DEVICE_ID` 控制，不能只依靠 `CUDA_VISIBLE_DEVICES`；该值是容器中
EGL 可见设备的索引。仅暴露一张显卡的容器通常使用索引 0。

两张卡片可选择同一已准备环境，并保存非敏感的 `environment_job`；两个 bundle 均需部署
`MOTION_ENVIRONMENT_URL` / `MOTION_ENVIRONMENT_TOKEN`，由后台解析服务凭证。手动连接也可设置
`MOTION_SERVICE_URL` / `MOTION_API_TOKEN`。已保存凭证不会返回 iframe；手动输入的凭证仅用于配置提交。
仅打开卡片、启动项目或重启容器都不会启动训练或机器人动作。

## 动作数据

首版接受 `.npz`，最多 64 MiB，解压后最多 128 MiB，`allow_pickle=False`。
必须包含以下数组：

| 字段 | 形状 / 含义 |
| --- | --- |
| `joint_names` | Unicode 字符串数组 `[J]`，与 profile 顺序相同 |
| `joint_pos`, `joint_vel` | `[T, J]`，弧度及弧度/秒 |
| `root_pos_w` | `[T, 3]`，世界坐标米 |
| `root_quat_w` | `[T, 4]`，单位四元数 wxyz |
| `root_lin_vel_w`, `root_ang_vel_w` | `[T, 3]`，世界坐标速度 |
| `fps` | 单值，必须等于 profile.control_hz |

网页通过现有 `call` 桥按小片上传，服务端按偏移验证并校验完整文件；无需新增
Core 文件接口。不会自动猜关节顺序、重采样或把录像直接当作机器人轨迹。

## 任务和模型

`POST /v1/call` 使用 `Authorization: Bearer …`，请求为 `{action, ...参数}`。
`GET /health` 也需要认证。主要操作：

- `catalog`、`dataset_preview`、`upload_begin/chunk/finish`。
- `dataset_render(dataset_id,offset?,width?,height?,azimuth?,elevation?,distance?,lookat?,jpeg_quality?)`：
  从已校验 NPZ 的根姿态和具名关节构建真实参考帧，返回 JPEG、帧号、时间和相机。
  独立于训练与交互会话，不调用 `mj_step`；只允许单自由根、单自由度关节机器人。
- `resources`：服务器 GPU、显存、负载、共享占用和可选状态。
- `plan(num_envs,preset,gpu_mode?,gpu_ids?)`：只读预计分配及能否排队，不预占显卡。
- `train(profile_id,dataset_id,num_envs,max_iterations,preset,gpu_mode?,gpu_ids?)`。
- `jobs`、`job(job_id)`、`cancel(job_id)`。
- `evaluate(profile_id,dataset_id,model_id)`。
- `export(model_id)`、`artifact_read(package_id,offset,limit)`。
- `training_preview(job_id,width?,height?,frame?,sample?)`：最多8个真实训练环境的抽样画面。
  `sample` 选择本次采样中的一个实例放大；省略显示全部真实样本。
  `frame` 为最近采样窗口内的位置（最多120帧）；省略读取当前状态。
  返回 `frame_id`、实际采样率、采样时间、画面年龄及可回放范围；相同状态不会
  被视为新帧。回放窗口会随新采样移动，`selected_sequence` 标识实际读取的记录。
- `simulation_create/inspect/run/pause/step/reset/close/perturb/record/record_stop/replay`。
- `simulation_render(session_id,width,height,azimuth,elevation,distance,lookat,offset?,jpeg_quality?,detail?)`
  返回 MuJoCo 离屏渲染的 JPEG、同一帧状态和会话信息；`offset` 指定已录制帧。
  `jpeg_quality` 为40–95的整数，默认88；`detail` 默认`full`，保留完整诊断状态。
  `compact`仅返回时间、步数、根姿态、关节名称/位置、接触数量和物体ID/名称，
  不传输几何体数组、接触点数组或逐物体矩阵。实时与回放图像均绑定对应帧状态。
- `simulation_inspect` 同样支持 `detail=full|compact`，默认`full`。
  空闲状态轮询可使用`compact`，诊断视图使用`full`；步进等操作仍返回完整状态。
- `simulation_list(view_id?)` 列出存活会话，仿真卡可重新接入，恢复刷新、超时或驱动重启后丢失的绑定。
- 新工作台每次打开生成独立 `view_id`。`simulation_create` 同时传 `request_id`，同一创建重试复用该标识，
  服务端返回原会话；重用标识改模型或机器人会被拒绝。幂等记录在当前服务进程保留最近1024个请求。
- `simulation_claim(session_id,view_id,control_epoch,takeover?)` 按观察到的 epoch 原子取得控制；
  已被占用时必须显式 `takeover=true`。`simulation_heartbeat` 续约，`simulation_release` 暂停并释放；
  它们以及所有物理/录制/关闭操作都需相同 `view_id` 与 `control_epoch`。
  返回 `control={owner_view_id,epoch,expires_in_s,lease_seconds,is_controller}`，只读查询不会续约。
- 插件 `simulation_attach` 只建立当前窗口的观看绑定；`simulation_detach` 只离开该窗口。
  主页面关闭回调不负责停止物理，服务端租约负责失联暂停。旧无 view 的会话在滚动升级期间仍可使用；
  首次 claim 后永久启用控制校验，无 view 请求不能操作它。服务重启无法恢复物理状态，只能重新创建会话。

提交立即返回任务 ID，作业单队列运行，记录持久化到数据目录。取消会终止该作业
launcher和各rank进程，包括另建进程组的torchrun子进程；服务重启将旧未完成记录
标为 `interrupted`，不自动恢复训练。界面的
“暂停仿真”只影响交互会话，与训练恢复无关。

### 单机多 GPU

`gpu_mode=auto` 不传 `gpu_ids`，服务自动选择显卡数量和物理编号；省略模式且
未提供显卡编号时也使用自动调度。显式 `gpu_ids` 的旧请求仍按手动模式处理。
自动任务提交时的分配只是预估；排队后真正启动前重新查询显存和负载、固定 UUID、
写入实际请求与配置，再启动训练。训练过程中不改变显卡数，不自动迁移或抢占
其他任务。临时等待任务不会挡住后续已有足够资源的任务，服务内同时运行一个训练或评测作业。

自动模式采用可配置的资源保留策略：small默认每卡目标1024个实例、至少8 GiB
空闲显存；standard为512个实例、至少32 GiB。实例数超过每卡目标时，显存保留
门槛按比例增加。卡数须整除总实例数，优先选接近目标且资源足够的组合；卡内
按利用率低、空闲显存多排序，已有显存占用的卡可共享。默认仅自动选择利用率
不高于90%的卡。以上是调度策略，不是峰值显存预测或加速比保证。
可用 `MOTION_AUTO_ENVS_PER_GPU_SMALL/STANDARD`、`MOTION_AUTO_FREE_MIB_SMALL/STANDARD`
及 `MOTION_AUTO_MAX_UTILIZATION` 调整；前两组变量分别使用正整数实例数和MiB。
瞬时显存/负载不足时自动等待；当前硬件总容量也不满足策略的请求会明确报错。

`gpu_ids` 可选择1–8张已分配显卡。`num_envs` 始终是全任务机器人实例总数，
必须能被显卡数量整除，例如64实例、8卡为每卡8实例。多卡采用独立torchrun任务，
各rank采样不同数据，共同训练一份世界模型、actor和critic。网络初始参数和buffers
从rank0广播；每个优化步骤在反向传播之后、梯度裁剪之前平均各卡梯度。
观测归一化沿用上游支持分布式的全局count/mean/variance统计。

学习batch等分到各rank；不能整除时向上补齐为显卡数量的倍数，原始和实际batch
均写入 `run.json`。多卡路径关闭Torch编译和CUDA graph更新捕获，保证每次更新执行
同步；自动微分及上游学习器保留。仅rank0保存检查点、导出ONNX并登记模型。
发布成功前比较各rank的更新次数与网络/归一化状态哈希，任一rank失败则整个任务失败。
当前提供单机多卡，不提供跨服务器多机训练。

任务列表从各rank实际进度文件聚合迭代、阶段、耗时和更新次数；缺rank时不虚报
整体进度。工作台默认连续监看真实环境状态，每rank默认最多每秒采样4次，全任务
最多8个实例；`MOTION_TRAINING_PREVIEW_HZ` 可设置1–8的上限。CUDA状态复制到
单个异步缓冲区，后台写入最近120次采样，缓冲区忙时跳过而不积压。服务器独立
渲染，不推进训练物理；相同快照复用已渲染JPEG。

采样频率是上限，实际速度取决于真实环境步进和学习更新；算法更新期间没有新
物理帧，页面显示已有帧和更新时间，不补造姿态。这是连续采样监看及采样回放，
不是30fps录像，也不包含MP4导出。旧任务没有历史采样时只能显示其最终快照。
缺少rank、过期或尚未采样均明确返回状态。暂停画面或离开页面不停止训练。

训练固定使用 `proprio` 观测和 profile 的历史长度。`small` 预设缩小学习批次，
让少量并行环境也能及时更新，并关闭 PyTorch 编译优化以减少小作业冷启动耗时；
自动微分和真实训练更新保留。`standard` 保留上游学习批次与编译优化。
两者均记录实际参数和编译开关。
没有发生世界模型及策略更新的短任务不会被标成成功训练。每次训练保存真实的最终
actor/world model 权重，符合上游条件时导出 motion ONNX。超过 15000 帧的数据仍可
训练，但不会被标成已有可部署 ONNX。模型与数据、profile、运行配置、版本一同保存。

正式评测在 mjlab 中运行原策略并比较 ONNX 输出，记录关节 RMSE、提前终止次数、
超时次数和回放数据。`evaluated` 仅表示完成评测，不表示达到真机部署阈值。
模型包包括 ONNX、参考动作、profile、运行配置、文件哈希及已有评测报告。
数据、检查点和 ONNX 在训练、评测、导出及加载前验证文件哈希；改变或损坏的文件会被拒绝。
当前工作台生成服务器侧模型包；下载可使用已认证的 `artifact_read` 分片接口。

原生仿真策略适配只接受验证过的 G1 motion/proprio ONNX 元数据，包含历史和未来
参考动作；不支持的布局明确拒绝。策略结束或发生错误时停止该调试会话。
经过验证、具有有限力矩范围且启用物理关节约束的原生位置执行器，允许策略目标角度
超出机械角度范围，与训练端的控制方式保持一致；这不关闭 MuJoCo 力矩限制或关节约束。
其他执行器仍保留目标范围检查，非有限输出、非法初始状态和异常物理状态会被拒绝。

当前一次训练绑定一个参考动作；录制回放尚不生成训练 NPZ，速度/方向指令的导出和
部署尚未接入。预置联调样例及其短训模型不代表稳定动作能力；已完成三秒试跑并不
等于不摔倒或通过质量评测。

工作台默认显示服务器 MuJoCo 渲染的机器人外壳、网格、地面与光照。浏览器负责
显示 JPEG 和提交相机参数；骨架与简化几何仅作为可选诊断视图。每个会话的 OpenGL
上下文在专用线程中使用独立模型和状态，只执行 `mj_forward`，不推进物理。渲染等待
不持有物理会话锁；录制回放渲染也不改变实时环境。画面不可用时会显示具体错误。
交互帧可使用640×360、质量70、`compact`；停止交互后使用960×540、质量88、
`compact`。相同比例共用一个960×540原生渲染上下文，编码前缩放图像；清晰度和
JPEG质量切换不会重新加载模型网格。宽高比改变时才需要重建上下文。

## 验证

```bash
cd actucore
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_motion_service.py \
  tests/test_motion_card.py tests/test_motion_policy.py tests/test_motion_simulation.py \
  tests/test_motion_asset_binding.py tests/test_motion_runtime_binding.py -q
```

本地测试包含真实 MuJoCo 物理步进和真实 ONNX Runtime 推理。训练更新与 GPU
评测必须额外在安装了指定上游版本的 NVIDIA 服务器运行；UI、mock subprocess
测试不能替代这项证据。停止服务和禁用 `motion.enabled` 即可撤下新增入口，已有
VLA、导航及机器人上肢控制路径不依赖该服务。
