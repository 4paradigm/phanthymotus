# SuperMimic 接入审阅与验收指南

操作步骤、服务器连接和环境部署见 [USER_GUIDE_ZH.md](USER_GUIDE_ZH.md)。本文供 reviewer 判断代码范围、执行路径、测试证据与尚未完成的能力。

## 1. 本次提交的审阅目标

主仓库交付 ActuCore 动作学习卡片、独立 Motion 服务、环境安装器、共用工作台源码与测试。配套 Driver PR 交付 `simulator/mujoco` 仿真调试卡片；两者通过 HTTP 服务协作，运行时不读取另一仓库的源码。

主 PR 还包含一处必要的通用宿主改动：`agent-core/web/js/workbench.js` 增加工作台 `config` 消息桥，用于读取和保存卡片明确声明的连接字段。最新主分支已有 iframe 工作台，但缺少本功能所需的配置消息处理；只让插件自己保存，不能替代宿主会重放的画布配置。该变更不修改 Agent 推理或工具规划逻辑，应单独审阅字段范围、凭证回传与保存行为。配套 Driver 工作台也依赖它。

桥接仅处理当前卡片显式声明的共享字符串字段，不开放实例配置或其他卡片配置。插件应用与 Canvas 保存不是跨进程原子事务：保存响应异常时先回读确认，只有旧完整配置可恢复时才回滚；否则明确报告“连接已应用但保存失败，请重试保存”，不返回成功或刷新为成功状态。

主要行为是：在主画布打开内嵌工作台，选择机器人和参考动作，提交真实 SuperMimic 训练，查看真实训练采样、模型版本及检查结果，再在原生 MuJoCo 中试跑对应策略。服务器未准备时，可以通过独立安装器发起固定配方安装并查看实际进度。

这不是稳定跑步或通用全身技能模型的发布，也不是实体 G1 部署。请分别审阅“功能链路正确”和“策略效果合格”；本次只交付前者的实现与验证路径。

## 2. 建议阅读顺序

| 路径 | 关注点 |
| --- | --- |
| `agent-core/web/js/workbench.js` | 最小通用 `config` 消息桥；按卡片声明过滤字段，保存环境引用，避免凭证回传 |
| `actucore/plugins/motion/plugin.py`、`client.py`、`session.py` | MCP 操作白名单、连接存储、环境引用、异步任务与卡片会话绑定 |
| `actucore/motion_service/contracts.py`、`store.py`、`prepare_profile.py` | NPZ 格式、模型与 profile 哈希、MJB／MuJoCo 版本绑定、上传与资产管理 |
| `actucore/motion_service/resources.py`、`jobs.py` | GPU 授权范围、自动计划、排队和启动时重查、取消与服务重启语义 |
| `actucore/motion_service/worker.py`、`distributed.py` | 真实上游学习器、世界模型／策略更新、多 rank 同步、最终产物校验与 ONNX 导出 |
| `actucore/motion_service/policy.py`、`simulation.py` | G1 ONNX 合同、控制历史、参考动作绑定、PD／原生位置执行器语义、控制租约和 epoch |
| `actucore/motion_service/rendering.py`、`dataset_render.py`、`training_preview.py`、`camera_fit.py` | 真实 MuJoCo 画面、参考预览与物理运行分离、异步采样、帧状态一致性 |
| `actucore/motion_environment/` | 固定配方、本地 Docker 权限边界、下载与缓存校验、安装进程持久性、验证后才允许连接 |
| `actucore/plugins/motion/ui/` | 动作／版本／训练／试跑的页面流、两卡共用控制与连接逻辑、错误恢复 |
| `actucore/tools/build_motion_workbenches.py` | 从共用源码生成两份自包含 HTML，并检查是否过期 |
| 配套 Driver 的 `simulator/mujoco/` | 主页面独立仿真入口、观看／接管、与 Motion 服务一致的连接方式 |

共享前端源由主仓库维护。两仓各自提交生成后的 HTML；部署 Driver 不需要运行时安装构建器或访问主仓库。不要只修改生成文件，也不要只更新一个仓库的产物。

## 3. 关键合同与审阅问题

### 3.1 数据、机器人和模型

- 当前训练任务绑定单个 `dataset_id`，最终模型绑定同一动作、profile、运行配置和文件哈希。
- profile 从官方实际 GPU 任务提取；保留完整编译模型、求解器、执行器和网格资产，而不是根据机器人名称猜参数。
- 训练与正式评测使用 mjlab/MuJoCo Warp；交互会话使用原生 MuJoCo。共享资产与控制合同并不等于两个后端逐帧数值完全一致。
- `evaluated` 表示评测执行过，不是团队部署门槛；UI 的“动作完成”表示播放到片段末尾，不判断是否摔倒。
- ONNX 导出绑定参考动作；超过 15,000 帧的动作不会被标为已有当前适配器可运行的 ONNX。

### 3.2 多 GPU 和任务生命周期

- 自动调度仅在 `MOTION_GPU_IDS` 授权范围选择卡，不终止其他使用者的进程；提交计划与实际启动分配分别记录。
- 一个服务同时运行一个训练或评测作业。当前只支持单机多卡，不支持多机、训练中扩缩卡或跨卡迁移。
- 每个 rank 采样不同经验，同步网络参数、梯度和观测归一化统计。检查点发布前比较更新次数和状态哈希。
- 多卡禁用不适合显式同步的编译／CUDA graph 更新捕获，保留上游自动微分和学习器。
- 取消必须覆盖 launcher 及各 rank；服务重启把旧未完成训练标为 interrupted，不宣称自动恢复。

### 3.3 试跑误停修复

此前 native MuJoCo 接入把策略输出的期望关节目标与机械关节范围直接比较，导致部分合法的训练端 PD 目标在第一步触发 `policy target exceeds robot joint limits`。

修复只放行已验证的原生位置执行器：固定 kp/kd、单位 joint gear、有限力矩范围、有限关节范围，而且物理约束启用。MuJoCo 的力矩限制、物理关节约束和已有显式控制范围仍生效。一般电机、无力矩边界的执行器、关闭约束、非有限输出和非法初始姿态仍被拒绝。

请检查这里没有通过剪裁 qpos、修改机器人参数或关闭约束让测试“通过”。关节限位属于软物理约束，也不应把修复描述成“实际关节角度绝不瞬时越过边界”。

### 3.4 多窗口控制

- 每个 iframe 有独立 `view_id`；读取状态和观看不会获得控制权。
- 操作携带当前 `control_epoch`；接管原子转移所有权并暂停，旧窗口迟到请求不能继续控制。
- 可见控制窗口每 5 秒续约，20 秒租约到期后服务端暂停。直接关闭主页面弹窗依赖超时，不承诺即时暂停。
- 切换模型前先确认真实控制权，再释放或离开；网络失败或无法确认释放时不能把“新选中模型”显示成仍绑定旧会话的成功结果。
- 切换服务保留原服务资产；服务重启不恢复进程内物理状态。

### 3.5 安装器与连接

- 浏览器选择管理员登记的目标和固定配方，不能注入任意 SSH、shell、镜像或挂载路径。
- 安装器操作其所在控制环境的 Docker；`target.host` 不是远程 SSH 执行配置。
- 独立环境不自动发布宿主机端口，返回容器 bridge 地址。reviewer 应从实际 ActuCore／Driver 网络位置验证可达性。
- `connection` 返回的服务凭证仅供后端解析。前端保存环境任务引用，公开状态与日志不应暴露凭证。
- HTTP 安装器重启不杀掉已启动的独立 worker；任务失败／取消只清理本任务所属资源，保留已有运行服务。
- 每次发布同时更新运行服务和安装源码快照。只修当前容器会导致后续冷安装仍得到旧版本。

## 4. 本地可重复的检查

使用独立 Python 环境。Motion 服务的轻量依赖在 `actucore/motion_service/requirements.txt`；测试另需 pytest，ONNX 合同测试需 `onnx`，UI 测试需 Playwright 和 Chromium。安装具体依赖以对应测试文件的导入条件为准。

```sh
python -m pip install -r actucore/motion_service/requirements.txt
python -m pip install pytest onnx playwright pyyaml
python -m playwright install chromium
```

可以用 `MOTION_TEST_CHROMIUM_EXECUTABLE` 指定已安装的 Chromium／Chrome 可执行文件。不要使用操作员正在使用的浏览器 profile 或 CDP 会话来运行验收。

从主仓库根目录运行：

```sh
MOTION_TEST_DRIVER_ROOT=/path/to/phanthymotus-driver \
  PYTHONPATH=actucore PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  actucore/tests -q
```

完整 ActuCore 回归同时覆盖已有卡片；只验证 Motion 时可将路径改成 `actucore/tests/test_motion_*.py`。跨卡片测试需要配套 Driver checkout；默认查找主仓库旁的 `phanthymotus-driver`，隔离 worktree 或其他目录布局使用 `MOTION_TEST_DRIVER_ROOT` 显式指定。缺失时会跳过，必须把跳过项写入验收结果，不能把 skip 计为覆盖完成。浏览器依赖缺失的 skip 同样不算 UI 验收通过。全量测试中已有导航卡片等测试还有自身依赖，缺失项应按实际 pytest 报告记录。

对两份生成 HTML 检查：

```sh
python3 actucore/tools/build_motion_workbenches.py \
  --driver-output /path/to/phanthymotus-driver/simulator/mujoco/workbench.html \
  --check
```

需要真实离屏图形环境的测试显式启用，例如在配置好 EGL 的 GPU 测试环境：

```sh
MOTION_TEST_DRIVER_ROOT=/path/to/phanthymotus-driver \
  MUJOCO_GL=egl MOTION_TEST_GL=1 MOTION_RENDER_TEST_GL=1 \
  PYTHONPATH=actucore PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  actucore/tests/test_motion_dataset_render.py \
  actucore/tests/test_motion_rendering.py -q
```

这些命令验证对应合同和图形路径，不会自动完成正式策略训练，也不能替代实际 GPU 多 rank 和冷安装验证。

## 5. Reviewer 手工验收场景

建议使用专用服务数据目录和已授权 GPU，不复用他人正在运行或录制的会话。下面每项记录版本、任务 ID、配置、结果和实际观察到的限制，避免只记录一张“成功”截图。

| 场景 | 操作 | 应观察到的结果 |
| --- | --- | --- |
| 两卡入口 | 主画布执行各卡 `ui` | 页面在主应用弹窗内；没有独立网页跳转 |
| 已有服务连接 | 两卡填写同一 Motion URL/token | 同一机器人、数据、模型、会话；凭证不出现在返回配置中 |
| 环境准备 | 检查 → 安装／复用 → 主动连接 | 有实际阶段和日志；安装成功不自动切换；失败能定位阶段 |
| 参考动作预览 | 导入合法 G1 NPZ，播放／拖动 | 帧与时间对应，网格正确；未创建策略运行或训练任务 |
| 非法数据 | 错关节顺序、错频率或非有限值 | 导入被明确拒绝，不自动猜测或修补 |
| 小规模训练 | 自动调度提交短任务 | 有真实更新、rank 进度、最终权重和来源记录；过短无更新任务失败 |
| 资源等待 | 提交暂时没有足够资源的自动任务 | 明确排队，资源变化后重查；可取消，不抢占已有进程 |
| 多卡训练 | 选择两卡以上的合法配置 | 多 rank 实际运行、共同更新，最终状态一致；一个 rank 失败则整体失败 |
| 训练监看 | 训练中切分组／单实例／历史帧 | 真实采样帧，正确显示采样时间；暂停观看不停止训练 |
| 版本绑定 | 选择不同训练版本进入试跑 | 始终使用被选模型和其绑定动作，初始暂停，不暗中运行 |
| 策略完整片段 | 点击开始运行，观察到结束 | 不再第一步误判 PD 目标限位；保留力矩与物理约束；单独记录是否摔倒 |
| 重置与再次运行 | 结束后重置并运行 | 返回原参考初始状态，仍绑定原模型 |
| 跨卡观看／接管 | Driver 打开 Motion 会话，再明确接管 | 观看不能修改，接管先暂停，原窗口操作被拒绝 |
| 租约过期 | 关闭控制窗口或停止续约 | 租约到期后暂停；另一观看者不会被当成控制者 |
| 仿真录制回放 | 录制状态，暂停后逐帧回放 | 画面对应记录帧，实时物理不随回放变化；不生成训练 NPZ |
| 模型检查与导出 | 开始检查，再生成模型包 | 检查结果和绑定资产可追溯；不标为已通过真机验收 |
| 重启恢复 | 重启服务／安装器 | 训练中断如实标记，原物理会话不伪恢复；安装 worker 生命周期与 HTTP 服务分离 |

## 6. 已有实测证据与本次 PR 回归的区别

### 6.1 2026-11-17 整理 PR 时的本地回归

主仓基于 `7b16c43`，Driver 基于 `b3e8e788`，在独立 worktree 中验证；未在开发者原来混有其他任务的工作目录提交。以下是独立测试命令的结果，有重叠用例，不相加作为覆盖总数。

| 检查 | 结果 | 范围与限制 |
| --- | --- | --- |
| 完整 `actucore/tests` | 1112 passed，43 skipped | 包含已有 VLA／导航回归。首次运行尚未指定配套 Driver 路径，19 项跨仓检查及 2 项显式 GL 检查在下一项补测；另外 21 项导航测试缺少 cv2，1 项分布式测试缺少本地 torch |
| 指定 `MOTION_TEST_DRIVER_ROOT` 并启用真实 GL | 85 passed，0 skipped | 连接切换、跨卡控制、开发启动器、真实 MuJoCo JPEG 与参考帧，补齐上述 21 项环境性跳过 |
| `agent-core/tests/test_workbench_config_bridge.py` | 24 passed | 真实 opaque iframe／MessagePort，实际 Motion HTML 的连接保存、后续配置重放、重开恢复及失败分支；服务与 Canvas API 使用测试后端 |
| Driver 两份 `test_mujoco_*.py` | 20 passed，0 skipped | 指定 `MOTION_TEST_MAIN_ROOT`；真实 MuJoCo、独立浏览器、当前 UI 的观看／控制／步进／录制回放／重置／渲染错误恢复 |
| 两份 HTML 生成一致性 | 通过 | 使用构建器 `--driver-output ... --check`；源与交付 HTML 一致 |
| 发布内容检查 | 通过 | Python 编译、补丁空白检查、示例文档链接与 JSON/Python 代码块检查；未提交真实部署凭据或内部连接信息 |

本地物理测试使用 Python 3.13 与 MuJoCo 3.3.7，历史 GPU 部署使用固定配方的 MuJoCo 3.16.0；MJB 资产必须由相同版本加载。Motion 服务 15 个 Python 源文件的 SHA256 与下述五模型服务器验证时完全一致。此次整理补充了宿主配置桥、部署变量／持久目录、普通服务镜像的编译工具链及测试可移植性；没有把这些变化描述为已重新完成 GPU 长时训练或整套冷安装。

### 6.2 先前专用服务器上的实测

以下是整理 PR 前在专用联调部署中的历史证据，用于说明已经验证过哪些真实执行路径。**它们不能自动视作基于最新主分支整理后的最终 PR commit 已全部复测。** 本次最终 commit、测试命令、通过／跳过数量和差异由 PR 正文及 CI 记录给出。

| 历史实测项 | 观察到的结果 | 结论范围 |
| --- | --- | --- |
| GPU 训练与多卡接入 | 真实训练任务完成世界模型和策略更新，保存权重；单机多 rank 路径有更新与状态一致性检查 | 证明学习和同步链路运行，不证明短训策略稳定，也不保证多卡加速比 |
| 固定配方环境安装 | 独立环境通过 CUDA／Warp 运算、官方 G1 GPU 物理、原生 MuJoCo 渲染；双卡 NCCL 探针通过 | 证明该部署拓扑及固定配方可运行，不代表任意服务器无需管理员准备 |
| 限位修复的五个现有模型 | 各完成 150 控制步／3 秒，逐物理步力矩范围检查无越界，模型指纹保持，重置后可继续运行 | 证明旧接入误停已解除；五个短训策略仍会摔倒 |
| 当前主页面试跑 | 内嵌工作台运行至片段结束，MuJoCo 画面随状态更新 | 不再出现第一步误停；实际画面暴露模型失衡 |
| 主页面验收脚本的限制 | 首轮记录器遗漏 Motion 专用的 `simulation_frame` 模式，未收集到帧统计而提前断言失败；截图和运行状态已确认，随后经插件接口重置及交出控制 | 不能把该首轮完整脚本标为全部通过，也不能声称本轮完成了 UI 重置按钮的端到端验收 |
| 发布一致性 | 同时更新运行服务、安装源快照和已有隔离环境，保留服务连接及初始模型选择 | 避免新安装回退到旧限位逻辑；不是任意会话物理状态的热迁移 |

测试应区分：带伪造后端的 UI 测试、真实 CPU MuJoCo／ONNX 单元测试、真实 GPU 训练、真实浏览器主页面验证，以及最终策略质量。任何一类都不能替代另一类。

## 7. 当前未完成的能力

- 生产可用的跑步、站立或其他成熟策略及其质量基线。
- 从遥操作／动捕／视频自动生成可训练动作；人体动作重定向、重采样和质量筛选。
- 工作台中的多动作集合训练和一个模型选择多个参考动作。
- 速度／方向控制任务的完整导出、工作台交互和运行适配。
- 跨服务器多机训练、运行中弹性扩缩 GPU、自动断点续训。
- 完整训练录像／MP4 导出；当前为有限窗口的真实采样监看。
- 实体机器人下发、部署验收或安全接管。
- 任意服务器 SSH 注册及远端安装；当前安装器是管理员配置的本地 Docker 控制服务。
- 当前固定配方之外的任意机器人／仿真引擎通用支持。

这些限制应保留在 PR 描述和使用文档中，不能通过把短任务改为“已完成”、把评测状态改为“合格”，或隐藏摔倒画面来规避。

## 8. 升级和撤回

升级前记录源码版本、上游版本、profile 和产物哈希，备份状态目录与服务配置。等待训练／评测停止，主动暂停并交出交互会话控制后再更新服务；服务重启会丢失进程内物理会话，需重新创建。安装源快照与两份生成 HTML 同步更新。

撤回入口时关闭 `plugins.motion.enabled`，停止对应新增 bundle／服务即可；不要删除模型数据卷来“卸载页面”。已有模型与参考动作保留哈希绑定。固定安装器创建的容器和数据卷有归属标签，清理前按任务记录确认所有权，不能按模糊名称批量删除。
