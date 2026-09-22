# PICO 遥操卡片

`teleop` 是普通 ActuCore 内的一张卡片。天轶的日常流程是：**在 Canvas 建立 teleop → motion_control → arm 三卡连线 → 安装并配对 PICO → 开启智能控制 → 在 PICO 开始遥操 → 双握把控制 → 在 PICO 结束或关闭智能控制收臂**。Canvas 负责连线、配置和项目权限；PICO 提供双手柄输入、透视画面及就地操作按钮。日常使用不需要后端脚本。

此前天轶双臂已完成实体 PICO 跟随、松握把保持、恢复与结束收臂的现场验证；这些是旧操作链路的证据。本轮已验证真实关节反馈、EEF 求解、ActuCore 显示回传及 Canvas 三卡 Shadow 启停。2026-09-23 经用户批准，通过 ADB 将实体北京 PICO 迁移到 release 0.3.18，邀请预填、一键配对、冷启动自动重连，以及透视、按钮和静态模型显示通过；随后发现开始时旧 Native parser 拒绝末端能力字段并持续重连，已修复并覆盖安装 0.3.19；保留配对，空闲连接与跟踪正常。尚未完成 PICO 浏览器安装或修复后的手柄动态模型、真机动作验收。灵巧手、其他机型及长期稳定性不因此视为通过；北京 G1 的展示与真实运动验收单独记录。

## 先找到卡片

1. 打开机器人 PhanthyMotus 页面，进入 Canvas。
2. 在侧栏 **执行（ActuCore）** 中找到 `teleop`，拖到画布。如果已经有遥操卡片，直接使用原卡片；当前只支持一个实例。
3. 新天轶卡片选择 `control_backend=motion_control`。在 **连接机器人卡片** 中选定当前天轶 Driver，点击 **建立三段连线**：`teleop` 的 `control/eef` 输出 → `motion_control` 的末端输入，后者的 `control/joint` 输出 → 同 Driver 的 `arm`。保存画布；两段都要求唯一目标，不能并接其他命令来源。也可按对应端口手动连接。
4. Canvas 自动显示 motion_control 返回 teleop 的反馈虚线；它不是动作输入，不纳入启动依赖图，不需要再画 arm 到 motion_control 的反馈边。Driver 内部直接共享实测快照。
5. 展开卡片查看连接 Driver、智能控制、PICO 连接和硬件执行状态。天轶提供“暂停保持 / 恢复遥操 / 结束并收臂 / 立即停止”等操作；日常开始入口在 PICO。G1 未接入这套项目启停与收臂生命周期，见下方机型边界。

开启智能控制时，Core 根据已保存连线、当前 MCP 注册地址和 Driver 的 `x-teleop-target` 声明解析目标，核对机型、`motus.control/2` 端口、执行资源及真实 ROS topic；motion_control 与 arm 必须属于同一 Driver。缺少连线、多目标、错误端口或不兼容 Driver 会明确拒绝开启，不会回退到历史配置地址。

侧栏没有 `teleop` 时，表示当前 ActuCore 未启用插件，或镜像没有提供该能力。添加一张普通参数卡不能替代服务部署。卡片存在但没有中文面板时，检查 Agent Core 是否支持 `x-connection-panel: teleop-v1`。

## 配置服务

关闭智能控制，等待收臂及控制权释放完成后，展开 **服务配置**。保存需要当前项目停止且具备 Canvas 编辑权；停止失败时先处理原因并重试停止，不通过改连线或改配置绕过。保存配置不会开始运动。

配置按职责分开，字段名与实际卡片 schema 一致：

| 卡片 / 字段 | 含义 |
|---|---|
| teleop：`mode` | `shadow` 只预览；`live` 才可能申请执行权。默认 Shadow |
| teleop：`robot_profile` | 新链路为 `tianyi2`；G1 保留 `g1_23` 旧路径 |
| teleop：`control_backend` | 新链路选 `motion_control`；已有 `legacy` 项目不自动迁移 |
| teleop：`mapping_version` | 天轶使用 `relative_v1`；G1 的 PR #152 映射仍仅用于旧路径 |
| teleop：`position_scale` | 相对位移比例，默认 0.5，可设 0.01～1 |
| teleop：`controller_to_palm` | 左右 `left/right` 手柄到手掌的变换对象；每侧为 `position` 和 `orientation`，四元数顺序 xyzw |
| motion_control：`calibration_path` | Driver 容器内已挂载的标定 JSON 路径，包含控制 URDF、TCP 和碰撞边界；这是导入入口，不是文件编辑器 |
| motion_control：`joint_velocity_rad_s` | 双臂速度，表单默认 1 rad/s；必须大于 0，不超过实际 URDF 限速且不超过 1.5 rad/s，检查 info 读回值 |

teleop 中保留的 `calibration_path` 只服务 legacy 本地求解，新链路模型在 Driver 配置；`shadow_feedback_source=driver_joints` 只用于 G1 Shadow。项目管理时 `namespace`、`driver_mcp_url` 从真实连线解析，不靠日常手填地址绑定执行对象。

点击 **保存并应用配置**后检查读回值。Driver 修改模型或速度要求没有 Live/Preview 身份、准备会话、已有动作或收臂；候选 URDF/TCP/碰撞/速度全部验证成功才原子替换。配置后需重新标定，失败保留原配置，不修改原标定 JSON 或 config.yaml。结束并释放后仍有 Preview 身份时，先明确停止对应会话，再保存。

**诊断与维护 → 重新标定**在新链路调用 Driver 验证模型与当前实测状态；开始时也会重新准备。它不移动机器人，也不重新测量真实尺寸。TLS、配对状态和管理密钥仍属部署资料。

`motion_control` 保存时，Core 先调用 Driver `config`，再通过 `info.config` 核对一致，最后保存确认过的完整配置。拒绝、超时或读回不一致不保存候选，页面显示失败原因；回执丢失时运行配置可能已经变化，应核对后重试，不假装已回滚。ActuCore 另保留已接受 teleop 参数的恢复缓存。重启恢复参数，不恢复 armed、动作或执行租约。使用完整表单保持 Core 与服务保存值一致。

## 首次配对与后续连接

1. 展开 **安装 PICO 与一键连接**，点击 **生成安装链接**，查看 APK 版本、二维码和 15 分钟短地址。APK 从当前 ActuCore 容器经 Core 受控代理下载；每个链接最多下载三次（含重试），重新生成会撤销旧链接。Core 重启后保留剩余有效期。
2. 在 PICO 系统浏览器打开地址，下载并经系统确认安装；手机扫码只会打开手机网页，不能自动安装到头显。PICO 系统扫码是否可用仍需现场验证，也可输入短地址。
3. 在 Canvas 点击 **生成一次性连接邀请**，使用更新后的二维码/链接在 PICO 浏览器打开。安装后返回该页，点击 **打开应用并连接此机器人**；连接页预填机器人配置，再点击 **连接当前机器人**。
4. 邀请有效期 15 分钟、单次兑换、可撤销，生成新邀请使旧邀请失效。客户端先验证机器人证书，再兑换配对；下载不占用旧 120 秒配对窗口。直接点安装器的“打开”不会继承原链接，须回原安装页再点连接，或使用下面的备用流程。

已配对冷启动会自动连接保存的机器人；网络暂断使用退避重连。返回连接管理页可取消或更换设备，其他机器人邀请不会静默覆盖原身份。**断开头显**保留配对；**撤销配对**后需重新申请。配对、重连和安装都不会开始运动。

备用流程仍可用：Canvas **连接与配对 → 允许新设备配对**打开 120 秒窗口，PICO 通过 mDNS 或手动地址申请，核对两端指纹后分别确认。邀请失效、证书变化或凭证撤销时不能跳过身份校验。

发布 APK 使用稳定 release 签名。旧 debug APK 与该签名不同，不能直接覆盖；须安排明确的人工迁移，卸载会清除本地配对信息，随后重新安装配对。代码不自动卸载应用，也不静默清空凭据；安装后的系统权限、账号或 MR 安全边界确认由操作员完成。

## 从 Canvas 开始一次遥操

以下流程用于天轶双臂；G1 不支持这里的项目启停与收臂步骤。

1. 完成上述配置、完整三卡连线和 PICO 配对，核对 `Shadow`／`Live` 模式并保存画布。
2. 在 Canvas **开启智能控制**。遥操卡片只进入 `armed`、显示“等待 PICO 开始”，不会因这一步创建运动会话、申请执行权或发送双臂目标。下游 motion_control 与 arm 准备完成后才允许开始；项目中其他卡片仍遵循各自的启停行为。
3. 正确佩戴 PICO，左右手柄分别持于身体两侧，打开透视采集页，松开双握把和扳机。用手柄射线指向 **开始遥操**并扣一下扳机，等待“等待双握把”。开始入口按实测反馈标定并准备会话；该标定不移动机器人，也不重新测量模型尺寸。准备时若仅遇到短暂反馈过期，会在同一 500 ms 截止时间内等待新反馈重试，停止请求可以取消；持续过期或真实安全错误仍返回具体原因，不放宽运行中的反馈门槛。
4. 同时按住左右侧面握把，移动手柄。只有 Live 且 Driver 反馈“硬件正在执行”时，才表示正在下发真实动作；Shadow 的模型运动不会驱动机器人。
5. 松开任意握把，双臂保持。再次握住时，从当前手柄和实测双臂姿态建立新的相对基准。
6. 正常结束可以在 PICO 点击 **结束并收臂**，也可以在 Canvas **关闭智能控制**。观察收臂和释放控制权的最终结果，不把按钮回执当作已完成。

未开启智能控制时，PICO 保留按钮并显示“请先开启智能控制”，开始按钮不可用；结束与立即停止仍可请求。旧服务缺少 `operator.armed` 时也不允许开始。开启期间已按住的扳机不会自动触发开始，需要松开后重新点击。

| 操作 | 结果 |
|---|---|
| 暂停保持 | 暂停输入并请求保持；保留会话供恢复 |
| 恢复遥操 | 与开始共用机型操作入口，要求智能控制仍已开启及当前 PICO 连接有效；按当前反馈重新准备，之后满足双握把条件 |
| PICO／卡片“结束并收臂” | ActuCore 先停止输入，再由 Driver motion_control 按实测反馈、碰撞与速度约束返回厂商模型双臂零位（本轮自然下垂目标，14 关节 q=0）；确认到位及停止后释放控制权。智能控制仍可保持开启，下一轮需显式点击开始 |
| Canvas 关闭智能控制 | 先禁止新开始，再复用同一收臂流程；不要求 PICO 在线。收臂与控制权释放确认后，Core 才停止其余卡片并标记项目已停止 |
| 立即停止 | 请求停止并释放控制权，不主动收臂；网络失败或停止未确认时，页面会保留未知或错误状态 |

PICO 或卡片发起的 `finish` 要求当前配对头显连接仍有效；头显已离线时可从 Canvas 关闭智能控制完成收臂。新链路的收臂规划、下发与确认由 Driver 持有，ActuCore 只请求 finish 并查询 finish_status。收臂目标不是上次遥操停止姿态，也不是本轮接管时的任意姿态。Shadow 或从未准备遥操的会话结束不发送真实回零动作；重复结束不会重新执行已完成的收臂。

断线、碰撞、反馈异常或停止未确认会保留具体错误，不显示为“已收臂”。Canvas 停止失败会保留项目和 Driver 反馈通路，原因解除后显式重试关闭智能控制；PICO 结束失败可在连接恢复后重试结束。立即停止可取消正在进行的收臂，但不会代替正常收臂。收臂逐步检查已有碰撞边界，不是绕障规划器；服务重启、部署或单纯断开 PICO 不会隐式回零。

天轶 PICO 的结束与停止可在握把按住时中止当前输入。G1 目前没有接入这套头显操作按钮，不应据此判断其配对失败。

## 如何看状态与头显模型

卡片区分三件事：PICO 是否在线、会话是否正在处理输入、Driver 是否实际输出。反馈过期时显示“执行状态未知”，不能用“已连接”或“遥操中”代替真实执行反馈。

天轶头显中的显示含义：

- 绿色线：新鲜关节实测值计算的双臂位置。
- 黄／橙色线：完整 IK 结果；它不是已经执行到位的证明。
- 粉色目标：相对映射后的末端目标。
- 带 `HELD` 标记的橙色模型：最后有效解的历史快照，不是当前有效指令。
- 白色躯干参考：辨认朝向；外框表示标定的活动边界，框内并非所有姿态都可达。

越界或短暂 IK 失败时，失败目标不下发，保持靠近最后有效执行位置。天轶在保持已确认、输入和反馈恢复有效后可继续，不做额外的越界目标搜索。普通松握把后重握会重新建立相对基准；真正故障的恢复仍需满足 Driver 条件。

G1 与天轶的模型点数、历史显示及恢复实现有差异，不能将天轶显示和恢复验收直接套用到 G1。

## 机型支持边界

| 能力 | 天轶 2.0 | G1_23 |
|---|---|---|
| 双臂 IK | 新链路在 Driver，每臂 7 关节 | 保留 ActuCore 旧链路，每臂 5 关节 |
| 配对、RTC 输入、Canvas 管理 | 已实现 | 已实现 |
| Driver 连线与智能控制启停 | 三卡连线与 Shadow 智能控制启停已实测；Live 跟随及关闭收臂仍待验收 | 未支持；不声明 `project_start/project_stop` |
| PICO 内开始／结束／停止按钮 | 已实现 | 尚未接入 |
| 结束并收臂 | 已实现，仅已标定双臂模式 | 尚未实现；不声明或显示此操作 |
| 手部 | 新链路只接收双臂，手部不在本轮范围；旧开合能力保留 | 固定假手，不输出手部命令 |
| 普通 Driver 只读 Shadow | 不提供此适配 | 支持 `driver_joints`，没有执行权 |
| 录制管理入口 | 10 秒 Shadow 录制 | 尚未接入 |

北京 G1 可沿用独立 Shadow 路径展示卡片、输入和 IK，不经过上述智能控制收臂流程；将 G1 遥操卡片放入该项目生命周期会明确拒绝开启，不会伪装支持收臂。IK 仍需要带 `pinocchio.casadi` 的 G1 数值环境，不能直接复用天轶依赖锁。若要真实运动，还需要其配套执行 Driver、模型／碰撞标定和独立验收；仅修改 `robot_profile` 不足以完成适配。

G1 的七个原始碰撞网格共 9,876,188 字节，以厂商固定提交为来源，保存在项目 COS，不存入 Git。源码开发/数值测试前执行 `python3 deploy/fetch_g1_collision.py`；用 `--check` 可以禁网核验本地缓存。遥操镜像构建会显式取得相同文件并验证 `models/g1_collision/sha256.json`，之后无需运行时网络。下载失败或哈希不符会使构建失败；运行时缺网格会拒绝初始化，不能关闭碰撞检查来绕过。网格来源与许可证见 [NOTICE.md](NOTICE.md)。

## 部署与接口

生产结构是一套 ActuCore：VLA 与 `teleop` 共用进程、ROS executor 和 MCP **15730**。PICO WSS **15741** 是插件内部的输入与显示服务，不是第二个 ActuCore。旧独立 `actucore-teleop` 与 15740 现场布局不再作为新部署方案。

JetPack 6.1 的普通 ActuCore 镜像自动包含遥操依赖，沿用 `deploy/build_actucore.sh --jp-version 6.1`；不要求特殊遥操构建开关或更新 bot。卡片仍默认关闭。该发布脚本在配置远端仓库凭据时会推送镜像；仅本地构建须使用项目约定的 Dockerfile 构建方式。构建、挂载和配置恢复约束见 [ActuCore README](../../README.md#遥操与-vla-共用-actucore)。共享依赖锁是 `requirements.bundle.lock`；CPU 隔离验证镜像不构成另一套生产服务，JP5.1.1 不声明遥操数值支持。

将 [配置示例](config.example.yaml) 的 `plugins.teleop` 合入现有配置，保留其他插件。新路径把控制模型/标定挂到 Driver；ActuCore 仍须提供 TLS、配对状态、管理密钥与 DDS 文件的挂载；标准 service 片段尚未自动生成全部遥操挂载，不能只打开 `enabled` 就宣称部署完成。`/deploy/dds-local.xml` 必须和宿主挂载的 DDS profile 一致。Core 的 `TELEOP_MANAGEMENT_URL` 指向共享的本机 15730 MCP，管理凭据只由 Core／ActuCore 持有。

ActuCore 启动时检查站点资料。缺少或无效时，不注册遥操工具，VLA 仍正常注册；`tools/list` 的 `_meta.required_site_config` 及直接 `teleop` 的 `info` 返回缺失字段和错误码，不暴露路径、密钥或证书内容。检查包含管理密钥、WSS 证书/域名/私钥、可写状态目录和必要依赖；legacy 另检查本地标定模型哈希。motion_control 后端缺少 Driver 模型时保留安装、配对和配置入口，Driver 拒绝标定/执行，不将其冒充就绪。Core 另行核对管理 URL 与共享密钥。补齐站点挂载后需要重启 ActuCore 重新注册；已就绪卡片的日常参数、配对和启停仍通过 Canvas/PICO 完成。站点检查不等于模型或停止行为的实物验收。

运行异常只返回已知的协议与恢复错误码；底层文件、超时或未知异常分别归为 `teleop_io_error`、`teleop_timeout`、`teleop_not_ready`，不把路径、URL 或凭据带到页面和头显。本地日志保留异常类别与 errno。关闭时即使录制或运行时清理失败，仍尝试独立释放 Driver；释放未确认优先报告 `stop_unconfirmed`，其他清理失败保留资源供重试，不报告配置保存成功。 Runtime 的负向关闭回执同样视为失败；仅旧执行线程完全退出后重试停止和清理，不重放目标或重建运动会话。

低频 MCP 操作包括 `info/config/project_start/project_stop/start/pause/resume/finish/stop`、配对、标定、自检及录制，按机型声明支持项。Core 将唯一连线解析出的 `driver_binding` 传给 `project_start`；它只设置内存权限，不直接调用 Driver 动作。`project_stop` 等待收臂及控制权释放，不将异步受理回执当作完成。

新链路通过本机 ROS domain 42：`/{namespace}/motion/control/command` 发送末端目标，Driver 解算后向 `/{namespace}/motion/arm/command` 发关节目标；执行反馈复用 `/{namespace}/motion/teleop/feedback`，teleop 状态仍为 `/{namespace}/teleop/status`。Core 不逐帧转发。

两段命令使用 `motus.control/2`，携带启动/会话、序号、原输入序号、映射代次、单调时钟期限、模型/标定版本及 HMAC。末端是左右 xyz 米 + xyzw，关节是 descriptor 顺序的 rad。末端目标最多 300 ms，关节目标最多 100 ms，且不能超过原输入期限；解算失败或过期不重发最后失败结果。详细字段见 [架构接口](../../../docs/design/teleop-architecture.md#绑定与接口)。旧 `control/teleop` 及 `motus.control/1` 保留。

普通镜像根据 [APK 清单](../../openxr_capture_native/package-manifest.json) 下载确定性 gzip 构建制品，验证外层及原 APK 的大小/SHA256 后解压内置；Canvas 用户下载原始 APK，无需处理 gzip。APK 与签名私钥不提交 Git，构建入口无需特殊 flag；完整客户端构建与许可说明见 [原生客户端](../../openxr_capture_native/README.md)。

天轶录制入口 `record_start/record_stop/record_status` 仅在 Shadow 且无租约时使用。`record_start` 等待有效双握把输入后录制 10 秒，记录输入、反馈和版本信息，不录入配对凭据；当前 Canvas 未提供录制按钮。离线分析和回放接口位于 [acceptance.py](acceptance.py)，真实回放必须与纯分析区分。

## 验证与开发资料

本地检查示例（仓库根目录，使用已准备的依赖环境）：

```bash
python -m pytest agent-core/tests/test_teleop_panel_metadata.py -q
python -m pytest agent-core/tests/test_teleop_project_lifecycle.py -q
python -m pytest actucore/tests/test_operator_session.py actucore/tests/test_teleop_configuration_restart.py -q
python -m pytest actucore/tests/test_teleop_project_chain.py -q
python -m pytest actucore/tests/test_motion_control_adapter.py actucore/tests/test_onboarding.py -q
# 设置 TIANYI_DRIVER_SOURCE 为配套 Driver 的 x-humanoid/tianyi2.0 目录后：
python -m pytest actucore/tests/test_motion_control_chain.py -q
node agent-core/tests/teleop-panel.browser.cjs
```

分别记录代码/离线测试、APK、目标架构镜像及部署/真机验收。浏览器检查使用本地 fixture，不等于 PICO 系统下载、侧载或深链接实测；`test_tianyi_execution_chain.py` 需要将 `TIANYI_DRIVER_SOURCE` 指向配套 Driver 源码。内存替身、隔离 ROS、镜像启动及实机动作是不同证据等级，不能相互替代。跨仓新链测试使用真实 IK、共享门禁和有限速度 plant，MCP/DDS 为内存替身。现场/回放仍应关联输入与求解、Driver 决策、本体 cmd_pos、实测 arm/status 四段记录。跟随误差与延迟如实报告，不另设精度通过门槛。

架构边界与后续问题见 [遥操架构说明](../../../docs/design/teleop-architecture.md)。现场证据和一次性迁移记录单独保存，不构成通用安装流程。模型与客户端来源、许可证见 [NOTICE.md](NOTICE.md) 和 [ADOPTED_SOURCE.json](ADOPTED_SOURCE.json)。
