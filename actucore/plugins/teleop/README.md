# PICO 遥操卡片

`teleop` 是默认关闭、启用后默认 Shadow 的 ActuCore processor 卡片。卡片管理 PICO 配对、WSS/RTC 输入、相对映射和 IK；天轶 Driver 接收机器人目标、执行并返回实测反馈。Shadow 不获取硬件执行权。

本 PR 包含 ActuCore、原生 PICO 客户端及诊断/回放工具，不包含 Agent Core 代理与 Canvas 专用面板。`x-connection-panel: teleop-v1` 元数据不表示未经配套更新的 Core 已支持该面板。管理请求需要回环地址及 `TELEOP_MANAGEMENT_KEY_FILE` 对应密钥；不要向浏览器暴露密钥。

## 配置与行为

- 天轶采用 `robot_profile=tianyi2`、14 关节双臂、胸部固定坐标系。每次重新握持以控制器及机器人实测姿态重建相对基准；必须同时按住双握把才跟随。
- `hands_enabled: false` 为双臂模式，忽略扳机且不下发手部命令。手部模式需要单独标定开合端点，尚未在本次完成验收。
- CPU Pinocchio/SciPy IK 使用实测初值，验证模型哈希、机械限位、目标及运动段碰撞；失败结果不执行。不使用额外搜索将不可达目标替换为另一目标。
- 最新目标经同机 ROS domain 42 传输；低频管理走 MCP。目标带租约身份、序号、单调时间和签名。输入预算与命令 TTL 分开，不能用头显时钟判断机器人指令是否过期。
- 短暂 IK 失败进入可恢复保持。反馈新鲜、保持确认且新目标有效后同会话继续；失败期间不重放旧目标。松开重握重建基准；租约失效等走完整恢复。
- 显式开始建立持续操作会话，握把只控制是否跟随。立即停止保持并释放；只有显式结束并收臂才执行回零路径。收臂可取消，断连/故障进入保持，不以超时伪造已停止。
- 速度由标定与模型共同限制；天轶位置接口上限 1.5 rad/s，默认 0.2 rad/s，现场回放采用 1 rad/s。跟随误差如实报告，不另加准确率门槛。

配置模板见 [config.example.yaml](config.example.yaml) 与 [calibration.example.json](calibration.example.json)。模板未实物验收，不可直接作为 Live 标定。两仓须使用相同标定及兼容执行协议。来源和许可证见 [NOTICE.md](NOTICE.md)，迁入时来源哈希见 [ADOPTED_SOURCE.json](ADOPTED_SOURCE.json)。

## PICO 显示与操作

客户端在 `actucore/openxr_capture_native`。透视内绿色为新鲜实测 FK，橙色为 IK 姿态，粉色为映射目标；HELD/STALE 明确表示历史快照。模型采用平视背后视角及躯干参考，不是相机测量，也没有与现实机器人配准。

显示以独立最新快照推送，不阻塞 IK；历史模型常亮不代表硬件正在运动。原生页面提供开始、结束并收臂和立即停止，经已配对 WSS 异步执行并去重；停止具有优先级。

## 构建、测试与回放

从仓根执行 `bash deploy/build_tianyi_actucore.sh LOCAL_IMAGE_TAG`。CPU ARM64 构建锁定数值和通信依赖，不要求 GPU。G1 兼容模块另需其锁定的 CasADi/Pinocchio ABI 环境。

```sh
export TIANYI_DRIVER_SOURCE=/absolute/path/to/phanthymotus-driver/x-humanoid/tianyi2.0
python -m pytest -q actucore/tests
sh actucore/openxr_capture_native/tests/run_host_tests.sh
```

`deploy/test_tianyi_teleop_isolated.sh` 使用真实 ROS/MCP 但合成本体反馈，不是硬件验收。原始现场录制、密钥、租约 journal 和设备配置不提交。

回放工具支持录制、离线 IK、独立关节执行与同源实时 IK 对照。`deploy/run_tianyi_chain.py` 默认预检，`--execute` 才请求真实运动；脚本包含现场路径与录制 ID，使用前须核对本机实际配置，不能作为通用即装即用入口。四段证据覆盖输入/IK、Driver 接收、独立本体命令和实测关节；重复命令的匹配可能有歧义。

本次 PR 的范围、测试失败和验收边界见 [交付记录](../../../docs/plans/tianyi-teleop-pr.md)。完整连续遥操、实体结束/停止按钮、15 分钟及 10 轮恢复仍需验收；已有运动或一次收臂成功不代替这些项目。
