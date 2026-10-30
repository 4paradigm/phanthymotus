# 通用人物视觉跟随

输入是 Perception 的 VOP 检测，输出是已协商的实际头部/云台关节位置。
策略不包含机器人名称、地址、SDK、模型或图像库。TRON2 是首个配套驱动适配，
换机器只需驱动声明并实现下述头部接口，不复制跟随策略。

## 使用

画布：`机器人头部相机 → VOP → headfollow → 机器人头部执行卡片`。
最后一路为标准 `control/joint`，JSON 遵循既有 `motus.control/1`。Agent Core
从接线下游的 `info.control_interface` 取得实际动作空间并传给本卡片的 `start`。
反馈话题在接口中声明，由插件自动订阅，不需要画一条循环依赖连线。

启动容器/项目只接线、订阅和协商，不会自动转头。执行默认 `follow` 后优先锁定靠近
中心的人，但也可从边缘获得候选；至少两个不同的新鲜结果确认。没有 pause/status 表单。
内部 start/stop/info 供生命周期使用；停止只阻止后续目标，不承诺中途制动已受理动作。
初次未见人或短暂丢失时保持，连续至少3个有效空结果且持续1.5秒后回到驱动声明的 neutral。
断流、错误、过期、相同时间戳的重复包都不作为无人证据。多人交叉或候选相近时保持关联，
不作人脸身份识别。头部任务未知、另一控制者介入、驱动重启或反馈异常时停止，不重试/回正。

头部只接受实际存在的 yaw 和/或 pitch，一或二轴；轴索引、正方向、限位、速度、回正点
以下游声明为准。没有可动头部/云台的固定相机不能用本卡片代替，尤其不能自动转动底盘。
支持通用接口不等于所有品牌均已实机验证。当前测试覆盖TRON2声明、另一组轴顺序/方向/
neutral声明和单轴云台；只有TRON2有实际驱动适配，其余是假接口契约测试。

## VOP 输入

订阅声明中的 `camera_topic + '/objects'`，拒绝其它相机误接。内容为：

```json
{"timestamp":1790000000.0,"count":1,"latency_ms":30,
 "objects":[{"name":"person","position":[0.3,-0.2],"confidence":0.9}]}
```

旧版可省略count（从objects推导），显式错误/不可用消息不算无人。position 是相对于图像中心的−1～1坐标（x向右、y向下）。bbox可有可无，不依赖图像尺寸。
有效人物类别为 person/human/人，置信度≥0.55。timestamp 必须推进且足够新鲜；有 latency_ms
时同时检查处理起始时间。VOP目前未携带硬件帧ID/真实采样时间，重复采集同一冻结图像而重新
打处理时间戳的情况不能靠这份结果单独识别；info明确报告计时依据和measurement_time_verified=false。
不将发布年龄冒充硬件图像新鲜度，不改造或自动升级Perception来掩盖此限制。

## 驱动声明

完整示例见 `tests/fixtures/head_tron2.json`。标准 descriptor 外的扩展是：

```json
"head_tracking": {
  "schema":"motus.head-tracking/1",
  "axes":{"pitch":{"index":0,"sign":-1,"image_axis":1,"gain_rad":0.22},"yaw":{"index":1,"sign":1,"image_axis":0,"gain_rad":0.26}},
  "neutral":[0,0], "max_duration_s":15,
  "camera_topic":"/robot/head_camera", "feedback_topic":"/robot/head_state"
}
```

标准字段需声明 joint_position、rad、一或二轴名称、每轴 lower/upper/max_velocity、
rate.max_hz/max_obs_age_ms 和 force_torque（如无感知须明确null）。sign=+1代表该轴数值增加
会让**与该接口配套的相机**向上（pitch）或左（yaw）转；−1相反。image_axis指定该轴响应VOP的x(0)还是y(1)，gain_rad是每单位归一化误差对应的保守角度增益（0～0.5rad），由相机安装/视场与执行端调定；策略不假定各机器人相机内参相同。不能只凭关节名猜轴方向。
neutral必须在工作限位内。声明应只覆盖实际头部，不能拿两轴机械臂假冒。

## 反馈与执行

下游按声明的 feedback_topic 发布 `std_msgs/String` JSON：

```json
{"schema":"motus.head-state/1","session":"driver-process-id","timestamp":1790000000.0,
 "sample_stamp":12345,"available":true,"stream_enabled":true,"ready":true,"stationary":true,
 "values":[0,0],"velocities":[0,0],"motion_id":null,"request":null,"completion_verified":false}
```

- values/velocities依descriptor顺序；sample_stamp为持续推进的原始反馈标记，不假定时间单位。
  每包timestamp为主机发布时间，只有反馈本身新鲜完整才available=true。session在驱动重启后改变。
- ready需包含原机器人模式、互斥和失败保护，不能只写“网络已连接”。stationary需要实际反馈。
- 本卡片每次只发一个目标，不是连续速度伺服，也不会在转动途中改目标。除标准控制字段外，
  消息带 duration_s、head_session、reference（反馈标记/实际关节值/motion_id）。驱动须在
  发送前核对TTL、会话、参考姿态、动作所有权、速度和原有保护；过期或来源改变拒绝。
- 接收后反馈 request={source,seq,outcome,action_id}。outcome为pending/accepted/completed/
  rejected/failed/unknown。只有实际到位并完成原有回调后才能completed且completion_verified=true。
  静止/受理不等于成功。未知或失败不能清除旧保护，也不能由上游重发。
- 目标发布前记录pending；发布失败即保留未知结果并停止。已完成后还要等之后的新鲜VOP结果，
  才计算下一目标。连续静止新反馈用于初始准入；之后可复用驱动仍有效的到位证明。
- 关闭数据流应使旧代次尚未发送的请求失效；已受理的动作按原驱动约定完成。不要自动复位。

协议扩展属于执行插件与驱动的配套接口，不要求修改Agent Core、Perception或公共控制协议。

## 构建和验证

随既有ActuCore镜像的`COPY actucore/plugins/`打包，没有新Dockerfile、pip/apt依赖、模型下载
或TRON2专属运行服务。需要配套机器人驱动；仅更新插件不能给固定相机凭空增加执行器。

```bash
PYTHONPATH=perception/utils python3 -m pytest -q actucore/tests
```

测试不连接实体硬件。实际ARM64镜像构建、旧Perception版本输出、DDS连通性和实体跟随/回正
仍须现场验收。VOP可作为后续视觉抓取的识别入口，但二维位置本身不是可执行抓取位姿；深度、
相机标定/坐标变换、可达性、碰撞检查和抓取成功确认仍是独立环节。
