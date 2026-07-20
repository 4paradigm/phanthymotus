# ROS 插件生命周期稳定性设计

## 背景

OCR 榜单在连续完成 58 个 case 后，`perception_spin` 线程因
`rclpy.InvalidHandle: cannot use Destroyable because destruction was requested`
退出。MCP HTTP 服务仍然存活并继续返回 `running`，导致后续 case 全部等待
120 秒超时。根因是 MCP 请求线程在 executor 仍处于 `spin()` 时销毁
subscription 或 Node。

同类生命周期操作也存在于 ASR、TTS、VOP 和 HTMSG。此次修复覆盖全部 ROS
插件，不改变它们的 MCP 输入输出协议、topic 命名和模型实现。

## 目标

- 连续至少 250 次 `start -> process -> stop` 不销毁正在被 executor 使用的句柄。
- 普通 `stop/start` 复用 ROS Node，不重复创建 DDS 实体。
- 并发 `start/stop/config` 不产生重复 Node、丢失 Node 或字典竞态。
- 已停止或已被新一代 worker 取代的后台任务不得发布旧结果。
- `perception_spin` 异常必须使服务停止并以非零状态退出，禁止假活。
- 进程关闭时先停止 worker/子进程，再停止 executor，最后销毁 Node。

## 非目标

- 不调整 ASR、TTS、OCR、VOP 的模型、阈值、推理设备或输出结构。
- 不引入新的 ROS executor 或改变现有 `MultiThreadedExecutor`。
- 不在正常请求路径中重启整个 perception 容器。

## 方案选择

采用 Node 常驻复用方案。普通停止只改变业务状态并停止后台任务；Node、publisher
和 subscription 保持注册。输入回调在 idle 状态直接返回。只有同一实例的 topic
发生变化时，旧 Node 才从 executor 移除并进入 retired 集合，但在 executor 停止前
不调用 `destroy_node()`。

未采用“把全部生命周期命令投递到 spin 线程”的方案，因为需要额外控制队列、
guard condition 和响应同步，改动面过大。未采用“延迟若干毫秒后销毁”，因为它
不能证明 wait-set 已释放句柄。

## 生命周期模型

### 插件级串行化

ASR、TTS、OCR、VOP、HTMSG 每个插件持有一个 `threading.RLock`。`start`、
`stop`、`config`、模型替换和 shutdown 状态变更均在该锁内完成。耗时推理不持有
生命周期锁。

### Node 正常启停

- 首次 `start`：创建 Node、加入 executor、创建 worker。
- 重复 `start`：同一实例和 topic 复用 Node；已运行时幂等返回。
- `stop`：先将状态切到 idle/stopping，再通知当前 worker，等待其退出；不调用
  `destroy_subscription()`、`remove_node()` 或 `destroy_node()`。
- 再次 `start`：创建新的 stop event 和 generation，启动新 worker。
- idle 状态的 subscription 回调立即返回，不缓存输入。

### 配置和 topic 变化

语言、阈值、adapter 等不改变 ROS 拓扑的配置直接更新现有 idle Node；运行中的
Node 先正常停止，再更新，保持当前“配置后需重新 start”的行为。

同一 instance 的输入 topic 变化时，旧 Node 正常停止并从 executor 移除，保存到
retired 集合；新 Node 使用新 topic。retired Node 只在 executor 完全停止后销毁。

### Worker 代次

每次 `start` 创建独立的 stop event 并递增 generation。worker 在每次发布前同时
检查本地 stop event 和 generation。即使一次推理超过 join timeout，旧 worker
也不能在下一次 start 清除停止信号后恢复，更不能发布到下一 case。

ASR 和 VOP 在阻塞推理返回后增加发布前检查；TTS 在每个音频帧发布前检查；OCR
保留现有推理后检查并改为独立 stop event。

## 全局故障传播

HTTP server 在 spin 线程之前创建。spin 包装函数捕获任何未处理异常，记录完整
堆栈并调用 `server.shutdown()`。主线程离开 `serve_forever()` 后完成有序清理，
随后重新抛出 spin 异常，使容器非零退出，由部署平台重启或明确标记失败。

MCP 不会在 spin 已死亡时继续提供假 `running` 状态。

## 有序关停

`PerceptionBundle` 提供两个阶段：

1. `prepare_shutdown()`：调用每个插件的停止方法，终止 worker、VAD/HTMSG 子进程，
   但不销毁 ROS Node。
2. executor `shutdown()` 完成后调用 `destroy_nodes()`：销毁 active 和 retired Node，
   清空引用；最后调用 `rclpy.shutdown()`。

单个插件清理失败只记录异常，其他插件仍继续清理。

## 插件适配

- **OCR**：删除普通 stop 中的 subscription/Node 销毁和请求时 retired reap；Node
  在相同 topic 上复用，adapter/language 可更新。
- **ASR**：subscription 常驻；每次 start 重建 VAD 队列和子进程；停止后的转写
  结果不得发布。
- **TTS**：Node 常驻；配置更新 adapter；旧合成 worker 使用独立停止信号。
- **VOP**：Node 常驻；推理后检查 generation；模型加载和 Node 创建纳入同一
  生命周期锁，防止并发双创建。
- **HTMSG**：主 Node 常驻；stop 仅停止 odometry pipeline；重新 start 复用 Node；
  最终关停才销毁。

## 错误处理

- worker 超过 join timeout：记录 warning，保留 Node 引用；generation 阻止旧输出。
- topic 变化后的旧 worker 尚未退出：Node 进入 retired 集合，最终关停统一销毁。
- spin 异常：关闭 MCP server，完成清理并非零退出。
- 插件 shutdown 异常：记录插件名称和堆栈，继续清理其余插件。

## 测试

- OCR 连续 250 次 start/stop 只创建一个 Node，普通 stop 不 remove/destroy。
- ASR、TTS、VOP 的普通 stop 复用 Node，并验证 concurrent start 只创建一个 Node。
- worker stop 后立即 restart 时，旧 generation 的 OCR/ASR/VOP/TTS 结果不发布。
- instance topic 变化时旧 Node 被 remove 但直到 `destroy_nodes()` 才销毁。
- HTMSG stop/start 复用 Node，odometry 子进程按次启停。
- spin executor 抛异常时 server 被 shutdown，主流程返回非零失败。
- shutdown 顺序为 worker/子进程、executor、Node、rclpy。
- 运行全部 `perception/tests`，并在 Jetson 上执行真实 rclpy 250 次启停压力测试。

## 验收标准

- 本地全部单元测试通过，无新增大于 1 MB 文件。
- Jetson 镜像能够构建并启动。
- 连续 250 次 OCR case 不出现 `InvalidHandle`、spin 线程退出或 120 秒连续超时。
- 正常停止后的旧 worker 不产生跨 case 输出。
