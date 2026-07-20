# OCR ROS 生命周期稳定性设计

## 背景

OCR 榜单连续完成 58 个 case 后，`perception_spin` 线程因
`rclpy.InvalidHandle: cannot use Destroyable because destruction was requested`
退出。MCP HTTP 服务仍继续返回 `running`，后续 case 全部等待 120 秒超时。

根因是 OCR 的 MCP `stop` 请求在 executor 仍处于 `spin()` 时销毁 subscription
和 Node。本阶段只修复 OCR 及全局 spin 故障传播；ASR、TTS、VOP、HTMSG 暂不
改动。

## 目标

- 连续至少 250 次 OCR `start -> process -> stop` 不销毁 executor 正在使用的句柄。
- 相同实例和 topic 的普通 `stop/start` 复用同一个 OCR Node。
- 并发 OCR `start/stop/config` 不产生重复 Node 或字典竞态。
- 已停止或被新一代 worker 取代的任务不得发布旧结果。
- `perception_spin` 异常时停止 MCP 服务并使容器非零退出，禁止假活。
- 进程关闭时先停止 OCR worker，再停止 executor，最后销毁 OCR Node。

## 非目标

- 不修改 ASR、TTS、VOP、HTMSG 的生命周期代码。
- 不调整 OCR 模型、阈值、推理设备、输出结构或 topic 命名。
- 不改变现有 `MultiThreadedExecutor` 类型。

## OCR 生命周期

### 串行化

`OCRPlugin` 增加 `threading.RLock`。`start`、`stop`、`config` 和 shutdown 状态
变化均在锁内完成，耗时 OCR 推理不持有该锁。

### 正常启停

- 首次 `start` 创建 Node、加入 executor、创建 subscription 和 worker。
- 同一实例和 topic 再次 `start` 复用 Node；运行中重复 start 幂等返回。
- `stop` 先切换状态并停止 worker，但不调用 `destroy_subscription()`、
  `remove_node()` 或 `destroy_node()`。
- idle 状态的 subscription 回调立即返回，不缓存输入。
- 再次 start 创建独立 stop event 和新的 generation，然后启动新 worker。

### 配置与 topic 变化

语言或 adapter 配置变化时，运行中的 Node 先正常停止，再更新 Node 的 adapter、
language 和限帧参数，保持“配置后需重新 start”的现有行为。

同一 instance 的 input topic 变化时，旧 Node 正常停止并从 executor 移除，保存到
retired 集合；新 Node 使用新 topic。retired Node 在 executor 停止前绝不销毁。

### Worker 代次

每次 start 创建独立 stop event 并递增 generation。worker 在发布前同时检查本地
stop event 和 generation。即使一次推理超过 join timeout，旧 worker 也不能在
下一次 start 后恢复或向下一 case 发布结果。

worker 超过 join timeout 时记录 warning 并保留 Node 引用，不在请求线程销毁它。

## Spin 故障传播

HTTP server 在 spin 线程之前创建。spin 包装函数捕获未处理异常，保存异常并调用
`server.shutdown()`。主线程离开 `serve_forever()` 后执行有序清理，随后重新抛出
spin 异常，使容器非零退出。

这样平台会明确看到失败并执行重启或终止，而不是让 MCP 继续返回假 `running`。

## 有序关停

`PerceptionBundle` 增加两阶段关停：

1. `prepare_shutdown()` 调用 OCR 插件，停止 active/retired worker，但不销毁 Node。
2. executor `shutdown()` 后调用 `destroy_nodes()`，销毁 active/retired OCR Node并清空
   引用，最后调用 `rclpy.shutdown()`。

其他插件在本阶段保持现状。调用可选关停方法时，单个插件异常只记录日志，不阻断
其余清理。

## 测试

- 连续 250 次 OCR start/stop 只创建一个 Node，普通 stop 不 remove/destroy。
- 并发 start 只创建一个 OCR Node。
- idle 状态 callback 不接收图片。
- stop 后立即 restart 时，旧 generation 结果不发布。
- instance topic 变化时旧 Node 被 remove，但直到 `destroy_nodes()` 才销毁。
- adapter/language 配置更新复用相同 topic 的 Node。
- spin 抛异常时 HTTP server 被 shutdown，异常传回主线程。
- 验证关停顺序为 OCR worker、executor、OCR Node、rclpy。
- 运行全部 `perception/tests`；Jetson 上执行真实 rclpy 250 次启停压力测试。

## 验收标准

- 本地全部测试通过，无新增大于 1 MB 文件。
- Jetson 镜像能够构建并启动。
- 250 次 OCR case 不出现 `InvalidHandle`、spin 线程假活或连续 120 秒超时。
- 正常 stop 后没有跨 case OCR 输出。
