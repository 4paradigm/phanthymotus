# Nav2 action 响应竞态修复

## 范围与根因

只修复本仓 ActuCore Nav2 action 客户端的请求登记与响应读取竞态。
旧 rclpy 在发送请求后才登记 Future，多线程 executor 可先消费响应并永久丢弃。
现场日志出现 unexpected goal response，目标已完成但桥接状态仍为 starting，
速度输入停止后按原安全规则输出 shadow_velocity_stale。
同一已部署镜像在本地隔离 Docker 中注入发送后延迟，复现了警告和永不完成的 Future。

## 实施

- 为目标、取消、结果请求登记和 take_data 使用同一短临界区。
- execute、用户回调、等待 Future 不持锁，保留并发回调能力。
- 不修改 topic/schema、5 Hz、250 ms TTL、超时与停车语义。
- 使用真实 ROS action server/client 在无网络隔离容器中覆盖提前响应、拒绝、取消和终态。
- 执行 ActuCore 回归并复核 README；用户追加授权提交并部署北京 G1；构建本地 ARM64 镜像，核验空闲后只切换 ActuCore，不触发 BOT review。

## 验证

- 旧部署镜像注入发送后 300 ms 延迟：复现 unexpected goal response，Future 未完成。
- 同镜像挂载本地修复，真实 ROS action 测试 5/5 通过：目标提前响应、结果提前响应、
  取消提前响应与 cancelled 终态、拒绝目标、慢响应保持 pending 后正常完成。
- 目标响应回调内继续请求结果并收取 succeeded 终态，验证回调重入无死锁。
- ActuCore 回归：437 passed、7 skipped、72 subtests passed。
  5 项 ROS 测试因宿主无 ROS 跳过，已在上述容器中实际通过；其余为 OpenCV 与 Linux procfs。
- README 已补充响应竞态与 stale 诊断。公开接口及部署参数不变，无需修改接口/部署说明。
- 修复阶段未提交或部署；后续按用户授权提交、构建、推送和部署，现场验收由用户完成。
- 部署沿用现有 Compose，只替换 actucore 镜像；保留旧镜像与配置备份。切换前后核对 Canvas 停止、自动启动关闭、无编辑者/部署任务、导航空闲及 Driver 控制状态。

隔离 ROS 测试（仓库根目录运行；使用已存在的本地部署镜像）：

```sh
docker run --rm --network none -e ROS_DOMAIN_ID=223 \
  -v "$PWD/actucore:/test/actucore:ro" --entrypoint python3 \
  local/phanthy-motus/actucore:release.260920.f7a6ded-jetson-jp5.11 \
  /test/actucore/tests/test_nav2_action_client_ros.py -v
```

宿主回归：在 actucore 目录运行 `../agent-core/.venv/bin/python -m pytest tests -q -rs`。

