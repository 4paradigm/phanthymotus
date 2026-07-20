# OCR ROS 生命周期稳定性实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 修复 OCR 连续启停导致 rclpy executor 线程崩溃，并让 spin 异常触发容器失败而不是假活。

**架构：** OCR Node 在相同实例和 topic 上常驻 executor，普通 stop 只停止当前 generation 的 worker。topic 改变时旧 Node 只从 executor 移除并保留到最终关停；主流程通过 spin guard 将线程异常传回主线程，并按 worker、executor、Node、rclpy 的顺序清理。

**技术栈：** Python 3.10、rclpy、`threading`、`unittest`、ROS2 `MultiThreadedExecutor`

---

## 文件职责

- 修改 `perception/plugins/ocr.py`：OCR Node generation、常驻复用、生命周期锁、retired Node 和两阶段关停。
- 修改 `perception/main.py`：bundle 两阶段关停、spin 异常传播和主流程清理顺序。
- 修改 `perception/tests/test_ocr_contract.py`：OCR 连续启停、并发、旧 worker 和 topic 变化回归测试。
- 创建 `perception/tests/test_main_lifecycle.py`：spin guard 和 bundle 关停顺序测试。

### 任务 1：锁定 OCR 正常启停契约

**文件：**
- 修改：`perception/tests/test_ocr_contract.py`

- [ ] **步骤 1：编写普通 stop 复用 Node 的失败测试**

构造 mock `_OCRNode`，连续调用 250 次 `start/stop`，断言 `_OCRNode` 和
`executor.add_node` 各调用一次，`executor.remove_node` 与 `destroy_node` 均未调用。

- [ ] **步骤 2：编写并发 start 的失败测试**

使用 `ThreadPoolExecutor(max_workers=2)` 同时启动相同 instance/topic，通过阻塞
Node 构造函数放大竞态，断言只创建一个 Node。

- [ ] **步骤 3：运行测试确认失败**

运行：

```bash
python3 -m unittest \
  perception.tests.test_ocr_contract.OCRContractTest.test_repeated_start_stop_reuses_one_ros_node \
  perception.tests.test_ocr_contract.OCRContractTest.test_concurrent_starts_create_one_ocr_node
```

预期：FAIL，当前实现每次 stop 都 remove/destroy，且没有 lifecycle lock。

### 任务 2：实现 OCR Node 常驻和 generation

**文件：**
- 修改：`perception/plugins/ocr.py`
- 修改：`perception/tests/test_ocr_contract.py`

- [ ] **步骤 1：实现 lifecycle lock 和普通 stop 复用**

为 `OCRPlugin` 增加 `threading.RLock`，`dispatch()` 在锁内执行。普通 stop 调用
`node.stop()` 但保留 `_nodes` 和 executor 注册。`_OCRNode.stop()` 不销毁
subscription；`_image_cb()` 在非 running 状态立即返回。

- [ ] **步骤 2：运行任务 1 测试确认通过**

运行任务 1 的两项测试，预期 PASS。

- [ ] **步骤 3：编写旧 generation 禁止发布的失败测试**

启动 worker A，让 adapter 推理阻塞；stop A 后立即 start worker B；释放 A，断言 A
不调用 publisher，B 仍可正常工作。测试必须证明每次 start 使用不同 stop event。

- [ ] **步骤 4：实现独立 stop event 和 generation**

每次 start 新建 `threading.Event` 并递增 `_generation`，worker 接收二者的快照；
发布前调用 `_is_generation_active(generation, stop_event)`。join 超时记录 warning，
不清除旧 event，不销毁 Node。

- [ ] **步骤 5：运行 OCR contract 测试**

```bash
python3 -m unittest perception.tests.test_ocr_contract
```

预期：全部 PASS。

### 任务 3：安全处理配置与 topic 变化

**文件：**
- 修改：`perception/plugins/ocr.py`
- 修改：`perception/tests/test_ocr_contract.py`

- [ ] **步骤 1：编写配置复用和 topic retirement 失败测试**

验证 language/adapter 配置先停止 Node 再更新其字段且不 remove；验证同一 instance
换 topic 时旧 Node 被 remove 并进入 retired 集合，但不会立即 destroy。

- [ ] **步骤 2：实现 Node 更新和 retirement**

增加 `_configure_node()`、`_retire_node()`、`prepare_shutdown()` 和
`destroy_nodes()`。`start` 检测同一 key 的 topic 变化并 retirement；最终销毁同时
覆盖 active 与 retired Node，按对象身份去重。

- [ ] **步骤 3：运行 OCR 测试确认通过**

```bash
python3 -m unittest perception.tests.test_ocr_contract
```

预期：全部 PASS。

### 任务 4：让 spin 异常终止 MCP 假活

**文件：**
- 创建：`perception/tests/test_main_lifecycle.py`
- 修改：`perception/main.py`

- [ ] **步骤 1：编写 spin guard 失败测试**

使用抛出 `RuntimeError("spin failed")` 的 fake executor 和记录 `shutdown()` 的 fake
server 调用 `_spin_executor()`，断言异常被保存且 server 被关闭。

- [ ] **步骤 2：编写 bundle 两阶段关停测试**

构造带 `prepare_shutdown()`、`destroy_nodes()` 的 fake plugin，断言 bundle 分别调用
两个阶段；缺少可选方法的旧插件不报错。

- [ ] **步骤 3：实现 spin guard 和两阶段关停**

在 `main.py` 中先创建 HTTP server，再启动 spin thread。包装函数捕获
`BaseException`，保存异常并 shutdown server。`finally` 中执行
`bundle.prepare_shutdown()`、`executor.shutdown()`、`bundle.destroy_nodes()`、
`rclpy.shutdown()`；清理后若存在 spin 异常则重新抛出。

- [ ] **步骤 4：运行 main 生命周期测试**

```bash
python3 -m unittest perception.tests.test_main_lifecycle
```

预期：全部 PASS。

### 任务 5：完整验证与提交

**文件：**
- 检查：全部变更文件

- [ ] **步骤 1：运行完整测试**

```bash
python3 -m unittest discover -s perception/tests -p 'test_*.py'
python3 -m compileall -q perception
git diff --check
```

预期：全部命令退出码为 0。

- [ ] **步骤 2：检查提交范围和大文件**

```bash
git status --short
find perception docs/superpowers -type f -size +1M -print
```

预期：只有本计划列出的源码、测试和文档发生变化；大文件检查无输出。

- [ ] **步骤 3：提交实现**

```bash
git add perception/main.py perception/plugins/ocr.py \
  perception/tests/test_ocr_contract.py perception/tests/test_main_lifecycle.py \
  docs/superpowers/plans/2026-07-20-ocr-ros-lifecycle.md
git commit -m "fix(ocr): make ROS lifecycle restart-safe"
```

- [ ] **步骤 4：Jetson 验证**

在 Jetson 构建镜像后连续执行 250 次 OCR start/process/stop。验收日志不得包含
`InvalidHandle` 或 `Exception in thread perception_spin`，所有 case 均应在评测超时
前产生结果。
