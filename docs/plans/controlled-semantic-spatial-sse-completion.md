# 导航完成事件 SSE 路径统一

## 范围

ActuCore 注册 MCP 地址为 /mcp，Agent Core 在该地址后追加 /sse。
将 ActuCore 唯一 SSE 路由从 /sse 改为 /mcp/sse，不保留旧地址别名。
已检索 ActuCore 与 Core 调用方及文档，未发现依赖旧地址的仓内客户端。
不改变终态回调、action_id、导航 topic、运动或超时语义。

## 验证与交付

本地真实 HTTP 测试验证 /mcp/sse 订阅与 action_complete 事件传输、查询参数、旧地址 404；
结合既有导航终态回调测试与 Core SSE 订阅测试，并运行 ActuCore 回归。
同步 README 的接口地址。用户追加授权提交并部署北京 G1：本地构建 ARM64 镜像，只切换现有 ActuCore 服务；切换前确认控制隔离，保存回滚配置。部署后验证 SSE 路由与 Core 订阅，不触发 BOT review，真机到达验收另行进行。

## 本地结果

- HTTP SSE 与导航回调定向：12 passed。
- Core SSE 订阅生命周期：6 passed。
- ActuCore 全量：440 passed、7 skipped、72 subtests passed。
  跳过项为 5 项 ROS runtime、OpenCV、Linux procfs；本次未更改 ROS action 实现。
- git diff --check 通过；README 地址已与唯一入口一致，旧地址仅保留在负向测试及变更说明。
- 未部署，尚未验证真机到达后 Core 唤醒与后续对话。
