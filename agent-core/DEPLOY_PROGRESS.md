# 部署进度改进方案

## 问题

在 AgiBot X2 上部署 perception 时遇到以下问题：

1. **超时频繁**：16GB 镜像拉取时间长，经常超时
2. **进度不可见**：用户不知道部署进行到哪一步
3. **错误不明确**：磁盘不足等问题的报错不清晰

## 解决方案

### 1. 预检查机制（Preflight Checks）

部署前自动检查：
- **磁盘空间**：需要 2.5 倍镜像大小（拉取 + 解压）
- **网络连接**：测试到仓库的延迟
- **仓库认证**：验证 docker login 状态
- **容器冲突**：检查是否有同名容器

文件：`phanthymotus/agent-core/src/api/preflight.py`

### 2. WebSocket 实时进度推送

通过 WebSocket 实时推送部署状态：

```
/ws/deploy/{driver_id}
```

事件类型：
- `start` - 开始部署
- `check` - 预检查结果（磁盘、网络等）
- `progress` - 拉取/启动进度（包含百分比、速度、ETA）
- `error` - 错误及解决建议
- `done` - 部署完成

文件：`phanthymotus/agent-core/src/api/deploy_stream.py`

### 3. 异步部署引擎

带进度追踪的异步部署实现：
- 逐层追踪镜像拉取进度
- 计算下载速度和剩余时间
- 更友好的错误提示

文件：`phanthymotus/agent-core/src/api/drivers_async.py`

### 4. 前端进度 UI

可视化部署进度的 JavaScript 组件：
- 实时进度条
- 预检查结果展示
- 错误提示和建议
- 详细日志

文件：`phanthymotus/agent-core/web/js/deploy-progress.js`

## API 使用

### 1. 预检查（可选，部署前手动调用）

```bash
GET /api/drivers/{driver_id}/preflight
```

返回：
```json
{
  "code": 200,
  "data": {
    "overall_status": "warning",
    "can_proceed": true,
    "checks": {
      "disk": {
        "status": "warning",
        "free_gb": 15.3,
        "total_gb": 57.0,
        "message": "磁盘空间紧张：仅 15.3 GB 可用（建议 20 GB）",
        "suggestion": "建议清理旧镜像：docker image prune -a"
      },
      "network": {
        "status": "pass",
        "latency_ms": 45.2,
        "message": "网络连接良好 (45 ms)"
      },
      "registry": {
        "status": "pass",
        "registry": "ccr.ccs.tencentyun.com",
        "message": "已连接到仓库"
      }
    },
    "recommendations": [
      "建议清理旧镜像：docker image prune -a"
    ]
  }
}
```

### 2. 部署（新版，带进度）

```bash
POST /api/drivers/{driver_id}/deploy-v2
```

请求体与原 `/deploy` 相同：
```json
{
  "image": "registry/namespace/image:tag"
}
```

返回：
```json
{
  "code": 200,
  "data": {
    "status": "starting",
    "service": "perception",
    "container_name": "embodied-perception"
  }
}
```

### 3. WebSocket 监听进度

```javascript
const ws = new WebSocket('wss://host:15678/ws/deploy/perception?token=YOUR_TOKEN');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  switch (data.type) {
    case 'check':
      console.log(`Check: ${data.message} [${data.status}]`);
      break;
      
    case 'progress':
      console.log(`${data.stage}: ${data.message} (${data.percent}%)`);
      if (data.speed) {
        console.log(`  速度: ${data.speed}`);
      }
      break;
      
    case 'error':
      console.error(`Error: ${data.message}`);
      if (data.suggestion) {
        console.log(`  建议: ${data.suggestion}`);
      }
      break;
      
    case 'done':
      console.log(`完成: ${data.message} (耗时 ${data.elapsed}s)`);
      ws.close();
      break;
  }
};
```

## 前端集成

### 简单方式（使用封装好的 UI）

```javascript
// 引入脚本
<script src="/js/deploy-progress.js"></script>

// 部署时显示进度
const ui = new DeployProgressUI('perception', 'Perception 感知层');
ui.show();

// 同时发起部署请求
fetch('/api/drivers/perception/deploy-v2', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ image: 'registry/perception:latest' })
});
```

### 自定义方式

```javascript
const monitor = new DeployProgressMonitor('perception');

monitor.onProgress = (event) => {
  // 更新你的 UI
  if (event.type === 'progress') {
    updateProgressBar(event.percent);
    updateStatus(event.message);
  }
};

monitor.onError = (event) => {
  showError(event.message, event.suggestion);
};

monitor.onDone = (event) => {
  showSuccess(event.message);
};

monitor.connect();
```

## 错误提示改进

### 磁盘空间不足

**旧版**：
```
pull failed: no space left on device
```

**新版**：
```
错误: 磁盘空间不足（15.3 GB 可用 / 57 GiB 总计）
建议: 必须清理磁盘空间：docker image prune -a && docker builder prune -a
```

### 网络超时

**旧版**：
```
pull failed: timeout
```

**新版**：
```
错误: 连接 ccr.ccs.tencentyun.com 超时
建议: 请检查网络连接和防火墙设置
```

### 仓库认证失败

**新版**：
```
错误: 仓库认证失败: ccr.ccs.tencentyun.com
建议: 请运行 docker login 登录仓库
```

## 部署流程对比

### 旧流程

1. 调用 `/deploy` → 立即返回
2. 轮询 `/status` 查看日志
3. 看到 `[deploy] done` 或错误

**问题**：
- 不知道进度百分比
- 不知道预计剩余时间
- 出错了才知道磁盘不够

### 新流程

1. （可选）调用 `/preflight` 检查环境
2. 打开 WebSocket `/ws/deploy/{driver_id}`
3. 调用 `/deploy-v2` 开始部署
4. 实时收到进度事件：
   - ✓ 磁盘空间充足：45.2 GB 可用
   - ✓ 网络连接良好 (45 ms)
   - ✓ 已连接到仓库
   - 开始拉取镜像…
   - 拉取进度: 12 层 (45.3%) · 2.3 MB/s
   - 拉取进度: 12 层 (78.9%) · 2.1 MB/s
   - 镜像拉取完成
   - 准备容器配置…
   - 启动容器…
   - 部署完成：embodied-perception (耗时 412.3s)

## 测试

### 测试预检查

```bash
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:15678/api/drivers/perception/preflight
```

### 测试完整部署

```bash
# 终端 1：监听 WebSocket
websocat "ws://localhost:15678/ws/deploy/perception?token=$TOKEN"

# 终端 2：发起部署
curl -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"image": "ccr.ccs.tencentyun.com/phanthy-motus/perception:release.260901.abc"}' \
  http://localhost:15678/api/drivers/perception/deploy-v2
```

## 兼容性

- 旧的 `/deploy` 端点保持不变，继续可用
- 新的 `/deploy-v2` 是额外功能，不影响现有系统
- 如果 WebSocket 连接失败，部署仍会正常进行（只是看不到进度）
- 前端可以先尝试 `/deploy-v2`，失败时回退到 `/deploy`

## 后续优化建议

1. **断点续传**：大镜像拉取中断后能从断点继续
2. **多任务队列**：同时部署多个 driver 时排队处理
3. **历史记录**：保存每次部署的时间、速度、是否成功
4. **智能推荐**：根据网络状况推荐最佳部署时间
5. **镜像预热**：空闲时提前拉取常用镜像

## 文件清单

新增文件：
- `phanthymotus/agent-core/src/api/preflight.py` - 预检查逻辑
- `phanthymotus/agent-core/src/api/deploy_stream.py` - WebSocket 进度推送
- `phanthymotus/agent-core/src/api/drivers_async.py` - 异步部署引擎
- `phanthymotus/agent-core/src/api/drivers_v2_endpoint.py` - 新 API 端点
- `phanthymotus/agent-core/web/js/deploy-progress.js` - 前端进度 UI

修改文件：
- `phanthymotus/agent-core/src/start.py` - 注册新路由

## 部署到测试环境

```bash
cd phanthymotus/agent-core

# 本地测试
./run.zsh

# 构建镜像（在 Orin 6 上原生构建最快）
cd ../deploy
./build_core.sh

# 部署到测试机
ssh nvidia@10.100.121.16
docker pull <刚构建的镜像>
# 通过 web console 更新 agent-core
```

## 注意事项

1. **Docker SDK 依赖**：确保 `docker` Python 包已安装
2. **WebSocket 认证**：使用与其他 WebSocket 相同的 token 参数
3. **并发安全**：`_deploy_sync_inner` 已有 `_deploy_slot` 防止同一 driver 并发部署
4. **日志兼容**：新版继续使用 `_log_deploy` 写日志，旧版轮询仍能看到
