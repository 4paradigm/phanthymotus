# Inspection Stack

独立的音视频采集卡片宿主。当前分支实现 Gate 1：MCP 合同、Audio Inspector / Video Inspector 生命周期、Core 注册和 Dashboard 编排；writer 为明确标记的 fake writer，不会生成本地文件，也不会访问 COS。

## 本地运行

```bash
cd inspection
python3 -m pip install -r requirements.txt
python3 main.py
```

健康检查：

```bash
curl http://127.0.0.1:15671/health
```

查看两张卡片：

```bash
curl -s http://127.0.0.1:15671/mcp \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
```

运行测试：

```bash
python3 -m unittest discover -s inspection/tests -v
python3 -m unittest discover -s agent-core/tests -v
node --check agent-core/web/js/sidebar.js
node --check agent-core/web/js/canvas.js
node --check agent-core/web/js/flow-view.js
```

## 当前边界

- `runtime_mode=gate1-contract-only` 表示只验证卡片协议和编排；
- `storage_ready=false`，不会宣称已经保存或上传数据；
- `testupload` 返回 `unsupported`，直到 Gate 4 接入真实 COS backend；
- COS 凭证只能由部署 secret 提供，不进入卡片配置或日志。
