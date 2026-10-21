# 文件下载卡片

从 AgentCore 工具列表添加 `file_download`，将支持下载的来源卡片右侧“可下载文件”输出口连到下载卡片左侧的同名输入口。选择全部文件、照片或视频，点击“查看文件并下载”。列表支持逐个下载及刷新；项目停止时也可使用。

## 来源能力声明

driver 在工具的 `inputSchema` 中声明只读的文件列表动作：

```json
"x-file-download": {"protocol": "motus.files/1", "action": "list_files"}
```

动作名可以自定义，须同时加入工具的 action schema 并实现 dispatch。Agent Core 的现有 MCP 发现会保留整个 `inputSchema`，前端根据声明自动显示输出口，不匹配工具名或厂商名。声明的列表动作不显示在原卡片的执行下拉菜单中。未声明或协议不受支持的工具保持原行为，不显示文件输出口。

下载卡片通过现有 `/api/mcp/{id}/call` 调用来源工具及其声明动作；多实例工具携带来源卡片的 `instance_id`。动作返回普通 MCP 文本 JSON（或直接 JSON 对象）：

```json
{"ok": true, "files": [{"name": "IMG_20261022.jpg", "path": "/opt/phanthy-motus/data/vision_capture/photos/IMG_20261022.jpg", "size": 12345, "mime_type": "image/jpeg"}]}
```

`name` 是文件名，`path` 是 Agent Core 可访问的绝对文件路径，`size` 是正数的字节大小；未提供 `mime_type` 时按 `application/octet-stream` 处理，仅显示在“全部文件”。错误返回 `{"ok": false, "message": "原因"}`。来源负责查询持久历史、排除未完成文件；列表查询不得启动拍摄或录制。目录尚未创建可返回空列表，读取失败应返回错误。

首个接入工具是 Bumi driver 的 `vision_capture`：新增只读 `list_files`，从原有 `output_dir/photos/*.jpg`、`output_dir/videos/*.mp4` 查询历史，排除空文件、符号链接和录制中的视频；照片写入与扫描互斥，避免列出半成品照片。相机离线也能查询。已删除文件在刷新后消失。原拍照、录像和取消接口保持不变。此能力需要配套更新 driver；旧 driver 仍可拍摄，但没有下载输出口。

## 连线和文件传输

连线继续保存在现有 `canvas_layout.connections`，使用 `format: file/reference`，两端端口索引为 `files`。它绑定来源卡片身份，不是 DDS 话题，不发布文件字节，不参与 DDS 话题推导、订阅或项目启动依赖。

下载经带鉴权的 `/api/file/download` 获取文件，浏览器保存到用户电脑。路径必须在 Agent Core 容器内可读，两个容器应在相同路径挂载共享目录；连线不会建立跨容器共享。首版不包含压缩包批量下载、已删除文件恢复或 driver 异常退出后残留视频的完整性检查。

## 验收

- 任意工具名均可通过声明获得输出口，未声明的工具不显示。
- 未连接来源时提示连线，不调用拍摄或文件接口。
- 连接后列出历史照片、已完成视频；刷新页面仍可通过保存的连接查询。
- 删除文件后刷新列表；下载鉴权失败时显示错误。
- 录制中及取消后的未完成视频不提供下载。
- 断开连线后不再查询来源；连接及下载均不触发拍照或录像。
- 原拍照、录像、参数显隐、结果展示和普通数据连线保持原行为。

本地自动测试及模拟浏览器检查不代表实机验收；此变更不会自动部署到机器人。
