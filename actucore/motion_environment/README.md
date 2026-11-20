# Motion 环境管理

这是独立于 Motion API 和 Agent Core 的主机引导服务。目标服务器没有安装 Motion 时，主页面仍可通过它检查环境、安装固定 G1 运行环境、查看真实日志并取消任务。任务在独立进程运行，关闭页面或重启 HTTP 服务不会终止安装。

管理员先通过自己的 SSH 管理入口在受信任的 Linux x86_64 / Docker 控制环境部署一次本服务。Core 不需要 Docker socket，也不保存操作员电脑的 SSH 私钥。当前接口只操作管理员配置的本机目标和 `supermimic-g1-v1` recipe，不接受用户提供的 shell、镜像名称、挂载路径或任意 SSH 地址。

## 前置条件

- 主机已安装 NVIDIA 驱动、Docker 和 NVIDIA Container Toolkit，运行账户能使用 Docker。
- 可访问官方软件源；可在管理员配置中设置仅用于此次下载/构建的 `download_proxy`。不修改 Docker daemon、主机代理配置或驱动。
- 准备 Motion 源码快照，以及通过 `python -m motion_environment.package_profile --profile <已验证profiles.json> --output <g1.tgz>` 生成的机器人包。包内只有完整 G1 MJB、配置和校验清单，不包含训练动作或策略。
- `source_dir/motion_service` 必须是完整发布快照，包含会话控制、分组训练预览及其 `camera_fit.py` 等导入依赖。发布时先核对全部 Python 文件摘要，再原子替换目录；不要仅更新已运行容器而保留旧安装源。每次安装的 `source_sha256` 涵盖源码、配方、机器人包及 wheel 清单，版本变化会创建独立容器。
- 机器人包绑定 MuJoCo 3.16.0；安装固定上游仓库提交和兼容性关键依赖。其他传递依赖仍由包管理器解析，不能将 recipe 宣称为所有依赖完全锁定的离线发行版。

管理员配置文件示例（路径和目标由管理员提供，0600 权限）：

```json
{
  "source_dir": "/opt/motion-environment/source",
  "profile_package": "/opt/motion-environment/assets/g1.tgz",
  "targets": [{
    "id": "gpu-server",
    "label": "训练服务器",
    "host": "服务器管理地址",
    "gpu_ids": [0, 1, 2, 3, 4, 5, 6, 7],
    "render_gpu": 2,
    "existing": {
      "endpoint": "http://私网地址:15732",
      "token_file": "/opt/motion-environment/credentials/existing.token",
      "container": "已有Motion容器名称",
      "python": "/已有环境/bin/python"
    }
  }]
}
```

`existing` 可省略；独立安装不依赖已有 Motion API。已有环境复用会执行实际 CUDA / Warp 算术、G1 物理步进和真实 JPEG 渲染验证；检测会话随即关闭，不训练、不连接物理机器人。

独立环境还会创建官方 G1 GPU 任务、执行两步真实物理，并核对完整机器人模型指纹。配置至少两张 GPU 时，会用两 rank、每 rank 1 MiB 的真实 NCCL all-reduce 验证容器共享内存和通信。只有这些检查及原生 MuJoCo 渲染通过后才提供连接；没有运行策略训练。

工作台兼容检查会创建带 `view_id` 的临时会话，验证控制者、只读观察者和心跳，携带 `control_epoch` 完成步进并关闭会话。只有旧仿真接口、缺少新控制权合同的服务不会被判为可连接。

服务启动：

```sh
PYTHONPATH=/opt/motion-environment/source python3 -m motion_environment.server \
  --host <私网监听地址> --port 15734 \
  --root /opt/motion-environment/state \
  --config /opt/motion-environment/config.json \
  --token-file /opt/motion-environment/credentials/manager.token
```

凭证应随机生成且至少 32 字符，文件权限 0600。监听私网或回环地址，由已有后端通过认证访问；不要公开服务端口。单个状态目录只允许一个 HTTP 管理进程。停止 HTTP 服务会保留已运行的任务；取消安装应调用任务的 `cancel` 操作。

## 接口

`GET /health` 和 `POST /v1/call` 均要求 `Authorization: Bearer …`。POST body 是含 `action` 的 JSON 对象。

| action | 参数 | 结果 |
|---|---|---|
| `targets` | 无 | 配置的目标、能力、已有地址、`latest_job` |
| `check` | `target_id` | 立即返回异步检查任务 |
| `install` | `target_id`, `mode: reuse / isolated`, 可选 `recipe_id` | 立即返回异步任务；同目标同方式的活动请求复用原任务 |
| `status` | `job_id`, 可选 `cursor` | 任务状态、最多 200 条新日志、下一字节游标；省略 job_id 返回最近任务列表 |
| `cancel` | `job_id` | 写入取消标记；停止本任务的命令和验证进程，保留缓存与持久数据 |
| `connection` | 成功安装任务的 `job_id` | **仅后端消费** endpoint / api_key，用于连接；不得透传 iframe、浏览器存储或日志 |

公开状态为 `queued / running / succeeded / failed / cancelled / interrupted`。`check` 成功只表示检查通过；只有 `install` 成功可获取连接。失败保留原始阶段和日志，重试创建新任务并复用已校验下载和 Docker 缓存。进程消失后状态变为 interrupted；不会伪装成成功，也不会覆盖已提交的成功终态。

独立安装创建按源码摘要命名的容器及数据卷，不发布宿主机端口。所有复用资源先检查归属，拒绝无标签或归属不同的同名资源；新版本创建独立环境。取消或失败只停止本任务创建/启动的容器，已经运行的复用服务保留。

下载代理启用时，官方 Python 镜像通过固定 OCI index 获取，逐项校验 manifest、压缩层、展开层和 Docker image ID 后 `docker load`，无需修改 daemon 配置。日志包含真实下载字节、Git 进度和包安装输出；没有编造的总体百分比或完成时间。

## 下载受限时准备原始包缓存

管理员可在网络较快的电脑上运行以下命令，为固定 Linux x86_64 / Python 3.13 / glibc 2.36 目标准备原始 wheel；下载电脑本身可以是 macOS：

```sh
PYTHONPATH=actucore python3 -m motion_environment.wheel_cache \
  --directory /本次部署缓存/wheels --mirror
```

CUDA 和通用依赖的版本及摘要取自官方 PyPI；Torch 取自官方 PyTorch cu126 索引，Warp 取自 NVIDIA 官方 release。`--mirror` 只选择传输渠道（PyPI 包走 Aliyun、Torch 走 NJU），完整文件始终需要匹配官方 SHA256。下载分块可续传；完整摘要失败会清除该文件的分块，重试不会反复复用损坏片段。

将完整缓存目录及 `manifest.json` 传到引导服务所在机器，然后在管理员配置增加 `"wheel_cache": "/本次部署缓存/wheels"`。未完成的上传不要使用最终文件名。安装任务再次校验固定版本、目标 wheel 标签、包内名称/版本、文件长度和完整 SHA；容器构建中再次校验后从原始包安装。这里没有复制任何已安装 venv 或 site-packages。小型传递依赖及上游源码仍需要网络，缓存不是完整离线发行包。

若管理员正上传缓存，可显式设置 `cache_wait_seconds`（0–7200，默认 0）为有限等待时长。最后才原子提交完整 `manifest.json`。等待期间任务显示“准备安装包”和实际接收字节，支持取消；缺少可信完整清单时不显示百分比，也不启动构建。超时保留缓存并报告可重试的真实原因。
