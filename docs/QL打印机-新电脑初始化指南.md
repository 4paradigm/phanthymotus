# QL 打印机：新前台电脑初始化操作指南

适用场景：机器人通过 SSH 将自带的 ParadigmVisitorBridge 安装包传到 Windows 前台电脑，再由电脑驱动打印机打印范式访客贴。无需人工复制安装包。

## 1. 准备条件

- 前台电脑使用 Windows x64，安装 Microsoft Edge，并保持操作员登录 Windows。不要注销用户。
- 安装打印机驱动、连接打印机，在 Windows 中完成一次测试打印，并设好标签纸尺寸、方向和打印质量。
- 前台电脑与机器人网络互通。新环境安装依赖时需要访问 python.org 和 PyPI。
- 机器人部署支持本功能的 core 和 perception；JetPack 镜像须匹配机器人。管理员需将 SSH 私钥及 known_hosts 挂载到 perception 的 `/run/ql-printer-auth`，只读挂载即可。
- 此功能适用于现有范式访客网站；更换其他访客系统需要适配。打印机会被设为该 Windows 用户的默认打印机，不会自动恢复原默认值。

## 2. 在前台电脑开启 SSH（人工首次操作）

右键 PowerShell，选择“以管理员身份运行”，逐条执行：

```powershell
whoami
ipconfig
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Set-Service -Name sshd -StartupType Automatic
Start-Service sshd
Get-Service sshd
```

记下 Windows 用户名和机器人能访问的 IPv4 地址；建议由网络管理员固定地址。如果安装提示需要重启，先完成重启。确认 Windows 防火墙允许机器人访问 TCP 22；不要关闭整个防火墙。

## 3. 给机器人授权（人工首次操作）

在机器人终端执行。若目标密钥文件已存在，保留原文件，不要覆盖：

```bash
sudo mkdir -p /opt/phanthy-motus/secrets/ql_printer
sudo ssh-keygen -t ed25519 -f /opt/phanthy-motus/secrets/ql_printer/frontdesk_bridge_ed25519 -N "" -C "robot-frontdesk-bridge"
sudo cat /opt/phanthy-motus/secrets/ql_printer/frontdesk_bridge_ed25519.pub
```

如果已有私钥但没有 `.pub`，使用以下命令显示公钥：

```bash
sudo ssh-keygen -y -f /opt/phanthy-motus/secrets/ql_printer/frontdesk_bridge_ed25519
```

在前台管理员 PowerShell 打开授权文件：

```powershell
New-Item -ItemType Directory -Path "C:\ProgramData\ssh" -Force | Out-Null
notepad "C:\ProgramData\ssh\administrators_authorized_keys"
```

将机器人输出的完整公钥粘贴为单独一行，保存。这里使用的 Windows 账户必须属于管理员组。同一公钥只保留一条；安装用公钥不能带 `command=...start-bridge...` 强制启动限制。保留其他机器人的公钥。此授权允许机器人执行管理员命令，私钥仅保存在机器人上。

设置文件权限：

```powershell
$keyFile = "C:\ProgramData\ssh\administrators_authorized_keys"
icacls.exe $keyFile /inheritance:r
icacls.exe $keyFile /grant:r "*S-1-5-32-544:F" "*S-1-5-18:F"
Restart-Service sshd
ssh-keygen -lf "C:\ProgramData\ssh\ssh_host_ed25519_key.pub"
```

记下最后输出的 SHA256 指纹。在机器人上执行以下命令，将 `电脑IP` 替换成实际地址：

```bash
ssh-keyscan -t ed25519 电脑IP 2>/dev/null > /tmp/ql-frontdesk-hostkey
ssh-keygen -lf /tmp/ql-frontdesk-hostkey
```

两边指纹必须完全相同。确认后，将扫描结果添加到 known_hosts（保留其他电脑记录）：

```bash
sudo sh -c 'cat /tmp/ql-frontdesk-hostkey >> /opt/phanthy-motus/secrets/ql_printer/known_hosts'
```

测试授权，把 `Windows用户名` 和 `电脑IP` 替换为实际值：

```bash
sudo ssh -F /dev/null -T -i /opt/phanthy-motus/secrets/ql_printer/frontdesk_bridge_ed25519 -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=/opt/phanthy-motus/secrets/ql_printer/known_hosts Windows用户名@电脑IP whoami
```

返回 Windows 用户身份才算通过。若返回 `STARTED`，密钥仍被限制为只能启动，不能安装。不要继续点初始化。

管理员在机器人 perception 服务中配置目录挂载：

```yaml
- /opt/phanthy-motus/secrets/ql_printer:/run/ql-printer-auth:ro
```

重建对应服务后，检查容器内两个文件可读。容器名称依实际部署填写，例如 `embodied-perception`：

```bash
docker exec embodied-perception sh -lc 'test -r /run/ql-printer-auth/frontdesk_bridge_ed25519 && test -r /run/ql-printer-auth/known_hosts && echo SSH_READY'
```

## 4. 在画布初始化电脑

1. 停止智能控制，将“感知”中的 `QL_printer` 拖入画布，点击卡片右上角齿轮。
2. 桥接服务地址填写 `http://电脑IP:8765`。不要填写机器人 IP，也不要填写 `127.0.0.1`。
3. 新电脑可填写至少 24 位随机访问令牌；安全浏览器环境下留空会在初始化时生成。若浏览器无法自动生成，请使用密码管理器生成并填写。升级已有电脑必须使用其现有 `config.json` 中的 `shared_token`。
4. SSH 用户填写刚才验证的 Windows 用户名，端口通常为 `22`，私钥路径填写 `/run/ql-printer-auth/frontdesk_bridge_ed25519`。
5. 点击“初始化/升级电脑”，保持窗口打开并等待结果，首次安装可能需要数分钟。不要重复点击，也不要在安装期间执行打印。
6. 机器人通过 SSH 传输安装包，安装 Python/依赖、配置桥接、注册登录启动和监护任务，并检查服务是否启动。
7. 成功后点击“刷新打印机”，选择实际标签打印机，开启“机器人远程启动桥接”，点击“保存”。

新安装的默认目录是 `C:\ProgramData\ParadigmVisitorBridge\app`。已有监护任务时使用其工作目录，保留原配置、数据库和浏览器资料；升级前备份代码。

## 5. 首次登录与验收

在前台电脑上登录访客网站。需要打开专用登录窗口时，在前台 PowerShell 执行：

```powershell
cd C:\ProgramData\ParadigmVisitorBridge\app
.\.venv\Scripts\python.exe .\login_edge.py --config .\config.json
```

若是升级旧安装，请使用原安装目录。登录窗口如提示操作步骤，按提示完成并退出登录工具，再使用桥接。不要同时启动多个使用同一浏览器资料目录的进程。

在画布启用智能控制：

1. 执行 `list`，确认返回当天未入场访客。
2. 找到经确认的测试预约，核对姓名、被访人、时间和手机尾号。
3. 执行 `print`，填写列表中的准确姓名和被访人。
4. 检查选中的打印机是否出纸、Logo/二维码和纸张布局是否正确，以及预约状态是否符合预期。

“安装成功”只表示桥接 HTTP 服务就绪，不代表网站已登录或实物打印已通过。全新电脑初始化须完成以上现场验收。

## 6. 常见问题

| 提示 | 处理 |
|---|---|
| SSH 返回 STARTED | 安装密钥仍有强制启动限制，检查授权文件及重复公钥 |
| Permission denied | 检查用户名、公钥、管理员身份及文件权限 |
| 主机指纹不匹配 | 核实电脑身份后再更新 known_hosts，不要跳过验证 |
| Existing token differs | 使用该电脑现有令牌，不要另生成令牌覆盖 |
| 刷新打印机 HTTP 404 | 运行中的桥接缺少新接口，检查升级结果和实际任务目录 |
| 查询要求登录/浏览器关闭 | 在前台专用 Edge 中登录，保持 Windows 用户登录 |
| 安装失败 | 查看下方安装日志，提供错误信息时遮住令牌 |

```powershell
Get-Content -LiteralPath "C:\ProgramData\ParadigmVisitorBridge\bootstrap.log" -Tail 30
```

日常 ACTION 只需 `list` 和 `print`。打印机刷新、初始化和安装进度都在齿轮窗口中完成。安装授权若被收回，后续远程升级需要重新授权；日常远程启动权限须另行保留并验证。
