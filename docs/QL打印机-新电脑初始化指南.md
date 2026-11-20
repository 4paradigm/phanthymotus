# QL 打印机：客户安装与使用指南

使用本功能后，机器人可以查询当天未入场的访客预约，并通过前台电脑打印访客贴。首次安装由客户与技术支持共同完成，安装包由机器人自动传输，客户不需要手动复制文件。

## 第一步：准备前台电脑（客户完成）

1. 打开电脑并登录 Windows，确认已经安装 Microsoft Edge 浏览器。
2. 连接打印机，安装打印机驱动并放好标签纸。
3. 在 Windows 中打印一张测试页，确认打印机正常。
4. 准备访客系统登录账号，并让电脑接入机器人可访问的网络。

完成标准：电脑能联网，打印机能够正常出纸。

## 第二步：首次连接授权（技术支持协助）

请联系技术支持，提供前台电脑的 IP 地址和 Windows 用户名；不知道在哪里查看时，由技术支持协助查询。

技术支持负责开启远程连接、配置机器人公钥、核对电脑身份并验证连接。这一步需要有权限的电脑管理员参与，每台新电脑首次安装时完成一次。客户无需自行输入后面的技术命令，也不需要将 Windows 密码填进卡片。

完成标准：技术支持确认“机器人已能连接这台电脑，可以初始化”。

## 第三步：点击自动安装（客户操作）

在机器人画布停止智能控制，拖入“感知”下的 `QL_printer` 卡片，点击右上角齿轮。

| 设置项 | 如何填写 |
|---|---|
| 桥接服务地址 | 技术支持提供，例如 `http://192.168.1.20:8765`，这里是前台电脑地址 |
| 访问令牌 | 使用技术支持提供的连接令牌；它不是 Windows 密码 |
| 前台 SSH 用户 | 技术支持确认的 Windows 用户名 |
| 前台 SSH 端口、私钥路径 | 按技术支持提供的配置填写，通常无需自行修改 |
| 机器人远程启动桥接 | 选择 Yes |

点击“初始化/升级电脑”，保持窗口打开，等待安装结果。首次安装可能需要几分钟，不要重复点击。

完成标准：窗口显示安装成功。如果提示失败，记录提示并联系技术支持，不要反复安装。

## 第四步：选择打印机（客户操作）

在同一个齿轮窗口中，点击“刷新打印机”，从“目标打印机”下拉框选择需要使用的标签打印机，点击“保存”。

注意：打印时会将所选设备设为当前 Windows 用户的默认打印机。标签纸尺寸、方向等由技术支持在电脑上设好。

完成标准：列表中能找到打印机，并已选择、保存。

## 第五步：登录并试打（客户与技术支持共同完成）

在技术支持打开的专用 Edge 窗口中登录访客系统。普通浏览器中的登录状态不一定能供桥接使用。

回到画布开启智能控制：

1. ACTION 选择 `list`，点击执行，查看当天未入场的访客预约。
2. 选一条已确认的测试预约，核对姓名、被访人、时间和手机尾号。
3. ACTION 选择 `print`，填写该预约的准确姓名和被访人，点击执行一次。
4. 检查选中的打印机是否出纸，姓名、Logo、二维码和排版是否正常，并确认预约状态。

完成标准：查询正常，目标打印机实际出纸。安装成功提示本身不代表打印测试已完成。

## 以后每天怎么用

前台电脑保持开机、联网，Windows 用户保持登录。配置正确时，机器人会按需启动桥接，无需每天输入启动命令。访客网站登录过期时，需要重新登录。

日常只使用 `list` 和 `print`；换打印机时，在齿轮中刷新、重新选择并保存。更换电脑时，从第二步重新配置。

## 遇到问题怎么办

| 情况 | 客户先做什么 |
|---|---|
| 连不上电脑 | 确认电脑开机、联网且已登录 Windows，然后联系技术支持 |
| 打印机列表为空或刷新失败 | 确认打印机驱动已安装，记录窗口提示交给技术支持 |
| 提示令牌不正确 | 请技术支持核对连接配置，不要随意换令牌 |
| 查不到预约 | 确认预约日期为当天、状态为未入场，并检查网站是否仍登录 |
| 显示失败或没有出纸 | 先检查纸张、打印机和打印队列，记录结果；不要连续重复点打印 |

提供错误截图时，请遮住访问令牌和访客隐私信息。

---

# 技术支持附录（客户无需自行执行）

以下为实施人员配置与排查参考。客户操作到此结束。首次全新电脑安装仍须现场验收；本指南不将模拟测试视为实机验收。

适用场景：机器人通过 SSH 将自带的 ParadigmVisitorBridge 安装包传到 Windows 前台电脑，再由电脑驱动打印机打印范式访客贴。无需人工复制安装包。

## 1. 准备条件

- 前台电脑使用 Windows x64，安装 Microsoft Edge，并保持操作员登录 Windows。不要注销用户。
- 安装打印机驱动、连接打印机，在 Windows 中完成一次测试打印，并设好标签纸尺寸、方向和打印质量。
- 前台电脑与机器人网络互通。新版镜像已在构建时打包 Python 和 Windows 依赖，前台从机器人接收并离线安装；访客网站仍需网络访问。
- 机器人部署支持本功能的 core 和 perception；JetPack 镜像须匹配机器人。新版 perception 部署模板会自动把宿主机 `/opt/phanthy-motus/secrets/ql_printer` 只读挂载到容器 `/run/ql-printer-auth`。每台机器人仍需单独生成自己的 SSH 密钥和 `known_hosts`，这些秘密文件不进入镜像或 Git 仓库。
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

新版 perception 镜像的标准服务模板已包含以下目录挂载，不需要在每台机器人上手工编辑 `docker-compose.yml`：

```yaml
- /opt/phanthy-motus/secrets/ql_printer:/run/ql-printer-auth:ro
```

通过部署控制台重新部署 perception 后，配置会自动合并到宿主机 compose。检查容器内两个文件可读。容器名称依实际部署填写，例如 `embodied-perception`：

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
