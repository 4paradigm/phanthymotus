Visitor_Badge_Printer 前台 SSH 自动化说明

机器人与前台电脑之间只使用 SSH，不开放 HTTP 业务端口，也不需要访问令牌。

卡片填写：前台电脑 IP、Windows SSH 用户名和密码；连接后选择访客网页和目标打印机。

管理员首次启用 Windows OpenSSH Server、确认用户名密码可登录，并安装打印机驱动。
点击卡片的连接初始化按钮后，机器人通过 SSH/SCP 把按需自动化环境安装到：
D:\ParadigmVisitorAutomation
Windows 安装依赖仍随机器人镜像提供，初始化无需额外联网下载或手工登记公钥。

日常 list 和 print 均通过 SSH 创建一次性任务，再由 Windows 交互用户的计划任务
操作专用 Edge 和打印机。没有常驻网络服务；旧的 8765 服务、任务和防火墙规则会在
升级时停止并清理。

必须选择已经登录且能看到访客列表的网页；如自动化 Edge 提示登录，先完成登录。
list 返回当天全部未入场访客，Agent 按
姓名、被访人、时间和手机号后四位匹配后调用 print。
