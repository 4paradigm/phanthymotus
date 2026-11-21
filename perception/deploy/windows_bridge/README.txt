Visitor_Badge_Printer 前台 SSH 自动化说明

机器人与前台电脑之间只使用 SSH，不开放 HTTP 业务端口，也不需要访问令牌。

卡片只配置：前台电脑 IP、Windows SSH 用户名、目标打印机。

管理员首次启用 Windows OpenSSH Server、授权机器人公钥并安装打印机驱动。点击
“准备前台电脑”后，机器人通过 SSH/SCP 把按需自动化环境安装到：
D:\ParadigmVisitorAutomation

日常 list 和 print 均通过 SSH 创建一次性任务，再由 Windows 交互用户的计划任务
操作专用 Edge 和打印机。没有常驻网络服务；旧的 8765 服务、任务和防火墙规则会在
升级时停止并清理。

首次使用需要在专用 Edge 中登录访客网站。list 返回当天全部未入场访客，Agent 按
姓名、被访人、时间和手机号后四位匹配后调用 print。
