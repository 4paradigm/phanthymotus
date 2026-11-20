QL_printer 前台桥接使用说明
============================

一、文件位置

桥接安装目录：D:\ParadigmVisitorBridge
程序目录：D:\ParadigmVisitorBridge\app
安装和运行日志：D:\ParadigmVisitorBridge\bootstrap.log、D:\ParadigmVisitorBridge\app\data\bridge.log

二、机器人画布中的配置

普通配置只需要填写三项：

1. 前台电脑 IP
   填写前台电脑的固定 IPv4 地址，例如：10.100.10.14。
   卡片会自动使用 http://电脑IP:8765。

2. 前台 SSH 用户
   填写已授权机器人公钥的 Windows 用户，例如：4paradigm。

3. 目标打印机
   初始化成功后选择实际使用的标签打印机。

访问令牌、查询和打印超时、SSH 端口、远程启动开关和私钥路径均使用安全默认值，通常无需修改。
全新电脑首次初始化时令牌会自动生成。已经安装过桥接的电脑通过已授权 SSH 自动读取并复用原令牌，不影响其他机器人。
安装成功后卡片自动保存连接配置并读取打印机；选择打印机后再点保存。

三、首次安装

确认前台电脑已开启 OpenSSH Server、D 盘可用、机器人公钥已授权，然后在齿轮中点击“安装或升级前台桥接”。
安装成功后选择目标打印机并保存。首次使用前，需要在前台专用 Edge 中登录访客网站。

四、日常使用

QL_printer 的 list 会列出当天未入场访客；Agent 根据姓名、公司、预约人等信息选择正确记录后调用 print。
不要根据猜测直接打印，也不要重复打印同一张访客贴。

五、故障检查

1. 检查桥接状态：浏览器访问 http://127.0.0.1:8765/health
2. 查看安装日志：Get-Content D:\ParadigmVisitorBridge\bootstrap.log -Tail 50
3. 查看运行日志：Get-Content D:\ParadigmVisitorBridge\app\data\bridge.log -Tail 50
4. 查看任务：Get-ScheduledTask -TaskName "Paradigm Visitor Bridge Watchdog"
5. 如果画布无法读取打印机，先确认前台 IP、桥接状态和打印机驱动；换机器人时执行一次安装或升级以同步令牌和访问规则。

六、安全说明

不要删除 OpenSSH 的 authorized_keys，不要公开访问令牌或机器人私钥。
更换前台电脑后，需要重新授权该机器人的 SSH 公钥，并重新执行初始化。
