QL_printer 前台桥接使用说明
============================

一、文件位置

桥接安装目录：D:\ParadigmVisitorBridge
程序目录：D:\ParadigmVisitorBridge\app
安装和运行日志：D:\ParadigmVisitorBridge\bootstrap.log、D:\ParadigmVisitorBridge\app\data\bridge.log

二、机器人画布中的配置

1. 桥接服务地址
   填写前台电脑的固定 IPv4 地址，格式为：http://电脑IP:8765
   示例：http://10.100.10.14:8765

2. 访问令牌
   必须与前台 app\config.json 中的 shared_token 完全一致。
   令牌相当于打印权限密码，请勿发给无关人员或写入公开文档。

3. 查询超时
   查询当天访客列表的最长等待时间，默认 8 秒即可。

4. 打印超时
   等待网页定位访客、打开打印窗口并确认状态的最长时间，默认 60 秒。

5. 机器人远程启动桥接
   建议保持 Yes。桥接没有运行时，机器人会通过 SSH 启动前台任务。

6. 前台 SSH 用户和端口
   填写已授权机器人公钥的 Windows 用户，端口通常为 22。
   示例用户：4paradigm

7. 机器人 SSH 私钥路径
   这是机器人容器内的路径，不是 Windows 路径。
   默认：/run/ql-printer-auth/frontdesk_bridge_ed25519

8. 目标打印机
   打开齿轮时会自动读取前台电脑已安装的打印机。请选择实际使用的标签打印机。

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
5. 如果画布无法读取打印机，先确认前台 IP、访问令牌、桥接状态和打印机驱动。

六、安全说明

不要删除 OpenSSH 的 authorized_keys，不要公开访问令牌或机器人私钥。
更换前台电脑后，需要重新授权该机器人的 SSH 公钥，并重新执行初始化。
