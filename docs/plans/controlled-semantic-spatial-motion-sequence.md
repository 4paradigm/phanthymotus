# motion_sequence 全链路命名统一

## 范围

统一本仓导航端口、ROS topic、schema、launch/节点参数、状态字段、协议类型、
构造/发布函数与协议文件名为 motion_sequence。移除端口别名转换。
Driver 同事同步修改 loco 接收端，本 session 不改 Driver 仓库。

## 双端契约

- 端口：motion_sequence。
- ROS topic：/<namespace>/navigation/motion_sequence。
- schema：phanthy.navigation.motion_sequence.v1；ROS 类型仍 std_msgs/msg/String。
- 字段不变：nav_id、sequence、ttl_ms、issued_at_unix_ms、frame、nav_status、velocity、shadow_only、physical_execution，可选 reason。
- 5 Hz、RELIABLE KEEP_LAST(1)、TTL 最大 250 ms、终态零速及 Driver 执行仲裁不变。
- 新旧 schema 不混用、不双发。北京现有 Driver 拒收新 schema，必须同步升级并重接画布后测试。

## 验证与交付

运行协议/launch/统一卡片定向测试，再执行 ActuCore 回归。
本地和现场验收后才申请 BOT review；本次不触发 BOT、不单独部署新 ActuCore。

## 本地验证结果

- 协议/launch/统一卡片定向测试：54 passed、1 skipped、17 subtests passed。
- 最终路径调整后 ActuCore 全量：437 passed、2 skipped、72 subtests passed。
- 两个跳过项为本机无 OpenCV 和 Linux procfs；未进行 ROS 实机连通或运动验收。
- `git diff --check` 通过；ActuCore 内旧命名及错误的 nav2/motion_sequence 路径检索无命中。
- 已复核统一卡片、planning、ActuCore README 及协议文件，明确新 topic/schema 与双端同步升级要求。
- 本轮未提交、推送、部署或申请 BOT review。
