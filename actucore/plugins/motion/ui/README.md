# 工作台前端源码

动作学习与仿真调试仍是两张卡片，使用一份 `simulation.js` 实现相机、真实 MuJoCo 帧、控制权、录制和回放。`motion.html` 保留动作、版本、训练与评估流程，`simulation.html` / `simulation-shell.js` 提供会话列表和调试检查器。`design.css` 与 `workbench-v4.css` 是共享样式。

`connections.js` 负责两卡共用的环境引用目录和公开名称。候选只来自成功的安装任务；检查任务不是可连接环境。当前绑定可以单独查询，历史接口不可用时仅使用确实返回的最新任务和已有绑定。用户点击“连接此环境”后，通过 host `config` 保存 `environment_job`，准备任务成功不会自动切换。服务器名称、环境类型及准备时间来自公开元数据，凭证不进入该目录。

两份交付文件都是自包含 HTML，运行时不依赖相邻仓库、CDN、外部脚本或 iframe 网络请求。修改此目录后生成产物：

```sh
python3 actucore/tools/build_motion_workbenches.py \
  --driver-output /path/to/phanthymotus-driver/simulator/mujoco/workbench.html
```

仅构建 Motion 时省略 `--driver-output`。发布 Driver 时必须显式指定产物路径，随后使用同一命令加 `--check` 检查两份产物。HTML 注释记录共享组件 SHA-256；Driver Dockerfile 继续复制生成后的单个 HTML，不需要安装构建器。

每个 iframe 都生成独立 `view_id`，读取不获取控制权。显式接管携带最近 `control_epoch`；运行、重置、扰动、录制、结束均携带当前 epoch。控制中的可见窗口每 5 秒续约，服务端超时暂停。离开调试会话会显式暂停并交出当前控制；观看者离开不影响控制者。主应用关闭弹窗没有可靠回调，依赖服务端超时，不能声称立即暂停。

训练监控默认在短窗口显示四个完整机器人，支持切换后一组或单实例；采样画面与学习阶段、训练心跳分别显示。完成时间来自任务结束时间，结束任务不再显示不断增长的“最近更新”。
