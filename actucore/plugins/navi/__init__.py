"""视觉导航卡片 —— 看到目标就朝它走过去。

包结构和 `plugins/vla/` 一样，按「卡片 / 行为 / 几何 / 协议」分文件：

    plugin.py   卡片：工具声明、生命周期、订阅、定时发布
    policy.py   控制律。纯函数，无 ROS —— 所有行为分支都在这里，都可测
    depth.py    深度解码与采样。纯函数，无 ROS
    odom.py     motus.odom/1 的读取侧（故意不跨仓库 import，见文件头）
"""

from .plugin import NaviPlugin

__all__ = ["NaviPlugin"]
