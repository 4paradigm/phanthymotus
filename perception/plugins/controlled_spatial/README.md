# controlled_spatial Perception 卡片

`controlled_spatial` 是 Perception Bundle 中的单实例 `processor` 卡片，不属于 G1 Driver Bundle。

本目录当前只冻结卡片归属、端到端方案和上海 G1 固定只读检查入口；尚未增加插件实现、Bundle loader 或默认配置。先设计再实现，避免在旧 Driver 卡片上继续叠加算法与状态。

## 入口

- [传统导航整体方案](传统导航整体方案.html)
- [上海 G1 固定只读检查](scripts/shanghai-ready-check.sh)
- [shell 回归](tests/test-shells.sh)

固定检查命令：

```bash
cd perception/plugins/controlled_spatial
./scripts/shanghai-ready-check.sh
```

该脚本无参数，固定检查 `g1-sh-wifi / ubuntu / eth0`。它委托同一 workspace 中 `phanthymotus-driver` 的底层只读验收脚本，不执行部署、容器启停、建图或运动。

## 卡片归属

| 范围 | 正确 owner |
|---|---|
| `controlled_spatial` schema、14 个业务 action、状态机、地图/POI、重定位、全局引导、EGO 编排 | Perception 卡片 |
| MID360 / IMU / RGB / Depth / 机身状态 topic | G1 Driver sensor cards |
| Unitree DDS 适配、主运控、高层有限时长运动与 SmartMotion 安全门 | G1 Driver |
| 导航 episode 录制 | Perception 卡片内部旁路，失败不得影响导航 |

FAST-LIVO2、重定位、全局规划和 EGO 是 perception 算法面；它们可作为卡片 companion sidecars 运行。G1 Driver 不承载导航任务状态，也不能让规划器直接取得运动权限。

## Perception 路由约束

当前 `PerceptionBundle` 会按工具名的第一个 `_` 拆分 PREFIX。为保持用户看到的工具名严格为 `controlled_spatial`，实现时使用：

```python
class ControlledSpatialPlugin:
    PREFIX = "controlled"

    def get_tools(self):
        return [{"name": "spatial", "type": "processor", ...}]
```

Bundle 对外组合为 `controlled_spatial`，并将调用路由到 `ControlledSpatialPlugin.dispatch("spatial", args)`。不得直接使用 `PREFIX="controlled_spatial"`，否则当前路由会把它错误拆成 `controlled`。

业务 `action` enum 必须保持用户冻结的 14 项，不把框架内部 `info/config/start/stop` 混入业务动作。`info` 等生命周期调用需要在 Perception adapter 中作为内部控制路径处理；具体实现前先补契约测试。

## 迁移边界

- 旧 `phanthymotus-driver/unitree/g1/controlled_spatial.py` 只作为兼容参考，不能继续充当新卡片正本。
- 新 Perception 卡片注册前，目标部署必须避免 Driver 与 Perception 同时暴露同名 `controlled_spatial`。
- Driver 仓保留硬件传感器适配、最终安全执行器以及底层验收脚本。
- 算法镜像、地图/定位、规划编排和卡片设计逐步迁入本目录；迁移必须按可构建、可回滚的小步骤进行，不在本设计提交里一次性搬动运行代码。

## 当前验证

- HTML 为单文件，无 CDN、远程字体或运行时网络依赖。
- shell 回归使用 fake SSH 真实调用固定入口，并断言远端命令不包含部署写操作。
- 本次不连接机器人，不修改 Perception loader/config，不宣称卡片已经注册或具备真机导航能力。
