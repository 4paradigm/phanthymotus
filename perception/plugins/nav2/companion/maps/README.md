# G1 Nav2 地图目录

这里只保留说明，不提交真实地图。正式 Perception Compose 在 G1 上使用
`/opt/phanthy-motus/data/nav2/maps`，容器内挂载为 `/maps`。

每张已就绪地图占用一个独立目录：

```text
<map_name>/
├── map.yaml
├── map.pgm
├── map.posegraph
├── map.data
├── manifest.json
└── tags.json
```

- `map.yaml` / `map.pgm` 是 Map Server + AMCL 使用的 occupancy map。
- `map.posegraph` / `map.data` 是 SLAM Toolbox 可继续使用的 pose graph。
- `manifest.json` 记录 schema、地图状态、时间和四个核心文件的大小/SHA256。
- `tags.json` 保存地图坐标系中的语义点位。

建图时先写入同根目录的 `.mapping-*` staging 目录；只有四个核心文件
全部存在、非空且 `map.yaml` 合法后，才用同文件系统 `os.replace`
原子发布为 `<map_name>`。保存顺序固定为：

```text
save occupancy → pause SLAM → serialize pose graph → validate → atomic publish
```

地图名和 tag 名会拒绝空值、隐藏名、路径分隔符、控制字符和越界路径；
地图目录及核心文件不接受 symlink。`stop_mapping` 保存完成后由 runtime supervisor
自动切换 localization 并加载新地图；若初始位姿不确定，仍需显式全局重定位并由
owner 人工低速原地旋转帮助收敛。基础容器由正式 Perception Compose 管理，不配置
额外 crontab 或卡片私有自启脚本。
