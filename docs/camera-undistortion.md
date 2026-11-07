# 独立相机去畸变卡 undistort

`undistort` 是 Perception 中的多实例 processor，不需要模型下载。
它接收原始 JPEG，输出同一路校正后的 JPEG，供所有视觉消费者使用。
`visual_depth` 内部不再单独做去畸变。

## 画布接线

```text
camera_front → undistort ┬→ visual_depth ── 深度图/摘要 ─→ navi
                         ├→ vop ────────── objects ─────→ navi
                         └──────────────── RGB ─────────→ navi
```

三条分支都从同一张 undistort 卡取图。不要把原始 RGB 接到 VOP 或导航、
同时把校正图接到深度卡。深度发布为 640×480 时会通过现有继承逻辑缩放 K；
VOP 保留输入图尺寸，导航按声明的尺寸归一化检测框。
共享图像几何不等于严格帧同步：本卡保留输入 JPEG 的时间戳和 frame_id，
但下游模型频率不同，仍需检查导航自身的新鲜度和同步策略。

## 参数

每个实例设置 `calibration`、`fps`（默认 15）、`jpeg_quality`（默认 90）。
`calibration` 是 JSON 对象，包含准确的：

| 字段 | 含义 |
|---|---|
| `id` | 对应相机身份，例如 `deeprobotics/s10_pro/camera_front` |
| `width`、`height` | 标定对应的实际输入 RGB 像素尺寸 |
| `K` | 九个行优先数值：`[fx,0,cx,0,fy,cy,0,0,1]` |
| `distortion_model` | `plumb_bob`、`rational_polynomial`、`equidistant` |
| `D` | 与模型对应的准确畸变系数 |

系数顺序遵循 OpenCV：plumb_bob 为 `[k1,k2,p1,p2]` 或增加 `k3`；
rational_polynomial 为 `[k1,k2,p1,p2,k3,k4,k5,k6]`；equidistant 为
OpenCV fisheye 的 `[k1,k2,k3,k4]`。

`calibration={}` 表示使用上游 camera_info 的准确参数。缺少 K/D 或模型未知时
启动报错，不会默默原图透传。厂家参数范围或中值不能代替准确参数。
S10 当前驱动输出为 640×360，必须使用与这路图像对应的标定；若厂家参数对应
其他尺寸，需确认裁剪和缩放关系后转换，不能只根据分辨率猜测。

相机 ID、声明尺寸、实际帧尺寸都进行校验。已有 `perception/undistort`
处理记录的输入会拒绝再次校正，避免双重去畸变。

## 输出几何

先用原 K 进行 OpenCV remap，再裁出最大的全有效矩形。三路消费者共享
这一次裁剪，不会收到人工填充的黑边。输出 K 的主点随裁剪平移，D 为零，
宽高和视场角重新计算；原始参数及裁剪矩形保留在 vendor.undistortion。
物理相机 id 保留，pipeline 添加 `perception/undistort`。

每个实例有独立输出话题和缓存映射表。话题以 `/rgb` 结尾；现有导航通过
ROS 图上的 CompressedImage 类型识别这路 RGB，因此启动导航前应确认该话题
已被 ROS 发现。深度和检测话题仍遵循 `/visual_depth`、`/objects` 后缀。
队列只保留一帧，停止时先结束 worker，再移除和销毁
ROS 节点。解码或尺寸错误时停止发布该帧，并在 info 的 error 字段报告。

## 使用顺序

1. 取得准确参数；本卡不从测墙距离求解 K/D。用棋盘格标定见下节。
2. 停止下游导航、VOP、深度卡，记录原配置和标定结果。
3. 在画布中插入 undistort，填写参数；连接原始相机，启动并检查校正图。
4. 从同一校正输出连接三条分支，重新启动下游，使 camera_info 重新传递。
5. 清空旧深度采样，使用 `manual set`、a=1、b=0 重新采集多个实测距离。
   标定期间仅运行一张深度卡，避免原有标定器将同一系数应用于其他实例。
6. 用未参与拟合的距离验证，再把最终深度系数保存到深度卡配置。
7. 检查导航图中 RGB、检测框和深度的空间对应关系，并实测处理延迟。

更换参数或输入相机前必须停止去畸变实例；运行期间配置变更会被拒绝。
参数修改后也要重启下游并重新验证尺度，不自动沿用旧图像几何的系数。
配置可通过画布或 MCP `config` action 完成，使用固定 instance_id。

## 棋盘格标定

`perception/tools/calibrate_chessboard.py` 从一组 JPEG 帧求解 K/D，直接输出
本卡的 `calibration` JSON。在驱动容器里从 RTSP 源抓帧（分辨率高于驱动输出，
角点更准），拷回后运行：

```bash
docker exec <driver> ffmpeg -rtsp_transport tcp -i rtsp://<robot>:8554/video4 \
    -t 150 -vf fps=2 -q:v 2 /tmp/calib_front/front_%04d.jpg
```

```powershell
python perception/tools/calibrate_chessboard.py <frame_dir> --out <out_dir> \
    --cols 10 --rows 13 --square 0.018 \
    --camera-id deeprobotics/s10_pro/camera_front --target-width 640 --target-height 360
```

`--cols/--rows` 是内角点数（格数减 1），`--square` 是方格边长（米）。脚本剔除
模糊帧、相近姿态和离群帧，比较 plumb_bob、rational_polynomial、equidistant，
打印角点覆盖热力图（有 0 的格子要补拍），并把 K 按比例换算到 `--target-*`
的驱动输出尺寸（D 不变）。`calibration_card.json` 即本卡参数；`preview_*.jpg`
是原图与校正图并排，用于检查直线是否变直。采集要求：板子手持、每个姿态
停住约 1 秒、覆盖画面四角和四边并带 20–40° 倾斜；小板子距离不要超过 0.7 m。

### 云深处 S10 Pro 前置相机实测（2026-11-07）

相机为 Sunplus SPCA2650 UVC 模组，厂家节点 `usb_cam_multi_node` 以 1280×720
采集，RTSP `/video4` 输出 1024×576，驱动再缩放为 640×360。机器人上没有任何
厂家标定文件（`camera_info_url` 为空），厂家给出的参数范围对应 1600×1200
全幅模式，不能换算使用。两轮共 420 帧、使用 103 帧，RMS 0.213 px，
三种模型的 K 一致到 1 px 以内：

```json
{"id": "deeprobotics/s10_pro/camera_front", "width": 640, "height": 360,
 "distortion_model": "plumb_bob",
 "K": [314.963, 0, 336.651, 0, 314.934, 182.285, 0, 0, 1],
 "D": [-0.264916, 0.082800, -0.000180, 0.000031, -0.013244]}
```

对应 1024×576 流的 fx=fy=503.9、cx=538.9、cy=292.0，视场角 90.8°×59.5°。
桶形畸变在此 K 下校正后无黑边，有效区为全幅。注意驱动声明中的前置相机
`half_fov_rad=0.423`（全视场 48.5°）是早期墙面粗估，实际为 0.793 rad，
接入本卡后下游使用标定值；驱动中的粗估值应另行修正。其他三路相机尚未标定。

## 构建、测试和状态

本功能在 `phanthymotus/perception`，需要新的 Perception 镜像；仅更新
S10 driver 镜像不会带上它。`plugins.undistort.enabled` 默认启用卡片注册，
但不会自动订阅相机。已有部署需通过框架的 Perception 构建流程生成与
JetPack 匹配的镜像，保留模型挂载和现有配置，记录旧镜像以便回退。

本地测试覆盖三种畸变模型、有效区域裁剪、错误配置、上下游 K/D 继承、
JPEG 时间戳、多实例生命周期，以及同一帧送入实际 VOP/深度插件（假引擎）。
不等于 Jetson/TensorRT 实机精度和性能验收。目前未部署到机器人。

Windows 测试示例：

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
$env:PYTHONUTF8='1'
python -c "import cv2,pytest; cv2.setNumThreads(1); raise SystemExit(pytest.main(['perception/tests/test_undistortion_runtime.py','perception/tests/test_undistort_plugin.py','perception/tests/test_visual_depth_plugin.py','perception/tests/test_vop_plugin.py','perception/tests/test_camera_info_chain.py','perception/tests/test_bundle_dispatch.py','-q','--tb=short','--basetemp=.pytest-undistort-tmp']))"
```
