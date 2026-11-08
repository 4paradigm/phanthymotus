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
**`calibration` 通常留空 `{}`**，卡按下面的顺序自动选参数，`info` 的
`calibration_source` 字段说明实际用的是哪一种：

| 顺序 | 来源 | `calibration_source` |
|---|---|---|
| 1 | 卡片里手填的完整对象 | `manual` |
| 2 | 上游相机声明里的 K/D（`distortion_model` 为下表三种之一） | `upstream` |
| 3 | 内置标定表 `undistortion_runtime.CALIBRATIONS`，按上游声明的相机 id 查 | `table:<id>` |

都没有时启动报错，错误里写明是哪台相机、怎么补，不会默默原图透传，也不会
猜内参。上游声明了受支持的模型但数值无效时直接报错，不会被表悄悄替换。
厂家参数范围或中值不能代替准确参数。

所以四路相机在画布上各放一张 `undistort`、各接一路，不用填任何参数。
只有要覆盖某台机器人的参数时才手填，字段如下：

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

**内置表按相机型号共用。** 驱动规范要求相机 id 在同型号每台机器人上相同，
所以一台机器人上测的参数会作为同型号所有机器人的默认值。同型号镜头差异
一般很小；某台校正效果不好时，单独标定后手填覆盖即可。每行都带
`measured_on`、`rms_px`、`frames_used` 等出处字段，并由测试逐行校验。

S10 当前驱动输出为 640×360，表中参数就是这一尺寸；若驱动输出尺寸变化，
K 要按比例重新换算（D 不变），标定脚本可以直接重新输出。

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

1. 相机没有在内置表里时先标定（见下节）。本卡不从测墙距离求解 K/D。
2. 停止下游导航、VOP、深度卡，记录原配置。
3. 在画布中插入 undistort，`calibration` 留空；连接原始相机，启动，
   确认 `calibration_source` 和校正图（直线变直）。
4. 从同一校正输出连接三条分支，重新启动下游，使 camera_info 重新传递。
5. 深度卡的标定来源保持默认 `auto (by camera)`：相机和深度模型在预设表里有
   对应行时自动套用（见下文"深度标定"）。没有对应行时需要实测距离标定。
6. 检查导航图中 RGB、检测框和深度的空间对应关系，并实测处理延迟。

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
    --camera-id deeprobotics/s10_pro/camera_rear --camera-desc "..." \
    --target-width 640 --target-height 360
```

`--cols/--rows` 是内角点数（格数减 1），`--square` 是方格边长（米）。脚本剔除
模糊帧、相近姿态和离群帧，比较 plumb_bob、rational_polynomial、equidistant，
打印角点覆盖热力图（有 0 的格子要补拍），并把 K 按比例换算到 `--target-*`
的驱动输出尺寸（D 不变）。输出：

- `table_entry.json`：一行内置表，含出处字段，直接加入
  `undistortion_runtime.CALIBRATIONS`；
- `calibration_card.json`：只想在某台机器人上手填覆盖时用；
- `preview_*.jpg`：原图与校正图并排，检查直线是否变直；
- `report.json`：三种模型的完整结果和逐帧误差。

合格标准：RMS < 0.5 px，热力图没有 0 格，预览直线变直。采集要求：机器人
站立，板子手持，每个姿态停住约 1 秒，覆盖画面四角和四边并带 20–40° 倾斜，
最后专门压低覆盖画面下三分之一；小板子距离不要超过 0.7 m。采样前确认画面
没有花屏。

### 云深处 S10 Pro 四路相机实测（2026-11-07 / 08）

四路都是 Sunplus SPCA2650 UVC 模组，厂家节点 `usb_cam_multi_node` 以 1280×720
采集，RTSP 输出 1024×576，驱动再缩放为 640×360。机器人上没有任何厂家标定
文件（`camera_info_url` 为空），厂家给出的参数范围对应 1600×1200 全幅模式，
不能换算使用。四路都在 1024×576 RTSP 流上用棋盘格标定，机器人站立，结果
（640×360，plumb_bob）都已在内置表里：

| 相机 | RTSP | fx / fy | cx / cy | k1 | 帧数 | RMS (px) | 水平视场 |
|---|---|---|---|---|---|---|---|
| 前 `camera_front` | `/video4` | 315.0 / 314.9 | 336.7 / 182.3 | -0.265 | 103 | 0.213 | 90.8° |
| 后 `camera_rear` | `/video0` | 312.8 / 312.7 | 308.9 / 183.4 | -0.262 | 103 | 0.208 | 91.3° |
| 左 `camera_left` | `/video2` | 314.2 / 314.2 | 311.7 / 192.5 | -0.259 | 106 | 0.222 | 91.0° |
| 右 `camera_right` | `/video6` | 314.4 / 314.3 | 327.5 / 177.9 | -0.262 | 98 | 0.219 | 91.0° |

每路三种畸变模型的 K 相差不到 1.3 px；四路焦距相差不到 1%，符合同型号镜头。
完整 K/D 和出处字段见 `undistortion_runtime.CALIBRATIONS`。桶形畸变在这些 K
下校正后都没有黑边，有效区为全幅；校正图里桌沿、门框、玻璃墙边都变直。

两点说明：

- 每路原始画面有一侧边缘超出畸变多项式的单调范围（约 59°，前、右在左边，
  后、左在右边）。这不影响校正输出：remap 从校正图反查原图，超出部分在这个
  K 下本来就落在画面外。但原始画面最边缘几列不能用单一针孔角度描述。
- 主点各不相同（cx 相差约 28 px），说明厂家 1280×720 → 1024×576 的裁剪并非
  对每路都严格居中，或者镜头装配有偏差。所以四路不能共用一组参数。

驱动原先声明的前置 `half_fov_rad=0.423`（48.5°）是墙面粗估，推出的焦距是实测
的 2.26 倍；已在 phanthymotus-driver#501 改为 0.7926。另外三路在驱动里仍是
`unknown`，接入本卡后下游拿到的是标定值。

**已知的硬件现象**：右侧相机挂在 PCIe 扩展的 USB 控制器（5 号总线）上，曾在
开机时 USB 掉线（`uvcvideo: Failed to initialize the device`，`USB disconnect`），
厂家服务报 `Failed to init camera usb_camera_4` 并重试 10 轮后放弃，RTSP
`/video6` 返回 404。整机断电重启后恢复。若反复出现，需检查线缆或扩展卡。

### 深度标定（DepthART）

S10 前置相机使用 DepthART 后端。YOLO26 深度在这台相机上不可用：白墙上 15 帧
内读数跳动 20–26%，远近顺序反转。DepthART 的公制尺度来自 K，接入本卡后读数
稳定（散布 3–6%），但仍有随距离变化的比例偏差，需要两参数修正
`m = p^a · e^b`。

前置相机的修正已作为预设 `DeepRobotics S10 Pro front (DepthART)` 加入
`visual_depth.CALIBRATION_PRESETS`：`cal_a=1.36`、`cal_b=-0.92`，16 个站立、
正对墙面的点（0.6–4.0 m，两次开机会话），拟合平均误差 7.8%；未参与拟合的
0.7 / 1.3 / 2.3 m 验收误差 4.0% / 2.4% / 3.1%。深度卡标定来源保持
`auto (by camera)` 即自动选用。

这行预设声明了 `requires_pipeline: ["perception/undistort"]`：它是在校正后的
画面上测的，只有上游声明里记录经过了 `perception/undistort` 才会被选中。深度
卡直接接原始相机时退回模型默认标定，`info` 的 calibration 显示
`model-default (auto:needs-undistort(...))`，提示要改接线。

指数项需要离线在 log 空间拟合。测量和使用时的注意事项：

- 机器人**站立**。趴下时画面几乎没有地面，模型尺度明显偏大，同一位置差 30%。
- 画面里要有地面。相机离地约 35 cm，0.6 m 以内地面出画，读数不稳；
  导航的停障阈值不要设在 0.6 m 以内。
- 用平整实心的平面取样（墙、门、大板）。镂空物体（椅子）和低于相机的物体
  会让中央取样框混进背景，读数无效。
- 摘要里的 `nearest_by_region` 是各区域最近的像素，画面下半有地面时它是脚前
  地面的距离，不是墙距；验收墙距要取中央框的中位数。
- 每次采样前确认画面没有花屏。

其他三路相机的深度修正需要各自实测后再加预设。

## 构建、测试和状态

本功能在 `phanthymotus/perception`，需要新的 Perception 镜像；仅更新
S10 driver 镜像不会带上它。`plugins.undistort.enabled` 默认启用卡片注册，
但不会自动订阅相机。已有部署需通过框架的 Perception 构建流程生成与
JetPack 匹配的镜像，保留模型挂载和现有配置，记录旧镜像以便回退。

本地测试覆盖三种畸变模型、有效区域裁剪、错误配置、上下游 K/D 继承、
JPEG 时间戳、多实例生命周期、参数来源优先级（手填 / 上游 / 内置表）、四路
实例各自查表、内置表逐行校验、深度预设只在经过 undistort 时生效，以及同一帧
送入实际 VOP/深度插件（假引擎）。

真机状态：前置相机的去畸变和 DepthART 深度标定已在 S10 Pro 上用 PR 测试镜像
验证（手填参数）。自动查表和另外三路的去畸变需用新镜像再验证；另外三路的
深度修正尚未测量。不等于导航验收。

Windows 测试示例：

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
$env:PYTHONUTF8='1'
python -c "import cv2,pytest; cv2.setNumThreads(1); raise SystemExit(pytest.main(['perception/tests/test_undistortion_runtime.py','perception/tests/test_undistort_plugin.py','perception/tests/test_visual_depth_plugin.py','perception/tests/test_vop_plugin.py','perception/tests/test_camera_info_chain.py','perception/tests/test_bundle_dispatch.py','perception/tests/test_depthart_runtime.py','-q','--tb=short','--basetemp=.pytest-undistort-tmp']))"
```
