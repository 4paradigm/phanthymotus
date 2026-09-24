#!/usr/bin/env python3
"""What the obstacle logic is reading right now. One number per run.

**Reads sensors only. Issues no command and moves nothing.**

── why a tool for this ──────────────────────────────────────────────────────

`visual_depth` on r1_sz does not report metres. A table measured at 1.8 m with a
tape reads 5.8; the mapping is monotone but neither linear nor calibrated, and
the model's own re-calibration is log-affine rather than a scale factor.

navi's thresholds — `obstacle_stop_m`, `slow_distance_m`, `stop_distance_m` —
are compared directly against that reading while being *named* in metres. With
the reading running about 3.2x long, `obstacle_stop_m: 0.8` stopped the robot at
roughly 0.25 m. That was observed on the robot before it was predicted here, and
it is the whole reason this exists.

**The fix is not to calibrate the sensor. It is to calibrate the threshold.** A
monotone reading is enough to stop at a repeatable distance: put something where
you want the robot to stop, read the number, make that number the threshold.
Three tape measurements per robot, once, and no lens model, no depth model and
no fitting — which also makes it immune to the non-linearity, since only the
ordering is being relied on.

    # 把物体放到你希望它停下的距离，然后：
    docker exec -it embodied-actucore python3 /work/tools/read_depth_now.py --at 0.8

`--at` is what your tape says, and is used only to print the line to paste into
`config.yaml`; the measurement itself does not depend on it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, "/work")
sys.path.insert(0, __file__.rsplit("/", 2)[0])

from plugins.navi import depth as depth_mod        # noqa: E402
from plugins.navi import policy as policy_mod      # noqa: E402

DEPTH_SUFFIX = "/visual_depth"
OBJECTS_SUFFIX = "/objects"


def _discover(node, suffix: str, settle_s: float = 6.0) -> str:
    import rclpy

    deadline = time.monotonic() + settle_s
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
        matches = [name for name, _ in node.get_topic_names_and_types()
                   if name.endswith(suffix)]
        if matches:
            return matches[0]
    raise SystemExit(f"{settle_s:.0f}s 内没有找到以 {suffix} 结尾的话题")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="读出避障逻辑当前看到的数。只读传感器，不发任何指令。")
    parser.add_argument("--at", type=float,
                        help="尺子量到的真实距离（米），只用来打印可粘贴的配置行")
    parser.add_argument("--frames", type=int, default=8)
    args = parser.parse_args()

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                           ReliabilityPolicy)
    from sensor_msgs.msg import CompressedImage
    from std_msgs.msg import String

    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                     history=HistoryPolicy.KEEP_LAST, depth=2,
                     durability=DurabilityPolicy.VOLATILE)
    rclpy.init()
    node = Node("read_depth_now")
    depth_topic = _discover(node, DEPTH_SUFFIX)
    objects_topic = _discover(node, OBJECTS_SUFFIX)

    maps: list = []
    detections: list = []
    node.create_subscription(
        CompressedImage, depth_topic,
        lambda m: maps.append(depth_mod.decode(bytes(m.data))), qos)
    node.create_subscription(
        String, objects_topic,
        lambda m: detections.append(json.loads(m.data)), qos)

    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline and len(maps) < args.frames:
        rclpy.spin_once(node, timeout_sec=0.2)
    node.destroy_node()
    rclpy.shutdown()
    if not maps:
        raise SystemExit("没有收到深度帧")

    # Median across frames: one frame of a monocular depth map is noisy, and the
    # threshold being set from it will live in a config file for a long time.
    depth_map = np.nanmedian(np.stack(maps), axis=0)
    config = policy_mod.Config()
    keepout = config.half_width_m + config.clearance_margin_m

    clearance, coverage = depth_mod.corridor(
        depth_map, half_width_m=keepout, half_fov_rad=config.half_fov_rad,
        reference_m=config.obstacle_stop_m)
    bands = depth_mod.nearest_by_band(depth_map)
    valid = depth_map[np.isfinite(depth_map)]

    print()
    print(f"深度帧 {len(maps)}，检测帧 {len(detections)}")
    print(f"全图读数 {valid.min():.2f}..{valid.max():.2f}   5 分位 "
          f"{np.percentile(valid, 5):.2f}")
    print()
    print("—— 避障逻辑实际用到的数 ——")
    print(f"  走廊 clearance   {clearance}      （半宽 {keepout:.2f}，这是 "
          f"obstacle_stop 要比较的量）")
    print(f"  走廊 coverage    {coverage}")
    print(f"  三等分 nearest   {bands}")

    if detections:
        newest = detections[-1].get("objects") or []
        if newest:
            print()
            print("—— vop 看到的东西，以及 navi 会算出的距离 ——")
            source = {"map": depth_map, "bands": bands}
            for obj in sorted(newest, key=lambda o: -float(o.get("confidence") or 0))[:5]:
                distance = policy_mod.target_distance(obj, source, config)
                name = obj.get("name")
                bearing = policy_mod._bearing(obj)
                print(f"  {name:<22} 方位 {bearing:+.3f}  距离读数 {distance}")

    if args.at and clearance:
        print()
        print("=" * 62)
        print(f"尺子说 {args.at} m，走廊读到 {clearance}"
              f"（相差 {clearance / args.at:.2f} 倍）")
        print()
        print("把它贴进 actucore/config.yaml 的 navi 段 —— 注意这个数**不是米**，")
        print("是 visual_depth 的读数；名字保留 _m 只是为了不改代码，值必须按本机量。")
        print(f"    obstacle_stop_m: {clearance}")
        print()
        print("同样的办法再量两个点：希望开始减速的距离 → slow_distance_m，")
        print("希望停在目标前的距离 → stop_distance_m。三次尺子，一台机器人一次。")


if __name__ == "__main__":
    main()
