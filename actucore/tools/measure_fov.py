#!/usr/bin/env python3
"""Measure the camera geometry the navi corridor is built on.

**Reads sensors only. Issues no command and moves nothing.** Safe to run on a
robot that is standing, lying down, or doing something else entirely.

── why this exists ──────────────────────────────────────────────────────────

`plugins/navi/depth.py::corridor` converts every depth pixel into a lateral
offset in metres, and the whole obstacle test rests on that conversion being
right. It rests on two things nobody has ever measured on this robot:

  1. `half_fov_rad` — a **configuration value**, not a reading. Every corridor
     width, every sidestep check and every bearing-to-angle conversion scales
     with `tan` of it. It is currently 0.55 rad because that is a plausible
     number for a 63° lens.

  2. **What the depth channel actually contains.** `corridor` computes
     `lateral = tan(theta) * depth`, which is correct if `depth` is the
     perpendicular distance to the image plane (z-depth) and wrong if it is the
     distance along the ray (range). Over a 31° half-field the two differ by
     17% — enough to put a corridor edge in the wrong place at exactly the
     angles where a shoulder is.

Both fall out of one measurement, which is why this tool does them together.

── the two methods ──────────────────────────────────────────────────────────

**`wall`** (preferred — needs no tape measure and no known object). Point the
robot at a flat wall, square on, filling the frame. Then:

    z-depth      d(u) is flat across the wall
    ray range    d(u) = d0 / cos(theta(u)),  rising towards both edges

The shape alone answers question 2. If it is ray range, the curve's steepness
gives `half_fov` directly, and this is the *only* method here that measures it
without a tape measure. If it is z-depth the curve carries no angular
information at all and you need `object` below — which is itself worth knowing,
and is reported rather than papered over.

A learned monocular depth model is under no obligation to produce either shape.
If the fit is poor against **both**, that is the finding: the corridor's pinhole
assumption does not hold for this depth source, and no value of `half_fov_rad`
will make it hold.

**`object`** (needs a tape measure). Put something of known width in view —
a box, a monitor, a suitcase; anything vop detects with a tight box. Then

    tan(half_fov) = (W/2) / (D * b)

where `b` is the detection's half-width in normalised image units. `D` comes
from the depth map unless `--distance` is given, so passing a measured distance
also cross-checks the depth scale.

Run it at two distances. If the answers differ by more than a few percent,
something other than the field of view is wrong — lens distortion, or a depth
scale that is not linear.

── running it ───────────────────────────────────────────────────────────────

    docker exec -it phanthy-motus-actucore-1 \\
        python3 /work/tools/measure_fov.py wall
    docker exec -it phanthy-motus-actucore-1 \\
        python3 /work/tools/measure_fov.py object --width 0.60 --target box

Topics are discovered by the same name suffixes the card uses, so nothing has
to be passed unless the discovery picks the wrong one.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, "/work")
sys.path.insert(0, __file__.rsplit("/", 2)[0])

from plugins.navi import depth as depth_mod        # noqa: E402
from plugins.navi import policy as policy_mod      # noqa: E402

DEPTH_SUFFIX = "/visual_depth"
OBJECTS_SUFFIX = "/objects"

# Columns to fit over, as a fraction of the frame. The outer edges of a
# monocular depth map are its worst region and a wall rarely fills the frame
# corner to corner.
FIT_SPAN = (0.15, 0.85)
# Rows to fit over: the middle of the frame, away from ceiling and floor.
FIT_ROWS = (0.35, 0.65)


# ── ROS plumbing ─────────────────────────────────────────────────────────────

def _discover(node, suffix: str, explicit: str = "") -> str:
    if explicit:
        return explicit
    matches = [name for name, _ in node.get_topic_names_and_types()
               if name.endswith(suffix)]
    if not matches:
        raise SystemExit(
            f"没有找到以 {suffix} 结尾的话题。确认 perception 的卡片在运行，"
            f"并且这个进程和它在同一个 ROS_DOMAIN_ID / DDS profile 下。")
    if len(matches) > 1:
        print(f"[warn] {suffix} 匹配到多个话题，用第一个：{matches}", flush=True)
    return matches[0]


def _collect(*, want_depth: bool, want_objects: bool, frames: int,
             depth_topic: str, objects_topic: str, timeout_s: float):
    """Gather `frames` samples off the bus. Returns (depth_maps, payloads)."""
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                           ReliabilityPolicy)
    from sensor_msgs.msg import CompressedImage
    from std_msgs.msg import String

    # BEST_EFFORT, for the same reason the card uses it: a reliable subscriber
    # does not match a best-effort publisher, and ROS reports that by silently
    # delivering nothing.
    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                     history=HistoryPolicy.KEEP_LAST, depth=2,
                     durability=DurabilityPolicy.VOLATILE)

    rclpy.init()
    node = Node("navi_measure_fov")
    depth_topic = _discover(node, DEPTH_SUFFIX, depth_topic) if want_depth else ""
    objects_topic = (_discover(node, OBJECTS_SUFFIX, objects_topic)
                     if want_objects else "")

    maps, payloads = [], []

    def on_depth(message):
        if len(maps) < frames:
            try:
                maps.append(depth_mod.decode(bytes(message.data)))
            except depth_mod.DepthError as error:
                print(f"[warn] 深度帧解码失败：{error}", flush=True)

    def on_objects(message):
        if len(payloads) < frames:
            try:
                payloads.append(json.loads(message.data))
            except Exception:                                  # noqa: BLE001
                pass

    if depth_topic:
        node.create_subscription(CompressedImage, depth_topic, on_depth, qos)
        print(f"[info] 深度图：{depth_topic}", flush=True)
    if objects_topic:
        node.create_subscription(String, objects_topic, on_objects, qos)
        print(f"[info] 检测结果：{objects_topic}", flush=True)

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if ((not want_depth or len(maps) >= frames)
                and (not want_objects or len(payloads) >= frames)):
            break
        rclpy.spin_once(node, timeout_sec=0.2)

    node.destroy_node()
    rclpy.shutdown()

    if want_depth and not maps:
        raise SystemExit(f"{timeout_s:.0f}s 内没有收到任何深度帧")
    if want_objects and not payloads:
        raise SystemExit(f"{timeout_s:.0f}s 内没有收到任何检测结果")
    return maps, payloads


# ── method: wall ─────────────────────────────────────────────────────────────

def _column_profile(maps) -> np.ndarray:
    """Median depth per column over the frames, for the rows we fit on."""
    top = int(depth_mod.HEIGHT * FIT_ROWS[0])
    bottom = int(depth_mod.HEIGHT * FIT_ROWS[1])
    stack = np.stack([m[top:bottom, :] for m in maps])
    with np.errstate(invalid="ignore"):
        return np.nanmedian(stack.reshape(-1, depth_mod.WIDTH), axis=0)


def _nrmse(observed, predicted, scale: float) -> float:
    """Residual against a prediction, as a fraction of the depth itself.

    Normalised RMS rather than R², because the discriminating case here is a
    **perfectly flat** profile: its variance is zero, so R² is 0/0. A flat wall
    reading flat is not an undefined fit — it is the answer.
    """
    mask = np.isfinite(observed) & np.isfinite(predicted)
    if mask.sum() < 20:
        return float("nan")
    residual = np.sqrt(np.mean((observed[mask] - predicted[mask]) ** 2))
    return float(residual / max(1e-9, scale))


def _draw_profile(observed, centre_depth: float, buckets: int = 24) -> None:
    """The depth across the frame, as a picture.

    Added after the first run on a robot: the fit said "neither hypothesis
    fits" and stopped there, which is true but not actionable — the operator
    cannot tell a depth source that violates pinhole geometry from a robot that
    is simply not pointed at a wall. The shape separates them at a glance. A
    wall is a smooth, symmetric curve; a room is lumpy and asymmetric.
    """
    edges = np.linspace(0, len(observed), buckets + 1).astype(int)
    values = []
    for index in range(buckets):
        chunk = observed[edges[index]:edges[index + 1]]
        chunk = chunk[np.isfinite(chunk)]
        values.append(float(np.median(chunk)) if chunk.size else float("nan"))

    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return
    low, high = min(finite), max(finite)
    span = max(1e-6, high - low)
    print()
    print(f"  横向剖面（左→右，{low:.2f}..{high:.2f} m）")
    for row in range(8, 0, -1):
        threshold = low + span * (row - 0.5) / 8
        line = "".join("█" if np.isfinite(v) and v >= threshold else " "
                       for v in values)
        print(f"  {low + span * row / 8:5.2f} |{line}")
    print(f"        +{'-' * buckets}")
    print(f"         左{' ' * (buckets - 4)}右")
    print("  一面正对的平墙是平滑对称的；房间是坑洼且不对称的。")


def measure_wall(args) -> None:
    maps, _ = _collect(want_depth=True, want_objects=False, frames=args.frames,
                       depth_topic=args.depth_topic, objects_topic="",
                       timeout_s=args.timeout)
    profile = _column_profile(maps)

    low = int(depth_mod.WIDTH * FIT_SPAN[0])
    high = int(depth_mod.WIDTH * FIT_SPAN[1])
    columns = np.arange(depth_mod.WIDTH)[low:high]
    observed = profile[low:high]
    valid = np.isfinite(observed)
    if valid.sum() < 50:
        raise SystemExit("墙面上有效深度太少，换一面纹理稍多的墙，或靠近一些")

    centre_px = (depth_mod.WIDTH - 1) / 2.0
    normalised = (columns - centre_px) / centre_px      # -1..1 across the span
    middle = len(observed) // 2
    centre_depth = float(np.nanmedian(observed[middle - 20:middle + 20]))
    edge_depth = float(np.nanmedian(
        np.concatenate([observed[:40], observed[-40:]])))
    ratio = edge_depth / max(1e-9, centre_depth)

    print()
    print(f"样本 {len(maps)} 帧，列 {low}..{high}，有效 {valid.sum()}/{len(columns)}")
    _draw_profile(observed, centre_depth=float(np.nanmedian(observed)))
    print(f"中心深度 {centre_depth:.3f} m   边缘深度 {edge_depth:.3f} m   "
          f"边缘/中心 = {ratio:.3f}")
    print()

    # ── hypothesis 1: z-depth. A flat wall reads flat. ──────────────────────
    flat = np.full_like(observed, centre_depth)
    error_flat = _nrmse(observed, flat, centre_depth)

    # ── hypothesis 2: ray range, d(u) = d0 / cos(theta). Fit half_fov. ──────
    best = None
    for half_fov in np.arange(0.20, 1.10, 0.002):
        angle = np.arctan(normalised * math.tan(half_fov))
        shape = 1.0 / np.cos(angle)
        mask = valid
        d0 = (np.sum(observed[mask] * shape[mask])
              / max(1e-9, np.sum(shape[mask] ** 2)))
        error = _nrmse(observed, d0 * shape, centre_depth)
        if best is None or (np.isfinite(error) and error < best[1]):
            best = (float(half_fov), float(error), float(d0))
    half_fov, error_range, d0 = best

    # The physically meaningful discriminator, stated as a number the operator
    # can sanity-check against the picture rather than as a fit statistic.
    span_edge = FIT_SPAN[1] * 2 - 1
    predicted_ratio = 1.0 / math.cos(math.atan(span_edge * math.tan(half_fov)))

    print("两种假设（残差是深度本身的百分比，越小越好）")
    print(f"  z-depth（墙读出来是平的）   残差 {error_flat * 100:5.2f}%   "
          f"预期 边缘/中心 = 1.000")
    print(f"  ray range（d0/cos θ）       残差 {error_range * 100:5.2f}%   "
          f"预期 边缘/中心 = {predicted_ratio:.3f}"
          f"   → half_fov {half_fov:.3f} rad（{math.degrees(half_fov) * 2:.1f}° 全视场）")
    print()

    configured = policy_mod.Config().half_fov_rad
    if error_flat < error_range * 0.7:
        print("结论：深度通道是 **z-depth**（到成像平面的垂直距离）。")
        print("      corridor 里的 `lateral = tan(θ) * depth` 是对的，不用改。")
        print("      但这个方法**量不出 half_fov** —— 平的曲线不含角度信息。")
        print(f"      请再跑一次 object 方法。当前配置 {configured} rad。")
    elif error_range < error_flat * 0.7:
        print("结论：深度通道是**沿射线的距离（range）**，不是 z-depth。")
        print("      **corridor 目前算错了**：它用的是 `lateral = tan(θ) * depth`，")
        print("      range 下应该是 `lateral = sin(θ) * depth`。31° 处差 17%。")
        print(f"      顺带量出 half_fov = {half_fov:.3f} rad（配置为 {configured}）。")
    else:
        print("结论：**两种假设都不像**。")
        print("      如果残差都很大，说明这个深度源不满足针孔几何 —— 那 corridor")
        print("      的整套换算就没有可信基础，换多少 half_fov_rad 都救不回来，")
        print("      得先搞清楚 visual_depth 输出的到底是什么量。")
        print("      如果只是区分不开，多半是墙没填满画面或没摆正，重试一次。")
    if not (0.5 < ratio < 2.0):
        print()
        print(f"[warn] 边缘/中心 = {ratio:.3f}，不像是一面正对着的平墙。")
        print("       摆正、贴近到墙填满画面，再测一次。")


# ── method: object ───────────────────────────────────────────────────────────

def measure_object(args) -> None:
    if not args.width:
        raise SystemExit("object 方法需要 --width：目标的真实宽度，米")

    maps, payloads = _collect(want_depth=args.distance is None, want_objects=True,
                              frames=args.frames, depth_topic=args.depth_topic,
                              objects_topic=args.objects_topic,
                              timeout_s=args.timeout)

    samples = []
    missing_bbox = 0
    for index, payload in enumerate(payloads):
        match = None
        for obj in (payload.get("objects") or []):
            if args.target.lower() in str(obj.get("name", "")).lower():
                if match is None or (float(obj.get("confidence") or 0)
                                     > float(match.get("confidence") or 0)):
                    match = obj
        if match is None:
            continue
        bbox = match.get("bbox_norm")
        if not bbox:
            missing_bbox += 1
            continue

        # `bbox_norm` is 0..1 of the frame; `position[0]` is -1..1. So the
        # box's angular span in those -1..1 units is twice its width in 0..1.
        span = 2.0 * (float(bbox[2]) - float(bbox[0]))
        if span <= 0:
            continue

        if args.distance is not None:
            distance = args.distance
        elif index < len(maps):
            distance = depth_mod.sample_box(maps[index], bbox)
        else:
            distance = None
        if not distance:
            continue

        # Both edges, rather than a half-width about the image centre:
        #
        #   tan(theta) = x * tan(half_fov)          x in -1..1
        #   W = D * (tan(theta_r) - tan(theta_l)) = D * tan(half_fov) * span
        #
        # so the object does **not** have to be centred. The half-width form
        # would have required it, because tan is not linear and an off-centre
        # object of a given width subtends a smaller span than a centred one.
        #
        # `D` here is the perpendicular distance. If the `wall` method says
        # this depth channel is ray range, this is a few percent optimistic at
        # the frame edge — one more reason to run `wall` first.
        samples.append((math.atan(args.width / (distance * span)), distance))

    if missing_bbox:
        print(f"[warn] {missing_bbox} 帧的检测没有 bbox —— vop 的 publish_bbox "
              f"可能是关的，这个方法需要它", flush=True)
    if not samples:
        raise SystemExit(
            f"没有取到任何可用样本：视野里要有 {args.target!r}，且 vop 要开 "
            f"publish_bbox，深度图上那块区域要有读数")

    angles = [a for a, _ in samples]
    distances = [d for _, d in samples]
    configured = policy_mod.Config().half_fov_rad
    median = statistics.median(angles)

    print()
    print(f"样本 {len(samples)} 帧，目标 {args.target!r}，真实宽度 {args.width} m")
    print(f"距离 中位数 {statistics.median(distances):.3f} m  "
          f"范围 {min(distances):.3f}..{max(distances):.3f} m"
          + ("（来自深度图）" if args.distance is None else "（由 --distance 给定）"))
    print()
    print(f"half_fov_rad 中位数 {median:.4f} rad  "
          f"（{math.degrees(median) * 2:.1f}° 全视场）")
    if len(angles) > 1:
        print(f"             范围   {min(angles):.4f}..{max(angles):.4f} rad  "
              f"标准差 {statistics.pstdev(angles):.4f}")
    print(f"             当前配置 {configured} rad")
    delta = (median - configured) / configured * 100
    print(f"             差 {delta:+.1f}%")
    print()
    if abs(delta) < 5:
        print("结论：配置值够用，走廊宽度的误差在 5% 以内。")
    else:
        print(f"结论：把 actucore/config.yaml 的 navi.half_fov_rad 改成 "
              f"{median:.3f}。")
        print(f"      走廊的横向换算与这个值的 tan 成正比，{delta:+.1f}% 的偏差"
              f"直接变成同量级的走廊宽度偏差。")
    if statistics.pstdev(angles) > 0.03:
        print()
        print("[warn] 各帧之间散得比较开。多半是检测框本身在抖 —— 换一个边缘")
        print("       清晰、检测框贴得紧的矩形目标（箱子、显示器）再测一次，")
        print("       并且在两个不同距离各测一次：如果两次结果差超过几个百分点，")
        print("       出问题的就不是视场角，而是镜头畸变或深度标定。")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="量 navi 走廊依赖的相机几何。只读传感器，不发任何指令。")
    parser.add_argument("method", choices=("wall", "object"))
    parser.add_argument("--width", type=float,
                        help="object 方法：目标的真实宽度，米")
    parser.add_argument("--target", default="box",
                        help="object 方法：vop 认得的目标名，默认 box")
    parser.add_argument("--distance", type=float,
                        help="object 方法：量出来的距离，米。不给就用深度图 —— "
                             "给了就等于顺带校验了深度的尺度")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--depth-topic", default="")
    parser.add_argument("--objects-topic", default="")
    args = parser.parse_args()

    if args.method == "wall":
        measure_wall(args)
    else:
        measure_object(args)


if __name__ == "__main__":
    main()
