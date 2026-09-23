#!/usr/bin/env python3
"""How wrong the ego-motion compensation gets, and how fast.

**Reads sensors only. Issues no command and moves nothing.** You drive the
robot — from the dashboard's `loco` card, by hand, or by letting navi run — and
this records what happens.

── the question this answers ────────────────────────────────────────────────

`plugins/navi/track.py` keeps following a target through an occlusion by moving
its estimate with the robot's own twist. Two decisions rest on how well that
works, and neither has a number behind it yet:

  * `max_coast_s` (1.2 s) — how long a hidden target may keep driving the robot.
  * whether the local memory grid proposed in `docs/visual-navigation.md` §八 is
    viable at all. R1's own `motus.odom/1` declares `pose_drift: unbounded`,
    which is precisely the warning that says *do not build a map with this*. A
    few-second window may still be fine. "May" is not a number.

So this does not measure pose drift in the abstract. It measures **the error in
the quantity the tracker actually consumes**: after T seconds of coasting, how
far off is the predicted bearing and range of a stationary landmark?

And it measures it **against doing nothing**. If compensating with odometry is
no better than assuming the world stands still, the compensation is not earning
its complexity and the memory grid is dead on arrival. That comparison is the
headline number, not the raw error.

The propagation here calls `track.Tracker._predict` rather than reimplementing
it — the point is to measure what runs, and a second copy of the transform could
be right while the shipped one is wrong.

── the two modes ────────────────────────────────────────────────────────────

**`static`** — leave the robot standing still (standing, not lying: a lying
robot's estimator may not be running). Anything that accumulates is bias, which
on a legged velocity estimate is usually the dominant term and is the cheapest
thing to find. Needs no landmark and no motion.

**`landmark`** — the real test. Put something vop reliably detects where it will
stay in view, and drive the robot around it: turn in place, walk past it, turn
again. Turning matters most — yaw is what moves a target across the frame
fastest, and it is where the compensation earns its keep.

    docker exec -it phanthy-motus-actucore-1 \\
        python3 /work/tools/measure_odom_drift.py static --seconds 60
    docker exec -it phanthy-motus-actucore-1 \\
        python3 /work/tools/measure_odom_drift.py landmark --target chair --seconds 120
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
from plugins.navi import track as track_mod        # noqa: E402

ODOM_SUFFIX = "/state/odom"
DEPTH_SUFFIX = "/visual_depth"
OBJECTS_SUFFIX = "/objects"

# Horizons to report, in seconds. 1.2 is `max_coast_s`; 5.0 is the window the
# proposed memory grid would need.
HORIZONS = (0.5, 1.0, 1.2, 2.0, 5.0)


def _discover(node, suffix: str, explicit: str = "") -> str:
    if explicit:
        return explicit
    matches = [name for name, _ in node.get_topic_names_and_types()
               if name.endswith(suffix)]
    if not matches:
        raise SystemExit(f"没有找到以 {suffix} 结尾的话题。")
    if len(matches) > 1:
        print(f"[warn] {suffix} 匹配到多个，用第一个：{matches}", flush=True)
    return matches[0]


def _record(args, *, want_landmark: bool):
    """Spin for `--seconds`, returning (odom samples, landmark observations).

    Each odom sample is `(monotonic_s, (vx, vy, wz))`; each observation is
    `(monotonic_s, body-frame point)`.

    Receipt time rather than the message stamp: at 10 Hz over horizons of
    seconds the arrival jitter is far below what is being measured, and using
    one clock removes any question of skew between the two publishers.
    """
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                           ReliabilityPolicy)
    from sensor_msgs.msg import CompressedImage
    from std_msgs.msg import String

    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                     history=HistoryPolicy.KEEP_LAST, depth=2,
                     durability=DurabilityPolicy.VOLATILE)

    config = policy_mod.Config()
    rclpy.init()
    node = Node("navi_measure_odom")

    odom_topic = _discover(node, ODOM_SUFFIX, args.odom_topic)
    print(f"[info] 里程计：{odom_topic}", flush=True)

    odom: list = []
    observations: list = []
    latest_depth = [None]
    unmeasured = [0]

    def on_odom(message):
        try:
            sample = json.loads(message.data)
        except Exception:                                      # noqa: BLE001
            return
        from plugins.navi import odom as odom_mod
        twist = tuple(odom_mod.axis_of(sample, axis) for axis in ("vx", "vy", "wz"))
        if all(value is None for value in twist):
            unmeasured[0] += 1
            return
        # An unmeasured axis becomes 0 here **only** after the all-null check
        # above: a robot that reports nothing is a different case from one that
        # reports zero, and the warning at the end depends on the difference.
        odom.append((time.monotonic(),
                     tuple(0.0 if v is None else float(v) for v in twist)))

    def on_depth(message):
        try:
            latest_depth[0] = depth_mod.decode(bytes(message.data))
        except depth_mod.DepthError:
            pass

    def on_objects(message):
        try:
            payload = json.loads(message.data)
        except Exception:                                      # noqa: BLE001
            return
        match = None
        for obj in (payload.get("objects") or []):
            if args.target.lower() not in str(obj.get("name", "")).lower():
                continue
            if match is None or (float(obj.get("confidence") or 0)
                                 > float(match.get("confidence") or 0)):
                match = obj
        if match is None:
            return
        depth = ({"map": latest_depth[0], "bands": {}}
                 if latest_depth[0] is not None else None)
        range_m = policy_mod.target_distance(match, depth, config)
        if range_m is None:
            return
        bearing = policy_mod._bearing(match) * config.half_fov_rad
        observations.append((time.monotonic(),
                             track_mod.point_of(range_m, bearing)))

    node.create_subscription(String, odom_topic, on_odom, qos)
    if want_landmark:
        depth_topic = _discover(node, DEPTH_SUFFIX, args.depth_topic)
        objects_topic = _discover(node, OBJECTS_SUFFIX, args.objects_topic)
        print(f"[info] 深度图：{depth_topic}", flush=True)
        print(f"[info] 检测结果：{objects_topic}（目标 {args.target!r}）", flush=True)
        node.create_subscription(CompressedImage, depth_topic, on_depth, qos)
        node.create_subscription(String, objects_topic, on_objects, qos)

    print(f"[info] 记录 {args.seconds:.0f}s —— 现在开始操作机器人", flush=True)
    deadline = time.monotonic() + args.seconds
    last_report = 0.0
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
        now = time.monotonic()
        if now - last_report > 10.0:
            last_report = now
            print(f"      … 剩 {deadline - now:.0f}s  odom {len(odom)} 条"
                  + (f"  观测 {len(observations)} 条" if want_landmark else ""),
                  flush=True)

    node.destroy_node()
    rclpy.shutdown()

    if unmeasured[0]:
        print(f"[warn] {unmeasured[0]} 条 odom 六个轴全是 null —— 这台机器人"
              f"根本不报速度，补偿无从谈起", flush=True)
    if not odom:
        raise SystemExit("没有收到任何里程计。确认驱动的 loco_state 在发布。")
    return odom, observations


def _propagate(point, odom, start_s: float, end_s: float, config):
    """Where a stationary point should be now, given what we did in between.

    Calls the tracker's own `_predict`, on a track whose target velocity is
    zero: the measurement has to be of the transform that ships, not of a
    second copy that could be right while the first one is wrong.
    """
    tracker = track_mod.Tracker()
    tracker.track = track_mod.Track(
        mean=np.concatenate([point, np.zeros(2)]), cov=np.eye(4) * 1e-9)
    import bisect

    stamps = [entry[0] for entry in odom]
    first = bisect.bisect_right(stamps, start_s)
    last = bisect.bisect_right(stamps, end_s)
    previous = start_s
    twist = odom[max(0, first - 1)][1] if odom else (0.0, 0.0, 0.0)
    for index in range(first, last):
        stamp, twist = odom[index]
        tracker._predict(twist, stamp - previous, config, True)
        previous = stamp
    # The tail. Odometry arrives at 10 Hz and `end_s` is an observation's
    # timestamp, so there is almost always a fraction of a tick left over —
    # dropping it under-integrates every horizon by up to one sample, which at
    # R1's minimum turn rate is 0.1 rad of phantom error on a **perfect**
    # odometry. That would have been read as drift.
    if end_s > previous:
        tracker._predict(twist, end_s - previous, config, True)
    return tracker.track.position


def _wrapped(angle: float) -> float:
    """A bearing difference folded into (-pi, pi].

    `atan2` returns (-pi, pi], so two bearings either side of straight behind
    differ by almost 2*pi while being almost identical. Unwrapped, those land in
    the tail of every percentile and make a working odometry look catastrophic.
    """
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _percentiles(values, label, unit="", scale=1.0):
    if not values:
        return f"  {label:<10} —"
    ordered = sorted(abs(v) * scale for v in values)
    median = statistics.median(ordered)
    p90 = ordered[int(0.9 * (len(ordered) - 1))]
    return (f"  {label:<10} 中位 {median:7.3f}{unit}  "
            f"p90 {p90:7.3f}{unit}  最大 {ordered[-1]:7.3f}{unit}  "
            f"n={len(ordered)}")


# ── mode: static ─────────────────────────────────────────────────────────────

def run_static(args) -> None:
    odom, _ = _record(args, want_landmark=False)

    position = np.zeros(2)
    heading = 0.0
    previous = odom[0][0]
    speeds, rates = [], []
    for stamp, twist in odom[1:]:
        dt = stamp - previous
        previous = stamp
        if dt <= 0 or dt > 1.0:
            continue
        cos, sin = math.cos(heading), math.sin(heading)
        position += np.array([cos * twist[0] - sin * twist[1],
                              sin * twist[0] + cos * twist[1]]) * dt
        heading += twist[2] * dt
        speeds.append(math.hypot(twist[0], twist[1]))
        rates.append(twist[2])

    elapsed = odom[-1][0] - odom[0][0]
    print()
    print(f"静止 {elapsed:.1f}s，{len(odom)} 条里程计")
    print(f"  累计位移  {np.linalg.norm(position):.3f} m  "
          f"({position[0]:+.3f}, {position[1]:+.3f})")
    print(f"  累计转角  {math.degrees(heading):+.2f}°")
    print(f"  每 5s 折合 {np.linalg.norm(position) / max(1e-6, elapsed) * 5:.3f} m / "
          f"{math.degrees(heading) / max(1e-6, elapsed) * 5:+.2f}°")
    print()
    print(_percentiles(speeds, "瞬时线速", " m/s"))
    print(_percentiles(rates, "瞬时角速", " rad/s"))
    print()
    drift_5s = np.linalg.norm(position) / max(1e-6, elapsed) * 5
    if drift_5s < 0.05:
        print("结论：静止偏置可以忽略。局部记忆栅格的可行性取决于运动中的漂移，")
        print("      跑 landmark 模式。")
    else:
        print(f"结论：**静止时每 5 秒就凭空漂 {drift_5s:.2f} m**。这是偏置，不是")
        print("      噪声 —— 它会稳定地往一个方向累积，而运动中只会更差。")
        print("      在修掉它之前，5 秒窗口的局部记忆栅格不用考虑。")


# ── mode: landmark ───────────────────────────────────────────────────────────

def run_landmark(args) -> None:
    odom, observations = _record(args, want_landmark=True)
    if len(observations) < 20:
        raise SystemExit(
            f"只取到 {len(observations)} 条 {args.target!r} 的观测 —— 目标要一直"
            f"在视野里，vop 要在跑，深度图上那块要有读数")

    config = policy_mod.Config()
    import bisect

    results = {h: {"compensated": [], "naive": [], "range": []} for h in HORIZONS}
    stamps = [stamp for stamp, _ in observations]
    for now_s, observed in observations:
        for horizon in HORIZONS:
            target_s = now_s - horizon
            # The observation closest to `horizon` ago, if there is one near
            # enough to be worth comparing.
            position = bisect.bisect_left(stamps, target_s)
            candidates = [observations[i] for i in (position - 1, position)
                          if 0 <= i < len(observations)]
            if not candidates:
                continue
            past = min(candidates, key=lambda o: abs(o[0] - target_s))
            # One odometry tick. Looser than this and the pair being compared
            # is not really `horizon` apart, which at R1's minimum turn rate
            # shows up as a tenth of a radian of error that is the matching,
            # not the odometry.
            if abs(past[0] - target_s) > 0.12:
                continue
            predicted = _propagate(past[1], odom, past[0], now_s, config)

            observed_range, observed_bearing = track_mod.polar_of(observed)
            predicted_range, predicted_bearing = track_mod.polar_of(predicted)
            _, naive_bearing = track_mod.polar_of(past[1])

            results[horizon]["compensated"].append(
                _wrapped(predicted_bearing - observed_bearing))
            results[horizon]["naive"].append(
                _wrapped(naive_bearing - observed_bearing))
            results[horizon]["range"].append(predicted_range - observed_range)

    print()
    print(f"里程计 {len(odom)} 条，{args.target!r} 的观测 {len(observations)} 条")
    print()
    print("按滑行时长，预测位置与实测位置的差：")
    print("（中位数是结论所在。p90 和最大值里混着一个采样下限 —— 配对的两帧")
    print("  最多可能差一个里程计周期，转身时那就是 0.1 rad 上下的假误差。）")
    print()
    for horizon in HORIZONS:
        bucket = results[horizon]
        if not bucket["compensated"]:
            print(f"{horizon:.1f}s  （没有取到样本）")
            continue
        marker = "  ← max_coast_s" if abs(horizon - config.max_coast_s) < 0.05 else ""
        print(f"{horizon:.1f}s{marker}")
        print(_percentiles(bucket["compensated"], "方位(补偿)", " rad"))
        print(_percentiles(bucket["compensated"], "  归一化", "",
                           scale=1.0 / config.half_fov_rad))
        print(_percentiles(bucket["naive"], "方位(不补偿)", " rad"))
        print(_percentiles(bucket["range"], "距离", " m"))
        print()

    # ── the headline ────────────────────────────────────────────────────────
    coast = min(HORIZONS, key=lambda h: abs(h - config.max_coast_s))
    bucket = results[coast]
    if not bucket["compensated"]:
        print("结论：滑行时长上没有样本，没法下结论。把 --seconds 加长，")
        print("      并且确保过程中真的转了几次身。")
        return

    compensated = statistics.median(abs(v) for v in bucket["compensated"])
    naive = statistics.median(abs(v) for v in bucket["naive"])
    normalised = compensated / config.half_fov_rad

    print("=" * 66)
    print(f"在 max_coast_s = {config.max_coast_s}s 处：")
    print(f"  补偿后方位误差 {compensated:.3f} rad = {normalised:.3f}（归一化）")
    print(f"  不补偿         {naive:.3f} rad")
    print(f"  align_tol      {config.align_tol}（归一化），"
          f"arrive_align_tol {config.arrive_align_tol}")
    print()
    gain = naive / max(1e-4, compensated)
    if naive <= compensated * 1.2:
        print("结论：**补偿没有带来好处。** 里程计在这个时间尺度上已经不比")
        print("      「假设世界没动」更准，那么自运动补偿在这台机器人上是白")
        print("      做的，而局部记忆栅格（取舍八里那条）不用考虑了。")
        print("      先查里程计本身：跑一次 static 模式看有没有偏置。")
    elif normalised > config.align_tol:
        print(f"结论：补偿有效（比不补偿好 {min(gain, 999):.0f} 倍），")
        print(f"      但滑行到 {config.max_coast_s}s 时误差已经超过 align_tol，")
        print(f"      也就是说滑行结束时机器人会朝着一个足以触发转向的错误方位。")
        print(f"      建议把 max_coast_s 调小到误差还在 align_tol 以内的那个档。")
    else:
        print(f"结论：补偿有效（比不补偿好 {min(gain, 999):.0f} 倍），")
        print(f"      且 {config.max_coast_s}s 的滑行误差仍在 align_tol 以内。")
        five = results.get(5.0, {}).get("compensated")
        if five:
            error_5s = statistics.median(abs(v) for v in five)
            print(f"      5s 处方位误差中位 {error_5s:.3f} rad —— 局部记忆栅格的")
            print(f"      格子尺寸不应小于这个角度在工作距离上折合的横向误差")
            print(f"      （2 m 处约 {2 * math.tan(error_5s):.2f} m）。")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="量自运动补偿的误差。只读传感器，不发任何指令 —— 机器人由你来开。")
    parser.add_argument("mode", choices=("static", "landmark"))
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--target", default="chair",
                        help="landmark 模式：一个**不动的**参照物，vop 认得的名字")
    parser.add_argument("--odom-topic", default="")
    parser.add_argument("--depth-topic", default="")
    parser.add_argument("--objects-topic", default="")
    args = parser.parse_args()

    if args.mode == "static":
        run_static(args)
    else:
        run_landmark(args)


if __name__ == "__main__":
    main()
