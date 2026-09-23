#!/usr/bin/env python3
"""
plugins/visual_depth.py — VideoDepthPerceptionPlugin: monocular depth from a plain RGB camera.

Subscribes to image/jpeg topics, runs a prebuilt YOLO26-depth TensorRT engine,
and publishes two things per frame:

    {input}/visual_depth          image/depth-zlib   for the dashboard's renderer
    {input}/visual_depth_summary  data/json          for the agent

Named after the tool, not after what they carry — see output_topics_for for why
`{input}/depth` could not stay.

Most robots in this fleet carry only an RGB camera — the RealSense on the
Realman RM75 is the exception — so this gives the rest a depth map without new
hardware.

Two constraints worth knowing before changing anything here:

* **640x480 is not a suggestion.** agent-core's DepthZlibRenderer
  (web/js/renderers/camera.js) hardcodes a 640x480 canvas and returns early
  when the decompressed buffer is shorter than 640*480 samples. A depth map
  published at the model's native resolution renders as a blank panel with
  nothing logged anywhere. Everything is resampled to 640x480 before publishing.

* **The output is metres, straight out of the engine.** This file used to claim
  the opposite — that the numbers were a relative scale until someone ran
  `model.calibrate()` — and labelled every payload `"scale": "relative"`. That
  was wrong, and it is the more dangerous direction of wrong: an agent told the
  distances are meaningless will not use them.

  The head predicts a relative log-depth field, but the metric transform is
  applied *inside* `Depth.forward` (`depth.pow(cal_a) * cal_b.exp()`, ultralytics
  `nn/modules/head.py`) **before** the export branch — so it is baked into the
  exported ONNX and into our TensorRT engine. The released yolo26n-depth weights
  ship with that fit already done (cal_a=1.0, cal_b=-0.1938).

  Measured on Orin5 against the reference `.pt`, same photos: a landscape gives
  2.4–35.7 m here vs 2.6–48.1 m there; ultralytics' bus.jpg gives 1.3–16.3 m vs
  2.2–17.8 m. Metres, with the error you would expect from fp16 at 640.

  What `model.calibrate()` buys is a refit for *your* camera. Until that is done
  these are metres from a general-purpose fit — good enough to compare and to
  reason about, not survey-grade. `cal_a` / `cal_b` in the config apply such a
  refit on top of the engine's own, using ultralytics' own parameterisation so a
  fit obtained there can be pasted here unchanged.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
import zlib
from typing import Optional

import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from plugins.image_input import BadInput, load_image_bytes
from utils.ros_lifecycle import dispose_node

log = logging.getLogger(__name__)

# Fixed by the dashboard renderer — see the module docstring.
DEPTH_WIDTH = 640
DEPTH_HEIGHT = 480

# Where a topic-less instance publishes. There is no input topic to derive an
# output from, so it is fixed — same idea as vop's DEFAULT_OUTPUT_TOPIC.
DEFAULT_DEPTH_TOPIC = "/perception/visual_depth"
DEFAULT_SUMMARY_TOPIC = "/perception/visual_depth_summary"
_DEFAULT_INSTANCE = "_default"


def output_topics_for(input_topic: Optional[str]) -> tuple[str, str]:
    """The one place the two output topics are derived from the input.

    Named after this tool, which is the convention everything else derived
    follows — `{input}/asr`, `{input}/tts`. These two were named after what they
    carry instead (`/depth`, `/depth_summary`), and on a robot whose camera
    already publishes its own depth map that produced a straight collision:
    visual_depth fed by `/nvidia_desktop/camera/head` derived
    `/nvidia_desktop/camera/head/depth`, which is exactly where the Tianyi
    driver's own `camera_depth` sensor publishes.

    Nothing detected it. The two producers simply took turns re-registering the
    topic under their own formats — `image/depth-zlib` against `image/depth-z16`
    — and every flip tore down and rebuilt the dashboard's subscription against
    a message type the other one was not using. 303 rebuilds later the panel had
    never received a frame, while both producers were publishing perfectly well.
    It read as visual_depth being stuck on its first frame.

    `_summary` is a sibling rather than a child (`…/visual_depth_summary`, not
    `…/visual_depth/summary`) so that neither topic is a path prefix of the
    other, which keeps prefix-matching anywhere downstream from confusing them.
    """
    if input_topic:
        return f"{input_topic}/visual_depth", f"{input_topic}/visual_depth_summary"
    return DEFAULT_DEPTH_TOPIC, DEFAULT_SUMMARY_TOPIC

_LOW_LAT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=2,
    durability=DurabilityPolicy.VOLATILE,
)

_PUB_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=DurabilityPolicy.VOLATILE,
)

# Fraction of the frame the `center` target box covers, per axis. Small enough
# that a calibration target fills it, large enough to average over noise.
_TARGET_BOX = 0.2

CALIBRATION_REGIONS = ("center", "left", "right", "full")

# A region whose interquartile spread exceeds this fraction of its own median
# is not one surface at one distance, whatever the operator believes.
_FLATNESS_LIMIT = 0.15

# A sample whose own error after the fit exceeds this is arguing with the rest.
# Most often a typo (2 for 20) or a reading taken facing something else.
_OUTLIER_PCT = 25.0

# How many recent frames one calibration sample is drawn from. At the card's
# default 2 fps this is about seven seconds of standing still, which is what
# the operator is doing anyway while holding a tape measure.
#
# Measured on r1_sz, robot stationary, same wall: fourteen consecutive frames
# spanned 1.68-2.09 m — 22% peak to peak, 6% standard deviation. One frame is
# a draw from that; the median of fourteen is worth about three times less
# scatter, which is the difference between a calibration and a coin toss.
CALIBRATION_FRAMES = 15

# Frame-to-frame spread, as a fraction of the median, past which the sample is
# reported as too noisy to build a calibration on. It does not refuse — the
# number is still the best estimate available — but it says so, because a fit
# whose inputs move by this much cannot support a two-parameter model and the
# temptation to add one is real.
_SCATTER_LIMIT = 0.20

# Distinguishes "the config had no cal_a" from "the config had cal_a = None",
# so a reset can put the dict back exactly as it found it.
_ABSENT = object()

CALIBRATION_PROCEDURE = (
    "把机器人开到一面平整的墙（或任何平面）正前方，让墙尽量正对、填满画面中央，"
    "用卷尺量出镜头到墙的真实距离，调用 calibrate 填进 distance_m。"
    "然后后退，换 1 米、2 米、3 米各做一次 —— 距离拉开才看得出误差是固定倍数还是随距离变化。"
    "量错了用 reset_calibration 清空重来。"
)



# Fits measured on a real camera, so that a robot of a known kind does not
# have to start from the engine's general-purpose one.
#
# **Every entry carries where it came from, and that is not decoration.** A
# preset is a number that looks like a fact, which is exactly what the engine's
# own `model-default` is — and applying that one to r1_sz's ultra-wide lens made
# it report 1.0 m as 3.2 m, which in turn made the navigation card stop 0.2 m
# from obstacles while believing it had stopped at 0.8. A preset chosen for the
# wrong camera fails the same way and just as quietly.
#
# Named for the robot, because that is what the person choosing one knows; the
# camera it is actually a property of is recorded in `camera` below. Two robots
# of the same model share a preset only because they share a lens, so a variant
# that ships a different camera needs its own entry rather than this one.
# The two non-preset choices in the same dropdown, so the field says what is in
# effect instead of leaving it to be inferred.
#
# An empty option rendered as a blank row in the dialog — a value you cannot
# see is a value you cannot choose deliberately, which is the same shape as
# every other silent default this card has been bitten by. "" is still accepted
# so an existing config keeps working, but it is not offered.
CAL_NONE = "no calibrate"
CAL_MANUAL = "manual set"

CALIBRATION_PRESETS = {
    "Unitree R1": {
        "cal_a": 1.0,
        "cal_b": -0.9753,
        "camera": "Unitree R1 主相机 1280x720（超广角，实测约 102° 全视场）",
        "measured_on": "r1_sz, 2026-09-23",
        "samples": 5,
        "range_m": "1.0–2.8",
        "residual_pct": 20,
        "note": "单帧取样拟合的 —— 多帧平均是这次之后才加的，同一距离先后两次"
                "测量曾差 31%。整体比例可信（5 个样本在 log 空间取均值），"
                "逐点残差约 ±20%。想要更准就用现在的 calibrate 重量两三个点。",
    },
}


TOOLS = [
    {
        "name": "visual_depth",
        "type": "processor",
        "multiInstance": True,
        "description": "视觉深度 — 用一个普通 RGB 摄像头估计每个像素的距离（单位：米），输出深度图与左/中/右三区的最近障碍摘要",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "start", "stop", "info", "config",
                        "recognize_by_photo", "recognize_by_url",
                        "calibrate", "reset_calibration",
                    ],
                    "description": "Action to perform"
                },
                "input_topic": {
                    "type": "string",
                    "description": "ROS2 image topic to subscribe (e.g. /hostname/camera/rgb). 可选：不填则卡片以按需模式启动，不订阅摄像头，只服务 recognize_by_photo / recognize_by_url"
                },
                # `format: file` makes the canvas render a file picker;
                # `uploadTo: mcp` posts it to /api/mcp/<id>/file/upload, which
                # streams the bytes to *this* service and returns the path they
                # landed on here — so the value this field receives is already a
                # path perception can open, with no shared mount.
                "image_path": {"type": "string", "format": "file", "accept": "image/*", "uploadTo": "mcp", "description": "图片文件。从卡片上传，或填一个容器可读的路径（如 /uploads/scene.jpg）。常见格式都支持，过大的图会本地缩放"},
                "url": {"type": "string", "description": "图片的 http(s) 地址，如 https://example.com/scene.jpg。下载后本地解码，格式限制同 image_path"},
                "distance_m": {"type": "number", "description": "标定用：镜头到那面墙/平面的真实距离（米），用卷尺量。墙要正对镜头、填满取样区域"},
                "region": {"type": "string", "enum": list(CALIBRATION_REGIONS), "description": "标定用：在画面的哪一块取样。默认 center（画面正中 20% 的方框）；墙占满整个画面时可以用 full，取样像素更多"},
                "reset": {"type": "boolean", "description": "标定用：等同于 reset_calibration，保留给已有调用方"},
            },
            "required": ["action"],
            "x-action-params": {
                "start":  {"params": ["input_topic"], "description": "启动。给 input_topic 则持续估计该摄像头话题的深度；不给则以按需模式启动，只服务单张图片"},
                "stop":   {"params": [], "description": "停止深度估计"},
                "info":   {"params": ["input_topic"], "description": "查看状态、输出话题与当前标定（engine 自带 / 站点重标定）"},
                "config": {"params": [], "description": "更新 fps / 站点标定参数 cal_a、cal_b"},
                "recognize_by_photo": {
                    "params": ["image_path"],
                    "description": "看一张图片的远近 — 一次性估计，不需要摄像头也不需要先 start。返回整体的最近/最远/平均距离（米），以及左/中/右三个方向各自的最近与平均",
                },
                "recognize_by_url": {
                    "params": ["url"],
                    "description": "看一张图片 URL 的远近 — 与 recognize_by_photo 相同，只是图片来自 http(s) 而非本地文件",
                },
                "calibrate": {
                    "params": ["distance_m", "region", "image_path", "url"],
                    "description": (
                        "用已知距离校准这台相机的深度尺度。" + CALIBRATION_PROCEDURE +
                        " 每次调用都会把新样本并进来重新拟合（不是覆盖），"
                        "结果立刻生效；但只在内存里，要写进卡片配置的 cal_a / cal_b 才能在重启后保留。"
                    ),
                },
                "reset_calibration": {
                    "params": [],
                    "description": "清空所有标定样本，恢复成 engine 自带的标定。量错了、或者换了相机就用这个",
                },
            },
        },
        "configSchema": {
            "type": "object",
            "properties": {
                "fps":          {"type": "integer", "description": "Max inference frames per second", "default": 2, "scope": "instance"},
                # Log-affine site calibration, applied on top of the one baked
                # into the engine: metres_out = metres_in**cal_a * exp(cal_b).
                # Same two parameters ultralytics' model.calibrate() fits, so a
                # result from there pastes in here unchanged. 1.0 / 0.0 is
                # identity — i.e. trust the engine.
                "cal_a": {"type": "number", "description": "站点标定指数 a（d^a）。默认 1.0 = 不额外修正，直接用 engine 自带的标定", "default": 1.0, "scope": "instance"},
                "cal_b": {"type": "number", "description": "站点标定偏移 b（乘 e^b）。默认 0.0 = 不额外修正。与 ultralytics model.calibrate() 的 cal_b 同一参数", "default": 0.0, "scope": "instance"},
                # Presets are for the common case (a known robot, a known
                # camera); cal_a/cal_b above stay for anything else, and a
                # non-identity value there wins — see _calibration_from_cfg.
                "calibration_preset": {
                    "type": "string",
                    "enum": [CAL_NONE, CAL_MANUAL] + sorted(CALIBRATION_PRESETS),
                    "description": "深度标定从哪来。"
                                   f"{CAL_NONE!r} = 用 engine 自带的通用标定（默认）；"
                                   f"{CAL_MANUAL!r} = 用上面填的 cal_a / cal_b；"
                                   "其余是按机器人量好的预设。"
                                   "**标定是相机的属性** —— 同型号换了镜头必须重标，"
                                   "选错和不标定一样危险，只是更不容易发现。",
                    "default": CAL_NONE,
                    "scope": "instance",
                },
                # max_depth_m 不在这里：见 perception/config.yaml。它是发布层的
                # 截断阈值，操作员没有判断依据，而每多一个 schema 字段就多一处
                # 静默覆盖 config.yaml 的机会。
            },
        },
        "topic_in":  [{"format": "image/jpeg", "desc": "camera image input"}],
        "topic_out": [
            {"format": "image/depth-zlib", "desc": "per-pixel depth map (640x480 uint16)"},
            {"format": "data/json",        "desc": "nearest-obstacle summary by region"},
        ],
    }
]


# There is deliberately no prose `note` on results any more. It said the same
# paragraph on every single call — and it said the wrong thing, pointing at
# ultralytics' model.calibrate(), which cannot run on a robot (see the
# `calibrate` action). The one-token `calibration` field carries the same fact,
# and the explanation belongs in the tool description and the README, which are
# read once rather than re-sent with every answer.


# ── Site calibration ─────────────────────────────────────────────────────────

def apply_site_calibration(depth_m: np.ndarray, cal_a: float, cal_b: float) -> np.ndarray:
    """metres**cal_a * exp(cal_b), on top of the engine's own calibration.

    Log-affine, not a plain multiplier, because that is the shape ultralytics
    fits (`exp(a·log d + b)`) — so a `model.calibrate()` result transfers here
    unchanged. This used to be a linear `depth_scale`, which is the same thing
    only when a == 1 and silently wrong otherwise.

    Identity (1.0, 0.0) short-circuits: the common case should not pay for two
    array passes per frame.
    """
    if cal_a == 1.0 and cal_b == 0.0:
        return depth_m
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.power(np.maximum(depth_m, 0.0), cal_a) * float(np.exp(cal_b))
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def fit_cal_b(predicted_m, measured_m) -> float:
    """b = mean(log gt − log pred), with a pinned at 1.0. The whole estimator.

    This is exactly what ultralytics fits. Its `select_calibration`
    (models/yolo/depth/calibrate.py) scores two candidates — identity and
    "scale-only" (a=1, b=mean(log_gt − log_pred)) — and its docstring records
    that an affine log-slope candidate *was* evaluated and removed, because the
    extra parameter overfits within-dataset and hurts cross-distribution. So
    `cal_a` stays a knob for a fit obtained elsewhere, but nothing here or in
    ultralytics ever fits it.

    Geometric mean, not arithmetic: the error that matters is multiplicative
    (δ1 is a ratio test), so a 4 m reading that should be 2 m and a 1 m reading
    that should be 2 m have to cancel, and in log space they do.

    Why we cannot just call ultralytics' own calibrate() on a robot: it wants a
    `.pt` checkpoint, torch, ultralytics itself, and a dataloader yielding
    ground-truth depth *maps*. The runtime image has none of those — it carries
    a TensorRT engine and nothing else. Reference distances typed in by whoever
    is standing next to the robot are the data that actually exists here.
    """
    pred = np.asarray(predicted_m, dtype=np.float64).ravel()
    gt = np.asarray(measured_m, dtype=np.float64).ravel()
    valid = np.isfinite(pred) & np.isfinite(gt) & (pred > 0) & (gt > 0)
    if not valid.any():
        raise ValueError("no usable (predicted, measured) pair — both must be > 0")
    return float(np.mean(np.log(gt[valid]) - np.log(pred[valid])))


def _calibration_message(samples: list) -> str:
    """What the operator should do next, given how much evidence there is.

    One sample fixes the average scale and nothing else; it is worth having and
    worth not trusting too far. The advice is to add readings at clearly
    different distances, because that is what reveals whether the error is a
    constant factor (which this can fix) or grows with range (which it cannot —
    `a` is pinned at 1.0, following ultralytics).
    """
    count = len(samples)
    if count == 1:
        return (
            "已用 1 个样本标定。这只固定了整体比例 —— 请把机器人挪到另一个距离"
            "（比如这次 1 米、下次 2 米、3 米）再各量一次："
            "两个以上、且距离拉开，才能看出误差是固定倍数（能修）还是随距离变化"
            "（修不了，a 固定为 1.0）。"
        )
    distances = [s["measured_m"] for s in samples]
    spread = max(distances) / max(min(distances), 1e-6)
    if spread < 1.5:
        return (
            f"已用 {count} 个样本标定，但它们的距离都差不多"
            f"（{min(distances):.2f}–{max(distances):.2f} 米）。"
            "把机器人再前后挪开一些（1 米 / 2 米 / 3 米），才知道这个比例在远处还成不成立。"
        )
    return (
        f"已用 {count} 个样本标定，覆盖 {min(distances):.2f}–{max(distances):.2f} 米。"
        "看一下 residuals 里各点的 error_pct：都小就说明这台相机的误差确实是一个"
        "固定倍数，标定管用；如果近处准、远处偏，那是随距离变化的误差，"
        "这个两参数标定修不了。"
    )


def _calibration_from_cfg(cfg: dict, default: tuple[float, float] = (1.0, 0.0)) -> tuple[float, float]:
    """Resolve (cal_a, cal_b): typed-in value, then preset, then legacy, then engine.

    **Non-identity wins over a preset, and that test is deliberate.** The canvas
    sends every field in the schema on every config call, defaults included, so
    "the operator typed 1.0/0.0" and "the operator left it alone" arrive
    identical — presence cannot distinguish them. Being non-identity can: 1.0/0.0
    *is* "no correction", so treating it as "no opinion" costs nothing, while
    any other value is something somebody put there on purpose.

    `depth_scale` was a linear multiplier, which is exactly cal_b = log(scale) at
    cal_a = 1 — so an existing card keeps the behaviour it was configured for
    rather than silently reverting to identity.
    """
    cal_a = float(cfg.get("cal_a", default[0]))
    cal_b = float(cfg.get("cal_b", default[1]))
    # **The dropdown decides — when there is one.**
    #
    # The canvas sends `calibration_preset` on every config call, so its
    # presence means a selector is driving. Its *absence* means this config came
    # from `perception/config.yaml`, which has no dropdown and where writing
    # `cal_b: -0.9` has always meant "apply this". Making the selector
    # mandatory would silently stop that working, which is how a file config
    # becomes decoration — the exact failure the navi card had with its own
    # schema defaults.
    choice = cfg.get("calibration_preset")
    if choice is not None:
        choice = str(choice or CAL_NONE)
        if choice == CAL_MANUAL:
            return cal_a, cal_b
        if choice in CALIBRATION_PRESETS:
            preset = CALIBRATION_PRESETS[choice]
            return float(preset["cal_a"]), float(preset["cal_b"])
        return 1.0, 0.0
    if (cal_a, cal_b) != (1.0, 0.0):
        return cal_a, cal_b

    # Legacy keeps its original rule — **presence**, not value. Writing
    # `cal_b: 0.0` next to a `depth_scale` is how you turn the old multiplier
    # off, and there is an existing test for exactly that. The non-identity
    # rule above is only about presets, where the canvas's habit of sending
    # every default makes presence useless.
    legacy = cfg.get("depth_scale")
    if legacy not in (None, "") and "cal_b" not in cfg:
        legacy = float(legacy)
        if legacy > 0:
            return cal_a, float(np.log(legacy))

    return 1.0, 0.0


def calibration_origin(cfg: dict) -> str:
    """Which of the four sources is in effect, for `info()`.

    A card whose depth is 3.2x out and a card whose depth is right look the same
    from outside; this is the one field that separates them, so it says which
    kind of number is in play rather than only whether one was applied.
    """
    choice = cfg.get("calibration_preset")
    if choice is not None:
        choice = str(choice or CAL_NONE)
        if choice == CAL_MANUAL:
            return "manual"
        if choice in CALIBRATION_PRESETS:
            return f"preset:{choice}"
        return "model-default"
    if (float(cfg.get("cal_a", 1.0)), float(cfg.get("cal_b", 0.0))) != (1.0, 0.0):
        return "manual"
    if cfg.get("depth_scale") not in (None, "") and "cal_b" not in cfg:
        return "legacy-depth_scale"
    return "model-default"


def calibration_conflict(cfg: dict) -> str:
    """A typed-in fit that the dropdown is ignoring, or "".

    Someone who fills in cal_a/cal_b and forgets to switch the selector gets
    their numbers dropped. Applying them instead would be a guess; dropping
    them *quietly* would be the failure this card keeps running into. So they
    are dropped, and said out loud.
    """
    choice = cfg.get("calibration_preset")
    if choice is None or str(choice or CAL_NONE) == CAL_MANUAL:
        return ""
    choice = str(choice or CAL_NONE)
    pair = (float(cfg.get("cal_a", 1.0)), float(cfg.get("cal_b", 0.0)))
    if pair == (1.0, 0.0):
        return ""
    return (f"配置里填了 cal_a={pair[0]:g} / cal_b={pair[1]:g}，但标定来源选的是 "
            f"{choice!r} —— 这两个数**没有生效**。要用它们请把来源改成 "
            f"{CAL_MANUAL!r}。")


def sample_region(depth_m: np.ndarray, region: str = "center") -> dict:
    """One representative distance for part of the frame, plus how flat it is.

    Distance is the **median**, not the mean: the surface rarely fills the box
    exactly, and whatever is behind it at the edges would drag a mean off. The
    median holds as long as the surface covers more than half the box.

    `flatness` is the interquartile range over that median — a scale-free
    measure of how much the readings disagree. Facing a wall square-on it is
    near zero; pointed at a corridor, a corner, or a person standing in front
    of the wall it is not. That distinction is the whole reason the operator is
    asked for a flat plane: one number can only stand for the region if the
    region is genuinely all at one distance.
    """
    if region not in CALIBRATION_REGIONS:
        raise ValueError(f"region must be one of {CALIBRATION_REGIONS}, got {region!r}")
    height, width = depth_m.shape[:2]
    if region == "full":
        patch = depth_m
    elif region == "center":
        half_w, half_h = width * _TARGET_BOX / 2, height * _TARGET_BOX / 2
        cx, cy = width / 2, height / 2
        patch = depth_m[int(cy - half_h):int(cy + half_h), int(cx - half_w):int(cx + half_w)]
    else:
        third = round(width / 3)
        patch = depth_m[:, :third] if region == "left" else depth_m[:, 2 * third:]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size == 0:
        raise ValueError(f"no valid depth in the {region} region of this frame")
    median = float(np.median(valid))
    q1, q3 = np.percentile(valid, [25, 75])
    return {
        "distance_m": median,
        "flatness": round(float((q3 - q1) / max(median, 1e-6)), 3),
        "pixels": int(valid.size),
    }


def sample_region_depth(depth_m: np.ndarray, region: str = "center") -> float:
    """Just the distance from `sample_region`."""
    return sample_region(depth_m, region)["distance_m"]


# ── Depth encoding ───────────────────────────────────────────────────────────

def encode_depth(depth_m: np.ndarray, max_depth_m: float) -> bytes:
    """Renderer contract: zlib of 640x480 little-endian uint16 millimetres.

    Mirrors the driver-side encoder in
    phanthymotus-driver/realman/rm75_6f_v/realsense.py, including its rule that
    0 means "no reading": an out-of-range value must never wrap around into a
    plausible near-field distance, because the consumer cannot tell the
    difference and a wrapped 70 m reading looks like an obstacle at arm's length.
    """
    if depth_m.shape != (DEPTH_HEIGHT, DEPTH_WIDTH):
        raise ValueError(
            f"Expected a {DEPTH_WIDTH}x{DEPTH_HEIGHT} depth map, got "
            f"{depth_m.shape[1]}x{depth_m.shape[0]}"
        )
    mm = np.rint(np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0) * 1000.0)
    ceiling = min(65535.0, max(1.0, max_depth_m) * 1000.0)
    mm[(mm < 1) | (mm > ceiling)] = 0
    return zlib.compress(mm.astype("<u2").tobytes(), 1)


def sample_region_over_frames(frames, region: str = "center") -> dict:
    """`sample_region` over several frames, plus how much they disagreed.

    The median of the per-frame medians, not the median of everything pooled:
    one bad frame should move the answer by nothing, and pooling lets it move
    the answer by its share of the pixels.

    `scatter` is the peak-to-peak spread of the per-frame readings as a
    fraction of the median — the quantity that, left unmeasured on r1_sz, let a
    22% frame-to-frame wobble be mistaken for a 12% improvement from a second
    fit parameter.
    """
    readings = []
    flatness = []
    for frame in frames:
        try:
            one = sample_region(frame, region)
        except Exception:                                      # noqa: BLE001
            continue
        if one.get("distance_m") and one["distance_m"] > 0:
            readings.append(float(one["distance_m"]))
            flatness.append(float(one.get("flatness") or 0.0))
    if not readings:
        raise ValueError("no usable depth in that region")

    values = np.asarray(readings, dtype=np.float64)
    median = float(np.median(values))
    spread = float(values.max() - values.min()) / max(median, 1e-6)
    return {
        "distance_m": median,
        "flatness": float(np.median(flatness)) if flatness else 0.0,
        "frames": len(readings),
        "scatter": round(spread, 3),
    }


def summarize_depth(depth_m: np.ndarray, scale: str = "metric", bands: int = 3) -> dict:
    """Nearest valid reading per vertical band, plus the overall range.

    Uses the 5th percentile rather than the raw minimum: a monocular depth map
    routinely has a handful of near-zero outlier pixels at object edges, and a
    summary driven by the single closest pixel reports an obstacle that is not
    there. The percentile is over valid pixels only.
    """
    valid = np.isfinite(depth_m) & (depth_m > 0)
    width = depth_m.shape[1]
    edges = [round(i * width / bands) for i in range(bands + 1)]
    names = ["left", "center", "right"] if bands == 3 else [f"band{i}" for i in range(bands)]

    regions = {}
    for i, name in enumerate(names):
        chunk = depth_m[:, edges[i]:edges[i + 1]]
        chunk_valid = valid[:, edges[i]:edges[i + 1]]
        if not chunk_valid.any():
            regions[name] = None
            continue
        regions[name] = round(float(np.percentile(chunk[chunk_valid], 5)), 3)

    overall = depth_m[valid]
    return {
        "scale": scale,
        "unit": "m" if scale == "metric" else "relative",
        "nearest_by_region": regions,
        "range": [round(float(overall.min()), 3), round(float(overall.max()), 3)] if overall.size else None,
        "valid_fraction": round(float(valid.mean()), 3),
    }


def measure_depth(depth_m: np.ndarray, scale: str = "metric", bands: int = 3) -> dict:
    """`summarize_depth` plus the averages a one-shot answer needs.

    The streamed summary stays deliberately small — it is published several
    times a second and an agent re-reads it constantly. A single photo is asked
    about once, so it can afford the mean per region as well as the nearest,
    which is what separates "one close object against a far wall" from
    "everything in that direction is close".
    """
    stats = summarize_depth(depth_m, scale, bands)

    valid = np.isfinite(depth_m) & (depth_m > 0)
    width = depth_m.shape[1]
    edges = [round(i * width / bands) for i in range(bands + 1)]
    names = list(stats["nearest_by_region"].keys())

    averages: dict = {}
    for i, name in enumerate(names):
        chunk = depth_m[:, edges[i]:edges[i + 1]]
        chunk_valid = valid[:, edges[i]:edges[i + 1]]
        averages[name] = (round(float(chunk[chunk_valid].mean()), 3)
                          if chunk_valid.any() else None)

    overall = depth_m[valid]
    stats["average_by_region"] = averages
    stats["nearest"] = stats["range"][0] if stats["range"] else None
    stats["farthest"] = stats["range"][1] if stats["range"] else None
    stats["average"] = round(float(overall.mean()), 3) if overall.size else None

    known = {k: v for k, v in stats["nearest_by_region"].items() if v is not None}
    stats["closest_region"] = min(known, key=known.get) if known else None
    stats["farthest_region"] = max(known, key=known.get) if known else None
    return stats


# ── ROS2 Node (one per instance/topic) ───────────────────────────────────────

class _DepthNode(Node):
    """Per-topic depth inference node."""

    def __init__(self, input_topic: Optional[str], model, fps: float, cal_a: float,
                 cal_b: float, max_depth_m: float, node_suffix: str):
        super().__init__(f"visual_depth_{node_suffix}" if node_suffix else "visual_depth")
        # Topic-less is a supported mode, as in plugins/vop.py and plugins/tts.py:
        # a card driven only by recognize_by_photo has no camera to subscribe
        # to, but still wants somewhere to publish so the canvas shows the flow.
        self._input_topic = input_topic or ''
        self._depth_topic, self._summary_topic = output_topics_for(input_topic)
        self._model = model
        self._fps = fps
        self._frame_interval = 1.0 / max(fps, 0.1)
        self._cal_a = cal_a
        self._cal_b = cal_b
        self._scale_label = "metric"
        self._max_depth_m = max_depth_m

        self._depth_pub = self.create_publisher(CompressedImage, self._depth_topic, _PUB_QOS)
        self._summary_pub = self.create_publisher(String, self._summary_topic, _PUB_QOS)
        self._sub: Optional[object] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._last_inference_time = 0.0
        self._frame_count = 0
        self._running = False
        # Recent decoded depth, BEFORE site calibration — what the `calibrate`
        # action fits against.
        #
        # **A deque, not one frame.** It was one frame, and on r1_sz that made
        # the whole calibration a lottery: with the robot stationary and the
        # scene unchanged, fourteen consecutive frames of the same wall gave
        # 1.68–2.09 m, a 22% peak-to-peak spread; two `calibrate` calls at the
        # same 1.6 m gave 3.82 and 5.00, 31% apart. Four such samples were then
        # fitted with two parameters and the residuals argued convincingly for
        # a log-slope that was entirely noise.
        #
        # A reading whose reproducibility is ±30% cannot support an argument
        # about a 6% model improvement. Averaging is not polish here; it is the
        # difference between calibrating and guessing.
        from collections import deque as _deque
        self._raw_depth_history = _deque(maxlen=CALIBRATION_FRAMES)
        # See perception/README.md § "Plugin Concurrency" — every dispatch runs
        # on its own ThreadingHTTPServer thread and the canvas issues
        # config→start→stop→start within seconds.
        self._lifecycle_lock = threading.RLock()

    def request_stop(self) -> None:
        """Signal cancellation without taking the lock, so stop can abort a start."""
        self._stop_event.set()

    def start(self) -> dict:
        with self._lifecycle_lock:
            if self._running:
                return self._state("running")
            self._stop_event.clear()
            if self._input_topic and self._sub is None:
                self._sub = self.create_subscription(
                    CompressedImage, self._input_topic, self._image_cb, _LOW_LAT_QOS
                )
                self._worker = threading.Thread(target=self._inference_worker, daemon=True,
                                                name=f"visual_depth_worker_{self._input_topic}")
                self._worker.start()
            # Without a topic there is nothing to subscribe to and no frames to
            # consume, so no worker is spawned; the node exists to own the
            # publishers that one-shot results go out on.
            self._running = True
            log.info(f"[visual_depth] started: {self._input_topic or '(no topic, on-demand)'} "
                     f"→ {self._depth_topic}, {self._summary_topic}")
            return self._state("running")

    def stop(self) -> dict:
        # Worker only — the subscription is destroyed by destroy_node() after
        # the node leaves the executor. Destroying it here races the executor's
        # wait list and kills the spin thread with InvalidHandle, taking every
        # other subscription in the process down with it. See plugins/vop.py.
        self._stop_event.set()
        with self._lifecycle_lock:
            if self._worker and self._worker.is_alive():
                self._worker.join(timeout=3.0)
            self._worker = None
            self._running = False
            log.info(f"[visual_depth] stopped: {self._input_topic or '(no topic, on-demand)'}")
            return self._state("idle")

    def _state(self, state: str) -> dict:
        return {
            "state": state,
            "input": self._input_topic,
            "depth_topic": self._depth_topic,
            "summary_topic": self._summary_topic,
            "scale": self._scale_label,
            "mode": "stream" if self._input_topic else "on_demand",
        }

    def _image_cb(self, msg: CompressedImage):
        now = time.monotonic()
        if now - self._last_inference_time < self._frame_interval:
            return
        self._last_inference_time = now
        # Drop the stale frame rather than queue up: a depth map from two
        # seconds ago is worse than no depth map.
        try:
            self._frame_queue.put_nowait(msg.data)
        except queue.Full:
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frame_queue.put_nowait(msg.data)
            except queue.Full:
                pass

    def _inference_worker(self):
        import cv2
        from plugins.vision_runtime import decode_depth

        while not self._stop_event.is_set():
            try:
                jpeg_bytes = self._frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                outputs, meta = self._model.infer(frame)
                raw = decode_depth(outputs, meta)
                # Kept *uncalibrated* so `calibrate` can refit from a live
                # frame repeatedly without compounding its own correction —
                # fitting against already-corrected depth converges on
                # whatever the first guess was.
                self._raw_depth_history.append(raw)
                depth_m = apply_site_calibration(raw, self._cal_a, self._cal_b)
                # Resampled here, not by the model: the renderer's canvas is
                # fixed at 640x480 and a mismatch is dropped silently.
                if depth_m.shape != (DEPTH_HEIGHT, DEPTH_WIDTH):
                    depth_m = cv2.resize(depth_m, (DEPTH_WIDTH, DEPTH_HEIGHT),
                                         interpolation=cv2.INTER_NEAREST)
                self._publish(depth_m)
            except Exception as e:
                log.error(f"[visual_depth] inference error: {e}", exc_info=True)

    def _publish(self, depth_m: np.ndarray, summary: Optional[dict] = None):
        """Publish one depth map and its summary.

        `summary` is optional so a one-shot answer can reuse the richer stats it
        already computed instead of measuring the same array twice.
        """
        self._frame_count += 1

        depth_msg = CompressedImage()
        depth_msg.format = "16UC1; compressedDepth zlib"
        depth_msg.data = encode_depth(depth_m, self._max_depth_m)
        self._depth_pub.publish(depth_msg)

        summary = dict(summary) if summary is not None else summarize_depth(depth_m, self._scale_label)
        summary["timestamp"] = time.time()
        msg = String()
        msg.data = json.dumps(summary, ensure_ascii=False)
        self._summary_pub.publish(msg)


# ── Plugin class ─────────────────────────────────────────────────────────────

class VideoDepthPerceptionPlugin:
    PREFIX = "visual_depth"
    # `vdp` was the name this shipped under for one release. It said nothing to
    # anyone reading a card on the dashboard or the image's card list on
    # resource-center, so the tool is `visual_depth` now and the old spelling
    # stays as an alias — a card saved under `vdp` keeps dispatching instead of
    # coming back `state: error` after a restart. Same rule as the vop model
    # rename; see perception/README.md § "Vision".
    ALIASES = ("vdp",)

    def __init__(self, plugin_cfg: dict, namespace: str, executor):
        self._namespace = namespace
        self._executor = executor
        # Kept whole: image_input reads max_image_bytes and the path-confinement
        # settings straight from it (see plugins/image_input.py).
        self._plugin_cfg = dict(plugin_cfg or {})
        # What cal_a/cal_b were before `calibrate` overwrote them, so
        # `reset_calibration` can put the dict back exactly as it found it.
        self._cal_cfg_backup = None
        self._fps = int(plugin_cfg.get("fps", 2))
        self._cal_a, self._cal_b = _calibration_from_cfg(plugin_cfg)
        # (measured_m, predicted_m) reference readings from the `calibrate`
        # action, in call order. Refit from scratch on every addition, so a
        # bad sample can be undone with reset rather than compounding.
        self._cal_samples: list[dict] = []
        self._max_depth_m = float(plugin_cfg.get("max_depth_m", 20.0))
        self._model = None  # lazy load
        self._model_loading = False
        self._model_load_error = None
        # Same as vop: the downloader's progress line while bytes are moving.
        self._model_load_status = None
        self._model_lock = threading.Lock()
        self._nodes: dict[str, _DepthNode] = {}
        self._instance_configs: dict[str, dict] = {}
        # What the camera feeding each instance said about its optics
        # (`motus.camera/1`, arrives as `camera_info` on `start`). Kept so
        # `info()` can pass it on, rewritten for the image this card publishes —
        # see `plugins/camera_info.py`. Guarded by `_nodes_lock` like the dict
        # above it, per the plugin concurrency rules in the README.
        self._upstream_camera: dict[str, dict] = {}
        # Guards _nodes / _instance_configs; never held across a node start,
        # stop, or a model load.
        self._nodes_lock = threading.RLock()

    def _ensure_model(self):
        if self._model is not None:
            return
        with self._model_lock:
            if self._model is not None:
                return
            from plugins.vision_runtime import VisionEngineSession
            from utils.model_downloader import ensure_depth_model
            from utils.model_progress import fetch_status

            model_dir = os.environ.get("DEPTH_MODEL_DIR", "/models/depth")
            progress_cb, _ = fetch_status(
                lambda text: setattr(self, "_model_load_status", text), "yolo26n-depth")
            paths = ensure_depth_model(model_dir, progress_cb=progress_cb)
            engine = next(p for name, p in paths.items() if name.endswith(".engine"))
            log.info(f"[visual_depth] loading engine: {engine}")
            self._model = VisionEngineSession(engine)
            log.info(f"[visual_depth] engine loaded, input={self._model.input_size}")

    def _camera_info(self, instance_id, input_topic, nodes,
                     depth_topic: str, summary_topic: str) -> tuple:
        """`(declarations, note)` for this card's two output ports.

        The note exists because "the upstream camera declared nothing" and "this
        card dropped it on the floor" look identical from downstream, and only
        the first is somebody else's business to fix. Downstream already degrades
        correctly on a missing declaration — this is so the operator reading
        *this* card can see whose problem it is.
        """
        from plugins.camera_info import inherit

        key = instance_id if instance_id in (nodes or {}) else None
        if key is None:
            key = next(iter(nodes), None) if nodes else (input_topic or _DEFAULT_INSTANCE)
        with self._nodes_lock:
            upstream = self._upstream_camera.get(key) or {}
        if not upstream:
            return [], ("上游相机没有声明 camera_info —— 下游拿不到视场角，避障走廊"
                        "只能按保守兜底值算。相机卡片补上声明即可，见 "
                        "phanthymotus-driver/README_dev.md 的 Camera Parameters")

        out = inherit(upstream, topic=depth_topic, fmt="image/depth-zlib",
                      stage="perception/visual_depth",
                      width=DEPTH_WIDTH, height=DEPTH_HEIGHT)
        # The summary carries the same optics — it is the same picture reduced to
        # three numbers — so it is declared too, rather than left for a consumer
        # wired to the summary alone to guess at.
        out += inherit(upstream, topic=summary_topic, fmt="data/json",
                       stage="perception/visual_depth",
                       width=DEPTH_WIDTH, height=DEPTH_HEIGHT)
        return out, ""

    def _start_node(self, node_key: str, input_topic: Optional[str]):
        """Register before starting, so a concurrent stop can always cancel it."""
        with self._nodes_lock:
            if node_key in self._nodes:
                return
            icfg = self._instance_configs.get(node_key, {})
            node = _DepthNode(
                input_topic or None, self._model,
                fps=int(icfg.get("fps", self._fps)),
                **dict(zip(("cal_a", "cal_b"),
                           _calibration_from_cfg(icfg, (self._cal_a, self._cal_b)))),
                max_depth_m=float(icfg.get("max_depth_m", self._max_depth_m)),
                node_suffix=node_key.replace("/", "_").replace("-", "_").lstrip("_"),
            )
            self._executor.add_node(node)
            self._nodes[node_key] = node
        node.start()
        log.info(f"[visual_depth] node started (background): "
                 f"{input_topic or '(no topic, on-demand)'}")

    def _retire_node(self, node_key: str) -> Optional[dict]:
        with self._nodes_lock:
            node = self._nodes.pop(node_key, None)
        if node is None:
            return None
        with self._nodes_lock:
            # Goes with the node. Leaving it behind would let a re-wired card
            # answer `info()` with the optics of a camera it is no longer fed by,
            # which is a worse answer than no answer.
            self._upstream_camera.pop(node_key, None)
        node.request_stop()
        result = node.stop()
        # remove-then-destroy: the node must leave the executor before its
        # handles are destroyed, and it must be destroyed rather than merely
        # removed or the publishers and the ROS node name leak.
        dispose_node(self._executor, node, label=f"visual_depth/{node_key}")
        return result

    # ── one-shot depth measurement ───────────────────────────────────────────

    # ── site calibration from known distances ────────────────────────────────

    def _raw_depth_for_calibration(self, args: dict, instance_id: str):
        """Uncalibrated depth to fit against, as a **list of frames**.

        Prefers an explicitly supplied photo, because "here is a picture of a
        target at 2.0 m" is reproducible; otherwise takes the running
        instance's recent frames — plural, see `_raw_depth_history`. A photo is
        one frame by nature and is returned as a list of one, so the caller has
        a single shape to handle and the scatter simply comes out zero.
        """
        if args.get("image_path") or args.get("url") or args.get("image_url"):
            cfg = dict(self._plugin_cfg)
            data, source = load_image_bytes(args, cfg, url_action="calibrate")
            import cv2
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise BadInput("could not decode that file as an image", source)
            from plugins.vision_runtime import decode_depth
            outputs, meta = self._require_engine().infer(frame)
            return [decode_depth(outputs, meta)], source

        with self._nodes_lock:
            node = self._nodes.get(instance_id) if instance_id else None
            if node is None:
                node = self._nodes.get(_DEFAULT_INSTANCE)
            if node is None and len(self._nodes) == 1:
                node = next(iter(self._nodes.values()))
        if node is None:
            raise ValueError(
                "nothing to calibrate against — start this card on a camera "
                "first, or pass image_path / url"
            )
        frames = list(node._raw_depth_history)
        if not frames:
            raise ValueError(
                f"{node._input_topic or 'this card'} has not produced a frame yet"
            )
        return frames, node._input_topic or "(on-demand)"

    def _apply_calibration(self, cal_a: float, cal_b: float) -> None:
        """Set the fit here and on every running node, without a restart."""
        self._cal_a, self._cal_b = cal_a, cal_b
        with self._nodes_lock:
            nodes = list(self._nodes.values())
        for node in nodes:
            node._cal_a, node._cal_b = cal_a, cal_b

    def _calibration_report(self, extra: Optional[dict] = None) -> dict:
        """Current fit plus how well it matches the samples it was fit on."""
        residuals = []
        for sample in self._cal_samples:
            corrected = sample["predicted_m"] ** self._cal_a * float(np.exp(self._cal_b))
            residuals.append({
                "measured_m": sample["measured_m"],
                "corrected_m": round(corrected, 3),
                "error_pct": round(abs(corrected - sample["measured_m"])
                                   / sample["measured_m"] * 100, 1),
            })
        report = {
            "ok": True,
            "cal_a": round(self._cal_a, 6),
            "cal_b": round(self._cal_b, 6),
            "calibration": self._calibration_label(),
            "samples": len(self._cal_samples),
            "residuals": residuals,
            # The fit lives in memory. Saying so is the difference between a
            # calibration that survives a restart and one that quietly does not.
            "persist": (
                "生效了，但只在内存里。要长期保留，把 cal_a / cal_b 填进卡片配置"
                "（config action 或卡片的配置弹窗）。"
            ),
            "procedure": CALIBRATION_PROCEDURE,
        }
        if residuals:
            report["max_error_pct"] = max(r["error_pct"] for r in residuals)
        if extra:
            report.update(extra)
        return report

    def _calibration_label(self) -> str:
        """Which fit produced these metres, and **where that fit came from**.

        "site" was not enough. A refit typed in by hand, one chosen from a
        preset table, and one fitted live against a tape measure are three very
        different levels of evidence, and only the last was ever measured on
        *this* camera. A reader deciding whether to trust a distance needs to
        know which of the three is in play — `model-default` on r1_sz's
        ultra-wide lens reported 1.0 m as 3.2 m, and nothing said so.
        """
        if self._cal_samples:
            return f"site:calibrate({len(self._cal_samples)} 样本)"
        if self._cal_a == 1.0 and self._cal_b == 0.0:
            return "model-default"
        return f"site:{calibration_origin(self._plugin_cfg)}"

    def _require_engine(self):
        """Return a loaded engine, loading it on demand.

        A photo question is useful without any instance running — someone asks
        "how far away is this" before pointing a camera anywhere — so it
        triggers the same single-flight load a `start` would and waits for it,
        rather than reporting `loading` and making the caller poll. Each
        tools/call already has its own thread (ThreadingHTTPServer), so blocking
        here blocks nothing else. Same rule as plugins/vop.py.
        """
        self._ensure_model()
        return self._model

    def _recognize_image(self, args: dict, url_action: str) -> dict:
        """Estimate depth for one image and return the measurements.

        No prose summary. There was one, and it only restated `nearest` /
        `farthest` / `average` / the per-region numbers that are already in the
        reply — a model reading those can phrase them itself, and re-sending a
        paragraph of Chinese on every call is context spent to say nothing new.
        """
        cfg = dict(self._plugin_cfg)
        try:
            data, source = load_image_bytes(args, cfg, url_action=url_action)
        except BadInput as error:
            return error.as_result()

        import cv2

        started = time.time()
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return BadInput(
                "could not decode that file as an image — check it is a real "
                "picture and not, say, HTML returned by a redirect", source,
            ).as_result()

        try:
            model = self._require_engine()
        except Exception as error:  # noqa: BLE001 — surfaced to the caller
            log.error(f"[visual_depth] engine load failed during recognize: {error}",
                      exc_info=True)
            return {"ok": False, "reason": "engine_unavailable", "detail": str(error)}

        from plugins.vision_runtime import decode_depth

        outputs, meta = model.infer(frame)
        depth_m = apply_site_calibration(decode_depth(outputs, meta), self._cal_a, self._cal_b)
        scale_label = "metric"

        # Measured at the model's own resolution, not the renderer's 640x480:
        # the resample exists for the dashboard canvas, and the answer should
        # not be quantised by it.
        stats = measure_depth(depth_m, scale_label)

        height, width = frame.shape[:2]
        result = {
            "ok": True,
            "source": source,
            "image_size": [width, height],
            "latency_ms": int((time.time() - started) * 1000),
            **stats,
        }
        result["calibration"] = self._calibration_label()

        # Echo onto the card's output topics when an instance is running, so a
        # topic-less card wired into the canvas actually shows data flowing —
        # which is the only reason it is startable without a camera. Purely
        # additive, and deliberately not reported back: which topics this went
        # out on is not something the caller asked about, and every field here
        # is re-read by the model on every turn.
        self._publish_one_shot(args.get("instance_id", ""), depth_m, stats)
        return result

    def _publish_one_shot(self, instance_id: str, depth_m: np.ndarray,
                          stats: dict) -> Optional[list]:
        """Publish a one-shot result on the named instance, or the default one."""
        with self._nodes_lock:
            node = self._nodes.get(instance_id) if instance_id else None
            if node is None:
                node = self._nodes.get(_DEFAULT_INSTANCE)
            if node is None and len(self._nodes) == 1:
                node = next(iter(self._nodes.values()))
        if node is None:
            return None
        try:
            import cv2
            published = depth_m
            if published.shape != (DEPTH_HEIGHT, DEPTH_WIDTH):
                published = cv2.resize(published, (DEPTH_WIDTH, DEPTH_HEIGHT),
                                       interpolation=cv2.INTER_NEAREST)
            node._publish(published, stats)
            return [node._depth_topic, node._summary_topic]
        except Exception as error:  # noqa: BLE001 — never fail the answer on this
            log.warning(f"[visual_depth] could not echo one-shot result: {error}")
            return None

    def get_tools(self) -> list:
        return TOOLS

    def dispatch(self, name: str, args: dict) -> dict | None:
        action = args.get("action", name)
        instance_id = args.get("instance_id", "")

        if action == "info":
            if self._model_loading:
                return {"name": "VideoDepthPerception", "manufacture": "Embodied",
                        "model": "yolo26n-depth", "state": "loading",
                        "desc": "Loading depth engine..."}
            if self._model_load_error:
                return {"name": "VideoDepthPerception", "manufacture": "Embodied",
                        "model": "yolo26n-depth", "state": "error",
                        "desc": f"Engine load failed: {self._model_load_error}"}

            with self._nodes_lock:
                nodes = dict(self._nodes)
            instances = {
                key: {
                    "input": node._input_topic,
                    "depth_topic": node._depth_topic,
                    "summary_topic": node._summary_topic,
                    "fps": node._fps,
                    "scale": node._scale_label,
                    "frame_count": node._frame_count,
                }
                for key, node in nodes.items()
            }

            input_topic = args.get("input_topic", "")
            if not input_topic:
                topics_list = args.get("input_topics") or []
                if topics_list:
                    input_topic = topics_list[0]
            if instance_id and instance_id in nodes:
                input_topic = nodes[instance_id]._input_topic
            elif not input_topic and nodes:
                input_topic = next(iter(nodes.values()))._input_topic

            topics_in = [{"topic": input_topic, "format": "image/jpeg"}] if input_topic else []
            depth_topic, summary_topic = output_topics_for(input_topic)
            # `or nodes`: a topic-less instance has no input to derive from but
            # does publish, on the fixed default topics. Reporting nothing there
            # is what leaves a running on-demand card looking unwired.
            topics_out = ([
                {"topic": depth_topic, "format": "image/depth-zlib"},
                {"topic": summary_topic, "format": "data/json"},
            ] if (input_topic or nodes) else [])

            # Pass the camera's optics on, rewritten for what this card actually
            # publishes: the map is resampled to 640x480, which changes the
            # declared size and rescales `K` while leaving `half_fov_rad` alone
            # (a stretch, not a crop). See `plugins/camera_info.py`.
            camera_out, camera_note = self._camera_info(
                instance_id, input_topic, nodes, depth_topic, summary_topic)

            scale = "metric"
            info = {
                "name": "VideoDepthPerception", "manufacture": "Embodied",
                "model": "yolo26n-depth",
                "state": "running" if instances else "idle",
                "scale": scale,
                "instances": instances,
                "topic_in": topics_in,
                "topic_out": topics_out,
                "desc": "Monocular depth estimation (YOLO26-depth, TensorRT)",
            }
            info["unit"] = "m"
            info["calibration"] = self._calibration_label()
            if camera_out:
                info["camera_info"] = camera_out
            if camera_note:
                info["camera_info_note"] = camera_note
            return info

        elif action == "start":
            input_topic = args.get("input_topic")
            upstream_camera = args.get("camera_info")
            if not input_topic:
                topics_list = args.get("input_topics") or []
                if topics_list:
                    input_topic = topics_list[0]
            # No topic is a supported mode, as in plugins/vop.py and
            # plugins/tts.py: the card comes up on-demand, loads the engine and
            # owns its publishers, and answers recognize_by_photo /
            # recognize_by_url. It just has nothing to subscribe to.
            node_key = instance_id or input_topic or _DEFAULT_INSTANCE

            # Recorded before the node starts, so `info()` can answer with it
            # even while the engine is still loading — and recorded even when it
            # is empty, so a restart that no longer carries a declaration
            # replaces the old one rather than leaving a stale entry claiming a
            # lens that is no longer wired.
            if input_topic:
                from plugins.camera_info import for_topic as _camera_for_topic
                with self._nodes_lock:
                    self._upstream_camera[node_key] = _camera_for_topic(
                        upstream_camera, input_topic)

            with self._nodes_lock:
                running = self._nodes.get(node_key)
            if running is None:
                if self._model is None:
                    if self._model_loading:
                        return {"state": "loading",
                                "message": (self._model_load_status
                                            or "Engine is still loading, please wait...")}
                    if self._model_load_error:
                        return {"state": "error", "message": f"Engine failed to load: {self._model_load_error}"}

                    def _bg_start():
                        self._model_loading = True
                        self._model_load_error = None
                        self._model_load_status = None
                        try:
                            self._ensure_model()
                            self._model_loading = False
                            self._model_load_status = None
                            self._start_node(node_key, input_topic)
                        except Exception as e:
                            self._model_loading = False
                            self._model_load_error = str(e)
                            log.error(f"[visual_depth] engine load failed: {e}", exc_info=True)

                    threading.Thread(target=_bg_start, daemon=True, name="visual_depth_model_load").start()
                    return {"state": "loading", "input": input_topic,
                            "message": "Engine loading in background, will start automatically"}
                self._start_node(node_key, input_topic)
                with self._nodes_lock:
                    running = self._nodes.get(node_key)
                if running is None:
                    return {"state": "idle", "input": input_topic}
            return running.start()

        elif action == "stop":
            if instance_id:
                result = self._retire_node(instance_id)
                return result if result is not None else {"state": "idle"}
            with self._nodes_lock:
                keys = list(self._nodes.keys())
            results = [key for key in keys if self._retire_node(key) is not None]
            return {"state": "idle", "stopped_instances": results} if results else {"state": "idle"}

        elif action == "config":
            cfg = {k: v for k, v in args.items()
                   if k not in ("action", "instance_id") and v is not None and v != ""}
            if instance_id:
                with self._nodes_lock:
                    self._instance_configs[instance_id] = cfg
                    running = instance_id in self._nodes
                if running:
                    self._retire_node(instance_id)
                return {"status": "configured", "instance_id": instance_id, "config": cfg}
            if "fps" in cfg:
                self._fps = int(cfg["fps"])
            if any(k in cfg for k in ("cal_a", "cal_b", "depth_scale")):
                self._cal_a, self._cal_b = _calibration_from_cfg(
                    cfg, (self._cal_a, self._cal_b))
            if "max_depth_m" in cfg:
                self._max_depth_m = float(cfg["max_depth_m"])
            return {"status": "configured", "config": cfg}

        elif action in ("calibrate", "reset_calibration"):
            if action == "reset_calibration" or args.get("reset"):
                self._cal_samples = []
                # Undo what `calibrate` wrote into the config, so reset means
                # what it has always meant: back to whatever was configured
                # before anyone measured, which with an untouched card is the
                # engine's own fit.
                if self._cal_cfg_backup is not None:
                    for key, value in zip(
                            ("cal_a", "cal_b", "calibration_preset"),
                            self._cal_cfg_backup):
                        if value is _ABSENT:
                            self._plugin_cfg.pop(key, None)
                        else:
                            self._plugin_cfg[key] = value
                    self._cal_cfg_backup = None
                self._apply_calibration(*_calibration_from_cfg(self._plugin_cfg))
                return self._calibration_report({"message": "标定已清空，恢复成 engine 自带的标定"})

            distance = args.get("distance_m")
            if distance in (None, ""):
                return {"ok": False, "reason": "bad_input",
                        "detail": "distance_m is required. " + CALIBRATION_PROCEDURE}
            distance = float(distance)
            if distance <= 0:
                return {"ok": False, "reason": "bad_input",
                        "detail": f"distance_m must be positive, got {distance}"}

            region = args.get("region") or "center"
            try:
                frames, source = self._raw_depth_for_calibration(args, instance_id)
                reading = sample_region_over_frames(frames, region)
            except BadInput as error:
                return error.as_result()
            except Exception as error:  # noqa: BLE001 — surfaced to the caller
                return {"ok": False, "reason": "bad_input", "detail": str(error)}

            self._cal_samples.append({
                "measured_m": distance,
                "predicted_m": round(reading["distance_m"], 3),
                "region": region,
                "flatness": reading["flatness"],
                # How much this reading moved while nothing did. Recorded per
                # sample because it is the honest error bar on the fit, and
                # because without it the next person will do what was done
                # here: read structure into the residuals and add a parameter
                # to explain it.
                "frames": reading["frames"],
                "scatter": reading["scatter"],
                "source": source,
            })
            # Refit over every sample, not incrementally: `a` is pinned at 1.0
            # and `b` is a mean, so the whole history is one cheap pass and a
            # reset genuinely undoes things.
            cal_b = fit_cal_b([s["predicted_m"] for s in self._cal_samples],
                              [s["measured_m"] for s in self._cal_samples])
            self._apply_calibration(1.0, cal_b)
            # Write the fit into the card's own config as the manual pair.
            #
            # It makes the card internally consistent: `info()` then reports
            # `manual` rather than claiming a preset or the engine default is in
            # play, which is what the reader of a distance needs to know.
            #
            # The pre-calibration pair is snapshotted first, because
            # `reset_calibration` promises a return to *the engine's own fit* —
            # writing here without that would quietly redefine reset as "go back
            # to the last thing I measured", and there are two tests on the
            # original meaning.
            #
            # **It is not persistence.** The canvas owns this config and pushes
            # its own copy on every `config` call, including on project start,
            # which will overwrite these two numbers with whatever the dialog
            # holds. Until they are typed in there (or into
            # perception/config.yaml) a restart loses the fit — and it has,
            # twice, on r1_sz, each time silently restoring a 3.2x error.
            if self._cal_cfg_backup is None:
                self._cal_cfg_backup = (
                    self._plugin_cfg.get("cal_a", _ABSENT),
                    self._plugin_cfg.get("cal_b", _ABSENT),
                    self._plugin_cfg.get("calibration_preset", _ABSENT))
            self._plugin_cfg["cal_a"] = 1.0
            self._plugin_cfg["cal_b"] = round(cal_b, 6)
            # ...and switch the selector, or the rule above would ignore the fit
            # that was just measured — writing two numbers nothing reads is
            # worse than not writing them.
            self._plugin_cfg["calibration_preset"] = CAL_MANUAL
            log.info(f"[visual_depth] calibrated: {len(self._cal_samples)} sample(s) "
                     f"→ cal_a=1.0 cal_b={cal_b:.4f}")

            extra = {
                "sample": self._cal_samples[-1],
                # The exact pair to put in the card's config dialog. Spelled out
                # as a field rather than left in prose because it is the one
                # thing that has to survive a restart, and prose does not get
                # copied accurately.
                "save_to_config": {"calibration_preset": CAL_MANUAL,
                                   "cal_a": 1.0, "cal_b": round(cal_b, 6)},
                "message": _calibration_message(self._cal_samples),
            }
            warnings = []
            if reading["scatter"] > _SCATTER_LIMIT:
                warnings.append(
                    f"这个样本的帧间抖动是中位数的 {reading['scatter'] * 100:.0f}%"
                    f"（{reading['frames']} 帧，阈值 {_SCATTER_LIMIT * 100:.0f}%）。"
                    "读数本身就这么不稳，那么标定残差里的任何「规律」都可能只是"
                    "抓到了哪一帧 —— 先让画面稳下来（别动、别让人走过），"
                    "再判断误差是不是随距离变化。")
            if reading["flatness"] > _FLATNESS_LIMIT:
                warnings.append(
                    f"取样区域看起来不是一个平面：区域内深度的四分位跨度是中位数的 "
                    f"{reading['flatness'] * 100:.0f}%（阈值 {_FLATNESS_LIMIT * 100:.0f}%）。"
                    "一个距离代表不了这块画面 —— 请让机器人正对一面平整的墙，"
                    "确认墙填满取样框、画面里没有别的东西，然后 reset_calibration 重来。"
                )
            extra["flatness"] = reading["flatness"]
            report = self._calibration_report(extra)
            # A sample that disagrees with the rest after the fit is usually a
            # typo (2 for 20) or a reading taken facing something else. Name it
            # rather than letting it quietly drag the mean.
            outliers = [r for r in report["residuals"] if r["error_pct"] > _OUTLIER_PCT]
            if outliers and len(self._cal_samples) > 1:
                worst = max(outliers, key=lambda r: r["error_pct"])
                warnings.append(
                    f"有 {len(outliers)} 个样本和其余的对不上，最差的一个量的是 "
                    f"{worst['measured_m']} 米、标定后是 {worst['corrected_m']} 米"
                    f"（差 {worst['error_pct']}%）。要么那次量错了或对着别的东西，"
                    "要么这台相机的误差随距离变化 —— 后者这个两参数标定修不了。"
                )
            if warnings:
                report["warnings"] = warnings
            return report

        elif action == "recognize_by_photo":
            return self._recognize_image(args, url_action="recognize_by_url")

        elif action == "recognize_by_url":
            return self._recognize_image(args, url_action="recognize_by_url")

        return None
