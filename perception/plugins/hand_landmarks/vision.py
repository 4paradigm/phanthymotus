"""Hand landmark backends for the hand-drive card.

The card needs exactly one thing from vision: 21 landmarks in normalized image
coordinates. Three backends are provided:

* ``rtmpose`` -- the RTMPose-m hand5 TensorRT engine already shipped in
  ``perception/``. The engine is downloaded lazily from COS on first use,
  pinned by size and SHA256, and cached in the shared ``/models`` volume. This
  is the **default and the only backend that works out of the box on a Jetson**:
  no model has to be placed on the machine by hand.
* ``mediapipe`` -- the tasks-vision HandLandmarker with a ``.task`` bundle.
  Kept for local Mac development only: neither mediapipe nor the bundle is in
  the Jetson image.
* ``mock`` -- deterministic synthetic open/fist landmarks, for canvas and
  protocol testing without a camera or a model.

The RTMPose path uses ``hands_in_frame`` (whole-frame inference). It needs no
body keypoints, but it also cannot tell left from right: the handedness label
is therefore the canvas *configuration*, marked as ``configured`` on every
detection rather than passed off as something the model saw.

Nothing in this module knows about I20-lite or joint angles.
"""

from __future__ import annotations

import importlib
import math
import os
import time
from typing import Optional

import numpy as np

N_LANDMARKS = 21

# Joints that stay visible when the hand closes into a fist. The fingertip
# joints (6-8, 10-12, 14-16, 18-20) fold behind the palm and their detection
# scores collapse; judging a whole hand by the mean of all 21 scores rejects
# exactly the fist frames we need to drive a closing grip. Wrist + the five
# MCP/CMC roots survive the fold and are the right place to gate on.
PALMAR_JOINTS = (0, 1, 5, 9, 13, 17)

# Synthetic landmarks, only used by the mock backend. Coordinates are
# normalized (x right, y down), z is a placeholder because the mock has no
# depth model. The shapes are deliberately crude: they only have to separate
# "extended" from "folded" for the retarget tests.
_OPEN = np.array([
    [0.50, 0.90, 0.0],
    [0.42, 0.78, 0.0], [0.35, 0.68, 0.0], [0.28, 0.60, 0.0], [0.22, 0.52, 0.0],
    [0.45, 0.72, 0.0], [0.44, 0.58, 0.0], [0.43, 0.47, 0.0], [0.42, 0.36, 0.0],
    [0.51, 0.71, 0.0], [0.51, 0.55, 0.0], [0.51, 0.44, 0.0], [0.51, 0.31, 0.0],
    [0.57, 0.72, 0.0], [0.58, 0.57, 0.0], [0.59, 0.46, 0.0], [0.59, 0.34, 0.0],
    [0.62, 0.75, 0.0], [0.64, 0.62, 0.0], [0.65, 0.53, 0.0], [0.66, 0.44, 0.0],
], dtype=np.float32)

_FIST = np.array([
    [0.50, 0.90, 0.0],
    [0.42, 0.78, 0.0], [0.37, 0.70, 0.0], [0.39, 0.66, 0.0], [0.44, 0.70, 0.0],
    [0.45, 0.72, 0.0], [0.46, 0.66, 0.0], [0.47, 0.69, 0.0], [0.46, 0.71, 0.0],
    [0.51, 0.71, 0.0], [0.52, 0.65, 0.0], [0.53, 0.68, 0.0], [0.52, 0.70, 0.0],
    [0.57, 0.72, 0.0], [0.58, 0.66, 0.0], [0.59, 0.69, 0.0], [0.58, 0.71, 0.0],
    [0.62, 0.75, 0.0], [0.64, 0.69, 0.0], [0.65, 0.72, 0.0], [0.64, 0.74, 0.0],
], dtype=np.float32)


class LandmarkFilter:
    """One Euro filter for 21 landmarks (Casiez et al. 2012).

    Used in teleoperation (AnyTeleop, arXiv:2307.04577) to fuse detections
    temporally: a low-pass filter whose cutoff rises with speed. When the hand
    is still it heavily suppresses fingertip jitter; when the hand moves fast
    it follows with low latency. A fixed low-pass would blur fast motions.
    """

    def __init__(self, *, min_cutoff: float = 1.0, beta: float = 0.08,
                 d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._previous: Optional[np.ndarray] = None
        self._prev_time: Optional[float] = None
        self._derivative: Optional[np.ndarray] = None

    @staticmethod
    def _alpha(cutoff, dt: float):
        # cutoff may be a scalar or one value per landmark (21,).
        tau = 1.0 / (2.0 * math.pi * np.maximum(cutoff, 1e-6))
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def update(self, landmarks: np.ndarray, now: float) -> np.ndarray:
        points = np.asarray(landmarks, dtype=np.float32)[:, :2]
        if self._previous is None or self._prev_time is None:
            self._previous = points.copy()
            self._prev_time = float(now)
            self._derivative = np.zeros_like(points)
            return np.asarray(landmarks, dtype=np.float32).copy()

        dt = max(1e-3, float(now) - self._prev_time)
        self._prev_time = float(now)
        derivative = (points - self._previous) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self._derivative = (1.0 - a_d) * self._derivative + a_d * derivative

        speed = np.linalg.norm(self._derivative, axis=1)
        cutoff = self.min_cutoff + self.beta * speed
        a = self._alpha(cutoff, dt)[:, None]

        smoothed_xy = (1.0 - a) * self._previous + a * points
        self._previous = smoothed_xy.copy()
        out = np.asarray(landmarks, dtype=np.float32).copy()
        out[:, :2] = smoothed_xy
        return out

    def reset(self) -> None:
        self._previous = None
        self._prev_time = None
        self._derivative = None


class HandDetection:
    """One hand read from one frame.

    ``landmarks`` is (21, 3) in normalized image coordinates with x right and
    y down. ``z`` is model-relative, never a calibrated distance in metres.
    ``handedness`` is the backend's own label and may be empty.

    ``handedness_source`` is honest about where that label came from:
    ``model`` (MediaPipe reads it from the image), ``configured`` (the whole
    -frame RTMPose path cannot tell left from right, so the canvas setting is
    used) or ``mock``. The driver card must not mistake a configured label for
    a detected one.
    """

    __slots__ = ("landmarks", "handedness", "score", "handedness_source")

    def __init__(self, landmarks, handedness: str = "", score: float = 1.0,
                 handedness_source: str = "model"):
        array = np.asarray(landmarks, dtype=np.float32)
        if array.shape != (N_LANDMARKS, 3) or not np.isfinite(array).all():
            raise ValueError("expected 21 finite (x, y, z) landmarks")
        self.landmarks = array
        self.handedness = str(handedness or "")
        self.score = float(score)
        self.handedness_source = str(handedness_source or "model")


class MediaPipeHands:
    """tasks-vision HandLandmarker in single-image mode.

    Built at card start, not import time: a missing dependency or a bad model
    path must be reported by ``info()`` as a refused start, not take down the
    whole ActuCore bundle before any card has been wired.
    """

    def __init__(self, model_path: str, *, num_hands: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_presence_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        path = os.path.expanduser(str(model_path or ""))
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(
                f"手部关键点模型不存在：{path or '(未设置 model_path)'}。"
                "请设置 hand_landmarks.model_path 指向 hand_landmarker.task")
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError as error:
            raise RuntimeError(
                "mediapipe 未安装；本地测试可 pip install mediapipe，"
                "或把 hand_landmarks.backend 设为 mock") from error

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=path),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=max(1, int(num_hands)),
            min_hand_detection_confidence=float(min_detection_confidence),
            min_hand_presence_confidence=float(min_presence_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)

    def detect(self, frame_bgr) -> list:
        """Return at most ``num_hands`` detections from a BGR frame."""
        import cv2
        from mediapipe import Image, ImageFormat

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image(data=rgb, image_format=ImageFormat.SRGB)
        result = self._landmarker.detect(image)
        detections = []
        for landmarks, handedness in zip(
                result.hand_landmarks or [], result.handedness or []):
            points = np.array(
                [[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
            label = ""
            score = 1.0
            if handedness:
                label = str(handedness[0].category_name or "")
                score = float(handedness[0].score or 0.0)
            try:
                detections.append(HandDetection(points, label, score))
            except ValueError:
                continue
        return detections

    def close(self) -> None:
        close = getattr(self._landmarker, "close", None)
        if close is not None:
            close()


class MockHands:
    """Deterministic synthetic hand, with no camera and no dependency.

    ``pose`` is ``open``, ``fist`` or ``alternate``. It exists so the canvas,
    the negotiation and the control message can be exercised locally; it must
    never be the default on a machine connected to a real hand.
    """

    def __init__(self, pose: str = "open"):
        self.pose = str(pose or "open").lower()
        if self.pose not in ("open", "fist", "alternate"):
            raise ValueError(f"unknown mock pose {pose!r}")
        self._tick = 0

    def detect(self, frame_bgr=None) -> list:
        if self.pose == "alternate":
            points = _FIST if self._tick % 2 else _OPEN
            self._tick += 1
        else:
            points = _FIST if self.pose == "fist" else _OPEN
        return [HandDetection(points.copy(), "Right", 1.0,
                              handedness_source="mock")]

    def close(self) -> None:
        pass


class RTMPoseHands:
    """The shipped RTMPose-m hand5 engine, downloaded on first use.

    This is the backend that makes the card deployable on a fresh Jetson. No
    model is baked into the image and none has to be staged by hand:
    ``ensure_hand_model`` fetches ``rtmpose-m-hand5.engine`` from COS, checks
    its size and SHA256 against the pinned table, and caches it in the shared
    ``/models/hand``. The engine family (jp61 / jp511) is chosen from the
    importable TensorRT major version, not from an image build argument.

    ``hands_in_frame`` is whole-frame inference: it needs no body keypoints,
    so ``remote_camera`` alone is enough. The cost is that it cannot label a
    hand left or right — the configured side is attached as a *declared*
    label, with ``handedness_source='configured'`` so consumers never mistake
    it for a detection.
    """

    def __init__(self, *, model_dir: str | None = None,
                 confidence: float = 0.42, max_hands: int = 1,
                 handedness: str = "Right", family: str | None = None,
                 min_cutoff: float = 1.0, beta: float = 0.08,
                 hold_frames: int = 5, on_status=None):
        from utils.model_downloader import ensure_hand_model
        from utils.model_progress import fetch_status
        from plugins.hand_runtime import HandEngine, assert_simcc

        self._confidence = float(confidence)
        # Let the engine hand us low-scoring candidates (a fist), then gate
        # on the palm, which remains visible. Without this the all-joint mean
        # in hands_in_frame drops below `confidence` on every fist frame and
        # the downstream driver never sees a close command at all.
        self._candidate_confidence = min(self._confidence, 0.05)
        self._max_hands = max(1, int(max_hands))
        self._handedness = str(handedness or "either")
        directory = str(model_dir or os.environ.get("HAND_MODEL_DIR", "/models/hand"))
        progress_cb, stage_cb = fetch_status(on_status, "rtmpose-m-hand5")
        paths = ensure_hand_model(directory, family=family, progress_cb=progress_cb)
        engine = next((path for name, path in paths.items()
                       if name.endswith(".engine")), None)
        if engine is None:
            raise RuntimeError(f"手部模型目录没有 .engine 文件：{paths}")
        self._session = HandEngine(engine)
        assert_simcc(self._session)
        # Temporal fusion (AnyTeleop): smooth jitter and briefly hold the
        # last pose when a fist / side view loses detection momentarily.
        self._filter = LandmarkFilter(min_cutoff=min_cutoff, beta=beta)
        self._hold_frames = max(0, int(hold_frames))
        self._misses = 0
        self._last: Optional[HandDetection] = None

    def detect(self, frame_bgr) -> list:
        from plugins.hand_runtime import hands_in_frame

        height, width = frame_bgr.shape[:2]
        found = hands_in_frame(self._session, frame_bgr,
                               confidence=self._candidate_confidence,
                               max_hands=self._max_hands)
        detections = []
        for points, score in found:
            # hands_in_frame reports pixel coordinates and puts per-joint
            # confidence in the third column. Normalize x/y back to 0..1 for
            # the common HandDetection contract; keep z out of it entirely —
            # the engine has no calibrated depth and the value is not metric.
            array = np.asarray(points, dtype=np.float32).reshape(-1, 3)
            if array.shape[0] != N_LANDMARKS:
                continue
            # Gate on the palm, not the all-joint mean. Fingertips fold behind
            # the palm in a fist and drag the mean under the threshold; the
            # wrist and MCP roots stay visible and still score well.
            # Palm-facing hands have less texture than back-facing ones, so
            # their joint scores are systematically lower. Gate on how many
            # palm joints are confidently detected (not the raw mean), which
            # treats both orientations fairly and still rejects noise.
            palmar_scores = array[list(PALMAR_JOINTS), 2]
            confident = int(np.count_nonzero(palmar_scores >= self._confidence))
            palmar = float(np.mean(palmar_scores))
            if confident < 4 or palmar < self._confidence * 0.6:
                continue
            normalized = np.zeros((N_LANDMARKS, 3), dtype=np.float32)
            normalized[:, 0] = array[:, 0] / max(width - 1, 1)
            normalized[:, 1] = array[:, 1] / max(height - 1, 1)
            try:
                detections.append(HandDetection(
                    normalized, self._handedness, palmar,
                    handedness_source="configured"))
            except ValueError:
                continue
        if detections:
            detection = detections[0]
            detection.landmarks[:] = self._filter.update(
                detection.landmarks, time.monotonic())
            self._misses = 0
            self._last = detection
            return detections

        # Brief detection loss (fist / side view): freeze the last pose for a
        # few frames rather than telling the driver the hand vanished. The
        # driver watchdog is the safety net for a longer loss.
        if self._hold_frames and self._last is not None:
            self._misses += 1
            if self._misses <= self._hold_frames:
                return [self._last]
        self._last = None
        self._filter.reset()
        return []

    def close(self) -> None:
        close = getattr(self._session, "close", None)
        if close is not None:
            close()


def default_model_path() -> str:
    """Where the staged landmarker bundle lives in this workspace.

    Docker deployments should set the path explicitly (the workspace sibling
    directory is not copied into the ActuCore image). Kept as a convenience for
    the local Mac path only.
    """
    override = os.environ.get("HAND_LANDMARKER_MODEL", "")
    if override:
        return os.path.expanduser(override)
    return "/models/hand_landmarker.task"


def create_backend(name: str, cfg: dict, on_status=None):
    """Build the configured backend. Unknown names fail loudly at start."""
    backend = str(name or "rtmpose").lower()
    if backend == "rtmpose":
        return RTMPoseHands(
            model_dir=cfg.get("model_dir"),
            confidence=float(cfg.get("hand_confidence", 0.42)),
            max_hands=int(cfg.get("num_hands") or 1),
            handedness=cfg.get("hand", "Right"),
            family=cfg.get("model_family") or None,
            min_cutoff=float(cfg.get("landmark_min_cutoff", 1.0)),
            beta=float(cfg.get("landmark_beta", 0.08)),
            hold_frames=int(cfg.get("detection_hold_frames", 5)),
            on_status=on_status,
        )
    if backend == "mediapipe":
        return MediaPipeHands(
            cfg.get("model_path") or default_model_path(),
            num_hands=int(cfg.get("num_hands") or 1),
            min_detection_confidence=float(cfg.get("min_detection_confidence", 0.5)),
            min_presence_confidence=float(cfg.get("min_presence_confidence", 0.5)),
            min_tracking_confidence=float(cfg.get("min_tracking_confidence", 0.5)),
        )
    if backend == "mock":
        return MockHands(cfg.get("mock_pose", "open"))
    raise ValueError(
        f"unknown hand_landmarks backend {name!r}; use rtmpose, mediapipe or mock")
