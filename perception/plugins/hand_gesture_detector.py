#!/usr/bin/env python3
"""
plugins/hand_gesture_detector.py — HaGRIDv2-based single- and two-hand gesture
detection runtime for `plugins/pose.py`.

Runs two lightweight NMS-free end-to-end YOLO TensorRT engines (via
`plugins.vision_runtime.VisionEngineSession` and `decode_detections`):

1. `hand_det` (`YOLOv10n_hands`, ~2.27M params): locates individual hand
   bounding boxes (`[x1, y1, x2, y2]`) for both single-hand and two-hand poses,
   including natural resting hands.
2. `gesture_det` (`YOLOv10n_gestures`, ~2.27M params): classifies 33 HaGRIDv2
   gesture classes (+ `no_gesture`), emitting single-hand boxes for single-hand
   gestures and united two-hand boxes for two-hand combination gestures.

Body keypoints from COCO-17 (`yolo26s-pose` / `yolo26n-pose`) anchor each
detected single-hand box to the person's own `left` or `right` hand (not image
left/right), handling crossed arms (`xsign`), T-poses (`timeout`) and close
praying hands (`holy`) cleanly without overshooting the wrist anchor.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from plugins.vision_runtime import COCO_INDEX, decode_detections

log = logging.getLogger(__name__)

# Official 34-class HaGRIDv2 index order (0..32 gestures, 33 = no_gesture).
HAGRID_CLASS_NAMES: Tuple[str, ...] = (
    "grabbing",         # 0
    "grip",             # 1
    "holy",             # 2
    "point",            # 3
    "call",             # 4
    "three3",           # 5
    "timeout",          # 6
    "xsign",            # 7
    "hand_heart",       # 8
    "hand_heart2",      # 9
    "little_finger",    # 10
    "middle_finger",    # 11
    "take_picture",     # 12
    "dislike",          # 13
    "fist",             # 14
    "four",             # 15
    "like",             # 16
    "mute",             # 17
    "ok",               # 18
    "one",              # 19
    "palm",             # 20
    "peace",            # 21
    "peace_inverted",   # 22
    "rock",             # 23
    "stop",             # 24
    "stop_inverted",    # 25
    "three",            # 26
    "three2",           # 27
    "two_up",           # 28
    "two_up_inverted",  # 29
    "three_gun",        # 30
    "thumb_index",      # 31
    "thumb_index2",     # 32
    "no_gesture",       # 33
)

# Canonical 30-class gesture vocabulary (1..23 single-hand, 24..30 two-hand).
HAGRID_TO_CANONICAL: Dict[str, Optional[str]] = {
    "grabbing": "grabbing",
    "grip": "grip",
    "holy": "holy",
    "point": "pointing",
    "call": "call",
    "three3": "three3",
    "timeout": "timeout",
    "xsign": "xsign",
    "hand_heart": "hand_heart",
    "hand_heart2": "hand_heart2",
    "little_finger": "little_finger",
    "middle_finger": "middle_finger",
    "take_picture": "take_picture",
    "dislike": "thumbs_down",
    "fist": "fist",
    "four": "four",
    "like": "thumbs_up",
    "mute": "mute",
    "ok": "ok",
    "one": "one",
    "palm": "palm",
    "peace": "victory",
    "peace_inverted": "victory",
    "rock": "rock",
    "stop": "stop",
    "stop_inverted": "stop",
    "three": "three",
    "three2": "three2",
    "two_up": "two_up",
    "two_up_inverted": "two_up",
    "three_gun": "three_gun",
    "thumb_index": "thumb_index",
    "thumb_index2": "thumb_index2",
    "no_gesture": None,
}

SINGLE_HAND_GESTURES: Tuple[str, ...] = (
    "stop", "fist", "pointing", "victory", "ok", "thumbs_up", "one",
    "thumbs_down", "palm", "two_up", "three", "three2", "three3", "four",
    "call", "mute", "rock", "little_finger", "middle_finger", "grabbing",
    "grip", "thumb_index", "three_gun",
)

TWO_HAND_GESTURES: Tuple[str, ...] = (
    "holy", "timeout", "xsign", "hand_heart", "hand_heart2",
    "take_picture", "thumb_index2",
)

TWO_HAND_GESTURE_SET = frozenset(TWO_HAND_GESTURES)

ALL_DETECTOR_GESTURES: Tuple[str, ...] = SINGLE_HAND_GESTURES + TWO_HAND_GESTURES

DETECTOR_GESTURE_LABELS_ZH: Dict[str, str] = {
    "stop": "停止",
    "fist": "握拳",
    "pointing": "食指指向",
    "victory": "V 字",
    "ok": "OK",
    "thumbs_up": "拇指向上",
    "one": "数字一",
    "thumbs_down": "拇指向下",
    "palm": "平掌",
    "two_up": "数字二",
    "three": "数字三（手型一）",
    "three2": "数字三（手型二）",
    "three3": "数字三（手型三）",
    "four": "数字四",
    "call": "电话手势",
    "mute": "嘘声手势",
    "rock": "摇滚手势",
    "little_finger": "小指伸出",
    "middle_finger": "中指伸出",
    "grabbing": "抓取手型",
    "grip": "捏握手型",
    "thumb_index": "拇指食指组合",
    "three_gun": "三指枪形手势",
    "holy": "双手合十",
    "timeout": "双手T形暂停",
    "xsign": "双手交叉X形",
    "hand_heart": "双手比心（手型一）",
    "hand_heart2": "双手比心（手型二）",
    "take_picture": "双手取景框",
    "thumb_index2": "双手拇指食指组合",
}

DEFAULT_HAND_DET_CONF = 0.25
DEFAULT_GESTURE_DET_CONF = 0.25
DEFAULT_TWO_HAND_CONF = 0.25
DEFAULT_WRIST_EXTRAPOLATION = 0.10
DEFAULT_KPT_CONF = 0.20


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Compute IoU between two xyxy boxes."""
    if not box_a or not box_b or len(box_a) != 4 or len(box_b) != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = (float(v) for v in box_a)
    bx1, by1, bx2, by2 = (float(v) for v in box_b)
    if ax2 <= ax1 or ay2 <= ay1 or bx2 <= bx1 or by2 <= by1:
        return 0.0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / union) if union > 0.0 else 0.0


def nms_class_agnostic(dets: Sequence[dict], iou_thr: float = 0.65) -> List[dict]:
    """Greedy class-agnostic NMS sorted by confidence descending."""
    kept: List[dict] = []
    for det in sorted(dets, key=lambda d: float(d.get("conf", 0.0)), reverse=True):
        if all(box_iou(det["box"], k["box"]) < iou_thr for k in kept):
            kept.append(det)
    return kept


def _box_center(box: Sequence[float]) -> np.ndarray:
    return np.array([0.5 * (float(box[0]) + float(box[2])),
                     0.5 * (float(box[1]) + float(box[3]))], dtype=np.float32)


def _wrist_anchor(
    keypoints: Optional[np.ndarray],
    side: str,
    *,
    alpha: float = DEFAULT_WRIST_EXTRAPOLATION,
    kpt_conf: float = DEFAULT_KPT_CONF,
) -> Optional[np.ndarray]:
    """Return the person's left/right wrist anchor in frame pixels.

    Uses a mild 0.10 forearm extrapolation rather than 0.35: a larger offset
    overshoots across the midline when hands press together (e.g. `holy`),
    swapping left and right.
    """
    if keypoints is None:
        return None
    k = np.asarray(keypoints, dtype=np.float32)
    if k.ndim != 2 or k.shape[0] < 17 or k.shape[1] < 3:
        return None
    w_idx = COCO_INDEX[f"{side}_wrist"]
    e_idx = COCO_INDEX[f"{side}_elbow"]
    if float(k[w_idx, 2]) < kpt_conf:
        return None
    wrist = k[w_idx, :2]
    if alpha > 0.0 and float(k[e_idx, 2]) >= kpt_conf:
        elbow = k[e_idx, :2]
        return wrist + float(alpha) * (wrist - elbow)
    return wrist.copy()


def _body_center_x(person: Optional[dict], img_w: int) -> float:
    if person is None:
        return img_w * 0.5
    kpts = person.get("keypoints")
    if kpts is not None:
        k = np.asarray(kpts, dtype=np.float32)
        if k.ndim == 2 and k.shape[0] >= 17 and k.shape[1] >= 3:
            ls, rs = COCO_INDEX["left_shoulder"], COCO_INDEX["right_shoulder"]
            if float(k[ls, 2]) >= DEFAULT_KPT_CONF and float(k[rs, 2]) >= DEFAULT_KPT_CONF:
                return 0.5 * (float(k[ls, 0]) + float(k[rs, 0]))
            nose = COCO_INDEX["nose"]
            if float(k[nose, 2]) >= DEFAULT_KPT_CONF:
                return float(k[nose, 0])
    box = person.get("box")
    if box is not None and len(box) == 4:
        return 0.5 * (float(box[0]) + float(box[2]))
    return img_w * 0.5


def assign_hands_lr(
    hand_dets: Sequence[dict],
    person: Optional[dict],
    img_w: int,
    *,
    alpha: float = DEFAULT_WRIST_EXTRAPOLATION,
    kpt_conf: float = DEFAULT_KPT_CONF,
) -> Dict[str, Optional[dict]]:
    """Assign up to two detected single-hand boxes to person's own left/right."""
    if not hand_dets:
        return {"left": None, "right": None}
    dets = nms_class_agnostic(hand_dets, iou_thr=0.65)[:2]

    kpts = person.get("keypoints") if person is not None else None
    l_anc = _wrist_anchor(kpts, "left", alpha=alpha, kpt_conf=kpt_conf)
    r_anc = _wrist_anchor(kpts, "right", alpha=alpha, kpt_conf=kpt_conf)
    cx_body = _body_center_x(person, img_w)

    if len(dets) == 1:
        det = dets[0]
        c = _box_center(det["box"])
        if l_anc is not None and r_anc is not None:
            side = "left" if np.linalg.norm(c - l_anc) <= np.linalg.norm(c - r_anc) else "right"
        elif l_anc is not None:
            diag = float(np.hypot(det["box"][2] - det["box"][0], det["box"][3] - det["box"][1]))
            if float(np.linalg.norm(c - l_anc)) < max(1.8 * diag, 0.25 * img_w):
                side = "left"
            else:
                side = "left" if c[0] >= cx_body else "right"
        elif r_anc is not None:
            diag = float(np.hypot(det["box"][2] - det["box"][0], det["box"][3] - det["box"][1]))
            if float(np.linalg.norm(c - r_anc)) < max(1.8 * diag, 0.25 * img_w):
                side = "right"
            else:
                side = "left" if c[0] >= cx_body else "right"
        else:
            # Facing-camera fallback: person's left hand is on image right.
            side = "left" if c[0] >= cx_body else "right"
        return {"left": det if side == "left" else None,
                "right": det if side == "right" else None}

    d0, d1 = dets[0], dets[1]
    c0, c1 = _box_center(d0["box"]), _box_center(d1["box"])
    if l_anc is not None and r_anc is not None:
        cost_0l = float(np.linalg.norm(c0 - l_anc) + np.linalg.norm(c1 - r_anc))
        cost_1l = float(np.linalg.norm(c1 - l_anc) + np.linalg.norm(c0 - r_anc))
        return {"left": d0, "right": d1} if cost_0l <= cost_1l else {"left": d1, "right": d0}
    if l_anc is not None:
        return ({"left": d0, "right": d1}
                if float(np.linalg.norm(c0 - l_anc)) <= float(np.linalg.norm(c1 - l_anc))
                else {"left": d1, "right": d0})
    if r_anc is not None:
        return ({"left": d1, "right": d0}
                if float(np.linalg.norm(c0 - r_anc)) <= float(np.linalg.norm(c1 - r_anc))
                else {"left": d0, "right": d1})
    return {"left": d0, "right": d1} if c0[0] >= c1[0] else {"left": d1, "right": d0}


def pick_primary_person_index(persons: Sequence[dict], hand_dets: Sequence[dict]) -> int:
    """Choose the index of the person most plausibly owning the top detected hands."""
    if len(persons) <= 1 or not hand_dets:
        return 0
    top_h = list(hand_dets)[:2]
    best_idx = 0
    best_score = -1e9
    for idx, person in enumerate(persons):
        box = person.get("box")
        if box is None or len(box) != 4:
            continue
        x1, y1, x2, y2 = (float(v) for v in box)
        pw, ph = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        ex1, ey1 = x1 - 0.25 * pw, y1 - 0.25 * ph
        ex2, ey2 = x2 + 0.25 * pw, y2 + 0.25 * ph
        kpts = person.get("keypoints")
        k_arr = np.asarray(kpts, dtype=np.float32) if kpts is not None else None
        inside = 0
        dist_sum = 0.0
        for hd in top_h:
            c = _box_center(hd["box"])
            if ex1 <= c[0] <= ex2 and ey1 <= c[1] <= ey2:
                inside += 1
            w_dists = []
            if k_arr is not None and k_arr.ndim == 2 and k_arr.shape[0] >= 17:
                for wi in (COCO_INDEX["left_wrist"], COCO_INDEX["right_wrist"]):
                    if float(k_arr[wi, 2]) >= DEFAULT_KPT_CONF:
                        w_dists.append(float(np.linalg.norm(c - k_arr[wi, :2])))
            dist_sum += (min(w_dists) / max(pw, ph)) if w_dists else 2.0
        score = inside * 10.0 - dist_sum + 0.5 * float(person.get("score", 0.0)) + 1e-7 * (pw * ph)
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def _round_box(box: Sequence[float], width: int, height: int) -> List[float]:
    x1 = max(0.0, min(float(width), float(box[0])))
    y1 = max(0.0, min(float(height), float(box[1])))
    x2 = max(0.0, min(float(width), float(box[2])))
    y2 = max(0.0, min(float(height), float(box[3])))
    if x2 <= x1 or y2 <= y1:
        return []
    return [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]


def _empty_hand_fields() -> dict:
    return {
        "hand_bboxes": {"left": [], "right": []},
        "hand_gestures": {"left": None, "right": None},
        "hand_gesture_scores": {"left": None, "right": None},
        "hands": {},
    }


def _decode_gesture_dets(
    outputs,
    meta,
    conf_thr: float,
    ox: float = 0.0,
    oy: float = 0.0,
) -> List[dict]:
    boxes, scores, class_ids = decode_detections(outputs, meta, conf_thr)
    out: List[dict] = []
    for box, score, cls_id in zip(boxes, scores, class_ids):
        cid = int(cls_id)
        raw_name = HAGRID_CLASS_NAMES[cid] if 0 <= cid < len(HAGRID_CLASS_NAMES) else None
        out.append({
            "box": [float(box[0]) + ox, float(box[1]) + oy,
                    float(box[2]) + ox, float(box[3]) + oy],
            "conf": float(score),
            "cls_id": cid,
            "raw_name": raw_name,
            "label": HAGRID_TO_CANONICAL.get(raw_name) if raw_name is not None else None,
        })
    out.sort(key=lambda d: d["conf"], reverse=True)
    return out


def _copy_hand_fields(fields: dict, allowed_set: Optional[set] = None) -> dict:
    """Return a fresh copy of detector hand fields, optionally filtered by `allowed_set`."""
    raw_boxes = fields.get("hand_bboxes") or {}
    raw_gests = fields.get("hand_gestures") or {}
    raw_scores = fields.get("hand_gesture_scores") or {}
    raw_pts = fields.get("hands_detected") or {}
    boxes = {side: list(raw_boxes.get(side) or []) for side in ("left", "right")}
    gests: Dict[str, Optional[str]] = {}
    scores: Dict[str, Optional[float]] = {}
    for side in ("left", "right"):
        lbl = raw_gests.get(side)
        if lbl is not None and allowed_set is not None and lbl not in allowed_set:
            gests[side] = None
            scores[side] = None
        else:
            gests[side] = lbl
            scores[side] = raw_scores.get(side) if lbl is not None else None
    pts = {
        side: [list(pt) for pt in pts_list]
        for side, pts_list in raw_pts.items()
        if pts_list
    }
    return {
        "hand_bboxes": boxes,
        "hand_gestures": gests,
        "hand_gesture_scores": scores,
        "hands_detected": pts,
        "hands": {side: [list(pt) for pt in pts_list] for side, pts_list in pts.items()},
    }


class HandGestureDetectorChannel:
    """Runs YOLOv10n hand + gesture detection and attaches left/right results to persons."""

    def __init__(
        self,
        gesture_session,
        hand_session=None,
        *,
        hand_conf: float = DEFAULT_HAND_DET_CONF,
        gesture_conf: float = DEFAULT_GESTURE_DET_CONF,
        two_hand_conf: float = DEFAULT_TWO_HAND_CONF,
        kpt_conf: float = DEFAULT_KPT_CONF,
        adaptive_zoom: bool = True,
        interval_s: float = 0.0,
        hold_s: float = 0.5,
    ):
        self._gesture_session = gesture_session
        self._hand_session = hand_session
        self._hand_conf = float(hand_conf)
        self._gesture_conf = float(gesture_conf)
        self._two_hand_conf = float(two_hand_conf)
        self._kpt_conf = float(kpt_conf)
        self._adaptive_zoom = bool(adaptive_zoom)
        self._interval_s = max(0.0, float(interval_s))
        self._hold_s = max(0.0, float(hold_s))
        self._last_t: float = -1e9
        self._last_by_person: Dict[Any, dict] = {}
        self._last_synthetic: Optional[dict] = None
        self.ran: int = 0
        self.skipped: Dict[str, int] = {"throttled": 0}
        self.last_error: Optional[str] = None

    @property
    def input_size(self) -> int:
        size = getattr(self._gesture_session, "input_size", (640, 640))
        return int(size[0]) if isinstance(size, (tuple, list)) else int(size)

    def reset(self) -> None:
        """Clear throttle/cache state when the owning stream node stops."""
        self._last_t = -1e9
        self._last_by_person.clear()
        self._last_synthetic = None

    def _reapply_held(
        self,
        persons: List[dict],
        now: float,
        *,
        allowed_set: Optional[set],
        synthesize_person: bool,
    ) -> int:
        """Reapply the most recent per-person detector result during a throttle interval."""
        hold_window = max(self._hold_s, self._interval_s) if self._hold_s > 0.0 else 0.0
        found = 0
        if persons:
            for p in persons:
                pid = p.get("id")
                entry = self._last_by_person.get(pid) if pid is not None else None
                if (
                    entry is not None
                    and hold_window > 0.0
                    and 0.0 <= (now - entry["seen"]) <= hold_window
                ):
                    fields = _copy_hand_fields(entry["fields"], allowed_set)
                    p.update(fields)
                    found += sum(
                        1 for side in ("left", "right")
                        if fields["hand_bboxes"][side] or fields["hand_gestures"][side]
                    )
                else:
                    p.update(_empty_hand_fields())
            return found

        if (
            synthesize_person
            and self._last_synthetic is not None
            and hold_window > 0.0
            and 0.0 <= (now - self._last_synthetic["seen"]) <= hold_window
        ):
            src = self._last_synthetic["person"]
            fields = _copy_hand_fields(self._last_synthetic["fields"], allowed_set)
            person = {
                "id": src["id"],
                "score": src["score"],
                "box": list(src["box"]),
                "keypoints": src["keypoints"].copy(),
                "position": list(src["position"]),
                "verdict": dict(src["verdict"]),
                **fields,
            }
            persons.append(person)
            found = sum(
                1 for side in ("left", "right")
                if fields["hand_bboxes"][side] or fields["hand_gestures"][side]
            )
        return found

    def _run_hands(self, frame: np.ndarray, ox: float = 0.0, oy: float = 0.0) -> List[dict]:
        if self._hand_session is None:
            return []
        outputs, meta = self._hand_session.infer(frame)
        boxes, scores, _ = decode_detections(outputs, meta, self._hand_conf)
        dets = [
            {"box": [float(b[0]) + ox, float(b[1]) + oy, float(b[2]) + ox, float(b[3]) + oy],
             "conf": float(s)}
            for b, s in zip(boxes, scores)
        ]
        dets.sort(key=lambda d: d["conf"], reverse=True)
        return dets

    def _run_gestures(self, frame: np.ndarray, ox: float = 0.0, oy: float = 0.0) -> List[dict]:
        outputs, meta = self._gesture_session.infer(frame)
        return _decode_gesture_dets(outputs, meta, self._gesture_conf, ox=ox, oy=oy)

    def update(
        self,
        persons: List[dict],
        frame: np.ndarray,
        now: float = 0.0,
        *,
        allowed: Optional[Iterable[str]] = None,
        synthesize_person: bool = False,
    ) -> dict:
        """Detect hands and gestures on `frame` and attach results to `persons` in place.

        Returns a diagnostic note dict (`{"ok": True, "found": ..., "ran": ..., ...}`).
        """
        height, width = frame.shape[:2]
        allowed_set = set(allowed) if allowed is not None else None
        if self._interval_s > 0.0 and 0.0 <= (now - self._last_t) < self._interval_s:
            self.skipped["throttled"] += 1
            held_found = self._reapply_held(
                persons,
                now,
                allowed_set=allowed_set,
                synthesize_person=synthesize_person,
            )
            return {
                "ok": True,
                "backend": "detector",
                "throttled": True,
                "found": held_found,
                "ran": self.ran,
            }

        try:
            hand_dets = self._run_hands(frame)
            gest_dets = self._run_gestures(frame)
        except Exception as error:  # noqa: BLE001 — never crash body pose on hand failure
            self.last_error = f"{type(error).__name__}: {error}"
            self._last_t = now
            self._last_by_person.clear()
            self._last_synthetic = None
            log.warning("[pose] hand gesture detector failed: %s", error)
            for p in persons:
                p.update(_empty_hand_fields())
            return {"ok": False, "error": self.last_error, "ran": self.ran}

        self.ran += 1
        self._last_t = now
        self.last_error = None
        self._last_by_person.clear()
        self._last_synthetic = None

        # Initialise empty hand fields on all persons first.
        for p in persons:
            p.update(_empty_hand_fields())

        # Adaptive person-crop zoom can rescue distant small hands even if the
        # full-frame pass missed both hands and gestures.
        if not hand_dets and not gest_dets:
            if not (self._adaptive_zoom and persons):
                return {"ok": True, "backend": "detector", "found": 0, "ran": self.ran, "zoomed": False}

        # Pick or synthesize the target person owning these hands.
        synthesized = False
        if persons:
            p_idx = (
                pick_primary_person_index(persons, hand_dets or gest_dets)
                if (hand_dets or gest_dets)
                else 0
            )
            if p_idx != 0:
                # Place the gesturing person first so single-subject callers
                # reading persons[0] always receive the active person. Keep
                # PoseTracker track ids intact on stream frames.
                persons.insert(0, persons.pop(p_idx))
                if all("track" not in p for p in persons):
                    for rank, p in enumerate(persons, start=1):
                        p["id"] = rank
            person = persons[0]
        elif synthesize_person:
            synthesized = True
            all_boxes = [d["box"] for d in (hand_dets or gest_dets)]
            ux1 = min(b[0] for b in all_boxes)
            uy1 = min(b[1] for b in all_boxes)
            ux2 = max(b[2] for b in all_boxes)
            uy2 = max(b[3] for b in all_boxes)
            half_w, half_h = max(width / 2.0, 1.0), max(height / 2.0, 1.0)
            cx, cy = 0.5 * (ux1 + ux2), 0.5 * (uy1 + uy2)
            person = {
                "id": 1,
                "score": round(float((hand_dets or gest_dets)[0]["conf"]), 2),
                "box": [ux1, uy1, ux2, uy2],
                "keypoints": np.zeros((17, 3), dtype=np.float32),
                "position": [round((cx - half_w) / half_w, 3),
                             round((cy - half_h) / half_h, 3)],
                "verdict": {"posture": "unknown", "posture_confidence": 0.0, "evidence": {}},
                **_empty_hand_fields(),
            }
            persons.append(person)
        else:
            return {"ok": True, "backend": "detector", "found": 0, "ran": self.ran, "zoomed": False}

        # Adaptive person-crop zoom when the hand is tiny, missed, or borderline.
        zoomed = False
        if self._adaptive_zoom and person.get("box") is not None:
            max_dim = float(max(width, height, 1))
            px1, py1, px2, py2 = (float(v) for v in person["box"])
            p_frac = max(px2 - px1, py2 - py1) / max_dim
            top_g_conf = gest_dets[0]["conf"] if gest_dets else 0.0
            smallest_h_frac = min(
                (max(d["box"][2] - d["box"][0], d["box"][3] - d["box"][1]) / max_dim
                 for d in hand_dets[:2]),
                default=0.0,
            )
            if p_frac < 0.70 and (not hand_dets or smallest_h_frac < 0.09 or top_g_conf < 0.72):
                pad = int(0.18 * max(px2 - px1, py2 - py1))
                cx1, cy1 = max(0, int(px1) - pad), max(0, int(py1) - pad)
                cx2, cy2 = min(width, int(px2) + pad), min(height, int(py2) + pad)
                if (cx2 - cx1) >= 64 and (cy2 - cy1) >= 64:
                    crop = frame[cy1:cy2, cx1:cx2]
                    try:
                        if self._hand_session is not None and (not hand_dets or smallest_h_frac < 0.09):
                            zoom_hands = self._run_hands(crop, ox=cx1, oy=cy1)
                            if zoom_hands and (len(zoom_hands) > len(hand_dets) or not hand_dets):
                                hand_dets = zoom_hands
                                zoomed = True
                        zoom_dets = self._run_gestures(crop, ox=cx1, oy=cy1)
                        if zoom_dets and (not gest_dets or zoom_dets[0]["conf"] >= top_g_conf):
                            gest_dets = zoom_dets
                            zoomed = True
                    except Exception as error:  # noqa: BLE001
                        log.debug("[pose] adaptive gesture zoom skipped: %s", error)

        if not hand_dets and not gest_dets:
            return {"ok": True, "backend": "detector", "found": 0, "ran": self.ran, "zoomed": zoomed}

        # Assign individual single-hand boxes to person's left/right.
        box_source = (
            hand_dets if self._hand_session is not None
            else [d for d in gest_dets if d["label"] not in TWO_HAND_GESTURE_SET]
        )
        assigned = assign_hands_lr(box_source, person, width, kpt_conf=self._kpt_conf)

        pred_boxes: Dict[str, List[float]] = {
            side: (_round_box(assigned[side]["box"], width, height)
                   if assigned[side] is not None else [])
            for side in ("left", "right")
        }
        pred_gestures: Dict[str, Optional[str]] = {"left": None, "right": None}
        pred_scores: Dict[str, Optional[float]] = {"left": None, "right": None}
        pred_hands_pts: Dict[str, List[List[float]]] = {}

        def _accept(label: Optional[str]) -> Optional[str]:
            if label is None:
                return None
            if allowed_set is not None and label not in allowed_set:
                return None
            return label

        # Check two-hand combination gestures first (united box).
        th_dets = [
            d for d in gest_dets
            if d["label"] in TWO_HAND_GESTURE_SET and d["conf"] >= self._two_hand_conf
        ]
        if th_dets:
            best_th = th_dets[0]
            lbl = _accept(best_th["label"])
            sc = round(float(best_th["conf"]), 4) if lbl is not None else None
            pred_gestures["left"] = lbl
            pred_gestures["right"] = lbl
            pred_scores["left"] = sc
            pred_scores["right"] = sc
        else:
            sh_dets = [d for d in gest_dets if d["label"] not in TWO_HAND_GESTURE_SET]
            for side in ("left", "right"):
                box = pred_boxes[side]
                if not box:
                    continue
                best_gd = None
                best_iou = 0.25
                for gd in sh_dets:
                    iou = box_iou(box, gd["box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gd = gd
                if best_gd is not None:
                    lbl = _accept(best_gd["label"])
                    pred_gestures[side] = lbl
                    pred_scores[side] = round(float(best_gd["conf"]), 4) if lbl is not None else None

            # Recover any gesturing hand found by gesture_det that hand_det missed.
            if self._hand_session is not None:
                unmatched = [
                    gd for gd in nms_class_agnostic(sh_dets, 0.60)
                    if all(box_iou(gd["box"], pred_boxes[s]) < 0.30
                           for s in ("left", "right") if pred_boxes[s])
                ]
                if unmatched and (not pred_boxes["left"] or not pred_boxes["right"]):
                    combined = [assigned[s] for s in ("left", "right") if assigned[s] is not None] + unmatched
                    re_assigned = assign_hands_lr(combined, person, width, kpt_conf=self._kpt_conf)
                    for side in ("left", "right"):
                        if not pred_boxes[side] and re_assigned[side] is not None:
                            d = re_assigned[side]
                            pred_boxes[side] = _round_box(d["box"], width, height)
                            if d in unmatched:
                                lbl = _accept(d["label"])
                                pred_gestures[side] = lbl
                                pred_scores[side] = round(float(d["conf"]), 4) if lbl is not None else None

        # Populate representative hand center keypoint per detected side.
        for side in ("left", "right"):
            box = pred_boxes[side]
            if box:
                cx = round(0.5 * (box[0] + box[2]), 1)
                cy = round(0.5 * (box[1] + box[3]), 1)
                sc = pred_scores[side]
                if sc is None:
                    det = assigned[side]
                    sc = round(float(det["conf"]), 2) if det is not None else 0.5
                else:
                    sc = round(float(sc), 2)
                pred_hands_pts[side] = [[cx, cy, sc]]

        person["hand_bboxes"] = pred_boxes
        person["hand_gestures"] = pred_gestures
        person["hand_gesture_scores"] = pred_scores
        person["hands_detected"] = pred_hands_pts
        # Keep `hands` compatible with pose.py photo/stream formatting when RTMPose is not running.
        person["hands"] = pred_hands_pts

        found = sum(1 for side in ("left", "right") if pred_boxes[side])
        has_any = found > 0 or any(pred_gestures[side] is not None for side in ("left", "right"))
        if has_any:
            cached_fields = _copy_hand_fields(person)
            if synthesized:
                self._last_synthetic = {
                    "person": {
                        "id": person["id"],
                        "score": person["score"],
                        "box": list(person["box"]),
                        "keypoints": person["keypoints"].copy(),
                        "position": list(person["position"]),
                        "verdict": dict(person["verdict"]),
                    },
                    "fields": cached_fields,
                    "seen": now,
                }
            else:
                pid = person.get("id")
                if pid is not None:
                    self._last_by_person[pid] = {
                        "fields": cached_fields,
                        "seen": now,
                    }

        return {
            "ok": True,
            "backend": "detector",
            "found": found,
            "ran": self.ran,
            "zoomed": zoomed,
        }


def detector_gesture_catalogue() -> List[dict]:
    """Structured vocabulary for `pose` `list_actions`."""
    return [
        {
            "name": name,
            "name_zh": DETECTOR_GESTURE_LABELS_ZH.get(name, name),
            "two_hand": name in TWO_HAND_GESTURE_SET,
        }
        for name in ALL_DETECTOR_GESTURES
    ]
