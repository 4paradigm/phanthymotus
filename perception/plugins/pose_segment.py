"""Recognize a received video segment once, after the input becomes quiet.

The JPEG buffer is independent of the live pose frame-rate gate. All selected
observations came from this subscription; no file boundaries or future input
are needed. The frozen action models use the same person and hand crops.
"""
from collections import deque
from pathlib import Path
import json
import logging
import math
import threading

import numpy as np

from plugins.action_vocab import OFFICIAL_126_CLASSES
from plugins.pose_action import PoseTracker, body_height
from plugins.pose_stgcn import SkeletonActionBackend, looks_like_probabilities, softmax


log = logging.getLogger(__name__)

_FINGER_IDS = frozenset({84, 85, 86, 87, 88, 89, 90, 95})
_DESKTOP_IDS = frozenset({29, 32, 35, 40, 48, 49, 54, 55, 56, 57, 58, 60})
_KITCHEN_IDS = frozenset(range(13, 29))
_CONTROL_IDS = frozenset(range(103, 117))


def _build_router_group_masks():
    masks = np.zeros((5, 126), dtype=np.float64)
    for cid in range(1, 127):
        if cid in _FINGER_IDS:
            masks[0, cid - 1] = 1.0
        elif cid in _DESKTOP_IDS:
            masks[1, cid - 1] = 1.0
        elif cid in _KITCHEN_IDS:
            masks[2, cid - 1] = 1.0
        elif cid in _CONTROL_IDS:
            masks[3, cid - 1] = 1.0
        else:
            masks[4, cid - 1] = 1.0
    return masks


_ROUTER_GROUP_MASKS = _build_router_group_masks()


class _SkeletonFrame:
    """Tracker/head fields without posture geometry that neither consumes."""

    __slots__ = ('t', 'box', 'keypoints', 'image_size', 'height')

    def __init__(self, t, box, keypoints, min_conf, image_size=None):
        shape = getattr(keypoints, 'shape', None)
        if shape is None or len(shape) != 2 or shape[0] != 17 or shape[1] < 3:
            raise ValueError(f'keypoints must be (17, 3+); got {shape}')
        self.t = float(t)
        self.box = [float(v) for v in box]
        self.keypoints = keypoints
        self.image_size = (tuple(int(v) for v in image_size)
                           if image_size and image_size[0] and image_size[1]
                           else None)
        self.height = body_height(box)


class _ImageShape:
    """Zero-byte stand-in carrying `.shape` when pixel data is unused."""

    __slots__ = ('shape',)

    def __init__(self, shape):
        self.shape = tuple(int(v) for v in shape)


class SegmentBuffer:
    def __init__(self, quiet_s=.4, max_s=30., max_bytes=12*1024**2, max_patches=24,
                 short_min_frames=0, short_quiet_s=0.0):
        self.quiet_s = float(quiet_s)
        self.short_min_frames = max(0, int(short_min_frames))
        self.short_quiet_s = max(self.quiet_s, float(short_quiet_s)) if short_quiet_s else self.quiet_s
        self.max_s = float(max_s)
        self.max_bytes = int(max_bytes)
        self.max_patches = int(max_patches)
        self._lock = threading.Lock()
        self._frames = deque()
        self._ready = deque(maxlen=2)
        self._by_time = {}
        self._bytes = 0
        self._received = 0
        self._segment_gen = 0
        self.dropped_segments = 0

    def _effective_quiet_s(self):
        if (self.short_min_frames > 0
                and self.short_quiet_s > self.quiet_s
                and len(self._frames) < self.short_min_frames):
            return self.short_quiet_s
        return self.quiet_s

    def _snapshot(self):
        return (list(self._frames), {'received_frames': self._received,
                'retained_frames': len(self._frames),
                'truncated_frames': self._received-len(self._frames),
                'segment_gen': self._segment_gen})

    def _take(self):
        result = self._snapshot()
        self._frames.clear()
        self._bytes = self._received = 0
        self._segment_gen += 1
        return result

    def snapshot_posed(self):
        """Return (segment_gen, posed_records) for the currently accumulating segment."""
        with self._lock:
            posed = [r for r in self._frames if r[2] is not None]
            return self._segment_gen, posed

    def current_gen(self):
        with self._lock:
            return self._segment_gen

    def append(self, payload, now):
        with self._lock:
            if self._frames and now-self._frames[-1][0] >= self._effective_quiet_s():
                if len(self._ready) == self._ready.maxlen:
                    self.dropped_segments += 1
                    self._forget(self._ready.popleft()[0])
                self._ready.append(self._take())
            data = bytes(payload)
            record = [float(now), data, None]
            self._frames.append(record)
            self._by_time[float(now)] = record
            self._bytes += len(data)
            self._received += 1
            while self._frames and (self._bytes > self.max_bytes or
                                    now-self._frames[0][0] > self.max_s):
                removed = self._frames.popleft()
                self._bytes -= len(removed[1])
                self._by_time.pop(removed[0], None)

    def record_pose(self, now, boxes, scores, keypoints, image_shape, rgb_patch=None):
        """Attach the real worker result (and optional pre-cropped RGB patch) to its record."""
        with self._lock:
            record = self._by_time.get(float(now))
            if record is not None:
                base = (np.array(boxes, copy=True), np.array(scores, copy=True),
                        np.array(keypoints, copy=True), tuple(image_shape[:2]))
                record[2] = (*base, rgb_patch) if rgb_patch is not None else base
                if rgb_patch is not None and self.max_patches > 0:
                    patched = [r for r in self._frames
                               if r[2] is not None and len(r[2]) > 4 and r[2][4] is not None]
                    if len(patched) > self.max_patches:
                        # Drop every other interior patch while preserving endpoints and all poses.
                        for r in patched[1:-1:2]:
                            r[2] = r[2][:4]

    def drop_payload(self, now):
        """Release JPEG bytes for a frame evicted from the pose queue before inference."""
        with self._lock:
            record = self._by_time.get(float(now))
            if record is not None and record[2] is None and record[1]:
                self._bytes -= len(record[1])
                record[1] = b''

    def pop_if_same_ready(self, now, last_t):
        """Confirm an in-flight preview immediately if quiet_s elapsed with no new frames."""
        with self._lock:
            if (not self._ready and self._frames
                    and self._frames[-1][0] == float(last_t)
                    and float(now) - float(last_t) >= self._effective_quiet_s()):
                result = self._take()
                self._forget(result[0])
                return result
        return None

    def _forget(self, frames):
        for record in frames:
            self._by_time.pop(record[0], None)

    def next_wait_s(self, now, preview_s=None, previewed_last_t=None):
        """Return seconds until the active segment reaches preview or quiet confirm."""
        with self._lock:
            if self._ready:
                return 0.0
            if not self._frames:
                return None
            last_t = self._frames[-1][0]
            target_s = self._effective_quiet_s()
            if (preview_s is not None and 0.0 < float(preview_s) < target_s
                    and (previewed_last_t is None or last_t != float(previewed_last_t))):
                target_s = float(preview_s)
            return max(0.0, last_t + target_s - float(now))

    def snapshot_if_quiet(self, now, quiet_s):
        """Read a provisional segment without closing it on a brief pause."""
        with self._lock:
            if self._frames and now-self._frames[-1][0] >= quiet_s:
                return self._snapshot()
        return None

    def pop_ready(self, now):
        with self._lock:
            if self._ready:
                result = self._ready.popleft()
                self._forget(result[0])
                return result
            if self._frames and now-self._frames[-1][0] >= self._effective_quiet_s():
                result = self._take()
                self._forget(result[0])
                return result
        return None

    def reset(self):
        with self._lock:
            self._frames.clear()
            self._ready.clear()
            self._by_time.clear()
            self._bytes = self._received = 0
            self._segment_gen += 1
            self.dropped_segments = 0


def crop_square(rgb, cx, cy, side, out_hw=192):
    import cv2
    h, w = rgb.shape[:2]
    half = max(side*.5, 24.)
    if half*2 >= max(h, w):
        return cv2.resize(rgb, (out_hw, out_hw), interpolation=cv2.INTER_LINEAR)
    x0, y0 = int(round(cx-half)), int(round(cy-half))
    x1, y1 = x0+int(round(2*half)), y0+int(round(2*half))
    if x0 < 0:
        x1, x0 = min(w, x1-x0), 0
    if y0 < 0:
        y1, y0 = min(h, y1-y0), 0
    if x1 > w:
        x0, x1 = max(0, x0-(x1-w)), w
    if y1 > h:
        y0, y1 = max(0, y0-(y1-h)), h
    patch = rgb[y0:y1, x0:x1]
    if patch.size == 0:
        patch = rgb
    return cv2.resize(patch, (out_hw, out_hw), interpolation=cv2.INTER_LINEAR)


def rgb_frame_indices(length, n_per_view=4):
    # Preserve the saved models' 16-observation cache -> observations per view.
    sixteen = np.linspace(0, length-1, 16).round().astype(int)
    return sixteen[np.linspace(0, 15, n_per_view).round().astype(int)]


def _extended_wrists(kp, width, height, extension=.35):
    points = []
    for wi, ei in [(9, 7), (10, 8)]:
        if kp[wi, 2] > .20:
            x, y = float(kp[wi, 0]), float(kp[wi, 1])
            if kp[ei, 2] > .20:
                x += extension*(x-float(kp[ei, 0]))
                y += extension*(y-float(kp[ei, 1]))
            points.append((wi, float(np.clip(x, 0, width-1)), float(np.clip(y, 0, height-1))))
    return points


def _clip_crop_mode_e(p0, width, height, side, dist_thresh=.12):
    dists = []
    two_frames = 0
    for kp in p0:
        pts = _extended_wrists(kp, width, height, extension=.35)
        if len(pts) == 2:
            two_frames += 1
            dists.append(float(np.hypot(pts[0][1]-pts[1][1], pts[0][2]-pts[1][2]))/max(side, 1.))
    if two_frames >= max(1, int(.6*len(p0))) and dists and float(np.median(dists)) <= dist_thresh:
        return 'B'
    return 'C'


def _hand_crop_for_frame_e(mode, kp, cx, cy, side, width, height):
    pts = _extended_wrists(kp, width, height, extension=.35)
    if mode == 'B':
        if len(pts) == 2:
            (_, x1, y1), (_, x2, y2) = pts
            dx, dy = abs(x1-x2), abs(y1-y2)
            hx, hy = .5*(x1+x2), .5*(y1+y2)
            hs = min(max(max(dx, dy)+.28*side, .32*side, 96.), 1.15*side)
            return hx, hy, hs
        if len(pts) == 1:
            return pts[0][1], pts[0][2], max(.32*side, 96.)
        return cx, cy-.12*side, max(.42*side, 96.)

    anchors = [(x, y) for _, x, y in pts]
    sh_pts = [(float(kp[i, 0]), float(kp[i, 1])) for i in (5, 6) if kp[i, 2] > .20]
    if len(sh_pts) == 2:
        sx, sy = .5*(sh_pts[0][0]+sh_pts[1][0]), .5*(sh_pts[0][1]+sh_pts[1][1])
        anchors.append((sx, sy+.16*side))
    elif sh_pts:
        anchors.append((sh_pts[0][0], sh_pts[0][1]+.16*side))
    if anchors:
        ax = [p[0] for p in anchors]
        ay = [p[1] for p in anchors]
        hx, hy = .5*(min(ax)+max(ax)), .5*(min(ay)+max(ay))
        span = max(max(ax)-min(ax), max(ay)-min(ay))
        hs = min(max(span+.24*side, .52*side, 128.), 1.15*side)
        return hx, hy, hs
    return cx, cy-.08*side, max(.55*side, 128.)


def build_rgb_input(rgb_frames, keypoints, crop_mode='A'):
    """RGB source images plus the whole segment's primary COCO-17 poses."""
    p0 = np.asarray(keypoints, np.float32)
    h, w = rgb_frames[0].shape[:2]
    cx, cy, side = w*.5, h*.5, float(min(h, w))
    valid = p0[..., 2] > .20
    if valid.any():
        xs, ys = p0[..., 0][valid], p0[..., 1][valid]
        cx = .5*(float(np.percentile(xs, 5))+float(np.percentile(xs, 95)))
        cy = .5*(float(np.percentile(ys, 3))+float(np.percentile(ys, 97)))
        side = max(float(np.percentile(xs, 97)-np.percentile(xs, 3)),
                   float(np.percentile(ys, 98)-np.percentile(ys, 2)))*1.30
    use_crop_e = str(crop_mode or 'A').upper() == 'E'
    if use_crop_e:
        p_side = max(side, 120.)
        sub_mode = _clip_crop_mode_e(p0, w, h, p_side)
        people, hands = [], []
        for rgb, fi in zip(rgb_frames, rgb_frame_indices(len(p0), len(rgb_frames))):
            ch, cw = rgb.shape[:2]
            people.append(crop_square(rgb, cx, cy, p_side).transpose(2, 0, 1))
            hx, hy, hs = _hand_crop_for_frame_e(sub_mode, p0[fi], cx, cy, p_side, cw, ch)
            hands.append(crop_square(rgb, hx, hy, hs).transpose(2, 0, 1))
    else:
        wrist_scores = []
        for wi in [9, 10]:
            ok = p0[:, wi, 2] > .25
            wrist_scores.append(float(-p0[ok, wi, 1].mean()) if ok.any() else -9999.)
        prefer_left = wrist_scores[0] >= wrist_scores[1]
        people, hands = [], []
        for rgb, fi in zip(rgb_frames, rgb_frame_indices(len(p0), len(rgb_frames))):
            h, w = rgb.shape[:2]
            people.append(crop_square(rgb, cx, cy, max(side, 120.)).transpose(2, 0, 1))
            hx, hy, hs = cx, cy-.12*side, max(.42*side, 96.)
            kp = p0[fi]
            points = []
            for wi, ei in [(9, 7), (10, 8)]:
                if kp[wi, 2] > .20:
                    x, y = float(kp[wi, 0]), float(kp[wi, 1])
                    if kp[ei, 2] > .20:
                        x += .30*(x-float(kp[ei, 0]))
                        y += .30*(y-float(kp[ei, 1]))
                    points.append((np.clip(x, 0, w-1), np.clip(y, 0, h-1)))
            if len(points) == 2:
                (x1, y1), (x2, y2) = points
                distance = float(np.hypot(x1-x2, y1-y2))
                if distance <= .50*side:
                    hx, hy = .5*(x1+x2), .5*(y1+y2)
                    hs = max(distance+.34*side, .38*side, 96.)
                else:
                    hx, hy = points[0 if prefer_left else 1]
                    hs = max(.38*side, 96.)
            elif points:
                hx, hy = points[0]
                hs = max(.38*side, 96.)
            hands.append(crop_square(rgb, hx, hy, hs).transpose(2, 0, 1))
    frames = np.concatenate([people, hands]).astype(np.float32)
    mean = np.array([123.675, 116.28, 103.53], np.float32)[None, :, None, None]
    std = np.array([58.395, 57.12, 57.375], np.float32)[None, :, None, None]
    return ((frames-mean)/std)[None]


def build_rgb_views_from_crops(people, hands):
    mean = np.array([123.675, 116.28, 103.53], np.float32)[None, :, None, None]
    std = np.array([58.395, 57.12, 57.375], np.float32)[None, :, None, None]
    vp = ((np.asarray(people, np.float32)-mean)/std)[None]
    vh = ((np.asarray(hands, np.float32)-mean)/std)[None]
    return vp, vh


def build_ctx_view_from_crops(ctx_crops, target_len):
    mean = np.array([123.675, 116.28, 103.53], np.float32)[None, :, None, None]
    std = np.array([58.395, 57.12, 57.375], np.float32)[None, :, None, None]
    vc = ((np.asarray(ctx_crops, np.float32)-mean)/std)[None]
    ctx_k = int(vc.shape[1])
    if ctx_k <= 0 or ctx_k == int(target_len):
        return vc
    rep = max(1, (int(target_len) + ctx_k - 1) // ctx_k)
    return np.repeat(vc, rep, axis=1)[:, :int(target_len)]


def prepare_rgb_crop_geometry(keypoints, frame_hw, crop_mode='A'):
    p0 = np.asarray(keypoints, np.float32)
    h, w = int(frame_hw[0]), int(frame_hw[1])
    cx, cy, side = w*.5, h*.5, float(min(h, w))
    valid = p0[..., 2] > .20
    if valid.any():
        xs, ys = p0[..., 0][valid], p0[..., 1][valid]
        cx = .5*(float(np.percentile(xs, 5))+float(np.percentile(xs, 95)))
        cy = .5*(float(np.percentile(ys, 3))+float(np.percentile(ys, 97)))
        side = max(float(np.percentile(xs, 97)-np.percentile(xs, 3)),
                   float(np.percentile(ys, 98)-np.percentile(ys, 2)))*1.30
    use_crop_e = str(crop_mode or 'A').upper() == 'E'
    p_side = max(side, 120.)
    if use_crop_e:
        return p0, cx, cy, side, p_side, True, _clip_crop_mode_e(p0, w, h, p_side), True
    wrist_scores = []
    for wi in [9, 10]:
        ok = p0[:, wi, 2] > .25
        wrist_scores.append(float(-p0[ok, wi, 1].mean()) if ok.any() else -9999.)
    return p0, cx, cy, side, p_side, False, 'A', (wrist_scores[0] >= wrist_scores[1])


def crop_rgb_pair_for_frame(rgb, fi, geom):
    p0, cx, cy, side, p_side, use_crop_e, sub_mode, prefer_left = geom
    ch, cw = rgb.shape[:2]
    p_crop = crop_square(rgb, cx, cy, p_side).transpose(2, 0, 1)
    if use_crop_e:
        hx, hy, hs = _hand_crop_for_frame_e(sub_mode, p0[fi], cx, cy, p_side, cw, ch)
    else:
        hx, hy, hs = cx, cy-.12*side, max(.42*side, 96.)
        kp = p0[fi]
        points = []
        for wi, ei in [(9, 7), (10, 8)]:
            if kp[wi, 2] > .20:
                x, y = float(kp[wi, 0]), float(kp[wi, 1])
                if kp[ei, 2] > .20:
                    x += .30*(x-float(kp[ei, 0]))
                    y += .30*(y-float(kp[ei, 1]))
                points.append((np.clip(x, 0, cw-1), np.clip(y, 0, ch-1)))
        if len(points) == 2:
            (x1, y1), (x2, y2) = points
            distance = float(np.hypot(x1-x2, y1-y2))
            if distance <= .50*side:
                hx, hy = .5*(x1+x2), .5*(y1+y2)
                hs = max(distance+.34*side, .38*side, 96.)
            else:
                hx, hy = points[0 if prefer_left else 1]
                hs = max(.38*side, 96.)
        elif points:
            hx, hy = points[0]
            hs = max(.38*side, 96.)
    h_crop = crop_square(rgb, hx, hy, hs).transpose(2, 0, 1)
    return p_crop, h_crop


def _crop_square_patch(bgr, cx, cy, side, out_hw):
    """Crop square region around (cx, cy, side) from BGR and return (rgb, meta)."""
    import cv2
    h, w = bgr.shape[:2]
    half = .5*side
    if side >= max(h, w):
        rx0, ry0, rx1, ry1 = 0, 0, w, h
    else:
        rx0, ry0 = int(round(cx-half)), int(round(cy-half))
        s_int = int(round(side))
        rx1, ry1 = rx0+s_int, ry0+s_int
        if rx0 < 0:
            rx1, rx0 = min(w, rx1-rx0), 0
        if ry0 < 0:
            ry1, ry0 = min(h, ry1-ry0), 0
        if rx1 > w:
            rx0, rx1 = max(0, rx0-(rx1-w)), w
        if ry1 > h:
            ry0, ry1 = max(0, ry0-(ry1-h)), h
    sub = bgr[ry0:ry1, rx0:rx1]
    if sub.size == 0:
        rx0, ry0, sub = 0, 0, bgr
    sh, sw = sub.shape[:2]
    if max(sh, sw) > out_hw:
        scale = out_hw/float(max(sh, sw))
        nw = max(1, int(round(sw*scale)))
        nh = max(1, int(round(sh*scale)))
        sub = cv2.resize(sub, (nw, nh), interpolation=cv2.INTER_AREA)
    else:
        nw, nh = sw, sh
    rgb = cv2.cvtColor(sub, cv2.COLOR_BGR2RGB)
    return rgb, (rx0, ry0, nw/float(max(sw, 1)), nh/float(max(sh, 1)))


def _primary_person_geom(boxes, scores, keypoints, h, w):
    boxes_arr = np.asarray(boxes, np.float32)
    scores_arr = np.asarray(scores, np.float32)
    kps_arr = np.asarray(keypoints, np.float32)
    if len(boxes_arr) == 0:
        return None, .5*w, .5*h, float(max(h, w)), 0.0
    areas = (np.maximum(1., (boxes_arr[:, 2]-boxes_arr[:, 0])
                        * (boxes_arr[:, 3]-boxes_arr[:, 1]))
             / max(1., float(w*h)))
    best = int((scores_arr*(.5+areas)).argmax())
    b0 = boxes_arr[best]
    x0, y0, x1, y1 = float(b0[0]), float(b0[1]), float(b0[2]), float(b0[3])
    kp = kps_arr[best] if len(kps_arr) > best else None
    if kp is not None:
        valid = kp[:, 2] > .20
        if valid.any():
            x0 = min(x0, float(kp[valid, 0].min()))
            x1 = max(x1, float(kp[valid, 0].max()))
            y0 = min(y0, float(kp[valid, 1].min()))
            y1 = max(y1, float(kp[valid, 1].max()))
    body_side = max(x1-x0, y1-y0, 120.)
    return kp, .5*(x0+x1), .5*(y0+y1), max(body_side, 160.)*1.55, body_side


def make_person_roi_patch(bgr, boxes, scores, keypoints, roi_hw=320):
    """Pre-crop a 1.55x RGB square ROI at <=roi_hw while frame is already decoded."""
    h, w = bgr.shape[:2]
    _, cx, cy, side, _ = _primary_person_geom(boxes, scores, keypoints, h, w)
    return _crop_square_patch(bgr, cx, cy, side, roi_hw)


def make_context_patch(bgr, ctx_hw=192):
    """Build a letterboxed full-frame RGB context patch (uint8 HxWx3)."""
    import cv2
    from plugins.vision_runtime import letterbox
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    canvas, _ = letterbox(rgb, int(ctx_hw), int(ctx_hw))
    return canvas


def make_dual_roi_patch(bgr, boxes, scores, keypoints, person_hw=224, hand_hw=192,
                        ctx_hw=0):
    """Pre-crop native-res person (person_hw), hand (hand_hw), and optional context RGB patches."""
    h, w = bgr.shape[:2]
    kp, cx, cy, p_full_side, body_side = _primary_person_geom(
        boxes, scores, keypoints, h, w)
    p_pair = _crop_square_patch(bgr, cx, cy, p_full_side, person_hw)
    ctx_rgb = make_context_patch(bgr, ctx_hw=ctx_hw) if int(ctx_hw) > 0 else None
    def _pack(h_pair):
        return (p_pair, h_pair, ctx_rgb) if ctx_rgb is not None else (p_pair, h_pair)
    if kp is None or body_side <= 0.0:
        return _pack(None)
    p_side_est = max(body_side*1.30, 120.)
    pts = _extended_wrists(kp, w, h, extension=.35)
    sh_pts = [(float(kp[i, 0]), float(kp[i, 1])) for i in (5, 6) if kp[i, 2] > .20]
    anchors = [(x, y) for _, x, y in pts]
    if len(sh_pts) == 2:
        anchors.append((.5*(sh_pts[0][0]+sh_pts[1][0]),
                        .5*(sh_pts[0][1]+sh_pts[1][1])+.16*p_side_est))
    elif sh_pts:
        anchors.append((sh_pts[0][0], sh_pts[0][1]+.16*p_side_est))
    if not anchors:
        return _pack(None)
    ax, ay = [p[0] for p in anchors], [p[1] for p in anchors]
    hx, hy = .5*(min(ax)+max(ax)), .5*(min(ay)+max(ay))
    span = max(max(ax)-min(ax), max(ay)-min(ay))
    hs = min(max(span+.32*p_side_est, .62*p_side_est, 144.), 1.15*p_side_est)
    if hs >= .88*p_full_side:
        return _pack(None)
    return _pack(_crop_square_patch(bgr, hx, hy, hs, hand_hw))


def _patch_covers_square(patch_rgb, roi_meta, cx, cy, side, cw, ch, tol=4.0):
    rx0, ry0, sx, sy = roi_meta
    px1 = rx0 + patch_rgb.shape[1] / max(sx, 1e-6)
    py1 = ry0 + patch_rgb.shape[0] / max(sy, 1e-6)
    half = max(side * .5, 24.)
    tx0, tx1 = max(0.0, cx - half), min(float(cw), cx + half)
    ty0, ty1 = max(0.0, cy - half), min(float(ch), cy + half)
    return (rx0 <= tx0 + tol and ry0 <= ty0 + tol
            and px1 >= tx1 - tol and py1 >= ty1 - tol)


def crop_from_roi_patch(rgb_patch, roi_meta, cx, cy, side, out_hw=192):
    rx0, ry0, sx, sy = roi_meta
    return crop_square(rgb_patch, (cx-rx0)*sx, (cy-ry0)*sy,
                       side*.5*(sx+sy), out_hw=out_hw)


def crop_rgb_pair_from_patch(rgb_patch, roi_meta, fi, geom, frame_hw):
    p0, cx, cy, side, p_side, use_crop_e, sub_mode, prefer_left = geom
    ch, cw = int(frame_hw[0]), int(frame_hw[1])
    if roi_meta is None and isinstance(rgb_patch, tuple):
        p_pair, h_pair = rgb_patch[0], rgb_patch[1]
        p_rgb, p_meta = p_pair
    else:
        p_rgb, p_meta, h_pair = rgb_patch, roi_meta, None
    p_crop = crop_from_roi_patch(p_rgb, p_meta, cx, cy, p_side).transpose(2, 0, 1)
    if use_crop_e:
        hx, hy, hs = _hand_crop_for_frame_e(sub_mode, p0[fi], cx, cy, p_side, cw, ch)
    else:
        hx, hy, hs = cx, cy-.12*side, max(.42*side, 96.)
        kp = p0[fi]
        points = []
        for wi, ei in [(9, 7), (10, 8)]:
            if kp[wi, 2] > .20:
                x, y = float(kp[wi, 0]), float(kp[wi, 1])
                if kp[ei, 2] > .20:
                    x += .30*(x-float(kp[ei, 0]))
                    y += .30*(y-float(kp[ei, 1]))
                points.append((np.clip(x, 0, cw-1), np.clip(y, 0, ch-1)))
        if len(points) == 2:
            (x1, y1), (x2, y2) = points
            distance = float(np.hypot(x1-x2, y1-y2))
            if distance <= .50*side:
                hx, hy = .5*(x1+x2), .5*(y1+y2)
                hs = max(distance+.34*side, .38*side, 96.)
            else:
                hx, hy = points[0 if prefer_left else 1]
                hs = max(.38*side, 96.)
        elif points:
            hx, hy = points[0]
            hs = max(.38*side, 96.)
    if h_pair is not None and _patch_covers_square(
            h_pair[0], h_pair[1], hx, hy, hs, cw, ch):
        h_src_rgb, h_src_meta = h_pair
    else:
        h_src_rgb, h_src_meta = p_rgb, p_meta
    h_crop = crop_from_roi_patch(h_src_rgb, h_src_meta, hx, hy, hs).transpose(2, 0, 1)
    return p_crop, h_crop


_ERF_VEC = np.vectorize(math.erf, otypes=[np.float64])


def _gelu_exact(x):
    x64 = np.asarray(x, dtype=np.float64)
    return (0.5 * x64 * (1.0 + _ERF_VEC(x64 * (1.0 / math.sqrt(2.0))))).astype(np.float32)


def _normalize_rgb_head_cfg(raw_cfg):
    cfg = {}
    for k, v in raw_cfg.items():
        if k == 'head_type':
            cfg[k] = str(v.item()) if hasattr(v, 'item') else str(v)
        elif k == 'ctx_frames':
            cfg[k] = int(v.item()) if hasattr(v, 'item') else int(v)
        else:
            cfg[k] = np.asarray(v, dtype=np.float32)
    return cfg


def fuse_dual_event_head(fp, fh, vh_norm, head_cfg, fc=None):
    """Apply DualEventMotionHead (with optional context residual branch) in NumPy."""
    b, t = vh_norm.shape[:2]
    eh_raw = np.zeros((b, t), dtype=np.float32)
    if t >= 2:
        eh_raw[:, 1:] = np.abs(vh_norm[:, 1:] - vh_norm[:, :-1]).mean(axis=(2, 3, 4))
    tp = fp.shape[1]
    if tp != t and tp > 0 and t % tp == 0:
        eh_seq = eh_raw.reshape(b, tp, t // tp).mean(axis=-1, keepdims=True)
    else:
        eh_seq = eh_raw[:, :tp, None]

    f_cat = np.concatenate([np.asarray(fp, np.float32), np.asarray(fh, np.float32)], axis=-1)
    mean_f = f_cat.mean(axis=1)
    df_cat = np.zeros_like(f_cat)
    if tp >= 2:
        df_cat[:, 1:] = f_cat[:, 1:] - f_cat[:, :-1]

    h_att = _gelu_exact(
        f_cat @ head_cfg['att_f_w'].T
        + np.abs(df_cat) @ head_cfg['att_d_w'].T
        + eh_seq @ head_cfg['att_e_w'].T
        + head_cfg['att_e_b']
    )
    att_logits = h_att @ head_cfg['att_out_w'].T + head_cfg['att_out_b']
    att_logits = att_logits - att_logits.max(axis=1, keepdims=True)
    alpha = np.exp(att_logits)
    alpha = alpha / alpha.sum(axis=1, keepdims=True)

    ev_res = (alpha * f_cat).sum(axis=1) - mean_f
    dv_res = (alpha * df_cat).sum(axis=1)
    pooled = mean_f + head_cfg['gamma_ev'] * ev_res + head_cfg['gamma_dv'] * dv_res
    if fc is not None:
        c_pool = np.asarray(fc, np.float32).mean(axis=1)
        head_type = str(head_cfg.get('head_type', 'res_ctx'))
        if head_type == 'res_ctx' and 'ctx_p_w' in head_cfg:
            logits_ph = pooled @ head_cfg['fc_w'].T + head_cfg['fc_b']
            h_ctx = _gelu_exact(
                pooled @ head_cfg['ctx_p_w'].T
                + c_pool @ head_cfg['ctx_c_w'].T
                + head_cfg['ctx_c_b']
            )
            return (logits_ph + h_ctx @ head_cfg['ctx_fc_w'].T + head_cfg['ctx_fc_b']).astype(np.float32)
        if head_type == 'gated_concat' and 'gate_p_w' in head_cfg:
            h_g = _gelu_exact(
                pooled @ head_cfg['gate_p_w'].T
                + c_pool @ head_cfg['gate_c_w'].T
                + head_cfg['gate_c_b']
            )
            g_logit = h_g @ head_cfg['gate_out_w'].T + head_cfg['gate_out_b']
            g = 1.0 / (1.0 + np.exp(-np.clip(g_logit, -30.0, 30.0)))
            feat_all = np.concatenate([pooled, g * c_pool], axis=-1).astype(np.float32)
            return (feat_all @ head_cfg['fc_w'].T + head_cfg['fc_b']).astype(np.float32)
        if 'ctx_frames' in head_cfg:
            feat_all = np.concatenate([pooled, c_pool], axis=-1).astype(np.float32)
            return (feat_all @ head_cfg['fc_w'].T + head_cfg['fc_b']).astype(np.float32)
    return (pooled @ head_cfg['fc_w'].T + head_cfg['fc_b']).astype(np.float32)


def _temp_scale_probs(probs, inv_t):
    lp = np.log(np.clip(probs, 1e-7, 1.0))*float(inv_t)
    lp -= lp.max()
    ep = np.exp(lp)
    return ep/ep.sum()


def _norm_entropy(probs):
    pc = np.clip(probs, 1e-7, 1.0)
    return float(-(pc*np.log(pc)).sum()/math.log(126.0))


def fuse_calibrated_router(body, rgb, router_cfg, observation_fps=None):
    """Apply 10-DOF OOF-calibrated conditional routing between Body and RGB."""
    theta = np.asarray(router_cfg['theta'], dtype=np.float64)
    if theta.shape != (10,):
        raise ValueError(f'router theta must have 10 scalars; got {theta.shape}')
    mode = str(router_cfg.get('mode', 'poe_gate'))
    low_hi = float(router_cfg.get('low_fps_hi', 20.0))
    low_lo = float(router_cfg.get('low_fps_lo', 13.0))
    if observation_fps is None or observation_fps <= 0.0:
        is_low_fps = 0.0
    else:
        is_low_fps = float(np.clip((low_hi-float(observation_fps))/max(low_hi-low_lo, 1e-6), 0.0, 1.0))

    inv_tb = float(np.clip(theta[0], 0.5, 2.5))
    inv_tr = float(np.clip(theta[1], 0.5, 2.5))
    g_logits = theta[2:7]
    g_sharp_r, g_sharp_b, g_fps = float(theta[7]), float(theta[8]), float(theta[9])

    pb = _temp_scale_probs(np.asarray(body, dtype=np.float64), inv_tb)
    pr = _temp_scale_probs(np.asarray(rgb, dtype=np.float64), inv_tr)
    sharp_b = 1.0 - _norm_entropy(pb)
    sharp_r = 1.0 - _norm_entropy(pr)
    base_logit = (g_logits[:, None]*_ROUTER_GROUP_MASKS).sum(axis=0)
    dyn_logit = base_logit + g_sharp_r*sharp_r - g_sharp_b*sharp_b + g_fps*is_low_fps
    w_rgb = 1.0/(1.0 + np.exp(-np.clip(dyn_logit, -8.0, 8.0)))

    if mode == 'poe_gate':
        log_f = (1.0-w_rgb)*np.log(np.clip(pb, 1e-7, 1.0)) + w_rgb*np.log(np.clip(pr, 1e-7, 1.0))
        log_f -= log_f.max()
        out = np.exp(log_f)
        return (out/out.sum()).astype(np.float32)

    out = (1.0-w_rgb)*pb + w_rgb*pr
    return (out/out.sum()).astype(np.float32)


def fuse_probabilities(body, rgb, ntu_rgb_weight=.08):
    # These model heads share this fixed ontology recipe. Non-NTU classes are
    # the kitchen actions, none, and the traffic/control gesture family.
    weights = np.full(126, ntu_rgb_weight, np.float32)
    weights[np.array([38]+list(range(103, 117)))-1] = .30
    weights[12:28] = .65
    result = (1-weights)*np.asarray(body)+weights*np.asarray(rgb)
    return result/result.sum()


class SegmentActionBackend:
    def __init__(self, pose_model, body_engine, rgb_engine, min_score=.02,
                 confidence=.35, observations=48, ntu_rgb_weight=.08,
                 crop_mode='A', router=None, rgb_head=None):
        self.pose_model = pose_model
        self.body = SkeletonActionBackend(engine=body_engine, modality='jb',
                                         prenorm_mode='fix', min_motion=0.)
        self.rgb_engine = rgb_engine
        self.rgb_head = (_normalize_rgb_head_cfg(rgb_head)
                         if rgb_head is not None else None)
        self.rgb_single_view = self.rgb_head is not None
        self.rgb_ctx_frames = (int(self.rgb_head.get('ctx_frames', 0))
                               if self.rgb_head is not None else 0)
        shape = getattr(rgb_engine, 'input_shape', None) or getattr(
            rgb_engine, 'optimization_shape', None)
        if self.rgb_single_view:
            if (shape is None or len(shape) != 5 or tuple(shape[2:]) != (3, 192, 192)
                    or int(shape[1]) <= 0):
                raise ValueError('Single-view RGB engine must take (N, K, 3, 192, 192)')
            self.rgb_n_per_view = int(shape[1])
        else:
            if (shape is None or len(shape) != 5 or tuple(shape[2:]) != (3, 192, 192)
                    or int(shape[1]) <= 0 or int(shape[1]) % 2):
                raise ValueError('RGB action engine must take (N, 2*K, 3, 192, 192)')
            self.rgb_n_per_view = int(shape[1])//2
        self.ntu_rgb_weight = float(ntu_rgb_weight)
        self.crop_mode = str(crop_mode or 'A').upper()
        self.router = dict(router) if isinstance(router, dict) and 'theta' in router else None
        self.min_score = float(min_score)
        self.confidence = float(confidence)
        self.observations = int(observations)
        self.last_error = None

    supports_need_image = True
    supports_precomputed_rgb = True

    def warmup(self, *, include_rgb=True):
        """Run one dummy pass per segment TensorRT engine so case 1 sees warm latency.

        Passing `include_rgb=False` defers the heavier RGB feature engine's first
        allocation until the first real segment, avoiding simultaneous 10-instance
        startup memory spikes on 8 GB Jetson devices.
        """
        engines = ((self.body._engine, self.rgb_engine)
                   if include_rgb else (self.body._engine,))
        for engine in engines:
            infer = getattr(engine, 'infer', None)
            shape = getattr(engine, 'optimization_shape', None) or getattr(engine, 'input_shape', None)
            if not callable(infer) or not shape:
                continue
            try:
                dims = tuple(int(d) for d in shape)
                if any(d <= 0 for d in dims):
                    continue
                dtype = getattr(engine, 'input_dtype', None) or np.float32
                infer(np.zeros(dims, dtype=dtype))
            except Exception as error:
                log.warning('Segment engine warmup skipped: %s', error)

    @classmethod
    def from_directory(cls, pose_model, directory, **kwargs):
        from utils.tensorrt_runtime import TensorRTEngine
        root = Path(directory)
        manifest = root/'models.json'
        files = json.loads(manifest.read_text()) if manifest.exists() else {}
        body = TensorRTEngine(root/files.get('body_engine', 'stgcnpp-action126-v2p.engine'), shared_scratch=True)
        try:
            rgb = TensorRTEngine(root/files.get('rgb_engine', 'tsm-mnv2-dual8f-action126.engine'), shared_scratch=True)
        except Exception:
            body.close()
            raise
        try:
            if 'rgb_head' in files and files['rgb_head'] and 'rgb_head' not in kwargs:
                with np.load(root/files['rgb_head']) as npz:
                    kwargs['rgb_head'] = _normalize_rgb_head_cfg({k: npz[k] for k in npz.files})
            kwargs.setdefault('ntu_rgb_weight', files.get('rgb_ntu_weight', .08))
            kwargs.setdefault('crop_mode', files.get('crop_mode', 'A'))
            kwargs.setdefault('router', files.get('router'))
            return cls(pose_model, body, rgb, **kwargs)
        except Exception:
            body.close()
            rgb.close()
            raise

    def _infer_rgb_from_blobs(self, vp_blob, vh_blob, vc_blob, stop_event=None):
        if stop_event is not None and stop_event.is_set():
            return None
        if self.rgb_single_view:
            fp = np.asarray(self.rgb_engine.infer(vp_blob)[0], np.float32)
            if stop_event is not None and stop_event.is_set():
                return None
            fh = np.asarray(self.rgb_engine.infer(vh_blob)[0], np.float32)
            fc = None
            if vc_blob is not None:
                if stop_event is not None and stop_event.is_set():
                    return None
                fc = np.asarray(self.rgb_engine.infer(vc_blob)[0], np.float32)
            raw_rgb = fuse_dual_event_head(fp, fh, vh_blob, self.rgb_head, fc=fc).reshape(-1)
        else:
            rgb_blob = np.concatenate([vp_blob, vh_blob], axis=1)
            raw_rgb = np.asarray(self.rgb_engine.infer(rgb_blob)[0], np.float32).reshape(-1)
        return raw_rgb if looks_like_probabilities(raw_rgb) else softmax(raw_rgb)

    def _build_cached_rgb_blobs(self, received, samples, indices, primary_records,
                                orig_h, orig_w, sample_patches, image_indices,
                                ctx_indices, stop_event=None, *,
                                need_image=False, snapshot_index=None):
        import cv2
        from plugins.vision_runtime import letterbox
        image_index_set = set(image_indices)
        ctx_index_set = set(ctx_indices)
        snapshot_bgr = None
        vc_blob = None
        patched_received = [
            (ri, rec[2][4]) for ri, rec in enumerate(received)
            if len(rec) == 3 and rec[2] is not None and len(rec[2]) > 4 and rec[2][4] is not None
        ]
        if patched_received:
            geom_orig = prepare_rgb_crop_geometry(
                primary_records, (orig_h, orig_w), crop_mode=self.crop_mode)
            crops = {}
            for fi in sorted(image_index_set):
                if stop_event is not None and stop_event.is_set():
                    return None
                patch_entry = sample_patches.get(fi)
                if patch_entry is None:
                    orig_ri = int(indices[fi])
                    _, patch_entry = min(
                        patched_received, key=lambda item: abs(item[0] - orig_ri))
                if isinstance(patch_entry[0], tuple):
                    crops[fi] = crop_rgb_pair_from_patch(
                        patch_entry, None, fi, geom_orig, (orig_h, orig_w))
                else:
                    rgb_patch, roi_meta = patch_entry
                    crops[fi] = crop_rgb_pair_from_patch(
                        rgb_patch, roi_meta, fi, geom_orig, (orig_h, orig_w))
            if ctx_indices:
                ctx_received = [
                    (ri, entry[2]) for ri, entry in patched_received
                    if isinstance(entry, tuple) and len(entry) >= 3 and entry[2] is not None
                ]
                if ctx_received:
                    ctx_crops = []
                    for ci in ctx_indices:
                        p_entry = sample_patches.get(ci)
                        if isinstance(p_entry, tuple) and len(p_entry) >= 3 and p_entry[2] is not None:
                            ctx_rgb = p_entry[2]
                        else:
                            orig_ri = int(indices[ci])
                            _, ctx_rgb = min(
                                ctx_received, key=lambda item: abs(item[0] - orig_ri))
                        ctx_crops.append(ctx_rgb.transpose(2, 0, 1))
                    vc_blob = build_ctx_view_from_crops(ctx_crops, self.rgb_n_per_view)
                    del ctx_crops
            if need_image and snapshot_index is not None:
                snap_bytes = samples[snapshot_index][1]
                if snap_bytes:
                    snapshot_bgr = cv2.imdecode(
                        np.frombuffer(snap_bytes, np.uint8), cv2.IMREAD_COLOR)
                    if snapshot_bgr is None:
                        raise ValueError('segment contains an undecodable JPEG')
                else:
                    snap_entry = sample_patches.get(snapshot_index)
                    if snap_entry is None:
                        orig_ri = int(indices[snapshot_index])
                        _, snap_entry = min(
                            patched_received, key=lambda item: abs(item[0] - orig_ri))
                    p_entry = snap_entry[0] if isinstance(snap_entry[0], tuple) else snap_entry
                    sp_rgb, (rx0, ry0, sx, sy) = p_entry
                    snapshot_bgr = np.zeros((orig_h, orig_w, 3), np.uint8)
                    tw = min(orig_w - rx0, max(1, int(round(sp_rgb.shape[1] / max(sx, 1e-6)))))
                    th = min(orig_h - ry0, max(1, int(round(sp_rgb.shape[0] / max(sy, 1e-6)))))
                    if tw > 0 and th > 0:
                        sp_bgr = cv2.cvtColor(sp_rgb, cv2.COLOR_RGB2BGR)
                        if (sp_bgr.shape[1], sp_bgr.shape[0]) != (tw, th):
                            sp_bgr = cv2.resize(sp_bgr, (tw, th), interpolation=cv2.INTER_LINEAR)
                        snapshot_bgr[ry0:ry0+th, rx0:rx0+tw] = sp_bgr
            vp_blob, vh_blob = build_rgb_views_from_crops(
                [crops[i][0] for i in image_indices],
                [crops[i][1] for i in image_indices],
            )
            del crops
            return vp_blob, vh_blob, vc_blob, snapshot_bgr

        # Legacy cached-pose path when records carry raw JPEG bytes without rgb_patch.
        crop_flag = (cv2.IMREAD_REDUCED_COLOR_2
                     if min(orig_h, orig_w) >= 720 else cv2.IMREAD_COLOR)
        crops = {}
        ctx_map = {}
        geom = None
        needed_rgb = image_index_set | ctx_index_set
        target_indices = (sorted(needed_rgb | {snapshot_index})
                          if (need_image and snapshot_index is not None) else sorted(needed_rgb))
        for fi in target_indices:
            if stop_event is not None and stop_event.is_set():
                return None
            raw_buf = np.frombuffer(samples[fi][1], np.uint8)
            if need_image and fi == snapshot_index and crop_flag != cv2.IMREAD_COLOR:
                snapshot_bgr = cv2.imdecode(raw_buf, cv2.IMREAD_COLOR)
                if snapshot_bgr is None:
                    raise ValueError('segment contains an undecodable JPEG')
            if fi in needed_rgb:
                bgr = cv2.imdecode(raw_buf, crop_flag)
                if bgr is None:
                    raise ValueError('segment contains an undecodable JPEG')
                if need_image and fi == snapshot_index and snapshot_bgr is None:
                    snapshot_bgr = bgr
                rgb_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if fi in image_index_set:
                    if geom is None:
                        hr, wr = rgb_img.shape[:2]
                        if (hr, wr) != (orig_h, orig_w) and orig_w > 0 and orig_h > 0:
                            sx, sy = wr / float(orig_w), hr / float(orig_h)
                            scaled_records = [
                                np.column_stack([kp[:, 0]*sx, kp[:, 1]*sy, kp[:, 2]])
                                for kp in primary_records
                            ]
                        else:
                            scaled_records = primary_records
                        geom = prepare_rgb_crop_geometry(
                            scaled_records, (hr, wr), crop_mode=self.crop_mode)
                    crops[fi] = crop_rgb_pair_for_frame(rgb_img, fi, geom)
                if fi in ctx_index_set:
                    ctx_canvas, _ = letterbox(rgb_img, 192, 192)
                    ctx_map[fi] = ctx_canvas.transpose(2, 0, 1)
                del rgb_img
                if snapshot_bgr is not bgr:
                    del bgr
            elif need_image and fi == snapshot_index and snapshot_bgr is None:
                snapshot_bgr = cv2.imdecode(raw_buf, cv2.IMREAD_COLOR)
                if snapshot_bgr is None:
                    raise ValueError('segment contains an undecodable JPEG')
        vp_blob, vh_blob = build_rgb_views_from_crops(
            [crops[i][0] for i in image_indices],
            [crops[i][1] for i in image_indices],
        )
        if ctx_indices:
            vc_blob = build_ctx_view_from_crops(
                [ctx_map[i] for i in ctx_indices], self.rgb_n_per_view)
        del crops, ctx_map
        return vp_blob, vh_blob, vc_blob, snapshot_bgr

    def compute_rgb_probs(self, received, stop_event=None):
        """Background-safe RGB branch pass over currently cached posed observations.

        Returns `(rgb_probs, obs_count)` or `None` if cancelled / insufficient poses.
        """
        if len(received) < 4 or len(received[0]) != 3:
            return None
        indices = np.linspace(0, len(received)-1,
                              min(self.observations, len(received))).round().astype(int)
        samples = [received[i] for i in indices]
        image_indices = list(rgb_frame_indices(len(samples), self.rgb_n_per_view))
        ctx_indices = (list(rgb_frame_indices(len(samples), self.rgb_ctx_frames))
                       if (self.rgb_single_view and self.rgb_ctx_frames > 0) else [])
        frame_records = []
        tracker = PoseTracker(timeout_s=2., history_s=30.,
                              frame_factory=_SkeletonFrame)
        track_weights = {}
        base_time = samples[0][0]
        sample_patches = {}
        for fi, (_, _data, pose) in enumerate(samples):
            if stop_event is not None and stop_event.is_set():
                return None
            if pose is None:
                return None
            boxes, scores, kps, shape = pose[:4]
            if len(pose) > 4 and pose[4] is not None:
                sample_patches[fi] = pose[4]
            h, w = shape
            order = np.argsort(np.asarray(scores, np.float32))[::-1][:4]
            boxes, scores, kps = boxes[order], scores[order], kps[order]
            timestamp = samples[fi][0]-base_time
            tracks = tracker.update(boxes, kps, timestamp, image_size=(w, h))
            records = []
            for track, box, score, kp in zip(tracks, boxes, scores, kps):
                area = max(1., float(box[2]-box[0])*float(box[3]-box[1]))/max(1., float(w*h))
                track_weights[track.id] = track_weights.get(track.id, 0.)+float(score)*(.5+area)
                records.append((track.id, box, float(score), kp))
            frame_records.append((timestamp, (w, h), records))
        if not track_weights:
            return None
        primary = max(track_weights, key=track_weights.get)
        primary_records = []
        snapshot_index = None
        for fi, (_timestamp, _size, records) in enumerate(frame_records):
            chosen = next((r for r in records if r[0] == primary), None)
            if chosen is not None:
                snapshot_index = fi
            if chosen is None and records:
                chosen = records[0]
            kp = chosen[3] if chosen else np.zeros((17, 3), np.float32)
            primary_records.append(kp)
        if snapshot_index is None or (stop_event is not None and stop_event.is_set()):
            return None
        _, snap_wh, _ = frame_records[snapshot_index]
        orig_w, orig_h = int(snap_wh[0]), int(snap_wh[1])
        built = self._build_cached_rgb_blobs(
            received, samples, indices, primary_records, orig_h, orig_w,
            sample_patches, image_indices, ctx_indices, stop_event,
            need_image=False, snapshot_index=None,
        )
        if built is None:
            return None
        vp_blob, vh_blob, vc_blob, _ = built
        rgb = self._infer_rgb_from_blobs(vp_blob, vh_blob, vc_blob, stop_event)
        del vp_blob, vh_blob, vc_blob
        if rgb is None or rgb.shape != (126,):
            return None
        return rgb, len(received)

    def recognize(self, received, stop_event=None, *, need_image=True, precomputed_rgb=None):
        import cv2
        from plugins.vision_runtime import decode_poses, letterbox
        if len(received) < 4:
            return None
        indices = np.linspace(0, len(received)-1,
                              min(self.observations, len(received))).round().astype(int)
        samples = [received[i] for i in indices]
        image_indices = list(rgb_frame_indices(len(samples), self.rgb_n_per_view))
        image_index_set = set(image_indices)
        ctx_indices = (list(rgb_frame_indices(len(samples), self.rgb_ctx_frames))
                       if (self.rgb_single_view and self.rgb_ctx_frames > 0) else [])
        ctx_index_set = set(ctx_indices)
        rgb_images, frame_records = {}, []
        tracker = PoseTracker(timeout_s=2., history_s=30.,
                              frame_factory=_SkeletonFrame)
        track_weights = {}
        base_time = samples[0][0]
        def record_detections(fi, shape, boxes, scores, kps):
            h, w = shape
            order = np.argsort(np.asarray(scores, np.float32))[::-1][:4]
            boxes, scores, kps = boxes[order], scores[order], kps[order]
            timestamp = samples[fi][0]-base_time
            tracks = tracker.update(boxes, kps, timestamp, image_size=(w, h))
            records = []
            for track, box, score, kp in zip(tracks, boxes, scores, kps):
                area = max(1., float(box[2]-box[0])*float(box[3]-box[1]))/max(1., float(w*h))
                track_weights[track.id] = track_weights.get(track.id, 0.)+float(score)*(.5+area)
                records.append((track.id, box, float(score), kp))
            frame_records.append((timestamp, (w, h), records))

        cached_poses = len(samples[0]) == 3
        sample_patches = {}
        if cached_poses:
            for fi, (_, _data, pose) in enumerate(samples):
                if stop_event is not None and stop_event.is_set():
                    return None
                if pose is None:
                    raise ValueError('segment observation has no processed pose')
                boxes, scores, kps, shape = pose[:4]
                if len(pose) > 4 and pose[4] is not None:
                    sample_patches[fi] = pose[4]
                record_detections(fi, shape, boxes, scores, kps)
        for start in ([] if cached_poses else range(0, len(samples), 8)):
            if stop_event is not None and stop_event.is_set():
                return None
            images = [cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                      for _, data in samples[start:start+8]]
            if any(image is None for image in images):
                raise ValueError('segment contains an undecodable JPEG')
            infer_many = getattr(self.pose_model, 'infer_batch', None)
            outputs = (infer_many(images) if infer_many else
                       [self.pose_model.infer(image) for image in images])
            for offset, (image, (raw, meta)) in enumerate(zip(images, outputs)):
                fi = start+offset
                h, w = image.shape[:2]
                if fi in image_index_set or fi in ctx_index_set:
                    rgb_images[fi] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                boxes, scores, kps = decode_poses(raw, meta, self.confidence)
                record_detections(fi, (h, w), boxes, scores, kps)
            del images, outputs
        if not track_weights:
            return None
        primary = max(track_weights, key=track_weights.get)
        poses, primary_records = [], []
        snapshot_index = None
        for fi, (timestamp, size, records) in enumerate(frame_records):
            chosen = next((r for r in records if r[0] == primary), None)
            if chosen is not None:
                snapshot_index = fi
            if chosen is None and records:
                chosen = records[0]
            box = chosen[1] if chosen else np.zeros(4, np.float32)
            kp = chosen[3] if chosen else np.zeros((17, 3), np.float32)
            poses.append(_SkeletonFrame(timestamp, box, kp, .3, image_size=size))
            primary_records.append(kp)
        if snapshot_index is None or (stop_event is not None and stop_event.is_set()):
            return None
        blob = self.body._build_input(poses, self.body._window_frames())
        body = self.body._run(blob)
        if not looks_like_probabilities(body):
            body = softmax(body)

        _, snap_wh, records = frame_records[snapshot_index]
        orig_w, orig_h = int(snap_wh[0]), int(snap_wh[1])
        snapshot_bgr = None
        used_precomputed_rgb = False
        if precomputed_rgb is not None:
            cand_rgb = np.asarray(precomputed_rgb, np.float32).reshape(-1)
            if cand_rgb.shape == (126,) and np.isfinite(cand_rgb).all():
                rgb = cand_rgb if looks_like_probabilities(cand_rgb) else softmax(cand_rgb)
                used_precomputed_rgb = True
                if need_image and cached_poses:
                    snap_bytes = samples[snapshot_index][1]
                    if snap_bytes:
                        snapshot_bgr = cv2.imdecode(
                            np.frombuffer(snap_bytes, np.uint8), cv2.IMREAD_COLOR)
                        if snapshot_bgr is None:
                            raise ValueError('segment contains an undecodable JPEG')
        if not used_precomputed_rgb:
            vc_blob = None
            if cached_poses:
                built = self._build_cached_rgb_blobs(
                    received, samples, indices, primary_records, orig_h, orig_w,
                    sample_patches, image_indices, ctx_indices, stop_event,
                    need_image=need_image, snapshot_index=snapshot_index,
                )
                if built is None:
                    return None
                vp_blob, vh_blob, vc_blob, snapshot_bgr = built
            else:
                rgb_blob_full = build_rgb_input([rgb_images[i] for i in image_indices],
                                                primary_records, crop_mode=self.crop_mode)
                vp_blob = rgb_blob_full[:, :self.rgb_n_per_view]
                vh_blob = rgb_blob_full[:, self.rgb_n_per_view:]
                if ctx_indices:
                    ctx_crops = [letterbox(rgb_images[i], 192, 192)[0].transpose(2, 0, 1)
                                 for i in ctx_indices]
                    vc_blob = build_ctx_view_from_crops(ctx_crops, self.rgb_n_per_view)
                    del ctx_crops
            rgb = self._infer_rgb_from_blobs(vp_blob, vh_blob, vc_blob, stop_event)
            del vp_blob, vh_blob, vc_blob
            if rgb is None:
                return None
        if body.shape != (126,) or rgb.shape != (126,):
            raise ValueError('segment action heads must both emit 126 classes')
        if self.router is not None:
            span_s = float(received[-1][0] - received[0][0]) if len(received) > 1 else 0.0
            obs_fps = (len(received) - 1)/span_s if span_s > 1e-3 else 30.0
            probabilities = fuse_calibrated_router(body, rgb, self.router, observation_fps=obs_fps)
        else:
            probabilities = fuse_probabilities(body, rgb, self.ntu_rgb_weight)
        action_id = int(probabilities.argmax())+1
        score = float(probabilities[action_id-1])
        activity = None
        if score >= self.min_score:
            zh, name = OFFICIAL_126_CLASSES[action_id]
            activity = {'action_id': action_id, 'name': name, 'name_zh': zh,
                        'score': round(score, 3), 'source': 'segment-dual'}
        if not need_image:
            image = _ImageShape((orig_h, orig_w, 3))
        elif snapshot_bgr is not None:
            image = snapshot_bgr
        elif snapshot_index in rgb_images:
            image = cv2.cvtColor(rgb_images[snapshot_index], cv2.COLOR_RGB2BGR)
        else:
            image = cv2.imdecode(np.frombuffer(samples[snapshot_index][1], np.uint8), cv2.IMREAD_COLOR)
        primary_box = next(r[1] for r in records if r[0] == primary)
        self.last_error = None
        return {'activity': activity, 'probabilities': probabilities,
                'evidence': {'frames_used': len(samples), 'primary_track': primary,
                             'poses_computed_during_reception': cached_poses,
                             'rgb_precomputed': used_precomputed_rgb,
                             'RGB_frames_per_view': self.rgb_n_per_view,
                             'RGB_NTU_weight': self.ntu_rgb_weight,
                             'source': 'received-segment', 'best_action_id': action_id,
                             'best_score': round(score, 3)},
                'image': image, 'boxes': np.array([r[1] for r in records]),
                'scores': np.array([r[2] for r in records]),
                'keypoints': np.array([r[3] for r in records]),
                'primary_box': primary_box}
