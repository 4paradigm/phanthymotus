"""Bounded, serializable visual sessions; no motion or transport dependencies."""

from collections import OrderedDict
from copy import deepcopy
import math
import threading
import time
from uuid import uuid4

from .contracts import (
    RESULT,
    Refusal,
    canonical,
    center,
    digest,
    fields,
    goal,
    need,
    number,
    observation,
    text,
    vector,
)


class Engine:
    def __init__(self, config, *, wall=None):
        self.wall = wall or time.time
        self.lock = threading.RLock()
        self.config = deepcopy(config)
        canonical(self.config)
        self.sources = self.config.get("sources", {})
        text(config["piper_id"])
        self.sessions = {}
        self.requests = OrderedDict()
        self.latest = {}
        self.seen = OrderedDict()
        self.last_wall = None
        self.running = True
        self.max_age = number(config.get("max_age_s", 2))
        need(0.05 <= self.max_age <= 5, "INVALID_MAX_AGE")
        self.ttl = number(config.get("session_ttl_s", 600))
        need(1 <= self.ttl <= 3600, "INVALID_SESSION_TTL")
        for name, binding in self.sources.items():
            need(name == binding["source_id"], "SOURCE_BINDING_MISMATCH")
            for key in ("source_id", "camera_id", "frame_id", "distortion"):
                text(binding[key])

    def _now(self):
        now = number(self.wall())
        need(self.last_wall is None or now >= self.last_wall, "CLOCK_ROLLBACK")
        self.last_wall = now
        return now

    def _session(self, args, now):
        s = self.sessions.get(args.get("session_id"))
        need(s is not None, "UNKNOWN_SESSION")
        need(args.get("session_token") == s["session_token"], "RETIRED_SESSION")
        need(now < s["expires_at"], "SESSION_EXPIRED")
        return s

    def inspect(self):
        with self.lock:
            return dict(
                running=self.running,
                piper_id=self.config["piper_id"],
                sources=[
                    dict(
                        source_id=k,
                        camera_id=v["camera_id"],
                        kind=v.get("kind", "normalized"),
                    )
                    for k, v in self.sources.items()
                ],
                sessions=[
                    dict(
                        session_id=s["session_id"],
                        task_id=s["task_id"],
                        source_id=s["source_id"],
                        has_before=s["before"] is not None,
                        has_marker=s["marker"] is not None,
                        last_result=deepcopy(s["result"]),
                    )
                    for s in self.sessions.values()
                ],
            )

    def ingest(self, raw):
        with self.lock:
            now = self._now()
            need(self.running, "STOPPED")
            need(
                isinstance(raw, dict) and raw.get("source_id") in self.sources,
                "UNKNOWN_SOURCE",
            )
            binding = self.sources[raw["source_id"]]
            o = observation(raw, binding, now, self.max_age)
            key = (o["source_id"], o["observation_id"])
            hashed = digest(o)
            if key in self.seen:
                need(self.seen[key] == hashed, "OBSERVATION_ID_CONFLICT")
                return deepcopy(o)
            previous = self.latest.get(o["source_id"])
            if previous:
                need(o["captured_at"] > previous["captured_at"], "OUT_OF_ORDER_TIME")
                if o["source_epoch"] == previous["source_epoch"]:
                    need(o["sequence"] > previous["sequence"], "OUT_OF_ORDER_SEQUENCE")
                else:
                    for s in self.sessions.values():
                        if s["source_id"] == o["source_id"] and s["before"]:
                            s["identity_lost"] = True
            self.latest[o["source_id"]] = o
            self.seen[key] = hashed
            while len(self.seen) > 512:
                self.seen.popitem(last=False)
            # A visible gap cannot silently be healed by another same-colour
            # object appearing later. A new baseline is needed after such loss.
            for s in self.sessions.values():
                if (
                    s["source_id"] == o["source_id"]
                    and s["before"]
                    and o["captured_at"] > s["before"]["captured_at"]
                ):
                    candidates = self._candidates(o, s["selector"], s["min_confidence"])
                    if not candidates:
                        s["identity_lost"] = True
            return deepcopy(o)

    def external_allowed(self, raw):
        with self.lock:
            need(
                isinstance(raw, dict) and raw.get("source_id") in self.sources,
                "UNKNOWN_SOURCE",
            )
            known = self.seen.get(
                (raw["source_id"], raw.get("observation_id"))
            ) == digest(raw)
            binding = self.sources[raw["source_id"]]
            need(
                known
                or (
                    binding.get("accept_external_observations") is True
                    and binding.get("external_observation_record")
                ),
                "EXTERNAL_OBSERVATION_DISABLED",
                "此来源只能使用已接收的真实视觉数据；外部规范观测需由owner显式配置可信生产者",
            )

    @staticmethod
    def _candidates(o, selector, confidence):
        return [
            obj
            for obj in o["objects"]
            if obj["class_name"] == selector["class_name"]
            and (not selector.get("color") or obj.get("color") == selector["color"])
            and obj["confidence"] >= confidence
            and obj["bbox"][0] > 0
            and obj["bbox"][1] > 0
            and obj["bbox"][2] < o["image_size"][0]
            and obj["bbox"][3] < o["image_size"][1]
        ]

    def replay(self, action, args):
        with self.lock:
            now = self._now()
            need(self.running, "STOPPED")
            if action != "start_session":
                self._session(args, now)
            key = (
                args.get("session_id", "new"),
                args.get("session_token", "new"),
                args.get("request_id"),
            )
            if key not in self.requests:
                return None
            saved = self.requests[key]
            need(
                saved[0] == digest(dict(action=action, args=args)),
                "REQUEST_ID_CONFLICT",
            )
            return deepcopy(saved[1])

    def acquisition(self, args, *, after=False):
        with self.lock:
            s = self._session(args, self._now())
            if after:
                need(
                    s["marker"] and args.get("marker_id") == s["marker"]["marker_id"],
                    "AFTER_MARKER_REQUIRED",
                )
            return s["source_id"], s["marker"]["not_before"] if after else None

    def record_failure(self, args, exc):
        """A later failed attempt must not leave an earlier PASS as latest."""
        with self.lock:
            now = self._now()
            s = self._session(args, now)
            out = self._result(s, getattr(exc, "code", "INPUT_ERROR"), now)
            out.update(
                message=str(exc),
                after_observation_id=(args.get("observation") or {}).get(
                    "observation_id"
                ),
            )
            s["result"] = deepcopy(out)
            s["history"].append(deepcopy(out))
            s["history"] = s["history"][-50:]
            key = (s["session_id"], s["session_token"], args.get("request_id"))
            try:
                # A conflicting retry must never replace the original receipt.
                if key not in self.requests:
                    self.requests[key] = (
                        digest(dict(action="verify_after", args=args)),
                        deepcopy(out),
                    )
            except Refusal:
                pass
            while len(self.requests) > 256:
                self.requests.popitem(last=False)
            return out

    def call(self, action, args, *, observed=None):
        with self.lock:
            now = self._now()
            need(self.running, "STOPPED")
            canonical(args)
            if action == "start_session":
                s = None
            else:
                s = self._session(args, now)
            if action in ("get_result", "export_session"):
                result = deepcopy(
                    s
                    if action == "export_session"
                    else s["result"] or self._result(s, "NOT_VERIFIED", now)
                )
                if action == "export_session":
                    result.pop("session_token", None)
                return result
            rid = text(args.get("request_id"))
            key = (args.get("session_id", "new"), args.get("session_token", "new"), rid)
            hashed = digest(dict(action=action, args=args))
            if key in self.requests:
                saved = self.requests[key]
                need(saved[0] == hashed, "REQUEST_ID_CONFLICT")
                return deepcopy(saved[1])
            if action == "start_session":
                for expired in [
                    sid
                    for sid, item in self.sessions.items()
                    if now >= item["expires_at"]
                ]:
                    del self.sessions[expired]
                need(len(self.sessions) < 32, "SESSION_LIMIT")
                sid = uuid4().hex
                source = text(args["source_id"])
                need(source in self.sources, "UNKNOWN_SOURCE")
                fields(
                    args["selector"],
                    ("class_name",),
                    ("color", "detection_id", "observation_id", "track_id"),
                )
                for v in args["selector"].values():
                    text(v)
                if "detection_id" in args["selector"]:
                    need(
                        args["selector"].get("observation_id"),
                        "SELECTION_FRAME_REQUIRED",
                    )
                if "track_id" in args["selector"]:
                    need(
                        self.sources[source].get("verified_persistent_tracks") is True,
                        "TRACK_IDENTITY_UNPROVEN",
                    )
                confidence = number(args.get("min_confidence", 0.7))
                need(0.1 <= confidence <= 1, "INVALID_CONFIDENCE")
                displacement = number(args.get("max_displacement_norm", 0.35))
                need(0 < displacement <= 1.4143, "INVALID_DISPLACEMENT")
                s = dict(
                    session_id=sid,
                    session_token=uuid4().hex,
                    task_id=text(args["task_id"]),
                    piper_id=self.config["piper_id"],
                    source_id=source,
                    selector=deepcopy(args["selector"]),
                    min_confidence=confidence,
                    max_displacement_norm=displacement,
                    goal=goal(args["goal"]),
                    created_at=now,
                    expires_at=now + self.ttl,
                    before=None,
                    selected=None,
                    marker=None,
                    result=None,
                    history=[],
                    observations=[],
                    after_high=None,
                    identity_lost=False,
                )
                self.sessions[sid] = s
                out = self._public(s)
            elif action == "reset_session":
                s.update(
                    session_token=uuid4().hex,
                    before=None,
                    selected=None,
                    marker=None,
                    result=None,
                    history=[],
                    observations=[],
                    after_high=None,
                    identity_lost=False,
                    created_at=now,
                    expires_at=now + self.ttl,
                )
                out = self._public(s)
            elif action == "close_session":
                out = dict(state="closed", session_id=s["session_id"])
                del self.sessions[s["session_id"]]
            elif action == "set_goal":
                need(s["before"] is None, "RESET_BEFORE_GOAL_CHANGE")
                s["goal"] = goal(args["goal"])
                out = self._public(s)
            elif action == "mark_operation_complete":
                need(s["before"] is not None, "BEFORE_REQUIRED")
                s["marker"] = dict(marker_id=uuid4().hex, not_before=now)
                s["result"] = None
                out = deepcopy(s["marker"])
            elif action in ("capture_before", "verify_after"):
                if "observation" in args:
                    self.external_allowed(args["observation"])
                raw = (
                    args.get("observation")
                    or observed
                    or self.latest.get(s["source_id"])
                )
                need(raw is not None, "NO_OBSERVATION")
                o = self.ingest(raw)
                need(o["source_id"] == s["source_id"], "SOURCE_BINDING_MISMATCH")
                if action == "capture_before":
                    need(s["before"] is None, "BASELINE_ALREADY_SET")
                    if s["selector"].get("detection_id"):
                        need(
                            o["observation_id"] == s["selector"]["observation_id"],
                            "SELECTION_FRAME_CHANGED",
                        )
                    matches = self._candidates(o, s["selector"], s["min_confidence"])
                    selected = [
                        x
                        for x in matches
                        if (
                            not s["selector"].get("detection_id")
                            or x["detection_id"] == s["selector"]["detection_id"]
                        )
                        and (
                            not s["selector"].get("track_id")
                            or x.get("track_id") == s["selector"]["track_id"]
                        )
                    ]
                    need(
                        len(selected) == 1,
                        "AMBIGUOUS_OR_MISSING_BASELINE",
                        "选择一个明确可见的目标；同类多物体需颜色或检测标识",
                    )
                    s.update(
                        before=o, selected=deepcopy(selected[0]), identity_lost=False
                    )
                    s["observations"] = [deepcopy(o)]
                    out = dict(
                        state="before_captured",
                        **self._public(s),
                        observation_id=o["observation_id"],
                        selected=deepcopy(s["selected"]),
                    )
                else:
                    need(
                        s["before"] is not None and s["marker"] is not None,
                        "AFTER_MARKER_REQUIRED",
                    )
                    need(
                        args.get("marker_id") == s["marker"]["marker_id"],
                        "RETIRED_MARKER",
                    )
                    out = self._verify(s, o, now)
                    s["result"] = deepcopy(out)
                    s["history"].append(deepcopy(out))
                    s["history"] = s["history"][-50:]
                    afters = [*s["observations"][1:], deepcopy(o)]
                    s["observations"] = [s["before"], *afters[-50:]]
            else:
                raise Refusal("UNKNOWN_ACTION")
            self.requests[key] = (hashed, deepcopy(out))
            while len(self.requests) > 256:
                self.requests.popitem(last=False)
            return out

    @staticmethod
    def _public(s):
        return {
            k: deepcopy(s[k])
            for k in (
                "session_id",
                "session_token",
                "task_id",
                "piper_id",
                "source_id",
                "goal",
                "expires_at",
            )
        }

    def _result(self, s, code, now, **extra):
        return dict(
            schema=RESULT,
            state="verified",
            result="INCONCLUSIVE",
            code=code,
            message=code.replace("_", " ").lower(),
            session_id=s["session_id"],
            task_id=s["task_id"],
            piper_id=s["piper_id"],
            verified_at=now,
            data_valid=False,
            before_observation_id=(s["before"] or {}).get("observation_id"),
            after_observation_id=None,
            units=s["goal"]["units"],
            before_position=None,
            after_position=None,
            displacement=None,
            target_error=None,
            inside_region=None,
            within_tolerance=None,
            criterion="visible_bbox_center",
            physical_motion_confirmed=False,
            **extra,
        )

    def _verify(self, s, o, now):
        result = self._result(s, "INSUFFICIENT_EVIDENCE", now)
        result["after_observation_id"] = o["observation_id"]
        result["provenance"] = o["provenance"]
        result["goal"] = deepcopy(s["goal"])
        b = s["before"]
        try:
            need(
                o["observation_id"] != b["observation_id"]
                and o["sequence"] > b["sequence"]
                and o["captured_at"] > b["captured_at"],
                "REPLAYED_OBSERVATION",
            )
            need(
                o["captured_at"] >= s["marker"]["not_before"],
                "PRE_OPERATION_OBSERVATION",
            )
            need(now - s["marker"]["not_before"] <= 120, "MARKER_EXPIRED")
            previous = s.get("after_high")
            need(
                previous is None
                or (
                    o["observation_id"] != previous["observation_id"]
                    and o["sequence"] > previous["sequence"]
                    and o["captured_at"] > previous["captured_at"]
                ),
                "REPLAYED_AFTER_OBSERVATION",
                "新验证请求必须使用比上一次更新的后观测；原请求重试使用原request_id",
            )
            s["after_high"] = {
                k: o[k] for k in ("observation_id", "sequence", "captured_at")
            }
            for key in ("source_epoch", "camera_id", "frame_id", "distortion"):
                need(o[key] == b[key], "SOURCE_GEOMETRY_CHANGED")
            binding = self.sources[s["source_id"]]
            matches = self._candidates(o, s["selector"], s["min_confidence"])
            need(matches, "TARGET_NOT_OBSERVED", "未可靠检测到目标，不等同于任务失败")
            selected = s["selected"]
            method = "unique_class_color_bounded_motion"
            track = selected.get("track_id")
            if track and binding.get("verified_persistent_tracks") is True:
                need(
                    b.get("track_epoch")
                    and b.get("track_epoch") == o.get("track_epoch"),
                    "TRACK_EPOCH_CHANGED",
                )
                candidates = [x for x in matches if x.get("track_id") == track]
                need(len(candidates) == 1, "TRACK_LOST_OR_AMBIGUOUS")
                final = candidates[0]
                method = "source_verified_persistent_track"
            else:
                need(
                    not s["identity_lost"],
                    "IDENTITY_LOST",
                    "目标曾消失或来源重启，需重新建立基准",
                )
                baseline = self._candidates(b, s["selector"], s["min_confidence"])
                need(
                    len(baseline) == len(matches) == 1,
                    "IDENTITY_AMBIGUOUS",
                    "同类同外观多目标，无法证明跨操作身份",
                )
                final = matches[0]
                a = [x / n for x, n in zip(center(selected), b["image_size"])]
                z = [x / n for x, n in zip(center(final), o["image_size"])]
                need(
                    math.dist(a, z) <= s["max_displacement_norm"],
                    "DISPLACEMENT_AMBIGUOUS",
                )

                def area(x, im):
                    return (
                        (x["bbox"][2] - x["bbox"][0])
                        * (x["bbox"][3] - x["bbox"][1])
                        / (im[0] * im[1])
                    )

                need(
                    0.25
                    <= area(final, o["image_size"]) / area(selected, b["image_size"])
                    <= 4,
                    "APPEARANCE_CHANGED",
                )
                if s["goal"]["units"] == "plane_mm":
                    need(
                        b["view_id"] == o["view_id"],
                        "MOVING_VIEW_IDENTITY_UNPROVEN",
                        "跨视角需要已验证的持续身份来源，短窗口或类别颜色不足以证明身份",
                    )
            result["image_positions"] = dict(
                units="pixels",
                frame_id=o["frame_id"],
                before=center(selected),
                after=center(final),
                comparable=b["view_id"] == o["view_id"]
                and b["image_size"] == o["image_size"],
            )
            before, error_b = self._position(s, b, selected, now)
            after, error_a = self._position(s, o, final, now)
            if s["goal"]["units"] != "plane_mm":
                need(
                    b["image_size"] == o["image_size"] and b["view_id"] == o["view_id"],
                    "VIEW_OR_RESOLUTION_CHANGED",
                    "图像视角或尺寸已变化，图像区域不能跨视角复用；请使用经验证平面标定或重建会话",
                )
            g = s["goal"]
            delta = [y - x for x, y in zip(before, after)]
            if g["shape"] == "circle":
                deviation = [y - x for x, y in zip(g["center"], after)]
                distance = math.hypot(*deviation)
                outside = distance - g["radius"]
            else:
                x1, y1, x2, y2 = g["bounds"]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                deviation = [after[0] - cx, after[1] - cy]
                distance = math.hypot(*deviation)
                dx = max(x1 - after[0], after[0] - x2)
                dy = max(y1 - after[1], after[1] - y2)
                outside = (
                    math.hypot(max(dx, 0), max(dy, 0))
                    if max(dx, dy) > 0
                    else max(dx, dy)
                )
            inside = outside <= 0
            within = outside <= g["tolerance"] + 1e-12
            if error_a > 0:
                need(abs(outside - g["tolerance"]) > error_a, "UNCERTAIN_BOUNDARY")
            result.update(
                result="PASS" if within else "FAIL",
                code="GOAL_MET" if within else "OUTSIDE_GOAL",
                message="可见目标中心满足目标区域及容差"
                if within
                else "可靠观测证明可见目标中心仍在目标区域之外",
                data_valid=True,
                before_position=before,
                after_position=after,
                displacement=delta,
                calibration_ids=[b.get("calibration_id"), o.get("calibration_id")],
                displacement_length=math.hypot(*delta),
                target_error=deviation,
                target_error_length=distance,
                signed_region_error=outside,
                inside_region=inside,
                within_tolerance=within,
                association_method=method,
                after_detection_id=final["detection_id"],
                calibration_error_bound=error_a,
                displacement_error_bound=error_a + error_b,
                evidence=dict(
                    before_sha256=digest(b),
                    after_sha256=digest(o),
                    goal_sha256=digest(g),
                    marker=deepcopy(s["marker"]),
                    camera_id=o["camera_id"],
                    source_id=o["source_id"],
                    view_id=o["view_id"],
                ),
            )
        except Refusal as exc:
            result.update(code=exc.code, message=str(exc))
        return result

    def _position(self, s, o, obj, now):
        g = s["goal"]
        p = center(obj)
        w, h = o["image_size"]
        if g["units"] != "plane_mm":
            need(g["frame_id"] == o["frame_id"], "GOAL_FRAME_MISMATCH")
            if g["shape"] == "circle":
                bound = [w, h] if g["units"] == "pixels" else [1, 1]
                need(
                    all(0 <= v <= m for v, m in zip(g["center"], bound)),
                    "GOAL_OUTSIDE_IMAGE",
                )
            else:
                bound = [w, h, w, h] if g["units"] == "pixels" else [1] * 4
                need(
                    all(0 <= v <= m for v, m in zip(g["bounds"], bound)),
                    "GOAL_OUTSIDE_IMAGE",
                )
            return (p if g["units"] == "pixels" else [p[0] / w, p[1] / h]), 0
        # Only owner-installed measured calibrations may turn pixels into mm.
        import numpy as np

        cal = self.config.get("calibrations", {}).get(o.get("calibration_id"))
        need(
            isinstance(cal, dict) and cal.get("provenance") == "operator_measured",
            "CALIBRATION_REQUIRED",
        )
        need(
            cal.get("piper_id") == s["piper_id"]
            and cal.get("camera_id") == o["camera_id"]
            and cal.get("frame_id") == g["frame_id"]
            and cal.get("view_id") == o["view_id"]
            and cal.get("image_size") == o["image_size"]
            and cal.get("distortion") == o["distortion"],
            "CALIBRATION_BINDING",
        )
        need(
            number(cal["created_at"])
            <= o["captured_at"]
            <= now
            < number(cal["expires_at"]),
            "CALIBRATION_EXPIRED",
        )
        text(cal["measurement_record"])
        text(cal["object_plane_record"])
        need(cal.get("class_name") == obj["class_name"], "OBJECT_PLANE_UNPROVEN")
        matrix = np.array(
            [vector(row, 3) for row in cal["pixel_to_plane_mm"]], dtype=float
        )
        need(
            matrix.shape == (3, 3)
            and np.isfinite(matrix).all()
            and np.linalg.cond(matrix) < 1e9,
            "INVALID_CALIBRATION",
        )
        xmin, ymin, xmax, ymax = vector(cal["valid_pixel_bounds"], 4)
        need(xmin <= p[0] <= xmax and ymin <= p[1] <= ymax, "OUTSIDE_CALIBRATED_REGION")
        mapped = matrix @ np.r_[p, 1]
        need(abs(mapped[2]) > 1e-9, "SINGULAR_PROJECTION")
        error = number(cal["max_error_mm"])
        need(0 < error <= 10, "CALIBRATION_ERROR_REQUIRED")
        return (mapped[:2] / mapped[2]).tolist(), error

    def stop(self):
        with self.lock:
            self.running = False
            self.sessions.clear()
            self.requests.clear()
            self.latest.clear()
            self.seen.clear()
