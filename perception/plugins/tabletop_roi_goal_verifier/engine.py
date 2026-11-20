"""Bounded ROI sessions with fresh-frame ordering and idempotent request receipts."""

import threading
import time
from collections import OrderedDict
from copy import deepcopy
from uuid import uuid4

from .contracts import RESULT, Refusal, canonical, digest, need, number, text
from .roi import settings


class Engine:
    def __init__(self, config, *, wall=None):
        self.config = deepcopy(config)
        self.sources = self.config["sources"]
        self.wall = wall or time.time
        self.last_wall = None
        self.lock = threading.RLock()
        self.max_age = config.get("max_age_s", 2)
        self.ttl = config.get("session_ttl_s", 600)
        self.sessions = {}
        self.roi_images = {}
        self.requests = OrderedDict()
        self.running = True

    def _now(self):
        now = number(self.wall())
        if self.last_wall is not None and now < self.last_wall:
            for session in self.sessions.values():
                session["clock_invalid"] = True
                session["result"] = None
            self.roi_images.clear()
            self.requests.clear()
            self.last_wall = now
            need(False, "CLOCK_ROLLBACK")
        self.last_wall = now
        return now

    def _session(self, args, now):
        s = self.sessions.get(args.get("session_id"))
        need(s is not None, "UNKNOWN_SESSION")
        need(args.get("session_token") == s["session_token"], "RETIRED_SESSION")
        need(now < s["expires_at"], "SESSION_EXPIRED")
        need(not s.get("clock_invalid"), "SESSION_CLOCK_INVALID")
        return s

    @staticmethod
    def _key(args):
        return (
            args.get("session_id", "new"),
            args.get("session_token", "new"),
            args.get("request_id"),
        )

    def _remember(self, action, args, out):
        self.requests[self._key(args)] = (
            digest({"action": action, "args": args}),
            deepcopy(out),
        )
        while len(self.requests) > 256:
            self.requests.popitem(last=False)
        return out

    def replay(self, action, args):
        need(self.running, "STOPPED")
        saved = self.requests.get(self._key(args))
        if saved:
            need(
                saved[0] == digest({"action": action, "args": args}),
                "REQUEST_ID_CONFLICT",
            )
            # Reset/close receipts remain retryable after retiring their token.
            return deepcopy(saved[1])
        if action != "start_session":
            self._session(args, self._now())
        return None

    def inspect(self):
        with self.lock:
            return {
                "running": self.running,
                "verifier_id": self.config["verifier_id"],
                "sources": [
                    {
                        "source_id": k,
                        "camera_id": b["camera_id"],
                        "kind": b["kind"],
                        "provenance": b.get("provenance", "real"),
                    }
                    for k, b in self.sources.items()
                ],
                "sessions": [
                    {
                        "session_id": s["session_id"],
                        "has_before": s["before"] is not None,
                        "has_marker": s["marker"] is not None,
                    }
                    for s in self.sessions.values()
                ],
            }

    def acquisition(self, args, *, after=False):
        s = self._session(args, self._now())
        if after:
            need(
                s["marker"] and args.get("marker_id") == s["marker"]["marker_id"],
                "AFTER_MARKER_REQUIRED",
            )
        return s["source_id"], s["marker"]["not_before"] if after else None

    def acquisition_mode(self, args):
        s = self._session(args, self._now())
        return "roi_change", deepcopy(s["roi_options"])

    def call(self, action, args, *, observed=None):
        with self.lock:
            canonical(args)
            need(self.running, "STOPPED")
            now = self._now()
            if action in ("get_result", "export_session"):
                s = self._session(args, now)
                out = deepcopy(
                    s
                    if action == "export_session"
                    else s["result"] or self._result(s, "NOT_VERIFIED", now)
                )
                out.pop("session_token", None)
                return out
            cached = self.replay(action, args)
            if cached is not None:
                return cached
            text(args["request_id"])
            if action == "start_session":
                for sid in [
                    sid for sid, s in self.sessions.items() if now >= s["expires_at"]
                ]:
                    self.sessions.pop(sid)
                    self.roi_images.pop(sid, None)
                need(len(self.sessions) < 32, "SESSION_LIMIT")
                source = text(args["source_id"])
                need(source in self.sources, "UNKNOWN_SOURCE")
                region, options = settings(args["target_roi"], args.get("roi_options"))
                s = {
                    "session_id": uuid4().hex,
                    "session_token": uuid4().hex,
                    "task_id": text(args["task_id"]),
                    "verifier_id": self.config["verifier_id"],
                    "source_id": source,
                    "verification_mode": "roi_change",
                    "target_roi": region,
                    "roi_options": options,
                    "created_at": now,
                    "expires_at": now + self.ttl,
                    "before": None,
                    "marker": None,
                    "result": None,
                    "history": [],
                    "observations": [],
                    "after_high": None,
                }
                self.sessions[s["session_id"]] = s
                out = self._public(s)
            else:
                s = self._session(args, now)
                if action == "reset_session":
                    self.roi_images.pop(s["session_id"], None)
                    s.pop("roi_evidence", None)
                    s.update(
                        session_token=uuid4().hex,
                        before=None,
                        marker=None,
                        result=None,
                        history=[],
                        observations=[],
                        after_high=None,
                        created_at=now,
                        expires_at=now + self.ttl,
                    )
                    out = self._public(s)
                elif action == "close_session":
                    self.roi_images.pop(s["session_id"], None)
                    self.sessions.pop(s["session_id"])
                    out = {"state": "closed", "session_id": s["session_id"]}
                elif action == "mark_operation_complete":
                    need(s["before"] is not None, "BEFORE_REQUIRED")
                    need(now > s["before"]["captured_at"], "PRE_BASELINE_MARKER")
                    s["marker"] = {"marker_id": uuid4().hex, "not_before": now}
                    s["result"] = None
                    out = deepcopy(s["marker"])
                elif action in ("capture_before", "verify_after"):
                    out = self._roi_capture(s, action, args, observed)
                else:
                    need(False, "UNKNOWN_ACTION")
            return self._remember(action, args, out)

    @staticmethod
    def _public(s):
        return {
            k: deepcopy(s[k])
            for k in (
                "session_id",
                "session_token",
                "task_id",
                "verifier_id",
                "source_id",
                "verification_mode",
                "target_roi",
                "roi_options",
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
            verifier_id=s["verifier_id"],
            verified_at=now,
            data_valid=False,
            before_observation_id=(s["before"] or {}).get("observation_id"),
            after_observation_id=None,
            criterion="roi_visual_change",
            physical_motion_confirmed=False,
            identity_confirmed=False,
            **extra,
        )

    def record_failure(self, args, exc):
        with self.lock:
            s = self._session(args, self._now())
            if self._key(args) in self.requests:
                # An invalid conflicting retry must not overwrite the original receipt or result.
                return self.replay("verify_after", args)
            # A new failed observation must not display a previous attempt's after crop.
            s.get("roi_evidence", {}).pop("after_target_jpeg_base64", None)
            out = self._result(s, getattr(exc, "code", "INPUT_ERROR"), self._now())
            out.update(
                message=str(exc),
                target_roi=s["target_roi"],
                roi_options=s["roi_options"],
                roi_evidence=deepcopy(s.get("roi_evidence", {})),
                evidence={
                    "before": deepcopy(s["before"]),
                    "marker": deepcopy(s["marker"]),
                },
            )
            s["result"] = deepcopy(out)
            s["history"] = [*s["history"], deepcopy(out)][-50:]
            return self._remember("verify_after", args, out)

    def stop(self):
        with self.lock:
            self.running = False
            self.sessions.clear()
            self.roi_images.clear()
            self.requests.clear()

    def _roi_capture(self, s, action, args, observed):
        from .roi import compare, crop, jpeg, prepare, statistics

        # Sources acquisition (including waits for distinct post-marker frames)
        # must finish before this clock sample. Never use the request/marker time
        # to validate a subsequently captured image, or derive now from its stamp.
        now = self._now()
        self._session(args, now)  # expiry / rollback may occur during acquisition
        binding = self.sources[s["source_id"]]
        if action == "capture_before":
            need(s["before"] is None, "BASELINE_ALREADY_SET")
            meta, image = prepare(observed, binding, now, self.max_age)
            patch, _ = crop(image, s["target_roi"])
            evidence = {
                "before_target_jpeg_base64": jpeg(patch),
                "before_statistics": statistics(image, s["target_roi"]),
            }
            # Commit only once all baseline evidence has been generated.
            s["before"] = meta
            s["observations"] = [deepcopy(meta)]
            self.roi_images[s["session_id"]] = image
            s["roi_evidence"] = evidence
            return dict(
                state="before_captured",
                **self._public(s),
                observation_id=meta["observation_id"],
                roi_evidence=deepcopy(s["roi_evidence"]),
            )
        need(
            s["before"] is not None and s["marker"] is not None, "AFTER_MARKER_REQUIRED"
        )
        need(args.get("marker_id") == s["marker"]["marker_id"], "RETIRED_MARKER")
        need(now - s["marker"]["not_before"] <= 120, "MARKER_EXPIRED")
        need(
            isinstance(observed, list)
            and len(observed) == s["roi_options"]["stable_frames"],
            "ROI_FRAMES_REQUIRED",
        )
        baseline = s["before"]
        previous = s.get("after_high") or baseline
        images, metas = [], []
        for packet in observed:
            meta, image = prepare(packet, binding, now, self.max_age)
            need(
                meta["captured_at"] > s["marker"]["not_before"],
                "PRE_OPERATION_OBSERVATION",
            )
            need(
                meta["captured_at"] > previous["captured_at"]
                and meta["sequence"] > previous["sequence"]
                and meta["observation_id"] != previous["observation_id"],
                "REPLAYED_AFTER_OBSERVATION",
            )
            for field in (
                "source_epoch",
                "camera_id",
                "frame_id",
                "view_id",
                "image_size",
                "distortion",
                "provenance",
            ):
                need(meta[field] == baseline[field], "SOURCE_GEOMETRY_CHANGED")
            previous = meta
            metas.append(meta)
            images.append(image)
        s["after_high"] = deepcopy(metas[-1])
        out = self._result(
            s,
            "ROI_CHANGE_UNCERTAIN",
            now,
            verification_mode="roi_change",
            target_roi=deepcopy(s["target_roi"]),
            after_observation_ids=[m["observation_id"] for m in metas],
            provenance=metas[-1]["provenance"],
        )
        out["after_observation_id"] = metas[-1]["observation_id"]
        try:
            out.update(
                compare(
                    self.roi_images[s["session_id"]],
                    images,
                    s["target_roi"],
                    s["roi_options"],
                )
            )
        except Refusal as exc:
            out.update(code=exc.code, message=str(exc))
        out["message"] = {
            "PASS": "目标 ROI 稳定出现明显变化；不证明物体身份或机械臂运动",
            "FAIL": "有效稳定画面中目标 ROI 基本没有变化",
        }.get(out["result"], out["message"])
        patch, _ = crop(images[-1], s["target_roi"])
        s["roi_evidence"]["after_target_jpeg_base64"] = jpeg(patch)
        out["roi_evidence"] = deepcopy(s["roi_evidence"])
        out["evidence"] = {
            "before": deepcopy(baseline),
            "after": deepcopy(metas),
            "marker": deepcopy(s["marker"]),
            "options": deepcopy(s["roi_options"]),
        }
        s["result"] = deepcopy(out)
        s["history"] = [*s["history"], deepcopy(out)][-50:]
        s["observations"] = [baseline, *metas]
        return out
