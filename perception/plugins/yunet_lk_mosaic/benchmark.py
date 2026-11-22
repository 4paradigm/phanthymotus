"""Paced JPEG replay for the current Card and pinned PR #382 baseline.

Run each variant in its own interpreter with PYTHONPATH set to that checkout's
perception directory. See PERFORMANCE.md for reproducible commands.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import threading
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from utils.latest_frame import LatestFrame


class Timings:
    def __init__(self):
        self.samples = defaultdict(list)

    def record(self, stage, seconds):
        self.samples[stage].append(seconds * 1000)

    def report(self):
        return {
            stage: {"n": len(values), "mean": round(float(np.mean(values)), 3),
                    "p50": round(float(np.percentile(values, 50)), 3),
                    "p95": round(float(np.percentile(values, 95)), 3)}
            for stage, values in self.samples.items() if values
        }


class TimedCV:
    def __init__(self, timings):
        self.timings = timings

    def __getattr__(self, name):
        return getattr(cv2, name)

    def calcOpticalFlowPyrLK(self, *args, **kwargs):
        start = time.perf_counter()
        try:
            return cv2.calcOpticalFlowPyrLK(*args, **kwargs)
        finally:
            self.timings.record("lk", time.perf_counter() - start)


def load_clip(args):
    video = cv2.VideoCapture(str(args.video))
    try:
        video.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)
        fps = video.get(cv2.CAP_PROP_FPS)
        if fps < 29.9:
            raise ValueError("expected source video at about 30 FPS or higher")
        frames = []
        source_index = -1
        for frame_index in range(round(args.seconds * 30)):
            seek_index = round(frame_index * fps / 30)
            while source_index < seek_index:
                ok, frame = video.read()
                if not ok:
                    raise ValueError("video ended early")
                source_index += 1
            if frame.shape[0] * 16 != frame.shape[1] * 9:
                raise ValueError("expected a 16:9 source clip")
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise ValueError("input encoding failed")
            frames.append(encoded.tobytes())
        return frames
    finally:
        video.release()


def replay_old(args, frames, timings, publish):
    from plugins.mosaic_smooth_runtime import DemoConfig, SmoothMosaic

    class Baseline(SmoothMosaic):
        def _detect(self, image):
            start = time.perf_counter()
            try:
                return super()._detect(image)
            finally:
                timings.record("detect", time.perf_counter() - start)

    engine = Baseline(str(args.model), DemoConfig(), cv=TimedCV(timings))
    # Warm the same model outside the measurement window.
    engine.process(frames[0])
    engine.number = engine.detect_calls = engine.fallback_frames = 0
    engine.last_boxes = []
    engine.points = engine.prev_gray = None
    timings.samples.clear()
    inbox = LatestFrame()
    counts = defaultdict(int)

    def run():
        while not inbox.closed:
            pending = inbox.pop(0.02)
            if pending is None:
                continue
            jpg, sequence, capture = pending
            if time.monotonic() - capture > .5:
                counts["expired_input"] += 1
                continue
            started = time.perf_counter()
            encoded, faces, fallback = engine.process(jpg)
            timings.record("process", time.perf_counter() - started)
            counts["processed"] += 1
            stamp_age = time.monotonic() - capture
            if stamp_age > .6:
                counts["expired_output"] += 1
                continue
            counts["coarse_frames"] += int(fallback)
            counts["face_frames"] += int(bool(faces))
            publish(sequence, capture, encoded, [(0, 0, 640, 360)])

    def result():
        return {**counts, "input_overwritten": inbox.dropped,
                "detection_calls": engine.detect_calls, "backend": "opencv"}

    return run, inbox.close, lambda jpg, i, t: inbox.push((jpg, i, t)), result, engine.close


def replay_new(args, frames, timings, publish):
    from plugins.yunet_lk_mosaic.config import MosaicConfig
    from plugins.yunet_lk_mosaic.runtime import MosaicEngine
    from plugins.yunet_lk_mosaic.stream import InputFrame, MosaicStream

    config = MosaicConfig().updated({
        "target_fps": args.fps, "publish_delay_ms": args.delay,
        "adaptive_detection": args.adaptive, "confidence_threshold": args.threshold,
        "miss_compensation": not args.no_compensation,
        "detect_every_frames": args.interval,
    })
    engine = MosaicEngine(str(args.model), config, timing=timings.record)
    engine.detector.detect(engine.decode(frames[0]))
    timings.samples.clear()

    def on_output(frame, encoded, rectangles):
        publish(frame.stamp, frame.captured_at, encoded, rectangles)

    def on_error(exc, _frame):
        raise exc

    stream = MosaicStream(engine, config, on_output, on_error)

    def send(jpg, sequence, capture):
        stream.inbox.push(InputFrame(jpg, sequence, capture, capture))

    return stream.run, stream.stop, send, stream.snapshot, engine.close


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("old", "new"), required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--start", type=float, default=30)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--delay", type=int, default=0)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--threshold", type=float, default=.28)
    parser.add_argument("--interval", type=int, default=4)
    parser.add_argument("--no-compensation", action="store_true")
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(args.threads)  # standalone benchmark process only
    frames = load_clip(args)
    digest = hashlib.sha256()
    timings = Timings()
    latencies, output_times, stamps, effect_ratios = [], [], [], []
    failures = []
    for jpg in frames:
        digest.update(len(jpg).to_bytes(4, "little"))
        digest.update(jpg)

    def publish(sequence, capture, _encoded, effect):
        now = time.monotonic()
        stamps.append(sequence)
        output_times.append(now)
        latencies.append(now - capture)
        # Input replay and configured output are both 640x360 for this clip.
        union = np.zeros((360, 640), dtype=bool)
        for x1, y1, x2, y2 in effect:
            union[y1:y2, x1:x2] = True
        effect_ratios.append(float(union.mean()))

    setup = replay_old if args.variant == "old" else replay_new
    run, stop, send, result, close = setup(args, frames, timings, publish)

    def worker():
        try:
            run()
        except Exception as exc:
            failures.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    started = time.monotonic() + .03
    try:
        for index, jpg in enumerate(frames):
            capture = started + index / 30
            time.sleep(max(0, capture - time.monotonic()))
            send(jpg, index + 1, capture)
        time.sleep(max(0, started + len(frames) / 30 + args.delay / 1000 + .15
                       - time.monotonic()))
    finally:
        stop()
        thread.join()
        close()
    if failures:
        raise failures[0]
    if len(stamps) < 2 or any(a >= b for a, b in zip(stamps, stamps[1:])):
        raise RuntimeError("no output or timestamps out of order")
    clip_s = len(frames) / 30
    counts = result()
    for latency in latencies:
        timings.record("source_to_publish", latency)
    report = {
        "variant": args.variant,
        "options": {k: getattr(args, k) for k in (
            "start", "seconds", "fps", "delay", "adaptive", "threshold", "interval",
            "no_compensation", "threads")},
        "host": {"platform": platform.machine(), "python": platform.python_version(),
                 "opencv": cv2.__version__, "numpy": np.__version__,
                 "cpu_affinity": sorted(os.sched_getaffinity(0))},
        "input_frames": len(frames), "input_jpeg_sequence_sha256": digest.hexdigest(),
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "processed_fps": round(counts["processed"] / clip_s, 3),
        "output_fps": round(len(stamps) / clip_s, 3),
        "output_active_fps": round((len(stamps) - 1) /
                                   (output_times[-1] - output_times[0]), 3),
        "published": len(stamps),
        "mean_image_effect_fraction": round(float(np.mean(effect_ratios)), 5),
        "counts": counts, "timing_ms": timings.report(),
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
