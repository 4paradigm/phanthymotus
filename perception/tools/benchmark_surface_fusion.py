"""Run from perception with PYTHONPATH=. on Linux."""
import resource
import statistics
import time
import zlib
from types import SimpleNamespace

import cv2
import numpy as np

from plugins.semantic_surface_fusion import NSEC, Runtime, Settings


def main():
    for width, height in [(640, 480), (1920, 1080)]:
        shape = height, width
        mask = np.ones(shape, np.uint8)
        depth = np.full(shape, 2000, "<u2")
        report = dict(width=width, height=height, classes=[
            dict(id=0, name="background", pixels=0),
            dict(id=1, name="object", pixels=width * height)])
        pair = (NSEC, "camera_optical_frame", SimpleNamespace(
            format="png", data=cv2.imencode(".png", mask)[1].tobytes()), report,
            SimpleNamespace(format="16UC1; compressedDepth zlib", data=zlib.compress(depth.tobytes())))
        k = [600, 0, width / 2, 0, 600, height / 2, 0, 0, 1]
        for count in (0, 4000):
            runtime = SimpleNamespace(calibration=(width, height, k, "camera"),
                                      settings=Settings(max_points=count))
            Runtime.fuse(runtime, pair)
            timings = []
            cpu = time.process_time()
            for _ in range(25):
                start = time.perf_counter()
                Runtime.fuse(runtime, pair)
                timings.append((time.perf_counter() - start) * 1000)
            cpu_ms = (time.process_time() - cpu) * 1000 / len(timings)
            print(f"{width}x{height} points={count}: p50={statistics.median(timings):.1f} ms "
                  f"p95={np.percentile(timings, 95):.1f} ms cpu={cpu_ms:.1f} ms/frame "
                  f"process_peak_rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f} MiB")


if __name__ == "__main__":
    main()
