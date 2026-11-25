"""Offline replay or loopback MCP/Canvas preview; never starts a camera by default."""
import argparse
import json
import time
from pathlib import Path
from .sources.capture import FileSource
from .engine import Engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="local JSON config")
    parser.add_argument("--video", help="replay this local video")
    parser.add_argument("--output", help="output annotated MP4")
    parser.add_argument("--metadata", help="output per-frame JSONL")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=0, help="0 selects an unused loopback port")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.serve:
        from .server import serve
        serve(cfg, args.port)
        return
    if not args.video:
        parser.error("--video required unless --serve")
    import cv2
    source = engine = writer = metadata = None
    try:
        source = FileSource(args.video, paced=False)
        engine = Engine(cfg)
        metadata = open(args.metadata, "w") if args.metadata else None
        frames = rendered = 0
        begin = time.monotonic()
        while (frame := source.read()) is not None:
            frames += 1
            result = engine.process(frame)
            if result is None:
                continue
            rendered += 1
            if args.output:
                if writer is None:
                    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), source.fps, frame.size)
                    if not writer.isOpened():
                        raise RuntimeError("output video encoder unavailable")
                writer.write(result.image)
            if metadata:
                metadata.write(json.dumps(result.metadata, allow_nan=False)+"\n")
        print(json.dumps({"input_frames": frames, "output_frames": rendered,
                          "throughput_fps": rendered/(time.monotonic()-begin),
                          "mode": "unpaced offline replay; not source-to-display FPS"}))
    finally:
        if metadata:
            metadata.close()
        if writer:
            writer.release()
        if engine:
            engine.close()
        if source:
            source.close()
            source.finish()


if __name__ == "__main__":
    main()
