#!/usr/bin/env python3
"""tools/export_vision_engines.py — build the vop / visual_depth TensorRT engines.

Run this INSIDE a container built from the target perception image — not on the
Jetson host, and not in any other container.

An engine plan only loads on the exact TensorRT that built it, and the image
ships its own TensorRT independently of the host's. On Orin 6 the host carries
TensorRT 10.3 while the jp6.1 perception image carries 10.4, so an engine built
on that host deserializes nowhere: every jp6.1 robot rejects it with
"engine plan file is not compatible with this version of TensorRT, expecting
library version 10.4.0.26". An earlier version of this note said "on a host of
the target JetPack line", which is how that happened.

    docker run --rm --runtime nvidia --network host \
      -v "$PWD/out:/work/exp" -w /work/exp -e YOLO_CONFIG_DIR=/work/exp \
      --entrypoint bash <perception-image> -lc \
      'source /etc/dla-fallback.env; python3 export_vision_engines.py --out /work/exp/engines'

`source /etc/dla-fallback.env` is required: on vendor BSPs missing
libnvdla_compiler.so, importing tensorrt fails outright without it, and the
image's own CMD sources it for exactly this reason.

The image must still carry ultralytics, which the runtime image no longer does
— use a pre-removal tag, or pip install it into the throwaway container.

Install the export-only dependencies yourself, pinning numpy to whatever the
image ships:

    pip3 install -i <reachable-mirror> onnx onnxslim "numpy==$(python3 -c 'import numpy;print(numpy.__version__)')"

Three reasons, each of which cost a failed build:

* **onnx is not in the image** and ultralytics' AutoUpdate cannot install it
  here — the Orins reach github.com but not pypi.org, so the automatic
  `pip install` fails and the export dies on `No module named 'onnx'`. Naming
  a reachable mirror is the fix; `mirrors.tencent.com` works from the office.
* **Pin numpy or onnx will raise it**, and the base's cv2 and torch are built
  against the version the image ships. Unpinned, the next import fails with
  `numpy.core.multiarray failed to import`.
* **Never run this in a live container.** AutoUpdate, when it does have a
  route, silently installs onnx and drags protobuf from 3.6.1 to 5.x — a
  shared dependency of onnxruntime and sherpa. A running perception container
  was polluted that way once, and `docker restart` does not undo it.

The engines MUST come from ultralytics' own exporter rather than trtexec: the
plugins load them back through `YOLO("....engine")`, and that loader requires
the metadata the ultralytics exporter embeds. A trtexec-built engine
deserializes fine and is then rejected on load.

    python3 tools/export_vision_engines.py --out /tmp/engines
    python3 tools/export_vision_engines.py --model vop --imgsz 640

Outputs, per model:
    yoloe-26s-seg.engine + vocab.json     (vop)
    yolo26n-depth.engine                  (visual_depth)

Then upload to COS and record size + SHA256 of the *uploaded* copy (re-download
it and hash that) in utils/model_downloader.py. See that file's bundle tables.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys

# ── The frozen vocabulary ────────────────────────────────────────────────────
#
# This list IS the product: an exported open-vocabulary model can no longer be
# re-prompted, so whatever is here is everything vop will ever detect on the
# robots this engine ships to. Err wide — an unused class costs a little
# latency (region-text similarity is computed per class on every forward pass,
# ~19% from 80 to 1200 classes by ultralytics' measurement), a missing one
# costs a rebuild and a republish.
COCO_80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

# Indoor / office / warehouse vocabulary these robots actually operate in.
ROBOT_EXTRA = [
    "door", "doorway", "elevator", "stairs", "handrail", "corridor", "window", "curtain",
    "desk", "office chair", "whiteboard", "projector", "screen", "monitor", "printer",
    "server rack", "cable", "power outlet", "light switch", "trash can", "box",
    "cardboard box", "pallet", "forklift", "shelf", "cabinet", "drawer",
    "robot", "quadruped robot", "humanoid robot", "drone", "robot arm", "charging dock",
    "traffic cone", "warning sign", "fire extinguisher", "first aid kit", "exit sign",
    "badge", "lanyard", "helmet", "safety vest", "glasses", "mask", "glove",
    "water dispenser", "coffee machine", "microphone", "speaker", "camera", "tripod",
    "plant", "painting", "poster", "banner", "sofa", "stool", "table", "mat", "carpet",
    "puddle", "obstacle", "hand", "face",
]


def vocabulary() -> list[str]:
    return COCO_80 + [c for c in ROBOT_EXTRA if c not in COCO_80]


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_vop(out_dir: str, imgsz: int, workspace: float | None) -> list[str]:
    from ultralytics import YOLOE

    vocab = vocabulary()
    model = YOLOE("yoloe-26s-seg.pt")
    try:
        model.set_classes(vocab)
    except TypeError:
        # Older signature wants the precomputed text embeddings explicitly.
        model.set_classes(vocab, model.get_text_pe(vocab))
    print(f"[export] vop vocabulary frozen: {len(vocab)} classes", flush=True)

    # nms=False selects YOLO26's NMS-free end-to-end head, so the engine emits
    # final boxes and plugins/vision_runtime.py only has to filter by score and
    # undo the letterbox. Exporting with NMS instead changes the output layout
    # and the decoder refuses it rather than misreading it.
    engine = model.export(format="engine", imgsz=imgsz, half=True,
                          workspace=workspace, nms=False)
    target = os.path.join(out_dir, "yoloe-26s-seg.engine")
    shutil.move(str(engine), target)

    # Shipped beside the engine so the plugin can name what it detects without
    # this list being restated (and drifting) in plugin code.
    vocab_path = os.path.join(out_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as handle:
        json.dump(
            {"model": "yoloe-26s-seg", "imgsz": imgsz, "classes": vocab},
            handle, ensure_ascii=False, indent=1,
        )
    return [target, vocab_path]


def export_depth(out_dir: str, imgsz: int, workspace: float | None) -> list[str]:
    from ultralytics import YOLO

    engine = YOLO("yolo26n-depth.pt").export(
        format="engine", imgsz=imgsz, half=True, workspace=workspace
    )
    target = os.path.join(out_dir, "yolo26n-depth.engine")
    shutil.move(str(engine), target)
    return [target]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("vop", "depth", "both"), default="both")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out", default="./engines")
    # Jetson memory is shared between CPU and GPU, so an unbounded TensorRT
    # builder workspace is not a soft preference — on an 8 GB Orin already
    # running the perception stack it gets the build OOM-killed outright
    # (observed on jp5.11: "Killed" mid-[GpuLayer], no Python traceback). Cap it.
    parser.add_argument("--workspace", type=float, default=2.0,
                        help="TensorRT builder workspace in GB (0 = unbounded)")
    args = parser.parse_args()
    workspace = args.workspace if args.workspace > 0 else None

    os.makedirs(args.out, exist_ok=True)

    try:
        import tensorrt as trt
        print(f"[export] TensorRT {trt.__version__}", flush=True)
    except ImportError:
        print("[export] TensorRT is not importable here — run this on a Jetson "
              "of the target JetPack line", file=sys.stderr)
        return 2

    produced: list[str] = []
    if args.model in ("both", "vop"):
        produced += export_vop(args.out, args.imgsz, workspace)
    if args.model in ("both", "depth"):
        produced += export_depth(args.out, args.imgsz, workspace)

    print("\n[export] record these in utils/model_downloader.py — but re-hash "
          "the COPY DOWNLOADED BACK FROM COS, not these local files: a pin that "
          "hashes the source cannot catch a bad upload.\n", flush=True)
    for path in produced:
        print(f'    "{os.path.basename(path)}": {{'
              f'"size": {os.path.getsize(path)}, '
              f'"sha256": "{sha256(path)}"}},', flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
