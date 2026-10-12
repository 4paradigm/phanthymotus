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
    yolo26s-pose.engine                   (pose; --pose-weights to pick another
                                           size, and update the bundle table)

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


# Which pose weights to export. `s` rather than `n` on purpose: the keypoints
# feed geometric action rules, so precision here shows up as fewer misjudged
# actions rather than as a nicer picture (57.2 → 63.0 mAP(pose) for ~12 MB more
# resident and ~8 ms more per frame). Overridable with --pose-weights, so
# re-exporting another size needs no code change — but then also update
# POSE_MODEL_BUNDLES, whose paths name the model.
DEFAULT_POSE_WEIGHTS = "yolo26s-pose.pt"


def export_pose(out_dir: str, imgsz: int, workspace: float | None,
                weights: str = DEFAULT_POSE_WEIGHTS) -> list[str]:
    """Build the human-keypoint engine for plugins/pose.py.

    COCO-17, single class. The YOLO11 equivalent is tried as a fallback because
    which weights the image's ultralytics can fetch is a property of the image
    rather than something to assume; it should not fire, since ultralytics
    publishes yolo26{n,s,m,l,x}-pose and this repo already runs YOLO26 elsewhere.
    Either decodes through plugins/vision_runtime.decode_poses, which picks the
    layout by content.
    """
    from ultralytics import YOLO

    candidates = [weights]
    fallback = weights.replace("yolo26", "yolo11")
    if fallback != weights:
        candidates.append(fallback)

    model = None
    errors = []
    for candidate in candidates:
        try:
            model = YOLO(candidate)
            weights = candidate
            break
        except Exception as exc:                        # noqa: BLE001
            errors.append(f"{candidate}: {exc}")
    if model is None:
        raise RuntimeError("no pose weights could be loaded — tried:\n  "
                           + "\n  ".join(errors))
    print(f"[export] pose weights: {weights}", flush=True)

    engine = model.export(format="engine", imgsz=imgsz, half=True,
                          workspace=workspace, nms=False)
    # Named after the weights actually used, so the bundle in
    # utils/model_downloader.py and the file cannot disagree about which model
    # a robot is running.
    target = os.path.join(out_dir, weights.replace(".pt", ".engine"))
    shutil.move(str(engine), target)
    return [target]


# ── skeleton-action engine ───────────────────────────────────────────────────

DEFAULT_ACTION_WINDOW = 48


DEFAULT_HAND_IMGSZ = 448

#: The ultralytics that produced the published ONNX. Recorded, not just for
#: provenance: the same weights exported by 8.4.33 and 8.4.175 disagree by a
#: median NME of 0.0145 with one finger decision in seventy flipped, which is
#: about thirty times the fp16 quantisation difference — and every size and
#: SHA256 check still passes. See HAND_MODEL_BUNDLES.
HAND_ONNX_ULTRALYTICS = "8.4.175"


def export_hand(out_dir: str, workspace: float | None,
                onnx_path: str | None = None,
                imgsz: int = DEFAULT_HAND_IMGSZ) -> list[str]:
    """Build the 21-keypoint hand engine for plugins/hand_runtime.py.

    Unlike vop/depth/pose this one does **not** go through ultralytics'
    exporter. It builds with `trtexec` from the pinned ONNX, for two reasons:

    * The plugins load these plans through `utils.tensorrt_runtime` directly,
      not through `YOLO("...engine")`, so the metadata header the ultralytics
      exporter embeds is not needed. (The header-free plan is also why
      `hand_runtime.assert_end2end` checks the output *shape* rather than an
      `end2end` metadata flag — there is no metadata at all.)
    * Re-exporting from the `.pt` would re-run whatever ultralytics the
      container happens to have, and that changes the numbers. Building from
      the pinned ONNX keeps the plan comparable across rebuilds.

    So a rebuild is reproducible and does not need torch, ultralytics or the
    `.pt` in the container — only TensorRT, which the perception image has.

    If you *do* need to regenerate the ONNX, the export is
    `YOLO(weights).export(format="onnx", imgsz=448, opset=17, simplify=True,
    nms=True)` — and `nms=True` is not optional on ultralytics >= 8.4.17x:
    without it the export is the raw head, which is one column narrower and
    decodes silently as a valid pose layout. Then re-pin HAND_ONNX.
    """
    if onnx_path is None:
        from utils.model_downloader import ensure_hand_onnx
        paths = ensure_hand_onnx(os.environ.get("HAND_MODEL_DIR", "/models/hand"))
        onnx_path = next(iter(paths.values()))
        print(f"[export] hand ONNX: {onnx_path}", flush=True)
    if not os.path.isfile(onnx_path):
        raise RuntimeError(f"hand ONNX not found: {onnx_path!r}")

    trtexec = "/usr/src/tensorrt/bin/trtexec"
    if not os.path.isfile(trtexec):
        found = shutil.which("trtexec")
        if not found:
            raise RuntimeError(
                "trtexec not found. Run this inside a container built from the "
                "target perception image, which ships it at "
                f"{trtexec} — not on the Jetson host, whose TensorRT is a "
                "different version and whose plans load nowhere.")
        trtexec = found

    engine = os.path.join(out_dir, "yolo26s-hand21.engine")
    command = [trtexec, f"--onnx={onnx_path}", f"--saveEngine={engine}", "--fp16"]
    if workspace:
        command.append(f"--memPoolSize=workspace:{int(workspace * 1024)}")

    print(f"[export] building {engine} at {imgsz}x{imgsz} (fp16)", flush=True)
    import subprocess
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.isfile(engine):
        tail = "\n".join((result.stderr or result.stdout or "").splitlines()[-25:])
        raise RuntimeError(f"trtexec failed ({result.returncode}):\n{tail}")

    # Verified here rather than left to the robot: a plan that deserializes and
    # then reports the wrong row width is the one failure mode that produces
    # thousands of plausible hands instead of an error.
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from plugins.hand_runtime import assert_end2end
        from plugins.vision_runtime import VisionEngineSession

        session = VisionEngineSession(engine)
        shape = assert_end2end(session)
        print(f"[export] end2end check passed, output {shape}", flush=True)
        session.close()
    except Exception as error:
        raise RuntimeError(
            f"the plan built but failed its end2end check: {error}") from error

    return [engine]


def export_action(out_dir: str, window: int, workspace: float | None,
                  checkpoint: str | None) -> list[str]:
    """Build the ST-GCN++ skeleton-action engine for plugins/pose_stgcn.py.

    Unlike the three above, this one needs a **checkpoint you supply**: pass
    `--action-checkpoint` pointing at a PYSKL ST-GCN++ NTU60-XSub *2D* joint
    weight file (the `j.pth` from the stgcn++_ntu60_xsub_hrnet config). There is
    no ultralytics-style auto-download, and guessing a URL for weights whose
    licence and provenance matter is not something this script should do.

    It also needs `pyskl`/`mmaction2` + torch importable, which the perception
    image does not carry — so this step runs in a throwaway container like the
    others, with those installed alongside onnx/onnxslim.

    Input is (N, M, T, V, C) = (1, 2, window, 17, 3), PYSKL's FormatGCNInput
    order, matching `SkeletonActionBackend._build_input` — M is
    NUM_PERSON_SLOTS and C is NUM_CHANNELS (x, y, score), both read from
    plugins/pose_stgcn.py rather than restated. An earlier version of this
    line said M=1 and C=2; the checkpoint's `data_bn.weight` is 51 = 3x17,
    which settles C, and getting either wrong produces a graph the robot's
    tensor cannot be fed to. The temporal size is
    baked into the engine, and the backend reads it back off the engine rather
    than trusting its own constant — but the two still have to agree about the
    *layout*, which is why both name it in the same order.
    """
    try:
        import torch
    except ImportError as exc:                           # pragma: no cover
        raise RuntimeError("torch is required to export the action engine") from exc

    if not checkpoint:
        # Fetched from the project's own mirror, pinned by size+SHA256 like
        # every other artefact. No URL is guessed: the upstream openmmlab path
        # is recorded in utils/model_downloader.py beside the pins, and the
        # mirrored copy was verified byte-identical to it.
        from utils.model_downloader import ensure_action_checkpoint
        paths = ensure_action_checkpoint(os.environ.get("ACTION_MODEL_DIR",
                                                        "/models/action"))
        checkpoint = next(iter(paths.values()))
        print(f"[export] action checkpoint: {checkpoint}", flush=True)
    if not os.path.isfile(checkpoint):
        raise RuntimeError(f"action checkpoint not found: {checkpoint!r}")

    raise RuntimeError(
        "the ONNX export path for ST-GCN++ is not implemented here yet.\n"
        "\n"
        "What is missing is only the graph export, and it is deliberately not "
        "guessed: PYSKL's recogniser wraps the backbone in a test-time pipeline "
        "(`forward_test` averages over clips and people), so exporting the "
        "recogniser gives a graph whose input is not the tensor the robot has. "
        "The backbone + head have to be traced directly on a "
        f"(1, 2, {window}, 17, 3) input, and that wiring depends on the pyskl "
        "version in the container.\n"
        "\n"
        "The checkpoint itself is in hand and pinned, and its state_dict says "
        "what has to be traced: `backbone.data_bn`, ten `backbone.gcn.N` "
        "blocks each with a `.gcn` (adjacency `A` stored in the checkpoint, so "
        "the graph does not have to be rebuilt) and a six-branch `.tcn`, then "
        "`cls_head.fc_cls`. Reimplementing that by hand in plain torch is "
        "possible and is NOT the path taken here: a mis-wired branch loads "
        "fine and returns wrong numbers, and there is no reference output to "
        "check against without pyskl. Install pyskl in the container and trace "
        "its own modules.\n"
        "\n"
        "Until then the pose card runs `action_backend: rules` and says so in "
        "`info` (action_backend_effective / action_backend_note)."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",
                        choices=("vop", "depth", "pose", "hand", "action",
                                 "both"),
                        default="both")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out", default="./engines")
    # Jetson memory is shared between CPU and GPU, so an unbounded TensorRT
    # builder workspace is not a soft preference — on an 8 GB Orin already
    # running the perception stack it gets the build OOM-killed outright
    # (observed on jp5.11: "Killed" mid-[GpuLayer], no Python traceback). Cap it.
    parser.add_argument("--hand-onnx", default=None,
                        help="hand ONNX to build from for --model hand "
                             "(default: the pinned copy from COS). Exported "
                             f"with ultralytics {HAND_ONNX_ULTRALYTICS}; see "
                             "export_hand on why a re-export is not free")
    parser.add_argument("--action-checkpoint", default=None,
                        help="PYSKL ST-GCN++ NTU60-XSub-2D joint checkpoint "
                             "(.pth) for --model action")
    parser.add_argument("--action-window", type=int, default=DEFAULT_ACTION_WINDOW,
                        help="frames the action engine takes (default %(default)s)")
    parser.add_argument("--pose-weights", default=DEFAULT_POSE_WEIGHTS,
                        help="pose weights to export (default %(default)s). "
                             "Changing this also means updating "
                             "POSE_MODEL_BUNDLES, whose COS paths name the model")
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
    if args.model in ("both", "pose"):
        produced += export_pose(args.out, args.imgsz, workspace,
                                weights=args.pose_weights)
    # Not in "both": the hand engine is an opt-in channel on the pose card, and
    # its input size is 448 rather than --imgsz, so folding it into a bulk build
    # would silently build it at whatever the others were asked for.
    if args.model == "hand":
        produced += export_hand(args.out, workspace, onnx_path=args.hand_onnx)
    # Not in "both": it needs a checkpoint the caller supplies, so it would
    # break every vop/depth/pose export if it were on by default.
    if args.model == "action":
        produced += export_action(args.out, args.action_window, workspace,
                                  args.action_checkpoint)

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
