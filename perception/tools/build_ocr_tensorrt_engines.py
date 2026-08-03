#!/usr/bin/env python3
"""Build deployment-local FP16 TensorRT engines for the OCR plugin.

TensorRT engines are tied to the target TensorRT/CUDA/GPU stack. Run this
tool on the Jetson that will execute the engines; do not commit its outputs.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path


Shape = tuple[int, int, int, int]


@dataclass(frozen=True)
class ShapeProfile:
    name: str
    minimum: Shape
    optimum: Shape
    maximum: Shape

    def contains(self, shape: Shape) -> bool:
        return all(
            lower <= value <= upper
            for value, lower, upper in zip(
                shape, self.minimum, self.maximum
            )
        )


# The detector resizes each side to a multiple of 32 and caps the longer side
# at 1600. Overlapping aspect-ratio profiles keep common inputs near an
# optimum while the final profile guarantees complete 32..1600 coverage.
DETECTOR_PROFILES = (
    ShapeProfile(
        "landscape",
        (1, 3, 32, 32),
        (1, 3, 704, 1216),
        (1, 3, 1216, 1600),
    ),
    ShapeProfile(
        "portrait",
        (1, 3, 32, 32),
        (1, 3, 1216, 704),
        (1, 3, 1600, 1216),
    ),
    ShapeProfile(
        "general",
        (1, 3, 32, 32),
        (1, 3, 896, 896),
        (1, 3, 1600, 1600),
    ),
)


# Recognition crops have a fixed height. Most lines fit the exact-width 320
# profile, where batching yields the largest gain. Wider crops stay batch 1 to
# keep tactic/activation memory bounded on an 8 GB Orin. Widths above 2048 fall
# back to MNN at run time when configured by the OCR plugin.
RECOGNIZER_PROFILES = (
    ShapeProfile(
        "short",
        (1, 3, 48, 320),
        (4, 3, 48, 320),
        (8, 3, 48, 320),
    ),
    ShapeProfile(
        "wide",
        (1, 3, 48, 328),
        (1, 3, 48, 640),
        (1, 3, 48, 2048),
    ),
)


def _validate_profiles(profiles: tuple[ShapeProfile, ...]) -> None:
    for profile in profiles:
        for lower, optimum, upper in zip(
            profile.minimum, profile.optimum, profile.maximum
        ):
            if not lower <= optimum <= upper:
                raise ValueError(f"invalid TensorRT profile: {profile}")


def build_engine(
    onnx_path: Path,
    output_path: Path,
    profiles: tuple[ShapeProfile, ...],
    *,
    workspace_mb: int,
    optimization_level: int,
) -> None:
    import tensorrt as trt

    _validate_profiles(profiles)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(
            str(parser.get_error(index)) for index in range(parser.num_errors)
        )
        raise RuntimeError(f"failed to parse {onnx_path}:\n{errors}")
    if network.num_inputs != 1:
        raise RuntimeError(
            f"OCR ONNX must have exactly one input, got {network.num_inputs}"
        )
    input_tensor = network.get_input(0)
    if len(tuple(input_tensor.shape)) != 4:
        raise RuntimeError(
            f"OCR ONNX input must be rank 4, got {input_tensor.shape}"
        )

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_mb) * 1024 * 1024
    )
    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = int(optimization_level)
    for shape_profile in profiles:
        profile = builder.create_optimization_profile()
        accepted = profile.set_shape(
            input_tensor.name,
            shape_profile.minimum,
            shape_profile.optimum,
            shape_profile.maximum,
        )
        # TensorRT 10.4 returns None on success while newer releases return
        # bool, so reject only an explicit False value.
        if accepted is False:
            raise RuntimeError(
                f"TensorRT rejected profile {shape_profile.name}"
            )
        profile_index = config.add_optimization_profile(profile)
        if isinstance(profile_index, int) and profile_index < 0:
            raise RuntimeError(
                f"TensorRT failed to add profile {shape_profile.name}"
            )

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT failed to build {onnx_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            delete=False,
        ) as temporary:
            temporary.write(bytes(serialized))
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_name = temporary.name
        os.replace(temporary_name, output_path)
        os.chmod(output_path, 0o644)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--det-onnx", type=Path)
    parser.add_argument("--rec-onnx", type=Path)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--component",
        choices=("all", "det", "rec"),
        default="rec",
        help="build both engines or only one component",
    )
    parser.add_argument("--workspace-mb", type=int, default=512)
    parser.add_argument(
        "--builder-optimization-level", type=int, choices=range(0, 6), default=3
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required = [args.keys]
    if args.component in ("all", "det"):
        if args.det_onnx is None:
            raise ValueError("--det-onnx is required for detector builds")
        required.append(args.det_onnx)
    if args.component in ("all", "rec"):
        if args.rec_onnx is None:
            raise ValueError("--rec-onnx is required for recognizer builds")
        required.append(args.rec_onnx)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.workspace_mb <= 0:
        raise ValueError("--workspace-mb must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.component in ("all", "det"):
        print(f"Building detector from {args.det_onnx}", flush=True)
        build_engine(
            args.det_onnx,
            args.output_dir / "det.engine",
            DETECTOR_PROFILES,
            workspace_mb=args.workspace_mb,
            optimization_level=args.builder_optimization_level,
        )
    if args.component in ("all", "rec"):
        print(f"Building recognizer from {args.rec_onnx}", flush=True)
        build_engine(
            args.rec_onnx,
            args.output_dir / "rec.engine",
            RECOGNIZER_PROFILES,
            workspace_mb=args.workspace_mb,
            optimization_level=args.builder_optimization_level,
        )
    shutil.copyfile(args.keys, args.output_dir / "keys.txt")
    print(f"TensorRT OCR bundle written to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
