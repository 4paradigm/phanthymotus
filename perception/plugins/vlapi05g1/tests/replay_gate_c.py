#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import statistics
import sys
import time
from pathlib import Path


CARD_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CARD_DIR.parent))

from vlapi05g1.policy import (
    ACTION_DIM,
    ACTION_SHAPE,
    ACTION_SPACE,
    ACTION_STEPS,
    NUM_INFERENCE_STEPS,
    ObservationSnapshot,
    PolicyClient,
    build_action_proposal,
    build_policy_payload,
    derive_health_url,
    validate_jpeg,
    validate_policy_response,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def process_cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def process_max_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def load_state(path: Path) -> tuple[float, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != 29:
        raise ValueError("recorded state must be a 29-value JSON array")
    state = []
    for index, value in enumerate(payload):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"state[{index}] must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"state[{index}] must be finite")
        state.append(number)
    return tuple(state)


def action_hash(action_chunk: list[list[float]]) -> str:
    encoded = json.dumps(action_chunk, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def max_abs_difference(
    baseline: list[list[float]],
    candidate: list[list[float]],
) -> float:
    return max(
        abs(baseline[row][column] - candidate[row][column])
        for row in range(ACTION_STEPS)
        for column in range(ACTION_DIM)
    )


def validate_health(payload: dict) -> None:
    expected = {
        "ok": True,
        "action_dim": ACTION_DIM,
        "action_chunk_size": ACTION_STEPS,
        "action_space": ACTION_SPACE,
        "fresh_inference_per_request": True,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "task_text_supported": True,
        "tokenizer_loaded": True,
        "allow_zero_observation": False,
    }
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"policy health contract mismatch: {mismatches}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run vlapi05g1 recorded-data Gate C")
    parser.add_argument("--policy-url", required=True)
    parser.add_argument("--health-url")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.runs < 3:
        raise ValueError("Gate C requires at least three runs")
    image_path = args.image.resolve()
    state_path = args.state.resolve()
    image_bytes = validate_jpeg(image_path.read_bytes(), 16 * 1024 * 1024)
    state = load_state(state_path)
    health_url = args.health_url or derive_health_url(args.policy_url)
    client = PolicyClient(args.policy_url, args.timeout_s, health_url)
    health = client.health()
    validate_health(health)

    snapshot = ObservationSnapshot(
        image_bytes=image_bytes,
        image_topic="recorded://g1-fixed-frame.jpg",
        image_frame_id="recorded_g1_front_camera",
        image_stamp=image_path.stat().st_mtime,
        image_stamp_source="recorded_file_mtime",
        image_received_at=time.time(),
        image_age_at_request_s=0.0,
        state=state,
        state_topic="recorded://g1-real-runtime-v2-step10.state.json",
        state_received_at=time.time(),
        state_age_at_request_s=0.0,
    )
    request = build_policy_payload(snapshot, args.task, args.seed)

    runs = []
    chunks: list[list[list[float]]] = []
    total_started = time.monotonic()
    cpu_started = process_cpu_seconds()
    for index in range(args.runs):
        started = time.monotonic()
        response = client.predict(request)
        validated = validate_policy_response(response, args.seed)
        elapsed = time.monotonic() - started
        proposal = build_action_proposal(
            request_id=f"vlapi05g1-gate-c-{index + 1:03d}",
            created_at=time.time(),
            snapshot=snapshot,
            task=args.task,
            seed=args.seed,
            validated_response=validated,
            card_elapsed_seconds=elapsed,
        )
        if proposal["execution_authorized"] is not False:
            raise ValueError("Gate C proposal unexpectedly authorizes execution")
        json.dumps(proposal)
        chunk = validated["action_chunk"]
        chunks.append(chunk)
        runs.append(
            {
                "run": index + 1,
                "action_sha256": action_hash(chunk),
                "end_to_end_seconds": elapsed,
                "policy_infer_seconds": validated["policy_infer_seconds"],
                "cuda_max_memory_allocated": response.get("cuda_max_memory_allocated"),
            }
        )
    total_elapsed = time.monotonic() - total_started
    cpu_elapsed = process_cpu_seconds() - cpu_started

    differences = [max_abs_difference(chunks[0], chunk) for chunk in chunks[1:]]
    max_difference = max(differences, default=0.0)
    action_hashes = [run["action_sha256"] for run in runs]
    if max_difference != 0.0 or len(set(action_hashes)) != 1:
        raise ValueError(
            f"determinism gate failed: max_abs_difference={max_difference}, hashes={action_hashes}"
        )

    end_to_end = [run["end_to_end_seconds"] for run in runs]
    policy_times = [run["policy_infer_seconds"] for run in runs]
    cuda_peaks = [
        run["cuda_max_memory_allocated"]
        for run in runs
        if isinstance(run["cuda_max_memory_allocated"], int)
    ]
    report = {
        "status": "PASS",
        "gate": "C",
        "recorded_at": time.time(),
        "model": "xiaopeng-wu/pi05_unitree_g1",
        "revision": args.revision,
        "policy_url": args.policy_url,
        "health_url": health_url,
        "health": health,
        "source": {
            "image_path": str(image_path),
            "image_format": "JPEG 1920x1080 RGB",
            "image_bytes": len(image_bytes),
            "image_sha256": sha256_file(image_path),
            "state_path": str(state_path),
            "state_format": "JSON array of 29 finite joint positions",
            "state_sha256": sha256_file(state_path),
            "task": args.task,
            "seed": args.seed,
        },
        "expected": {
            "action_shape": ACTION_SHAPE,
            "action_space": ACTION_SPACE,
            "fresh_inference_per_request": True,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "execution_authorized": False,
            "determinism_tolerance": 0.0,
        },
        "runs": runs,
        "determinism": {
            "action_sha256": action_hashes[0],
            "max_abs_difference": max_difference,
            "strict_equal": True,
        },
        "performance": {
            "end_to_end_p50_seconds": statistics.median(end_to_end),
            "end_to_end_p95_seconds": percentile(end_to_end, 95.0),
            "policy_p50_seconds": statistics.median(policy_times),
            "policy_p95_seconds": percentile(policy_times, 95.0),
            "throughput_requests_per_second": args.runs / total_elapsed,
            "client_cpu_seconds": cpu_elapsed,
            "client_cpu_percent_one_core": cpu_elapsed / total_elapsed * 100.0,
            "client_max_rss_bytes": process_max_rss_bytes(),
            "policy_cuda_peak_bytes": max(cuda_peaks) if cuda_peaks else None,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
