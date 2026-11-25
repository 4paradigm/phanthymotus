"""Two real NCCL ranks, 1 MiB each; no simulator, optimizer, or training."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time


def rank_check(rank):
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=45))
    try:
        value = torch.full((262144,), float(rank + 1), device=f"cuda:{rank}")
        dist.all_reduce(value)
        torch.cuda.synchronize()
        if not bool(torch.all(value == 3).item()):
            raise RuntimeError("NCCL all-reduce returned incorrect values")
        print(json.dumps({"rank": rank, "device": rank, "elements": value.numel(),
                          "sum": 3, "nccl": True}), flush=True)
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--rank", type=int, choices=(0, 1))
    args = parser.parse_args()
    if not re.fullmatch(r"[a-f0-9]{32}", args.job_id):
        raise ValueError("invalid probe job id")
    if args.rank is not None:
        rank_check(args.rank)
        return
    # Children intentionally inherit this process group. The host cleanup
    # verifies the job marker before terminating this group on cancel/timeout.
    try:
        os.setsid()
    except PermissionError:
        if os.getpgrp() != os.getpid():
            raise
    marker = Path("/tmp/phanthymotus-environment-probe-" + args.job_id + ".pid")
    fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(str(os.getpid()))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    children = []
    began = time.monotonic()
    try:
        for rank in range(2):
            env = dict(os.environ, MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
                       WORLD_SIZE="2", RANK=str(rank), LOCAL_RANK=str(rank))
            children.append(subprocess.Popen([sys.executable, __file__, "--job-id", args.job_id,
                                               "--rank", str(rank)], env=env))
        deadline = began + 100
        while any(child.poll() is None for child in children):
            if any(child.returncode not in (None, 0) for child in children):
                raise RuntimeError("NCCL rank failed; see its real output")
            if time.monotonic() > deadline:
                raise TimeoutError("NCCL verification exceeded 100 seconds")
            time.sleep(.1)
        if any(child.returncode for child in children):
            raise RuntimeError("NCCL rank failed; see its real output")
        print(json.dumps({"nccl_all_reduce": True, "world_size": 2, "bytes_per_rank": 1048576,
                          "elapsed_s": round(time.monotonic() - began, 3),
                          "training_started": False}), flush=True)
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)


if __name__ == "__main__":
    main()
