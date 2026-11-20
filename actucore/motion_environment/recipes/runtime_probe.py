"""Tiny real GPU computation; no learning, replay buffer or physical robot."""
import argparse
import json
import os
from pathlib import Path
import re

# Publish ownership before slow torch imports so cancellation can reap this exact
# exec process even when the Docker CLI has already disconnected.
parser = argparse.ArgumentParser()
parser.add_argument("--job-id", required=True)
args = parser.parse_args()
if not re.fullmatch(r"[a-f0-9]{32}", args.job_id):
    raise ValueError("invalid probe job id")
try:
    os.setsid()
except PermissionError:
    pass
pid_path = Path("/tmp/phanthymotus-environment-probe-" + args.job_id + ".pid")
fd = os.open(pid_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, "w") as stream:
    stream.write(str(os.getpid()))

import torch
import warp as wp


@wp.kernel
def increment(values: wp.array(dtype=wp.float32)):
    index = wp.tid()
    values[index] = values[index] + 1.0


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA compute unavailable")
    value = torch.arange(16, device="cuda", dtype=torch.float32)
    total = float((value * value).sum().cpu())
    if total != 1240.0:
        raise RuntimeError("CUDA arithmetic check failed")
    wp.init()
    data = wp.zeros(16, dtype=wp.float32, device="cuda:0")
    wp.launch(increment, dim=16, inputs=[data], device="cuda:0")
    wp.synchronize()
    if not (data.numpy() == 1).all():
        raise RuntimeError("Warp kernel check failed")
    print(json.dumps({"cuda_compute": True, "warp_compute": True,
                      "torch": torch.__version__, "warp": wp.__version__,
                      "gpu": torch.cuda.get_device_name(0)}))
