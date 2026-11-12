#!/usr/bin/env python3
"""Fetch the fixed CUDA Whisper runtime and Q5 model on first use."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.parse
import urllib.request

REVISION = "e29905dbd76212ba987e8a486fbd46c475591af3"
MANIFEST_SHA256 = "8049a2aaf4e93d69ea0192f1767457170740a1ea3befada2d3fdde3fc764a980"
BASE_URL = ("https://www.modelscope.cn/models/"
            "Flame4pd/asr-whisper-large-v3-turbo-q5-v1/resolve/" + REVISION)
MODEL_DIR = "/models/whisper-turbo/" + REVISION


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fetch(url, target, sha256, size=None, on_status=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    if (target.is_file() and (size is None or target.stat().st_size == size)
            and _sha256(target) == sha256):
        return
    partial = target.with_name(target.name + ".part")
    for attempt in range(4):
        try:
            if on_status:
                on_status("Downloading Whisper Turbo: " + target.name)
            request = urllib.request.Request(
                url, headers={"User-Agent": "phanthymotus-asr-installer"})
            digest, count = hashlib.sha256(), 0
            with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as out:
                for block in iter(lambda: response.read(1 << 20), b""):
                    out.write(block)
                    digest.update(block)
                    count += len(block)
            if size is not None and count != size:
                raise RuntimeError("Unexpected size: " + target.name)
            if digest.hexdigest() != sha256:
                raise RuntimeError("SHA256 mismatch: " + target.name)
            partial.replace(target)
            return
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


def ensure_release(model_dir, jetpack, on_status=None):
    """Install one JetPack runtime; serialize shared-cache first use."""
    if jetpack not in ("511", "61"):
        raise RuntimeError("Whisper Turbo requires JetPack 5.1.1 or 6.1; "
                           "WHISPER_TURBO_JP_VERSION must be 511 or 61")
    root = Path(model_dir)
    root.mkdir(parents=True, exist_ok=True)
    runtime = root / "runtime" / ("jp" + jetpack)
    with (root / ".install.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest_path = root / "release.json"
        _fetch(BASE_URL + "/release.json", manifest_path, MANIFEST_SHA256,
               on_status=on_status)
        manifest = json.loads(manifest_path.read_text())
        files = {item["path"]: item for item in manifest["files"]}
        prefix = "jp" + jetpack + "/"
        wanted = [name for name in files if name.startswith(prefix)]
        worker_name = prefix + "bin/whisper_turbo_worker"
        if worker_name not in files:
            raise RuntimeError("Whisper release has no worker for JetPack " + jetpack)
        wanted += ["model/ggml-large-v3-turbo-q5_0.bin"]
        wanted += sorted(name for name in files if name.startswith("lora/"))
        wanted += sorted(name for name in files if name.startswith("numeric-itn/"))
        wanted += ["numeric.json"]
        wanted += ["LICENSE.whisper", "LICENSE.whisper.cpp", "MODEL_CARD.json"]
        for name in wanted:
            item = files[name]
            if name.startswith(prefix):
                target = runtime / name[len(prefix):]
            elif name.startswith("model/"):
                target = root / "model.bin"
            else:
                target = root / name
            _fetch(BASE_URL + "/" + urllib.parse.quote(name), target,
                   item["sha256"], item["size"], on_status)
            if name.startswith(prefix + "bin/"):
                target.chmod(0o755)
        # Releases carry canonical libraries; recreate loader aliases locally.
        for real in sorted((runtime / "lib").glob("*.so.*.*")):
            library, version = real.name.split(".so.", 1)
            for alias in (library + ".so", library + ".so." + version.split(".")[0]):
                link = real.with_name(alias)
                if link.is_symlink() and os.readlink(link) == real.name:
                    continue
                link.unlink(missing_ok=True)
                link.symlink_to(real.name)
        # JP5 needs a private CUDA 11.8 runtime for lazy module loading. Never
        # install these packages into the system used by other plugins.
        packages = manifest.get("cuda_runtime_packages", {}).get("jp" + jetpack, [])
        cuda_root = runtime / "cuda-runtime"
        marker = cuda_root / ".release"
        if packages and (not marker.is_file() or marker.read_text() != MANIFEST_SHA256):
            for package in packages:
                archive = runtime / package["filename"]
                _fetch(package["url"], archive, package["sha256"],
                       package["size"], on_status)
                subprocess.run(["dpkg-deb", "-x", str(archive), str(cuda_root)], check=True)
                archive.unlink()
            marker.write_text(MANIFEST_SHA256)
    return runtime / "bin" / "whisper_turbo_worker"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jp", required=True, choices=["511", "61"])
    parser.add_argument("--model-dir", default=MODEL_DIR)
    args = parser.parse_args()
    ensure_release(args.model_dir, args.jp, on_status=lambda value: print(value, flush=True))


if __name__ == "__main__":
    main()
