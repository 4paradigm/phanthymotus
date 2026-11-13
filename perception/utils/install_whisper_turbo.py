#!/usr/bin/env python3
"""Fetch the fixed CUDA Whisper runtime and Q5 model on first use."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess

REVISION = "e29905dbd76212ba987e8a486fbd46c475591af3"
MANIFEST_SHA256 = "8049a2aaf4e93d69ea0192f1767457170740a1ea3befada2d3fdde3fc764a980"
MANIFEST_SIZE = 7790
BASE_URL = ("https://www.modelscope.cn/models/"
            "Flame4pd/asr-whisper-large-v3-turbo-q5-v1/resolve/" + REVISION)
MODEL_DIR = "/models/whisper-turbo/" + REVISION
API_URL = ("https://modelscope.cn/api/v1/models/"
           "Flame4pd/asr-whisper-large-v3-turbo-q5-v1/repo?Revision="
           + REVISION + "&FilePath={file}")


def _sources():
    # Match other bundles: an operator's verified mirror replaces the first
    # source, while the fixed public endpoint remains available for fallback.
    return [os.environ.get("WHISPER_TURBO_MODEL_BASE_URL") or BASE_URL, API_URL]


def ensure_release(model_dir, jetpack, on_status=None):
    """Install one JetPack runtime; serialize shared-cache first use."""
    if jetpack not in ("511", "61"):
        raise RuntimeError("Whisper Turbo requires JetPack 5.1.1 or 6.1; "
                           "WHISPER_TURBO_JP_VERSION must be 511 or 61")
    from utils.model_downloader import ensure_verified_bundle
    from utils.model_progress import fetch_status

    sources = _sources()
    progress, _ = fetch_status(on_status, "Whisper Turbo")
    root = Path(model_dir)
    root.mkdir(parents=True, exist_ok=True)
    runtime = root / "runtime" / ("jp" + jetpack)
    with (root / ".install.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest_path = root / "release.json"
        ensure_verified_bundle("whisper-turbo-manifest", str(root), sources,
                               {"release.json": {"size": MANIFEST_SIZE,
                                                 "sha256": MANIFEST_SHA256}})
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
        payload = {}
        for name in wanted:
            item = files[name]
            if name.startswith(prefix):
                target = runtime / name[len(prefix):]
            elif name.startswith("model/"):
                target = root / "model.bin"
            else:
                target = root / name
            payload[target.relative_to(root).as_posix()] = {
                "size": item["size"], "sha256": item["sha256"], "source_path": name}
        ensure_verified_bundle("whisper-turbo", str(root), sources, payload, progress_cb=progress)
        for name in wanted:
            if name.startswith(prefix + "bin/"):
                (runtime / name[len(prefix):]).chmod(0o755)
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
                progress, stage = fetch_status(on_status, "Whisper Turbo CUDA runtime")
                ensure_verified_bundle(
                    "whisper-turbo-cuda", str(runtime), package["url"].rsplit("/", 1)[0],
                    {package["filename"]: {"size": package["size"],
                                            "sha256": package["sha256"]}}, progress_cb=progress)
                if stage:
                    stage("extract")
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
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
