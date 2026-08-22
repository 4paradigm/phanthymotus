"""
utils/model_downloader.py — Auto-download sherpa-onnx models from COS if missing.
"""

from __future__ import annotations

import logging
import os
import fcntl
import hashlib
import json
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlretrieve
from urllib.request import urlopen

log = logging.getLogger(__name__)

COS_BASE = "https://agi-phanthy-dev-1252788780.cos.ap-beijing.myqcloud.com/public"


def _progress_hook(name: str):
    """Create a reporthook for urlretrieve that logs download progress."""
    last_pct = [0]
    def hook(block_num, block_size, total_size):
        if total_size > 0:
            pct = min(int(block_num * block_size * 100 / total_size), 100)
            if pct >= last_pct[0] + 10:
                last_pct[0] = pct
                mb_done = block_num * block_size / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                log.info(f"[model_downloader] {name}: {pct}% ({mb_done:.1f}/{mb_total:.1f} MB)")
    return hook

MODELS = {
    "asr": {
        "url": f"{COS_BASE}/sherpa-onnx-streaming-paraformer-bilingual-zh-en.zip",
        "check_file": "tokens.txt",
    },
    "asr_en": {
        "url": f"{COS_BASE}/sherpa-onnx-streaming-zipformer-en-2023-06-26.zip",
        "check_file": "tokens.txt",
    },
    "asr_sensevoice": {
        "url": f"{COS_BASE}/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.zip",
        "check_file": "tokens.txt",
    },
    "asr_paraformer_offline": {
        "url": f"{COS_BASE}/sherpa-onnx-paraformer-zh-small-2024-03-09.tar.bz2",
        "check_file": "tokens.txt",
    },
    "asr_x_asr": {
        "url": f"{COS_BASE}/x-asr-zh-en-punct-int8-robot.zip",
        "check_file": "tokens.txt",
    },
    "tts": {
        "url": f"{COS_BASE}/matcha-icefall-zh-en.tar.bz2",
        "check_file": "model-steps-3.onnx",
    },
    "tts_vocoder": {
        "url": f"{COS_BASE}/vocos-16khz-univ.onnx",
        "check_file": "vocos-16khz-univ.onnx",
        "single_file": True,
    },
    "kws": {
        "url": f"{COS_BASE}/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2",
        "check_file": "tokens.txt",
    },
    "kws_zh": {
        "url": f"{COS_BASE}/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.zip",
        "check_file": "tokens.txt",
    },
    "kws_en": {
        "url": f"{COS_BASE}/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.zip",
        "check_file": "tokens.txt",
    },
    "vad": {
        "url": f"{COS_BASE}/silero_vad.onnx",
        "check_file": "silero_vad.onnx",
        "single_file": True,  # Not an archive, just a single file download
    },
    "denoise": {
        "url": f"{COS_BASE}/gtcrn_simple.onnx",
        "check_file": "gtcrn_simple.onnx",
        "single_file": True,
    },
    # Manifest-based release; handled by ensure_vits2_model below.
    "vits2": {
        "manifest": True,
        "check_file": ".release_manifest.json",
    },
}


def ensure_model(name: str, model_dir: str) -> None:
    """Ensure model files exist in model_dir. Download from COS if missing."""
    info = MODELS.get(name)
    if not info:
        raise ValueError(f"Unknown model name: {name}")
    if info.get("manifest"):
        ensure_vits2_model(model_dir)
        return

    check_path = os.path.join(model_dir, info["check_file"])
    if os.path.exists(check_path):
        log.info(f"[model_downloader] {name}: already exists at {model_dir}")
        return

    url = info["url"]
    os.makedirs(model_dir, exist_ok=True)
    log.info(f"[model_downloader] {name}: downloading from {url} ...")

    if info.get("single_file"):
        # Direct file download (not an archive)
        dest = os.path.join(model_dir, info["check_file"])
        urlretrieve(url, dest, reporthook=_progress_hook(name))
        log.info(f"[model_downloader] {name}: done.")
        return

    # Determine suffix from URL
    if url.endswith(".zip"):
        suffix = ".zip"
    else:
        suffix = ".tar.bz2"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name

    try:
        urlretrieve(url, tmp_path, reporthook=_progress_hook(name))
        log.info(f"[model_downloader] {name}: extracting to {model_dir} ...")

        if suffix == ".zip":
            _extract_zip(tmp_path, model_dir)
        else:
            _extract_tar(tmp_path, model_dir)

        log.info(f"[model_downloader] {name}: done.")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # Verify
    if not os.path.exists(check_path):
        raise RuntimeError(
            f"[model_downloader] {name}: download completed but {info['check_file']} "
            f"not found in {model_dir}"
        )


def _extract_zip(zip_path: str, model_dir: str) -> None:
    """Extract zip, stripping common top-level directory prefix."""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Filter out __MACOSX and directory entries
        names = [n for n in zf.namelist()
                 if not n.endswith('/') and not n.startswith('__MACOSX')]
        if not names:
            raise RuntimeError(f"Empty archive: {zip_path}")

        prefix = _common_prefix_from_names(names)
        for name in names:
            stripped = name[len(prefix):] if prefix else name
            if not stripped:
                continue
            dest = os.path.join(model_dir, stripped)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with zf.open(name) as src, open(dest, 'wb') as dst:
                dst.write(src.read())


def _extract_tar(tar_path: str, model_dir: str) -> None:
    """Extract tar.bz2, stripping common top-level directory prefix."""
    with tarfile.open(tar_path, "r:bz2") as tf:
        members = tf.getmembers()
        if not members:
            raise RuntimeError(f"Empty archive: {tar_path}")

        names = [m.name for m in members if not m.isdir()]
        prefix = _common_prefix_from_names(names)
        for m in members:
            if m.isdir():
                continue
            if prefix:
                m.name = m.name[len(prefix):]
            if not m.name:
                continue
            m.name = m.name.lstrip("/")
            tf.extract(m, model_dir)


def _common_prefix_from_names(names: list[str]) -> str:
    """Find common top-level directory prefix from file name list."""
    dirs_with_slash = [n.split("/", 1) for n in names if "/" in n]
    if not dirs_with_slash:
        return ""
    first_parts = set(parts[0] for parts in dirs_with_slash)
    if len(first_parts) == 1:
        return first_parts.pop() + "/"
    return ""


# VITS2 uses a file-level manifest rather than the archive format above.
VITS2_MODEL_ID = "Starlight777/VITS2-ZH-EN-Male-16k"
VITS2_MODEL_REVISION = "14954122c4baf4e80b44436c4b2b167e38db4103"
VITS2_MANIFEST_SHA256 = "2a8537b3abe7faffa81b20120136745e14e6f6d9e1599271f873e4d9192ab0f8"
VITS2_BASE_URL = f"https://www.modelscope.cn/models/{VITS2_MODEL_ID}/resolve"
VITS2_LOCAL_MANIFEST = ".release_manifest.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _vits2_runtime_target() -> tuple[str, int]:
    try:
        import tensorrt
    except ModuleNotFoundError as exc:
        raise RuntimeError("TensorRT is required for the VITS2 TTS runtime") from exc

    version = str(tensorrt.__version__)
    try:
        major = int(version.split(".", 1)[0])
    except ValueError as exc:
        raise RuntimeError(f"Unrecognized TensorRT version: {version}") from exc

    targets = {8: "jp511", 10: "jp61"}
    target = targets.get(major)
    if target is None:
        raise RuntimeError(
            f"No VITS2 runtime target is defined for TensorRT {version}"
        )
    return target, major


def _vits2_runtime_path(remote_path: str, target_name: str) -> str:
    prefix = f"engines/{target_name}/"
    return "engines/" + remote_path[len(prefix):] if remote_path.startswith(prefix) else remote_path


def _manifest_runtime_target(manifest: dict, target_name: str) -> dict | None:
    targets = manifest.get("runtime_targets")
    if isinstance(targets, dict):
        target = targets.get(target_name)
        return target if isinstance(target, dict) else None
    if isinstance(targets, list):
        for target in targets:
            if isinstance(target, dict) and target.get("name") == target_name:
                return target
        return None
    target = manifest.get("runtime_target")
    return target if isinstance(target, dict) else None


def _load_vits2_manifest(path: Path, target_name: str, tensorrt_major: int) -> dict:
    if _sha256_file(path) != VITS2_MANIFEST_SHA256:
        raise RuntimeError("VITS2 release manifest SHA256 mismatch")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("model_id") != VITS2_MODEL_ID:
        raise RuntimeError("Unsupported VITS2 release manifest")
    target = _manifest_runtime_target(manifest, target_name)
    if target is None:
        raise RuntimeError(
            f"The VITS2 release has no runtime target for {target_name}/"
            f"TensorRT {tensorrt_major}"
        )
    if target.get("name") != target_name or target.get("tensorrt_major") != tensorrt_major:
        raise RuntimeError(
            f"The VITS2 release target does not match {target_name}/"
            f"TensorRT {tensorrt_major}"
        )
    return manifest


def _vits2_entry_target(entry: dict) -> str | None:
    target = entry.get("runtime_target")
    if isinstance(target, str):
        return target
    path = entry.get("path", "")
    for target_name in ("jp511", "jp61"):
        if path.startswith(f"engines/{target_name}/"):
            return target_name
    return None


def _vits2_required_files(manifest: dict, target_name: str) -> list[dict]:
    files = [
        entry for entry in manifest.get("files", [])
        if entry.get("runtime_required")
        and _vits2_entry_target(entry) in (None, target_name)
    ]
    if not files:
        raise RuntimeError("VITS2 release manifest has no runtime files")
    return files


def _vits2_complete(model_dir: Path, target_name: str, tensorrt_major: int) -> bool:
    manifest_path = model_dir / VITS2_LOCAL_MANIFEST
    if not manifest_path.is_file():
        return False
    try:
        manifest = _load_vits2_manifest(manifest_path, target_name, tensorrt_major)
        for entry in _vits2_required_files(manifest, target_name):
            path = model_dir / _vits2_runtime_path(entry["path"], target_name)
            if (not path.is_file() or path.stat().st_size != int(entry["bytes"])
                    or _sha256_file(path) != entry["sha256"]):
                return False
    except (KeyError, OSError, TypeError, ValueError, RuntimeError, json.JSONDecodeError):
        return False
    return True


def _vits2_file_url(base_url: str, revision: str, path: str) -> str:
    encoded = "/".join(quote(part, safe="") for part in path.split("/"))
    return f"{base_url.rstrip('/')}/{quote(revision, safe='')}/{encoded}"


def _download_verified(url: str, destination: Path, expected_bytes: int, expected_sha256: str) -> None:
    digest = hashlib.sha256()
    size = 0
    with urlopen(url, timeout=120) as response, destination.open("wb") as output:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            output.write(block)
            digest.update(block)
            size += len(block)
    if (expected_bytes >= 0 and size != expected_bytes) or digest.hexdigest() != expected_sha256:
        raise RuntimeError(f"VITS2 release file verification failed: {destination.name}")


def ensure_vits2_model(model_dir: str) -> None:
    """Install a complete verified VITS2 TensorRT release on first use."""
    target_name, tensorrt_major = _vits2_runtime_target()
    target = Path(model_dir).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.parent / f".{target.name}.download.lock"
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if _vits2_complete(target, target_name, tensorrt_major):
            log.info("[model_downloader] vits2: verified release at %s", target)
            return
        revision = os.getenv("VITS2_MODEL_REVISION", VITS2_MODEL_REVISION).strip()
        if not revision or revision == "REPLACE_WITH_MODELSCOPE_COMMIT":
            raise RuntimeError("VITS2 ModelScope revision is not configured")
        base_url = os.getenv("VITS2_MODEL_BASE_URL", VITS2_BASE_URL).strip()
        if not base_url:
            raise RuntimeError("VITS2 model base URL is empty")

        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
        backup = target.parent / f".{target.name}.previous"
        try:
            manifest_path = staging / VITS2_LOCAL_MANIFEST
            _download_verified(_vits2_file_url(base_url, revision, "release_manifest.json"),
                               manifest_path, -1, VITS2_MANIFEST_SHA256)
            manifest = _load_vits2_manifest(
                manifest_path, target_name, tensorrt_major
            )
            for entry in _vits2_required_files(manifest, target_name):
                destination = staging / _vits2_runtime_path(entry["path"], target_name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                _download_verified(_vits2_file_url(base_url, revision, entry["path"]),
                                   destination, int(entry["bytes"]), entry["sha256"])
            if not _vits2_complete(staging, target_name, tensorrt_major):
                raise RuntimeError("Downloaded VITS2 release is incomplete")
            if backup.exists():
                shutil.rmtree(backup)
            if target.exists():
                target.rename(backup)
            staging.rename(target)
            if backup.exists():
                shutil.rmtree(backup)
            log.info("[model_downloader] vits2: installed verified release at %s", target)
        except Exception:
            if not target.exists() and backup.exists():
                backup.rename(target)
            raise
        finally:
            if staging.exists():
                shutil.rmtree(staging)
