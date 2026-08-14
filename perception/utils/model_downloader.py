"""
utils/model_downloader.py — Auto-download sherpa-onnx models from COS if missing.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tarfile
import tempfile
import time
import zipfile
from urllib.error import URLError
from urllib.request import urlopen, urlretrieve

log = logging.getLogger(__name__)

COS_BASE = "https://agi-phanthy-dev-1252788780.cos.ap-beijing.myqcloud.com/public"
# Artifact: sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.
# Source: https://github.com/Gilgamesh-J/X-ASR (Apache-2.0).
# The internal mirror below is pinned by byte size and SHA256 for image builds.
X_ASR_BASE = os.environ.get(
    "ASR_X_ASR_MODEL_BASE_URL",
    "http://172.28.4.81:34567/zengzhitao/embodied-ai/x_asr_punct_int8_robot343_20260814",
)

X_ASR_FILES = {
    "encoder-epoch-99-avg-1.int8.onnx": {
        "size": 161015713,
        "sha256": "7f6aa62056efd8af9da13e0faa81cd3f284d2fb2e3b63de56fd2dfd3450910dc",
    },
    "decoder-epoch-99-avg-1.onnx": {
        "size": 11309084,
        "sha256": "72f47405d3c1033bebccbef82f90071e7b4ba3e71b9c986f2b74244b25723aed",
    },
    "joiner-epoch-99-avg-1.int8.onnx": {
        "size": 2581422,
        "sha256": "aedb7fa697b2ab43f20499826fff7c997eea7d67db77be97769aeeeb726e63b3",
    },
    "tokens.txt": {
        "size": 58806,
        "sha256": "b818a60878b9aae978cbb8ad594acbd403d76d1af2e31ef4197c84e2dbdba27c",
    },
    "bpe.model": {
        "size": 119265,
        "sha256": "f87a38025a5fdd1e4e9591f6a44bb81295097ce0b80df6f4ab9f44e52c64ca5f",
    },
    "bpe.vocab": {
        "size": 69594,
        "sha256": "28fc94d67aae53d8c58010fcfb16fc8c2f8dd263e03f3490c98210e354e8f914",
    },
    "hotwords.txt": {
        "size": 3496,
        "sha256": "202f2558ca13e19578d69c9c4b1e9402974efc9b0e35774f0ca24372cf063708",
    },
}


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
    "asr_x_asr": {
        "base_url": X_ASR_BASE,
        "files": X_ASR_FILES,
        "check_file": "tokens.txt",
    },
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
}


def ensure_model(name: str, model_dir: str) -> None:
    """Ensure model files exist in model_dir. Download from COS if missing."""
    info = MODELS.get(name)
    if not info:
        raise ValueError(f"Unknown model name: {name}")

    files = info.get("files")
    if files:
        if _bundle_exists(model_dir, files):
            log.info(f"[model_downloader] {name}: already exists at {model_dir}")
            return
        _download_bundle(name, info["base_url"], model_dir, files)
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


def _bundle_exists(model_dir: str, files: dict) -> bool:
    """Accept complete non-empty local bundles, including user-supplied weights."""
    return all(
        os.path.isfile(os.path.join(model_dir, filename))
        and os.path.getsize(os.path.join(model_dir, filename)) > 0
        for filename in files
    )


def _verify_download(path: str, metadata: dict) -> None:
    actual_size = os.path.getsize(path)
    if actual_size != metadata["size"]:
        raise ValueError(
            f"size mismatch for {os.path.basename(path)}: "
            f"expected {metadata['size']}, got {actual_size}"
        )

    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != metadata["sha256"]:
        raise ValueError(
            f"SHA256 mismatch for {os.path.basename(path)}: "
            f"expected {metadata['sha256']}, got {actual_sha256}"
        )


def _download_bundle(name: str, base_url: str, model_dir: str, files: dict) -> None:
    """Download and verify a multi-file model before replacing its destination."""
    os.makedirs(model_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=model_dir) as staging:
        for filename, metadata in files.items():
            if os.path.basename(filename) != filename:
                raise ValueError(f"Invalid model filename: {filename}")
            url = f"{base_url.rstrip('/')}/{filename}"
            destination = os.path.join(staging, filename)
            last_error = None
            for attempt in range(1, 4):
                try:
                    log.info(
                        f"[model_downloader] {name}: downloading {filename} "
                        f"(attempt {attempt}/3)"
                    )
                    with urlopen(url, timeout=120) as response, open(destination, "wb") as output:
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            output.write(chunk)
                        output.flush()
                        os.fsync(output.fileno())
                    _verify_download(destination, metadata)
                    break
                except (URLError, TimeoutError, OSError, ValueError) as error:
                    last_error = error
                    if os.path.exists(destination):
                        os.unlink(destination)
                    if attempt < 3:
                        time.sleep(3)
            else:
                raise RuntimeError(
                    f"[model_downloader] {name}: failed to download {filename}"
                ) from last_error

        for filename in files:
            os.replace(
                os.path.join(staging, filename),
                os.path.join(model_dir, filename),
            )
    log.info(f"[model_downloader] {name}: verified bundle ready at {model_dir}")


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
