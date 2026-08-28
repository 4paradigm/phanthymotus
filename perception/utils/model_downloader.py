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
from urllib.parse import quote
from urllib.request import urlopen, urlretrieve

try:  # Linux only; the perception images are Linux, dev hosts may not be.
    import fcntl
except ImportError:  # pragma: no cover - Windows/macOS dev hosts
    fcntl = None

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
}


def ensure_model(name: str, model_dir: str) -> None:
    """Ensure model files exist in model_dir. Download from COS if missing.

    Serialized per (model_dir, name) with a file lock, and every download lands
    through a temporary file in the destination directory: the check_file must
    not exist until the model behind it is complete. Writing the final name
    directly meant a second caller saw check_file the moment the transfer
    started and loaded a partial model — observed as
    "Load model from .../vocos-16khz-univ.onnx failed: Protobuf parsing failed"
    while the log still showed that file at 30%.
    """
    info = MODELS.get(name)
    if not info:
        raise ValueError(f"Unknown model name: {name}")

    check_path = os.path.join(model_dir, info["check_file"])
    if os.path.exists(check_path):
        log.info(f"[model_downloader] {name}: already exists at {model_dir}")
        return

    os.makedirs(model_dir, exist_ok=True)
    lock_path = os.path.join(model_dir, f".{name}.lock")
    with open(lock_path, "a+b") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            if os.path.exists(check_path):
                log.info(f"[model_downloader] {name}: fetched by another instance")
                return
            _download_model(name, info, model_dir, check_path)
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _download_model(name: str, info: dict, model_dir: str, check_path: str) -> None:
    """Fetch one legacy model into model_dir. Caller holds the per-model lock."""
    url = info["url"]
    log.info(f"[model_downloader] {name}: downloading from {url} ...")

    if info.get("single_file"):
        # Direct file download (not an archive). Staged in the destination
        # directory so the rename is atomic (same filesystem).
        with tempfile.NamedTemporaryFile(dir=model_dir, suffix=".part",
                                         delete=False) as tmp:
            tmp_path = tmp.name
        try:
            urlretrieve(url, tmp_path, reporthook=_progress_hook(name))
            os.chmod(tmp_path, 0o644)
            os.replace(tmp_path, check_path)
            log.info(f"[model_downloader] {name}: done.")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
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

        # Extract beside the destination, then move the files in, so a partly
        # extracted archive never publishes check_file either.
        with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=model_dir) as staging:
            if suffix == ".zip":
                _extract_zip(tmp_path, staging)
            else:
                _extract_tar(tmp_path, staging)
            if not os.path.exists(os.path.join(staging, info["check_file"])):
                raise RuntimeError(
                    f"[model_downloader] {name}: download completed but "
                    f"{info['check_file']} not found in the archive"
                )
            _merge_tree(staging, model_dir)

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


# ── Verified bundles (OCR / obstacle TensorRT artefacts) ─────────────────────
# Pure additions consumed by the vision plugins' thin wrappers. The legacy
# ensure_model() above (sherpa-onnx archives, X-ASR) is intentionally left
# untouched. Every file in a verified bundle carries a pinned size and SHA256:
# existing files are re-verified before reuse, downloads are staged next to
# the destination, verified, and only then moved into place. Concurrent
# instances sharing /models serialize on a per-bundle file lock. Entries that
# ship one bundle per JetPack family use {"jp511": {...}, "jp61": {...}} keys
# selected by the TensorRT that is actually importable
# (see utils.tensorrt_runtime).


MODELS_ROOT = "/models"


def require_models_subpath(path: str, root: str = MODELS_ROOT) -> str:
    """Validate that a caller-supplied model_dir stays inside the models tree.

    model_dir is accepted over MCP config and the downloader runs as root in
    the container, so an unchecked value would let a caller create or
    overwrite files at arbitrary container paths.

    A lexical check is not enough: ``/models/link`` passes it while ``link``
    is a symlink pointing outside the tree, and every later makedirs/open/
    os.replace would follow it. Resolve symlinks on both sides — for the
    deepest component that exists, since the target directory is usually
    created later — and compare the resolved paths. Returns the resolved
    absolute path, which callers must use for all filesystem work.
    """
    candidate = os.path.normpath(os.path.join("/", str(path)))
    root_real = os.path.realpath(root)

    # Resolve the longest existing prefix, then re-attach the missing tail:
    # realpath() on a not-yet-created directory cannot detect a symlinked
    # parent otherwise.
    existing = candidate
    tail: list[str] = []
    while not os.path.exists(existing) and existing not in ("/", ""):
        existing, name = os.path.split(existing)
        tail.append(name)
    resolved = os.path.join(os.path.realpath(existing), *reversed(tail))
    resolved = os.path.normpath(resolved)

    if resolved != root_real and not resolved.startswith(root_real + os.sep):
        raise ValueError(
            f"model_dir must resolve under {root_real}/: got {path!r}"
        )
    return resolved


def select_bundle_family(bundles: dict, family: str | None = None) -> str:
    """Pick the bundle key ("jp511"/"jp61") for the runtime TensorRT.

    An explicit ``family`` (or alias such as "61"/"511") wins; otherwise the
    family is derived from the importable TensorRT major version. Never
    depends on a Docker build argument or image ENV.
    """
    from utils.tensorrt_runtime import normalize_family, tensorrt_family

    if family is not None:
        key = normalize_family(family)
        if key is None:
            raise ValueError(f"Unknown model bundle family: {family!r}")
    else:
        key = tensorrt_family()
    if key not in bundles:
        raise RuntimeError(
            f"No model bundle for TensorRT family {key}; available: {sorted(bundles)}"
        )
    return key


def ensure_verified_bundle(
    name: str, model_dir: str, base_url: str, files: dict
) -> dict[str, str]:
    """Ensure a size/SHA256-pinned bundle is present and valid in model_dir.

    existing files → size check → SHA256 check → reuse
    otherwise      → lock → re-check → download (retry) → verify → replace
    Returns ``{filename: absolute path}``.
    """
    paths = {
        filename: os.path.join(model_dir, filename) for filename in files
    }
    if _bundle_matches(model_dir, files):
        log.info(f"[model_downloader] {name}: verified bundle already at {model_dir}")
        return paths

    os.makedirs(model_dir, exist_ok=True)
    # Platform instances share /models. Serialize the download so a cold
    # multi-instance launch fetches one copy instead of one per process; a
    # waiter re-checks the bundle once it gets the lock.
    lock_path = os.path.join(model_dir, f".{name.replace('/', '_')}.lock")
    with open(lock_path, "a+b") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            if _bundle_matches(model_dir, files):
                log.info(f"[model_downloader] {name}: verified by another instance")
                return paths
            _download_verified_bundle(name, base_url, model_dir, files)
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    return paths


def _file_matches(path: str, metadata: dict) -> bool:
    """Return whether one file exists and matches its pinned size and SHA256."""
    try:
        if not os.path.isfile(path):
            return False
        _verify_pinned_file(path, metadata)
    except (OSError, ValueError):
        return False
    return True


def _bundle_matches(model_dir: str, files: dict) -> bool:
    """Return whether every bundle file matches its pinned size and SHA256."""
    return all(
        _file_matches(os.path.join(model_dir, filename), metadata)
        for filename, metadata in files.items()
    )


def _check_bundle_relpath(filename: str) -> None:
    """Reject a bundle key that would escape model_dir or break the URL join.

    Keys are relative paths, not bare filenames: the VITS2 release ships
    ``engines/jp61/flow.plan`` and ``nltk_data/taggers/...`` and its consumers
    expect that layout on disk. A key is only allowed to descend — no absolute
    path, no ``..``, no empty or ``.`` segment, no backslash (which is a plain
    character in a POSIX name but a separator once it reaches a URL).
    """
    if not filename or filename != filename.strip():
        raise ValueError(f"Invalid model filename: {filename!r}")
    if filename.startswith("/") or "\\" in filename:
        raise ValueError(f"Invalid model filename: {filename!r}")
    parts = filename.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"Invalid model filename: {filename!r}")


def _verify_pinned_file(path: str, metadata: dict) -> None:
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


def _fetch_pinned_file(
    name: str, url: str, destination: str, metadata: dict, label: str = ""
) -> None:
    """Download one URL to destination, verifying its pinned size and SHA256.

    Retries three times with a short backoff, leaving no partial file behind:
    a truncated download fails _verify_pinned_file, which is caught here, so a
    flaky link costs a retry rather than a corrupt model.
    """
    label = label or os.path.basename(destination)
    last_error = None
    for attempt in range(1, 4):
        try:
            log.info(
                f"[model_downloader] {name}: downloading {label} "
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
            _verify_pinned_file(destination, metadata)
            os.chmod(destination, 0o644)
            return
        except (URLError, TimeoutError, OSError, ValueError) as error:
            last_error = error
            if os.path.exists(destination):
                os.unlink(destination)
            if attempt < 3:
                time.sleep(3)
    raise RuntimeError(
        f"[model_downloader] {name}: failed to download {label}"
    ) from last_error


def _download_verified_bundle(
    name: str, base_url: str, model_dir: str, files: dict
) -> None:
    """Download and verify a multi-file model before replacing its destination."""
    os.makedirs(model_dir, exist_ok=True)
    staging_prefix = f".{name.replace('/', '_')}-"
    with tempfile.TemporaryDirectory(prefix=staging_prefix, dir=model_dir) as staging:
        for filename, metadata in files.items():
            _check_bundle_relpath(filename)
            url = "/".join(
                [base_url.rstrip("/")] + [quote(part) for part in filename.split("/")]
            )
            destination = os.path.join(staging, filename)
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            _fetch_pinned_file(name, url, destination, metadata, label=filename)

        for filename in files:
            final = os.path.join(model_dir, filename)
            os.makedirs(os.path.dirname(final), exist_ok=True)
            os.replace(os.path.join(staging, filename), final)
    log.info(f"[model_downloader] {name}: verified bundle ready at {model_dir}")



# ── sherpa-onnx GPU weight variants (device: gpu) ──────────────────────────
# The bundles in MODELS above are all int8, which is the right choice for the CPU
# and the wrong one for the GPU: ONNX Runtime's CUDA provider has no int8 kernels,
# falls back to CPU node by node, and measured 1.25x-3.3x *slower* than the CPU on
# the same audio. These are the non-quantised variants that `device: gpu` loads —
# which weights belong to which model is declared in plugins/asr.py ASR_MODELS.
#
# These use ensure_verified_bundle rather than ensure_model because they are the
# largest downloads in the stack (the paraformer encoder alone is 636 MB) and
# ensure_model's only integrity check is "does check_file exist in the archive".
# A truncated 780 MB transfer passes that and then fails at session creation in a
# way nobody can diagnose. Here every file is pinned by size and SHA256.
#
# Provenance: derived from pengzhendong's ModelScope mirrors of the k2-fsa model
# zoo, accepted only after that mirror's int8 weights were confirmed byte-identical
# to the copies we already deploy from COS. The fp16 files are converted from the
# mirror's fp32 with tools/convert_onnx_fp16.py.
SHERPA_GPU_MODEL_BASE = os.environ.get(
    "SHERPA_GPU_MODEL_BASE_URL", f"{COS_BASE}/sherpa-onnx-gpu"
)
SHERPA_GPU_BUNDLES = {
    # Streaming paraformer, fp32. fp16 exists but is NOT used here: on CUDA it
    # emits nothing but </s> (correct on CPU, so the conversion is fine and the
    # CUDA+fp16+streaming combination is not), and it is slower than fp32 anyway
    # (2077 ms vs 1859 ms).
    "asr_gpu": {
        "base_url": f"{SHERPA_GPU_MODEL_BASE}/streaming-paraformer-bilingual-zh-en-fp32",
        "files": {
            "encoder.onnx": {
                "size": 636348877,
                "sha256": "832c8e8d3f758f4ab0fcfc011eec91154ecd129b7305564a7b461b20064ebcc6",
            },
            "decoder.onnx": {
                "size": 228464044,
                "sha256": "e178f5a7dd4efbf5905a797807006d773b12116eb39fed3d16758e68f9f50921",
            },
            "tokens.txt": {
                "size": 75756,
                "sha256": "59aba8873a2ed1e122c25fee421e25f283b63290efbde85c1f01a853d83cb6e6",
            },
        },
    },
    # Offline SenseVoice, fp16 — faster than fp32 on CUDA (344 ms vs 416 ms), half
    # the size, and transcript-identical to fp32 on both providers.
    "asr_sensevoice_gpu": {
        "base_url": f"{SHERPA_GPU_MODEL_BASE}/sense-voice-zh-en-ja-ko-yue-2024-07-17-fp16",
        "files": {
            "model.fp16.onnx": {
                "size": 470225401,
                "sha256": "b6b71a306afa7ccb48d2319b91567dfeefeb51f0f4eed9c88ec139cb10c14e09",
            },
            "tokens.txt": {
                "size": 315894,
                "sha256": "f449eb28dc567533d7fa59be34e2abca8784f771850c78a47fb731a31429a1dc",
            },
        },
    },
}


def ensure_gpu_model(name: str, model_dir: str) -> dict[str, str]:
    """Ensure a `device: gpu` weight bundle is present and SHA256-verified."""
    bundle = SHERPA_GPU_BUNDLES.get(name)
    if bundle is None:
        raise KeyError(
            f"No GPU weight bundle named {name!r}; "
            f"available: {sorted(SHERPA_GPU_BUNDLES)}"
        )
    return ensure_verified_bundle(name, model_dir, bundle["base_url"],
                                  bundle["files"])


# ── OCR (PP-OCRv6 small, TensorRT engines; one bundle per JetPack family) ──
# The engines are built per TensorRT major and are not portable, so the
# bundle is chosen from the TensorRT that is importable at runtime. Only the
# base URL is provenance-specific: switching the distribution host (e.g. to
# COS) means changing OCR_MODEL_BASE only.
OCR_MODEL_BASE = os.environ.get(
    "OCR_MODEL_BASE_URL",
    "https://www.modelscope.cn/models/Flame4pd/"
    "ppocrv6-small-edge-ocr/resolve/"
    "0301e9299b3abe09c6a60796d7bed74c23fcc525",
)
_OCR_KEYS = {
    "size": 74947,
    "sha256": "b5f2bfe2bdd9448429e3e82b51c789775d9b42f2403d082b00662eb77e401c5d",
}
OCR_MODEL_BUNDLES = {
    "jp61": {
        "base_url": f"{OCR_MODEL_BASE}/tensorrt-jp6-trt10.4-orin-batch8-cls8",
        "files": {
            "det.engine": {
                "size": 11194324,
                "sha256": "3b36aae43b2cc4a1b1e2d74d846a1319b4b6f42fbc6d97747d8d72e12c74a1ef",
            },
            "rec.engine": {
                "size": 23303292,
                "sha256": "8149fa68d5418f2c0763b8c4e5088987cb679a407317c7510f88ab6de38dd641",
            },
            "cls.engine": {
                "size": 1046484,
                "sha256": "148a6895260d3b6b6f86e0c5787121fc1bba316f3427397f654421196c13cb77",
            },
            "keys.txt": _OCR_KEYS,
        },
    },
    "jp511": {
        "base_url": f"{OCR_MODEL_BASE}/tensorrt-jp511-trt8.5-orin-batch8-cls8",
        "files": {
            "det.engine": {
                "size": 12334256,
                "sha256": "1bb32a027e93b06d5319ac61e38bb3e447137b01465eacefa7a652f58130ebdf",
            },
            "rec.engine": {
                "size": 19915466,
                "sha256": "1e204f0469beba33d8590b29c06419cf1073d98d41243b5ee316d2f877340b61",
            },
            "cls.engine": {
                "size": 1038858,
                "sha256": "02c722e56e621b56a36678cc8aa124a31b41e9e3c9ca350b11e4de0d5bbd0a35",
            },
            "keys.txt": _OCR_KEYS,
        },
    },
}


# ── Obstacle distance (ZipDepth + YOLO26n TensorRT engines) ─────────────
OBSTACLE_MODEL_REVISION = "8c50e1f5faf6b588b90e0bab4b6a696b5ef207ac"
OBSTACLE_MODEL_BASE = os.environ.get(
    "OBSTACLE_MODEL_BASE_URL",
    "https://www.modelscope.cn/models/Flame4pd/"
    f"obstacle-distance-jetson-int8/resolve/{OBSTACLE_MODEL_REVISION}",
)
OBSTACLE_MODEL_BUNDLES = {
    "jp61": {
        "base_url": f"{OBSTACLE_MODEL_BASE}/jp61",
        "files": {
            "zipdepth-base-npu-512x384-int8.engine": {
                "size": 7935428,
                "sha256": "aa34296bcaeed28a5176b423f074da3923c996e7be06702a3952d475000a8887",
            },
            "yolo26n-depth-int8.engine": {
                "size": 7778158,
                "sha256": "8174652d6ba72af15c10caccf95629d585d33245e5242aa1f1734317d5a23f7c",
            },
            "yolo26n-seg-int8.engine": {
                "size": 5641961,
                "sha256": "7cb85598bc50b82ab5835102dab9214f6e58a0061c6a1891ee018387346bae30",
            },
        },
    },
    "jp511": {
        "base_url": f"{OBSTACLE_MODEL_BASE}/jp511",
        "files": {
            "zipdepth-base-npu-512x384-int8.engine": {
                "size": 14259938,
                "sha256": "2b3f81cf5162136e25b0d26e2a8bb453d13158d2884c2eacbf17e0105682f684",
            },
            "yolo26n-depth-int8.engine": {
                "size": 6746230,
                "sha256": "816ca14c23af37ee2961ec09db51a54888462c5e4bb296bbaba2a569e6f2bb64",
            },
            "yolo26n-seg-int8.engine": {
                "size": 4920200,
                "sha256": "ed5e0f8dcb968440866f5b0433f7b813d14f2e89f810be6e8910945e0af42635",
            },
        },
    },
}


def ensure_ocr_model(model_dir: str, family: str | None = None) -> dict[str, str]:
    """Ensure the OCR TensorRT bundle matching the runtime TensorRT is present."""
    model_dir = require_models_subpath(model_dir)
    key = select_bundle_family(OCR_MODEL_BUNDLES, family)
    entry = OCR_MODEL_BUNDLES[key]
    log.info(f"[model_downloader] ocr: using {key} bundle")
    return ensure_verified_bundle(
        f"ocr/{key}", model_dir, entry["base_url"], entry["files"]
    )


def ensure_verified_archive(name: str, model_dir: str, url: str, entry: dict) -> None:
    """Ensure a size/SHA256-pinned archive has been unpacked into model_dir.

    The bundle helper above fetches one URL per file, which is right for a
    handful of engines. A release that also carries its frontend data (VITS2
    ships ~30 files, most of them small NLTK corpora) is cheaper as a single
    compressed download, so this variant pins the archive instead: one size +
    SHA256 covers every member, and 154 MB of engines and FSTs travel as 60 MB.

    A ``.<name>.installed`` marker holding the archive's SHA256 records what is
    unpacked, so a warm start costs one small read instead of re-hashing every
    engine. Delete the marker (or the directory) to force a reinstall.
    """
    flat = name.replace("/", "_")
    marker = os.path.join(model_dir, f".{flat}.installed")
    if _archive_installed(marker, entry["sha256"]):
        log.info(f"[model_downloader] {name}: verified archive already at {model_dir}")
        return

    os.makedirs(model_dir, exist_ok=True)
    # Same rationale as ensure_verified_bundle: instances share /models, so a
    # cold multi-instance launch must fetch one copy, not one per process.
    lock_path = os.path.join(model_dir, f".{flat}.lock")
    with open(lock_path, "a+b") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            if _archive_installed(marker, entry["sha256"]):
                log.info(f"[model_downloader] {name}: installed by another instance")
                return
            with tempfile.TemporaryDirectory(prefix=f".{flat}-", dir=model_dir) as staging:
                archive = os.path.join(staging, os.path.basename(url))
                _fetch_pinned_file(name, url, archive, entry)
                payload = os.path.join(staging, "payload")
                os.makedirs(payload)
                _extract_verified_tar(archive, payload)
                os.unlink(archive)
                _merge_tree(payload, model_dir)
            # Written last: until the marker exists the install is incomplete
            # and the next call redoes it, so a crash mid-extract cannot leave
            # a half-unpacked release looking ready.
            tmp_marker = f"{marker}.tmp"
            with open(tmp_marker, "w") as handle:
                handle.write(entry["sha256"])
            os.replace(tmp_marker, marker)
            log.info(f"[model_downloader] {name}: unpacked verified archive to {model_dir}")
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _archive_installed(marker: str, sha256: str) -> bool:
    """Return whether the marker records this exact archive as unpacked."""
    try:
        with open(marker) as handle:
            return handle.read().strip() == sha256
    except OSError:
        return False


def _extract_verified_tar(archive: str, destination: str) -> None:
    """Extract a tar archive, refusing anything that could escape destination.

    tarfile's ``filter="data"`` would cover this, but it only exists from
    Python 3.12 and the jp511 image is on 3.8 — so the member checks are
    explicit: regular files and directories only, relative paths only, no
    symlink or device entries.
    """
    with tarfile.open(archive, "r:*") as handle:
        members = handle.getmembers()
        if not members:
            raise RuntimeError(f"Empty archive: {archive}")
        for member in members:
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsupported archive entry: {member.name}")
            _check_bundle_relpath(member.name)
        handle.extractall(destination, members=members)


def _merge_tree(source: str, destination: str) -> None:
    """Move every file under source into destination, creating parents."""
    for root, _, files in os.walk(source):
        for filename in files:
            src = os.path.join(root, filename)
            final = os.path.join(destination, os.path.relpath(src, source))
            os.makedirs(os.path.dirname(final), exist_ok=True)
            os.replace(src, final)


# ── VITS2 TTS (ZH/EN VITS2 16 kHz, TensorRT engines; one archive per JetPack) ──
# TensorRT plans are not portable across TensorRT majors, so the archive is
# chosen from the TensorRT that is importable at runtime, never from a build
# argument — same rule as OCR above.
#
# Each archive carries the frontend the engines need, unpacked to the layout
# frontend/release_paths.py expects: engines/<family>/*.plan, config.json,
# frontend_data/, tn_cache/ (compiled WeText TN FSTs) and nltk_data/ (cmudict +
# perceptron tagger — shipped precisely so the container never has to call
# nltk.download() at runtime). Upstream is
# modelscope.cn/models/Starlight777/VITS2-ZH-EN-Male-16k at revision
# 14954122c4baf4e80b44436c4b2b167e38db4103; the runtime-required files of that
# revision were repacked per family and mirrored to COS, so devices pull one
# 60 MB file from the same host as every other model here. The fp32 ONNX graphs
# the plans were built from are not included — they are build inputs, not
# runtime files.
VITS2_MODEL_BASE = os.environ.get("VITS2_MODEL_BASE_URL", COS_BASE)
VITS2_MODEL_ARCHIVES = {
    "jp61": {
        "archive": "vits2-zh-en-male-16k-tensorrt-jp61-trt10.4-orin.tar.gz",
        "size": 61834952,
        "sha256": "f04ab439588cd3106ccd245f64af548199ba888d31627827c07ac28368225805",
    },
    "jp511": {
        "archive": "vits2-zh-en-male-16k-tensorrt-jp511-trt8.5-orin.tar.gz",
        "size": 62994881,
        "sha256": "01ffce0516f1a68f3fcedce6ff9caff784f428a1d09da2d760fcba599116e8c7",
    },
}


def ensure_vits2_model(model_dir: str, family: str | None = None) -> str:
    """Ensure the VITS2 release matching the runtime TensorRT is installed.

    Returns the engine directory for this runtime, which is what the adapter
    hands to TensorRT — the caller never has to work out the family itself.
    """
    model_dir = require_models_subpath(model_dir)
    key = select_bundle_family(VITS2_MODEL_ARCHIVES, family)
    entry = VITS2_MODEL_ARCHIVES[key]
    log.info(f"[model_downloader] vits2: using {key} archive")
    ensure_verified_archive(
        f"vits2/{key}",
        model_dir,
        f"{VITS2_MODEL_BASE.rstrip('/')}/{entry['archive']}",
        entry,
    )
    return os.path.join(model_dir, "engines", key)


def ensure_obstacle_models(
    model_dir: str, bundle: str | None = None
) -> dict[str, str]:
    """Ensure the obstacle TensorRT engines matching the runtime TensorRT.

    ``bundle`` (or the OBSTACLE_MODEL_BUNDLE environment variable) is an
    explicit test override such as "jp61"/"jp511"; production deployments
    leave it unset and follow the importable TensorRT version.
    """
    family = bundle or os.environ.get("OBSTACLE_MODEL_BUNDLE") or None
    model_dir = require_models_subpath(model_dir)
    key = select_bundle_family(OBSTACLE_MODEL_BUNDLES, family)
    entry = OBSTACLE_MODEL_BUNDLES[key]
    log.info(f"[model_downloader] obstacle: using {key} bundle")
    return ensure_verified_bundle(
        f"obstacle/{key}", model_dir, entry["base_url"], entry["files"]
    )
