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
from urllib.request import Request, urlopen, urlretrieve

try:  # Linux only; the perception images are Linux, dev hosts may not be.
    import fcntl
except ImportError:  # pragma: no cover - Windows/macOS dev hosts
    fcntl = None

log = logging.getLogger(__name__)

COS_BASE = "https://agi-phanthy-dev-1252788780.cos.ap-beijing.myqcloud.com/public"


def _notify_stage(name: str, stage_cb, stage: str) -> None:
    """Tell the caller which wait it is now in: "download" or "extract".

    Percentages alone are not enough for an archive. Once the bytes are in, a
    515 MB Kokoro tarball still has to be decompressed and merged, and during
    that the last progress line — "100% (515/515 MB)" — sits frozen on the card
    for tens of seconds, which reads exactly like a hang. Callers that only
    render percentages pass nothing; a failing callback must never abort a
    download that is otherwise fine.
    """
    if stage_cb is None:
        return
    try:
        stage_cb(stage)
    except Exception as error:  # pragma: no cover - defensive
        log.debug(f"[model_downloader] {name}: stage_cb failed: {error}")


def _progress_hook(name: str, progress_cb=None):
    """Create a reporthook for urlretrieve that logs download progress.

    `progress_cb(pct, mb_done, mb_total)` is for callers that surface progress in
    a UI rather than only in the log — the dashboard shows one status string per
    plugin, and "downloading 60%" is a very different thing to wait for than
    "loading". Called on the same 10%-step schedule as the log line, so it costs
    nothing extra; exceptions from it are swallowed because a status update
    failing must never abort a download that is otherwise fine.
    """
    last_pct = [0]
    def hook(block_num, block_size, total_size):
        if total_size > 0:
            pct = min(int(block_num * block_size * 100 / total_size), 100)
            if pct >= last_pct[0] + 10:
                last_pct[0] = pct
                mb_done = block_num * block_size / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                log.info(f"[model_downloader] {name}: {pct}% ({mb_done:.1f}/{mb_total:.1f} MB)")
                if progress_cb is not None:
                    try:
                        progress_cb(pct, mb_done, mb_total)
                    except Exception as error:  # pragma: no cover - defensive
                        log.debug(f"[model_downloader] {name}: progress_cb failed: {error}")
    return hook

MODELS = {
    "asr_sensevoice": {
        "url": f"{COS_BASE}/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.zip",
        "check_file": "tokens.txt",
    },
    "asr_parakeet_en": {
        "url": f"{COS_BASE}/sherpa-onnx-nemo-parakeet_tdt_ctc_110m-en-36000-int8.tar.bz2",
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


def ensure_model(name: str, model_dir: str, progress_cb=None,
                 stage_cb=None) -> None:
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
            _download_model(name, info, model_dir, check_path, progress_cb,
                            stage_cb)
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _download_model(name: str, info: dict, model_dir: str, check_path: str,
                    progress_cb=None, stage_cb=None) -> None:
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
            urlretrieve(url, tmp_path, reporthook=_progress_hook(name, progress_cb))
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
        urlretrieve(url, tmp_path, reporthook=_progress_hook(name, progress_cb))
        log.info(f"[model_downloader] {name}: extracting to {model_dir} ...")
        _notify_stage(name, stage_cb, "extract")

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

        # Drop the leading "./" GNU tar writes for archives built with `tar -c .`
        # BEFORE computing the prefix. Otherwise every member shares a "." first
        # component, _common_prefix_from_names strips just "./", and the archive's
        # real top-level directory survives — so check_file ends up one level
        # below where the caller looks and the download is reported as corrupt.
        # (sherpa-onnx publishes both layouts; the NeMo Parakeet asset is "./".)
        for m in members:
            m.name = _strip_dot_slash(m.name)

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


def _strip_dot_slash(name: str) -> str:
    """Remove leading "./" components from an archive member name."""
    while name.startswith("./"):
        name = name[2:]
    return name


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
    name: str, model_dir: str, base_url, files: dict, progress_cb=None
) -> dict[str, str]:
    """Ensure a size/SHA256-pinned bundle is present and valid in model_dir.

    existing files → size check → SHA256 check → reuse
    otherwise      → lock → re-check → probe sources → download → verify → replace
    Returns ``{filename: absolute path}``.

    `base_url` may be a single base URL, as it always was, or a **list of
    sources** to choose between — see the multi-source note above. A source is
    a base URL or a template containing `{file}`. They are probed once per
    bundle and tried fastest-first, falling through on failure; the pinned
    size and SHA256 are what make that safe.
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
            _download_verified_bundle(name, base_url, model_dir, files,
                                      progress_cb=progress_cb)
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
    name: str, url: str, destination: str, metadata: dict, label: str = "",
    progress_cb=None, done_bytes: int = 0, total_bytes: int = 0,
) -> None:
    """Download one URL to destination, verifying its pinned size and SHA256.

    Retries three times with a short backoff, leaving no partial file behind:
    a truncated download fails _verify_pinned_file, which is caught here, so a
    flaky link costs a retry rather than a corrupt model.

    `progress_cb(pct, mb_done, mb_total)` matches _progress_hook's contract so a
    caller can pass the same callback on either path. `done_bytes`/`total_bytes`
    place this file inside a larger bundle, so a two-file bundle reports one
    monotonic 0-100% instead of restarting at 0 for the second file. The pinned
    size is the denominator — no reliance on Content-Length.
    """
    label = label or os.path.basename(destination)
    total_bytes = total_bytes or int(metadata.get("size") or 0)
    last_error = None
    for attempt in range(1, 4):
        try:
            log.info(
                f"[model_downloader] {name}: downloading {label} "
                f"(attempt {attempt}/3)"
            )
            fetched = 0
            last_pct = 0
            with urlopen(url, timeout=120) as response, open(destination, "wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    fetched += len(chunk)
                    if progress_cb is not None and total_bytes > 0:
                        pct = min(int((done_bytes + fetched) * 100 / total_bytes), 100)
                        # Same 10%-step schedule as the archive path, so this
                        # costs nothing extra and reads the same in the UI.
                        if pct >= last_pct + 10:
                            last_pct = pct
                            try:
                                progress_cb(pct,
                                            (done_bytes + fetched) / (1024 * 1024),
                                            total_bytes / (1024 * 1024))
                            except Exception as error:  # pragma: no cover
                                log.debug(f"[model_downloader] {name}: "
                                          f"progress_cb failed: {error}")
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


# ── multi-source ─────────────────────────────────────────────────────────────
#
# Weights are getting large — a SmolVLA deployment is ~3 GB across two repos —
# and no single host is fastest from everywhere. Measured on the same wheel:
# pypi.jetson-ai-lab served a rig at 12 KB/s while COS served it at 5.7 MB/s;
# ModelScope, for models it mirrors, is another 5.8 MB/s and needs no staging
# step at all.
#
# So a bundle may register several sources and the machine picks. **This is only
# safe because every file is pinned by size and SHA256**: the integrity check
# does not care which host answered, so falling through to a second source costs
# nothing in guarantees. Without the pins, "try another mirror" would mean
# "fetch something unverified from wherever".
#
# A source is a base URL, or a template containing `{file}` for hosts that do
# not serve paths directly — ModelScope's repo API wants
# `...?Revision=master&FilePath=model.safetensors`.
_PROBE_BYTES = 512 * 1024
_PROBE_TIMEOUT = 8
# Below this a source is treated as unusable rather than slow, so an
# unreachable-but-resolving host does not win by returning its error page fast.
_MIN_USEFUL_BPS = 50 * 1024


def _source_url(source: str, filename: str) -> str:
    quoted = "/".join(quote(part) for part in filename.split("/"))
    if "{file}" in source:
        return source.replace("{file}", quoted)
    return source.rstrip("/") + "/" + quoted


def _probe_source(source: str, filename: str) -> float:
    """Bytes per second for a short ranged read, or 0.0 if unusable.

    A Range request rather than a full download: the point is to choose, not to
    transfer, and a probe that pulls a gigabyte to decide has already lost.
    Hosts that ignore Range simply deliver the first chunk before we stop
    reading, which measures the same thing.
    """
    url = _source_url(source, filename)
    request = Request(url, headers={"Range": f"bytes=0-{_PROBE_BYTES - 1}"})
    start = time.monotonic()
    try:
        with urlopen(request, timeout=_PROBE_TIMEOUT) as response:
            read = len(response.read(_PROBE_BYTES))
    except Exception as error:      # noqa: BLE001 — an unusable source is a result
        log.info(f"[model_downloader] probe failed for {url}: {error}")
        return 0.0
    # A host that answers instantly with four bytes of error page measures as
    # *extremely* fast — rate alone would rank it first and then the real
    # download would fail over to the good source having already lost the
    # choice. Require the probe to have actually delivered the content.
    if read < _PROBE_BYTES // 2:
        log.info(f"[model_downloader] probe returned {read} B (wanted "
                 f"{_PROBE_BYTES}) ← {source}; treating as unusable")
        return 0.0
    elapsed = max(time.monotonic() - start, 1e-6)
    rate = read / elapsed
    log.info(f"[model_downloader] probe {rate / 1024:.0f} KB/s ← {source}")
    return rate if rate >= _MIN_USEFUL_BPS else 0.0


def _order_sources(name: str, sources: list, files: dict) -> list:
    """Sources fastest-first, measured once per bundle.

    Probed on the *largest* file: it is the one whose transfer time dominates,
    and small files are often served from a different tier than large ones.
    A single source is returned as-is — measuring it would only add latency to
    a decision with one outcome.
    """
    if len(sources) < 2:
        return list(sources)
    biggest = max(files, key=lambda f: int(files[f].get("size") or 0))
    if int(files[biggest].get("size") or 0) < _PROBE_BYTES:
        # Nothing here is big enough to measure with. Whichever source we pick,
        # the transfer is over before the choice could have mattered.
        return list(sources)
    scored = [(_probe_source(source, biggest), source) for source in sources]
    usable = [source for rate, source in sorted(scored, reverse=True) if rate > 0]
    if not usable:
        # Every probe failed. Rather than give up here, hand back the original
        # order and let the real download produce the real error — a probe is a
        # heuristic, and a transient failure during it should not mask a host
        # that would have worked.
        log.warning(f"[model_downloader] {name}: all source probes failed; "
                    f"trying them in declared order")
        return list(sources)
    return usable


def _download_verified_bundle(
    name: str, base_url: str, model_dir: str, files: dict, progress_cb=None
) -> None:
    """Download and verify a multi-file model before replacing its destination.

    Progress is reported across the *bundle*, not per file: every size is pinned
    up front, so a 437 MB model plus a 10 KB tokens.txt reads as one monotonic
    0-100% rather than jumping back to 0% for the second file.
    """
    sources = base_url if isinstance(base_url, (list, tuple)) else [base_url]
    sources = _order_sources(name, list(sources), files)

    os.makedirs(model_dir, exist_ok=True)
    total_bytes = sum(int(m.get("size") or 0) for m in files.values())
    done_bytes = 0
    staging_prefix = f".{name.replace('/', '_')}-"
    with tempfile.TemporaryDirectory(prefix=staging_prefix, dir=model_dir) as staging:
        for filename, metadata in files.items():
            _check_bundle_relpath(filename)
            destination = os.path.join(staging, filename)
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            # Fall through the remaining sources on failure. Safe precisely
            # because `_fetch_pinned_file` verifies size and SHA256 before
            # accepting anything: a second host cannot smuggle in a different
            # file, only serve the same one faster or not at all.
            last_error = None
            for index, source in enumerate(sources):
                try:
                    _fetch_pinned_file(name, _source_url(source, filename),
                                       destination, metadata, label=filename,
                                       progress_cb=progress_cb,
                                       done_bytes=done_bytes,
                                       total_bytes=total_bytes)
                    last_error = None
                    break
                except Exception as error:      # noqa: BLE001 — try the next host
                    last_error = error
                    remaining = len(sources) - index - 1
                    # Name the *root* cause: after its retries _fetch_pinned_file
                    # raises a uniform "failed to download X", so without this
                    # every mirror failure reads the same whether the host 404ed,
                    # timed out, or served a file whose SHA256 did not match.
                    cause = error.__cause__ if error.__cause__ is not None else error
                    log.warning(
                        f"[model_downloader] {name}: {filename} failed from "
                        f"{source}: {error} ({type(cause).__name__})"
                        + (f"; {remaining} source(s) left" if remaining else "")
                    )
            if last_error is not None:
                raise last_error
            done_bytes += int(metadata.get("size") or 0)

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
    # Offline NeMo Parakeet CTC 110M, fp32. The int8 archive this model's cpu
    # entry uses is deliberately NOT reused here: ONNX Runtime's CUDA provider
    # has no int8 kernels and falls back per node. No fp16 variant is published
    # upstream, so fp32 is the only gpu option and there is nothing to compare
    # it against — which, given what fp16 did to sensevoice on CUDA, is fine.
    "asr_parakeet_en_gpu": {
        "base_url": f"{SHERPA_GPU_MODEL_BASE}/nemo-parakeet-tdt-ctc-110m-en-fp32",
        "files": {
            "model.onnx": {
                "size": 458161021,
                "sha256": "936806cf3dd0db5aba53f8c7410bb5632d7a8ad6b2c51009f5e4fc0890ec76bf",
            },
            "tokens.txt": {
                "size": 9953,
                "sha256": "450e56bd2f036fe5b6aa821865838cc5aa9d8b0106134ce9a9ba0664abe6cd10",
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


def ensure_gpu_model(name: str, model_dir: str, progress_cb=None) -> dict[str, str]:
    """Ensure a `device: gpu` weight bundle is present and SHA256-verified.

    Takes the same `progress_cb(pct, mb_done, mb_total)` as ensure_model. These
    are the largest downloads in the stack — parakeet's fp32 weights are 437 MB
    and took 81 s on an Orin — and without a callback the card sat on a static
    "fetching" line for that whole time, which is indistinguishable from hung.
    """
    bundle = SHERPA_GPU_BUNDLES.get(name)
    if bundle is None:
        raise KeyError(
            f"No GPU weight bundle named {name!r}; "
            f"available: {sorted(SHERPA_GPU_BUNDLES)}"
        )
    return ensure_verified_bundle(name, model_dir, bundle["base_url"],
                                  bundle["files"], progress_cb=progress_cb)


# ── SoundEvent (Google YAMNet TFLite) ───────────────────────────────────────
SOUNDEVENT_MODEL_DIR = "/models/soundevent"
SOUNDEVENT_MODEL_FILENAME = "yamnet_classification.tflite"
SOUNDEVENT_MODEL_BASE = os.environ.get(
    "SOUNDEVENT_MODEL_BASE_URL", f"{COS_BASE}/soundevent"
)
SOUNDEVENT_MODELSCOPE_BASE = (
    "https://www.modelscope.cn/models/zhangyiqun/"
    "yamnet-audio-classification-tflite/resolve/master"
)
SOUNDEVENT_MODEL_FILES = {
    SOUNDEVENT_MODEL_FILENAME: {
        "size": 4126810,
        "sha256": "10c95ea3eb9a7bb4cb8bddf6feb023250381008177ac162ce169694d05c317de",
    },
}
# Two hosts rather than one, so the shared downloader probes them and uses the
# fastest — COS wins from inside the VPC, ModelScope from several of the rigs.
# Declared order is only the tiebreak when every probe fails. Deciding by
# measurement is safe because the file is pinned by size and SHA256 above, so
# both hosts must deliver byte-identical content.
SOUNDEVENT_MODEL_SOURCES = [SOUNDEVENT_MODEL_BASE, SOUNDEVENT_MODELSCOPE_BASE]


def ensure_soundevent_model(progress_cb=None) -> str:
    """Fetch pinned YAMNet from whichever of its sources answers fastest."""
    model_dir = require_models_subpath(SOUNDEVENT_MODEL_DIR)
    paths = ensure_verified_bundle(
        "soundevent", model_dir, SOUNDEVENT_MODEL_SOURCES, SOUNDEVENT_MODEL_FILES,
        progress_cb=progress_cb,
    )
    return paths[SOUNDEVENT_MODEL_FILENAME]


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


def ensure_ocr_model(model_dir: str, family: str | None = None,
                     progress_cb=None) -> dict[str, str]:
    """Ensure the OCR TensorRT bundle matching the runtime TensorRT is present."""
    model_dir = require_models_subpath(model_dir)
    key = select_bundle_family(OCR_MODEL_BUNDLES, family)
    entry = OCR_MODEL_BUNDLES[key]
    log.info(f"[model_downloader] ocr: using {key} bundle")
    return ensure_verified_bundle(
        f"ocr/{key}", model_dir, entry["base_url"], entry["files"],
        progress_cb=progress_cb,
    )


# ── Face recognition (InsightFace buffalo_sc: SCRFD detector + ArcFace) ──
# Plain ONNX, run by the standalone onnxruntime, so — unlike the OCR bundle —
# there is nothing JetPack-specific about these files and no family selection:
# one bundle serves both Jetson lines and any x86 dev host.
#
# Re-hosted on COS rather than fetched from the upstream GitHub release. The
# release URL redirects to a signed, expiring `release-assets.githubusercontent`
# URL, which cannot be pinned, and the robots have no reliable route to GitHub
# anyway (see CLAUDE.md § "When a page or API won't load").
FACE_MODEL_BASE = os.environ.get(
    "FACE_MODEL_BASE_URL", f"{COS_BASE}/face/buffalo_sc"
)
# Pinned against the files re-hosted from the insightface v0.7 `buffalo_sc.zip`
# release; the COS copies were re-downloaded and re-hashed after upload, so
# these are the bytes a robot will actually receive.
FACE_MODEL_FILES = {
    # SCRFD-500M-BNKPS — detection + the 5 landmarks ArcFace alignment needs.
    # 9 outputs: score/bbox/kps for strides 8, 16, 32 (verified against the
    # decoder in plugins/face_runtime.py).
    "det_500m.onnx": {
        "size": 2524817,
        "sha256": "5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a",
    },
    # ArcFace MobileFaceNet trained on Glint360K — 112x112 in, 512-d out.
    "w600k_mbf.onnx": {
        "size": 13616099,
        "sha256": "9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f",
    },
}


FACE_MODEL_BUNDLES = {
    "face": (FACE_MODEL_BASE, FACE_MODEL_FILES),
}


def ensure_face_model(model_dir: str, bundle: str = "face",
                      progress_cb=None) -> dict[str, str]:
    """Ensure a face detection + recognition ONNX pair is present.

    `bundle` selects which pinned set to fetch, so a second model added to
    `plugins/face_runtime.FACE_MODELS` brings its own sizes and hashes rather than
    reusing these. One entry today; the parameter exists so adding the second does not
    have to touch the call site.
    """
    spec = FACE_MODEL_BUNDLES.get(bundle)
    if spec is None:
        raise ValueError(
            f"unknown face model bundle {bundle!r}; this build has "
            f"{sorted(FACE_MODEL_BUNDLES)}")
    base, files = spec
    model_dir = require_models_subpath(model_dir)
    return ensure_verified_bundle(bundle, model_dir, base, files,
                                  progress_cb=progress_cb)


def ensure_verified_archive(name: str, model_dir: str, url: str, entry: dict,
                            progress_cb=None, stage_cb=None) -> None:
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
                _fetch_pinned_file(name, url, archive, entry,
                                   progress_cb=progress_cb)
                payload = os.path.join(staging, "payload")
                os.makedirs(payload)
                _notify_stage(name, stage_cb, "extract")
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


def ensure_vits2_model(model_dir: str, family: str | None = None,
                       progress_cb=None, stage_cb=None) -> str:
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
        progress_cb=progress_cb,
        stage_cb=stage_cb,
    )
    return os.path.join(model_dir, "engines", key)


# The Thai TTS voice: an ONNX export of VIZINTZOR/MMS-TTS-THAI-MALE-NARRATOR,
# produced by tools/export_mms_thai_onnx.py. One pinned tarball rather than a
# per-file bundle because the payload is a model plus its token table plus the
# licence note, and the archive checksum then covers all three.
#
# Licence: CC-BY-NC-4.0, inherited from facebook/mms-tts. NON-COMMERCIAL —
# see the LICENSE file inside the archive.
THAI_TTS_MODEL_BASE = os.environ.get("THAI_TTS_MODEL_BASE_URL", COS_BASE)
THAI_TTS_ARCHIVE = {
    "archive": "mms-tts-thai-male-narrator-16k.tar.gz",
    # Verified by re-downloading the uploaded object and hashing that copy, not
    # the local file that was uploaded — the point of the pin is to catch a bad
    # transfer, and hashing the source cannot.
    "size": 105246833,
    "sha256": "85aba3adca3017e955993a3f1ca0fd9aed3216a24b8c64b210f02375b12a4eb4",
}


def ensure_thai_tts_model(model_dir: str, progress_cb=None,
                          stage_cb=None) -> str:
    """Ensure the Thai VITS model + tokens are installed; return the directory."""
    model_dir = require_models_subpath(model_dir)
    if not THAI_TTS_ARCHIVE.get("sha256") or not THAI_TTS_ARCHIVE.get("size"):
        # Refuse rather than download unpinned: every other model here is
        # size+SHA256 verified, and a Thai voice that skipped that would be the
        # one unauthenticated blob in the image's supply chain.
        raise RuntimeError(
            "THAI_TTS_ARCHIVE has no pinned size/sha256 — publish the tarball to "
            "COS and record them (see tools/export_mms_thai_onnx.py)"
        )
    ensure_verified_archive(
        "thai-tts",
        model_dir,
        f"{THAI_TTS_MODEL_BASE.rstrip('/')}/{THAI_TTS_ARCHIVE['archive']}",
        THAI_TTS_ARCHIVE,
        progress_cb=progress_cb,
        stage_cb=stage_cb,
    )
    return model_dir


# ── Kokoro TTS (Kokoro-82M v1.0, 24 kHz, ONNX; one archive per device) ─────────
# Keyed by **device**, not by JetPack family: Kokoro is plain ONNX Runtime, so
# unlike the VITS2 TensorRT plans above there is nothing tied to a TensorRT major
# and select_bundle_family does not apply. What does differ per device is the
# weights themselves — provider_for_device refuses int8 on CUDA (ONNX Runtime's
# CUDA provider falls back to CPU per quantised node and measured slower than
# fp32), so gpu must get fp32 and cpu wants int8. Two archives rather than one
# holding both means a robot downloads ~330 MB or ~120 MB, not 450 MB of which
# half is never loaded.
#
# Each extracts into its own `<device>/` subdirectory, the same shape
# ensure_vits2_model uses for `engines/<family>/`. Sharing one directory would put
# two ensure_verified_archive installs in the same tree, where the second's
# staging replace could take the first's weights with it; separate subdirectories
# make the question not arise, and flipping `device` back finds its files still
# there.
#
# Repacked from the sherpa-onnx release asset kokoro-multi-lang-v1_0.tar.bz2 by
# tools/repack_kokoro_v1_0.py, which drops three things upstream ships that this
# deployment can never read:
#   - lexicon-us-en.txt / lexicon-gb-en.txt — unreachable. sherpa-onnx takes the
#     espeak path for non-Chinese text whenever `lang` is non-empty, and `lang`
#     defaults to the model's own meta_data.voice ("en-us"), so it is never empty.
#   - dict/ (the jieba dictionary) — ignored since sherpa-onnx v1.12.15; passing
#     dict_dir now only logs a warning.
# lexicon-zh.txt IS reachable (the Chinese branch does not consult `lang`) and is
# required for lang=zh, so it stays, as do the three ZH rule FSTs.
#
# Licence: Apache-2.0, inherited from hexgrad/Kokoro-82M — see the LICENSE file
# inside the archive.
KOKORO_MODEL_BASE = os.environ.get("KOKORO_MODEL_BASE_URL", COS_BASE)
KOKORO_MODEL_ARCHIVES = {
    # Verified by re-downloading the uploaded object and hashing that copy, not the
    # local file that was uploaded — the point of the pin is to catch a bad
    # transfer, and hashing the source cannot. (Same note as THAI_TTS_ARCHIVE.)
    "gpu": {
        "archive": "kokoro-multi-v1_0-24k-fp32.tar.gz",
        "size": 337021832,
        "sha256": "519afd6a443c5eb4c9c75d4f677c43beb0aa56f33c73c063d0456d4ceeb58156",
    },
    "cpu": {
        "archive": "kokoro-multi-v1_0-24k-int8.tar.gz",
        "size": 124706598,
        "sha256": "ccf70f4fd809a1c697333c3a036c9d94799f1620aedf932271ea163cd72b97fc",
    },
}


def ensure_kokoro_model(model_dir: str, device: str = "gpu",
                        progress_cb=None, stage_cb=None) -> str:
    """Ensure the Kokoro release for `device` is installed; return its directory.

    Returns `<model_dir>/<device>`, which is what the adapter passes to
    sherpa-onnx — the caller never assembles the subdirectory itself.
    """
    model_dir = require_models_subpath(model_dir)
    key = "gpu" if str(device).strip().lower() == "gpu" else "cpu"
    entry = KOKORO_MODEL_ARCHIVES[key]
    if not entry.get("sha256") or not entry.get("size"):
        # Refuse rather than download unpinned, the same rule ensure_thai_tts_model
        # states: every other model here is size+SHA256 verified, and an
        # unauthenticated 330 MB blob would be the one hole in that.
        raise RuntimeError(
            f"KOKORO_MODEL_ARCHIVES[{key!r}] has no pinned size/sha256 — build the "
            "tarball with tools/repack_kokoro_v1_0.py, publish it to COS, and "
            "record the size and SHA256 of the *uploaded* copy here"
        )
    target = os.path.join(model_dir, key)
    log.info(f"[model_downloader] kokoro: using {key} archive")
    ensure_verified_archive(
        f"kokoro/{key}",
        target,
        f"{KOKORO_MODEL_BASE.rstrip('/')}/{entry['archive']}",
        entry,
        progress_cb=progress_cb,
        stage_cb=stage_cb,
    )
    return target


# ── Vision engines (vop detection, visual_depth monocular depth) ────────────
#
# Both plugins run a prebuilt TensorRT engine, so these follow OCR's shape:
# one bundle per JetPack family, selected by the TensorRT that is actually
# importable. Engines are not portable across TensorRT majors.
#
# They are produced by tools/export_vision_engines.py, which drives
# ultralytics' exporter on a host of the matching JetPack line. What has to come
# from ultralytics is the *ONNX*, with set_classes() already applied, or the
# open-vocabulary class list is not baked into the weights at all. The engine
# build itself could be done by trtexec — read_engine_file() strips the
# ultralytics JSON header when present and accepts a plain engine otherwise —
# but going through ultralytics end to end keeps the class names inside the
# engine, which is where the plugin reads them from.
#
# vop's bundle also carries `vocab.json` beside the engine. That is the
# fallback, not the source of truth: an engine exported without names would
# otherwise leave vop labelling detections by index. The class list is frozen
# into the weights at export time (ultralytics raises on set_classes() for an
# exported model), so neither copy can be changed on a robot.
VISION_MODEL_BASE = os.environ.get("VISION_MODEL_BASE_URL", f"{COS_BASE}/vision")

# The jp61 bundle is built against TensorRT 10.4, which is what the jp6.1
# *image* ships — not the 10.3 its Jetson hosts carry. An engine plan only
# loads on the TensorRT that built it, so a bundle built on the host was
# rejected by every jp6.1 robot. The version is in the path so the mismatch is
# visible without deserializing anything.
#
# Every pin below was taken from the copy downloaded back out of COS, not from
# the file that was uploaded — the point of the pin is to catch a bad transfer,
# and hashing the source cannot. (Same note as THAI_TTS_ARCHIVE / KOKORO.)
#
# vocab.json is byte-identical across both families; the two bundles carry
# their own copy anyway so a family is one self-contained download.
_VOP_VOCAB = {
    "size": 1969,
    "sha256": "5aaa0f34df07fff0037318c4100f40bf55b62beb439b89f60b6641924f17fd3b",
}

VOP_MODEL_BUNDLES = {
    "jp61": {
        "base_url": f"{VISION_MODEL_BASE}/yoloe-26s-seg/tensorrt-jp61-trt10.4-orin-640",
        "files": {
            "yoloe-26s-seg.engine": {
                "size": 24780908,
                "sha256": "b8cb77a0685a399ef7d83dfc4d0777b54e66ea110d1085a005c4f153366e4099",
            },
            "vocab.json": _VOP_VOCAB,
        },
    },
    "jp511": {
        "base_url": f"{VISION_MODEL_BASE}/yoloe-26s-seg/tensorrt-jp511-trt8.5-orin-640",
        "files": {
            "yoloe-26s-seg.engine": {
                "size": 23742701,
                "sha256": "49df478a308de3a1f996d4784d2b00245486c04a40e0b7b6005b7226674da4fe",
            },
            "vocab.json": _VOP_VOCAB,
        },
    },
}

DEPTH_MODEL_BUNDLES = {
    "jp61": {
        "base_url": f"{VISION_MODEL_BASE}/yolo26n-depth/tensorrt-jp61-trt10.4-orin-640",
        "files": {
            "yolo26n-depth.engine": {
                "size": 14020431,
                "sha256": "d7fd1096fd2d29226b85693693a9ec11b65b0097ad0e783803b5fc7218d8f23b",
            },
        },
    },
    "jp511": {
        "base_url": f"{VISION_MODEL_BASE}/yolo26n-depth/tensorrt-jp511-trt8.5-orin-640",
        "files": {
            "yolo26n-depth.engine": {
                "size": 13059848,
                "sha256": "2f9da78b4eb689a30860996c7b962770fd09d4d86a4578f1010844c3ef6d68c5",
            },
        },
    },
}


def _ensure_vision_bundle(
    kind: str, bundles: dict, model_dir: str, family: str | None = None,
    progress_cb=None,
) -> dict[str, str]:
    """Shared body of ensure_vop_model / ensure_depth_model.

    Refuses an unpinned entry rather than downloading it, for the reason
    ensure_kokoro_model states: every other model here is size+SHA256 verified,
    and a placeholder would be the one hole in that. A bundle whose pins are
    still zero has not been published yet.
    """
    model_dir = require_models_subpath(model_dir)
    key = select_bundle_family(bundles, family)
    entry = bundles[key]
    unpinned = [
        name for name, meta in entry["files"].items()
        if not meta.get("sha256") or not meta.get("size")
    ]
    if unpinned:
        raise RuntimeError(
            f"{kind.upper()}_MODEL_BUNDLES[{key!r}] has no pinned size/sha256 for "
            f"{sorted(unpinned)} — build the engine with "
            "tools/export_vision_engines.py on a host of that JetPack line, "
            "publish it to COS, and record the size and SHA256 of the *uploaded* "
            "copy here"
        )
    log.info(f"[model_downloader] {kind}: using {key} bundle")
    return ensure_verified_bundle(
        f"{kind}/{key}", model_dir, entry["base_url"], entry["files"],
        progress_cb=progress_cb,
    )


def ensure_vop_model(model_dir: str, family: str | None = None,
                     progress_cb=None) -> dict[str, str]:
    """Ensure the vop detection engine + its frozen vocabulary are present."""
    return _ensure_vision_bundle("vop", VOP_MODEL_BUNDLES, model_dir, family,
                                 progress_cb=progress_cb)


def ensure_depth_model(model_dir: str, family: str | None = None,
                       progress_cb=None) -> dict[str, str]:
    """Ensure the monocular depth engine matching the runtime TensorRT is present."""
    return _ensure_vision_bundle("depth", DEPTH_MODEL_BUNDLES, model_dir, family,
                                 progress_cb=progress_cb)


# Indoor distance models are independent of the visual_depth model bundle.
_OBSTACLE_MODEL_BASE = (
    "https://modelscope.cn/api/v1/models/Flame4pd/obstacle-indoor-yolo26s-trt/repo"
    "?Revision=dd09f801bf732586b09ba9c1aa15f944d465c535&FilePath="
)
OBSTACLE_MODEL_BUNDLES = {
    "jp61": {
        "base_url": _OBSTACLE_MODEL_BASE + "jp61/{file}",
        "files": {
            "indoor-metric.engine": {
                "size": 30838180,
                "sha256": "6b8afab1f7f4633ce9d100211e3f39622c0478f34cff39589f4e3222601dde26",
            },
        },
    },
    "jp511": {
        "base_url": _OBSTACLE_MODEL_BASE + "jp511/{file}",
        "files": {
            "indoor-metric.engine": {
                "size": 27019922,
                "sha256": "4cb00f5bd4d2609c8a91eb0a9b8759484699eb7075a6c806ecafa5bd590a4029",
            },
        },
    },
}


def ensure_obstacle_models(model_dir: str, family: str | None = None,
                           progress_cb=None) -> dict[str, str]:
    """Fetch the pinned indoor distance engine for the running TensorRT family."""
    return _ensure_vision_bundle("obstacle", OBSTACLE_MODEL_BUNDLES, model_dir, family,
                                 progress_cb=progress_cb)
