"""
utils/model_downloader.py — Auto-download sherpa-onnx models from COS if missing.
"""

from __future__ import annotations

import fcntl
import logging
import os
import tarfile
import tempfile
import zipfile
from urllib.request import urlretrieve

log = logging.getLogger(__name__)

COS_BASE = "https://agi-phanthy-dev-1252788780.cos.ap-beijing.myqcloud.com/public"

# VITS2 TRT TTS 资产（frontend / vits2 源码 / nltk / TRT engine）
VITS2_COS_BASE = "https://dou-1257995901.cos.ap-guangzhou.myqcloud.com/source_code"


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
    # VITS2 TRT TTS：前端 G2P + vits2 源码 + nltk 数据（解压到 model_dir 根）
    "vits2_mix": {
        "url": f"{VITS2_COS_BASE}/vits2_model.tar.gz",
        "check_file": "frontend/cleaner.py",
    },
    # VITS2 TRT engine（JP5 / TRT 8.5.2）
    "vits2_trt_mel20full_d50_jp5": {
        "url": f"{VITS2_COS_BASE}/trt_mel20full_d50.tar.gz",
        "check_file": "encoder.trt",
    },
    # VITS2 TRT engine（JP6 / TRT 10.4）
    "vits2_trt_mel20full_d50_jp6": {
        "url": f"{VITS2_COS_BASE}/trt_mel20full_d50_jp6.tar.gz",
        "check_file": "encoder.trt",
    },
    # VITS2 TRT engine（JP5 / TRT 8.5.2，mxd_m45d5e6 微调解码器）
    "vits2_trt_mel20full_d50_mxd_jp5": {
        "url": f"{VITS2_COS_BASE}/trt_mel20full_d50_mxd.tar.gz",
        "check_file": "encoder.trt",
    },
    # VITS2 TRT engine（JP6 / TRT 10.4，mxd_m45d5e6 微调解码器）
    "vits2_trt_mel20full_d50_mxd_jp6": {
        "url": f"{VITS2_COS_BASE}/trt_mel20full_d50_mxd_jp6.tar.gz",
        "check_file": "encoder.trt",
    },
}


def ensure_model(name: str, model_dir: str) -> None:
    """Ensure model files exist in model_dir. Download from COS if missing.

    用文件锁串行化下载，避免多个进程共享同一 model_dir 时并发解压冲突 /
    重复下载。check_file 作为完成标记，锁内做二次检查。
    """
    info = MODELS.get(name)
    if not info:
        raise ValueError(f"Unknown model name: {name}")

    check_path = os.path.join(model_dir, info["check_file"])
    if os.path.exists(check_path):
        log.info(f"[model_downloader] {name}: already exists at {model_dir}")
        return

    os.makedirs(model_dir, exist_ok=True)
    lock_path = os.path.join(model_dir, ".download.lock")

    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)  # 阻塞：同一时间只有一个进程下载
        try:
            # 二次检查：等锁期间别的进程可能已经下载完成
            if os.path.exists(check_path):
                log.info(f"[model_downloader] {name}: downloaded by another process")
                return

            url = info["url"]
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
            elif url.endswith(".tar.gz") or url.endswith(".tgz"):
                suffix = ".tar.gz"
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
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)


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
    """Extract tar.bz2 / tar.gz, stripping common top-level directory prefix."""
    mode = "r:gz" if tar_path.endswith((".gz", ".tgz")) else "r:bz2"
    with tarfile.open(tar_path, mode) as tf:
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
