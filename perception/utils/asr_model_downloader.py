#!/usr/bin/env python3
"""Download the offline ASR model used by the Jetson image."""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


DEFAULT_BASE_URL = (
    "http://172.28.4.81:34567/zengzhitao/embodied-ai/official_paraformer"
)

OFFICIAL_PARAFORMER_FILES = (
    "config.json",
    "model.int8.onnx",
    "tokens.txt",
)

# x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03
# transducer 三件套（encoder/decoder/joiner），需 asr_offline.py 的 transducer 分支
# bpe.vocab + hotwords.txt 为 hotwords 偏置所需（asr_offline.py 加载时逐字编码）
X_ASR_PUNCT_INT8_FILES = (
    "encoder-epoch-99-avg-1.int8.onnx",
    "decoder-epoch-99-avg-1.onnx",
    "joiner-epoch-99-avg-1.int8.onnx",
    "tokens.txt",
    "bpe.model",
    "bpe.vocab",
    "hotwords.txt",
)

# FireRedVAD (DFSMN, exported to ONNX — see plugins/firered_vad.py header)
# .onnx.data holds the weights (external-data format from the torch 2.x exporter)
FIRERED_VAD_FILES = (
    "firered_vad.onnx",
    "firered_vad.onnx.data",
    "cmvn.npz",
)

DOWNLOAD_TIMEOUT = 120
MAX_RETRIES = 3
RETRY_DELAY = 3.0


def _download_file(
    url: str,
    destination: Path,
    *,
    timeout: float,
    retries: int,
    retry_delay: float,
) -> None:
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if retries < 1:
        raise ValueError("retries must be at least one")

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            size = 0
            with urlopen(url, timeout=timeout) as response, destination.open(
                "wb"
            ) as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    size += len(chunk)
                output.flush()
                os.fsync(output.fileno())
            if size == 0:
                raise ValueError(f"downloaded file is empty: {url}")
            return
        except (URLError, TimeoutError, OSError, ValueError) as error:
            last_error = error
            destination.unlink(missing_ok=True)
            if attempt < retries:
                print(
                    f"Retry {attempt}/{retries} after error: {error}",
                    flush=True,
                )
                time.sleep(retry_delay)

    assert last_error is not None
    raise last_error


def download_model(
    base_url: str,
    output_dir: str | Path,
    filenames: tuple[str, ...],
    *,
    timeout: float = DOWNLOAD_TIMEOUT,
    retries: int = MAX_RETRIES,
    retry_delay: float = RETRY_DELAY,
) -> None:
    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    if not filenames:
        raise ValueError("filenames must not be empty")
    if any(Path(filename).name != filename or not filename for filename in filenames):
        raise ValueError("filenames must be plain file names")

    with tempfile.TemporaryDirectory(
        prefix="asr-model-", dir=destination_dir.parent
    ) as staging_directory:
        staging = Path(staging_directory)
        for filename in filenames:
            url = f"{base_url.rstrip('/')}/{filename}"
            print(f"Downloading {url}", flush=True)
            _download_file(
                url,
                staging / filename,
                timeout=timeout,
                retries=retries,
                retry_delay=retry_delay,
            )

        for filename in filenames:
            os.replace(staging / filename, destination_dir / filename)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--output-dir", default="/models/sherpa-onnx/asr-offline"
    )
    parser.add_argument(
        "--model",
        choices=("x_asr", "paraformer", "firered_vad"),
        default="paraformer",
        help="model family to download (default: paraformer)",
    )
    args = parser.parse_args()
    if args.model == "x_asr":
        filenames = X_ASR_PUNCT_INT8_FILES
    elif args.model == "firered_vad":
        filenames = FIRERED_VAD_FILES
    else:
        filenames = OFFICIAL_PARAFORMER_FILES
    download_model(args.base_url, args.output_dir, filenames)


if __name__ == "__main__":
    main()
