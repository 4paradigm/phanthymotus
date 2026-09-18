#!/usr/bin/env python3
"""Repack the sherpa-onnx Kokoro v1.0 release into the two archives we deploy.

Upstream `kokoro-multi-lang-v1_0.tar.bz2` (fp32) and
`kokoro-int8-multi-lang-v1_0.tar.bz2` are ~350 MB / ~132 MB and each carries three
things this deployment can never read:

  - `lexicon-us-en.txt`, `lexicon-gb-en.txt` — unreachable. In
    `KokoroMultiLangLexicon::ConvertNonChineseToTokenIDs`, a non-empty `voice`
    short-circuits straight to espeak and the lexicon is never consulted; `lang`
    resolves to the model's own `meta_data.voice` ("en-us") when unset, so it is
    never empty. Shipping 11.6 MB that cannot be opened invites the next person to
    assume English pronunciation comes from a dictionary here. It does not.
  - `dict/` — the jieba dictionary. Ignored since sherpa-onnx v1.12.15; passing
    `dict_dir` now only logs "you don't need to provide dict_dir".

`lexicon-zh.txt` is kept: the Chinese branch of that same function does not look at
`voice`, so it is genuinely used, and `lang=zh` needs it. The three ZH rule FSTs are
kept for number/date/phone normalisation.

Also writes a `manifest.json`, mirroring tools/export_mms_thai_onnx.py. The adapter
reads it *before* constructing OfflineTts, because a `version >= 2` Kokoro model with
both `lexicon` and `lang` empty makes sherpa-onnx call `SHERPA_ONNX_EXIT(-1)` — a
process exit that would take ASR, VOP and OCR down with it, not an exception
main.py could catch.

Every value in the manifest is read out of the ONNX metadata, never hardcoded: the
point is to record what the shipped graph actually says, so a future upstream repack
at a different sample rate fails the adapter's check instead of being pitch-shifted.

Run on the x86 build host (root@172.18.66.241), which holds the COS credentials:

    python3 tools/repack_kokoro_v1_0.py --work /tmp/kokoro --out /tmp/kokoro/dist

then upload both tarballs to COS and record the size + SHA256 of the *uploaded*
copies — re-download and hash those, not the local files. Hashing the source cannot
catch a bad transfer, which is the whole reason the pins exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request

UPSTREAM = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models"

# device -> (upstream archive, upstream weight filename, weight name we ship)
VARIANTS = {
    "fp32": ("kokoro-multi-lang-v1_0.tar.bz2", "model.onnx", "model.onnx"),
    "int8": ("kokoro-int8-multi-lang-v1_0.tar.bz2", "model.int8.onnx", "model.int8.onnx"),
}

# Everything besides the weights that goes into every archive. espeak-ng-data is a
# directory; the four files sherpa-onnx's Validate() insists on live inside it.
SHARED_FILES = ("voices.bin", "tokens.txt", "lexicon-zh.txt",
                "date-zh.fst", "number-zh.fst", "phone-zh.fst", "LICENSE")
SHARED_DIRS = ("espeak-ng-data",)
ESPEAK_REQUIRED = ("phontab", "phonindex", "phondata", "intonations")

# Refuse anything else. The adapter downsamples 24000 -> 16000 with a filter
# designed for exactly that 2:3 ratio; another rate would need its own filter, and
# silently accepting one would ship audio at the wrong pitch.
EXPECTED_SAMPLE_RATE = 24000
EXPECTED_VERSION = 2

# Which speaker ids belong to which language, so the adapter can warn when a
# `lang`/voice pair is incoherent (e.g. an Italian voice reading Spanish phonemes).
# Derived from the speaker names in the ONNX metadata rather than written out here.
# The prefix convention is hexgrad's: first letter = language (a American, b
# British, e Spanish, f French, h Hindi, i Italian, j Japanese, p Portuguese,
# z Chinese), second = f/m for the speaker's voice.
LANG_PREFIXES = {
    "en-us": ("af_", "am_"),
    "en-gb": ("bf_", "bm_"),
    "es": ("ef_", "em_"),
    "fr": ("ff_", "fm_"),
    "hi": ("hf_", "hm_"),
    "it": ("if_", "im_"),
    "ja": ("jf_", "jm_"),
    "pt-br": ("pf_", "pm_"),
    "zh": ("zf_", "zm_"),
}


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, dest: str) -> None:
    if os.path.exists(dest):
        print(f"[repack] have {os.path.basename(dest)}")
        return
    print(f"[repack] downloading {url}")
    tmp = dest + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)


def _extract(archive: str, dest: str) -> str:
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    os.makedirs(dest)
    with tarfile.open(archive) as tar:
        # Upstream tarballs are a single top-level directory; strip it so callers
        # get a flat tree regardless of what that directory is called.
        tar.extractall(dest)
    entries = [e for e in os.listdir(dest) if not e.startswith(".")]
    if len(entries) == 1 and os.path.isdir(os.path.join(dest, entries[0])):
        return os.path.join(dest, entries[0])
    return dest


def _read_onnx_meta(model_path: str) -> dict:
    """Read the metadata sherpa-onnx's scripts/kokoro/v1.0/add_meta_data.py wrote.

    onnx.load() on a 310 MB graph would read the whole thing; onnxruntime's session
    would build one. Neither is needed — a metadata-only read is enough, and
    onnxruntime is present in the build host's perception environment.
    """
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
    return dict(session.get_modelmeta().custom_metadata_map)


def _build_manifest(meta: dict, device: str, weights: str) -> dict:
    sample_rate = int(meta.get("sample_rate") or 0)
    version = int(meta.get("version") or 0)
    if sample_rate != EXPECTED_SAMPLE_RATE:
        raise SystemExit(
            f"[repack] ERROR: graph declares sample_rate={sample_rate}, expected "
            f"{EXPECTED_SAMPLE_RATE}. utils/resample.py's filter is designed for the "
            "24000->16000 2:3 ratio; shipping another rate needs a new filter first."
        )
    if version != EXPECTED_VERSION:
        raise SystemExit(
            f"[repack] ERROR: graph declares version={version}, expected "
            f"{EXPECTED_VERSION}. Only Kokoro >= 1.0 honours `lang`, which the "
            "language selector depends on."
        )

    names = [n for n in (meta.get("speaker_names") or "").split(",") if n]
    if not names:
        raise SystemExit("[repack] ERROR: graph has no speaker_names metadata")
    n_speakers = int(meta.get("n_speakers") or len(names))
    if n_speakers != len(names):
        raise SystemExit(
            f"[repack] ERROR: n_speakers={n_speakers} but {len(names)} speaker_names"
        )

    languages = {
        lang: [i for i, name in enumerate(names) if name.startswith(prefixes)]
        for lang, prefixes in LANG_PREFIXES.items()
    }
    for lang, ids in languages.items():
        if not ids:
            raise SystemExit(f"[repack] ERROR: no speaker ids found for {lang}")

    return {
        "source": "sherpa-onnx tts-models/kokoro-multi-lang-v1_0",
        "upstream": "https://huggingface.co/hexgrad/Kokoro-82M (v1.0)",
        "model_version": version,
        "sample_rate": sample_rate,
        "n_speakers": n_speakers,
        "voice_default": meta.get("voice") or "en-us",
        "device": device,
        "weights": weights,
        "id2speaker": {str(i): name for i, name in enumerate(names)},
        "languages": languages,
        "dropped": ["lexicon-us-en.txt", "lexicon-gb-en.txt", "dict/", "README.md"],
    }


def _stage(src: str, staging: str, weight_src: str, weight_dst: str,
           manifest: dict) -> None:
    if os.path.isdir(staging):
        shutil.rmtree(staging)
    os.makedirs(staging)

    shutil.copy2(os.path.join(src, weight_src), os.path.join(staging, weight_dst))
    for name in SHARED_FILES:
        path = os.path.join(src, name)
        if os.path.exists(path):
            shutil.copy2(path, os.path.join(staging, name))
        elif name == "LICENSE":
            print("[repack] WARNING: upstream has no LICENSE file")
        else:
            raise SystemExit(f"[repack] ERROR: upstream is missing {name}")
    for name in SHARED_DIRS:
        shutil.copytree(os.path.join(src, name), os.path.join(staging, name))

    espeak = os.path.join(staging, "espeak-ng-data")
    missing = [f for f in ESPEAK_REQUIRED if not os.path.exists(os.path.join(espeak, f))]
    if missing:
        raise SystemExit(
            f"[repack] ERROR: espeak-ng-data lacks {missing}; sherpa-onnx's "
            "Validate() checks for exactly these and would refuse the release"
        )

    with open(os.path.join(staging, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _tar(staging: str, out_path: str) -> None:
    """Flat gzip tarball — members at the root, matching ensure_verified_archive.

    Sorted, and with uid/gid/mtime normalised, so rebuilding from the same input
    gives the same bytes and a re-upload does not change the pin for no reason.
    """
    def _reset(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mtime = 0
        return info

    with tarfile.open(out_path, "w:gz") as tar:
        for name in sorted(os.listdir(staging)):
            tar.add(os.path.join(staging, name), arcname=name, filter=_reset,
                    recursive=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", default="/tmp/kokoro-repack",
                        help="scratch directory for downloads and extraction")
    parser.add_argument("--out", default=None,
                        help="where the finished tarballs go (default: <work>/dist)")
    parser.add_argument("--variant", choices=sorted(VARIANTS), action="append",
                        help="build only this variant (repeatable; default both)")
    args = parser.parse_args()

    work = os.path.abspath(args.work)
    out = os.path.abspath(args.out or os.path.join(work, "dist"))
    os.makedirs(work, exist_ok=True)
    os.makedirs(out, exist_ok=True)

    device_for = {"fp32": "gpu", "int8": "cpu"}
    results = []

    for variant in (args.variant or sorted(VARIANTS)):
        archive, weight_src, weight_dst = VARIANTS[variant]
        device = device_for[variant]
        print(f"\n[repack] ===== {variant} ({device})")

        local = os.path.join(work, archive)
        _download(f"{UPSTREAM}/{archive}", local)
        src = _extract(local, os.path.join(work, f"src-{variant}"))

        meta = _read_onnx_meta(os.path.join(src, weight_src))
        manifest = _build_manifest(meta, device, weight_dst)
        print(f"[repack] rate={manifest['sample_rate']} version={manifest['model_version']} "
              f"speakers={manifest['n_speakers']}")
        for lang, ids in sorted(manifest["languages"].items()):
            print(f"[repack]   {lang}: ids {min(ids)}-{max(ids)} ({len(ids)} voices)")

        staging = os.path.join(work, f"stage-{variant}")
        _stage(src, staging, weight_src, weight_dst, manifest)
        staged_size = int(subprocess.check_output(["du", "-sb", staging]).split()[0])

        out_path = os.path.join(out, f"kokoro-multi-v1_0-24k-{variant}.tar.gz")
        _tar(staging, out_path)
        results.append((out_path, os.path.getsize(out_path), _sha256(out_path),
                        staged_size, device))

    print("\n[repack] ===== paste into utils/model_downloader.py after uploading")
    print("[repack] (record the size/sha256 of the RE-DOWNLOADED copy, not these)")
    for path, size, digest, staged, device in results:
        print(f'\n    "{device}": {{')
        print(f'        "archive": "{os.path.basename(path)}",')
        print(f'        "size": {size},')
        print(f'        "sha256": "{digest}",')
        print("    },")
        print(f"    # unpacked {staged / 1e6:.0f} MB, archive {size / 1e6:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
