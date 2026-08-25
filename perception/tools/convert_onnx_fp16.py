#!/usr/bin/env python3
"""
tools/convert_onnx_fp16.py — Convert sherpa-onnx fp32 weights to fp16.

The `device: gpu` bundles are ours, not upstream's, so the recipe has to live in
the repo rather than only in whatever produced the blob on COS.

    python3 tools/convert_onnx_fp16.py model.onnx --out model.fp16.onnx

Requires `onnx` and `onnxconverter-common`, which are NOT in the perception image:

    pip3 install onnx==1.14.1 onnxconverter-common==1.14.0

Two flags are load-bearing and both were learned the hard way:

- `keep_io_types=True`. sherpa-onnx hands the session fp32 features, so the graph's
  inputs and outputs must stay fp32 with only the interior in fp16. Without it the
  session rejects sherpa's tensors.

- `disable_shape_infer=False` (i.e. leave shape inference ON). Ops that cannot take
  fp16 — `Range` above all — are already in onnxconverter-common's default
  op_block_list, and the converter fences them with Cast nodes. Placing those Casts
  needs shape inference. Turn it off to save 30 seconds and you get a file that
  saves fine and then fails at session creation with:

      Type 'tensor(float16)' of input parameter
      (/encoder/embed/Constant_10_output_0) of operator (Range) in node
      (/encoder/embed/Range_1) is invalid

**Converting successfully is not the same as converting correctly, and this script
cannot tell you the difference.** `--check-session` proves ONNX Runtime will load
the result; it says nothing about whether the model still transcribes. Streaming
paraformer fp16 loaded, ran 20x faster than fp16-on-CPU, gave byte-identical
results across three thread counts on CUDA — and emitted nothing but `</s>`. The
same file on CPU was fine. So before adding any (model, device) pair to
ASR_MODELS, decode real audio on the target device and *read the text*. See
perception/README.md § sherpa-onnx Device Selection.
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def convert(src: str, dst: str, extra_block: tuple[str, ...] = ()) -> None:
    import onnx
    from onnxconverter_common import float16

    block = set(getattr(float16, "DEFAULT_OP_BLOCK_LIST", ()))
    block |= set(extra_block)

    print(f"loading {src} ({os.path.getsize(src) / 1048576:.1f} MB)")
    t0 = time.perf_counter()
    model = onnx.load(src)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    print(f"converting to fp16 (op_block_list: {len(block)} ops)")
    t0 = time.perf_counter()
    converted = float16.convert_float_to_float16(
        model,
        keep_io_types=True,        # sherpa feeds fp32 features
        disable_shape_infer=False,  # required for correct Cast fencing — see docstring
        op_block_list=sorted(block),
    )
    print(f"  converted in {time.perf_counter() - t0:.1f}s")

    del model
    onnx.save(converted, dst)
    print(f"saved {dst} ({os.path.getsize(dst) / 1048576:.1f} MB)")


def check_session(paths: dict, kind: str, tokens: str, provider: str) -> None:
    """Create a real ONNX Runtime session. Necessary, nowhere near sufficient."""
    import sherpa_onnx

    print(f"\ncreating a {kind} session on {provider} ...")
    if kind == "sense_voice":
        sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=paths["model"], tokens=tokens, num_threads=2,
            provider=provider, use_itn=True)
    elif kind == "streaming_paraformer":
        sherpa_onnx.OnlineRecognizer.from_paraformer(
            encoder=paths["encoder"], decoder=paths["decoder"], tokens=tokens,
            num_threads=2, provider=provider, sample_rate=16000,
            decoding_method="greedy_search")
    else:
        raise SystemExit(f"unknown --check-session kind: {kind}")
    print("  session created OK")
    print("  NOTE: this does not prove the model transcribes. Decode real audio on "
          "the target device and read the text before shipping it.")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description="Convert sherpa-onnx fp32 weights to fp16.",
        epilog="Loading successfully is not correctness — read the transcripts.")
    p.add_argument("src", help="fp32 .onnx file")
    p.add_argument("--out", help="output path (default: <src stem>.fp16.onnx)")
    p.add_argument("--block-op", action="append", default=[],
                   metavar="OP",
                   help="extra op to keep in fp32, on top of the library default")
    p.add_argument("--check-session", choices=("sense_voice", "streaming_paraformer"),
                   help="after converting, build an ONNX Runtime session")
    p.add_argument("--tokens", help="tokens.txt, required by --check-session")
    p.add_argument("--check-provider", default="cpu", choices=("cpu", "cuda"),
                   help="provider for --check-session (default cpu)")
    p.add_argument("--decoder", help="decoder .onnx, for --check-session "
                                     "streaming_paraformer")
    args = p.parse_args(argv)

    if not os.path.isfile(args.src):
        p.error(f"no such file: {args.src}")
    dst = args.out or args.src.replace(".onnx", "") + ".fp16.onnx"
    if os.path.exists(dst):
        p.error(f"refusing to overwrite {dst}")

    convert(args.src, dst, tuple(args.block_op))

    if args.check_session:
        if not args.tokens:
            p.error("--check-session needs --tokens")
        if args.check_session == "streaming_paraformer" and not args.decoder:
            p.error("--check-session streaming_paraformer needs --decoder")
        check_session({"model": dst, "encoder": dst, "decoder": args.decoder},
                      args.check_session, args.tokens, args.check_provider)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
