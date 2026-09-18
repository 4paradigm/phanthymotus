#!/usr/bin/env python3
"""
tools/export_mms_thai_onnx.py — export an MMS Thai VITS checkpoint for sherpa-onnx.

The Thai voice is a fine-tune of Meta's MMS VITS (VIZINTZOR/MMS-TTS-THAI-*), and
those are published as HF `VitsModel` PyTorch weights. sherpa-onnx wants a single
ONNX graph plus a `tokens.txt`, so the recipe has to live in the repo rather than
only in whatever produced the tarball on COS — same reason as
tools/convert_onnx_fp16.py.

    python3 tools/export_mms_thai_onnx.py \
        --repo VIZINTZOR/MMS-TTS-THAI-MALE-NARRATOR \
        --out /tmp/thai-tts

Requires packages that are NOT in the perception image; run it on a dev host:

    pip3 install torch transformers onnx onnxruntime soundfile

Four things are load-bearing:

- **The input signature must be exactly what sherpa-onnx's generic VITS path
  sends.** `OfflineTtsVitsModel::Run` dispatches on the ONNX `comment` metadata:
  anything containing "piper", "coqui" or "Inflect" takes a different, shorter
  argument list. `comment=mms` falls through to `RunVits`, which feeds
  ``(x, x_length, noise_scale, length_scale, noise_scale_w)`` positionally and
  appends `sid` only when the 6th input is literally named `sid` or `speaker`.
  Names and order, not just shapes, decide whether the model loads.

- **The scales have to be graph inputs, not baked constants.** HF's
  `VitsModel.forward` reads `self.noise_scale` / `self.noise_scale_duration` and
  derives `length_scale` from `speaking_rate`, so exporting it as-is freezes the
  speed at whatever the config said and the plugin's `speed` field stops doing
  anything. Hence the wrapper below reimplements the generation path with tensor
  scales.

- **`add_blank`, `sample_rate` and the token table are read from the checkpoint,
  never hardcoded.** The Thai fine-tunes disagree with each other on sample rate
  (MALE-NARRATOR is 16 kHz, FEMALEV2 is 22.05 kHz) and the plugin asserts 16 kHz
  at load, so a hardcoded value would turn a wrong-voice mistake into wrong-pitch
  audio.

- **`tokens.txt` carries no `<unk>` line.** The tokenizer's vocab.json holds ids
  0-70 and adds `<unk>` at 71, but sherpa-onnx simply skips characters it cannot
  find, so the reference layout (willwade/mms-tts-multilingual-models-onnx,
  `tha/tokens.txt`) lists only the 71 real tokens. The space token is written as
  a literal space followed by the separator, i.e. the line is ``"  70"``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

DEFAULT_REPO = "VIZINTZOR/MMS-TTS-THAI-MALE-NARRATOR"

# What the plugin expects. A voice that disagrees is a different integration.
EXPECTED_SAMPLE_RATE = 16000


def _tokens_txt(vocab: dict[str, int]) -> str:
    """Render vocab.json as sherpa-onnx's `<token> <id>` table.

    Sorted by id, not by token: sherpa-onnx trusts the id column, but a table
    whose rows are out of order is impossible to diff against the reference.
    """
    lines = [f"{token} {index}" for token, index in sorted(vocab.items(), key=lambda kv: kv[1])]
    return "\n".join(lines) + "\n"


def build_wrapper(model):
    """Wrap VitsModel so its scales become graph inputs.

    The body follows `transformers.models.vits.VitsModel.forward`; the only
    changes are that noise_scale / length_scale / noise_scale_w arrive as tensors
    and the return value is the bare waveform.
    """
    import torch
    from torch import nn

    class ExportableVits(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x, x_length, noise_scale, length_scale, noise_scale_w):
            inner = self.inner
            mask_dtype = inner.text_encoder.embed_tokens.weight.dtype
            attention_mask = (
                torch.arange(x.shape[1], device=x.device).unsqueeze(0) < x_length.unsqueeze(1)
            ).to(torch.int64)
            input_padding_mask = attention_mask.unsqueeze(-1).to(mask_dtype)

            encoded = inner.text_encoder(
                input_ids=x,
                padding_mask=input_padding_mask,
                attention_mask=attention_mask,
                return_dict=True,
            )
            hidden_states = encoded.last_hidden_state.transpose(1, 2)
            input_padding_mask = input_padding_mask.transpose(1, 2)
            prior_means = encoded.prior_means
            prior_log_variances = encoded.prior_log_variances

            if inner.config.use_stochastic_duration_prediction:
                log_duration = inner.duration_predictor(
                    hidden_states, input_padding_mask, None,
                    reverse=True, noise_scale=noise_scale_w,
                )
            else:
                log_duration = inner.duration_predictor(hidden_states, input_padding_mask, None)

            duration = torch.ceil(torch.exp(log_duration) * input_padding_mask * length_scale)
            predicted_lengths = torch.clamp_min(torch.sum(duration, [1, 2]), 1).long()

            indices = torch.arange(
                predicted_lengths.max(), dtype=predicted_lengths.dtype, device=predicted_lengths.device
            )
            output_padding_mask = (indices.unsqueeze(0) < predicted_lengths.unsqueeze(1))
            output_padding_mask = output_padding_mask.unsqueeze(1).to(input_padding_mask.dtype)

            attn_mask = torch.unsqueeze(input_padding_mask, 2) * torch.unsqueeze(output_padding_mask, -1)
            batch_size, _, output_length, input_length = attn_mask.shape
            cum_duration = torch.cumsum(duration, -1).view(batch_size * input_length, 1)
            indices = torch.arange(output_length, dtype=duration.dtype, device=duration.device)
            valid_indices = (indices.unsqueeze(0) < cum_duration).to(attn_mask.dtype)
            valid_indices = valid_indices.view(batch_size, input_length, output_length)
            padded_indices = valid_indices - nn.functional.pad(
                valid_indices, [0, 0, 1, 0, 0, 0]
            )[:, :-1]
            attn = padded_indices.unsqueeze(1).transpose(2, 3) * attn_mask

            prior_means = torch.matmul(attn.squeeze(1), prior_means).transpose(1, 2)
            prior_log_variances = torch.matmul(attn.squeeze(1), prior_log_variances).transpose(1, 2)

            prior_latents = (
                prior_means
                + torch.randn_like(prior_means) * torch.exp(prior_log_variances) * noise_scale
            )
            latents = inner.flow(prior_latents, output_padding_mask, None, reverse=True)
            waveform = inner.decoder(latents * output_padding_mask, None).squeeze(1)
            return waveform

    wrapper = ExportableVits(model)
    wrapper.eval()
    return wrapper


def add_metadata(path: str, meta: dict[str, str]) -> None:
    """Write sherpa-onnx's metadata onto an existing ONNX file, in place."""
    import onnx

    model = onnx.load(path)
    # Clear first: re-running the export otherwise appends duplicate keys, and
    # ONNX Runtime returns the first match, so a corrected value would be ignored.
    del model.metadata_props[:]
    for key, value in meta.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = str(value)
    onnx.save(model, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="HF repo id of the MMS Thai voice")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--opset", type=int, default=15)
    parser.add_argument("--allow-sample-rate", type=int, default=EXPECTED_SAMPLE_RATE,
                        help="refuse a checkpoint whose sample rate differs; the plugin "
                             "asserts 16 kHz because the ROS topic is audio/pcm-16k")
    parser.add_argument("--check-text", default="สวัสดีครับ ระบบพร้อมแล้ว",
                        help="Thai sentence used for the torch-vs-onnx comparison")
    args = parser.parse_args()

    import numpy as np
    import onnxruntime
    import torch
    from transformers import VitsModel, VitsTokenizer

    os.makedirs(args.out, exist_ok=True)

    print(f"[export] loading {args.repo}")
    tokenizer = VitsTokenizer.from_pretrained(args.repo)
    model = VitsModel.from_pretrained(args.repo)
    model.eval()

    sample_rate = int(model.config.sampling_rate)
    if sample_rate != args.allow_sample_rate:
        print(f"[export] ERROR: {args.repo} is {sample_rate} Hz, expected "
              f"{args.allow_sample_rate}. The perception TTS topic is audio/pcm-16k "
              f"and the adapter asserts that rate; exporting this voice would need a "
              f"resampler added first.", file=sys.stderr)
        return 1
    if int(model.config.num_speakers) > 1:
        print(f"[export] ERROR: {args.repo} declares {model.config.num_speakers} "
              f"speakers; the adapter only accepts speaker_id=0.", file=sys.stderr)
        return 1

    # tokens.txt straight from the tokenizer's own table, so it cannot drift.
    vocab = tokenizer.get_vocab()
    vocab = {token: index for token, index in vocab.items() if token != tokenizer.unk_token}
    tokens_path = os.path.join(args.out, "tokens.txt")
    with open(tokens_path, "w", encoding="utf-8") as handle:
        handle.write(_tokens_txt(vocab))
    print(f"[export] wrote {tokens_path} ({len(vocab)} tokens)")

    if "ำ" in vocab:
        print("[export] NOTE: this table contains ำ, so plugins/thai_frontend.py "
              "will skip its sara-am rewrite. Verify that is right for this voice.")
    else:
        print("[export] table has no ำ (expected for MMS) — the frontend rewrites it to ํา")

    wrapper = build_wrapper(model)

    encoded = tokenizer(text=args.check_text, return_tensors="pt")
    x = encoded["input_ids"]
    x_length = torch.tensor([x.shape[1]], dtype=torch.int64)
    noise_scale = torch.tensor([float(model.noise_scale)], dtype=torch.float32)
    length_scale = torch.tensor([1.0], dtype=torch.float32)
    noise_scale_w = torch.tensor([float(model.noise_scale_duration)], dtype=torch.float32)

    onnx_path = os.path.join(args.out, "model.onnx")
    print(f"[export] tracing to {onnx_path} (opset {args.opset})")
    torch.onnx.export(
        wrapper,
        (x, x_length, noise_scale, length_scale, noise_scale_w),
        onnx_path,
        opset_version=args.opset,
        do_constant_folding=True,
        # The TorchScript tracer, not the dynamo exporter. torch>=2.9 defaults to
        # dynamo, which refuses this graph: HF's attention path emits
        # `aten._is_all_true` from an internal assertion and there is no ONNX
        # decomposition for it ("No ONNX function found"). The tracer drops
        # assertions, which is exactly what an inference-only export wants.
        dynamo=False,
        # Names and order are the contract with RunVits; see the module docstring.
        input_names=["x", "x_length", "noise_scale", "length_scale", "noise_scale_w"],
        output_names=["y"],
        dynamic_axes={"x": {0: "N", 1: "L"}, "x_length": {0: "N"}, "y": {0: "N", 1: "T"}},
    )

    add_metadata(onnx_path, {
        "model_type": "vits",
        # Must not contain piper/coqui/Inflect — those select other input layouts.
        "comment": "mms",
        # THE key that decides whether the model loads at all. sherpa-onnx's
        # OfflineTtsVitsImpl::InitFrontend dispatches on this exact string: only
        # "characters" selects OfflineTtsCharacterFrontend. Anything else falls
        # through to the lexicon branch, which refuses an empty --vits-lexicon with
        # "Not a model using characters as modeling unit" and calls exit(-1) — so
        # the failure is a hard process exit, not an exception the plugin can
        # report. `comment=mms` alone is not enough; it only steers the *input
        # layout*, not the text frontend.
        "frontend": "characters",
        "language": "Thai",
        "voice": "tha",
        "sample_rate": sample_rate,
        "add_blank": int(bool(getattr(tokenizer, "add_blank", True))),
        "n_speakers": 1,
        "speaker_id": 0,
        "has_espeak": 0,
        "punctuation": "",
        "url": f"https://huggingface.co/{args.repo}",
        "license": "CC-BY-NC-4.0 (inherited from facebook/mms-tts)",
    })
    size_mb = os.path.getsize(onnx_path) / 1e6
    print(f"[export] wrote {onnx_path} ({size_mb:.1f} MB)")

    # A manifest beside the model, mirroring what the VITS2 release ships, so the
    # adapter can validate before it constructs sherpa_onnx.OfflineTts. That check
    # is not belt-and-braces: sherpa-onnx answers a wrong `frontend` value with
    # SHERPA_ONNX_EXIT(-1), a process exit that takes ASR, VOP and OCR down with
    # it and that main.py's try/except around the TTS plugin cannot catch.
    manifest = {
        "model": "model.onnx",
        "tokens": "tokens.txt",
        "repo": args.repo,
        "sample_rate": sample_rate,
        "frontend": "characters",
        "comment": "mms",
        "n_speakers": 1,
        "license": "CC-BY-NC-4.0",
        "exported_by": "tools/export_mms_thai_onnx.py",
    }
    manifest_path = os.path.join(args.out, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"[export] wrote {manifest_path}")

    # ── self-check: torch vs onnxruntime on the same input ────────────────
    #
    # VITS samples latents from a normal distribution, so the two waveforms can
    # never be identical. What must match is duration and loudness: a broken
    # export usually produces the right shape and silence, or a wildly different
    # length because a scale got frozen or transposed.
    print("[export] self-check: torch vs onnxruntime")
    with torch.no_grad():
        torch_wave = wrapper(x, x_length, noise_scale, length_scale, noise_scale_w).numpy()[0]

    session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_wave = session.run(None, {
        "x": x.numpy(), "x_length": x_length.numpy(),
        "noise_scale": noise_scale.numpy(),
        "length_scale": length_scale.numpy(),
        "noise_scale_w": noise_scale_w.numpy(),
    })[0][0]

    def rms(wave):
        return float(np.sqrt(np.mean(np.square(wave.astype(np.float64)))) + 1e-12)

    torch_seconds = len(torch_wave) / sample_rate
    onnx_seconds = len(onnx_wave) / sample_rate
    print(f"[export]   torch: {torch_seconds:.2f}s rms={rms(torch_wave):.4f}")
    print(f"[export]   onnx : {onnx_seconds:.2f}s rms={rms(onnx_wave):.4f}")

    problems = []
    if rms(onnx_wave) < 1e-3:
        problems.append("the ONNX output is effectively silent")
    if not 0.8 <= (onnx_seconds / max(torch_seconds, 1e-6)) <= 1.25:
        problems.append(f"durations disagree ({torch_seconds:.2f}s vs {onnx_seconds:.2f}s)")
    ratio = rms(onnx_wave) / rms(torch_wave)
    if not 0.5 <= ratio <= 2.0:
        problems.append(f"loudness disagrees (ratio {ratio:.2f})")

    # length_scale must actually change the duration, or `speed` on the card is
    # decoration — this is the check that catches a frozen scale, which the
    # comparison above cannot see.
    slow = session.run(None, {
        "x": x.numpy(), "x_length": x_length.numpy(),
        "noise_scale": noise_scale.numpy(),
        "length_scale": np.array([1.5], dtype=np.float32),
        "noise_scale_w": noise_scale_w.numpy(),
    })[0][0]
    slow_ratio = len(slow) / max(len(onnx_wave), 1)
    print(f"[export]   length_scale=1.5 → {slow_ratio:.2f}x duration")
    if slow_ratio < 1.2:
        problems.append(f"length_scale does not affect duration ({slow_ratio:.2f}x); "
                        "the plugin's speed field would do nothing")

    try:
        import soundfile

        wav_path = os.path.join(args.out, "sample.wav")
        soundfile.write(wav_path, onnx_wave, sample_rate)
        print(f"[export] wrote {wav_path} — listen to it before uploading")
    except ImportError:
        print("[export] soundfile not installed; skipped the sample wav")

    with open(os.path.join(args.out, "LICENSE"), "w", encoding="utf-8") as handle:
        handle.write(
            f"Source: https://huggingface.co/{args.repo}\n"
            "License: CC-BY-NC-4.0, inherited from facebook/mms-tts.\n"
            "NON-COMMERCIAL. Re-evaluate before shipping this in a product.\n"
        )

    if problems:
        print("[export] FAILED:", file=sys.stderr)
        for problem in problems:
            print(f"[export]   - {problem}", file=sys.stderr)
        return 1

    print("[export] OK. Next: tar it up, upload to COS, and pin size+sha256 in "
          "utils/model_downloader.py THAI_TTS_ARCHIVE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
