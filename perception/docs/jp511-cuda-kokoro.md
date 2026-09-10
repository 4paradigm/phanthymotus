# jp5.11's CUDA renders Kokoro's Japanese wrongly

**Status:** unresolved. The guard in `plugins/kokoro_worker.py` detects it and refuses
the device; this document records what it is *not*, so the next person does not repeat
five experiments to find out.

## The symptom

On jp5.11 (Orin 5, L4T R35.6.5), `KokoroDirect` — the standalone ONNX Runtime session
this code builds for Japanese — produces audio whose duration does not agree with its
own CPU rendering of the same input, and which is **garbled when listened to**.

Same code, same model, on jp6.1 (Orin 6, L4T R36.4.3) the CUDA output matches the CPU
exactly, down to the same RMS.

    10-token probe "konnichiwa", expected 47400 samples

    jp6.1   cpu   47400, 47400, 47400, 47400
            cuda  47400, 47400, 47400, 47400      <- correct
    jp5.11  cpu   47400, 47400, 47400, 47400
            cuda  17400, 21000, 21600, 20400      <- 54-63% off, and different each run

Whole sentences, native 24 kHz, no resampling or framing in the way:

| | jp5.11 cpu | jp5.11 cuda | jp6.1 cpu | jp6.1 cuda |
|---|---|---|---|---|
| long sentence | 8.35 s | **6.47 s** (22% short) | 8.35 s | 8.35 s |
| short sentence | 2.45 s | **3.77 s** (54% long) | 2.45 s | 2.45 s |

**The direction is not consistent** — the long sentence comes back short and the short
one long, and three renders of one input gave 1.4 s, 1.9 s and 2.95 s. So it is not a
scaling error, and "the audio is rushed" (an early description of mine, inferred from
the long sentence alone) was wrong.

Confirmed by ear: jp6.1's CUDA output is fine, jp5.11's is garbled.

## What it is not

Each ruled out by measurement. Re-running these is not necessary unless something in the
stack changes.

| ruled out | how |
|---|---|
| the two-runtime collision (see `plugins/ort_worker.py`) | reproduced in a process with exactly one ONNX Runtime mapped in `/proc/self/maps` and `sherpa_onnx` never imported |
| a different model file | `model.onnx`, `voices.bin` and `tokens.txt` hash identically on both boxes (`b40f62b1…`, `1c5a5b98…`, `6ebb6bb2…`) |
| TF32 on Ampere | `NVIDIA_TF32_OVERRIDE=0` changes nothing — 63/56/53/56% off either way |
| the ONNX Runtime **version** | NVIDIA's official 1.16.0 wheel for JetPack 5 behaves exactly like the 1.15.1 that ships |
| the CUDA **provider binary** | the Dockerfile treats the lines differently — jp6.1 overwrites the wheel's provider with sherpa's, jp5.11 keeps the wheel's. Substituting sherpa's on jp5.11, with versions matched at 1.16.0, gives the same wrong lengths |

That last row is worth spelling out, because the asymmetry in `Dockerfile.jetson` makes
it a reasonable suspect: jp6.1's working CUDA provider is **sherpa's build**, not the
COS wheel's, so "1.15.1 versus 1.18.x" was never the only difference between the lines.
It is still not the cause.

Note the ABI floor found on the way: sherpa's 1.16.0 provider next to a **1.15.1**
pybind dies with `Illegal instruction (core dumped)`. The version-matched substitution
(1.16.0 pybind + 1.16.0 provider) runs fine and is what the table above reports.

## What is left

The platform stack itself, which cannot be swapped inside a container:

    jp5.11   L4T R35.6.5   CUDA 11.4   cuDNN 8
    jp6.1    L4T R36.4.3   CUDA 12.6   cuDNN 9

The next step, if anyone takes it, is comparing an intermediate tensor between the two
providers to find the first op that diverges — the graph has a second output
(`onnx::Shape_3411`, int64, from a Squeeze) that looks like the predicted length and
would localise it quickly.

## What is *not* broken on jp5.11

"GPU works on jp5.11" is also true, and this document is not evidence against it:

- **sherpa-onnx's own engines** run on the GPU there and always have — ASR, and every
  TTS language except Japanese;
- **face recognition** runs on the GPU there, measured at 27.1 ms per frame.

`KokoroDirect` is the only thing affected. It exists only for Japanese, and it was
pinned to the CPU from the day it was written, so this path had never been exercised on
that line until the shared-worker branch opened it.

## Why the guard is a measurement and not a version check

`plugins/kokoro_worker.py` renders a fixed probe on the chosen device at startup — the
warmup it had to run anyway — and refuses the device if the duration is more than 10%
off. It records the verdict in `.cuda-duration-verdict` next to the model so the
question is answered once per machine rather than once per process start: a rejected
CUDA session leaves memory behind that unloading does not return, measured at 3.8 GB on
Orin 5, and paying that on every restart is what tipped that box into the OOM killer.

A version check would have been wrong **twice** over the course of this investigation:

- written against 1.15.1, it would have let the official 1.16.0 through — which produces
  the same garbled audio;
- written against "the wheel's own provider", it would have let sherpa's through — same
  again.

The failure is real and its cause is unknown, which is exactly the situation where
testing the behaviour beats naming a suspect.
