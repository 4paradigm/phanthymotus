"""A policy that is not a policy — a sine wave, for proving the path.

This exists so the command path can be tested before any model is involved:
canvas wiring, descriptor negotiation, message construction, the driver's check
chain, the watchdog, and what happens when the link is cut. Every one of those
can be wrong in a way that is dangerous, and none of them needs a VLA to be
wrong.

It is also the only provider that keeps working with no network, no GPU, no
weights and no torch, which makes it the regression harness for everything
above: change ControlSink, change the negotiation, bump the schema version, and
this is what re-proves the path on a laptop.

Two deliberate properties:

- **It is bounded by construction.** The trajectory is a fraction of the
  *declared* half-span around the midpoint of each joint's range, so it cannot
  leave the limits even before the driver's own checks. A test signal that can
  reach a joint limit is a test signal that will, at three in the morning.

- **The amplitude default is small and the frequency is slow.** First run of
  anything on a real arm is at reduced speed with a person watching; a default
  that assumes otherwise is a default that gets used.
"""

from __future__ import annotations

import math


class MockProvider:
    """Sine trajectory shaped to whatever action space it is told about.

    Args:
        descriptor: the downstream driver's action space (see
            phanthymotus-driver/README_dev.md § "Continuous Control"). The mock
            adapts to it rather than declaring its own, which is what makes it
            usable against any arm without configuration.
        amplitude: fraction of each joint's half-range, 0-1.
        period_s: seconds per full cycle.
        chunk_size: how many future steps to emit per inference, so the chunk
            path is exercised rather than only the single-step one.
    """

    def __init__(self, descriptor: dict, *, amplitude: float = 0.05,
                 period_s: float = 8.0, chunk_size: int = 10,
                 control_hz: float = 30.0):
        if not 0 < amplitude <= 1:
            raise ValueError("amplitude must be in (0, 1]")
        if period_s <= 0 or control_hz <= 0 or chunk_size < 1:
            raise ValueError("period_s, control_hz and chunk_size must be positive")

        limits = (descriptor or {}).get("limits") or {}
        self._lower = list(limits.get("lower") or [])
        self._upper = list(limits.get("upper") or [])
        self._dof = int((descriptor or {}).get("dof") or len(self._lower))
        if self._dof <= 0:
            raise ValueError("mock provider needs a descriptor with a dof")
        if len(self._lower) != self._dof or len(self._upper) != self._dof:
            raise ValueError("descriptor limits do not match its dof")

        self._amplitude = amplitude
        self._period_s = period_s
        self._chunk = chunk_size
        self._hz = control_hz
        self._phase = 0.0

    # ── provider protocol ────────────────────────────────────────────────────

    def capabilities(self) -> dict:
        return {
            "model": f"mock-sine@{self._period_s:g}s",
            "action_dim": self._dof,
            "chunk_size": self._chunk,
            "control_hz": self._hz,
            # Nothing is read: a sine wave is open-loop by definition, and
            # saying otherwise would make the card demand a camera it does not
            # use, and hide a wiring mistake behind a stream that still moves.
            "needs_state": False,
            "n_cameras": 0,
            "image_size": 0,
            "supports_rtc": False,
        }

    def infer(self, observation=None, inference_delay: int = 0) -> list:
        """One chunk of `chunk_size` future joint targets.

        `observation` is ignored and `inference_delay` only advances the phase,
        which is the honest behaviour for an open-loop signal: a mock that
        pretended to react would make a broken observation path look healthy.
        """
        step = 1.0 / self._hz
        start = self._phase + inference_delay * step
        chunk = []
        for i in range(self._chunk):
            chunk.append(self._sample(start + i * step))
        self._phase = start + self._chunk * step
        return chunk

    def health(self) -> bool:
        return True

    def close(self) -> None:
        return None

    # ── the signal ───────────────────────────────────────────────────────────

    def _sample(self, t: float) -> list:
        wave = math.sin(2 * math.pi * t / self._period_s)
        out = []
        for lo, hi in zip(self._lower, self._upper):
            mid = (lo + hi) / 2.0
            half = (hi - lo) / 2.0
            out.append(mid + wave * half * self._amplitude)
        return out


def PROVIDER(descriptor: dict, config: dict | None = None,
             on_status=None) -> MockProvider:
    """Factory. `config` carries the card's provider-specific settings.

    `on_status` is accepted and ignored: a sine wave has no weights, so there is
    no download to report. It is in the signature because the card passes it to
    whichever provider it built, without asking which one that is.
    """
    del on_status
    config = config or {}
    return MockProvider(
        descriptor,
        amplitude=float(config.get("amplitude", 0.05)),
        period_s=float(config.get("period_s", 8.0)),
        chunk_size=int(config.get("chunk_size", 10)),
        control_hz=float(config.get("control_hz", 30.0)),
    )


# The discovery check looks for the four methods on whatever PROVIDER is; a
# factory function has none of them, so they are advertised here.
for _name in ("capabilities", "infer", "health", "close"):
    setattr(PROVIDER, _name, getattr(MockProvider, _name))

# No model to name. A sine wave has no weights and no server, so the card hides
# the model field entirely rather than offering a name that means nothing here.
PROVIDER.MODEL_NAMES = None
