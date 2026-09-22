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

Three deliberate properties:

- **It is bounded by construction.** The trajectory is a fraction of the
  distance from its rest pose to the *declared* limit, so it cannot leave the
  limits even before the driver's own checks. A test signal that can reach a
  joint limit is a test signal that will, at three in the morning.

- **It starts and ends at rest, and never jumps.** The wave is `(1-cos)/2`, so
  it leaves the rest pose at zero velocity and returns to it every cycle.

- **The amplitude default is small and the frequency is slow.** First run of
  anything on a real arm is at reduced speed with a person watching; a default
  that assumes otherwise is a default that gets used.

── Why "rest pose" and not the midpoint of the range ─────────────────────────

It used to centre the sine on the midpoint of each joint's declared range, on
the reasoning that the midpoint is the point furthest from both limits. That is
true and it is not the point. A joint's range is not symmetric about its rest
position, so the midpoint is a *pose*, and on a real robot it was not a pose
anyone had asked for. Measured on Tianyi (26 dof):

    left_shoulder_roll   [-0.262, 2.618] rad   midpoint  +1.178 rad = +67.5°
    left_elbow_pitch     [-2.618, 0.262] rad   midpoint  -1.178 rad
    hand_* (12 fingers)  [0, 1] normalized     midpoint   0.5  (half closed)

So the very first command — before the sine had moved at all — asked for both
arms raised 67° to the side with the hands half closed, and the driver ramped
there at `max_delta_per_step` over about a second. What looked like "the mock's
amplitude is too large" was the *centre*, and no amount of reducing `amplitude`
would have fixed it.

The rest pose is zero, clamped into the range: every rotary joint on these arms
straddles zero (arms hanging), and a normalized 0-1 finger reads 0 as open. So
zero is both the natural home and, for the fingers, one end of travel — which
is why the wave is unipolar rather than a sine: it can start *on* a bound.
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
        amplitude: fraction of the travel available from rest, 0-1. Either one
            number for every joint, or a dict keyed by group name — because the
            same fraction does not read the same on every group. On Tianyi 5%
            of an arm joint's reach is 7.5° and plainly visible, while 5% of a
            0-1 finger is 0.05 and looks like the hand is not being driven at
            all. A group the dict does not mention falls back to `default`.
        period_s: seconds per full cycle.
        chunk_size: how many future steps to emit per inference, so the chunk
            path is exercised rather than only the single-step one.
    """

    DEFAULT_AMPLITUDE = 0.05

    # 末端姿态摆动的满幅。见 `_sample` —— 四元数必须整体生成，所以这里是**角度**
    # 而不是四个分量各自的幅度。
    MOCK_MAX_ANGLE = math.pi / 4

    # A normalized 0-1 axis is a gripper, not a joint: 5% of a grip is a twitch
    # nobody can see, and "the hand is not being driven" is exactly how it was
    # reported. Half of the grip is unmistakable and still returns to open every
    # cycle. Keyed on the unit rather than on Tianyi's group names so it is the
    # default on any robot whose descriptor says `normalized`.
    DEFAULT_AMPLITUDE_BY_UNIT = {"normalized": 0.5}

    def __init__(self, descriptor: dict, *, amplitude=DEFAULT_AMPLITUDE,
                 period_s: float = 8.0, chunk_size: int = 10,
                 control_hz: float = 30.0):
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

        self._amplitude = _per_joint_amplitude(
            amplitude, (descriptor or {}).get("groups"), self._dof)
        self._period_s = period_s
        self._chunk = chunk_size
        self._hz = control_hz
        self._phase = 0.0
        self._mode = str((descriptor or {}).get("mode") or "joint_position")
        # 末端位姿段里四元数的起始下标（每段 7 维：xyz + qx,qy,qz,qw）。
        # 逐分量的正弦对四元数是**错的**：它生成的不是单位四元数，而
        # `ControlSink._check_contract` 会把每一条都拒掉——于是通路依然验不成，
        # 只是失败挪后了一道。见 `_sample`。
        self._quat_offsets = []
        for group in (descriptor or {}).get("groups") or []:
            if not isinstance(group, dict) or group.get("mode") != "eef_pose":
                continue
            offset, count = int(group.get("offset", 0)), int(group.get("count", 0))
            for start in range(offset, offset + count, 7):
                self._quat_offsets.append(start + 3)

        # Precomputed rather than derived per sample: this runs at control rate,
        # and the two numbers depend only on the descriptor.
        self._rest, self._reach = [], []
        for lo, hi in zip(self._lower, self._upper):
            rest = min(max(0.0, lo), hi)
            # Toward whichever side has more room, so a joint that rests on one
            # of its bounds still moves, and moves inward.
            up, down = hi - rest, rest - lo
            self._rest.append(rest)
            self._reach.append(up if up >= down else -down)

    # ── provider protocol ────────────────────────────────────────────────────

    def capabilities(self) -> dict:
        return {
            "model": f"mock-sine@{self._period_s:g}s",
            # **跟着下游 descriptor 走，不写死。** 这个信号本来就是照着下游的
            # limits 生成的 —— 它没有自己的动作空间，只有下游那个。
            #
            # 写死 `joint_position` 的后果是 mock **验不了任何非关节空间的卡片**，
            # 而"验证通路"是它存在的全部理由。真机实测 2026-09-21（G1）：接到一张
            # `eef_pose` 卡片上，协商当场拒掉——
            #
            #   模型 mock-sine@8s，下游 eef_pose/17 关节：模型输出
            #   'joint_position' 空间的动作，下游接受 'eef_pose'
            #
            # 拒得对，但那说明这条通路根本没法用 mock 验。
            "control_mode": self._mode,
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
        # Unipolar: 0 at rest, 1 at the far end of the swing, back to 0. A plain
        # sine would need a centre with room on both sides, which a finger
        # resting at 0.0 does not have.
        wave = (1.0 - math.cos(2 * math.pi * t / self._period_s)) / 2.0
        values = [rest + wave * reach * amp
                  for rest, reach, amp in zip(self._rest, self._reach, self._amplitude)]
        for offset in self._quat_offsets:
            # 绕 z 转 `wave × amp × MOCK_MAX_ANGLE`，从单位四元数出发。**必须整体
            # 生成**，不能逐分量摆：四元数的四个分量各自是一条正弦的话，合起来不是
            # 单位长度，sink 会在契约检查处拒掉每一条。
            angle = wave * self._amplitude[offset] * self.MOCK_MAX_ANGLE
            values[offset:offset + 4] = [0.0, 0.0, math.sin(angle / 2),
                                         math.cos(angle / 2)]
        return values


def _per_joint_amplitude(amplitude, groups, dof: int) -> list:
    """One fraction per joint, from a number or a dict.

    A dict key is either a **group name** (`hand_l`) or a group's **unit**
    (`normalized`), with `default` for the rest; a name beats a unit. Units are
    in the keys because that is what makes a default portable — group names are
    invented per robot, whereas `rad` and `normalized` come from the descriptor
    spec and mean the same thing on every arm.

    A group whose `offset`/`count` do not describe a real slice is skipped
    rather than raised on: this is a test signal, and refusing to start because
    one group in a descriptor is odd would take away the tool people reach for
    *when* a descriptor is odd.
    """
    def _fraction(value, where: str) -> float:
        value = float(value)
        if not 0 < value <= 1:
            raise ValueError(f"amplitude{where} must be in (0, 1], got {value}")
        return value

    if not isinstance(amplitude, dict):
        return [_fraction(amplitude, "")] * dof

    default = _fraction(amplitude.get("default", MockProvider.DEFAULT_AMPLITUDE),
                        "['default']")
    out = [default] * dof
    for group in groups or []:
        if not isinstance(group, dict):
            continue
        offset, count = group.get("offset"), group.get("count")
        if not isinstance(offset, int) or not isinstance(count, int):
            continue
        for key in (group.get("name"), group.get("unit")):
            if key in amplitude and key != "default":
                value = _fraction(amplitude[key], f"[{key!r}]")
                for i in range(offset, min(offset + count, dof)):
                    out[i] = value
                break
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
    # The default is the dict, not the scalar: a bare 0.05 would mean the same
    # invisible twitch on a gripper that got this reported in the first place.
    # An operator who writes a scalar has overridden that on purpose.
    amplitude = config.get("amplitude")
    if amplitude is None:
        amplitude = {"default": MockProvider.DEFAULT_AMPLITUDE,
                     **MockProvider.DEFAULT_AMPLITUDE_BY_UNIT}
    return MockProvider(
        descriptor,
        # Not coerced with float(): a dict here is the per-group form.
        amplitude=amplitude if isinstance(amplitude, dict) else float(amplitude),
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
