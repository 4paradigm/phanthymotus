"""The VLA card, its provider discovery, and the wire format it produces.

No ROS, no model, no robot: `next_command` is deliberately separate from the
timer that publishes it, so the whole of the message construction — sequence
numbers, the two timestamps, chunk indices — is testable here, and `_tick` is
only this plus a publish.

What these cover, in the order the card does them:

  discovery      a provider file is the whole of adding a provider; a broken
                 one is reported, not raised
  negotiation    the card refuses to start on a mismatch rather than failing
                 one command at a time at 30 Hz
  the message    obs_stamp_ms is the observation's age, not the command's —
                 passing stamp_ms for both looks right in every test that does
                 not involve latency and disables the receiver's staleness check
  the mock       bounded by construction, so a test signal cannot reach a limit

Run: cd actucore && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_vla_card.py -q
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plugins.vla import VLAPlugin  # noqa: E402
from plugins.vla.plugin import FORMATS, Observation  # noqa: E402
from plugins.vla import negotiate  # noqa: E402
from plugins.vla.message import build as build_message  # noqa: E402
from plugins.vla.providers import discover, REQUIRED  # noqa: E402
from plugins.vla.providers.mock import PROVIDER as MOCK  # noqa: E402


DESCRIPTOR = {
    "control_interface": "motus.control/1",
    "mode": "joint_position",
    "dof": 7,
    "joint_names": [f"joint{i}" for i in range(1, 8)],
    "units": {"angle": "rad"},
    "limits": {"lower": [-2.0] * 7, "upper": [2.0] * 7},
    "rate": {"max_hz": 100, "expected_hz": 30, "watchdog_ms": 200},
    "force_torque": None,
}



def make_card(**cfg):
    base = {"provider": "mock", "amplitude": 0.1, "period_s": 4.0, "chunk_size": 5}
    base.update(cfg)
    return VLAPlugin(base, executor=None)


def _started_card(**cfg):
    """A card in the state `start` leaves it in, without a ROS node.

    `_start` would build a publisher and a timer; these tests are about what is
    emitted and when, which `next_command` and `_tick` answer on their own.
    """
    card = make_card(**cfg)
    card._descriptor = DESCRIPTOR
    card._provider = MOCK(DESCRIPTOR, {"chunk_size": 5})
    card._ttl_ms = 100
    card._running = True
    return card


# ── provider discovery ───────────────────────────────────────────────────────

def test_mock_is_discovered_without_being_named_anywhere():
    assert "mock" in discover()


def test_the_schema_enum_is_built_from_what_was_found():
    """Adding a provider file must not require editing the card or its schema."""
    card = make_card()
    schema = card.get_tools()[0]["configSchema"]["properties"]["provider"]
    assert schema["enum"] == sorted(discover())


def test_every_discovered_provider_has_the_whole_protocol():
    for name, factory in discover().items():
        for method in REQUIRED:
            assert hasattr(factory, method), f"{name} is missing {method}"


def test_a_broken_provider_is_reported_not_raised(tmp_path, monkeypatch):
    """One backend's missing dependency must not take the card down with it."""
    import plugins.vla.providers as providers

    # A real module whose import genuinely fails, the way one missing torch
    # would. Not underscore-prefixed: discovery skips those, which would make
    # this pass for the wrong reason.
    pkg_dir = pathlib.Path(providers.__path__[0])
    broken = pkg_dir / "tmpbroken.py"
    broken.write_text("import definitely_not_a_real_module\n")
    try:
        found = discover()
        assert "tmpbroken" not in found
        assert "mock" in found                       # unaffected
        assert "tmpbroken" in discover.errors
    finally:
        broken.unlink()


# ── negotiation ──────────────────────────────────────────────────────────────

def test_matching_capabilities_pass():
    assert negotiate.check(MOCK(DESCRIPTOR).capabilities(), DESCRIPTOR) == []


def test_action_dim_mismatch_names_both_numbers():
    """"shape mismatch" sends somebody to read code; this sends them to a wire."""
    problems = negotiate.check(_caps(action_dim=32), DESCRIPTOR)
    assert problems
    assert "32" in problems[0] and "7" in problems[0]


def test_a_downstream_that_is_not_a_control_card_is_caught_first():
    problems = negotiate.check({"action_dim": 7}, {"control_interface": None})
    assert len(problems) == 1
    assert "motus.control/1" in problems[0]


def test_a_model_faster_than_the_hardware_is_refused():
    problems = negotiate.check(_caps(control_hz=500), DESCRIPTOR)
    assert any("500" in p for p in problems)


def test_every_problem_is_reported_at_once():
    """An operator fixing a canvas should see the whole disagreement."""
    problems = negotiate.check(_caps(action_dim=32, control_hz=500), DESCRIPTOR)
    assert len(problems) == 2


def test_rate_is_clamped_by_everyone_with_a_say():
    # The model's own rate wins over a faster request: publishing faster than
    # the actions were computed for changes what each one means.
    assert negotiate.effective_rate({"control_hz": 10}, DESCRIPTOR, 60) == 10
    # And the driver's ceiling wins over everything.
    slow = {**DESCRIPTOR, "rate": {**DESCRIPTOR["rate"], "max_hz": 5}}
    assert negotiate.effective_rate({"control_hz": 50}, slow, 50) == 5


def test_ttl_never_outlives_the_receivers_watchdog():
    """A generous ttl removes the protection while appearing to provide it."""
    assert negotiate.ttl_ms(1.0, DESCRIPTOR) == 200        # capped by watchdog_ms
    assert negotiate.ttl_ms(30.0, DESCRIPTOR) >= 50        # floored


# ── the message ──────────────────────────────────────────────────────────────

def test_message_carries_every_field_the_receiver_checks():
    message = build_message(seq=1, values=[0.0] * 7, mode="joint_position", dof=7,
                            source="test", stamp_ms=1000, obs_stamp_ms=900,
                            ttl_ms=100)
    for field in ("schema", "seq", "stamp_ms", "obs_stamp_ms", "ttl_ms",
                  "source", "priority", "mode", "dof", "values"):
        assert field in message
    assert message["schema"] == "motus.control/1"


def test_the_two_timestamps_are_not_the_same_number():
    """The characteristic failure of remote inference is only visible in obs_stamp."""
    card = make_card()
    card._descriptor = DESCRIPTOR
    card._provider = MOCK(DESCRIPTOR, {"chunk_size": 5})
    card._ttl_ms = 100

    first = card.next_command()
    second = card.next_command()

    # Both came from one inference, so they share an observation timestamp while
    # their own stamps advance — that is what makes an ageing chunk visible.
    assert first["obs_stamp_ms"] == second["obs_stamp_ms"]
    assert second["stamp_ms"] >= first["stamp_ms"]


def test_sequence_numbers_are_strictly_increasing_across_chunks():
    card = make_card()
    card._descriptor = DESCRIPTOR
    card._provider = MOCK(DESCRIPTOR, {"chunk_size": 3})
    card._ttl_ms = 100

    seqs = [card.next_command()["seq"] for _ in range(8)]      # spans 3 chunks

    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)


def test_chunk_indices_walk_the_chunk_then_refill():
    card = make_card()
    card._descriptor = DESCRIPTOR
    card._provider = MOCK(DESCRIPTOR, {"chunk_size": 3})
    card._ttl_ms = 100

    indices = [card.next_command()["chunk"]["index"] for _ in range(7)]

    assert indices == [0, 1, 2, 0, 1, 2, 0]


def test_an_empty_chunk_raises_rather_than_publishing_nothing_silently():
    class Empty:
        def capabilities(self): return _caps()
        def infer(self, obs=None, inference_delay=0): return []
        def health(self): return True
        def close(self): return None

    card = make_card()
    card._descriptor = DESCRIPTOR
    card._provider = Empty()

    with pytest.raises(RuntimeError):
        card.next_command()


# ── start-time refusals ──────────────────────────────────────────────────────

def test_start_without_a_downstream_descriptor_is_refused():
    """Without a wired consumer there is no action space to target."""
    result = make_card().dispatch("start", {"action": "start"})
    assert result["state"] == "error"
    assert "control/*" in result["message"]


def test_start_refuses_a_mismatched_model_instead_of_failing_per_command():
    class Wide:
        def capabilities(self): return {"action_dim": 32, "control_hz": 30}
        def infer(self, obs=None, inference_delay=0): return [[0.0] * 32]
        def health(self): return True
        def close(self): self.closed = True

    card = make_card(provider="wide")
    import plugins.vla.plugin as plugin_mod
    original = plugin_mod.discover
    # The factory takes on_status as of the download-progress work: the card
    # hands it to whichever provider it built, without asking which one.
    plugin_mod.discover = lambda: {"wide": lambda d, c, on_status=None: Wide()}
    plugin_mod.discover.errors = {}
    try:
        result = card.dispatch("start", {"action": "start",
                                         "control_interface": DESCRIPTOR})
    finally:
        plugin_mod.discover = original

    assert result["state"] == "error"
    assert "32" in result["message"]


def test_an_unknown_provider_lists_what_is_available():
    card = make_card(provider="nope")
    result = card.dispatch("start", {"action": "start",
                                     "control_interface": DESCRIPTOR})
    assert result["state"] == "error"
    assert "mock" in result["message"]


def test_a_mode_with_no_topic_format_is_refused():
    odd = {**DESCRIPTOR, "mode": "hyperdrive"}
    result = make_card().dispatch("start", {"action": "start",
                                            "control_interface": odd})
    assert result["state"] == "error"
    assert "hyperdrive" in result["message"]


def test_the_card_declares_a_resource_and_no_completion():
    schema = make_card().get_tools()[0]["inputSchema"]
    # A policy runs until stopped; an ACP pending held for its lifetime would
    # block every other actuator behind the barrier.
    assert "x-completion" not in schema
    assert schema["x-resource"] == ["arm"]          # pre-negotiation placeholder
    # An interrupt throws the planned chunk away rather than tearing the card
    # down: the driver's watchdog holds the arm, and a card that stopped itself
    # could not be started again by whoever interrupted it.
    assert schema["x-hooks"]["on_interrupt_all"]["action"] == "interrupt"


def test_the_negotiated_groups_replace_the_configured_resource():
    """Only the downstream driver knows what it actually owns.

    A humanoid's action space is several channels — Tianyi's is both arms and
    both hands. A card still claiming one configured `arm` would let something
    else drive the hands while a policy was moving them.
    """
    card = make_card()
    card._descriptor = {
        **DESCRIPTOR,
        "groups": [
            {"name": "arm_l", "offset": 0, "count": 7, "resource": "arm_l"},
            {"name": "hand_l", "offset": 7, "count": 6, "resource": "hand_l"},
        ],
    }

    assert card.get_tools()[0]["inputSchema"]["x-resource"] == ["arm_l", "hand_l"]


def test_a_descriptor_without_groups_keeps_the_configured_resource():
    card = make_card(resource="waist")
    card._descriptor = DESCRIPTOR                   # no groups — a single-arm driver

    assert card.get_tools()[0]["inputSchema"]["x-resource"] == ["waist"]


def test_info_answers_before_anything_has_started():
    """Agent Core probes liveness with info(); a card that cannot answer is offline."""
    info = make_card().dispatch("info", {"action": "info"})
    assert info["state"] == "idle"
    assert "mock" in info["providers_available"]


def test_action_enum_contains_info():
    enum = make_card().get_tools()[0]["inputSchema"]["properties"]["action"]["enum"]
    assert "info" in enum


def test_prefix_has_no_underscore():
    """dispatch routes on partition('_'); a prefix with one never matches."""
    assert "_" not in VLAPlugin.PREFIX


# ── the mock signal ──────────────────────────────────────────────────────────

def test_the_sine_cannot_leave_the_declared_limits():
    """A test signal that can reach a joint limit is one that will, at 3am."""
    provider = MOCK(DESCRIPTOR, {"amplitude": 1.0, "chunk_size": 64,
                                 "period_s": 1.0, "control_hz": 30})
    lower, upper = DESCRIPTOR["limits"]["lower"], DESCRIPTOR["limits"]["upper"]

    for _ in range(20):
        for values in provider.infer():
            for v, lo, hi in zip(values, lower, upper):
                assert lo <= v <= hi


def test_the_default_amplitude_is_small():
    provider = MOCK(DESCRIPTOR)
    extremes = []
    for _ in range(40):
        for values in provider.infer():
            extremes.append(abs(values[0]))
    # 5% of a half-span of 2.0
    assert max(extremes) <= 0.11


def test_the_mock_adapts_to_whatever_arm_it_is_given():
    six = {**DESCRIPTOR, "dof": 6,
           "limits": {"lower": [-1.0] * 6, "upper": [1.0] * 6}}
    assert MOCK(six).capabilities()["action_dim"] == 6
    assert len(MOCK(six).infer()[0]) == 6


def test_a_descriptor_whose_limits_contradict_its_dof_is_refused():
    bad = {**DESCRIPTOR, "dof": 7,
           "limits": {"lower": [-1.0] * 6, "upper": [1.0] * 6}}
    with pytest.raises(ValueError):
        MOCK(bad)


def test_the_mock_does_not_claim_to_read_anything():
    """Claiming otherwise would hide a broken observation path behind motion."""
    assert MOCK(DESCRIPTOR).capabilities()["needs_state"] is False
    assert MOCK(DESCRIPTOR).capabilities()["n_cameras"] == 0


# ── the mock on an arm whose range is not symmetric ──────────────────────────
#
# DESCRIPTOR above is [-2, 2] on every joint, which is the one shape that hides
# the bug these cover: with a symmetric range the midpoint *is* the rest pose,
# so centring on either gives the same numbers. Real arms are not symmetric.
# The numbers below are Tianyi's, read off its servo card.

ASYMMETRIC = {
    "control_interface": "motus.control/1",
    "mode": "joint_position",
    "dof": 4,
    "units": {"angle": "rad", "normalized": "0-1"},
    "limits": {
        #        shoulder_roll   elbow_pitch   finger  finger
        "lower": [-0.2617993878, -2.6179938780, 0.0,   0.0],
        "upper": [2.6179938780, 0.2617993878, 1.0,   1.0],
    },
    "groups": [
        {"name": "arm_l", "offset": 0, "count": 2, "unit": "rad", "resource": "arm_l"},
        {"name": "hand_l", "offset": 2, "count": 2, "unit": "normalized",
         "resource": "hand_l"},
    ],
    "rate": {"max_hz": 50, "expected_hz": 30, "watchdog_ms": 200},
}


def test_the_first_command_is_the_rest_pose_not_the_middle_of_the_range():
    """The bug as it was reported: "手臂抬得太高".

    Centring on the midpoint made the *first* command — before the wave had
    moved at all — ask for shoulder_roll +1.178 rad (+67.5°) and the fingers
    half closed. The driver then ramped there at max_delta_per_step, which is
    what an observer saw as the arms lifting. No value of `amplitude` fixed it,
    because the offending number was the centre.
    """
    first = MOCK(ASYMMETRIC, {"chunk_size": 1}).infer()[0]
    assert first == [0.0, 0.0, 0.0, 0.0]
    midpoints = [(lo + hi) / 2 for lo, hi in zip(ASYMMETRIC["limits"]["lower"],
                                                 ASYMMETRIC["limits"]["upper"])]
    assert midpoints[0] > 1.1      # the pose it used to start from
    assert first[0] != pytest.approx(midpoints[0])


def test_the_signal_returns_to_rest_every_cycle():
    """`(1-cos)/2`, not a sine: it leaves rest at zero velocity and comes back.

    A signal that ends a cycle somewhere other than where it started drifts,
    and a drifting test signal on a real arm is how a limit gets reached.
    """
    hz, period = 30.0, 4.0
    provider = MOCK(ASYMMETRIC, {"chunk_size": int(hz * period) + 1,
                                 "period_s": period, "control_hz": hz})
    chunk = provider.infer()
    assert chunk[0] == [0.0] * 4
    assert chunk[-1] == pytest.approx([0.0] * 4, abs=1e-9)
    assert max(step[0] for step in chunk) > 0      # it did move in between


def test_a_joint_resting_on_its_own_bound_still_moves_and_moves_inward():
    """A finger rests at 0.0, which is one end of its travel.

    A symmetric sine around that point would have to leave the range to move at
    all, so it would either clip to a constant or violate the limit.
    """
    chunk = MOCK(ASYMMETRIC, {"chunk_size": 40, "period_s": 1.0,
                              "control_hz": 30}).infer()
    finger = [step[2] for step in chunk]
    assert max(finger) > 0.1
    assert min(finger) >= 0.0


def test_the_signal_stays_inside_an_asymmetric_range_at_full_amplitude():
    lower, upper = ASYMMETRIC["limits"]["lower"], ASYMMETRIC["limits"]["upper"]
    provider = MOCK(ASYMMETRIC, {"amplitude": 1.0, "chunk_size": 64,
                                 "period_s": 1.0, "control_hz": 30})
    for _ in range(20):
        for values in provider.infer():
            for v, lo, hi in zip(values, lower, upper):
                assert lo <= v <= hi


def test_a_gripper_gets_a_visible_default_while_an_arm_stays_small():
    """The other half of the report: "没有驱动手指".

    The fingers *were* being driven — 5% of a 0-1 grip, around a half-closed
    centre. Both halves of that are invisible. The fraction is keyed on the
    group's unit so this default holds on any robot whose descriptor says
    `normalized`, not just the one it was reported on.
    """
    chunk = MOCK(ASYMMETRIC, {"chunk_size": 60, "period_s": 2.0,
                              "control_hz": 30}).infer()
    grip = max(step[2] for step in chunk)
    shoulder = max(step[0] for step in chunk)
    assert grip == pytest.approx(0.5, abs=1e-6)          # half the grip: visible
    assert shoulder == pytest.approx(0.05 * 2.618, rel=1e-3)   # 7.5°, still small


def test_amplitude_can_be_set_per_group_and_a_name_beats_a_unit():
    amplitude = {"default": 0.02, "normalized": 0.5, "hand_l": 0.1}
    chunk = MOCK(ASYMMETRIC, {"amplitude": amplitude, "chunk_size": 60,
                              "period_s": 2.0, "control_hz": 30}).infer()
    assert max(step[2] for step in chunk) == pytest.approx(0.1, abs=1e-6)
    assert max(step[0] for step in chunk) == pytest.approx(0.02 * 2.618, rel=1e-3)


def test_a_scalar_amplitude_still_applies_to_every_joint():
    """An operator who writes one number has overridden the per-unit default."""
    chunk = MOCK(ASYMMETRIC, {"amplitude": 0.25, "chunk_size": 60,
                              "period_s": 2.0, "control_hz": 30}).infer()
    assert max(step[2] for step in chunk) == pytest.approx(0.25, abs=1e-6)


def test_an_out_of_range_amplitude_is_refused():
    for bad in (0, 1.5, -0.1):
        with pytest.raises(ValueError):
            MOCK(ASYMMETRIC, {"amplitude": bad})
    with pytest.raises(ValueError):
        MOCK(ASYMMETRIC, {"amplitude": {"default": 0.05, "normalized": 2.0}})


def test_a_malformed_group_does_not_stop_the_test_signal():
    """The mock is what people reach for *when* a descriptor looks wrong."""
    odd = {**ASYMMETRIC, "groups": ["arm_l", {"name": "hand_l"}, None]}
    chunk = MOCK(odd, {"chunk_size": 10}).infer()
    assert len(chunk[0]) == 4


# ── the config form an operator actually sees ────────────────────────────────

def _properties(**cfg):
    return make_card(**cfg).get_tools()[0]["configSchema"]["properties"]


def test_remote_only_fields_are_hidden_for_a_local_provider():
    """A filled-in endpoint beside `provider: smolvla` reads as configured.

    It is ignored, and nothing says so — which is an afternoon of an operator's
    time. `x-show-when` is how every other card in this project expresses that.
    """
    props = _properties()

    for field in ("endpoint", "api_key", "timeout_ms"):
        assert props[field]["x-show-when"] == {"provider": ["vla_cloud"]}, field


def test_the_cloud_fields_are_ordered_the_way_they_are_filled_in():
    """表单顺序就是 schema 顺序（sidebar.js 遍历 Object.entries）。

    先有服务器，才有它认得的 key，才谈得上问它有哪些模型名。`cloud_model_name` 一度
    排在最前，于是表单第一个问的是一个只有服务器知道答案的名字 —— 一个纯排序问题，
    但它是操作员第一眼看到的东西。
    """
    keys = list(_properties())
    order = [keys.index(f) for f in ("endpoint", "api_key", "cloud_model_name")]
    assert order == sorted(order), f"云端三项的顺序不对：{keys}"


def test_unnorm_key_is_settable_from_the_form():
    """修了 provider 却没修配置表面，等于把卡片变成一个填不了的错误。

    The provider refuses to load a checkpoint whose normalisation statistics are
    grouped per dataset unless one is chosen — that refusal is the fix for a card
    that used to emit its policy's normalised space (≈ ±1) into a descriptor that
    reads degrees. But the form renders exactly this schema, so without a field
    here an operator is told to set something they have nowhere to set.

    Free text, not `enum`: the groups live inside the checkpoint and are only
    known once it is read. An empty <select> would be worse than a box.
    """
    props = _properties()
    assert "unnorm_key" in props, "表单里没有 unnorm_key，provider 的拒绝就无解了"
    assert props["unnorm_key"]["type"] == "string"
    assert "enum" not in props["unnorm_key"]


def test_unnorm_key_is_hidden_for_the_cloud_provider():
    """云端那侧由服务端自己选，机器人这边填了也不会被用上。"""
    condition = _properties()["unnorm_key"]["x-show-when"]["provider"]
    assert "vla_cloud" not in condition


def test_every_show_when_value_is_a_string_or_a_list_of_them():
    """A boolean here renders the field permanently hidden, silently.

    The frontend compares `actual === condVal`, or `condVal.includes(actual)`
    for a list, against a value that arrives from the form as a string — so a
    JSON boolean never matches. agent-core has a whole test
    (test_config_schema_show_when.py) about the afternoon that cost someone.
    """
    for key, definition in _properties().items():
        condition = definition.get("x-show-when")
        if not condition:
            continue
        for value in condition.values():
            values = value if isinstance(value, list) else [value]
            assert all(isinstance(v, str) for v in values), key


def test_the_model_field_offers_the_staged_checkpoints():
    props = _properties(models={"smolvla_base": {}, "smolvla_tianyi": {}})

    assert props["model_name"]["enum"] == ["smolvla_base", "smolvla_tianyi"]
    assert props["model_name"]["default"] == "smolvla_base"


def test_the_model_field_stays_free_text_with_nothing_staged():
    """vla_cloud's model names live on the server; they are not ours to list."""
    props = _properties()

    assert "enum" not in props["model_name"]
    assert props["model_name"]["type"] == "string"


def test_the_configured_model_is_the_default_shown():
    props = _properties(model_name="smolvla_tianyi",
                        models={"smolvla_base": {}, "smolvla_tianyi": {}})
    assert props["model_name"]["default"] == "smolvla_tianyi"


def test_mock_is_not_asked_for_a_model_name():
    """A sine wave has no weights and no server; offering a name is noise.

    Worse than noise, in fact: the name shown would be a real checkpoint, which
    reads as "this is what is running".
    """
    props = _properties(models={"smolvla_base": {}})

    shown_for = props["model_name"]["x-show-when"]["provider"]
    assert "mock" not in shown_for
    assert "smolvla" in shown_for


def test_a_remote_provider_gets_a_free_text_model_field():
    """`enum` renders as a <select>, so one shared field would lock vla_cloud
    to the locally staged names and leave no way to type the server's."""
    props = _properties(models={"smolvla_base": {}})

    assert "enum" not in props["cloud_model_name"]
    assert props["cloud_model_name"]["x-show-when"]["provider"] == ["vla_cloud"]
    assert "vla_cloud" not in props["model_name"]["x-show-when"]["provider"]


def test_the_two_model_fields_never_show_together():
    props = _properties(models={"smolvla_base": {}})

    local = set(props["model_name"]["x-show-when"]["provider"])
    remote = set(props["cloud_model_name"]["x-show-when"]["provider"])
    assert not (local & remote)


def test_the_field_lists_come_from_what_providers_declare():
    """Adding a provider stays "add a file" — the card enumerates nothing."""
    from plugins.vla.providers import discover

    props = _properties()
    declared_staged = sorted(n for n, f in discover().items()
                             if getattr(f, "MODEL_NAMES", None) == "staged")
    assert props["model_name"]["x-show-when"]["provider"] == declared_staged


# ── the levers over a running policy ─────────────────────────────────────────

def test_the_model_gets_the_levers_and_not_start_or_stop():
    """agent-core splits a tool into one LLM-callable function per
    `x-action-params` entry (mcp_client.py `_to_openai_schema`), so this list is
    the model's entire reach. `start` needs the downstream `control_interface`,
    which only agent-core can supply — a model that stopped this card could not
    start it again.
    """
    schema = make_card().get_tools()[0]["inputSchema"]

    assert set(schema["x-action-params"]) == {"execute", "pause", "interrupt",
                                              "resume"}
    # Still dispatchable by the canvas, just not offered to the model.
    assert {"start", "stop"} <= set(schema["properties"]["action"]["enum"])


def test_pause_keeps_the_planned_chunk_and_interrupt_throws_it_away():
    """The whole reason both exist.

    "Hold on a second" and "no, that is wrong" are different instructions, and
    carrying out the second by replaying a plan made before the objection would
    be the wrong answer.
    """
    card = _started_card()
    card.next_command()                      # pull one command, leaving a chunk
    pending = len(card._chunk) - card._chunk_index
    assert pending > 0, "this test needs a provider that plans ahead"

    assert card.dispatch("vla", {"action": "pause"})["chunk_pending"] == pending

    card.dispatch("vla", {"action": "resume"})
    assert card.dispatch("vla", {"action": "interrupt"})["chunk_pending"] == 0
    assert card._chunk == []


def test_a_halted_card_emits_nothing():
    """Emitting nothing is how the halt reaches the arm: the driver's watchdog
    holds it. Commanding a stop here would fight that hold."""
    card = _started_card()
    card._publisher = _CountingPublisher()

    card.dispatch("vla", {"action": "pause"})
    card._tick()

    assert card._publisher.published == 0


def test_resuming_after_an_interrupt_infers_again():
    card = _started_card()
    card.next_command()
    card.dispatch("vla", {"action": "interrupt"})
    card.dispatch("vla", {"action": "resume"})

    message = card.next_command()            # refills, because the chunk is gone

    assert card._chunk_index == 1
    assert message["seq"] > 1


def test_a_halt_reports_its_own_state_not_running():
    """An operator reading "running" beside a motionless arm goes looking for a
    fault that is not there."""
    card = _started_card()

    card.dispatch("vla", {"action": "pause"})
    assert card.dispatch("vla", {"action": "info"})["state"] == "paused"

    card.dispatch("vla", {"action": "resume"})
    assert card.dispatch("vla", {"action": "info"})["state"] == "running"


def test_halting_a_card_that_is_not_running_is_not_an_error():
    card = make_card()
    for action in ("pause", "interrupt", "resume"):
        assert card.dispatch("vla", {"action": action})["state"] == "idle"


class _CountingPublisher:
    def __init__(self):
        self.published = 0

    def publish(self, _message):
        self.published += 1


def test_the_output_topic_is_declared_not_only_discovered():
    """agent-core's fallback chain ends at this declaration.

    A consumer's `input_topic` comes from the source's running `info()`, then
    the connection's persisted `fromTopic`, then the source card's declared
    `topic_out`. Naming no topic here leaves all three empty, so any failure to
    start this card also fails the card downstream — reported as
    "连线缺少 topic", which sends someone to check wiring that is correct.

    Nothing needs discovering: the topic is configuration and is known here.
    """
    port = make_card().get_tools()[0]["topic_out"][0]

    assert port["topic"] == "/actucore/vla/cmd"
    assert port["format"] == "control/joint"


def test_a_configured_topic_reaches_the_declaration():
    card = make_card(topic="/robot/arm/cmd")
    assert card.get_tools()[0]["topic_out"][0]["topic"] == "/robot/arm/cmd"


# ── observations ─────────────────────────────────────────────────────────────

def _stub_ros_messages():
    """The two message modules `_bind_inputs` imports when it actually subscribes.

    Stubbed rather than skipped: the thing under test is which topic is given
    which role, and that decision must hold on a machine with ROS as well as on
    this one.
    """
    import types as _types

    # `_sensor_qos()` 要 rclpy.qos。真的装了就用真的 —— 那样这条断言在有 ROS 的
    # 机器上比的是真枚举，而不是一个自己造的、怎么写都成立的替身。
    try:
        import rclpy.qos  # noqa: F401
    except ImportError:
        qos = _types.ModuleType("rclpy.qos")

        class _Enum:
            def __init__(self, name):
                self.name = name

            def __repr__(self):
                return self.name

        qos.ReliabilityPolicy = _types.SimpleNamespace(
            BEST_EFFORT=_Enum("BEST_EFFORT"), RELIABLE=_Enum("RELIABLE"))
        qos.HistoryPolicy = _types.SimpleNamespace(KEEP_LAST=_Enum("KEEP_LAST"))
        qos.DurabilityPolicy = _types.SimpleNamespace(VOLATILE=_Enum("VOLATILE"))

        class _Profile:
            def __init__(self, **kw):
                self.__dict__.update(kw)

        qos.QoSProfile = _Profile
        package = _types.ModuleType("rclpy")
        package.qos = qos
        sys.modules.setdefault("rclpy", package)
        sys.modules["rclpy.qos"] = qos

    for name, members in (("sensor_msgs", ("CompressedImage", "Image")),
                          ("std_msgs", ("String",))):
        package = _types.ModuleType(name)
        module = _types.ModuleType(f"{name}.msg")
        for member in members:
            setattr(module, member, type(member, (), {}))
        package.msg = module
        sys.modules[name] = package
        sys.modules[f"{name}.msg"] = module


class _Graph:
    """A node stub that answers the ROS graph query and records subscriptions."""

    def __init__(self, topics):
        self._topics = topics
        self.subscribed = []

    def get_topic_names_and_types(self):
        return list(self._topics.items())

    def create_subscription(self, message_type, topic, callback, qos):
        # **qos 也要记下来。** 它此前被丢掉，而那正是订阅端 QoS 错了却没被任何
        # 用例抓到的原因：这个假件把除了出问题的那一项之外的一切都记了下来。
        self.subscribed.append((message_type.__name__, topic, callback, qos))


def _caps(**over):
    """一份能通过协商的 capabilities。

    `control_mode` 在这里，是因为 `negotiate.check()` **缺它就拒** —— 维度相同不代表
    动作空间相同（一个 23 维的末端位姿模型和一张 23 维的关节卡片，数字完全吻合），
    而猜错的代价是机械臂走到错误的地方。不测动作空间的用例用这个构造器拿到一份合法
    的，才不会被那条检查抢先触发。

    **文件里曾经有两个同名的 `_caps`**，后定义的把前面那个遮蔽掉了，于是「补了字段
    却还是失败」。只留这一个。
    """
    base = {"control_mode": "joint_position", "action_dim": 7,
            "chunk_size": 10, "control_hz": 30,
            "n_cameras": 0, "needs_state": False}
    base.update(over)
    return base


def test_roles_come_from_the_message_type_not_the_topic_name():
    """agent-core passes topic names and no formats, so the name is all a card
    would otherwise have — and `/robot/state` is free to be anything at all.
    The graph gives a definite answer, which is how t800 and lynx_m20 already
    resolve topics in this project."""
    _stub_ros_messages()
    card = make_card()
    node = _Graph({"/cam": ["sensor_msgs/msg/CompressedImage"],
                   "/st": ["std_msgs/msg/String"]})

    binding, problem = card._bind_inputs(node, ["/st", "/cam"],
                                         _caps(n_cameras=1, needs_state=True))

    assert problem == ""
    assert binding["state"] == "/st"
    assert list(binding["images"].values()) == ["/cam"]


def test_observations_are_subscribed_best_effort():
    """**这条是一次真机失败换来的。**

    这个项目里每一个传感器发布者都是 BEST_EFFORT，而 rclpy 的默认 profile 是
    RELIABLE —— 一个 RELIABLE 的订阅者收不到 BEST_EFFORT 的发布者，DDS 直接不
    匹配。此前 `_bind_inputs` 传的是 `1`（展开成默认 profile），于是 vla_cloud 与
    smolvla 在**任何一台真机上都拿不到观测**：卡片报 running、error 空、一条指令
    都不发，而唯一的线索是 ROS stderr 里一行 "incompatible QoS"。

    同一个文件里的 `_open_publisher` 一直是显式 BEST_EFFORT，还写了注释 —— 两边
    不对称了很久没人发现，因为唯一能暴露它的地方（真 DDS 匹配）在用例里是假的。
    """
    from rclpy.qos import ReliabilityPolicy

    _stub_ros_messages()
    card = make_card()
    node = _Graph({"/cam": ["sensor_msgs/msg/CompressedImage"],
                   "/st": ["std_msgs/msg/String"]})

    card._bind_inputs(node, ["/cam", "/st"], _caps(n_cameras=1, needs_state=True))

    assert len(node.subscribed) == 2, "相机和状态都要订上"
    for _type, topic, _cb, qos in node.subscribed:
        assert qos.reliability == ReliabilityPolicy.BEST_EFFORT, topic
        assert qos.depth == 1, f"{topic}：排队的观测就是过期的观测"


def test_a_subscription_that_never_delivers_is_reported():
    """订阅建立成功而一条消息都不来，是 DDS 里的常态，不是异常。

    `_bind_inputs` 只能核对「话题连上了没」，核对不了「消息收到了没」。此前这三种
    情况——QoS 不兼容、发布者没在发、域不同——表现完全一样：running / error 空 /
    published 0。现在持续缺失会把**缺的是什么**写进 error。
    """
    card = make_card()
    card._running = True
    card._capabilities = _caps(n_cameras=2, needs_state=True)
    card._publisher = object()

    card._report_starvation(card._capabilities)
    assert card._info()["error"] == "", "刚启动就报错会把所有正常启动也误伤"

    card._starved_since -= card.STARVED_AFTER_S + 1
    card._report_starvation(card._capabilities)

    error = card._info()["error"]
    assert "图像 0/2 路" in error and "本体状态" in error
    assert "QoS" in error, "要说出最常见的那个原因，否则只是换个地方说「没收到」"


def test_the_published_format_is_the_one_drivers_declare():
    """**画布按严格字符串相等匹配端口。** 差一个字就连不上，而且是静默的。

    真机上的表现：`vla` 的输出口怎么都拖不到 `servo_eef` 的输入口，没有提示、
    没有日志。原因是这边发 `control/waypoint`、驱动收 `control/eef`。

    `control/waypoint` 还不只是"另一个名字"——它在 agent-core 的格式表里是**导航**
    语义（navigate_to / goto）。用它会让一张导航卡片和一张手臂卡片在画布上可以
    互换着连。
    """
    assert FORMATS["eef_pose"] == "control/eef"
    assert "waypoint" not in FORMATS.values(), "waypoint 是导航，不是末端位姿"


def test_the_out_port_can_be_declared_before_anything_is_wired():
    """**连线发生在 start 之前，而 descriptor 要 start 之后才有。**

    不给这条路的话，末端位姿的驱动卡片永远连不上：要拿到 descriptor 得先连，要连
    得先有 descriptor。而画布是严格字符串相等匹配，连不上时**静默** —— 拖放没反
    应，没提示也没日志。真机上就卡在这里。
    """
    assert make_card()._format() == "control/joint", "默认必须是今天的行为"

    eef = make_card(action_space="eef_pose")
    assert eef._format() == "control/eef"
    assert eef.get_tools()[0]["topic_out"][0]["format"] == "control/eef"


def test_the_negotiated_descriptor_wins_over_the_configured_guess():
    """配置只是连线时的占位。真值一到就该换掉它，否则填错会一直挂在那儿。"""
    card = make_card(action_space="eef_pose")
    card._descriptor = {"mode": "joint_position"}
    assert card._format() == "control/joint"


def test_a_model_that_needs_a_camera_refuses_to_start_without_one():
    """The providers have always declared needs_state and n_cameras honestly;
    nothing read them, so picking smolvla without wiring a camera produced a
    card reporting `running` that never emitted a command, with the reason
    buried in info().error."""
    card = make_card()
    node = _Graph({"/st": ["std_msgs/msg/String"]})

    binding, problem = card._bind_inputs(node, ["/st"],
                                         _caps(n_cameras=1, needs_state=True))

    assert binding is None
    assert "相机" in problem


def test_an_open_loop_provider_needs_nothing_wired():
    """mock declares n_cameras=0; it is open-loop by construction and must not
    be made to demand a camera it would ignore."""
    card = make_card()
    binding, problem = card._bind_inputs(_Graph({}), [], _caps())

    assert problem == ""
    assert binding == {"images": {}, "state": None}


def test_an_unrecognised_input_is_named_in_the_refusal():
    card = make_card()
    node = _Graph({"/odd": ["geometry_msgs/msg/Twist"]})

    _, problem = card._bind_inputs(node, ["/odd"], _caps(n_cameras=1))

    assert "/odd" in problem


def test_the_capture_time_is_the_oldest_channel_not_now():
    """RTC's inference_delay is computed from observation age. Reporting now()
    claims every channel just updated, when the stalest one may be seconds old
    — and the compensation is then made against the wrong instant."""
    card = make_card()
    card._capabilities = _caps(n_cameras=1, needs_state=True)
    card._images = {"cam0": b"jpeg"}
    card._image_ms = {"cam0": 5_000}
    card._proprio = [0.0] * 26
    card._state_ms = 3_000

    assert card.observation().t_capture_ms == 3_000


def test_no_observation_means_no_command_rather_than_a_stale_one():
    """Publishing nothing lets the driver's watchdog hold the arm, which is the
    right state for "no policy is driving". Repeating the last command would
    keep driving on a policy that is no longer seeing anything."""
    card = _started_card()
    card._capabilities = _caps(n_cameras=1)
    card._publisher = _CountingPublisher()

    card._tick()

    assert card._publisher.published == 0


def test_the_observation_carries_the_task_as_the_prompt():
    card = make_card()
    card._capabilities = _caps()
    card._task = "把杯子递给我"

    assert card.observation().prompt == "把杯子递给我"


def test_the_card_declares_both_input_ports():
    ports = make_card().get_tools()[0]["topic_in"]
    assert [p["format"] for p in ports] == ["image/jpeg", "state/joint"]


# ── execute ──────────────────────────────────────────────────────────────────

def test_execute_is_how_a_model_tells_the_policy_what_to_do():
    card = _started_card()

    result = card.dispatch("vla", {"action": "execute", "task": "把杯子递给我"})

    assert result["task"] == "把杯子递给我"
    assert card.observation().prompt == "把杯子递给我"


def test_changing_the_task_throws_away_the_plan_made_for_the_old_one():
    """A chunk is a stretch of future computed for the previous instruction.
    Running it under the new one looks entirely reasonable for the first tens
    of milliseconds, which is what makes it the dangerous kind of wrong."""
    card = _started_card()
    card.dispatch("vla", {"action": "execute", "task": "第一个任务"})
    card.next_command()
    assert len(card._chunk) - card._chunk_index > 0

    card.dispatch("vla", {"action": "execute", "task": "第二个任务"})

    assert card._chunk == []
    assert card._chunk_index == 0


def test_execute_also_lifts_a_pause():
    """Someone who says "go and fetch the cup" does not mean "note that down
    but keep still"."""
    card = _started_card()
    card.dispatch("vla", {"action": "pause"})

    card.dispatch("vla", {"action": "execute", "task": "去拿杯子"})

    assert card.dispatch("vla", {"action": "info"})["state"] == "running"


def test_execute_without_a_task_is_refused():
    card = _started_card()
    result = card.dispatch("vla", {"action": "execute", "task": "   "})
    assert result["state"] == "error"


def test_execute_on_a_card_that_is_not_running_says_so():
    result = make_card().dispatch("vla", {"action": "execute", "task": "x"})
    assert result["state"] == "idle"


def test_the_model_can_reach_execute():
    """`x-action-params` is the model's entire reach (mcp_client.py
    `_to_openai_schema`). Without execute listed there, a policy could be
    paused and resumed but never told what to do."""
    schema = make_card().get_tools()[0]["inputSchema"]

    assert set(schema["x-action-params"]) == {"execute", "pause", "interrupt", "resume"}
    assert schema["x-action-params"]["execute"]["params"] == ["task"]
    assert {"start", "stop"} <= set(schema["properties"]["action"]["enum"])


# ── the refusal has to say *which* model and *which* card ────────────────────
#
# `模型输出 6 维动作，下游只接受 26 维` names what disagrees but not who. A robot
# runs several cards, and this exact message appeared twice on Tianyi with no way
# to tell which checkpoint was wired to which arm without opening the canvas.

def test_a_mismatch_names_the_model_and_the_downstream_card():
    class Wide:
        def capabilities(self):
            return {"action_dim": 32, "control_hz": 30, "model": "smolvla/ckpt-9000"}
        def infer(self, obs=None, inference_delay=0): return [[0.0] * 32]
        def health(self): return True
        def close(self): pass

    card = make_card(provider="wide")
    import plugins.vla.plugin as plugin_mod
    original = plugin_mod.discover
    plugin_mod.discover = lambda: {"wide": lambda d, c, on_status=None: Wide()}
    plugin_mod.discover.errors = {}
    try:
        result = card.dispatch("start", {"action": "start",
                                         "control_interface": DESCRIPTOR})
    finally:
        plugin_mod.discover = original

    assert result["state"] == "error"
    msg = result["message"]
    # the disagreement itself, unchanged
    assert "32" in msg and "7" in msg
    # who the model is
    assert "smolvla/ckpt-9000" in msg
    assert "wide" in msg
    # which card it was pointed at — mode, joint count, and the first joint name,
    # which is what actually separates two arms on one robot
    assert "joint_position" in msg
    assert "joint1" in msg


def test_the_model_label_falls_back_to_the_provider_name():
    """A provider that does not name itself still has to be identifiable."""
    from plugins.vla.plugin import _model_label
    assert _model_label("smolvla", {}) == "smolvla"
    assert _model_label("smolvla", {"model": "ckpt-9000"}) == "smolvla:ckpt-9000"
    # No stutter when the checkpoint path already carries the family name —
    # "smolvla:smolvla/ckpt-9000" reads like a bug in the error message itself.
    assert _model_label("smolvla", {"model": "smolvla/ckpt-9000"}) == "smolvla/ckpt-9000"


def test_the_downstream_label_survives_a_sparse_descriptor():
    """Never let the diagnostic be the thing that raises."""
    from plugins.vla.plugin import _downstream_label
    assert _downstream_label({}) == "?/? 关节"
    assert "joint_position" in _downstream_label({"mode": "joint_position", "dof": 26})
    assert "left_shoulder" in _downstream_label(
        {"mode": "joint_position", "dof": 26, "joint_names": ["left_shoulder_pitch"]})


# ── a bad descriptor must not take the tool list with it ─────────────────────
#
# All three of these are one incident. A descriptor was accepted at `start` with
# `groups: ["arm_l"]` — strings where the spec says objects. Nothing failed then.
# `_resources()` runs on every *schema* fetch, a different code path, and there
# it raised; `tools/list` returns the whole bundle, so agent-core got an RPC
# error and served the cards it already knew with no ports at all. `stop` did
# not help — the descriptor is not cleared — so only a container restart did.
#
# Fixed in three places because each of them alone leaves the failure possible:
# refuse it at the door, survive it if it gets in, and contain it to one card.

def test_a_descriptor_whose_groups_are_not_objects_is_refused_at_start():
    from plugins.vla import negotiate
    caps = _caps(chunk_size=10)
    problems = negotiate.check(caps, {**DESCRIPTOR, "groups": ["arm_l", "arm_r"]})
    assert problems and "groups" in problems[0]
    # names the offending indices, so a 26-dof descriptor does not have to be
    # eyeballed to find which entry is wrong
    assert "0, 1" in problems[0]

    assert negotiate.check(caps, {**DESCRIPTOR, "groups": "arm_l"})
    # absent and well-formed are both fine
    assert not negotiate.check(caps, DESCRIPTOR)
    assert not negotiate.check(
        caps, {**DESCRIPTOR,
               "groups": [{"name": "arm_l", "offset": 0, "count": 7,
                           "resource": "arm_l"}]})


def test_resources_skips_a_group_it_cannot_read_instead_of_raising():
    """This runs on every schema fetch; raising here empties the whole bundle."""
    card = make_card(resource="arm")
    card._descriptor = {"groups": ["arm_l", None,
                                   {"name": "hand_l", "resource": "hand_l"}]}
    assert card._resources() == ["hand_l"]

    card._descriptor = {"groups": ["arm_l"]}
    assert card._resources() == ["arm"]      # falls back to the configured value


def test_one_cards_broken_schema_does_not_empty_the_bundle():
    # `main` imports rclpy at module scope and the rest of this suite runs on a
    # laptop without ROS. Stubbed rather than skipped: the thing under test is
    # ten lines of pure Python, and a test that only runs on a robot is a test
    # that runs after the mistake has already shipped.
    import types
    for name in ("rclpy", "rclpy.executors"):
        sys.modules.setdefault(name, types.ModuleType(name))
    import main

    class Fine:
        PREFIX = "fine"

        def get_tools(self):
            return [{"name": "fine"}]

    class Broken:
        PREFIX = "broken"

        def get_tools(self):
            raise AttributeError("'str' object has no attribute 'get'")

    bundle = object.__new__(main.ActuCoreBundle)
    bundle._plugins = [Broken(), Fine()]
    assert [t["name"] for t in bundle.get_all_tools()] == ["fine"]


# ── 动作空间：维度相同不代表空间相同 ─────────────────────────────────────────


def test_a_model_in_a_different_action_space_is_refused():
    """这条检查补上之前，这一组输入是**协商通过**的。

    UnifoLM-VLA 的 G1 checkpoint 输出 23 维 EE_R6_G1（2 × [xyz(3) + R6(6) + 夹爪(1)]
    + 腰 rpy(3)），而天轶那类机器人的命令卡片是 23 维 joint_position。两个 23 完全
    吻合，`dof` 和 `control_hz` 都挑不出毛病，于是位姿被当成关节角发下去。
    """
    problems = negotiate.check(
        _caps(control_mode="eef_r6_g1", action_dim=7),
        DESCRIPTOR,                                   # mode=joint_position, dof=7
    )
    assert problems
    assert any("eef_r6_g1" in p and "joint_position" in p for p in problems)


def test_a_model_that_declares_nothing_is_refused():
    """「不确定就拒绝，不要猜」—— 猜错的代价不是报错，是机械臂走到错误的地方。"""
    caps = _caps()
    caps.pop("control_mode")
    problems = negotiate.check(caps, DESCRIPTOR)
    assert problems
    # 报错要说清楚去哪儿设，两侧各一处。
    assert any("CONTROL_MODE" in p and "control_mode" in p for p in problems)


def test_a_matching_action_space_passes():
    assert negotiate.check(_caps(control_mode="joint_position"), DESCRIPTOR) == []


# ── 混合向量：顶层一个 mode 说不清的那些 ─────────────────────────────────────

# 规范化之后的 G1 动作空间：两个末端位姿、两个归一化夹爪、三个腰关节角。
MIXED_GROUPS = [
    {"name": "eef_l", "offset": 0, "count": 7, "mode": "eef_pose"},
    {"name": "gripper_l", "offset": 7, "count": 1, "mode": "joint_position"},
    {"name": "eef_r", "offset": 8, "count": 7, "mode": "eef_pose"},
    {"name": "gripper_r", "offset": 15, "count": 1, "mode": "joint_position"},
    {"name": "waist", "offset": 16, "count": 3, "mode": "joint_position"},
]


def _mixed_descriptor(groups=None):
    return {
        "control_interface": "motus.control/1",
        "mode": "eef_pose",
        "dof": 19,
        "joint_names": [f"a{i}" for i in range(19)],
        "units": {"length": "m", "angle": "rad"},
        "limits": {"lower": [-2.0] * 19, "upper": [2.0] * 19},
        "rate": {"max_hz": 100, "expected_hz": 30, "watchdog_ms": 200},
        "force_torque": None,
        "groups": [dict(g) for g in (groups or MIXED_GROUPS)],
    }


def test_a_mixed_vector_that_agrees_segment_by_segment_passes():
    problems = negotiate.check(
        _caps(control_mode="eef_pose", action_dim=19,
              control_groups=[dict(g) for g in MIXED_GROUPS]),
        _mixed_descriptor(),
    )
    assert problems == []


def test_segments_that_line_up_differently_are_refused():
    """总维度相同、顶层 mode 相同，而分段错位 —— 每一段都把邻段的数字当成自己的。

    这是顶层那条检查看不见的分歧：两边都报 19 维的 `eef_pose`，一边第 7 维是夹爪、
    另一边第 7 维还是位姿的一部分。发下去不报错。
    """
    shifted = [
        {"name": "eef_l", "offset": 0, "count": 8, "mode": "eef_pose"},
        {"name": "gripper_l", "offset": 8, "count": 1, "mode": "joint_position"},
        {"name": "eef_r", "offset": 9, "count": 7, "mode": "eef_pose"},
        {"name": "gripper_r", "offset": 16, "count": 1, "mode": "joint_position"},
        {"name": "waist", "offset": 17, "count": 2, "mode": "joint_position"},
    ]
    problems = negotiate.check(
        _caps(control_mode="eef_pose", action_dim=19, control_groups=shifted),
        _mixed_descriptor(),
    )
    assert problems
    assert any("位置对不上" in p for p in problems)


def test_a_segment_in_the_wrong_space_is_refused():
    """腰那三个是关节角。一个把它们也当成笛卡尔量的模型，维度全对。"""
    wrong = [dict(g) for g in MIXED_GROUPS]
    wrong[-1] = {**wrong[-1], "mode": "eef_pose"}
    problems = negotiate.check(
        _caps(control_mode="eef_pose", action_dim=19, control_groups=wrong),
        _mixed_descriptor(),
    )
    assert problems
    assert any("waist" in p for p in problems)


def test_different_numbers_of_segments_are_refused_rather_than_zipped():
    """段数不同就无从逐段核对。按最短的那个 zip 过去会静默漏掉尾巴。"""
    problems = negotiate.check(
        _caps(control_mode="eef_pose", action_dim=19,
              control_groups=[{"name": "all", "offset": 0, "count": 19,
                               "mode": "eef_pose"}]),
        _mixed_descriptor(),
    )
    assert problems
    assert any("分成" in p for p in problems)


def test_a_driver_group_without_a_mode_inherits_the_top_level_one():
    """和 `motus.control/1` 驱动侧同一条规矩 —— 不写就是「和整体一样」。

    今天每一个已有的驱动都不写段 mode，所以这条不成立的话，它们全都会在协商时被
    判成和模型分歧。
    """
    inheriting = [
        {"name": "a", "offset": 0, "count": 10},         # 不写 → eef_pose
        {"name": "b", "offset": 10, "count": 9, "mode": "eef_pose"},
    ]
    problems = negotiate.check(
        _caps(control_mode="eef_pose", action_dim=19,
              control_groups=[{"name": "a", "offset": 0, "count": 10,
                               "mode": "eef_pose"},
                              {"name": "b", "offset": 10, "count": 9,
                               "mode": "eef_pose"}]),
        _mixed_descriptor(inheriting),
    )
    assert problems == []


# ── advisory vs optional：丢维要双方都同意过 ────────────────────────────────
#
# 由来是一次真机实测：G1 的 1 自由度腰上，`unifolm-vla-g1` 的 25 步动作块一步都
# 没通过，全数停在 `waist_roll outside [-0.02, 0.02]` —— 那个限位没错（腰确实动不
# 了），但它把整条指令拒掉了，连同两条本可以执行的手臂。
#
# 解法不是放宽限位（那会让 IK 按一个机器人到不了的躯干姿态解手臂，每拍差同样一点
# 而没有一处报错），而是两侧各声明一半：驱动说「我收下但不执行」（advisory），
# 模型说「任务不要求执行」（optional）。**这个函数是它们相遇的地方**，也是让
# 「静默丢掉几维」在这套协议里不可能发生的那道门。


def _advisory_waist_descriptor():
    groups = [dict(g) for g in MIXED_GROUPS]
    groups[-1]["advisory"] = True
    return _mixed_descriptor(groups)


def test_a_dropped_segment_the_model_requires_is_refused():
    """驱动不执行，而模型认为必须执行 —— 这不是可以两边各让一步的事。"""
    problems = negotiate.check(
        _caps(control_mode="eef_pose", action_dim=19,
              control_groups=[dict(g) for g in MIXED_GROUPS]),
        _advisory_waist_descriptor(),
    )
    assert problems
    assert any("advisory" in p and "optional" in p for p in problems)


def test_a_dropped_segment_the_model_allows_is_accepted():
    """许可到位就放行。这是 unifolm-vla-g1 在 1 自由度腰 G1 上真正走的那条路。"""
    allowed = [dict(g) for g in MIXED_GROUPS]
    allowed[-1]["optional"] = True
    assert negotiate.check(
        _caps(control_mode="eef_pose", action_dim=19, control_groups=allowed),
        _advisory_waist_descriptor(),
    ) == []


def test_permission_alone_changes_nothing():
    """模型说可丢、驱动说会执行 —— 那就执行，没有分歧。

    反过来理解这个字段（「模型说可丢，所以别执行了」）会让一台**真能弯腰**的
    29dof G1 从此不再弯腰，而没有任何一处报错。
    """
    allowed = [dict(g) for g in MIXED_GROUPS]
    allowed[-1]["optional"] = True
    assert negotiate.check(
        _caps(control_mode="eef_pose", action_dim=19, control_groups=allowed),
        _mixed_descriptor(),
    ) == []


def test_neither_side_declaring_anything_is_todays_every_model():
    """两个字段都缺省 false，所以这条检查对今天每一对声明都是透明的。"""
    assert negotiate.check(
        _caps(control_mode="eef_pose", action_dim=19,
              control_groups=[dict(g) for g in MIXED_GROUPS]),
        _mixed_descriptor(),
    ) == []


def test_a_single_space_model_is_not_forced_to_declare_segments():
    """一边有段一边没有不算分歧 —— 今天每个模型都是单一空间，没有段是常态。"""
    assert negotiate.check(
        _caps(control_mode="eef_pose", action_dim=19),
        _mixed_descriptor(),
    ) == []
    # 而顶层 mode 真的不同时，仍然由上面那条检查抓住 —— 不是因为没有段就放行。
    assert negotiate.check(
        _caps(control_mode="joint_position", action_dim=19),
        _mixed_descriptor(),
    )


def test_the_action_space_check_does_not_mask_the_others():
    """空间不对、维度也不对时，两条都要报出来。

    `check()` 收集全部理由而不是撞上第一条就返回 —— 在画布上改接线的人应该一次看到
    全部分歧，而不是修好一个再发现下一个。
    """
    problems = negotiate.check(
        _caps(control_mode="joint_velocity", action_dim=32), DESCRIPTOR
    )
    assert len(problems) == 2
    assert any("joint_velocity" in p for p in problems)
    assert any("32" in p and "7" in p for p in problems)


def test_the_local_providers_declare_their_action_space():
    """mock 和 smolvla 都发绝对关节角。

    不声明的话，它们自己会被上面那条检查拦下 —— 这个用例钉的是「新增 provider 时
    别忘了这个字段」，而不是某个具体的值。
    """
    from plugins.vla.providers import mock as mock_provider

    provider = mock_provider.PROVIDER(DESCRIPTOR, {})
    assert provider.capabilities()["control_mode"] in ("joint_position",)


# ── eef_state：增量模型的基准位姿 ────────────────────────────────────────────


class _Message:
    def __init__(self, data):
        self.data = data


def _state_message(values, eef=None, stamp_ms=7_000):
    payload = {"schema": "motus.control/1", "kind": "joint_state",
               "values": list(values), "stamp_ms": stamp_ms}
    if eef is not None:
        payload["eef"] = list(eef)
    return _Message(json.dumps(payload))


POSE = [0.3, 0.1, 0.2, 0.0, 0.0, 0.0, 1.0]


def test_the_end_effector_pose_rides_in_the_state_payload_not_a_second_topic():
    """`_bind_inputs` 按 **ROS 消息类型**分派角色，两路 `String` 它分不开。

    单开一路末端位姿话题会变成一个按连线顺序赌运气的绑定 —— 有时对，有时把本体
    状态当成位姿。所以发布方（driver 的 servo_eef）把 `eef` 放进同一条载荷。
    """
    card = make_card()
    card._capabilities = _caps(n_cameras=0, needs_state=True, needs_eef_state=True)
    card._on_state(_state_message([0.1] * 17, eef=POSE + [0.0] * 12))

    observation = card.observation()
    assert observation.state == [0.1] * 17
    assert observation.eef_state[:7] == POSE


def test_a_state_payload_without_the_field_leaves_it_none():
    """今天绝大多数驱动的状态载荷里没有 `eef`。多出来的这个字段不能让它们变成
    「报了一个空位姿」。"""
    card = make_card()
    card._capabilities = _caps(n_cameras=0, needs_state=True)
    card._on_state(_state_message([0.1] * 17))
    assert card.observation().eef_state is None


def test_a_delta_model_publishes_nothing_when_the_base_pose_is_missing():
    """和 `needs_state` 同样的处理。

    凑一个单位位姿上去，手臂会飞到原点附近一个看起来挺合理的地方，而上游每一道
    检查都满意 —— 不发这一拍，驱动的看门狗保持，那是正确的状态。
    """
    card = make_card()
    card._capabilities = _caps(n_cameras=0, needs_state=False, needs_eef_state=True)
    card._on_state(_state_message([0.1] * 17))          # 有 values，没有 eef
    assert card.observation() is None

    card._on_state(_state_message([0.1] * 17, eef=POSE))
    assert card.observation() is not None


def test_binding_refuses_to_start_a_delta_model_with_no_state_topic():
    """缺什么在**启动时**说清楚。不然卡片会报 running、一条指令都不发，而原因
    只藏在 info().error 里。"""
    _stub_ros_messages()
    card = make_card()
    node = _Graph({"/cam": ["sensor_msgs/msg/CompressedImage"]})

    _, problem = card._bind_inputs(
        node, ["/cam"], _caps(n_cameras=1, needs_state=False, needs_eef_state=True))

    assert "末端位姿" in problem


def test_the_provider_omits_the_field_entirely_when_there_is_no_pose():
    """服务端把 None 和 [] 分开看：后者是「机器人报了，但它是空的」，那是个错误。"""
    from plugins.vla.providers import vla_cloud

    provider = vla_cloud.VLACloudProvider({}, {"endpoint": "https://vla.test"})
    sent = {}

    def _post(path, payload):
        sent.update(payload)
        return {"seq": payload["seq"], "actions": [[0.0]]}

    provider._post = _post
    provider.infer(Observation(images={}, state=[0.0], eef_state=None))
    assert "eef_state" not in sent

    provider.infer(Observation(images={}, state=[0.0], eef_state=POSE))
    assert sent["eef_state"] == POSE
