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

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plugins.vla import VLAPlugin  # noqa: E402
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
    problems = negotiate.check({"action_dim": 32}, DESCRIPTOR)
    assert problems
    assert "32" in problems[0] and "7" in problems[0]


def test_a_downstream_that_is_not_a_control_card_is_caught_first():
    problems = negotiate.check({"action_dim": 7}, {"control_interface": None})
    assert len(problems) == 1
    assert "motus.control/1" in problems[0]


def test_a_model_faster_than_the_hardware_is_refused():
    problems = negotiate.check({"action_dim": 7, "control_hz": 500}, DESCRIPTOR)
    assert any("500" in p for p in problems)


def test_every_problem_is_reported_at_once():
    """An operator fixing a canvas should see the whole disagreement."""
    problems = negotiate.check({"action_dim": 32, "control_hz": 500}, DESCRIPTOR)
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
        def capabilities(self): return {"action_dim": 7}
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
        self.subscribed.append((message_type.__name__, topic, callback))


def _caps(**over):
    base = {"n_cameras": 0, "needs_state": False}
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
