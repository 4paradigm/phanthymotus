"""The one provider for everything that runs somewhere else.

No network: `urlopen` is stubbed, so what is tested is the contract — what goes
on the wire, what is refused, and what happens when the endpoint is slow or
wrong. That is the whole of this provider's job; the model's problems are the
server's.

The cases worth pinning:

  schema      an endpoint speaking a different version is refused at the
              handshake, before anything moves
  timestamps  `t_capture_ms` is the observation's, not the send time — passing
              the latter silently disables the server's RTC alignment
  ordering    a reply that does not match the request is dropped, because a
              chunk from an older observation cannot be told apart once applied
  failure     a timeout raises rather than retries: the receiver's watchdog
              holding the arm beats a staler chunk arriving late

Run: cd actucore && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_vla_cloud_provider.py -q
"""

from __future__ import annotations

import io
import json
import pathlib
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plugins.vla.providers.vla_cloud import SCHEMA, VLACloudProvider  # noqa: E402


CAPABILITIES = {
    "schema": SCHEMA,
    "model": "pi05@2026-02-01",
    "action_dim": 14,
    "chunk_size": 50,
    "control_hz": 30,
    "needs_state": True,
    "n_cameras": 2,
    "image_size": 224,
    "supports_rtc": True,
}


class FakeHTTP:
    """Records requests and replays canned responses."""

    def __init__(self, responses=None, error=None):
        self.requests = []
        self._responses = dict(responses or {})
        self._error = error

    def __call__(self, request, timeout=None):
        self.requests.append((request, timeout))
        if self._error is not None:
            raise self._error
        path = request.full_url.split("://", 1)[-1].split("/", 1)[-1]
        body = self._responses.get("/" + path, {})
        if callable(body):
            body = body(request)
        stream = io.BytesIO(json.dumps(body).encode())
        stream.__enter__ = lambda: stream
        stream.__exit__ = lambda *a: False
        return stream

    def payload(self, index=-1):
        request = self.requests[index][0]
        return json.loads(request.data.decode())


@pytest.fixture
def http(monkeypatch):
    import plugins.vla.providers.vla_cloud as mod

    fake = FakeHTTP()
    monkeypatch.setattr(mod.urllib.request, "urlopen", fake)
    return fake


def make_provider(**config):
    base = {"endpoint": "https://vla.internal:8443", "api_key": "k",
            "cloud_model_name": "pi05", "timeout_ms": 150}
    base.update(config)
    return VLACloudProvider({}, base)


def _observation(**overrides):
    base = dict(images={"main": b"jpegbytes"}, state=[0.0] * 14,
                prompt="pick it up", t_capture_ms=1_700_000_000_000)
    base.update(overrides)
    return types.SimpleNamespace(**base)


# ── configuration ────────────────────────────────────────────────────────────

def test_an_endpoint_is_required():
    """The robot does not host the model; without an address there is nothing."""
    with pytest.raises(ValueError) as excinfo:
        VLACloudProvider({}, {})
    assert "endpoint" in str(excinfo.value)


def test_configuration_is_just_address_key_model():
    provider = make_provider()
    assert provider._endpoint == "https://vla.internal:8443"
    assert provider._model == "pi05"


# ── the handshake ────────────────────────────────────────────────────────────

def test_capabilities_come_from_the_endpoint(http):
    http._responses["/capabilities"] = CAPABILITIES
    assert make_provider().capabilities()["action_dim"] == 14


def test_a_different_schema_version_is_refused(http):
    """A changed action width read as a version difference moves an arm wrongly."""
    http._responses["/capabilities"] = {**CAPABILITIES, "schema": "motus.vla/2"}

    with pytest.raises(RuntimeError) as excinfo:
        make_provider().capabilities()

    assert "motus.vla/2" in str(excinfo.value)


def test_the_handshake_happens_once(http):
    """The server pins a session to a checkpoint, so the answer cannot change."""
    http._responses["/capabilities"] = CAPABILITIES
    provider = make_provider()
    provider.capabilities()
    provider.capabilities()
    assert len(http.requests) == 1


def test_the_key_travels_as_a_bearer_token(http):
    http._responses["/capabilities"] = CAPABILITIES
    make_provider().capabilities()
    assert http.requests[0][0].headers["Authorization"] == "Bearer k"


def test_no_key_configured_sends_no_header(http):
    http._responses["/capabilities"] = CAPABILITIES
    make_provider(api_key="").capabilities()
    assert "Authorization" not in http.requests[0][0].headers


# ── what goes on the wire ────────────────────────────────────────────────────

def test_the_request_carries_the_observations_own_timestamp(http):
    """Sending now() instead silently disables the server's RTC alignment."""
    http._responses["/infer"] = lambda req: {"seq": 1, "actions": [[0.0] * 14]}

    make_provider().infer(_observation())

    payload = http.payload()
    assert payload["t_capture_ms"] == 1_700_000_000_000
    assert payload["schema"] == SCHEMA
    assert payload["prompt"] == "pick it up"
    assert payload["model"] == "pi05"


def test_already_encoded_frames_are_not_re_encoded(http):
    """The card's input is a video/mjpeg topic; re-encoding costs quality."""
    http._responses["/infer"] = lambda req: {"seq": 1, "actions": [[0.0] * 14]}

    make_provider().infer(_observation(images={"main": b"jpegbytes"}))

    import base64
    assert base64.b64decode(http.payload()["images"]["main"]) == b"jpegbytes"


def test_sequence_numbers_increase(http):
    http._responses["/infer"] = lambda req: {"actions": [[0.0] * 14]}
    provider = make_provider()
    provider.infer(_observation())
    provider.infer(_observation())
    assert [http.payload(0)["seq"], http.payload(1)["seq"]] == [1, 2]


def test_inference_delay_is_forwarded(http):
    http._responses["/infer"] = lambda req: {"actions": [[0.0] * 14]}
    make_provider().infer(_observation(), inference_delay=3)
    assert http.payload()["inference_delay"] == 3


def test_a_state_free_observation_omits_state(http):
    http._responses["/infer"] = lambda req: {"actions": [[0.0] * 14]}
    make_provider().infer(_observation(state=None))
    assert "state" not in http.payload()


# ── what comes back ──────────────────────────────────────────────────────────

def test_a_chunk_becomes_a_list_of_steps(http):
    http._responses["/infer"] = lambda req: {"seq": 1, "actions": [[1, 2], [3, 4]]}
    assert make_provider().infer(_observation()) == [[1.0, 2.0], [3.0, 4.0]]


def test_an_out_of_order_reply_is_refused(http):
    """A chunk from an older observation cannot be told apart once applied."""
    http._responses["/infer"] = lambda req: {"seq": 99, "actions": [[0.0] * 14]}

    with pytest.raises(RuntimeError) as excinfo:
        make_provider().infer(_observation())

    assert "out of order" in str(excinfo.value)


def test_an_empty_chunk_is_an_error_not_a_pause(http):
    http._responses["/infer"] = lambda req: {"seq": 1, "actions": []}
    with pytest.raises(RuntimeError):
        make_provider().infer(_observation())


# ── failure ──────────────────────────────────────────────────────────────────

def test_a_timeout_raises_rather_than_retries(monkeypatch):
    """The receiver's watchdog holding the arm beats a staler chunk arriving."""
    import plugins.vla.providers.vla_cloud as mod

    fake = FakeHTTP(error=TimeoutError("timed out"))
    monkeypatch.setattr(mod.urllib.request, "urlopen", fake)

    with pytest.raises(RuntimeError):
        make_provider().infer(_observation())

    assert len(fake.requests) == 1          # one attempt, no retry


def test_health_is_a_boolean_not_an_exception(monkeypatch):
    import plugins.vla.providers.vla_cloud as mod

    monkeypatch.setattr(mod.urllib.request, "urlopen",
                        FakeHTTP(error=OSError("unreachable")))
    assert make_provider().health() is False


def test_the_request_deadline_is_the_configured_one(http):
    http._responses["/infer"] = lambda req: {"actions": [[0.0] * 14]}
    make_provider(timeout_ms=150).infer(_observation())
    assert http.requests[0][1] == pytest.approx(0.150)


def test_the_default_deadline_is_500ms():
    """The point of giving up, not the expected latency.

    A late reply is not a safety problem: the driver drops anything whose
    observation is older than the descriptor's `max_obs_age_ms`, and its
    watchdog has already held the arm. What a long timeout costs is how quickly
    a dead endpoint is noticed — so there is no reason to set it far above that
    age limit, and no reason to squeeze it under the watchdog either.
    """
    provider = VLACloudProvider({}, {"endpoint": "https://vla.internal"})
    assert provider._timeout == pytest.approx(0.5)


def test_the_config_key_is_cloud_model_name_but_the_wire_field_is_model(http):
    """Three names, each for a different reader, and none interchangeable.

    `cloud_model_name` is what an operator sets for a remote provider;
    `model_name` is the locally staged checkpoint and must never be sent
    anywhere — it would be a plausible-looking request for something the server
    has never heard of; `model` is what motus.vla/1 puts on the wire.
    """
    http._responses["/infer"] = lambda req: {"actions": [[0.0] * 14]}

    make_provider(cloud_model_name="pi05-droid").infer(_observation())

    assert http.payload()["model"] == "pi05-droid"


def test_a_locally_staged_checkpoint_name_is_never_sent(http):
    """`model_name` belongs to the local providers and must not leak here.

    Sending it would be a plausible-looking request for a checkpoint the server
    has never heard of — the server would answer something, and the card would
    drive an arm with it.
    """
    http._responses["/infer"] = lambda req: {"actions": [[0.0] * 14]}
    provider = VLACloudProvider({}, {"endpoint": "https://vla.internal",
                                     "model_name": "smolvla_base"})

    provider.infer(_observation())

    assert "model" not in http.payload()
