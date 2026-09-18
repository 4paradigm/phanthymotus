"""Any model that runs somewhere else — one provider, `{endpoint, key, model}`.

The mirror image of how the local providers are organised, and deliberately so.
Locally there is one file per model because each one brings its own weights,
its own load and its own quirks. Remotely none of that is the robot's problem:
what arrives is an action chunk, and the only thing that varies between a π0.5
server, a LeRobot policy server and a vendor's endpoint is the address.

So there is one provider here and not four, and configuring it looks exactly
like configuring an LLM in this project already does — `agent-core`'s
`config.main['client']['llm']` is a list of `{url, key, model}` and there is not
one line of serving code beside it.

**The consequence, stated plainly:** this speaks `motus.vla/1`, our own
protocol, not openpi's msgpack-over-WebSocket or LeRobot's gRPC. Pointing it at
a raw upstream server will not work — the translation belongs on the server
side, in `phanthymotus-cloud`, which is where the throughput and autoscaling
problems already live. That is the same split OpenAI's ecosystem uses: the spec
is a document, vLLM and SGLang each implement it, and no client carries an
adapter per server.

Spec: phanthymotus/docs/vla-integration.md §"云端边界".
"""

from __future__ import annotations

import base64
import json
import logging
import urllib.error
import urllib.request

log = logging.getLogger(__name__)

SCHEMA = "motus.vla/1"


class VLACloudProvider:
    """A `motus.vla/1` endpoint.

    Config keys:
        endpoint   base URL, e.g. https://vla.internal:8443
        api_key    bearer token; issued and checked by the server
        cloud_model_name  which model to ask for, when the endpoint serves
                   several. Free text: these names live on the server.
        timeout_ms per-request deadline, default 500 ms.

                   This is the point of giving up, not the expected latency. A
                   reply that arrives late is not a safety problem — the driver
                   drops anything whose observation is older than the
                   descriptor's `max_obs_age_ms`, and its watchdog has already
                   held the arm. What a long timeout costs is how quickly a
                   dead endpoint is noticed, so there is little value in setting
                   it far above that age limit.
    """

    def __init__(self, descriptor: dict, config: dict | None = None):
        config = dict(config or {})
        self._descriptor = descriptor or {}
        self._endpoint = str(config.get("endpoint") or "").rstrip("/")
        if not self._endpoint:
            raise ValueError(
                "vla_cloud needs an `endpoint`. The robot does not host the "
                "model; point this at a motus.vla/1 server (phanthymotus-cloud)."
            )
        self._key = str(config.get("api_key") or "")
        # `cloud_model_name`, not `model_name`: that one holds a locally
        # staged checkpoint, and sending its name to a server would be a
        # plausible-looking request for something the server has never heard of.
        self._model = str(config.get("cloud_model_name") or "")
        self._timeout = float(config.get("timeout_ms", 500)) / 1000.0
        self._session = None
        self._seq = 0
        self._capabilities = None

    # ── provider protocol ────────────────────────────────────────────────────

    def capabilities(self) -> dict:
        """Handshake once, and let the card refuse before anything moves.

        Cached: negotiation happens at start and the answer cannot change
        under a running session — the server pins a session to a checkpoint
        version precisely so it does not.
        """
        if self._capabilities is None:
            self._capabilities = self._get("/capabilities")
            schema = self._capabilities.get("schema")
            if schema != SCHEMA:
                raise RuntimeError(
                    f"endpoint speaks {schema!r}, this card speaks {SCHEMA!r}. "
                    f"A mismatched action width read as a version difference is "
                    f"how a robot moves to the wrong place."
                )
        return dict(self._capabilities)

    def infer(self, observation=None, inference_delay: int = 0) -> list:
        if observation is None:
            raise ValueError("vla_cloud needs an observation")
        self._seq += 1
        payload = {
            "schema": SCHEMA,
            "session_id": self._session_id(),
            "seq": self._seq,
            # The observation's own timestamp, not now(): the server uses it for
            # RTC, and passing the send time instead silently disables that.
            "t_capture_ms": int(getattr(observation, "t_capture_ms", 0) or 0),
            "prompt": getattr(observation, "prompt", "") or "",
            "images": {
                name: base64.b64encode(_jpeg(image)).decode()
                for name, image in (getattr(observation, "images", None) or {}).items()
            },
            "inference_delay": int(inference_delay),
        }
        state = getattr(observation, "state", None)
        if state is not None:
            payload["state"] = [float(v) for v in state]
        if self._model:
            payload["model"] = self._model

        reply = self._post("/infer", payload)
        # Out-of-order replies are dropped rather than applied: a chunk computed
        # from an older observation is worse than no chunk, because the receiver
        # cannot tell it is stale once it is in flight.
        if reply.get("seq") not in (None, self._seq):
            raise RuntimeError(f"reply out of order: {reply.get('seq')} != {self._seq}")
        actions = reply.get("actions")
        if not isinstance(actions, list) or not actions:
            raise RuntimeError("endpoint returned no actions")
        return [[float(v) for v in step] for step in actions]

    def health(self) -> bool:
        try:
            self._get("/healthz")
            return True
        except Exception:      # noqa: BLE001 — the answer is the boolean
            return False

    def close(self) -> None:
        self._session = None
        self._capabilities = None

    # ── transport ────────────────────────────────────────────────────────────

    def _session_id(self) -> str:
        if self._session is None:
            # Derived from the endpoint and the start, not random: this file is
            # imported into a process where Math.random-style entropy is fine,
            # but a stable id makes a server-side log searchable.
            self._session = f"{self._endpoint}#{id(self)}"
        return self._session

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        return headers

    def _get(self, path: str) -> dict:
        request = urllib.request.Request(self._endpoint + path,
                                         headers=self._headers(), method="GET")
        return self._send(request)

    def _post(self, path: str, payload: dict) -> dict:
        body = json.dumps(payload).encode()
        request = urllib.request.Request(self._endpoint + path, data=body,
                                         headers=self._headers(), method="POST")
        return self._send(request)

    def _send(self, request) -> dict:
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as error:
            raise RuntimeError(
                f"{request.get_method()} {request.full_url} → {error.code}"
            ) from error
        except Exception as error:      # noqa: BLE001
            # Timeouts land here and are expected under load. Publishing nothing
            # lets the receiver's watchdog hold the arm, which is the right
            # state; retrying would only deliver a staler chunk.
            raise RuntimeError(f"{request.full_url}: {error}") from error


def _jpeg(image) -> bytes:
    """Encode one frame for the wire.

    Already-encoded bytes pass through: the card's input is a `video/mjpeg`
    topic, so the common case is that the frame arrived compressed and
    re-encoding it would cost quality and time for nothing.
    """
    if isinstance(image, (bytes, bytearray)):
        return bytes(image)
    import cv2                       # lazy: a bytes-in deployment never needs it

    ok, buffer = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("could not encode frame")
    return buffer.tobytes()


def PROVIDER(descriptor: dict, config: dict | None = None,
             on_status=None) -> VLACloudProvider:
    # `on_status` is part of the factory signature so the card can pass it to
    # any provider without asking which one it got. Ignored here: the weights
    # live on the server and nothing is downloaded to report on.
    del on_status
    return VLACloudProvider(descriptor, config)


for _name in ("capabilities", "infer", "health", "close"):
    setattr(PROVIDER, _name, getattr(VLACloudProvider, _name))

# Free text. The model names here live on the server and are not ours to
# enumerate — offering the locally staged checkpoints as a dropdown would be
# worse than useless, because the form renders an enum as a <select> and the
# operator could then not type the name the server actually knows.
PROVIDER.MODEL_NAMES = "remote"
