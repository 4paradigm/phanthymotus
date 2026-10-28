"""Language policy layered on the current asynchronous VITS2 implementation.

This module selects readings and returns an instruction to a dialogue caller.
It does not subscribe to ASR, translate text or generate a robot reply.
"""

from dataclasses import dataclass
import json
import logging
import math
import threading

from utils.log_sampling import SampledLogGate

from ..vits2_tts_trt.plugin import TTSPlugin as BaseTTSPlugin, _Vits2TTSNode, _node_suffix
from .policy import LanguagePolicy, LanguagePolicyError, SpeechRequest, _scripts, validate_reply_script
from .neutral_text import (
    InvalidDateError, is_neutral_text,
    normalize_mixed_text, normalize_neutral_text,
)
from .name_text import normalize_standalone_name, _english_version_names

log = logging.getLogger(__name__)
_LABELS = {"自动": "auto", "中文": "zh", "English": "en"}
_ASR_ALIASES = {
    "zh": "zh", "zh-cn": "zh", "zh-tw": "zh", "zh-hans": "zh", "zh-hant": "zh",
    "en": "en", "en-us": "en", "en-gb": "en", "en-au": "en",
}


def reply_language_mode(value):
    if isinstance(value, str):
        value = _LABELS.get(value, value)
    LanguagePolicy(value)
    return value


def validate_config(args):
    """Validate before either the facade or engine stores a shared setting."""
    if "reply_language" in args:
        reply_language_mode(args["reply_language"])
    if "speed" in args:
        try:
            speed = float(args["speed"])
        except (TypeError, ValueError):
            raise LanguagePolicyError("invalid_speed", "speed must be in (0, 4]") from None
        if not math.isfinite(speed) or not 0 < speed <= 4:
            raise LanguagePolicyError("invalid_speed", "speed must be in (0, 4]")
    if "speaker_id" in args and args["speaker_id"] != 0:
        raise LanguagePolicyError("invalid_speaker", "VITS2 supports only speaker_id=0")
    if any(key in args and args[key] != "" for key in {"api_key", "model", "voice", "url"}):
        raise LanguagePolicyError("unsupported_config", "Remote-service settings do not apply to VITS2")


def _language_input(args):
    if "asr_result" not in args:
        return args.get("text"), args.get("input_language")
    result = args["asr_result"]
    if not isinstance(result, dict) or not isinstance(result.get("text"), str) or not result["text"].strip():
        raise LanguagePolicyError("invalid_asr_result", "asr_result requires nonempty text")
    text = result["text"]
    if "text" in args and (not isinstance(args["text"], str) or args["text"].strip() != text.strip()):
        raise LanguagePolicyError("conflicting_user_text", "text and asr_result.text must agree")
    raw = result.get("lang")
    if raw is not None and not isinstance(raw, str):
        raise LanguagePolicyError("invalid_asr_language", "asr_result.lang must be a string")
    # `language` is an ASR setting, not a recognized tag; only `lang` is read.
    detected = _ASR_ALIASES.get((raw or "").strip().lower().replace("_", "-"))
    supplied = args.get("input_language")
    if supplied is not None and supplied not in ("zh", "en"):
        raise LanguagePolicyError("invalid_input_language", "input_language must be zh or en")
    if supplied is not None and detected is not None and supplied != detected:
        raise LanguagePolicyError("conflicting_input_language", "input_language conflicts with asr_result.lang")
    return text, supplied if supplied is not None else detected


@dataclass(frozen=True)
class LiteralSpeechRequest(SpeechRequest):
    """Directly supplied text: the selected language is a reading preference."""

    # Freeze only fixed panel preferences. Auto prose retains MIX's existing
    # local clause readings; explicit dialogue SpeechRequest uses its language.
    reading_language: str = ""
    direct_turn: object = None
    language_source: str = ""


@dataclass(frozen=True)
class PreparedSpeech(SpeechRequest):
    chunks: tuple


def _prepare_chunks(adapter, request):
    """Freeze frontend output once, before it is handed to the worker."""
    try:
        reading_language = (request.reading_language if isinstance(request, LiteralSpeechRequest)
                            else request.language)
        if is_neutral_text(request.text):
            text = normalize_neutral_text(request.text, request.language)
        else:
            text = request.text
            if reading_language == "en":
                text = _english_version_names(text)
            elif not reading_language or _scripts(text) == {"en"}:
                text = normalize_standalone_name(text, reading_language or request.language)
            if reading_language:
                text = normalize_mixed_text(text, reading_language)
        chunks, parts = [], []
        prepared_chunks = (adapter.iter_text_chunks(text, reading_language=reading_language)
                           if reading_language else adapter.iter_text_chunks(text))
        for normalized, ids in prepared_chunks:
            validate_reply_script(normalized, request.language, allow_neutral=True,
                                  frontend=True, require_primary=False)
            chunks.append((normalized, tuple(tuple(values) for values in ids)))
            parts.append(normalized)
        validate_reply_script(" ".join(parts), request.language, frontend=True,
                              require_primary=not isinstance(request, LiteralSpeechRequest))
        return PreparedSpeech(request.text, request.language, request.turn_id, tuple(chunks))
    except InvalidDateError as error:
        raise LanguagePolicyError("invalid_numeric_date", str(error)) from error
    except LanguagePolicyError:
        raise
    except Exception as error:
        raise LanguagePolicyError("frontend_error", "VITS2 could not prepare the supplied text") from error


def _direct_request(policy, text, direct_turn=None):
    direct_turn = direct_turn or policy.reserve_direct(text)
    turn = policy.preview_direct(direct_turn, text)
    if (turn["reply_language_mode"] == "auto" and turn["language_source"] == "default"
            and not is_neutral_text(text)
            and _scripts(normalize_standalone_name(text, "zh")) == {"en"}):
        turn = policy.preview_direct(direct_turn, text, requested_reply_language="en")
    # A manual utterance is not a generated dialogue reply. Read its supported
    # Chinese/English content literally, including standalone names or acronyms.
    # Explicit language -> speak still goes through policy.prepare and its turn
    # and primary-language checks; this does not bypass that dialogue contract.
    validate_reply_script(normalize_standalone_name(text, turn["reply_language"]),
                          turn["reply_language"], allow_neutral=is_neutral_text(text),
                          require_primary=False)
    return LiteralSpeechRequest(
        text, turn["reply_language"], turn["turn_id"],
        reading_language=turn["reply_language"] if turn["reply_language_mode"] != "auto" else "",
        direct_turn=direct_turn, language_source=turn["language_source"],
    )


@dataclass(frozen=True)
class _PreparedAdapterView:
    """Use validated chunks with the original locking, RNG and PCM code."""

    _adapter: object
    _request: PreparedSpeech

    def __getattr__(self, name):
        return getattr(self._adapter, name)

    def iter_text_chunks(self, text):
        if text != self._request.text:
            raise LanguagePolicyError("prepared_text_mismatch", "Speech context belongs to another request")
        yield from self._request.chunks


class _LanguageNode(_Vits2TTSNode):
    def __init__(self, input_topic, adapter, suffix, policy, prepare_topic, accept_speech):
        self.policy = policy
        self._prepare_topic = prepare_topic
        self._accept_speech = accept_speech
        self._topic_log_lock = threading.Lock()
        self._topic_rejection_logs = SampledLogGate(every=100)
        self._topic_failure_logs = SampledLogGate(every=100)
        super().__init__(input_topic, adapter, suffix)

    def enqueue(self, text, action_id="", context=None, generation=None):
        # Frontend preparation can be slow. An interrupt during preparation
        # must cancel this request rather than relabel it as a new utterance.
        if generation is None:
            generation = self._current_gen()
        if not isinstance(context, SpeechRequest) or context.text != text:
            raise LanguagePolicyError("unvalidated_request", "Missing language context for speech")
        try:
            if isinstance(context, LiteralSpeechRequest):
                # Deferred items are flushed in FIFO order. Resolve neutral
                # readings now, after preceding items passed their frontend.
                context = _direct_request(self.policy, text, context.direct_turn)
            prepared = _prepare_chunks(self._adapter, context)
            self._accept_speech(self, context, generation, lambda: super(_LanguageNode, self).enqueue(
                text, action_id=action_id, context=prepared, generation=generation))
        except LanguagePolicyError as error:
            self._record_request_error(error, context)
            raise
        except Exception:
            self._record_request_error(
                LanguagePolicyError("speech_error", "TTS could not accept speech"), context)
            raise

    def _record_request_error(self, error, context=None):
        self.policy.record_error(error, direct_turn=getattr(context, "direct_turn", None),
                                 turn_id=getattr(context, "turn_id", None))

    def _synthesize_stream(self, text, context=None):
        if not isinstance(context, PreparedSpeech):
            raise LanguagePolicyError("unvalidated_request", "Missing prepared speech")
        view = _PreparedAdapterView(self._adapter, context)
        yield from type(self._adapter).synthesize_stream(view, text)

    def _should_log_topic_error(self, gate):
        # Keep the shared sampler's counters safe under concurrent callbacks.
        # Fixed outcomes avoid using user text/exception strings as state keys.
        with self._topic_log_lock:
            return gate.check("error")[0]

    def _text_callback(self, message):
        if self.state != "running":
            return
        generation = self._current_gen()
        context = None
        try:
            try:
                payload = json.loads(message.data)
            except (ValueError, TypeError):
                payload = {"text": message.data.strip()}
            if not isinstance(payload, dict):
                # json.loads accepts a bare year/number; it is still raw ROS
                # text, not a structured envelope.
                payload = {"text": message.data.strip()}
            text, context = self._prepare_topic(payload)
            self.enqueue(text, context=context, generation=generation)
        except LanguagePolicyError as error:
            self._record_request_error(error, context)
            if self._should_log_topic_error(self._topic_rejection_logs):
                log.warning("[vits2] topic speech rejected: %s", error.code)
        except Exception:
            self._record_request_error(
                LanguagePolicyError("speech_error", "TTS could not accept topic speech"), context)
            if self._should_log_topic_error(self._topic_failure_logs):
                log.exception("[vits2] topic speech failed")

    def status(self):
        return {**super().status(), **self.policy.snapshot()}


class TTSPlugin(BaseTTSPlugin):
    """Keep the current engine lifecycle; add per-instance language state."""

    def __init__(self, plugin_cfg, executor):
        validate_config(plugin_cfg)
        self._language_lock = threading.RLock()
        self._default_reply_language = reply_language_mode(plugin_cfg.get("reply_language", "auto"))
        self._policies = {}
        self._explicit_turns = set()
        super().__init__(plugin_cfg, executor)
        self._cfg["reply_language"] = self._default_reply_language

    def _policy(self, key):
        if key not in self._policies:
            self._policies[key] = LanguagePolicy(self._default_reply_language)
        return self._policies[key]

    def _new_node(self, key, input_topic, adapter):
        with self._language_lock:
            policy = self._policy(key)
        return _LanguageNode(input_topic or None, adapter, _node_suffix(key), policy,
                             lambda args: self._prepare_speech(args, key),
                             lambda node, request, gen, enqueue: self._accept_speech(
                                 key, node, request, gen, enqueue))

    def _accept_speech(self, key, node, request, generation, enqueue):
        # Match stop/language lock ordering. No frontend/GPU work or thread
        # joins occur here; enqueue only puts into the existing unbounded queue.
        with self._state_lock, self._language_lock, node.policy.lock, node._interrupt_lock:
            if (self._nodes.get(key) is not node or self._policies.get(key) is not node.policy
                    or node._stop_event.is_set()):
                raise LanguagePolicyError("stale_instance", "Speech instance has stopped")
            enqueue()
            if generation != node._interrupt_gen:
                # Keep the base worker's stale-generation ACP completion, but
                # cancelled frontend work must not establish language history.
                if isinstance(request, LiteralSpeechRequest):
                    node.policy.cancel_direct(request.direct_turn)
                return
            if isinstance(request, LiteralSpeechRequest):
                node.policy.accept_direct(request.direct_turn, request.language, request.language_source)
            else:
                node.policy.clear_error(turn_id=request.turn_id)

    def _forget_instances(self, instance_id):
        # The base stop detaches nodes under _state_lock. Only short policy
        # updates nest here; no language holder may acquire the state lock.
        with self._language_lock:
            if instance_id:
                self._policies.pop(instance_id, None)
                self._explicit_turns.discard(instance_id)
            else:
                self._policies.clear()
                self._explicit_turns.clear()

    def _speech_metadata(self, context):
        # Report the accepted request, even if a concurrent new turn arrives.
        if isinstance(context, LiteralSpeechRequest):
            turn = context.direct_turn
            with turn.owner.lock:
                if turn.accepted_language is None:
                    # Cold-start queued speech has not passed the frontend.
                    # Its neutral auto reading may follow an earlier success.
                    return {"language_pending": not turn.completed, **(
                        {"reply_language": turn.mode} if turn.mode != "auto" else {})}
                return {"reply_language": turn.accepted_language, "turn_id": turn.turn_id}
        return {"reply_language": context.language, "turn_id": context.turn_id}

    def _prepare_speech(self, args, instance_id):
        with self._language_lock:
            policy = self._policy(instance_id or "_default")
            try:
                text = args.get("text")
                if "turn_id" not in args and instance_id not in self._explicit_turns:
                    request = _direct_request(policy, text)
                else:
                    request = policy.prepare(text, args.get("turn_id"))
                return request.text, request
            except LanguagePolicyError as error:
                policy.record_error(error)
                raise

    def _config(self, args):
        validate_config(args)
        # Blank remote fields are sent by Darvin even for a local engine.
        args = {key: value for key, value in args.items()
                if not (key in {"api_key", "model", "voice", "url"} and value == "")}
        mode = reply_language_mode(args["reply_language"]) if "reply_language" in args else None
        speed = float(self._cfg.get("speed", 1.0))
        if "speed" in args:
            speed = float(args["speed"])
        # Empty/repeated configuration must not invalidate a loader or turn.
        if speed != float(self._cfg.get("speed", 1.0)):
            super()._config({"speed": speed})
        with self._language_lock:
            if mode is not None:
                key = args.get("instance_id")
                if key:
                    self._policy(key).configure(mode)
                else:
                    self._default_reply_language = mode
                    self._cfg["reply_language"] = mode
                    for policy in self._policies.values():
                        policy.configure(mode)
            default = self._default_reply_language
        result = {"status": "configured", "speed": speed, "speaker_id": 0,
                  "default_reply_language": default}
        if mode is not None:
            result["reply_language"] = mode
        return result

    def dispatch(self, name, args):
        action = args.get("action") if name == "tts" else name
        # Ordinary lifecycle/direct speech keeps the base TTS fallback for an
        # omitted/blank routing field. Only explicit language turns require
        # an instance and a matching turn token. Do not mutate caller args.
        blank_routing = {
            key for key in ("instance_id", "turn_id")
            if key in args and (args[key] is None
                                or isinstance(args[key], str) and not args[key].strip())
        }
        if blank_routing:
            args = {key: value for key, value in args.items() if key not in blank_routing}
        if "instance_id" in args and not isinstance(args["instance_id"], str):
            raise LanguagePolicyError("instance_required", "instance_id must be a nonempty string")
        if action == "language":
            key = args.get("instance_id")
            if not key:
                raise LanguagePolicyError("instance_required", "language requires instance_id")
            text, detected = _language_input(args)
            # Match stop's state -> language ordering. These are only policy
            # operations; never hold either lock across frontend/GPU/join work.
            with self._state_lock:
                if key not in self._nodes and key not in self._pending_starts:
                    raise LanguagePolicyError("unknown_instance", "Start the instance before language")
                with self._language_lock:
                    policy = self._policy(key)
                    try:
                        result = policy.begin_turn(text, detected, args.get("requested_reply_language"))
                        self._explicit_turns.add(key)
                        return result
                    except LanguagePolicyError as error:
                        policy.record_error(error)
                        raise
        if action == "speak" and not args.get("instance_id"):
            with self._state_lock:
                key = next((key for key, node in self._nodes.items() if node.state == "running"), "_default")
            args = {**args, "instance_id": key}
        if action == "start":
            key = args.get("instance_id") or args.get("input_topic") or "_default"
            with self._language_lock:
                policy = self._policy(key)
                if "reply_language" in args:
                    policy.configure(reply_language_mode(args["reply_language"]))
        result = super().dispatch(name, args)
        if action in ("info", "speak") and isinstance(result, dict):
            with self._language_lock:
                result["default_reply_language"] = self._default_reply_language
                key = args.get("instance_id")
                if key and action == "info":
                    policy = self._policies.get(key) or LanguagePolicy(self._default_reply_language)
                    result.update(policy.snapshot())
        return result

    def synthesize_raw(self, text):
        """HTTP audition uses the shared preference, with no conversation turn."""
        with self._language_lock:
            mode = self._default_reply_language
        with self._state_lock:
            adapter = self._adapter
            if adapter is None:
                self._request_load_locked()
                state, error = self._adapter_state, self._load_error
        if adapter is None:
            raise RuntimeError(f"VITS2 model not ready ({state}): {error or 'still loading'}")
        request = _direct_request(LanguagePolicy(mode), text)
        prepared = _prepare_chunks(adapter, request)
        view = _PreparedAdapterView(adapter, prepared)
        return b"".join(type(adapter).synthesize_stream(view, request.text))
