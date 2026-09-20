"""SmolVLA on this robot, via LeRobot.

**One provider per model, named after the model** — the same shape
`perception/plugins/` has, where `asr.py`, `tts.py` and `vop.py` each own their
weights, their download and their load. A single `local` provider would have had
to grow a switch over model families, and every model's quirks would have piled
up behind it: SmolVLA's action padding, π0's JAX stack, UnifoLM's flash-attn
build. They do not belong in one file.

What is shared lives in the card and in `providers/__init__.py`: discovery, the
four-method protocol, and the negotiation against the arm. What is specific to
*this checkpoint family* lives here.

This is also the only kind of provider whose resource use lands on the robot's
own budget, which is why almost everything below is about *when* things are
loaded rather than about inference.

Three rules from the design (phanthymotus/docs/vla-integration.md §"接一个新模型"), and
each is a thing that goes wrong if skipped:

**Lazy import.** torch and lerobot are imported inside the functions that need
them. A card nobody selected must not pull a GPU stack into the process, and
`providers/__init__.py` reports an import failure rather than raising — so an
actucore image without lerobot still offers `mock` and still starts.

**Lazy download.** Weights come from COS with size and SHA256 pinned, through
perception's `model_downloader`, which already handles the parts that are easy
to get wrong: a check file that must not appear until the model behind it is
complete, a lock so concurrent starts fetch one copy, archives staged before
they are merged. Not reimplemented here.

**Lazy load, and load off the calling thread.** `__init__` reads the
checkpoint's *config* — cheap, and enough to answer `capabilities()` so the card
can negotiate against the arm — then loads the weights in the background.
`health()` is False until they are in. Blocking `start` on a multi-second load
is how a card reports ready before it can act, which the TTS card already taught
this project once.

**What this cannot do for you.** A SmolVLA checkpoint is trained for a specific
robot, and its action dimension is that robot's. Pointing it at a 26-DOF
humanoid will fail negotiation, correctly and immediately — that is a
fine-tuning or retargeting problem, not a configuration one, and the error says
so rather than letting the mismatch reach a motor.
"""

from __future__ import annotations

import json
import logging
import os
import threading

log = logging.getLogger(__name__)

# The checkpoint layout LeRobot writes. `config.json` is read directly rather
# than through the library so that capabilities are available before torch is
# imported at all.
CONFIG_FILE = "config.json"
# The preprocessor pipeline LeRobot writes beside the weights. It carries a
# *second*, independent reference to the backbone — the tokenizer's — and
# staging the weights does not fix it. See `_write_resolved_config`.
PREPROCESSOR_FILE = "policy_preprocessor.json"


class SmolVLAProvider:
    """A SmolVLA checkpoint loaded in this process.

    Config keys:
        model_dir      where the checkpoint lives (default /models/vla/smolvla)
        model_id       upstream id, for the record — ModelScope first (see
                       docs/vla-integration.md §"接一个新模型"). Not fetched
                       from directly: the robot pulls from a pinned manifest.
        weights        optional {base_url, files:{name:{size,sha256}}} manifest;
                       fetched into model_dir when the checkpoint is absent
        vlm_dir        where the backbone lives (default
                       /models/vla/smolvlm2_500m)
        vlm_weights    manifest for the backbone. A separate upstream repo,
                       ~2 GB, shared by every SmolVLA checkpoint built on it —
                       see _ensure_backbone
        device         "cuda" | "cpu" (default cuda, falling back to cpu)
        feature_map    {our observation name: the policy's input key}, e.g.
                       {"main": "observation.images.top", "state":
                       "observation.state"}

    Note there is no `chunk_size` key here: the action horizon comes from the
    checkpoint and nothing else. See `_chunk_size`.
    """

    def __init__(self, descriptor: dict, config: dict | None = None,
                 on_status=None):
        config = dict(config or {})
        self._descriptor = descriptor or {}
        # Which checkpoint. `smolvla_base` is the published 6-DOF one; a version
        # fine-tuned for a robot is registered beside it under a name that says
        # which robot (`smolvla_tianyi`), because the action space it fits is the
        # thing that has to match the arm.
        self._model = str(config.get("model_name") or "").strip()
        entry = self._model_entry(config)
        self._model_dir = (entry.get("model_dir")
                           or config.get("model_dir")
                           or f"/models/vla/{self._model or 'smolvla'}")
        self._device = str(config.get("device") or "cuda")
        self._feature_map = dict(entry.get("feature_map")
                                 or config.get("feature_map") or {})
        self._weights = entry.get("weights") or config.get("weights") or {}
        # Which dataset's statistics un-normalise the action. Only needed by a
        # checkpoint whose statistics are grouped per dataset — see
        # `_select_stats`, which refuses to load rather than guess.
        self._unnorm_key = str(entry.get("unnorm_key")
                               or config.get("unnorm_key") or "").strip()
        self._vlm_dir = config.get("vlm_dir") or "/models/vla/smolvlm2_500m"
        self._vlm_weights = config.get("vlm_weights") or {}
        # Set by _ensure_backbone when the checkpoint has to be presented with a
        # local backbone path; None means load straight from model_dir.
        self._resolve_dir = None

        self._policy = None
        # The checkpoint's own pipelines. Not optional decoration: the policy
        # cannot be called without them (see `infer`).
        self._preprocessor = None
        self._postprocessor = None
        self._error = ""
        # Where the card's status line comes from while weights are moving.
        # These are the largest downloads anywhere in the system — a SmolVLA
        # checkpoint is ~900 MB and its backbone another ~1 GB — and the card
        # reported nothing at all for the whole of it.
        self._on_status = on_status
        self._lock = threading.RLock()
        self._closed = False

        self._require_lerobot()
        self._ensure_checkpoint()
        self._ensure_backbone()
        self._config = self._read_config()
        # Weights in the background: `capabilities()` is answerable from the
        # config alone, so negotiation can fail fast on a mismatched action
        # space without having waited for several gigabytes to load.
        self._loader = threading.Thread(target=self._load, name="vla-local-load",
                                        daemon=True)
        self._loader.start()

    # ── provider protocol ────────────────────────────────────────────────────

    def capabilities(self) -> dict:
        features = self._config.get("input_features") or {}
        image_keys = [k for k in features if "image" in k]
        return {
            # The configured checkpoint name, not the architecture family:
            # `smolvla` is true of every one of them and tells an operator
            # reading `info()` nothing about which weights are loaded.
            "model": self._model or self._config.get("type") or "smolvla",
            # SmolVLA 的动作是绝对关节角，反归一化之后就是弧度。声明它不是形式：
            # negotiate 现在会拒绝一个不声明动作空间的模型，因为维度相同不代表空间
            # 相同（一个 23 维的末端位姿模型和一张 23 维的关节卡片，数字完全吻合）。
            #
            # 写成常量而不是从 checkpoint 读，是因为 LeRobot 的 config 里没有这个
            # 概念 —— 它由**微调数据集**决定，而不是由架构决定。将来出现一个用速度
            # 或末端位姿微调的 SmolVLA，这里必须跟着改，而那时 negotiate 会先拦住它。
            "control_mode": "joint_position",
            "action_dim": self._action_dim(),
            "chunk_size": self._chunk_size(),
            "control_hz": float(self._config.get("fps") or 30.0),
            "needs_state": any("state" in k for k in features),
            "n_cameras": len(image_keys),
            "image_size": self._image_size(features, image_keys),
            # LeRobot ships RTC for the flow-matching policies, but this
            # provider does not implement the prefix conditioning it needs, so
            # it must not claim it — a card that believed this would hand it an
            # inference_delay nothing acts on.
            "supports_rtc": False,
            "ready": self._policy is not None,
            "error": self._error,
        }

    def infer(self, observation=None, inference_delay: int = 0) -> list:
        """One chunk, in the arm's units.

        The two pipeline calls are not optional plumbing, and leaving them out
        fails in two different ways — one loud, one silent, which is the
        dangerous one:

          `preprocessor` normalizes the observation **and tokenizes the task
          string**. SmolVLA reads `observation.language.tokens`, not `task`, so
          without this the call raises `KeyError` on every inference.

          `postprocessor` unnormalizes the action. Without it the chunk is in
          the policy's normalized space while everything downstream — the
          descriptor, the limits, the arm — reads radians. Negotiation passes,
          the message validates, and the numbers are simply wrong.
        """
        policy = self._policy
        preprocessor, postprocessor = self._preprocessor, self._postprocessor
        if policy is None or preprocessor is None or postprocessor is None:
            raise RuntimeError(self._error or "weights are still loading")

        batch = preprocessor(self._batch(observation))
        chunk = self._predict(policy, batch, postprocessor)
        return [[float(v) for v in step] for step in chunk]

    def health(self) -> bool:
        return self._policy is not None and not self._error

    def close(self) -> None:
        with self._lock:
            self._closed = True
            policy, self._policy = self._policy, None
            self._preprocessor = self._postprocessor = None
        if policy is None:
            return
        del policy
        # CUDA context is never returned to the OS; freeing the cache is all
        # that can be done from inside the process, which is why the design
        # would rather run this provider in its own process on a small board.
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:       # noqa: BLE001 — teardown is best effort
            pass

    # ── loading ──────────────────────────────────────────────────────────────

    def _model_entry(self, config: dict) -> dict:
        """The `models:` entry for the selected checkpoint.

        Refuses an unknown name rather than falling back to a default: silently
        loading a different checkpoint than the operator picked would produce a
        policy that runs, moves, and is wrong — the failure mode this whole
        negotiation path exists to avoid.

        An empty `models:` is allowed, for a single-checkpoint deployment that
        configures `model_dir`/`weights` directly.
        """
        models = config.get("models") or {}
        if not models:
            return {}
        if not self._model:
            raise ValueError(
                f"`model_name` is not set. This deployment stages "
                f"{sorted(models)}; pick one."
            )
        entry = models.get(self._model)
        if entry is None:
            raise ValueError(
                f"unknown model {self._model!r}. Staged here: {sorted(models)}. "
                f"A checkpoint has to be registered under `models:` with its own "
                f"weights manifest before it can be selected."
            )
        return dict(entry)

    @staticmethod
    def _require_lerobot():
        """Refuse at start on an image that cannot have lerobot, and say why.

        `find_spec` rather than an import: the point of this file is that torch
        and lerobot are only imported when something is actually going to run,
        and a presence check must not undo that.

        This is the normal state on the JetPack 5.11 image line, and it is not
        a packaging oversight — that line is CUDA 11.4, lerobot needs
        torch >= 2.2.1, and no torch >= 2.2 supports CUDA 11.4. Local inference
        cannot exist there, so the honest thing is to fail the start with the
        reason instead of loading in the background and reporting `unhealthy`
        forever.
        """
        import importlib.util

        if importlib.util.find_spec("lerobot") is None:
            raise ModuleNotFoundError(
                "lerobot is not installed in this image. On JetPack 5.11 that is "
                "expected and permanent: CUDA 11.4 cannot host torch >= 2.2.1, "
                "which lerobot requires — use a remote provider there. On "
                "JetPack 6.1 the actucore base image carries it; check the image "
                "was built from jetson-base-actucore."
            )

    def _ensure_checkpoint(self):
        """Fetch the checkpoint from COS if it is not already on disk.

        Absent a `weights` manifest this only checks: pinning a size and a
        SHA256 for a file nobody has staged yet would be inventing them, and an
        unpinned download of something that drives motors is not an improvement
        on a clear error.
        """
        if os.path.exists(os.path.join(self._model_dir, CONFIG_FILE)):
            return
        manifest = self._weights
        base_url, files = manifest.get("base_url"), manifest.get("files")
        if not base_url or not files:
            raise FileNotFoundError(
                f"no checkpoint at {self._model_dir} and no `weights` manifest "
                f"configured. Stage the checkpoint on COS (ModelScope first — "
                f"see docs/vla-integration.md §接一个新模型) and put its base_url plus "
                f"per-file size/sha256 in the card's `weights` config."
            )
        # perception's downloader: existing → size → sha256 → reuse, otherwise
        # lock, re-check, download with retry, verify, atomic replace.
        from model_downloader import ensure_verified_bundle
        from model_progress import fetch_status

        progress_cb, _ = fetch_status(self._on_status, self._model or "checkpoint")
        ensure_verified_bundle("vla-local", self._model_dir, base_url, files,
                               progress_cb=progress_cb)

    def _ensure_backbone(self):
        """Stage the VLM the checkpoint is built on, and point it at the copy.

        **A SmolVLA checkpoint is not self-contained.** Its config names a
        backbone — `vlm_model_name: HuggingFaceTB/SmolVLM2-500M-Video-Instruct`
        — which LeRobot fetches from HuggingFace while loading. On a robot that
        is a two-gigabyte download from a host it cannot reach, and it happens
        *after* the policy weights are already on disk, so everything looks
        staged right up until it fails:

            OSError: We couldn't connect to 'https://huggingface.co'

        Discovering that at deploy time is the deployer's problem to work around;
        discovering it here makes it ours, which is where it belongs.

        The pinned files are never rewritten. `vlm_model_name` has to become a
        local path for the library to stop reaching out, so the edited copy goes
        in a sidecar directory and the originals keep matching their SHA256 —
        otherwise the first load would invalidate the manifest that verifies it.
        """
        config = self._read_config()
        backbone = config.get("vlm_model_name") or ""
        if backbone and os.path.isdir(backbone):
            # Already a local path, so there is nothing to stage — but do not
            # return here. The preprocessor names the backbone a *second* time
            # and that copy is the one that actually broke the robot.
            self._vlm_dir = backbone
        elif backbone:
            manifest = self._vlm_weights
            base_url, files = manifest.get("base_url"), manifest.get("files")
            if os.path.exists(os.path.join(self._vlm_dir, CONFIG_FILE)):
                pass
            elif base_url and files:
                from model_downloader import ensure_verified_bundle
                from model_progress import fetch_status

                # Named after the backbone, not the checkpoint: the two are
                # separate downloads and one shared label would read as a
                # restart at 0%.
                progress_cb, _ = fetch_status(self._on_status,
                                              backbone.split("/")[-1])
                ensure_verified_bundle("vla-smolvla-backbone", self._vlm_dir,
                                       base_url, files, progress_cb=progress_cb)
            else:
                raise FileNotFoundError(
                    f"this checkpoint needs the {backbone!r} backbone, which "
                    f"LeRobot would fetch from HuggingFace at load time — "
                    f"unreachable from a robot. Stage it on COS the way the "
                    f"policy weights are staged and configure `vlm_weights`, "
                    f"or put it at {self._vlm_dir}."
                )

        if self._names_remote(config):
            self._resolve_dir = self._write_resolved_config(config)

    def _names_remote(self, config: dict) -> bool:
        """Does anything in this checkpoint still name a hub id?

        Two places do, independently, and staging the weights fixes neither on
        its own — which is why this asks about both rather than inferring one
        from the other.
        """
        backbone = config.get("vlm_model_name") or ""
        if backbone and not os.path.isdir(backbone):
            return True
        tokenizer = self._preprocessor_tokenizer()
        return bool(tokenizer and not os.path.isdir(tokenizer))

    def _preprocessor_tokenizer(self) -> str:
        """Whatever the preprocessor pipeline will try to load a tokenizer from."""
        for step in (self._read_preprocessor().get("steps") or []):
            if step.get("registry_name") == "tokenizer_processor":
                return str((step.get("config") or {}).get("tokenizer_name") or "")
        return ""

    def _read_preprocessor(self) -> dict:
        """The staged preprocessor pipeline, or {} — absent is a valid layout."""
        path = os.path.join(self._model_dir, PREPROCESSOR_FILE)
        try:
            with open(path, encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:               # noqa: BLE001
            return {}

    def _write_resolved_config(self, config: dict) -> str:
        """A load-time view of the checkpoint with every hub id made local.

        **Two files name the backbone, not one.** `config.json` names it for the
        policy, and `policy_preprocessor.json` names it again for the tokenizer
        step. Rewriting only the first gets the weights to load and then fails
        the moment the pipeline is built:

            ValueError: Failed to instantiate processor step
            'tokenizer_processor' ... We couldn't connect to
            'https://huggingface.co'

        which reads like a weights problem and is not one. Both are rewritten
        here for the same reason and in the same place, so neither can be fixed
        without the other.

        Hard links rather than copies for the weights: a second 900 MB file on a
        57 GB eMMC for the sake of two edited JSON fields is not a trade worth
        making. Falls back to a copy across filesystems.
        """
        preprocessor = self._read_preprocessor()
        # Everything rewritten below must be excluded from the linking pass, or
        # the link wins and the edit is never written.
        rewritten = {CONFIG_FILE} | ({PREPROCESSOR_FILE} if preprocessor else set())

        resolved = os.path.join(self._model_dir, ".resolved")
        os.makedirs(resolved, exist_ok=True)
        for name in os.listdir(self._model_dir):
            if name.startswith("."):
                continue
            source = os.path.join(self._model_dir, name)
            target = os.path.join(resolved, name)
            if name in rewritten or os.path.exists(target):
                continue
            if os.path.isdir(source):
                # A checkpoint is a flat directory of files, but nothing stops
                # one from carrying a subdirectory, and neither os.link nor
                # shutil.copy2 takes one — the load would die on IsADirectoryError
                # while the weights beside it were perfectly fine.
                continue
            try:
                os.link(source, target)
            except OSError:
                import shutil
                shutil.copy2(source, target)

        self._write_sidecar(resolved, CONFIG_FILE,
                            {**config, "vlm_model_name": self._vlm_dir})

        if preprocessor:
            for step in preprocessor.get("steps") or []:
                if step.get("registry_name") != "tokenizer_processor":
                    continue
                step.setdefault("config", {})["tokenizer_name"] = self._vlm_dir
            self._write_sidecar(resolved, PREPROCESSOR_FILE, preprocessor)
        return resolved

    @staticmethod
    def _write_sidecar(resolved: str, name: str, document: dict):
        """Write one rewritten file, replacing rather than opening any link.

        **Unlinking first is the whole point.** An earlier version of this file
        rewrote only `config.json`, so every other file — the preprocessor
        included — is already a *hard link* to the pinned original in every
        checkpoint staged before this change. `open(..., "w")` on a hard link
        truncates the shared inode, which would destroy the very file whose
        SHA256 proves the download is intact, and it would do so on the first
        load after an upgrade.
        """
        path = os.path.join(resolved, name)
        if os.path.exists(path):
            os.unlink(path)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)

    def _read_config(self) -> dict:
        path = os.path.join(self._model_dir, CONFIG_FILE)
        try:
            with open(path, encoding="utf-8") as handle:
                return json.load(handle)
        except Exception as error:      # noqa: BLE001
            raise RuntimeError(
                f"could not read {path}: {error}. capabilities() is derived "
                f"from it, and without it the card cannot check the model "
                f"against the arm before moving anything."
            ) from error

    def _load(self):
        """Background weight load. Failures are recorded, never raised here."""
        try:
            policy, preprocessor, postprocessor = self._build_policy()
        except Exception as error:      # noqa: BLE001 — reported via health()
            self._error = f"{type(error).__name__}: {error}"
            log.warning("smolvla provider failed to load: %s", self._error)
            return
        with self._lock:
            if self._closed:            # stopped while we were loading
                return
            self._policy = policy
            self._preprocessor = preprocessor
            self._postprocessor = postprocessor
        log.info("smolvla provider ready: %s on %s",
                 self._config.get("type"), self._device)

    def _build_policy(self):
        """Construct the LeRobot policy and the two pipelines that feed it.

        Kept to a few lines on purpose: everything else in this file works
        against plain dicts and is tested without torch, so when LeRobot's API
        moves this is the only thing to fix.

        `device_processor` is overridden rather than rewritten into the staged
        JSON, because the checkpoint hardcodes `cuda` and the device is decided
        at load time — `_resolved_device` falls back to cpu when there is no
        GPU, and a staged file would contradict it.
        """
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        source = self._resolve_dir or self._model_dir
        device = self._resolved_device()
        policy = SmolVLAPolicy.from_pretrained(source)
        policy.to(device)
        policy.eval()
        preprocessor, postprocessor = make_pre_post_processors(
            policy.config, pretrained_path=source,
            preprocessor_overrides={"device_processor": {"device": device}})
        self._apply_norm_stats(preprocessor, postprocessor)
        return policy, preprocessor, postprocessor

    def _apply_norm_stats(self, *pipelines) -> None:
        """Point the normalizer steps at one dataset's statistics.

        A no-op for a checkpoint whose statistics are already keyed `action`.
        Split from `_select_stats` because only these four lines touch LeRobot:
        the decision is a dict transform and is tested without the library.

        The tensor cache is the part that is easy to miss. `NormalizerProcessorStep`
        converts `stats` into `_tensor_stats` in `__post_init__`, and `__call__`
        reads **only** the latter — so assigning `step.stats` alone changes
        nothing at all, silently, and the un-normalisation stays an identity.
        """
        from lerobot.processor.normalize_processor import to_tensor

        for pipeline in pipelines:
            for step in getattr(pipeline, "steps", []):
                chosen = self._select_stats(getattr(step, "stats", None), self._unnorm_key)
                if chosen is None:
                    continue
                step.stats = chosen
                step._tensor_stats = to_tensor(      # noqa: SLF001
                    chosen, device=step.device, dtype=step.dtype)
                log.info("smolvla norm stats: using %r → keys %s",
                         self._unnorm_key, sorted(chosen))

    @staticmethod
    def _select_stats(stats: dict | None, unnorm_key: str) -> dict | None:
        """One dataset's statistics, keyed the way the pipeline looks them up.

        Returns None when there is nothing to do — no stats, or already keyed
        `action`, which is what a checkpoint fine-tuned on one robot looks like.

        ── Why this has to exist ────────────────────────────────────────────
        `smolvla_base` is a **multi-dataset pretrained base**. Its statistics
        are grouped by the dataset they came from:

            so100.buffer.action / so100-blue.buffer.action / so100-red.buffer.action

        and the pipeline looks up plain `action`. A miss is not an error in
        LeRobot: the step simply passes the tensor through. So the policy's
        normalised output — roughly ±1 — is handed to the descriptor, the
        limits and the arm, all of which read degrees. Negotiation passes, the
        message validates, the numbers are wrong by two orders of magnitude,
        and nothing is logged.

        Measured on Tianyi 2026-09-20, before this fix: the card's output was
        |max| ≈ 0.9 while `so100`'s own statistics are mean ∈ [-27, 120],
        std ∈ [19, 59]. Commanding an arm whose joints span tens of degrees to
        a target near zero is not "a smaller motion" — it drives it to a limit.

        ── Why refusing is the right failure ────────────────────────────────
        There is no safe default. Picking the wrong group yields well-formed
        actions of the wrong magnitude, which is exactly the failure this whole
        negotiation path exists to prevent, so an unset key refuses to load and
        the card reports the error instead of driving anything.

        This is the same decision OpenVLA calls `unnorm_key`, and the cloud
        runtime (`phanthymotus-cloud`, runtimes/smolvla) makes it under that
        same name. The two were compared on the same checkpoint and the same
        inputs: with the key applied, their sampled action distributions agree
        to 0.35 of the sampling spread. Without it, one of them is in a
        different space entirely.
        """
        import re

        if not stats or "action" in stats:
            return None
        groups = sorted({key.split(".")[0] for key in stats})
        if not unnorm_key:
            raise ValueError(
                f"this checkpoint's normalisation statistics are grouped per "
                f"dataset ({', '.join(groups)}) and none is named `action`: it is "
                f"a pretrained base, not a model that can drive an arm as-is. Set "
                f"`unnorm_key` to one of them, or register a checkpoint fine-tuned "
                f"on your own data. Left unset, un-normalisation silently becomes "
                f"an identity: the actions are well-formed and off by two orders "
                f"of magnitude."
            )
        if unnorm_key not in groups:
            raise ValueError(
                f"unnorm_key={unnorm_key!r} is not in this checkpoint. "
                f"Available: {', '.join(groups)}"
            )
        prefix = re.compile(rf"^{re.escape(unnorm_key)}\.(?:buffer\.)?")
        return {prefix.sub("", key): value
                for key, value in stats.items() if key.startswith(unnorm_key + ".")}

    def _resolved_device(self) -> str:
        if self._device != "cuda":
            return self._device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except Exception:       # noqa: BLE001
            pass
        log.warning("cuda unavailable; smolvla provider falling back to cpu")
        return "cpu"

    # ── inference ────────────────────────────────────────────────────────────

    def _batch(self, observation) -> dict:
        """Our observation, in the keys this checkpoint was trained on.

        The mapping is configuration rather than convention because the keys
        travel with the training dataset — `observation.images.top` on one
        checkpoint and `observation.images.cam_high` on the next — and guessing
        produces a policy acting on a black image rather than an error.
        """
        import torch

        if observation is None:
            raise ValueError("smolvla provider needs an observation")

        expected = set(self._config.get("input_features") or {})
        batch, missing = {}, []
        device = self._resolved_device()

        for name, image in (getattr(observation, "images", None) or {}).items():
            key = self._feature_map.get(name)
            if not key:
                continue
            batch[key] = torch.as_tensor(image).to(device)

        state = getattr(observation, "state", None)
        if state is not None:
            key = self._feature_map.get("state", "observation.state")
            batch[key] = torch.as_tensor(state).to(device)

        for key in expected:
            if key not in batch:
                missing.append(key)
        if missing:
            raise KeyError(
                f"this checkpoint expects {sorted(missing)} and the card's "
                f"feature_map does not supply them. feature_map maps our "
                f"observation names to the policy's input keys."
            )

        batch["task"] = getattr(observation, "prompt", "") or ""
        return batch

    @staticmethod
    def _predict(policy, batch, postprocessor) -> list:
        """One chunk out of the policy. The second version-sensitive call.

        `predict_action_chunk` is what LeRobot's async and RTC paths use;
        `select_action` is the single-step fallback for a policy that has no
        chunk method, wrapped so the caller always sees a chunk.

        The postprocessor runs before the tensor leaves the GPU, and before the
        shape is normalized: it is part of producing the action, not a
        formatting step applied to one.
        """
        import torch

        with torch.no_grad():
            if hasattr(policy, "predict_action_chunk"):
                chunk = policy.predict_action_chunk(batch)
            else:
                chunk = policy.select_action(batch)

        chunk = postprocessor(chunk)
        chunk = chunk.detach().to("cpu")
        # (B, T, D) → (T, D); (T, D) stays; (D,) becomes one step.
        if chunk.ndim == 3:
            chunk = chunk[0]
        elif chunk.ndim == 1:
            chunk = chunk.unsqueeze(0)
        return chunk.tolist()

    # ── capabilities, from the checkpoint's own config ───────────────────────

    def _action_dim(self):
        """The checkpoint's action width, not the network's.

        SmolVLA pads its action tensor to a fixed internal maximum, so the
        network's width says nothing about the robot it was trained on. What
        matters is the `action` output feature's shape, which carries the
        dataset's real dimension — and that is the number negotiation must
        compare against the arm.
        """
        features = self._config.get("output_features") or {}
        action = features.get("action") or {}
        shape = action.get("shape")
        if isinstance(shape, (list, tuple)) and shape:
            return int(shape[-1])
        return None

    def _chunk_size(self):
        """The checkpoint's action horizon, and only ever the checkpoint's.

        Deliberately not overridable from config. `chunk_size` is also the
        *mock* provider's tuning knob — config.yaml sets it to 10 for the sine
        generator — and the config dict handed to a provider is flat and shared
        by all of them. Reading it here let the mock's knob silently replace the
        real horizon: smolvla_base returns chunks of 50 and advertised 10.
        """
        for key in ("n_action_steps", "chunk_size", "horizon"):
            value = self._config.get(key)
            if isinstance(value, int) and value > 0:
                return value
        return 1

    @staticmethod
    def _image_size(features: dict, image_keys: list):
        for key in image_keys:
            shape = (features.get(key) or {}).get("shape")
            if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                return int(shape[-1])
        return 0


def PROVIDER(descriptor: dict, config: dict | None = None,
             on_status=None) -> SmolVLAProvider:
    return SmolVLAProvider(descriptor, config, on_status=on_status)


# Discovery checks the four methods on whatever PROVIDER is; a factory function
# has none of them, so they are advertised here.
for _name in ("capabilities", "infer", "health", "close"):
    setattr(PROVIDER, _name, getattr(SmolVLAProvider, _name))

# Picks from the checkpoints staged under `models:`, so the card can offer them
# as a list — the names have to match a config key exactly, and typing one is
# the kind of thing a dropdown exists to prevent.
PROVIDER.MODEL_NAMES = "staged"
