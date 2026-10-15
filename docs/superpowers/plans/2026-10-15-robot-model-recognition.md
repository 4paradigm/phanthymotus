# Robot Model Recognition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in, entirely local Perception card that identifies supported robot models in photos and camera frames, and says `unknown` when visual evidence is inadequate.

**Architecture:** Reuse VOP's loaded detector for robot boxes; pass each crop to a CLIP ViT-B/32 image encoder hosted in the existing isolated ORT worker. Compare its normalized embedding with offline-generated candidate text vectors, then apply calibrated score and margin gates. One inference function serves one-shot and streaming actions.

**Tech Stack:** Python, ROS 2 `CompressedImage`/`String`, TensorRT VOP, ONNX Runtime worker, NumPy, pytest.

**Spec:** `docs/superpowers/specs/2026-10-15-robot-model-recognition-design.md`

## Global Constraints

- Inference and model files stay on the robot; no cloud vision call.
- Candidate labels are Driver IDs, not a Driver selection or control authority.
- A generic robot detection or an uncalibrated score never yields a specific model.
- Use `plugins.image_input.load_image_bytes` for photo and URL input.
- Keep the plugin disabled by default until held-out accuracy and both JetPack lines are measured.
- Do not create an ORT session in the Perception parent process.

## Review Focus

- Two robots in one photo: each box gets its own classification (Task 4 test).
- A robot absent from the candidate list: result remains `unknown` (Task 1 test).
- Similar top two scores: margin gate withholds the label (Task 1 test).
- Broken or missing model bundle: `engine_unavailable`, not a guessed label (Task 2/4 tests).
- A camera frame racing `stop`: no orphan publisher or stale result (Task 5 test).

---

### Task 1: Candidate catalogue and abstaining decision

**Files:**
- Create: `perception/plugins/robot_model_catalog.py`
- Create: `perception/tests/test_robot_model_catalog.py`

**Interfaces:**
- Produce `CANDIDATES: tuple[tuple[str, str], ...]`, `(Driver ID, English visual prompt)`, copied from `phanthymotus-driver/README.md` at revision `44a74e5e80597cb3682be60b6a4ebe9dbac1a43f`, documented in the file.
- Produce `decide_model(scores: dict[str, float], min_score: float | None, min_margin: float | None) -> dict` with `model_id`, `score`, `margin`, `reason`, and top candidate scores. `None` gates mean uncalibrated and always `unknown`.

- [ ] Write tests asserting every catalogue ID matches the Driver README snapshot, empty scores return `unknown`, an unlisted candidate ID cannot be returned, missing gates return `unknown`, and score or margin below a configured gate returns `unknown`.
- [ ] Run `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest perception/tests/test_robot_model_catalog.py -q`; confirm the missing module/function failure.
- [ ] Implement the catalogue and the smallest pure scoring function. Only finite scores and IDs in `CANDIDATES` are eligible.
- [ ] Run the focused test and commit `feat(perception): define robot model candidates and abstention`.

### Task 2: Local CLIP artifact and isolated embedding service

**Files:**
- Create: `perception/tools/export_robot_model_clip.py`
- Create: `perception/plugins/robot_model_service.py`
- Create: `perception/plugins/robot_model_proxy.py`
- Modify: `perception/utils/model_downloader.py`
- Create: `perception/tests/test_robot_model_service.py`
- Create: `perception/tests/test_robot_model_bundle.py`

**Interfaces:**
- Produce `export_bundle(checkpoint_path: str, output_dir: str) -> dict` using the official OpenAI CLIP ViT-B/32 checkpoint: image-only ONNX, normalized candidate text vectors, model metadata, and SHA256 manifest. Reject catalogue/vector order drift.
- Produce worker service `RobotModelService(model_dir: str, providers: list[str])` with `describe() -> dict` and `embed(image_bytes: bytes) -> list[float]` or a structured undecodable-image error.
- Produce parent `RobotModelProxy(model_dir: str, on_status=None)` with `embed(image_bytes: bytes) -> list[float]` and `close()`. It uses `ort_worker.get_worker().service`, not parent-process ORT.
- Produce `ensure_robot_model_bundle(model_dir: str, progress_cb=None) -> dict[str, str]`, checking pinned size and SHA256 before load. Keep runtime download opt-in and report a missing artifact explicitly.

- [ ] Write tests for vector order/normalization, crop preprocessing parity with CLIP, a bad hash, missing bundle, undecodable crop, and proof the parent proxy never imports `onnxruntime`.
- [ ] Run both focused test files; confirm feature-missing failures.
- [ ] Implement export, verified bundle loading, service, and proxy. Validate exported ONNX output against the source checkpoint on fixed inputs before pinning hashes; keep generated weights out of Git.
- [ ] Run focused tests and an actual ONNX versus source-model parity probe; record the commands and measured discrepancy in the commit/README. Commit `feat(perception): add local robot image encoder`.

### Task 3: Share VOP detection without a second engine

**Files:**
- Modify: `perception/plugins/vop.py`
- Modify: `perception/tests/test_vop_plugin.py`

**Interfaces:**
- Produce `VideoObjectPerceptionPlugin.detect_frame(frame: np.ndarray, confidence: float) -> list[dict]` returning the existing box/class/score shape without publishing. The current one-shot path calls it too; it does not load another engine.

- [ ] Write a test where one fake inference contains a person and two robot classes: `detect_frame` returns the established VOP results, and the existing one-shot action still returns its prior shape.
- [ ] Run the focused test; confirm it fails because `detect_frame` is absent.
- [ ] Extract only the shared inference/decoding path; keep VOP's existing payload and topic behavior intact.
- [ ] Run `test_vop_plugin.py` and commit `refactor(perception): expose VOP frame detection for robot model card`.

### Task 4: One-shot `robot_model` card

**Files:**
- Create: `perception/plugins/robot_model.py`
- Modify: `perception/main.py`
- Modify: `perception/config.yaml`
- Create: `perception/tests/test_robot_model_plugin.py`

**Interfaces:**
- Produce `RobotModelPlugin(plugin_cfg: dict, detector: VideoObjectPerceptionPlugin | None, executor)` with tool prefix `robot_model` and `recognize_by_photo`, `recognize_by_url`, `info`, `start`, `stop` actions.
- Produce `recognize_frame(frame: np.ndarray) -> dict` shared by photo and stream: filter VOP detections to robot classes; clamp each box, encode each crop, classify with Task 2's proxy and Task 1's gates. Every object has `bbox`, `detector_class`, `detector_score`, `model_id`, `score`, `margin`, and `reason`.

- [ ] Write tests for two robots with different predictions, no robot, invalid image, tiny/invalid box, unknown/ambiguous model, model bundle failure, and photo/URL input through the shared guard.
- [ ] Run the focused test; confirm the card/action is missing.
- [ ] Implement the opt-in card and main registration. With VOP disabled, `info` explains the detector dependency and recognition returns `engine_unavailable`.
- [ ] Run focused plugin tests plus VOP tests; commit `feat(perception): add robot model photo recognition card`.

### Task 5: Camera stream and lifecycle

**Files:**
- Modify: `perception/plugins/robot_model.py`
- Modify: `perception/tests/test_robot_model_plugin.py`

**Interfaces:**
- `start(input_topic: str)` subscribes to `image/jpeg` and publishes bounded-rate `data/json` to `{input_topic}/robot_models`; `stop` disposes the subscriber and publisher. Empty topic retains the on-demand card.

- [ ] Write tests for frame publication, FPS limiting, per-instance topic isolation, stop racing a queued frame, and latest-state `info`.
- [ ] Run the tests; confirm missing stream behavior.
- [ ] Implement with the repository's `LatestFrame`, QoS, and node-disposal pattern; call the same `recognize_frame` as Task 4.
- [ ] Run focused tests and commit `feat(perception): stream robot model results`.

### Task 6: Evaluation and operator guidance

**Files:**
- Create: `perception/tools/eval_robot_model.py`
- Create: `perception/tests/test_eval_robot_model.py`
- Modify: `perception/README.md`

**Interfaces:**
- Evaluation input is a manifest of image path, visible boxes/Driver IDs or `unknown`, calibration/test split, and physical-unit IDs. Derive score/margin gates from calibration only; report per-model recall, unknown false-identification rate, and latency on the disjoint test split.

- [ ] Write tests proving split/physical-unit overlap is rejected, unknown false identifications count correctly, and a model with no held-out examples receives no accuracy claim.
- [ ] Run the focused tests; confirm missing evaluator.
- [ ] Implement the smallest evaluator and document collecting photos, generating/pinning bundles, calibration, running the card, and the opt-in flag. Do not set non-null production gates without measured held-out evidence.
- [ ] Run evaluator tests and the full Perception suite in a compatible environment; record any environment-only failures by name. On actual JP5.11 and JP6.1 hardware, verify model load, provider, latency, memory, and false-identification metrics before enabling by default. Commit `docs(perception): add robot model evaluation workflow`.

## Completion check

Re-read the spec against the six task results. Run `git diff origin/main --check`, inspect every changed file, and report exact test counts and any missing model asset or device evidence. A branch with code but without held-out evaluation remains an opt-in prototype, not a validated recognizer.
