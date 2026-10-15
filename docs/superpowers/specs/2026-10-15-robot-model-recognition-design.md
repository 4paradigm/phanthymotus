# Robot model recognition card

## Intent and scope

Identify the make and model of robots **seen by a camera**, not the host robot's
own Driver. Inference runs entirely on the robot. The first version covers the
models named in the current `phanthymotus-driver` catalogue; it may return
`unknown` for any image, including an unlisted or visually ambiguous model.
It must never turn a generic `robot` detection into a specific model claim.

Success means a Perception card accepts one photo or a camera topic, reports
each visible robot's box and either a supported model or `unknown`, and exposes
enough evidence to evaluate false identifications. A model name is not a
control-plane identity and must not select a Driver or authorize movement.

## Chosen approach

1. Reuse the existing VOP detector and its `robot`, `quadruped robot`, and
   `humanoid robot` classes to find candidate regions. Keep this stage separate
   from model identification; VOP's current frozen vocabulary contains no
   specific model names.
2. Encode each robot crop locally with an OpenAI CLIP ViT-B/32 image encoder
   exported to ONNX. Build the text embeddings for the fixed candidate list
   offline; the robot does not load a text encoder or PyTorch. The model bundle
   includes the ONNX file, candidate embeddings, provenance, and pinned hashes.
   Run the image encoder in the existing isolated ONNX Runtime process used by
   other Perception models, to avoid mixing incompatible ONNX Runtime libraries
   in the Perception process.
3. Compare normalized image and candidate vectors. Return the highest-scoring
   model only if its score and lead over the runner-up pass thresholds calibrated
   on held-out photos. Until that calibration exists, report `unknown` with
   ranked candidate scores; never ship guessed thresholds as reliable identity.

The alternative of OCR on model badges misses unlabelled and occluded robots.
Training a dedicated classifier may outperform zero-shot CLIP but requires a
larger labelled corpus; use evaluation results to decide whether to pursue it.

## Card contract

Add one `robot_model` Perception card. It accepts `image/jpeg` camera input and
the established `recognize_by_photo` / `recognize_by_url` actions through the
shared image-input guard. One-shot actions work without a camera. Camera mode
publishes bounded-rate `data/json` results and latest-state information. Each
result contains `bbox`, detector class and score, `model_id` or `unknown`,
classification score, margin, and `reason` when withheld. The card reports
model-load errors explicitly without taking down other Perception cards.

The candidate IDs are the current Driver identifiers, including Unitree,
Noetix, Robotera, Deep Robotics, EngineAI, PNPbotics, DJI, and Chasing entries.
Similar variants may collapse to `unknown` where visual evidence is
insufficient; results must not silently merge distinct Driver IDs.

## Delivery and verification

Implement the single-photo path first, then the camera stream using the same
inference function. Tests cover multiple robots, no robot, invalid image,
unlisted robot, ambiguous top two candidates, and unavailable model artifacts.
Use a held-out set with different viewpoints, backgrounds, and physical units
from the reference or calibration images. Report per-model recall, false
identification rate on unknown models, and latency/memory on both supported
JetPack lines. Publish numerical acceptance claims only after these checks;
desktop unit tests alone establish the card contract, not recognition accuracy.

Model files are fetched as verified, immutable bundles, using the repository's
existing bundle mechanism. A missing bundle yields `engine_unavailable`, never
a fabricated model label. The first deployment stays opt-in (`enabled: false`)
until the held-out evaluation and device measurements support enabling it.
