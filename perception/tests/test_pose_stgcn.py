"""
tests/test_pose_stgcn.py — the ST-GCN++ skeleton-action backend.

The engine is injected, so everything here runs on a laptop with no TensorRT,
no checkpoint and no GPU. What is covered is the part that fails *silently* if
it is wrong: the normalisation arithmetic, the uniform sampling, the tensor
layout, the probability/logit handling, the NTU-60 label mapping, and the
failure paths that must not turn into a label.

What is NOT covered, and cannot be from here: whether ST-GCN++ actually agrees
with these labels on a real robot. There is no published engine yet. Treat a
green run as "the plumbing is right", not as "the model works".

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest perception/tests -q
"""

from __future__ import annotations

import numpy as np
import pytest

import vision_stubs  # noqa: F401

from plugins.pose_action import (  # noqa: E402
    RULE_TO_ACTIVITY,
    ACTION_PRIORITY,
    PoseActionClassifier,
    PoseFrame,
)
from plugins.pose_stgcn import (  # noqa: E402
    DEFAULT_MIN_SCORE,
    resample_clip,
    DEFAULT_WINDOW_FRAMES,
    NUM_CHANNELS,
    NUM_PERSON_SLOTS,
    PRENORM_SCORE_THRESHOLD,
    FALL_CLASS,
    MUTUAL_CLASSES,
    NTU60,
    NTU_TO_POSE_LABEL,
    USABLE_CLASSES,
    action_vocabulary,
    ntu_name,
    ActionBackendError,
    HybridActionBackend,
    SkeletonActionBackend,
    build_backend,
    looks_like_probabilities,
    pre_normalize_2d,
    softmax,
    uniform_sample_indices,
)
from plugins.vision_runtime import N_KEYPOINTS  # noqa: E402
from test_pose_action import (  # noqa: E402
    CX, H, TOP, _body, _kp, _lying_body, _lying_box, _standing_box,
)

FRAME = (1280, 720)

# -- reading the two-channel result ------------------------------------------
# posture and activity are separate channels; these express the old flattened
# view for assertions that only care about "in one word".

ACTIVITY_TO_RULE = {v: k for k, v in RULE_TO_ACTIVITY.items()}


def _action(result):
    activity = result.get("activity")
    if isinstance(activity, dict):
        return ACTIVITY_TO_RULE.get(activity["name"], activity["name"])
    if isinstance(activity, str):
        return ACTIVITY_TO_RULE.get(activity, activity)
    return result.get("posture") or "unknown"


def _labels(result):
    out = set()
    if result.get("posture"):
        out.add(result["posture"])
    name = _action(result)
    if name != "unknown":
        out.add(name)
    return out


N_CLASSES = 60


class _FakeEngine:
    """Stands in for TensorRTEngine: infer(blob) -> per-class scores."""

    def __init__(self, scores=None, *, window_frames=DEFAULT_WINDOW_FRAMES,
                 outputs=None, raise_on_infer=None):
        self.input_shape = (1, NUM_PERSON_SLOTS, window_frames,
                            N_KEYPOINTS, NUM_CHANNELS)
        self.calls = []
        self._raise = raise_on_infer
        if outputs is not None:
            self._outputs = outputs
        else:
            values = np.zeros(N_CLASSES, dtype=np.float32)
            for index, value in (scores or {}).items():
                values[index] = value
            self._outputs = [values[None]]

    def infer(self, blob):
        if self._raise:
            raise self._raise
        self.calls.append(np.asarray(blob).copy())
        return self._outputs


def _frames(joints_fn, *, n=30, fps=10.0, box=None, image_size=FRAME,
            static=False):
    """A clip. **Moving by default**, because a static one is now refused.

    The backend abstains below `MIN_MOTION` rather than asking the model, since
    "no action" is not a class NTU-60 has and a frozen clip gets a confident
    wrong answer instead of an unsure one. Tests that want the model consulted
    therefore need a clip with motion in it; `static=True` is for the tests of
    the abstention itself.
    """
    out = []
    for i in range(n):
        t = i / fps
        joints = joints_fn(t)
        if not static and n > 1:
            # A plain drift: enough to clear MIN_MOTION on the normalised
            # tensor, but under the geometry's `still_speed`, so it neither
            # colours what the fake engine is asked nor trips the rules that
            # require a limb to be stationary (pointing needs the other arm
            # still).
            shift = (i / (n - 1)) * 0.05 * H
            joints = {k: (v[0], v[1] - shift) for k, v in joints.items()}
        out.append(PoseFrame(t, box or _standing_box(), _kp(**joints),
                             0.3, image_size=image_size))
    return out


def _standing(_t):
    return _body()


# ── preprocessing: the part that fails silently ─────────────────────────────

def test_fix_mode_maps_the_frame_onto_minus_one_to_one():
    """`fix` is what the checkpoint was trained with: normalise by the FRAME.

    Where someone stands and how large they appear within the frame is
    information the network trained with; re-centring on the body would discard
    it. Pinned on exact corners because a wrong scale raises nothing — it just
    feeds the model a skeleton from a distribution it never saw.
    """
    corners = np.array([[[0.0, 0.0, 0.9], [1280.0, 720.0, 0.9],
                         [640.0, 360.0, 0.9]]], dtype=np.float32)
    out = pre_normalize_2d(corners, FRAME, "fix")
    assert out[0, 0, :2].tolist() == pytest.approx([-1.0, -1.0])
    assert out[0, 1, :2].tolist() == pytest.approx([1.0, 1.0])
    assert out[0, 2, :2].tolist() == pytest.approx([0.0, 0.0])


def test_fix_mode_refuses_a_zero_frame_size():
    with pytest.raises(ActionBackendError, match="unusable"):
        pre_normalize_2d(np.zeros((1, 17, 3), dtype=np.float32), (0, 720), "fix")


def test_an_unknown_normalisation_mode_is_refused():
    with pytest.raises(ActionBackendError, match="unknown normalisation"):
        pre_normalize_2d(np.zeros((1, 17, 3), dtype=np.float32), FRAME, "centred")


def test_auto_mode_is_scale_invariant():
    """The reason it is the default. Measured on a real falling skeleton at four
    distances, scoring A43 "falling down":

        fills ~42% of frame   fix 0.950   auto 0.948
        half that             fix 0.600   auto 0.948
        a quarter             fix 0.074   auto 0.948
        an eighth             fix 0.003   auto 0.948

    NTU's subjects all fill a similar fraction of frame, so a skeleton from
    further away lands outside the distribution under `fix`. A robot sees people
    across a room.
    """
    body = np.zeros((4, N_KEYPOINTS, 3), dtype=np.float32)
    body[:, :, 2] = 0.9
    body[:, :, 0] = np.linspace(400, 600, N_KEYPOINTS)
    body[:, :, 1] = np.linspace(200, 700, N_KEYPOINTS)
    near = pre_normalize_2d(body, FRAME, "auto")
    far = body.copy()
    far[:, :, :2] = (far[:, :, :2] - 500.0) * 0.2 + 500.0     # same pose, 5x away
    assert np.allclose(near[..., :2], pre_normalize_2d(far, FRAME, "auto")[..., :2],
                       atol=1e-4)
    # fix, by contrast, shrinks with the person.
    assert not np.allclose(pre_normalize_2d(body, FRAME, "fix")[..., :2],
                           pre_normalize_2d(far, FRAME, "fix")[..., :2], atol=1e-2)


def test_auto_mode_needs_no_frame_size():
    """It derives its scale from the skeleton, so a frame it has no use for
    must not be a reason to refuse the clip."""
    engine = _FakeEngine({42: 9.0})
    frames = _frames(_standing, n=20, image_size=None)
    result = SkeletonActionBackend(engine=engine).classify(frames)
    assert result.get("backend_error") is None
    assert engine.calls, "auto mode should have reached the engine"


def test_uniform_sampling_covers_the_clip_evenly():
    """PYSKL's UniformSample, made deterministic: bin the clip and take each
    bin's centre. Random offsets are for training; a robot wants the same
    answer twice for the same input."""
    assert uniform_sample_indices(48, 48).tolist() == list(range(48))
    short = uniform_sample_indices(6, 12)
    assert short.tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    long = uniform_sample_indices(96, 4)
    assert long.tolist() == [12, 36, 60, 84]


def test_uniform_sampling_never_indexes_past_the_clip():
    for available in (1, 2, 5, 31, 100):
        indices = uniform_sample_indices(available, 48)
        assert indices.min() >= 0 and indices.max() < available


def test_sampling_an_empty_sequence_raises_rather_than_returning_nothing():
    with pytest.raises(ActionBackendError):
        uniform_sample_indices(0, 48)


def test_the_input_tensor_matches_the_published_config():
    """(N, M, T, V, C) — FormatGCNInput's order, with M and C taken from the
    config the checkpoint was trained under, not chosen.

    `FormatGCNInput(num_person=2)` means two person slots with a single person
    zero-padded into the second; `backbone.data_bn.weight` has 51 = 3 x 17
    entries, so C is x, y **and score**. An earlier version built
    (1, 1, T, 17, 2) — wrong in two dimensions, which is exactly the kind of
    mistake that loads without complaint and returns confident nonsense.
    """
    engine = _FakeEngine({42: 9.0}, window_frames=48)
    backend = SkeletonActionBackend(engine=engine)
    backend.classify(_frames(_standing, n=30))
    assert engine.calls, "the engine was never called"
    assert engine.calls[0].shape == (1, 2, 48, N_KEYPOINTS, 3)
    assert engine.calls[0].dtype == np.float32
    assert NUM_PERSON_SLOTS == 2 and NUM_CHANNELS == 3


def test_the_unused_person_slot_is_zero():
    """`mode='zero'`, not `'loop'`: the second slot is padding, not a copy."""
    engine = _FakeEngine({42: 9.0})
    SkeletonActionBackend(engine=engine).classify(_frames(_standing, n=30))
    blob = engine.calls[0]
    assert np.any(blob[0, 0] != 0), "the real person must not be empty"
    assert np.all(blob[0, 1] == 0), "the padded slot must be all zeros"


def test_the_default_window_is_the_one_the_checkpoint_was_trained_with():
    """UniformSample(clip_len=100) in the published config. Not ours to pick."""
    assert DEFAULT_WINDOW_FRAMES == 100


def test_a_joint_below_the_prenorm_threshold_has_its_position_zeroed():
    """Part of the transform the weights were trained under, not our visibility
    gate: PreNormalize2D zeroes x and y while keeping the score, and skipping it
    feeds the network coordinates it was taught to read as absent."""
    keypoints = np.zeros((1, N_KEYPOINTS, 3), dtype=np.float32)
    keypoints[0, :, 0] = 640.0
    keypoints[0, :, 1] = 360.0
    keypoints[0, 0, 2] = 0.9                      # visible
    keypoints[0, 1, 2] = PRENORM_SCORE_THRESHOLD  # at the threshold -> absent
    out = pre_normalize_2d(keypoints, FRAME, "fix")
    assert out[0, 0, :2].tolist() == pytest.approx([0.0, 0.0])   # frame centre
    assert out[0, 1, :2].tolist() == [0.0, 0.0]                  # zeroed
    assert out[0, 1, 2] == pytest.approx(PRENORM_SCORE_THRESHOLD)  # score kept


def test_the_window_comes_from_the_engine_not_from_our_constant():
    """An ST-GCN export has a fixed temporal dimension; ours is only a fallback."""
    engine = _FakeEngine({42: 9.0}, window_frames=100)
    SkeletonActionBackend(engine=engine).classify(_frames(_standing, n=30))
    assert engine.calls[0].shape[2] == 100


def test_fix_mode_refuses_frames_with_no_declared_image_size():
    """Guessing the frame from the bounding box would silently misnormalise
    every input, and this model answers a misnormalised skeleton with confident
    nonsense rather than an error. Only `fix` needs the frame at all."""
    engine = _FakeEngine({42: 9.0})
    frames = _frames(_standing, n=20, image_size=None)
    result = SkeletonActionBackend(engine=engine,
                                   prenorm_mode="fix").classify(frames)
    assert result["backend_error"] is True
    assert "image_size" in result["evidence"]["reason"]
    assert engine.calls == []


def test_the_window_bounds_what_the_model_sees():
    engine = _FakeEngine({42: 9.0})
    frames = _frames(_standing, n=60, fps=10.0)      # 6 s of video
    SkeletonActionBackend(engine=engine, window_s=2.0).classify(frames)
    # 2 s at 10 fps is 21 frames inclusive; they are then resampled to T.
    assert engine.calls[0].shape[2] == DEFAULT_WINDOW_FRAMES


# ── logits vs probabilities ─────────────────────────────────────────────────

def test_a_softmaxed_output_is_detected_and_not_softmaxed_again():
    """Running softmax twice flattens the distribution towards uniform, which
    shows up as every score below threshold and the backend reporting nothing —
    with no error anywhere."""
    probabilities = np.zeros(N_CLASSES, dtype=np.float32)
    probabilities[42] = 0.9
    probabilities[0] = 0.1
    assert looks_like_probabilities(probabilities)
    engine = _FakeEngine(outputs=[probabilities[None]])
    result = SkeletonActionBackend(engine=engine).classify(_frames(_standing))
    assert _action(result) == "fall"
    assert result["activity"]["score"] == pytest.approx(0.9, abs=0.01)


def test_raw_logits_are_softmaxed():
    engine = _FakeEngine({42: 12.0})
    result = SkeletonActionBackend(engine=engine).classify(_frames(_standing))
    assert _action(result) == "fall"
    assert 0.9 < result["activity"]["score"] <= 1.0


def test_logits_are_not_mistaken_for_probabilities():
    logits = np.full(N_CLASSES, 0.01, dtype=np.float32)   # in [0,1] but sums to 0.6
    assert not looks_like_probabilities(logits)


def test_softmax_is_stable_on_large_logits():
    out = softmax(np.array([1000.0, 1001.0], dtype=np.float32))
    assert np.isfinite(out).all()
    assert out.sum() == pytest.approx(1.0)


# ── the model's own vocabulary ──────────────────────────────────────────────

def test_the_model_reports_its_own_class_not_one_of_ours():
    """An earlier version clipped sixty classes down to the five that coincided
    with labels the geometry happened to produce and discarded the rest. That
    inverted the relationship — the rules were written by hand in an afternoon,
    the model was trained on 56,000 clips — and threw away the half with the
    value in it: the health group and the interaction gestures."""
    engine = _FakeEngine({10: 9.0})           # A11 reading
    result = SkeletonActionBackend(engine=engine).classify(_frames(_standing))
    assert _action(result) == "reading"
    assert result["activity"]["ntu_class"] == 11
    assert result["activity"]["name_zh"] == "看书"


def test_classes_the_geometry_also_names_use_the_shared_name():
    """So the two halves do not report the same thing twice under different
    spellings."""
    for index, label in NTU_TO_POSE_LABEL.items():
        engine = _FakeEngine({index: 12.0})
        backend = SkeletonActionBackend(
            engine=engine, fall_min_score=0.1, min_score=0.1)
        assert _action(backend.classify(_frames(_standing))) == label


def test_the_vocabulary_is_the_full_ntu60_table():
    assert len(NTU60) == 60
    assert all(len(entry) == 2 and all(entry) for entry in NTU60)
    assert ntu_name(FALL_CLASS) == "falling down"
    assert ntu_name(22) == "hand waving"


def test_two_person_classes_are_excluded():
    """A50-A60 are defined on a **pair** of skeletons. This backend is fed one
    tracked person with the second slot zero-padded, so the evidence for those
    classes is structurally absent from the input — the model would answer, and
    the answer would be about a person who is not in the tensor."""
    assert MUTUAL_CLASSES == frozenset(range(49, 60))
    assert len(USABLE_CLASSES) == 49
    assert all(i not in USABLE_CLASSES for i in MUTUAL_CLASSES)
    # "hugging" must be unreachable however confident the model is about it.
    engine = _FakeEngine({54: 20.0})
    result = SkeletonActionBackend(engine=engine, min_score=0.01).classify(
        _frames(_standing))
    assert _action(result) == "unknown"


def test_the_catalogue_covers_every_usable_class_with_both_names():
    catalogue = action_vocabulary()
    assert len(catalogue) == 49
    assert all(entry["name"] and entry["name_zh"] for entry in catalogue)
    assert {e["ntu_class"] for e in catalogue} == {i + 1 for i in USABLE_CLASSES}


def test_the_health_group_is_reachable():
    """The half that was being discarded, and the reason this changed."""
    for index, expected in ((41, "staggering"), (43, "touch head"),
                            (47, "nausea / vomiting"), (40, "sneeze / cough")):
        engine = _FakeEngine({index: 9.0})
        result = SkeletonActionBackend(engine=engine).classify(_frames(_standing))
        assert _action(result) == expected, index


# ── failure paths must not become labels ────────────────────────────────────

def test_an_engine_failure_is_reported_as_an_error_not_as_unknown():
    """`unknown` would be indistinguishable from "nobody is doing anything" and
    would hide a broken engine for weeks."""
    engine = _FakeEngine(raise_on_infer=RuntimeError("deserialize failed"))
    result = SkeletonActionBackend(engine=engine).classify(_frames(_standing))
    assert result["backend_error"] is True
    assert "deserialize failed" in result["evidence"]["reason"]


def test_an_output_with_too_few_classes_is_refused():
    engine = _FakeEngine(outputs=[np.zeros((1, 10), dtype=np.float32)])
    result = SkeletonActionBackend(engine=engine).classify(_frames(_standing))
    assert result["backend_error"] is True
    assert "NTU-60" in result["evidence"]["reason"]


def test_a_wrong_keypoint_count_is_refused_at_the_boundary():
    """PoseFrame owns this invariant, because it indexes fixed COCO slots by
    name — a 13-joint array otherwise fails as an IndexError from deep in the
    geometry, which reads as a bug in the rules rather than the wrong input."""
    with pytest.raises(ValueError, match="COCO-17"):
        PoseFrame(0.0, _standing_box(), np.zeros((13, 3), dtype=np.float32),
                  0.3, image_size=FRAME)


def test_the_backend_also_guards_the_keypoint_count_itself():
    """Defence in depth: a caller that assembled the tensor some other way
    must not reach the engine with the wrong joint count."""
    engine = _FakeEngine({42: 9.0})
    frames = _frames(_standing, n=5)
    # Bypass PoseFrame's own check to exercise the backend's.
    frames[0].keypoints = np.zeros((13, 3), dtype=np.float32)
    result = SkeletonActionBackend(engine=engine).classify(frames)
    assert result["backend_error"] is True
    assert "17" in result["evidence"]["reason"]


def test_no_frames_is_unknown_without_touching_the_engine():
    engine = _FakeEngine({42: 9.0})
    assert _action(SkeletonActionBackend(engine=engine).classify([])) == "unknown"
    assert engine.calls == []


def test_a_single_frame_is_declined_rather_than_padded():
    """Padding one frame to T and calling it an action would produce a
    confident answer from a clip in which nothing moves."""
    engine = _FakeEngine({42: 9.0})
    result = SkeletonActionBackend(engine=engine).classify_frame(
        _frames(_standing, n=1)[0])
    assert _action(result) == "unknown"
    assert result["temporal"] is False
    assert result["activity_available"] is False
    assert engine.calls == []


# ── hybrid ──────────────────────────────────────────────────────────────────

def test_hybrid_takes_postures_from_geometry():
    """NTU-60 has no class for a motionless person — standing still is not an
    action — so the geometry is the only thing that can answer it."""
    engine = _FakeEngine(outputs=[np.zeros((1, N_CLASSES), dtype=np.float32)])
    result = HybridActionBackend(engine=engine).classify(_frames(_standing, n=20))
    assert _action(result) == "standing"
    assert result["posture"] == "standing"
    assert result["activity"] is None or result["activity"]["source"] == "rules"


def test_hybrid_lets_the_model_name_the_activity():
    """The model answers "what is this person doing" in its own words."""
    engine = _FakeEngine({10: 9.0})          # A11 reading
    result = HybridActionBackend(engine=engine).classify(_frames(_standing, n=20))
    assert _action(result) == "reading"
    assert result["activity"]["source"] == "stgcn"
    assert result["activity"]["name_zh"] == "看书"


def test_hybrid_reports_posture_and_activity_separately():
    """Two questions, two answers. "Reading" says what they are doing;
    "standing" says what shape their body is in. Neither replaces the other."""
    engine = _FakeEngine({10: 9.0})
    result = HybridActionBackend(engine=engine).classify(_frames(_standing, n=20))
    assert _action(result) == "reading"
    assert result["posture"] == "standing"
    assert result["activity"]["name"] == "reading"


def test_hybrid_keeps_the_geometrys_point_direction():
    """The model names the act; only the geometry measures where."""
    joints = _body()
    shoulder = (CX - 0.10 * H, TOP + 0.18 * H)
    wrist = (CX - 0.40 * H, TOP + 0.19 * H)
    joints.update({"left_shoulder": shoulder, "left_wrist": wrist,
                   "left_elbow": ((shoulder[0] + wrist[0]) / 2,
                                  (shoulder[1] + wrist[1]) / 2)})
    box = (CX - 0.45 * H, TOP, CX + 0.15 * H, TOP + H)
    engine = _FakeEngine({30: 9.0})
    result = HybridActionBackend(engine=engine).classify(
        _frames(lambda _t: joints, n=20, box=box))
    assert _action(result) == "pointing"
    assert result["point_direction"][0] == pytest.approx(-1.0, abs=0.05)


def test_hybrid_still_answers_when_the_model_is_broken():
    """A dead action engine must not take the postures down with it, and must
    leave a trace rather than looking like a quiet frame."""
    engine = _FakeEngine(raise_on_infer=RuntimeError("no engine"))
    result = HybridActionBackend(engine=engine).classify(_frames(_standing, n=20))
    assert _action(result) == "standing"
    assert "no engine" in result["evidence"]["stgcn_error"]


def test_hybrid_history_covers_both_backends():
    engine = _FakeEngine({42: 9.0})
    backend = HybridActionBackend(engine=engine, window_s=4.0)
    assert backend.history_s >= 4.0
    assert backend.history_s >= backend.rules.history_s


def test_an_ntu_transition_is_reported_as_itself_beside_the_posture():
    """`sit down` is NTU's name for the *act* of sitting down. It no longer gets
    bent into our `sitting` state — the posture channel already carries that,
    observed directly, and the two are different facts."""
    engine = _FakeEngine({7: 9.0})
    result = HybridActionBackend(engine=engine).classify(_frames(_standing, n=20))
    assert result["activity"]["name"] == "sit down"
    assert result["posture"] == "standing"


def test_a_single_frame_through_hybrid_falls_to_the_geometry():
    engine = _FakeEngine({42: 9.0})
    result = HybridActionBackend(engine=engine).classify_frame(
        _frames(_standing, n=1)[0])
    assert _action(result) == "standing"


# ── backend selection ───────────────────────────────────────────────────────

def test_build_backend_returns_each_kind():
    assert isinstance(build_backend("rules"), PoseActionClassifier)
    assert isinstance(build_backend("stgcn", engine=_FakeEngine({42: 1.0})),
                      SkeletonActionBackend)
    assert isinstance(build_backend("hybrid", engine=_FakeEngine({42: 1.0})),
                      HybridActionBackend)


def test_an_unknown_backend_name_raises_rather_than_falling_back():
    """A card silently running geometry while its config says `stgcn` is the
    failure mode this whole exercise is about."""
    with pytest.raises(ActionBackendError, match="unknown action_backend"):
        build_backend("stgcnpp")


def test_every_backend_offers_the_same_interface():
    engine = _FakeEngine({42: 1.0})
    for name in ("rules", "stgcn", "hybrid"):
        backend = build_backend(name, engine=engine)
        assert callable(backend.classify)
        assert callable(backend.classify_frame)
        assert isinstance(backend.history_s, float)


# ── the one label the robot acts on ─────────────────────────────────────────

def test_fall_is_held_to_a_higher_score_than_the_rest():
    """Measured: the built engine returns A43 "falling down" at **0.62** on pure
    Gaussian noise — above the general threshold, from a skeleton that is not a
    body. The class is where this network puts input it cannot parse, which is
    the worst possible default for the label a robot acts on."""
    from plugins.pose_stgcn import FALL_MIN_SCORE
    assert FALL_MIN_SCORE > DEFAULT_MIN_SCORE
    backend = SkeletonActionBackend(engine=_FakeEngine({42: 9.0}))
    assert backend._threshold_for({"ntu_class": FALL_CLASS + 1}) == FALL_MIN_SCORE
    assert backend._threshold_for({"ntu_class": 23}) == backend.min_score


def test_a_noise_level_fall_score_does_not_become_a_fall():
    """0.62, the measured noise response, must not clear the bar."""
    probabilities = np.zeros(N_CLASSES, dtype=np.float32)
    probabilities[42] = 0.62
    probabilities[0] = 0.38
    engine = _FakeEngine(outputs=[probabilities[None]])
    result = SkeletonActionBackend(engine=engine).classify(_frames(_standing))
    assert _action(result) != "fall"


def test_a_confident_fall_still_passes():
    probabilities = np.zeros(N_CLASSES, dtype=np.float32)
    probabilities[42] = 0.90
    probabilities[0] = 0.10
    engine = _FakeEngine(outputs=[probabilities[None]])
    assert _action(SkeletonActionBackend(engine=engine).classify(
        _frames(_standing))) == "fall"


def test_hybrid_withholds_a_fall_the_geometry_contradicts():
    """Second guard, and one the noise case cannot satisfy: a real fall ends
    with the person on the ground, which the geometry can see."""
    engine = _FakeEngine({42: 20.0})          # model is certain
    result = HybridActionBackend(engine=engine).classify(_frames(_standing, n=20))
    assert _action(result) != "fall", "a standing body cannot have just fallen"
    assert "fall_withheld" in str(result["evidence"])


def test_hybrid_accepts_a_fall_the_geometry_corroborates():
    engine = _FakeEngine({42: 20.0})
    frames = []
    for i in range(20):
        # Drifting, so the clip has motion and the model is actually consulted.
        joints = {k: (v[0], v[1] - (i / 19) * 0.05 * H)
                  for k, v in _lying_body().items()}
        frames.append(PoseFrame(i / 12, _lying_box(), _kp(**joints), 0.3,
                                image_size=FRAME))
    result = HybridActionBackend(engine=engine).classify(frames)
    assert _action(result) == "fall"
    assert result["activity"]["source"] == "stgcn"


# ── resampling: the step that quietly ate most of a fall's confidence ───────

def test_downsampling_keeps_upstreams_index_selection():
    """NTU clips are longer than 100 frames, so reducing is what the weights
    were fitted on. That path stays exactly as upstream wrote it."""
    clip = np.arange(200, dtype=np.float32)[:, None, None] * np.ones((1, 17, 3))
    out = resample_clip(clip.astype(np.float32), 100)
    assert len(out) == 100
    expected = clip[uniform_sample_indices(200, 100)]
    assert np.allclose(out, expected)


def test_upsampling_interpolates_rather_than_repeating():
    """Our situation is the reverse of NTU's: a 2.5 s window at 12 fps holds 30
    real frames and the network wants 100.

    Repeating each frame 3.3 times makes a staircase — plateaus of zero velocity
    separated by jumps — and an ST-GCN's temporal convolutions see velocity, so
    that is a motion signature the model was never trained on. Measured on a
    real falling skeleton, A43 "falling down":

        100 real frames, smooth              0.948
        30 frames repeated out to 100        0.572   <- under FALL_MIN_SCORE
        30 frames interpolated to 100        0.950

    The repeat version was being withheld as a fall entirely: the model had seen
    the event and said so, and the resampler had taken most of its confidence
    away first.
    """
    clip = np.linspace(0.0, 29.0, 30, dtype=np.float32)[:, None, None]
    clip = clip * np.ones((1, N_KEYPOINTS, 3), dtype=np.float32)
    out = resample_clip(clip, 100)
    assert out.shape == (100, N_KEYPOINTS, 3)
    # A linear ramp in must come back a linear ramp: no plateaus, no jumps.
    steps = np.diff(out[:, 0, 0])
    assert steps.min() > 0, "a repeated frame would give a zero step"
    assert np.allclose(steps, steps[0], atol=1e-4), "and a jump would give a spike"
    assert out[0, 0, 0] == pytest.approx(0.0)
    assert out[-1, 0, 0] == pytest.approx(29.0)


def test_resampling_an_exact_length_clip_is_a_copy():
    clip = np.random.default_rng(0).standard_normal(
        (100, N_KEYPOINTS, 3)).astype(np.float32)
    out = resample_clip(clip, 100)
    assert np.allclose(out, clip)
    assert out is not clip, "must not alias the caller's array"


def test_resampling_a_single_frame_fills_the_window():
    clip = np.ones((1, N_KEYPOINTS, 3), dtype=np.float32)
    out = resample_clip(clip, 100)
    assert out.shape == (100, N_KEYPOINTS, 3)
    assert np.allclose(out, 1.0)


def test_resampling_an_empty_clip_raises():
    with pytest.raises(ActionBackendError):
        resample_clip(np.zeros((0, N_KEYPOINTS, 3), dtype=np.float32), 100)


# ── a frozen clip must not reach the model ──────────────────────────────────

def test_a_motionless_clip_is_not_sent_to_the_model():
    """"No action" is not an answer NTU-60 contains — all 60 classes are things
    somebody is doing — so a frozen clip does not make this network unsure, it
    makes it confidently wrong.

    Measured on 100 identical frames of a real person lying on pavement: NTU's
    "play with phone/tablet" at **0.997**, entropy 0.03. The hybrid mapping
    happened to discard that (the class is not mapped to one of our labels) but
    that is luck, not a guard — the same mechanism landing on A43 would be a
    false fall straight through.
    """
    engine = _FakeEngine({42: 20.0})
    result = SkeletonActionBackend(engine=engine).classify(
        _frames(_standing, n=40, static=True))
    assert _action(result) == "unknown"
    assert "no motion" in result["evidence"]["reason"]
    assert engine.calls == [], "the engine must not have been consulted"


def test_a_moving_clip_still_reaches_the_model():
    def drifting(t):
        joints = dict(_body())
        shift = t * 0.8 * H
        return {k: (v[0], v[1] - shift) for k, v in joints.items()}
    engine = _FakeEngine({42: 20.0})
    frames = [PoseFrame(i / 12.0, _standing_box(), _kp(**drifting(i / 29)), 0.3,
                        image_size=FRAME) for i in range(30)]
    result = SkeletonActionBackend(engine=engine, fall_min_score=0.1).classify(frames)
    assert engine.calls, "a clip with motion must be classified"
    assert _action(result) == "fall"


def test_the_motion_figure_is_reported_either_way():
    """So "the model is wrong" can be told from "it was never asked"."""
    engine = _FakeEngine({42: 20.0})
    backend = SkeletonActionBackend(engine=engine)
    prediction = backend.predict(_frames(_standing, n=40, static=True))
    assert prediction["motion"] < backend.min_motion
    assert prediction["abstained"]


def test_motion_is_measured_on_the_tensor_that_would_be_sent():
    """After normalisation and resampling, so it reflects what the network sees
    rather than raw pixels — which scale with how close the person is."""
    still = np.zeros((1, 2, 10, N_KEYPOINTS, 3), dtype=np.float32)
    assert SkeletonActionBackend.clip_motion(still) == 0.0
    moving = still.copy()
    moving[0, 0, :, 0, 0] = np.linspace(0.0, 0.5, 10)
    assert SkeletonActionBackend.clip_motion(moving) == pytest.approx(0.5)
