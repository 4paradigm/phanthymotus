"""
tests/test_hand_runtime.py — hand ROI geometry, the end2end guard, and the
59-keypoint merge.

Synthetic throughout. The only hardware-bound call in the hand path is the
engine's `infer()`, so every decision about where to crop, when to refuse, and
how the result maps back into frame coordinates is testable here.

Bodies are built from a forearm length rather than in raw pixels, because that
is the scale the module itself works in — a test in absolute pixels would pass
or fail depending on how far away the imaginary person stood.

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest perception/tests -q
"""

from __future__ import annotations

import numpy as np
import pytest

import vision_stubs  # noqa: F401  (installs the cv2 / ROS stubs)

from plugins.hand_runtime import (  # noqa: E402
    DEFAULT_MIN_FOREARM_PX,
    HAND_INDEX,
    HAND_KEYPOINTS,
    HAND_PER_FOREARM,
    HAND_SKELETON,
    N_HAND_KEYPOINTS,
    RTMPOSE_MEAN,
    RTMPOSE_STD,
    TARGET_HAND_FRACTION,
    HandDecodeError,
    assert_simcc,
    box_to_frame,
    crop_roi,
    decode_simcc,
    forearm_length,
    keypoints_to_frame,
    merge_keypoints,
    merged_keypoint_names,
    merged_skeleton,
    roi_from_body,
    roi_from_previous,
    to_rtmpose_blob,
)
from plugins.vision_runtime import (  # noqa: E402
    COCO_INDEX, COCO_KEYPOINTS, COCO_SKELETON, N_KEYPOINTS, LetterboxMeta,
)


def _body(forearm=120.0, conf=0.9, wrist_conf=None, elbow_conf=None):
    """COCO-17 keypoints with a horizontal right forearm of a given length.

    Elbow at (400, 300), wrist `forearm` pixels to its right, so the hand is
    expected further right again.
    """
    keypoints = np.zeros((N_KEYPOINTS, 3), dtype=np.float32)
    keypoints[:, 2] = conf
    keypoints[COCO_INDEX["right_elbow"]] = (400.0, 300.0,
                                            conf if elbow_conf is None else elbow_conf)
    keypoints[COCO_INDEX["right_wrist"]] = (400.0 + forearm, 300.0,
                                            conf if wrist_conf is None else wrist_conf)
    keypoints[COCO_INDEX["left_elbow"]] = (200.0, 300.0,
                                           conf if elbow_conf is None else elbow_conf)
    # Both wrists honour `wrist_conf`: a test that occluded only one would see
    # the other hand still produced and read as "the cache was not dropped".
    keypoints[COCO_INDEX["left_wrist"]] = (200.0 - forearm, 300.0,
                                           conf if wrist_conf is None else wrist_conf)
    return keypoints


class _FakeSession:
    """Stands in for the engine in the guard tests.

    Returns tensors under given names, because the two SimCC outputs have the
    same shape and can only be told apart by name.
    """

    def __init__(self, shapes, names=("simcc_x", "simcc_y"), net=256):
        self._shapes = shapes
        self.output_names = list(names)
        self._net = net
        self.calls = 0
        self.input_dtype = np.float32

    @property
    def input_size(self):
        return (self._net, self._net)

    def infer(self, blob):
        self.calls += 1
        return [np.zeros(sh, dtype=np.float32) for sh in self._shapes]


# ── the keypoint set ─────────────────────────────────────────────────────────

def test_there_are_twenty_one_joints_and_the_names_are_unique():
    assert N_HAND_KEYPOINTS == 21
    assert len(set(HAND_KEYPOINTS)) == 21
    assert HAND_KEYPOINTS[0] == "wrist"


def test_every_joint_is_reachable_through_the_bone_table():
    """A joint in no bone draws as a loose dot and reads as a missing finger."""
    reachable = set()
    for a, b in HAND_SKELETON:
        reachable.add(a)
        reachable.add(b)
    assert reachable == set(range(N_HAND_KEYPOINTS))


def test_the_bone_table_only_references_real_joints():
    for a, b in HAND_SKELETON:
        assert 0 <= a < N_HAND_KEYPOINTS and 0 <= b < N_HAND_KEYPOINTS


# ── the SimCC head ──────────────────────────────────────────────────────────

def test_a_simcc_engine_passes_the_guard():
    session = _FakeSession([(1, 21, 512), (1, 21, 512)])
    assert assert_simcc(session) == ((1, 21, 512), (1, 21, 512))
    assert session.calls == 1


def test_an_engine_that_does_not_name_its_outputs_is_refused():
    """The two tensors are the same shape, so there is nothing in the numbers
    to tell x from y. Swapping them transposes every hand and still draws a
    perfectly plausible one — and this repository has already been bitten by
    TensorRT listing outputs in a different order on the two JetPack lines."""
    session = _FakeSession([(1, 21, 512), (1, 21, 512)],
                           names=("output0", "output1"))
    with pytest.raises(HandDecodeError) as excinfo:
        assert_simcc(session)
    assert "simcc_x" in str(excinfo.value)


def test_a_wrong_keypoint_count_is_refused():
    session = _FakeSession([(1, 17, 512), (1, 17, 512)])
    with pytest.raises(HandDecodeError):
        assert_simcc(session)


def test_a_bin_count_that_is_not_a_whole_split_ratio_is_refused():
    """A different RTMPose size changes the ratio, and a wrong one scales
    every hand by a constant while still looking like a hand."""
    session = _FakeSession([(1, 21, 500), (1, 21, 500)])
    with pytest.raises(HandDecodeError):
        assert_simcc(session)


def test_the_guard_runs_at_load_not_per_frame():
    session = _FakeSession([(1, 21, 512), (1, 21, 512)])
    assert_simcc(session)
    assert session.calls == 1


def test_simcc_decodes_an_argmax_to_input_pixels():
    net, bins = 256, 512
    sx = np.zeros((1, N_HAND_KEYPOINTS, bins), np.float32)
    sy = np.zeros((1, N_HAND_KEYPOINTS, bins), np.float32)
    sx[0, :, 100] = 0.9          # 100 bins / (512/256) = 50 px
    sy[0, :, 300] = 0.7          # 300 / 2 = 150 px
    out = decode_simcc(sx, sy, net)
    assert out.shape == (1, N_HAND_KEYPOINTS, 3)
    assert out[0, 0, 0] == pytest.approx(50.0)
    assert out[0, 0, 1] == pytest.approx(150.0)


def test_the_split_ratio_is_read_from_the_tensor_not_hardcoded():
    """A different input size changes it, and a constant would scale every
    hand by a fixed factor while still producing a hand-shaped thing."""
    sx = np.zeros((1, N_HAND_KEYPOINTS, 768), np.float32)
    sy = np.zeros((1, N_HAND_KEYPOINTS, 768), np.float32)
    sx[0, :, 300] = 1.0; sy[0, :, 300] = 1.0
    out = decode_simcc(sx, sy, 384)              # ratio 2 again
    assert out[0, 0, 0] == pytest.approx(150.0)


def test_joint_confidence_is_the_worse_of_the_two_axes():
    """A joint is only as well located as its worse axis; taking the larger
    would report a joint certain in x and guessed in y as certain."""
    sx = np.zeros((1, N_HAND_KEYPOINTS, 512), np.float32)
    sy = np.zeros((1, N_HAND_KEYPOINTS, 512), np.float32)
    sx[0, 0, 10] = 0.9
    sy[0, 0, 10] = 0.2
    assert decode_simcc(sx, sy, 256)[0, 0, 2] == pytest.approx(0.2)


def test_the_blob_is_imagenet_normalised_rgb_not_zero_to_one():
    """RTMPose subtracts a mean and divides by a standard deviation. Feeding
    it /255 yields a confident hand in the wrong place."""
    crop = np.zeros((256, 256, 3), np.uint8)
    crop[:, :, 0] = 10      # B
    crop[:, :, 1] = 20      # G
    crop[:, :, 2] = 30      # R
    blob = to_rtmpose_blob(crop, np.float32)
    assert blob.shape == (1, 3, 256, 256)
    # channel 0 of the blob is R, because the model wants RGB
    assert blob[0, 0, 0, 0] == pytest.approx((30 - RTMPOSE_MEAN[0]) / RTMPOSE_STD[0])
    assert blob[0, 2, 0, 0] == pytest.approx((10 - RTMPOSE_MEAN[2]) / RTMPOSE_STD[2])
    assert blob.min() < 0, "normalised input must not be in [0, 1]"


# ── forearm scale ────────────────────────────────────────────────────────────

def test_forearm_length_measures_elbow_to_wrist():
    assert forearm_length(_body(forearm=150.0), "right", 0.3) == pytest.approx(150.0)


def test_forearm_length_is_none_when_the_wrist_is_occluded():
    assert forearm_length(_body(wrist_conf=0.1), "right", 0.3) is None


def test_forearm_length_is_none_when_the_elbow_is_occluded():
    assert forearm_length(_body(elbow_conf=0.1), "right", 0.3) is None


# ── ROI derivation ───────────────────────────────────────────────────────────

def test_the_roi_is_sized_to_put_the_hand_in_the_measured_band():
    """The crop exists to land the hand at ~31% of the network input. A crop
    sized any other way is the failure the measurements were taken to avoid."""
    roi, reason = roi_from_body(_body(forearm=200.0), "right", min_conf=0.3)
    assert reason is None
    expected_hand = 200.0 * HAND_PER_FOREARM
    assert roi.hand_px == pytest.approx(expected_hand)
    assert roi.side == pytest.approx(expected_hand / TARGET_HAND_FRACTION)
    # ...and that is what makes the hand occupy the target fraction of the crop
    assert roi.hand_px / roi.side == pytest.approx(TARGET_HAND_FRACTION)


def test_the_roi_sits_past_the_wrist_along_the_forearm():
    """A hand hangs off the end of the arm; centring on the wrist puts a third
    of the crop on the forearm and clips the fingers."""
    roi, _ = roi_from_body(_body(forearm=120.0), "right", min_conf=0.3)
    wrist_x = 400.0 + 120.0
    assert roi.cx > wrist_x
    assert roi.cy == pytest.approx(300.0)


def test_the_roi_follows_the_other_arms_direction():
    roi, _ = roi_from_body(_body(forearm=120.0), "left", min_conf=0.3)
    assert roi.cx < 200.0 - 120.0        # left arm points the other way
    assert roi.side_name == "left"


def test_an_occluded_wrist_is_refused_rather_than_guessed():
    roi, reason = roi_from_body(_body(wrist_conf=0.1), "right", min_conf=0.3)
    assert roi is None and reason == "wrist_occluded"


def test_an_occluded_elbow_is_refused_because_there_is_no_scale():
    """The wrist alone gives a position but no size, and a guessed size is how
    a crop ends up with the hand at 500 px — 3 of 5 detections lost."""
    roi, reason = roi_from_body(_body(elbow_conf=0.1), "right", min_conf=0.3)
    assert roi is None and reason == "elbow_occluded"


def test_a_short_forearm_is_too_far():
    roi, reason = roi_from_body(_body(forearm=40.0), "right", min_conf=0.3)
    assert roi is None and reason == "too_far"


def test_the_distance_gate_is_configurable():
    """It has to be: pixels-to-metres depends on the camera's field of view,
    which this process does not know."""
    body = _body(forearm=100.0)
    assert roi_from_body(body, "right", min_conf=0.3,
                         min_forearm_px=200.0)[1] == "too_far"
    assert roi_from_body(body, "right", min_conf=0.3,
                         min_forearm_px=50.0)[1] is None


def test_the_default_gate_is_about_forty_five_pixels_of_hand():
    """Where the measured curve starts losing detections outright.

    Asserted as a *product* on purpose: the gate and the hand/forearm ratio
    have to move together. They were once consistent with each other and both
    wrong — a 0.45 ratio with a 90 px gate — and a test on either number alone
    would have passed throughout."""
    assert DEFAULT_MIN_FOREARM_PX * HAND_PER_FOREARM == pytest.approx(45.0)


def test_the_ratio_is_hand_length_not_hand_width():
    """0.45 is hand width; what the crop must contain is the box the model
    draws, which covers the fingers. Shipping the width halved every crop and
    put the hand at 70% of it, the oversized end where detection collapses."""
    assert HAND_PER_FOREARM > 0.7


def test_the_previous_box_gives_a_tighter_roi():
    roi = roi_from_previous([500.0, 280.0, 560.0, 340.0], "right")
    assert roi.source == "previous"
    assert roi.cx == pytest.approx(530.0) and roi.cy == pytest.approx(310.0)
    assert roi.hand_px == pytest.approx(60.0)
    assert roi.side == pytest.approx(60.0 / TARGET_HAND_FRACTION)


def test_no_previous_box_means_no_roi_so_the_caller_falls_back_to_the_body():
    assert roi_from_previous(None, "right") is None
    assert roi_from_previous([10.0, 10.0, 10.0, 10.0], "right") is None


# ── cropping and the coordinate round trip ───────────────────────────────────

def test_a_crop_comes_back_at_the_engines_input_size():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    roi, _ = roi_from_body(_body(forearm=200.0), "right", min_conf=0.3)
    crop = crop_roi(frame, roi, 448)
    assert crop.shape[:2] == (448, 448)


def test_a_crop_at_the_frame_edge_still_comes_back_square():
    """A raised hand is frequently at the edge of frame; the border is extended
    by reflection rather than filled with grey, which is a texture the weights
    never saw."""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    roi = roi_from_previous([-40.0, -40.0, 60.0, 60.0], "right")
    crop = crop_roi(frame, roi, 448)
    assert crop.shape[:2] == (448, 448)


def test_keypoints_map_back_onto_the_native_frame():
    roi = roi_from_previous([500.0, 280.0, 560.0, 340.0], "right")
    centre = np.array([[224.0, 224.0, 0.9]], dtype=np.float32)
    mapped = keypoints_to_frame(centre, roi, 448)
    assert mapped[0][0] == pytest.approx(roi.cx, abs=0.5)
    assert mapped[0][1] == pytest.approx(roi.cy, abs=0.5)
    assert mapped[0][2] == pytest.approx(0.9)


def test_the_round_trip_preserves_the_corners():
    roi = roi_from_previous([500.0, 280.0, 560.0, 340.0], "right")
    corners = np.array([[0.0, 0.0, 1.0], [448.0, 448.0, 1.0]], dtype=np.float32)
    mapped = keypoints_to_frame(corners, roi, 448)
    x1, y1, x2, y2 = roi.box
    assert mapped[0][0] == pytest.approx(x1) and mapped[0][1] == pytest.approx(y1)
    assert mapped[1][0] == pytest.approx(x2) and mapped[1][1] == pytest.approx(y2)


def test_a_fingertip_outside_the_frame_stays_outside():
    """Pinning it to the border invents a position that then reads as a real
    joint to the gesture rules — the same rule undo_letterbox_points follows."""
    roi = roi_from_previous([10.0, 10.0, 70.0, 70.0], "right")
    off = np.array([[0.0, 0.0, 0.9]], dtype=np.float32)
    mapped = keypoints_to_frame(off, roi, 448)
    assert mapped[0][0] < 0.0 and mapped[0][1] < 0.0


def test_a_box_maps_back_the_same_way_as_keypoints():
    roi = roi_from_previous([500.0, 280.0, 560.0, 340.0], "right")
    mapped = box_to_frame([0.0, 0.0, 448.0, 448.0], roi, 448)
    assert mapped == pytest.approx(list(roi.box))


# ── the merged skeleton payload ──────────────────────────────────────────────

def test_the_merge_appends_and_never_moves_an_existing_index():
    """The dashboard renderer keeps its own copy of the COCO bone table and any
    already-wired consumer indexes the first 17. Interleaving by anatomy would
    look tidier and silently move every existing index."""
    body = np.arange(N_KEYPOINTS * 3, dtype=np.float32).reshape(N_KEYPOINTS, 3)
    left = np.full((N_HAND_KEYPOINTS, 3), 7.0, dtype=np.float32)
    right = np.full((N_HAND_KEYPOINTS, 3), 9.0, dtype=np.float32)
    merged = merge_keypoints(body, left, right)
    assert merged.shape == (N_KEYPOINTS + 2 * N_HAND_KEYPOINTS, 3)
    assert np.array_equal(merged[:N_KEYPOINTS], body)
    assert np.all(merged[N_KEYPOINTS:N_KEYPOINTS + N_HAND_KEYPOINTS] == 7.0)
    assert np.all(merged[N_KEYPOINTS + N_HAND_KEYPOINTS:] == 9.0)


def test_the_merged_length_does_not_depend_on_what_was_in_frame():
    """A variable-length array would be a second shape for every consumer to
    handle; a zero-visibility hand is already indistinguishable from an
    invisible one downstream, since both the renderer and the rules gate on
    visibility."""
    body = np.zeros((N_KEYPOINTS, 3), dtype=np.float32)
    full = merge_keypoints(body, np.ones((N_HAND_KEYPOINTS, 3), np.float32), None)
    assert full.shape == (59, 3)
    assert np.all(full[N_KEYPOINTS + N_HAND_KEYPOINTS:, 2] == 0.0)
    assert merge_keypoints(body, None, None).shape == (59, 3)


def test_the_merged_names_line_up_with_the_merged_array():
    names = merged_keypoint_names(COCO_KEYPOINTS)
    assert len(names) == 59
    assert names[:N_KEYPOINTS] == list(COCO_KEYPOINTS)
    assert names[N_KEYPOINTS] == "left_hand_wrist"
    assert names[N_KEYPOINTS + N_HAND_KEYPOINTS - 1] == "left_hand_pinky_tip"
    assert names[-1] == "right_hand_pinky_tip"


def test_no_merged_name_is_duplicated():
    """COCO-17 already has `left_wrist` and so does the hand set. A plain
    `left_` prefix gives one payload two joints with the same name, and
    `keypoint_names` exists precisely so consumers can look joints up by
    name."""
    names = merged_keypoint_names(COCO_KEYPOINTS)
    assert len(set(names)) == 59
    assert names.count("left_wrist") == 1


def test_the_merged_bone_table_offsets_the_hands_and_joins_them_to_the_arms():
    edges = merged_skeleton(COCO_SKELETON, N_KEYPOINTS)
    assert [tuple(e) for e in edges[:len(COCO_SKELETON)]] == list(COCO_SKELETON)
    flat = {i for edge in edges for i in edge}
    assert flat == set(range(59))
    # each hand's wrist is tied to the arm's wrist, or the hand floats detached
    assert [COCO_INDEX["left_wrist"], N_KEYPOINTS + HAND_INDEX["wrist"]] in edges
    assert [COCO_INDEX["right_wrist"],
            N_KEYPOINTS + N_HAND_KEYPOINTS + HAND_INDEX["wrist"]] in edges


def test_the_merged_bone_table_has_no_out_of_range_index():
    for a, b in merged_skeleton(COCO_SKELETON, N_KEYPOINTS):
        assert 0 <= a < 59 and 0 <= b < 59


# ── the published bundle ─────────────────────────────────────────────────────
#
# These assert the shipped table, not the download. `_ensure_vision_bundle`
# refuses an unpinned entry rather than fetching it, so an unpinned table is a
# card that reports `state: error` on every robot — worth catching here rather
# than on a rig.

def test_both_jetpack_lines_are_published_and_pinned():
    from utils.model_downloader import HAND_MODEL_BUNDLES

    assert set(HAND_MODEL_BUNDLES) == {"jp61", "jp511"}
    for family, entry in HAND_MODEL_BUNDLES.items():
        assert entry["files"], family
        for name, meta in entry["files"].items():
            assert name.endswith(".engine"), name
            assert meta["size"] > 0, (family, name)
            assert len(meta["sha256"]) == 64, (family, name)


def test_the_two_lines_are_different_plans():
    """A plan only loads on the TensorRT that built it, so the same bytes under
    two families would mean one of them was never really built."""
    from utils.model_downloader import HAND_MODEL_BUNDLES

    digests = {family: next(iter(entry["files"].values()))["sha256"]
               for family, entry in HAND_MODEL_BUNDLES.items()}
    assert len(set(digests.values())) == len(digests)


def test_the_bundle_url_names_the_input_size_it_was_built_for():
    """The crop is sized to land the hand at a fraction of the *input*, so an
    engine built for another size moves the hand out of the band the
    measurements were taken in."""
    from utils.model_downloader import HAND_MODEL_BUNDLES

    for entry in HAND_MODEL_BUNDLES.values():
        assert entry["base_url"].endswith("-256")


def test_the_source_onnx_is_mirrored_and_pinned():
    """Build-time only, but pinned for the same reason everything else is: it
    is what makes an engine rebuild reproducible without re-exporting from the
    .pt, which would change the numbers if the ultralytics version differs."""
    from utils.model_downloader import HAND_ONNX

    assert len(HAND_ONNX) == 1
    meta = next(iter(HAND_ONNX.values()))
    assert meta["size"] > 0 and len(meta["sha256"]) == 64


# ── HandChannel: the budget and the bookkeeping ──────────────────────────────

from plugins.hand_runtime import (  # noqa: E402
    HandChannel, N_HAND_KEYPOINTS as NHK, hands_in_frame)


def _simcc(span_frac=0.6, conf=0.9, net=256, n=1):
    """SimCC tensors for a hand spanning `span_frac` of the input, centred."""
    bins = net * 2
    sx = np.zeros((n, NHK, bins), np.float32)
    sy = np.zeros((n, NHK, bins), np.float32)
    lo = (0.5 - span_frac / 2) * net
    for j in range(NHK):
        x = lo + span_frac * net * j / max(NHK - 1, 1)
        sx[:, j, int(x * 2)] = conf
        sy[:, j, int(net * 0.5 * 2)] = conf
    return sx, sy


class _HandSession:
    """Fake RTMPose engine. Counts inferences, so the budget is observable."""

    def __init__(self, conf=0.9, net=256, raises=False, span_frac=0.6):
        self.calls = 0
        self.net = net
        self.raises = raises
        self.conf = conf
        self.span_frac = span_frac
        self.output_names = ["simcc_x", "simcc_y"]
        self.input_dtype = np.float32

    @property
    def input_size(self):
        return (self.net, self.net)

    def infer(self, blob):
        self.calls += 1
        if self.raises:
            raise RuntimeError("engine exploded")
        sx, sy = _simcc(self.span_frac, self.conf, self.net)
        return [sx, sy]


def _frame():
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


def _person(track_id=1, forearm=200.0):
    return {"id": track_id, "keypoints": _body(forearm=forearm)}


def test_both_hands_are_attempted_within_the_budget():
    session = _HandSession()
    channel = HandChannel(session, max_rois=2, interval_s=0.0)
    person = _person()
    channel.update([person], _frame(), 0.0)
    assert session.calls == 2
    assert person["hands"]["left"] is not None
    assert person["hands"]["right"] is not None
    assert person["hands"]["left"].shape == (NHK, 3)


def test_the_budget_caps_inferences_per_frame():
    """11.26 ms per ROI against an 83 ms frame that the body channel already
    spends 41 ms of — the cap is the point, not a safety margin."""
    session = _HandSession()
    channel = HandChannel(session, max_rois=1, interval_s=0.0)
    people = [_person(1), _person(2)]
    channel.update(people, _frame(), 0.0)
    assert session.calls == 1
    produced = sum(1 for p in people for v in p["hands"].values() if v is not None)
    assert produced == 1


def test_the_budget_is_spent_on_the_largest_hand():
    """Largest means nearest, and a nearer hand is both likelier to be
    addressing the robot and the only one fingers can be resolved on."""
    session = _HandSession()
    channel = HandChannel(session, max_rois=1, interval_s=0.0)
    near, far = _person(1, forearm=400.0), _person(2, forearm=120.0)
    channel.update([near, far], _frame(), 0.0)
    assert any(v is not None for v in near["hands"].values())
    assert all(v is None for v in far["hands"].values())


def test_a_dropped_candidate_is_counted_not_silently_lost():
    session = _HandSession()
    channel = HandChannel(session, max_rois=1, interval_s=0.0)
    channel.update([_person(1), _person(2)], _frame(), 0.0)
    assert channel.skipped["throttled"] >= 1


def test_the_throttle_reuses_the_previous_keypoints():
    """Otherwise the payload would flicker between a hand and no hand at the
    difference between the card's fps and the hand channel's rate."""
    session = _HandSession()
    channel = HandChannel(session, max_rois=2, interval_s=10.0)
    person = _person()
    channel.update([person], _frame(), 0.0)
    first = person["hands"]["right"]
    assert session.calls == 2

    again = _person()
    channel.update([again], _frame(), 0.1)      # well inside the interval
    assert session.calls == 2                   # no new inference
    assert np.array_equal(again["hands"]["right"], first)
    assert channel.skipped["throttled"] >= 2


def test_the_throttle_lapses():
    session = _HandSession()
    channel = HandChannel(session, max_rois=2, interval_s=0.25)
    channel.update([_person()], _frame(), 0.0)
    calls = session.calls
    channel.update([_person()], _frame(), 5.0)
    assert session.calls > calls


def test_an_out_of_reach_hand_loses_its_cached_shape():
    """A cached hand would otherwise keep being reported after the person
    turned away — a hand shape from a moment that has passed."""
    session = _HandSession()
    channel = HandChannel(session, max_rois=2, interval_s=0.0)
    channel.update([_person(1, forearm=200.0)], _frame(), 0.0)

    gone = {"id": 1, "keypoints": _body(forearm=200.0, wrist_conf=0.05)}
    channel.update([gone], _frame(), 1.0)
    assert gone["hands"]["left"] is None and gone["hands"]["right"] is None
    assert channel.skipped["wrist_occluded"] >= 2


def test_a_too_far_person_is_counted_as_such():
    session = _HandSession()
    channel = HandChannel(session, max_rois=2, interval_s=0.0,
                          min_forearm_px=300.0)
    person = _person(forearm=100.0)
    channel.update([person], _frame(), 0.0)
    assert session.calls == 0
    assert channel.skipped["too_far"] == 2


def test_an_engine_error_does_not_kill_the_frame():
    """One hand failing must not cost the body result that was already
    computed for this frame."""
    session = _HandSession(raises=True)
    channel = HandChannel(session, max_rois=2, interval_s=0.0)
    person = _person()
    channel.update([person], _frame(), 0.0)
    assert person["hands"] == {"left": None, "right": None}
    assert "engine exploded" in (channel.last_error or "")


def test_an_empty_detection_drops_the_cache_rather_than_chasing_it():
    """So the next frame re-derives the ROI from the arm instead of following
    a box the model could not read. One call per hand, not two: there is no
    second scale to try."""
    session = _HandSession(conf=0.05)
    channel = HandChannel(session, max_rois=2, interval_s=0.0)
    person = _person()
    channel.update([person], _frame(), 0.0)
    assert session.calls == 2
    assert person["hands"]["right"] is None
    assert channel.ran == 2


def test_an_out_of_reach_hand_loses_its_cached_shape():
    """A cached hand would otherwise keep being reported after the person
    turned away — a hand shape from a moment that has passed."""
    session = _HandSession()
    channel = HandChannel(session, max_rois=2, interval_s=0.0)
    channel.update([_person(1, forearm=200.0)], _frame(), 0.0)

    gone = {"id": 1, "keypoints": _body(forearm=200.0, wrist_conf=0.05)}
    channel.update([gone], _frame(), 1.0)
    assert gone["hands"]["left"] is None and gone["hands"]["right"] is None
    assert channel.skipped["wrist_occluded"] >= 2


def test_a_too_far_person_is_counted_as_such():
    session = _HandSession()
    channel = HandChannel(session, max_rois=2, interval_s=0.0,
                          min_forearm_px=300.0)
    person = _person(forearm=100.0)
    channel.update([person], _frame(), 0.0)
    assert session.calls == 0
    assert channel.skipped["too_far"] == 2


def test_an_engine_error_does_not_kill_the_frame():
    """One hand failing must not cost the body result that was already
    computed for this frame."""
    session = _HandSession(raises=True)
    channel = HandChannel(session, max_rois=2, interval_s=0.0)
    person = _person()
    channel.update([person], _frame(), 0.0)
    assert person["hands"] == {"left": None, "right": None}
    assert "engine exploded" in (channel.last_error or "")


def test_a_hand_costs_exactly_one_inference():
    """No detector pass and no retry: this model reads the same hand across a
    2.2x range of box sizes, so there is nothing to retry at a second scale."""
    session = _HandSession()
    channel = HandChannel(session, max_rois=1, interval_s=0.0)
    channel.update([_person()], _frame(), 0.0)
    assert session.calls == 1


def test_keypoints_land_in_native_frame_coordinates():
    """The crop is cut out of the native frame, so what the gesture rules and
    the renderer see has to be in that frame's pixels, not the crop's."""
    session = _HandSession()
    channel = HandChannel(session, max_rois=2, interval_s=0.0)
    person = _person(forearm=200.0)
    channel.update([person], _frame(), 0.0)
    right = person["hands"]["right"]
    wrist_x = _body(forearm=200.0)[COCO_INDEX["right_wrist"]][0]
    # the hand sits past the wrist, so well to the right of 400 + 200
    assert right[:, 0].mean() > wrist_x


def test_reset_drops_all_cached_hands():
    session = _HandSession()
    channel = HandChannel(session, max_rois=2, interval_s=10.0)
    channel.update([_person()], _frame(), 0.0)
    channel.reset()
    person = _person()
    channel.update([person], _frame(), 0.05)
    assert session.calls == 4            # ran again rather than reusing


def test_every_skip_reason_is_a_declared_one():
    """`info` groups by these keys; an undeclared reason would be invisible."""
    from plugins.hand_runtime import SKIP_REASONS

    channel = HandChannel(_HandSession(), max_rois=2, interval_s=0.0)
    assert set(channel.skipped) == set(SKIP_REASONS)


def test_the_body_gate_applies_on_every_frame_not_just_the_first():
    """Once a hand has a box, cropping from that box alone would keep following
    it after the person walked out of range — so the distance gate would hold
    for one frame and then stop existing."""
    session = _HandSession()
    channel = HandChannel(session, max_rois=2, interval_s=0.0,
                          min_forearm_px=150.0)
    channel.update([_person(1, forearm=200.0)], _frame(), 0.0)
    assert session.calls == 2

    receding = _person(1, forearm=100.0)        # now inside the gate's cutoff
    channel.update([receding], _frame(), 1.0)
    assert session.calls == 2                   # no further inference
    assert receding["hands"] == {"left": None, "right": None}
    assert channel.skipped["too_far"] == 2


def test_the_previous_box_still_tightens_the_crop_while_the_gate_holds():
    """The default fake row happens to span exactly the target fraction, which
    makes the refined ROI identical to the arm-derived one and the assertion
    vacuous — so this uses a deliberately smaller hand."""
    session = _HandSession(span_frac=0.25)
    channel = HandChannel(session, max_rois=2, interval_s=0.0)
    channel.update([_person(1, forearm=200.0)], _frame(), 0.0)
    cached = channel._state[(1, "right")]["box"]
    refined = roi_from_previous(cached, "right")
    from_body, _ = roi_from_body(_body(forearm=200.0), "right", min_conf=0.3)
    assert refined.source == "previous"
    assert refined.side < from_body.side


def test_whole_frame_hands_come_back_inside_the_frame():
    """decode_poses already undoes the letterbox. Undoing it again inflated
    every hand by 1/scale and pushed it off the left edge — a 500x917 photo
    came back with a 770x634 hand starting at x = -159."""
    from plugins.hand_runtime import hands_in_frame

    frame = np.zeros((917, 500, 3), dtype=np.uint8)
    session = _HandSession(net=256)
    found = hands_in_frame(session, frame, confidence=0.4)
    assert found, "the fake engine always returns a hand"
    points, score = found[0]
    assert points.shape == (NHK, 3)
    xs, ys = points[:, 0], points[:, 1]
    # Asserted on the shape of the bug rather than on absolute bounds: undoing
    # the letterbox twice scales everything by 1/letterbox_scale, so the hand
    # came back *wider than the frame* with its centroid off the left edge.
    # A fingertip may legitimately sit just outside the frame; a whole hand
    # bigger than the picture may not.
    assert 0 <= float(xs.mean()) <= 500
    assert 0 <= float(ys.mean()) <= 917
    # The span is deliberately not asserted: this stub's "hand" is a
    # degenerate horizontal line, and the refine pass legitimately enlarges
    # it. The centroid is what separates the bug from the stub — the real
    # failure put the whole hand off the left edge, starting at x = -159.


# ── holding a hand through a miss ────────────────────────────────────────────
#
# Reported as "the fingers keep disappearing". The throttle is not the cause
# and was measured not to be — with a steady image, 129 of 129 published
# frames carried both hands, because the frames between inferences reuse the
# last result. What a miss did was erase the hand outright, and at the
# throttled rate that is a quarter second of nothing, which on a live camera
# reads as flicker.

def test_one_miss_does_not_erase_the_hand():
    found = _HandSession()
    channel = HandChannel(found, max_rois=2, interval_s=0.0, hold_s=0.5)
    first = _person()
    channel.update([first], _frame(), 0.0)
    kept = first["hands"]["right"]
    assert kept is not None

    channel._session = _HandSession(conf=0.05)          # the engine finds nothing
    during = _person()
    channel.update([during], _frame(), 0.1)
    assert during["hands"]["right"] is not None
    assert np.array_equal(during["hands"]["right"], kept)


def test_a_hand_that_stays_gone_is_forgotten():
    """The hold is a grace period, not a memory — a hand really put away must
    stop being reported."""
    channel = HandChannel(_HandSession(), max_rois=2, interval_s=0.0,
                          hold_s=0.5)
    channel.update([_person()], _frame(), 0.0)
    channel._session = _HandSession(conf=0.05)
    gone = _person()
    channel.update([gone], _frame(), 2.0)             # well past the hold
    assert gone["hands"]["right"] is None


def test_the_stale_box_is_dropped_even_while_the_keypoints_are_held():
    """The two were conflated. Cropping from the previous box after a miss
    would chase a hand that is no longer there; reporting the previous
    keypoints for a moment is merely saying "it was here an instant ago"."""
    channel = HandChannel(_HandSession(), max_rois=2, interval_s=0.0,
                          hold_s=5.0)
    channel.update([_person()], _frame(), 0.0)
    assert "box" in channel._state[(1, "right")]

    channel._session = _HandSession(conf=0.05)
    channel.update([_person()], _frame(), 0.1)
    assert "box" not in channel._state[(1, "right")]
    assert channel._state[(1, "right")].get("kpts") is not None


def test_a_blinking_wrist_does_not_erase_the_hand():
    """Visibility oscillating around the threshold is the other way a hand
    vanishes — the geometric gate refuses and the cache went with it."""
    channel = HandChannel(_HandSession(), max_rois=2, interval_s=0.0,
                          hold_s=0.5)
    channel.update([_person()], _frame(), 0.0)
    blink = {"id": 1, "keypoints": _body(forearm=200.0, wrist_conf=0.05)}
    channel.update([blink], _frame(), 0.1)
    assert blink["hands"]["right"] is not None


def test_the_hold_can_be_switched_off():
    channel = HandChannel(_HandSession(), max_rois=2, interval_s=0.0,
                          hold_s=0.0)
    channel.update([_person()], _frame(), 0.0)
    channel._session = _HandSession(conf=0.05)
    gone = _person()
    channel.update([gone], _frame(), 0.01)
    assert gone["hands"]["right"] is None


def test_the_confidence_gate_sits_between_the_measured_distributions():
    """A top-down model always returns 21 joints, so this gate rejects a box
    that landed on no hand — not a failure to detect. Measured: no hand in the
    box scored 0.11-0.37 (grey, shirt, background, face, jeans, noise), a real
    hand 0.48-0.74 including blurred frames. The default must sit in the gap.

    It was 0.25, inherited from the model this replaced, where the same field
    held a detection score — at which every one of those negatives except the
    flat grey would have been reported as a hand."""
    from plugins.hand_runtime import DEFAULT_HAND_CONFIDENCE

    assert 0.37 < DEFAULT_HAND_CONFIDENCE < 0.48
    channel = HandChannel(_HandSession(), max_rois=1)
    assert channel._confidence == DEFAULT_HAND_CONFIDENCE


def test_a_hand_that_cannot_be_found_is_throttled_too():
    """The throttle only ever applied to hands that were already working: a
    miss cleared the cache, `_due` then saw no entry and said yes, and an
    unfindable hand cost an inference on every frame — at two scales. On a
    stream where one of two hands was marginal that measured as +49 ms per
    frame."""
    session = _HandSession(conf=0.05)
    channel = HandChannel(session, max_rois=2, interval_s=1.0, hold_s=0.2)
    channel.update([_person()], _frame(), 0.0)
    first = session.calls
    assert first == 2                       # both hands attempted once

    for step in range(1, 10):               # ~0.75 s of frames, inside the interval
        channel.update([_person()], _frame(), step * 1 / 12.0)
    assert session.calls == first, "an unfindable hand was retried every frame"


def test_the_throttle_lapses_for_a_missing_hand_too():
    """It is a throttle, not a blacklist — a hand that comes back must be
    found again."""
    session = _HandSession(conf=0.05)
    channel = HandChannel(session, max_rois=2, interval_s=0.25, hold_s=0.2)
    channel.update([_person()], _frame(), 0.0)
    calls = session.calls
    channel.update([_person()], _frame(), 5.0)
    assert session.calls > calls


def test_the_hold_expires_on_the_last_success_not_the_last_attempt():
    """Recording a failed attempt against the hold would keep a stale hand on
    screen forever."""
    channel = HandChannel(_HandSession(), max_rois=2, interval_s=0.0,
                          hold_s=0.3)
    channel.update([_person()], _frame(), 0.0)
    channel._session = _HandSession(conf=0.05)
    for step in range(1, 12):               # keeps attempting, never succeeds
        p = _person()
        channel.update([p], _frame(), step * 0.1)
    assert p["hands"]["right"] is None, "a stale hand outlived its hold"




def test_the_hand_engine_wrapper_refuses_a_non_square_input():
    """The crop is a square cut out of the frame; a non-square input would
    stretch it and put every joint on a hand shape nobody has."""
    import plugins.hand_runtime as hr

    class _Eng:
        input_shape = (1, 3, 256, 192)
        optimization_shape = (1, 3, 256, 192)
        input_dtype = np.float32
        output_names = ["simcc_x", "simcc_y"]
        def __init__(self, *a, **k): pass
        def infer(self, blob): return []
        def close(self): pass

    import utils.tensorrt_runtime as trt
    original = trt.TensorRTEngine
    trt.TensorRTEngine = _Eng
    try:
        with pytest.raises(HandDecodeError) as excinfo:
            hr.HandEngine("/nowhere.engine")
        assert "square" in str(excinfo.value)
    finally:
        trt.TensorRTEngine = original
