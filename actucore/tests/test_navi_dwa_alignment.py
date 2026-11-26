"""Native DWA alignment about an offset front camera.

The body can rotate in place while its front camera traces an arc away from a
straight path reference.  Route following must not veto that useful alignment
after the complete physical stop has passed its actual footprint check.
"""
import math

import numpy as np
import pytest

from plugins.navi import dwa, local_path, motion
from plugins.navi.camera import HorizontalIntrinsics


def _case(side=1., yaw=.4, *, budgeted=True):
    geometry = local_path.camera_square(
        HorizontalIntrinsics(160, 100., 80.), (120, 100., 60.), .275,
        at_front=True)
    # A nearby wall corner prevents advancing before turning into its open side.
    occupied = ((abs(local_path.FORWARD - .2) < .026)
                & (side * local_path.RIGHT >= -.05))
    angle = math.radians(50.) * side
    options = dict(
        route=((0., 0.), (.6 * math.cos(angle), -.6 * math.sin(angle))),
        grid=(~occupied, occupied), geometry=geometry,
        footprint=motion.FootprintSpec(geometry.front, geometry.rear, .275, .0275, "camera_approx"),
        measured=(0., 0., side * yaw), previous=(0., 0., side * yaw),
        x_limit=(0., .15), y_limit=(0., 0.), yaw_limit=(-1., 1.),
        floors=(.15, 0., 1., .04), target=(3., -3. * side), stop_distance=.8,
        horizon=1.2, max_candidates=64, supported_region=local_path.SUPPORTED_REGION)
    if budgeted:
        options["candidate_budget"] = dwa.CandidateBudget(64)
    return options


def _context(options):
    return dwa.MotionContext(
        options["measured"], options["previous"], options["horizon"], 0.,
        options["geometry"], options["footprint"],
        local_path.occupied_squares(options["grid"][1]), options.get("remembered"),
        options["supported_region"], options["target"], options["stop_distance"])


def _score(options, command, trace, checked):
    reference = dwa._score_route(options["route"], .45)
    return dwa.score_candidate(
        command, trace, checked, reference, options["previous"],
        half_width=.3025,
        limits=(options["x_limit"], options["y_limit"], options["yaw_limit"]),
        progress_scale=.2)


@pytest.mark.parametrize("budgeted", [False, True], ids=["direct", "decision-budget"])
@pytest.mark.parametrize("side", [-1., 1.], ids=["right", "left"])
@pytest.mark.parametrize("yaw", [.4, 1.], ids=["recover-selection", "keep-turn-direction"])
def test_camera_arc_does_not_veto_physically_checked_in_place_alignment(budgeted, side, yaw):
    options = _case(side, yaw, budgeted=budgeted)
    command = (0., 0., side)
    predicted, checked = _context(options).check(command)
    assert checked.accepted and predicted.stopped
    initial = predicted.body_poses[0][:2]
    assert all(math.dist(pose[:2], initial) < 1e-12 for pose in predicted.body_poses)
    # The camera moves on the chassis radius, although the chassis has not translated.
    end = predicted.camera_poses[predicted.moving_count - 1]
    direction = math.atan2(options["route"][-1][1], options["route"][-1][0])
    assert abs(math.remainder(end[2] - direction, math.tau)) < abs(direction)

    diagnostics = {}
    selected = dwa.select(**options, diagnostics=diagnostics)
    assert selected is not None, diagnostics
    assert selected.command == command
    assert selected.check.accepted and selected.trace.stopped
    assert selected.progress == 0.
    assert diagnostics["rejected"].get("footprint_collision", 0) > 0


@pytest.mark.parametrize("side", [-1., 1.], ids=["right", "left"])
def test_alignment_still_checks_new_obstacles_in_the_rotating_footprint(side):
    options = _case(side)
    occupied = options["grid"][1].copy()
    row = local_path.START[0] + round(-.2 / local_path.RESOLUTION)
    col = local_path.START[1] + round(-side * .4 / local_path.RESOLUTION)
    occupied[row, col] = True
    options["grid"] = (~occupied, occupied)
    command = (0., 0., side)
    predicted, checked = _context(options).check(command)
    assert not checked.accepted and checked.reason == "footprint_collision"
    selected = dwa.select(**options)
    assert selected is None or selected.command != command


@pytest.mark.parametrize("side", [-1., 1.], ids=["right", "left"])
def test_alignment_does_not_ignore_residual_translation_before_stopping(side):
    options = _case(side)
    options.update(measured=(.3, 0., side * .4), previous=(.3, 0., side * .4))
    command = (0., 0., side)
    predicted, checked = _context(options).check(command)
    assert math.dist(predicted.body_poses[0][:2], predicted.body_poses[-1][:2]) > .05
    assert not checked.accepted and checked.reason == "footprint_collision"
    selected = dwa.select(**options)
    assert selected is None or selected.command != command


@pytest.mark.parametrize("side", [-1., 1.], ids=["right", "left"])
def test_turn_away_from_reference_is_not_admitted_as_alignment(side):
    options = _case(side)
    command = (0., 0., -side)
    predicted, checked = _context(options).check(command)
    assert checked.accepted
    components, progress, reason = _score(options, command, predicted, checked)
    assert components is None and reason == "insufficient_progress"


def test_translating_off_route_motion_keeps_its_existing_rejection():
    options = _case()
    free = np.ones(local_path.SHAPE, dtype=bool)
    options.update(grid=(free, ~free), measured=(.4, 0., 0.), previous=(.4, 0., 0.),
                   x_limit=(.4, .4), yaw_limit=(0., 0.))
    command = (.4, 0., 0.)
    predicted, checked = _context(options).check(command)
    assert checked.accepted and predicted.stopped
    components, progress, reason = _score(options, command, predicted, checked)
    assert progress > .025
    assert components is None and reason == "off_route"
    diagnostics = {}
    assert dwa.select(**options, diagnostics=diagnostics) is None
    assert diagnostics["rejected"] == {"off_route": 1}
