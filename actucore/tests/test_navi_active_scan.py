import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from plugins.navi import active_scan as scan
from plugins.navi.camera import HorizontalIntrinsics
from plugins.navi.depth import ground_mask
from plugins.navi.local_path import Geometry, body_square, camera_square
from plugins.navi.odom import PlanarPose


def _depth(stamp, occupied=None):
    free = np.ones(scan.SHAPE, dtype=bool)
    if occupied is None:
        occupied = np.zeros_like(free)
    return {"map": free, "ground": free, "capture_ms": stamp, "frame_id": stamp}


def _odom(stamp, wz=0):
    return {"vx": 0.0, "vy": 0.0, "wz": wz, "stamp_ms": stamp}


def _setup(monkeypatch):
    blocked = np.zeros(scan.SHAPE, dtype=bool)
    visible = np.ones_like(blocked)
    monkeypatch.setattr(scan, "observe",
                        lambda *args: (visible.copy(), blocked.copy()))
    lookout = scan.ActiveScan()
    geometry = Geometry(None, None, np.eye(4), .23, .23, None)
    return lookout, geometry, visible, blocked


def _tick(lookout, geometry, stamp, wz=0, capture=True):
    depth = _depth(stamp if capture is True else capture)
    lookout.update(depth, _odom(stamp, wz), geometry)
    return lookout.command(geometry, .33, .25, 600)


@pytest.mark.parametrize("command_only", [False, True])
def test_receipt_only_projects_only_the_latest_requested_frame(monkeypatch, command_only):
    geometry = camera_square(HorizontalIntrinsics(64, 50, 32), (48, 50, 24), .23)
    tangent = (np.arange(48) - 24) / 50
    image = np.broadcast_to(np.minimum(.42 / np.maximum(tangent, .001), 4.)[:, None],
                            (48, 64)).copy()
    ground = image < 4.
    newer = image.copy()
    newer[:25, 20:45] = 1.
    expected = scan.observe(newer, geometry, scan.RIGHT, scan.FORWARD, scan.RESOLUTION, ground)
    observe = scan.observe
    projected = []

    def project(frame, *args):
        projected.append(frame)
        return observe(frame, *args)

    monkeypatch.setattr(scan, "observe", project)
    look = scan.ActiveScan()
    for stamp, frame in ((1000, image), (1100, newer)):
        depth = {"map": frame, "ground": ground, "frame_id": stamp}
        if command_only:
            assert look.observe_single(depth, geometry, 500, stamp) is None
        else:
            odom = {**_odom(stamp), "received_ms": stamp}
            assert look.update(depth, odom, geometry, receipt_only=True) is None
    assert look.current_frame == 1100
    assert not projected
    for actual, wanted in zip(look.grid(), expected):
        np.testing.assert_array_equal(actual, wanted)
    look.grid()
    assert len(projected) == 1 and projected[0] is newer


def test_active_scan_turns_right_and_can_return_using_the_current_view(monkeypatch):
    lookout, geometry, _, blocked = _setup(monkeypatch)
    _tick(lookout, geometry, 1000)
    assert lookout.begin((1., -1.))
    assert _tick(lookout, geometry, 1050).yaw == 0
    assert _tick(lookout, geometry, 1150).yaw < 0
    for stamp in range(1250, 2550, 100):
        result = _tick(lookout, geometry, stamp, -.25)
    assert result.yaw == 0
    assert math.degrees(lookout.pose[2]) == pytest.approx(20., abs=2.)
    assert _tick(lookout, geometry, 2500, capture=2450).yaw == 0
    assert not lookout.ready
    assert lookout.phase == "return" and lookout.side_frame == 2450
    blocked[30, 38] = True
    assert _tick(lookout, geometry, 2600).yaw > 0
    blocked[30, 38] = False
    for stamp in range(2700, 4000, 100):
        result = _tick(lookout, geometry, stamp, .25)
    assert result.yaw == 0
    _tick(lookout, geometry, 4000)
    assert lookout.ready
    _, occupied = lookout.grid()
    assert occupied.any()
    assert not occupied[30, 38]  # the side observation was reprojected
    assert lookout.try_other_side()
    assert _tick(lookout, geometry, 4100).yaw == 0
    assert _tick(lookout, geometry, 4200).yaw > 0


def test_active_scan_holds_an_occupied_sweep(monkeypatch):
    lookout, geometry, _, blocked = _setup(monkeypatch)
    blocked[14, 28] = True
    _tick(lookout, geometry, 1000)
    assert lookout.begin((1., -1.))
    for stamp in (1050, 1200, 1250, 1350):
        assert _tick(lookout, geometry, stamp).yaw == 0
    assert not lookout.active
    assert not lookout.begin((1., -1.))


@pytest.mark.parametrize("feedback", ["pose", "receipt_twist", "aligned_twist"])
@pytest.mark.parametrize("start_delay_ms", [0, 250])
def test_scan_converges_with_delayed_feedback_and_chassis(monkeypatch, start_delay_ms, feedback):
    lookout, geometry, _, _ = _setup(monkeypatch)
    history = {stamp: 0. for stamp in range(99_600, 100_000, 100)}
    yaw, rate, turn_start = 0., 0., None
    measured_twist = feedback != "pose"
    angle = math.atan2(.45, 1.85) if measured_twist else scan.ANGLE
    for stamp in range(100_000, 115_000, 100):
        if turn_start is not None:
            moving_ms = min(100, max(0, stamp - turn_start - start_delay_ms))
            yaw += rate * moving_ms / 1000
        history[stamp] = yaw
        measured = stamp - 300
        rate_at_measurement = (history[measured] - history[measured - 100]) / .1
        odom = {"received_ms": stamp, "source_sample_id": measured, "stamp_ms": measured}
        if measured_twist:
            odom.update(vx=0., vy=0., wz=rate_at_measurement)
        else:
            pose = PlanarPose(measured, ("odom", "body"), 0., 0., history[measured])
            odom.update(motion_pose=pose, pose_feedback=(0., 0., rate_at_measurement),
                        pose_feedback_since_ms=measured - 100)
        depth = dict(_depth(stamp), control_ms=stamp, capture_ms=measured)
        lookout.update(depth, odom, geometry, receipt_only=feedback != "aligned_twist")
        if stamp == 100_000:
            assert lookout.begin((-1., 1.), advance_m=1.9 if measured_twist else None, turn_angle=angle)
        rate = lookout.command(geometry, .31, 1. if measured_twist else .83, 300,
                               min_yaw_rate=0. if measured_twist else .33, now_ms=stamp).yaw
        turn_start = (turn_start if turn_start is not None else stamp) if rate else None
        assert abs(yaw) <= angle + scan.ANGLE_TOLERANCE
        if lookout.ready:
            break

    assert lookout.ready
    assert yaw == pytest.approx(angle, abs=scan.ANGLE_TOLERANCE)


@pytest.mark.parametrize("stalled", [False, True])
def test_slow_measured_turn_excludes_stop_waits_from_stall_window(monkeypatch, stalled):
    look, geometry, _, _ = _setup(monkeypatch)
    rate = 0.
    start = None
    angle = math.radians(6)
    for stamp in range(1000, 11000, 100):
        # At low speed this chassis needs 200 ms to start each burst.
        measured = rate if start is not None and stamp - start > 200 and not stalled else 0.
        look.update(_depth(stamp), {**_odom(stamp, measured), "received_ms": stamp},
                    geometry, receipt_only=True)
        if stamp == 1000:
            assert look.begin((-1.,), advance_m=1., turn_angle=angle)
        result = look.command(geometry, .375, .1, 500, 800, now_ms=stamp)
        rate = result.yaw
        start = (start if start is not None else stamp) if rate else None
        if look.ready or "转向无实测进展" in result.reason:
            break
    if stalled:
        assert "转向无实测进展" in result.reason
    else:
        assert look.ready
        assert -look.pose[2] == pytest.approx(angle, abs=scan.ANGLE_TOLERANCE)


def test_measured_turn_deadline_stops_and_waits_for_new_feedback(monkeypatch):
    look, geometry, _, _ = _setup(monkeypatch)

    def tick(now, received, source, wz=0.):
        look.update(dict(_depth(now), control_ms=now),
                    {"vx": 0., "vy": 0., "wz": wz, "received_ms": received,
                     "source_sample_id": source}, geometry, receipt_only=True)
        return look.command(geometry, .375, 1., 500, 800).yaw

    tick(1000, 1000, 10)
    assert look.begin((-1.,), advance_m=1.9, turn_angle=math.radians(14))
    assert tick(1100, 1100, 11) == 0.
    assert tick(1200, 1200, 12) > 0.
    assert tick(1400, 1400, 13) > 0.
    assert tick(1500, 1500, 14) > 0.
    assert tick(1600, 1580, 14) == 0.  # deadline on the control clock, repeated input
    assert tick(1650, 1640, 14) == 0.
    assert tick(1700, 1680, 15) == 0.
    assert tick(1800, 1800, 16, wz=.12) == 0.  # in-flight rotation after braking
    assert tick(1900, 1900, 17, wz=-.08) == 0.
    assert tick(2050, 2050, 18) == 0.
    assert tick(2150, 2140, 18) == 0.  # a repeated sample cannot release the stop
    assert tick(2200, 2200, 19, wz=-.08) == 0.
    assert tick(2300, 2300, 20) > 0.


@pytest.mark.parametrize("case", [
    "continuous", "gap", "batched_gap", "duplicate", "translation", "expired",
])
def test_receipt_scan_checks_buffered_odom_independently_of_the_control_tick(monkeypatch, case):
    from plugins.navi import plugin

    look, geometry, _, _ = _setup(monkeypatch)
    card = plugin.NaviPlugin({"local_planning": True}, executor=None)
    now = 1000
    monkeypatch.setattr(plugin.time, "time", lambda: now / 1000)

    def sample(stamp, vx=0., wz=0., source=None):
        nonlocal now
        now = stamp
        payload = dict(schema="motus.odom/1", frame="body", stamp_ms=stamp if source is None else source,
                       twist=[vx, 0., None, None, None, wz])
        card._on_string("odom", SimpleNamespace(data=json.dumps(payload)))

    def update():
        odom = card._observation_snapshot(with_capture_timing=True)[2]
        return look.update(dict(_depth(now), control_ms=now), odom, geometry, receipt_only=True)

    def command():
        return look.command(geometry, .33, 1., 600)

    sample(1000)
    assert update() is None
    assert look.begin((-1.,), advance_m=1.5, route_segment=True)
    sample(1100)
    assert update() is None and command().yaw == 0.
    sample(1200)
    assert update() is None and command().yaw > 0.

    # Drive the callbacks between two control ticks, including a transient
    # excursion that has already ended by the time the planner takes a snapshot.
    if case == "gap":
        sample(1490, wz=.10)
    elif case == "batched_gap":
        sample(1300, wz=.10, source=1500)
    else:
        sample(1300, vx=.1 if case == "translation" else 0., wz=.10)
        sample(1420, wz=.10, source=1300 if case == "duplicate" else None)
    sample(1620 if case == "expired" else 1500, wz=.10, source=1300 if case == "duplicate" else None)
    hold = update()
    if case == "continuous":
        assert hold is None
        assert look.pose[2] == pytest.approx(-.030)
        assert command().yaw > 0.
    elif case == "expired":
        assert hold is None
        assert command().yaw == 0.
        assert look.turn_stop_ms == 1620 and look.turn_until_ms == 0
    elif case == "duplicate":
        assert hold is None and look.poses[-1][0] == 1300
        sample(1600, wz=.10, source=1300)
        assert update().yaw == 0.
        assert look.locked
    else:
        assert look.phase == "brake_for_replan" and hold.yaw == 0.
        assert ("检测到平移" if case == "translation" else "间隔过长") in hold.reason


@pytest.mark.parametrize("rate_hz", [10., 2.])
def test_card_uses_configured_period_only_to_start_a_turn_burst(monkeypatch, rate_hz):
    from plugins.navi import plugin

    look, geometry, _, _ = _setup(monkeypatch)
    card = plugin.NaviPlugin({"local_planning": True, "wz_max": 1.}, executor=None)
    card._config.local_geometry = geometry
    card._rate_hz = rate_hz
    card._ttl_ms = 400
    card._state.target = "person"
    card._state.active_scan = look
    now = 100_000
    history = []

    def snapshot(**kwargs):
        odom = dict(_odom(now), received_ms=now, source_sample_id=now, velocity_history=tuple(history))
        return {"objects": []}, dict(_depth(now), control_ms=now), odom, now

    def tick(stamp):
        nonlocal now
        for sample in range(now + 100, stamp, 100):
            history.append((sample, sample, (0., 0., 0.)))
        now = stamp
        history.append((now, now, (0., 0., 0.)))
        return card.next_command()

    monkeypatch.setattr(card, "_observation_snapshot", snapshot)
    monkeypatch.setattr(plugin.time, "monotonic", lambda: now / 1000)
    monkeypatch.setattr(plugin.time, "time", lambda: now / 1000)
    _, depth, odom, _ = snapshot()
    look.update(depth, odom, geometry, receipt_only=True)
    assert look.begin((-1.,), advance_m=1.9, route_segment=True)
    assert tick(100_100)["values"] == [0.] * 6

    first = tick(100_550)  # 450 ms since the last tick, even with fresh odometry.
    if rate_hz == 2.:
        assert first["values"] == [0.] * 6
        return

    assert first["values"][5] > 0. and first["ttl_ms"] == 400
    assert look.turn_until_ms == 100_950
    # The 150 ms left would fit the nominal 100 ms period, but not the
    # preceding 250 ms tick. Renewal keeps the existing conservative check.
    assert tick(100_800)["values"] == [0.] * 6
    assert look.turn_stop_ms == 100_800 and look.turn_until_ms == 0


@pytest.mark.parametrize("expiry", ["first_yaw", "renewal", "snapshot"])
def test_card_finishing_a_turn_command_after_its_deadline_sends_stop(monkeypatch, expiry):
    from plugins.navi import plugin

    look, geometry, _, _ = _setup(monkeypatch)
    card = plugin.NaviPlugin({"local_planning": True, "wz_max": 1.}, executor=None)
    card._config.local_geometry = geometry
    card._config.watchdog_ms = 500
    card._descriptor = {"rate": {"max_obs_age_ms": 600}}
    card._ttl_ms = 400
    card._state.target = "person"
    card._state.active_scan = look
    now = 100_000
    lag = 0

    def snapshot(**kwargs):
        return ({"objects": []}, dict(_depth(now), control_ms=now),
                dict(_odom(now), received_ms=now, source_sample_id=now), now)

    monkeypatch.setattr(card, "_observation_snapshot", snapshot)
    monkeypatch.setattr(plugin.time, "monotonic", lambda: now / 1000)
    monkeypatch.setattr(plugin.time, "time", lambda: (now + lag) / 1000)
    _, depth, odom, _ = snapshot()
    look.update(depth, odom, geometry, receipt_only=True)
    assert look.begin((-1.,), advance_m=1.9, turn_angle=math.radians(14), route_segment=True)
    now += 100
    assert card.next_command()["values"] == [0.] * 6
    now += 100
    if expiry == "first_yaw":
        lag = 450
    else:
        assert card.next_command()["values"][5] > 0.
        now += 200
        lag = 50
        renewal = card.next_command()
        assert renewal["values"][5] > 0.
        assert renewal["stamp_ms"] + renewal["ttl_ms"] <= look.turn_until_ms
        now = 100_600 if expiry == "snapshot" else 100_500
        lag = 100_650 - now
    stop = card.next_command()
    assert stop["values"] == [0.] * 6
    assert stop["obs_stamp_ms"] == stop["stamp_ms"] == 100_650
    assert look.turn_stop_ms == stop["stamp_ms"] and look.turn_until_ms == 0
    assert look.issued_yaw == (expiry != "first_yaw")
    assert not any(card._state.commanded)
    # A later clock with the same velocity sample cannot reopen the burst.
    assert look.command(geometry, .375, 1., 500, 800, now_ms=101_200).yaw == 0.
    assert look.turn_stop_ms == stop["stamp_ms"]


def test_active_scan_stops_with_stale_odom_and_cannot_restart_on_same_obstacle(monkeypatch):
    lookout, geometry, _, _ = _setup(monkeypatch)
    _tick(lookout, geometry, 1000)
    assert lookout.begin((1., -1.))
    _tick(lookout, geometry, 1050)
    assert _tick(lookout, geometry, 1200).yaw < 0
    lookout.update(_depth(1250), None, geometry)
    assert not lookout.active
    assert not lookout.begin((1., -1.))


@pytest.mark.parametrize("lose_depth, yaw_limit", [(False, .83), (False, .60), (True, .83)])
def test_command_only_scan_evaluates_when_the_timed_turn_finishes(monkeypatch, lose_depth, yaw_limit):
    look, geometry, _, _ = _setup(monkeypatch)
    look.observe_single(_depth(1000), geometry, 800, 1000)
    assert look.begin((1., -1.))
    rates = []
    first_turn = None
    for stamp in range(1100, 7400, 100):
        depth = None if lose_depth and first_turn is not None else _depth(stamp)
        look.observe_single(depth, geometry, 800, stamp)
        command = look.command(geometry, .31, yaw_limit, 300, 800, min_yaw_rate=.33)
        if command.yaw and first_turn is None:
            first_turn = stamp
        rates.append(command.yaw)
        if lose_depth and first_turn is not None and depth is None:
            assert command.yaw == 0 and not look.active
            return
        if look.ready:
            break
    assert first_turn is not None
    nonzero = [rate for rate in rates if rate]
    assert all(-yaw_limit <= rate <= -.33 for rate in nonzero)
    assert math.degrees(-sum(nonzero) * .1) == pytest.approx(20., abs=2.)
    assert look.ready and stamp == first_turn + 100 * len(nonzero)
    assert len(look.views) == 1 and not look.poses


@pytest.mark.parametrize("blocked", [False, True])
def test_command_only_scan_checks_obstacles_with_unseen_rear(monkeypatch, blocked):
    look, geometry, free, occupied = _setup(monkeypatch)
    free[:] = scan.FORWARD > 0.
    row, col = scan.START
    occupied[row + 4, col + 6] = blocked
    look.observe_single(_depth(1000), geometry, 800, 1000)
    assert look.begin((1.,))
    for stamp in range(1100, 1700, 100):
        look.observe_single(_depth(stamp), geometry, 800, stamp)
        result = look.command(geometry, .31, .83, 300, 800, min_yaw_rate=.33)
        if result.yaw:
            assert not blocked and result.yaw < 0.
            break
    assert look.issued_yaw is (not blocked)
    assert look.active is (not blocked)


@pytest.mark.parametrize("first", [-1., 1.])
def test_command_only_scan_turns_to_the_far_side_of_its_original_heading(monkeypatch, first):
    look, geometry, _, _ = _setup(monkeypatch)
    look.observe_single(_depth(1000), geometry, 800, 1000)
    assert look.begin((first, -first))
    yaw = heading = 0.
    switched = False
    for stamp in range(1100, 17000, 100):
        look.observe_single(_depth(stamp), geometry, 800, stamp)
        heading += yaw * .1
        yaw = look.command(geometry, .31, .83, 300, 800, min_yaw_rate=.33,
                           previous_yaw=yaw).yaw
        if look.ready and not switched:
            assert math.degrees(heading) == pytest.approx(-first * 20., abs=2.)
            assert look.try_other_side()
            switched = True
        elif look.ready:
            assert math.degrees(heading) == pytest.approx(first * 20., abs=2.)
            return
    pytest.fail("The other heading was never observed")


def test_command_only_scan_reuses_the_current_view_after_turning(monkeypatch):
    look, geometry, _, _ = _setup(monkeypatch)
    look.observe_single(_depth(1000), geometry, 800, 1000)
    assert look.begin((1.,))
    look.phase = "turn"
    look.timed_heading = look.turn_angle
    look.observe_single(_depth(1000), geometry, 800, 1100)
    assert look.command(geometry, .31, .83, 300).yaw == 0.
    assert look.ready and look.side_frame == 1000


def test_scan_rejects_unexpected_yaw_rate(monkeypatch):
    lookout, geometry, _, _ = _setup(monkeypatch)
    _tick(lookout, geometry, 1000)
    assert lookout.begin((-1., 1.))
    _tick(lookout, geometry, 1050)
    assert _tick(lookout, geometry, 1200).yaw > 0
    assert _tick(lookout, geometry, 1250, wz=.55).yaw == 0
    assert lookout.locked


def test_scan_time_limit_applies_even_with_fresh_frames_and_no_angular_progress(monkeypatch):
    lookout, geometry, _, _ = _setup(monkeypatch)
    _tick(lookout, geometry, 1000)
    assert lookout.begin((-1., 1.))
    _tick(lookout, geometry, 1050)
    for stamp in range(1200, 9200, 200):
        assert _tick(lookout, geometry, stamp).yaw > 0
    assert _tick(lookout, geometry, 9200).yaw == 0
    assert lookout.locked


@pytest.mark.parametrize("at_stop", [1250, 1050, None, 0])
def test_drift_before_first_yaw_can_replan_with_the_current_depth(monkeypatch, at_stop):
    look, geometry, _, _ = _setup(monkeypatch)

    def tick(stamp, source, vx=0.):
        look.update(dict(_depth(stamp), source_frame_id=source),
                    {**_odom(stamp), "received_ms": stamp, "vx": vx},
                    geometry, receipt_only=True)
        return look.command(geometry, .375, 1., 500)

    tick(1000, 1000)
    assert look.begin((1., -1.))
    assert tick(1050, 1050).yaw == 0.
    assert tick(1150, 1050, vx=.12).yaw == 0.
    assert "重新规划" in tick(1250, at_stop).reason
    assert not look.issued_yaw
    assert look.begin((-1., 1.))


def test_receipt_only_scan_replans_after_measured_translation_has_settled(monkeypatch):
    look, geometry, _, _ = _setup(monkeypatch)

    def tick(stamp, vx=0., vy=0., yaw=0.):
        look.update(_depth(stamp), {"vx": vx, "vy": vy, "wz": yaw, "received_ms": stamp},
                    geometry, receipt_only=True, max_depth_age_ms=800)
        return look.command(geometry, .33, 1., 600, 800)

    tick(1000)
    assert look.begin((1., -1.))
    assert tick(1050).yaw == 0.
    assert tick(1150).yaw < 0.
    assert tick(1250, vx=.113, vy=-.014, yaw=-.204).yaw == 0.
    assert look.active and not look.locked
    assert tick(1400, vx=-.006, vy=-.035).yaw == 0.
    assert not look.active and not look.locked and look.attempted


@pytest.mark.parametrize("interruption", ["translation", "gap"])
def test_interrupted_turn_can_replan_once_with_the_current_view(monkeypatch, interruption):
    look, geometry, _, _ = _setup(monkeypatch)

    def tick(stamp, source, vx=0.):
        hold = look.update(dict(_depth(stamp), source_frame_id=source),
                           dict(_odom(stamp), received_ms=stamp, vx=vx),
                           geometry, receipt_only=True)
        return hold or look.command(geometry, .33, 1., 600)

    tick(1000, 10)
    assert look.begin((-1.,), advance_m=1.5, route_segment=True)
    assert tick(1100, 11).yaw == 0.
    assert tick(1200, 12).yaw > 0.
    stopped = 1550 if interruption == "gap" else 1300
    assert tick(stopped, 13, vx=.1 if interruption == "translation" else 0.).yaw == 0.
    assert look.phase == "brake_for_replan" and look.attempted
    assert tick(stopped + 100, 12 if interruption == "gap" else 13).replanned_turn
    assert not look.active and not look.attempted and look.advance_m is None
    assert look.begin((-1.,), advance_m=1.5, route_segment=True)
    assert tick(stopped + 200, 17).yaw == 0.
    assert tick(stopped + 300, 18).yaw > 0.
    assert tick(stopped + 400, 19, vx=.1).yaw == 0.
    assert tick(stopped + 500, 20).yaw == 0.
    assert look.locked and not look.begin((-1.,))


def test_receipt_scan_with_no_yaw_progress_replans_before_standing_bias_locks_it(monkeypatch):
    look, geometry, _, _ = _setup(monkeypatch)

    def tick(stamp):
        look.update(_depth(stamp), {**_odom(stamp), "vy": -.039, "received_ms": stamp},
                    geometry, receipt_only=True, max_depth_age_ms=800)
        return look.command(geometry, .30, 1., 600, 800)

    tick(1000)
    assert look.begin((1., -1.))
    assert tick(1050).yaw == 0.
    assert tick(1150).yaw < 0.
    for stamp in range(1250, 2550, 100):
        tick(stamp)
    hold = tick(2550)
    assert hold.yaw == 0. and "转向无实测进展" in hold.reason
    # A subsequent missing frame must not make a stalled turn retryable.
    look.update(None, {**_odom(2600), "received_ms": 2600},
                geometry, receipt_only=True, max_depth_age_ms=800)
    tick(2650)
    assert tick(2800).yaw == 0.
    assert not look.active and not look.locked and look.attempted


@pytest.mark.parametrize("missing", ["depth", "odom", "poll_gap"])
def test_receipt_scan_stops_then_replans_after_temporary_input_loss(monkeypatch, missing):
    look, geometry, _, _ = _setup(monkeypatch)

    def update(stamp, depth=True, odom=True, wz=0.):
        return look.update(_depth(stamp) if depth else None,
                           {**_odom(stamp, wz), "received_ms": stamp} if odom else None,
                           geometry, receipt_only=True, max_depth_age_ms=800)

    update(1000)
    assert look.begin((1., -1.))
    update(1050)
    assert look.command(geometry, .33, 1., 600, 800).yaw == 0.
    update(1150)
    assert look.command(geometry, .33, 1., 600, 800).yaw < 0.
    loss = 1500 if missing == "poll_gap" else 1250
    hold = update(loss, depth=missing != "depth", odom=missing != "odom")
    assert hold.yaw == 0. and look.active and not look.locked
    for t, yaw in ((loss + 100, -.12), (loss + 200, 0.), (loss + 350, 0.)):
        update(t, wz=yaw)
        assert look.command(geometry, .33, 1., 600, 800).yaw == 0.
    assert not look.attempted and not look.has_result and not look.active and not look.locked
    assert look.begin((-1., 1.))


@pytest.mark.parametrize(("motion", "reason"), [
    ((-.0028, -.0353, -.2373), "仍在转动，停车后重规划"),
    ((.0936, -.0349, .0753), "检测到平移，停车后重规划"),
])
def test_receipt_scan_distinguishes_rotation_from_translation_while_stopping(monkeypatch, motion, reason):
    look, geometry, _, _ = _setup(monkeypatch)

    def update(stamp, motion=(0., 0., 0.)):
        vx, vy, wz = motion
        return look.update(_depth(stamp), dict(vx=vx, vy=vy, wz=wz, received_ms=stamp),
                           geometry, receipt_only=True)

    update(1000)
    assert look.begin((1.,))
    assert look.command(geometry, .33, 1., 600).yaw == 0.
    assert update(1358).yaw == 0.

    assert update(1444, motion) is None
    hold = look.command(geometry, .33, 1., 600)
    assert hold.yaw == 0.
    assert look.phase == "brake_for_replan"
    assert hold.reason == f"主动观察：{reason}"


@pytest.mark.parametrize("released", [True, False])
def test_receipt_scan_replan_after_travel_locks_only_before_a_released_straight_segment(
        monkeypatch, released):
    """Tianyi 2.0, 2026-11-20 06:06:59: depth paused for 2.3 s while the scan's
    straight segment had already covered 0.62 m. The replan hold must use
    the current view once feedback is available. Drift without a released segment
    is still an unexplained translation and keeps locking."""
    look, geometry, _, _ = _setup(monkeypatch)

    def update(stamp, vx=0., depth=True):
        return look.update(_depth(stamp) if depth else None,
                           {"vx": vx, "vy": 0., "wz": 0., "received_ms": stamp},
                           geometry, receipt_only=True, max_depth_age_ms=800)

    update(1000)
    assert look.begin((1., -1.), advance_m=1.)
    update(1050)
    assert look.command(geometry, .33, 1., 600, 800).yaw == 0.
    update(1150)
    assert look.command(geometry, .33, 1., 600, 800).yaw < 0.
    look.phase, look.issued_yaw = "evaluate", True
    if released:
        look.advance_origin = look.pose
    # 0.6 m of forward travel, as the policy's released straight segment produces.
    for stamp in range(1250, 5300, 100):
        assert update(stamp, vx=.15) is None
    assert math.dist(look.pose[:2], look.origin[:2]) > .18
    hold = update(5350, depth=False)
    assert hold.yaw == 0. and "停车后重规划" in hold.reason and look.phase == "brake_for_replan"
    for stamp in range(5450, 7650, 100):
        assert update(stamp, depth=False).reason == hold.reason
    assert update(7700) is None
    first = look.command(geometry, .33, 1., 600, 800)
    assert first.yaw == 0.
    if not released:
        assert not look.active and look.locked
        return
    assert "重新规划" in first.reason
    assert not look.active and not look.locked and not look.issued_yaw and not look.attempted


@pytest.mark.parametrize("missing", ["depth", "odom"])
def test_receipt_scan_retries_if_input_lapses_while_braking_before_first_turn(monkeypatch, missing):
    look, geometry, _, _ = _setup(monkeypatch)

    def update(stamp, *, depth=True, odom=True, vx=0.):
        return look.update(_depth(stamp) if depth else None,
                           {**_odom(stamp), "received_ms": stamp, "vx": vx} if odom else None,
                           geometry, receipt_only=True)

    for stamp in range(1000, 1900, 100):
        update(stamp, vx=.3)
    assert look.pose[0] > .18
    assert look.begin((-1., 1.))
    assert not look.origin_set and not look.issued_yaw
    update(1900, depth=missing != "depth", odom=missing != "odom")
    assert look.phase == "brake_for_replan" and look.attempted

    update(2000)
    assert "已停稳，重新规划" in look.command(geometry, .33, 1., 600).reason
    assert not look.locked
    assert look.begin((-1., 1.))


def test_scan_waits_for_the_odometry_bracket_instead_of_discarding_the_depth(monkeypatch):
    lookout, geometry, _, _ = _setup(monkeypatch)
    _tick(lookout, geometry, 1000)
    assert lookout.begin((-1., 1.))
    _tick(lookout, geometry, 1050)
    waiting = lookout.update(_depth(1120), _odom(1100), geometry)
    assert waiting.yaw == 0 and lookout.active
    assert lookout.update(_depth(1220), _odom(1200), geometry) is None
    assert lookout.current_frame == 1120
    assert lookout.command(geometry, .33, .25, 600).yaw > 0


def test_scan_allows_an_unobserved_rear_sweep(monkeypatch):
    lookout, geometry, visible, _ = _setup(monkeypatch)
    visible[:] = scan.FORWARD > 0
    _tick(lookout, geometry, 1000)
    assert lookout.clear_sweep(geometry, .33, scan.ANGLE, 0.)
    assert lookout.clear_sweep(geometry, .33, -scan.ANGLE, 0.)


def test_s10_search_sweep_is_checked_against_hits_not_the_rear_grid_extent():
    geometry = camera_square(object(), (480, 420, 243), .23)
    occupied = np.zeros_like(scan.RIGHT, dtype=bool)
    horizon = (600 + 800 + 50) / 1000
    sweep = (.60 + .10) * horizon
    lookout = scan.ActiveScan()

    for direction in (-1, 1):
        assert lookout.clear_sweep(geometry, .33, direction * sweep,
                                   scan.STILL_LINEAR * horizon, occupied=occupied)
    occupied[scan.START[0] + 6, scan.START[1]] = True
    for direction in (-1, 1):
        assert not lookout.clear_sweep(geometry, .33, direction * sweep,
                                       scan.STILL_LINEAR * horizon, occupied=occupied)


def test_body_geometry_does_not_assume_the_camera_or_cloud_is_the_yaw_pivot():
    geometry = camera_square(object(), (480, 420, 243), .23)
    transform = geometry.base_from_optical.copy()
    transform[:3, 3] = [.27, .04, .06]
    inverse = np.linalg.inv(transform).tolist()
    assert body_square(geometry, {"camera_optical_from_cloud": inverse,
                                  "cloud_frame_id": "lidar_link"}) is None
    mounted = body_square(geometry, {"camera_optical_from_cloud": inverse,
                                     "cloud_frame_id": "base_link"})
    np.testing.assert_allclose(mounted.base_from_optical, transform)


@pytest.mark.parametrize("half_width", [.23, .35])
@pytest.mark.parametrize("route_segment", [False, True])
def test_receipt_only_scan_turns_with_only_front_depth(half_width, route_segment):
    horizontal = HorizontalIntrinsics(640, 315, 337)
    vertical = (480, 420, 243)
    geometry = camera_square(horizontal, vertical, half_width)
    tangent = (np.arange(480) - 243) / 420
    floor = np.minimum(.42 / np.maximum(tangent - .02, .001), 4.0)
    image = np.broadcast_to(floor[:, None], (480, 640)).copy()
    look = scan.ActiveScan()
    ground = ground_mask(image, horizontal, vertical)
    assert ground is not None

    for stamp in (1000, 1050, 1150):
        look.update({"map": image, "ground": ground, "frame_id": stamp},
                    {**_odom(stamp), "received_ms": stamp}, geometry, receipt_only=True)
        if stamp == 1000:
            assert look.begin((1.,), advance_m=1., route_segment=route_segment)
        result = look.command(geometry, half_width + .05, .18, 600)
    assert result.yaw == -.18


@pytest.mark.parametrize("initial_hit", [False, True])
def test_receipt_only_sweep_uses_measured_cells_and_blocks_an_initial_hit(monkeypatch, initial_hit):
    look, geometry, _, occupied = _setup(monkeypatch)
    occupied[scan.START[0], scan.START[1] + 5] = initial_hit  # Hits in the 5 cm side margin count.
    for stamp in (1000, 1050, 1150):
        look.update(_depth(stamp), {**_odom(stamp), "received_ms": stamp},
                    geometry, receipt_only=True)
        if stamp == 1000:
            assert look.begin((1.,))
        result = look.command(geometry, .25, .18, 600)
    assert (result.yaw < 0) is (not initial_hit)


@pytest.mark.parametrize("floor", [0., .5])
def test_scan_uses_the_fastest_executable_rate_supported_by_the_sweep(monkeypatch, floor):
    look, geometry, _, occupied = _setup(monkeypatch)
    occupied[scan.START[0] + 3, scan.START[1] - 9] = True
    for stamp in (1000, 1050, 1150):
        look.update(_depth(stamp), {**_odom(stamp), "received_ms": stamp},
                    geometry, receipt_only=True)
        if stamp == 1000:
            assert look.begin((1.,))
        result = look.command(geometry, .33, 1., 600, min_yaw_rate=floor)
    if floor:
        assert result.yaw == 0.
    else:
        assert -.5 < result.yaw < -.2


def test_sweep_stops_for_an_occupied_cell_that_straddles_the_footprint(monkeypatch):
    look, _, _, occupied = _setup(monkeypatch)
    geometry = Geometry(None, None, np.eye(4), .22, .23, None)
    # This cell is centred at forward .20 m, left .30 m. It extends .005 m
    # past the front edge and the right turn sweeps through the exposed part.
    occupied[14, 22] = True
    _tick(look, geometry, 1000)
    assert look.begin((1.,))
    _tick(look, geometry, 1050)
    assert _tick(look, geometry, 1150).yaw == 0


def test_receipt_only_scan_rejects_a_republished_depth_frame(monkeypatch):
    look, geometry, _, _ = _setup(monkeypatch)
    view = {**_depth(1000), "source_frame_id": 80_000}
    look.update(view, {**_odom(1000), "received_ms": 1000}, geometry, receipt_only=True)
    assert look.begin((1., -1.))
    for stamp in (1050, 1150, 1250, 1350, 1450):
        look.update({**view, "frame_id": stamp},
                    {**_odom(stamp), "received_ms": stamp}, geometry, receipt_only=True)
        look.command(geometry, .33, .18, 600)
    assert look.current_frame == 1000
    look.update({**view, "frame_id": 1550},
                {**_odom(1550), "received_ms": 1550}, geometry, receipt_only=True)
    assert look.current_frame == 0
    assert not look.active and look.locked


def test_observed_turn_can_use_the_current_depth_after_turning(monkeypatch):
    look, geometry, _, _ = _setup(monkeypatch)

    def tick(stamp, source):
        look.update(dict(_depth(stamp), source_frame_id=source),
                    dict(_odom(stamp), received_ms=stamp), geometry, receipt_only=True)
        return look.command(geometry, .25, 1., 600)

    tick(1000, 10)
    assert look.begin((-1.,), advance_m=1.)
    assert tick(1100, 11).yaw == 0.
    look.pose = (0., 0., -look.turn_angle)
    assert tick(1200, 11).yaw == 0.
    assert look.ready
    assert look.side_frame == 1100


def test_receipt_stamp_cannot_extend_the_life_of_repeated_odometry(monkeypatch):
    look, geometry, _, _ = _setup(monkeypatch)

    def update(stamp, source):
        look.update(_depth(stamp), {**_odom(stamp), "received_ms": stamp,
                                    "source_sample_id": source}, geometry, receipt_only=True)

    update(1000, 1)
    assert look.begin((1., -1.))
    update(1050, 2)
    assert look.command(geometry, .33, .18, 600).yaw == 0
    update(1150, 3)
    assert look.command(geometry, .33, .18, 600).yaw < 0
    update(1250, 3)
    update(1350, 3)
    update(1450, 3)
    assert not look.active and look.locked


def test_obstacle_evidence_survives_unknown_then_clears_only_on_measured_free_cells():
    origin = PlanarPose(1000, ('room',), 0., 0., 0.)
    moved = PlanarPose(1200, ('room',), .50, 0., 0.)
    memory = scan.ObservedObstacles()
    occupied = np.zeros(scan.SHAPE, dtype=bool)
    occupied[scan.START[0] + 30, scan.START[1]] = True
    memory.remember(origin, occupied)
    reprojected = memory.project(moved)
    assert reprojected[scan.START[0] + 20, scan.START[1]]
    assert memory.project(PlanarPose(1250, ('room',), .50, 0., math.pi / 2))[
        scan.START[0], scan.START[1] + 20]
    assert memory.project(PlanarPose(1300, ('new',), 0., 0., 0.)) is None
    assert memory.project(moved, observed_free=np.zeros_like(occupied)).any()
    assert not memory.project(moved, observed_free=reprojected.copy()).any()
    assert not memory.cells
