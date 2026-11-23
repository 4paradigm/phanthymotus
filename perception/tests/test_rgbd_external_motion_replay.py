import json
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest

from plugins import external_odometry_fusion, rgbd_external_input
from test_external_odometry_fusion import attitude_packet, packet
from test_rgbd_mount_replay import mount, native_pose, streams  # noqa: F401
from test_rgbd_odometry_status import sessions  # noqa: F401


@pytest.fixture
def replay_stream(sessions, monkeypatch):
    _, now, _ = sessions
    tick = [0.]
    monkeypatch.setattr(time, 'monotonic', lambda: tick[0])
    streams = []

    def create(automatic=False):
        calibration = dict(frame_id='body', external_candidates=['/external', '/other'])
        if not automatic:
            calibration['external_topic'] = '/external'
        stream = rgbd_external_input.ExternalOdometryInput('/fused/state/odom', calibration, learning=False)
        streams.append(stream)
        return stream

    yield create, now, tick
    for stream in streams:
        stream.close()


def receive(stream, item, source='/external'):
    sub = next(s for s in stream.node.subscriptions if s.topic == source)
    sub.callback(NS(data=json.dumps(item)))


def flush(stream):
    stream._apply_ready(stream._take_ready())


def turn(capture):
    item = attitude_packet(capture-1., 0., (capture-7.)*.4, .4)
    item['pose']['process_covariance'] = (np.eye(6)*(.02*(capture-7.))).ravel().tolist()
    return item


@pytest.mark.parametrize('delay', [0., .35])
@pytest.mark.parametrize('replacement', [False, True])
def test_historical_pose_completes_turn_and_replays_process_noise(replay_stream, delay, replacement):
    create, now, tick = replay_stream
    stream = create()
    for capture in ((7., 8., 9., 10.) if replacement else (7., 9., 10.)):
        now[0] = round(capture*1e9)
        item = turn(capture)
        if capture == 8.:
            del item['pose']['process_covariance']
        receive(stream, item)
        flush(stream)

    published = list(stream.publisher.messages)
    receive(stream, turn(8.))
    now[0] += round(delay*1e9)
    tick[0] += delay
    flush(stream)
    assert stream.publisher.messages == published

    now[0] = 10_500_000_000
    receive(stream, turn(10.5))
    flush(stream)
    samples = [json.loads(message) for message in stream.publisher.messages]
    stamps = [sample['vendor']['motus_slam']['stamp_ns'] for sample in samples]
    assert stamps == ([8_000_000_000, 9_000_000_000] if replacement else []) + [
        10_000_000_000, 10_500_000_000]
    q = samples[-1]['pose']['quaternion']
    assert 2*np.arctan2(q[2], q[3]) == pytest.approx(1.4)
    np.testing.assert_allclose(np.reshape(samples[-1]['pose']['covariance'], (6, 6)),
                               np.eye(6)*1.07, atol=1e-12)


@pytest.mark.parametrize('old_first', [True, False])
def test_queued_history_can_follow_fresh_frontier_in_either_ingress_order(replay_stream, old_first):
    create, now, tick = replay_stream
    stream = create()
    now[0] = 7_000_000_000
    receive(stream, turn(7.))
    flush(stream)
    now[0] = 10_400_000_000
    for capture in ((8., 10., 9.) if old_first else (10., 9., 8.)):
        receive(stream, turn(capture))
    tick[0] = .35
    now[0] += 350_000_000
    flush(stream)
    samples = [json.loads(message) for message in stream.publisher.messages]
    sample, = samples
    assert sample['vendor']['motus_slam']['stamp_ns'] == 10_000_000_000
    assert sample['pose']['quaternion'][2] == pytest.approx(np.sin(.6))


@pytest.mark.parametrize('fresh_pair', [True, False])
def test_only_fresh_pair_selects_source_before_applying_queued_history(replay_stream, fresh_pair):
    create, now, tick = replay_stream
    stream = create(automatic=True)
    now[0] = 8_000_000_000
    receive(stream, turn(8.))
    receive(stream, turn(8.25), source='/other')
    now[0] = 10_000_000_000
    receive(stream, turn(7.5))  # Already stale at receipt on the unselected wire.
    if fresh_pair:
        receive(stream, turn(9.))
    receive(stream, turn(10.))
    tick[0] += .35
    flush(stream)
    if fresh_pair:
        assert stream.selected == '/external'
        assert sorted(stream.fuser.raw_motion) == [8_000_000_000, 9_000_000_000, 10_000_000_000]
        samples = [json.loads(message) for message in stream.publisher.messages]
        assert [s['vendor']['motus_slam']['stamp_ns'] for s in samples] == [9_000_000_000, 10_000_000_000]
        assert samples[-1]['pose']['quaternion'][2] == pytest.approx(np.sin(.4))
    else:
        assert stream.selected is None
        assert not stream.fuser.raw_motion
        assert not stream.publisher.messages
        assert set(stream.candidate_history['/external']) == {10_000_000_000}


@pytest.mark.parametrize('leading', ['visual', 'invalid_visual', 'stale_visual', 'future_motion', 'invalid_motion'])
def test_queued_pose_needs_an_accepted_frontier(replay_stream, leading):
    create, now, tick = replay_stream
    stream = create()
    now[0] = 10_000_000_000
    receive(stream, turn(8.))
    if leading == 'invalid_motion':
        item = turn(10.)
        item['pose']['quaternion'] = [0.]*4
        receive(stream, item)
    elif leading == 'future_motion':
        receive(stream, turn(20.))
    else:
        stamp = 8_400_000_000 if leading == 'stale_visual' else now[0]
        stream.visual(stamp, 'native', dict(position=[0.]*3,
                      quaternion=[0., 0., 0., 0. if leading == 'invalid_visual' else 1.]))
    tick[0] += .35
    flush(stream)
    if leading == 'visual':
        assert set(stream.fuser.raw_motion) == {8_000_000_000}
        sample, = stream.publisher.messages
        assert json.loads(sample)['vendor']['motus_slam']['stamp_ns'] == now[0]
    else:
        assert not stream.fuser.raw_motion
        assert not stream.publisher.messages


def test_old_eviction_cut_applies_before_queued_history_and_survives_the_batch(replay_stream, monkeypatch):
    create, now, tick = replay_stream
    monkeypatch.setattr(rgbd_external_input, '_MAX_PENDING_EVENTS', 2)
    stream = create()
    now[0] = 10_000_000_000
    receive(stream, packet(7., twist=[1.]*6))  # Evicted 8 s sample cuts the lost interval.
    receive(stream, turn(7.5))
    receive(stream, turn(10.))
    tick[0] += .35
    flush(stream)
    receive(stream, turn(7.75))
    flush(stream)
    assert sorted(stream.fuser.raw_motion) == [8_000_000_000, 10_000_000_000]
    assert stream.fuser.raw_motion[8_000_000_000].reset
    assert not stream.publisher.messages


def test_fresh_frontier_excludes_motion_beyond_its_history_window(replay_stream):
    create, now, _ = replay_stream
    stream = create()
    now[0] = 21_000_000_000
    receive(stream, turn(10.))
    receive(stream, turn(21.))
    assert stream.events.qsize() == 2
    flush(stream)
    assert set(stream.fuser.raw_motion) == {21_000_000_000}
    assert stream.dropped == 1
    assert not stream.publisher.messages


def test_late_pose_cannot_replace_a_frozen_checkpoint(replay_stream, monkeypatch):
    create, now, _ = replay_stream
    monkeypatch.setattr(external_odometry_fusion, '_MAX_EVENTS', 3)
    stream = create()
    for capture in (7., 8., 9., 10.):
        now[0] = round(capture*1e9)
        receive(stream, turn(capture))
        flush(stream)
    assert stream.fuser.base.stamp == 8_000_000_000
    original = stream.fuser.raw_motion[stream.fuser.base.stamp]
    published = list(stream.publisher.messages)
    receive(stream, attitude_packet(7., 500., 1.8, .4))
    assert stream.events.empty()
    assert stream.dropped == 1
    assert stream.fuser.raw_motion[stream.fuser.base.stamp] is original
    assert stream.publisher.messages == published


@pytest.mark.parametrize('split_drain', [False, True])
@pytest.mark.parametrize('earlier_visual', [False, True])
def test_motion_replay_uses_only_surviving_visual_frontier(streams, split_drain, earlier_visual):
    create, now, tick = streams
    stream = create()
    if earlier_visual:
        now[0] = 8_250_000_000
        mount(stream, now[0], 0.)
        stream.visual(now[0], 'native', native_pose(0.))
        flush(stream)

    now[0] = 10_000_000_000
    mount(stream, now[0], 0.)
    stream.visual(now[0], 'native', native_pose(0.))
    flush(stream)
    assert stream.fuser.current.stamp == now[0]
    mount(stream, now[0], None)
    if split_drain:
        flush(stream)
    receive(stream, packet(7., 100.))
    now[0] += 350_000_000
    tick[0] += .35
    flush(stream)

    assert 10_000_000_000 not in stream.fuser.raw_vision
    assert not stream.publisher.messages
    receive(stream, packet(8.3, 101.3))
    flush(stream)
    if earlier_visual:
        assert set(stream.fuser.raw_motion) == {8_000_000_000, 9_300_000_000}
        sample, = stream.publisher.messages
        assert json.loads(sample)['pose']['position'][0] == pytest.approx(1.3)
    else:
        assert set(stream.fuser.raw_motion) == {9_300_000_000}
        assert stream.dropped == 1
        assert not stream.publisher.messages


@pytest.mark.parametrize('split_drain', [False, True])
def test_withdrawn_visual_cannot_authorize_an_eviction_cut(streams, monkeypatch, split_drain):
    create, now, _ = streams
    monkeypatch.setattr(rgbd_external_input, '_MAX_PENDING_EVENTS', 2)
    stream = create()
    now[0] = 10_000_000_000
    mount(stream, now[0], 0.)
    stream.visual(now[0], 'native', native_pose(0.))
    flush(stream)
    mount(stream, now[0], None)
    if split_drain:
        flush(stream)
    receive(stream, packet(7., twist=[1.]*6))
    for stamp in (6_000_000_000, 7_000_000_000):
        mount(stream, stamp, None)
    now[0] += 350_000_000
    flush(stream)
    assert not stream.fuser.raw_vision
    assert not stream.fuser.raw_motion

    # The newly established timeline can still accept a pose preceding the
    # ineligible 8 s eviction. A spurious cutoff would block this bracket.
    receive(stream, packet(8.3, 101.5))
    flush(stream)
    receive(stream, packet(6.8, 100.))
    flush(stream)
    sample, = stream.publisher.messages
    assert json.loads(sample)['pose']['position'][0] == pytest.approx(1.5)


@pytest.mark.parametrize('split_drain', [False, True])
def test_first_seen_stale_visual_cannot_establish_motion_history(replay_stream, split_drain):
    create, now, _ = replay_stream
    stream = create()
    now[0] = 10_000_000_000
    stream.visual(8_400_000_000, 'native', native_pose(0.))
    if split_drain:
        flush(stream)
    receive(stream, turn(8.))
    flush(stream)
    receive(stream, turn(9.3))
    flush(stream)
    assert set(stream.fuser.raw_motion) == {9_300_000_000}
    assert not stream.publisher.messages
    receive(stream, turn(9.5))
    flush(stream)
    sample, = stream.publisher.messages
    assert json.loads(sample)['pose']['quaternion'][2] == pytest.approx(np.sin(.04))


@pytest.mark.parametrize('projected_live', [False, True])
def test_restoring_aged_visual_has_authority_only_if_projected_live(streams, projected_live):
    create, now, _ = streams
    stream = create()
    capture = now[0] = 8_400_000_000
    if projected_live:
        mount(stream, capture, 0.)
    stream.visual(capture, 'native', native_pose(0.))
    flush(stream)
    now[0] = 10_000_000_000
    mount(stream, capture, None)
    receive(stream, turn(8.))
    flush(stream)
    assert not stream.fuser.raw_motion
    mount(stream, capture, 0.)
    flush(stream)
    assert capture in stream.fuser.raw_vision
    receive(stream, turn(8.))
    flush(stream)
    assert set(stream.fuser.raw_motion) == ({8_000_000_000} if projected_live else set())


def test_stale_replacement_cannot_inherit_another_visual_owners_authority(replay_stream):
    create, now, _ = replay_stream
    stream = create()
    capture = now[0] = 8_400_000_000
    stream.visual(capture, 'first', native_pose(0.))
    flush(stream)
    now[0] = 10_000_000_000
    stream.visual(capture, 'replacement', native_pose(0.))
    receive(stream, turn(8.))
    flush(stream)
    assert stream.fuser.raw_vision[capture][0] == 'replacement'
    assert not stream.fuser.raw_motion


@pytest.mark.parametrize('authority', ['stale_only', 'frozen_visual', 'retained_visual', 'motion'])
def test_evicted_cut_after_pruning_requires_a_surviving_frontier(replay_stream, monkeypatch, authority):
    create, now, _ = replay_stream
    monkeypatch.setattr(external_odometry_fusion, '_MAX_EVENTS', 4)
    monkeypatch.setattr(rgbd_external_input, '_MAX_PENDING_EVENTS', 2)
    stream = create()
    if authority == 'motion':
        now[0] = 8_200_000_000
        receive(stream, turn(8.2))
        flush(stream)
    elif authority in ('frozen_visual', 'retained_visual'):
        now[0] = 7_000_000_000 if authority == 'frozen_visual' else 8_400_000_000
        stream.visual(now[0], 'native', native_pose(0.))
        flush(stream)
    now[0] = 10_000_000_000
    for capture in (7., 7.2, 7.4, 7.6, 8.4):
        stream.visual(round(capture*1e9), 'native', native_pose(0.))
        flush(stream)
    assert 7_000_000_000 <= stream.fuser.base.stamp < 8_000_000_000
    assert 8_400_000_000 in stream.fuser.raw_vision
    receive(stream, packet(7., twist=[1.]*6))  # The 8 s cut will be evicted.
    for capture in (7.7, 7.8):
        stream.visual(round(capture*1e9), 'native', native_pose(0.))
    flush(stream)
    if authority in ('retained_visual', 'motion'):
        assert stream.fuser.raw_motion[8_000_000_000].reset
        assert stream.motion_cutoffs['/external'] == 8_000_000_000
    else:
        assert not stream.fuser.raw_motion
        assert '/external' not in stream.motion_cutoffs
