"""Start-input diagnostics: name the missing input, never invent calibration."""
from copy import deepcopy

import pytest

from plugins.rgbd_odometry import RgbdOdometryPlugin
from plugins.rgbd_odometry_contract import start_topics, validate_start_inputs
from test_rgbd_odometry import Session
from test_rgbd_odometry_contract import declarations
from vision_stubs import _FakeExecutor


@pytest.fixture
def card(monkeypatch):
    from plugins import rgbd_odometry as module

    class RecordingSession(Session):
        def __init__(self, calibration, executor, topic):
            super().__init__(calibration, executor, topic)
            self.calibration = calibration

    monkeypatch.setattr(module, 'RgbdSession', RecordingSession)
    monkeypatch.setattr(module, 'require_runtime', lambda: None)
    Session.created = []
    value = RgbdOdometryPlugin({}, _FakeExecutor())
    yield value
    value.shutdown()


def start(card, **args):
    return card.dispatch('rgbd_odometry', dict(action='start', instance_id='one', **args))


def test_singular_core_input_is_diagnosed_instead_of_type_error(card):
    """Core sends one resolved topic as `input_topic` only."""
    assert start_topics({'input_topic': '/depth'}) == ['/depth']
    assert start_topics({'input_topics': ['/rgb', '/depth'], 'input_topic': '/rgb'}) == ['/rgb', '/depth']
    result = start(card, input_topic='/depth', camera_info=declarations())
    assert result['state'] == 'error'
    assert '1 input(s) connected' in result['message'] and '/depth: declared image/depth-zlib' in result['message']
    assert not Session.created


def test_missing_declaration_names_topic_role_and_unconnected_declarations(card):
    values = declarations(); values['/other'] = values.pop('/rgb')
    result = start(card, input_topics=['/rgb', '/depth'], camera_info=values)
    message = result['message']
    assert message.startswith('connect one calibrated color and one measured depth input: missing image/jpeg')
    assert message.index('/rgb: no camera_info declaration') < message.index('/depth: declared')
    assert 'declared but not connected: /other' in message
    assert not Session.created


@pytest.mark.parametrize('topics,needle', [
    (['/rgb', '/rgb'], 'duplicate /rgb'),
    (['/rgb', '/depth', '/imu'], 'imu_assisted=false, 3 input(s) connected'),
])
def test_count_and_duplicate_failures_are_specific(card, topics, needle):
    result = start(card, input_topics=topics, camera_info=declarations())
    assert result['state'] == 'error' and needle in result['message']


@pytest.mark.parametrize('mutate,needle', [
    (lambda d: d['/depth'].update(format='image/jpeg'), 'missing image/depth-zlib'),
    (lambda d: d['/depth'].pop('format'), '/depth: declared without format'),
])
def test_wrong_or_absent_format_is_named(card, mutate, needle):
    values = declarations(); mutate(values)
    result = start(card, input_topics=['/rgb', '/depth'], camera_info=values)
    assert result['state'] == 'error' and needle in result['message']


@pytest.mark.parametrize('mutate', [
    lambda d: d['/rgb'].pop('K'),
    lambda d: d['/depth']['vendor'].pop('registration'),
    lambda d: d['/depth']['vendor'].update(depth_unit='m'),
    lambda d: d['/depth']['vendor'].pop('depth_unit'),
])
def test_diagnostics_do_not_substitute_intrinsics_registration_or_units(card, mutate):
    values = declarations(); mutate(values)
    assert start(card, input_topics=['/rgb', '/depth'], camera_info=values)['state'] == 'error'
    assert not Session.created


def test_refreshed_declarations_require_restart_and_are_then_adopted(card):
    assert start(card, input_topics=['/rgb', '/depth'], camera_info=declarations())['state'] == 'running'
    refreshed = deepcopy(declarations())
    for entry in refreshed.values():
        entry['K'] = [410, 0, 320, 0, 410, 240, 0, 0, 1]
    changed = start(card, input_topics=['/rgb', '/depth'], camera_info=refreshed)
    assert changed['state'] == 'error' and 'stop RGB-D odometry' in changed['message']
    card.dispatch('rgbd_odometry', dict(action='stop', instance_id='one'))
    assert start(card, input_topics=['/depth', '/rgb'], camera_info=refreshed)['state'] == 'running'
    assert card.instances['one'][0].calibration['K'][0, 0] == 410


def test_imu_start_reports_missing_imu_link(card):
    card.dispatch('rgbd_odometry', dict(action='config', instance_id='one', imu_assisted=True))
    result = start(card, input_topics=['/rgb', '/depth'], camera_info=declarations())
    assert 'imu_assisted=true, 2 input(s) connected' in result['message']
    ok = validate_start_inputs(declarations(), ['/imu', '/depth', '/rgb'], True)
    assert ok['imu_topic'] == '/imu'
