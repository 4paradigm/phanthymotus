"""Wiring selects IMU assistance, including saved legacy settings."""
import pytest
from test_rgbd_start_inputs import card, start
from test_rgbd_odometry_contract import declarations


@pytest.mark.parametrize('legacy', [False, True, 'false'])
@pytest.mark.parametrize('imu', [False, True])
@pytest.mark.parametrize('fusion', [False, True])
def test_wiring_overrides_legacy_flags_and_gates_fusion(card, legacy, imu, fusion):
    result = card.dispatch('rgbd_odometry', dict(action='config', instance_id='one',
        imu_assisted=legacy, short_term_compensation=fusion, depth_geometry=fusion))
    assert result['state'] == 'configured'
    topics = ['/depth', '/rgb'] + (['/motion'] if imu else [])
    result = start(card, input_topics=topics, camera_info=declarations(), imu_assisted=legacy)
    assert result['state'] == 'running'
    c = card.instances['one'][0].calibration
    assert c['imu_assisted'] is imu
    assert c['short_term_compensation'] is imu
    assert c['depth_geometry'] is imu
    assert c['inertial_association'] is imu
    assert c['max_features'] == 1000
    assert result['config']['visual_frontend'] == 'auto'
    assert ('imu_topic' in c) is imu
    assert result['config']['imu_assisted'] is imu
    assert len(result['topic_in']) == len(topics)
    assert card.get_tools()[0]['configSchema']['properties'] == {}


def test_rewiring_requires_stop_and_preserves_other_instance(card):
    args = dict(camera_info=declarations(), input_topics=['/rgb','/depth','/motion'])
    assert start(card, **args)['state'] == 'running'
    original = card.instances['one'][0]
    assert start(card, camera_info=declarations(), input_topics=['/rgb','/depth'])['state'] == 'error'
    assert card.instances['one'][0] is original
    assert card.dispatch('rgbd_odometry', dict(action='start', instance_id='two', **args))['state'] == 'running'
    card.dispatch('rgbd_odometry', dict(action='stop', instance_id='one'))
    assert start(card, camera_info=declarations(), input_topics=['/rgb','/depth'])['state'] == 'running'
    assert card.instances['one'][0].calibration['imu_assisted'] is False
    assert card.instances['two'][0].calibration['imu_assisted'] is True


@pytest.mark.parametrize('inputs', [{}, {'input_topics': None}, {'input_topics': 3}])
def test_missing_or_malformed_inputs_return_error_before_session_creation(card, inputs):
    result = start(card, camera_info=declarations(), **inputs)
    assert result['state'] == 'error'
    assert not card.instances
