"""Perception-only tests: no legacy ActuCore checkout, model download or robot."""
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from vision_stubs import PERCEPTION_ROOT, _FakeExecutor, _FakeNode, frame_bytes


@pytest.fixture
def bundle(monkeypatch):
    from main import PerceptionBundle
    cfg = yaml.safe_load((PERCEPTION_ROOT / 'config.yaml').read_text())['plugins']
    settings = {k: {'enabled': False} for k in cfg}
    settings['hand_landmarks'] = cfg['hand_landmarks']
    assert settings['hand_landmarks']['enabled']
    monkeypatch.setattr(sys.modules['sensor_msgs.msg'], 'Image', type('Image', (), {}), raising=False)
    monkeypatch.setattr(_FakeNode, 'get_topic_names_and_types', lambda n: [('/camera', ['sensor_msgs/msg/CompressedImage'])], raising=False)
    monkeypatch.setattr(_FakeNode, 'create_timer', lambda n, interval, cb: setattr(n, 'tick', cb), raising=False)
    result = PerceptionBundle({'plugins': settings}, _FakeExecutor())
    yield result
    result.shutdown()


@pytest.mark.parametrize('pose,expected', [
    ('open', [0., .0037, 0., .0046, .008]),
    ('fist', [.9645, .907, .907, .907, .8771]),
])
def test_camera_to_driver_wire_contract(bundle, pose, expected):
    from sensor_msgs.msg import CompressedImage
    [tool] = bundle.get_all_tools()
    name = tool['name']
    assert name == 'hand_landmarks' and bundle.owns(name)
    def call(action, **args):
        return bundle.dispatch(name, {'action': action, **args})
    call('config', backend='mock', mock_pose=pose)
    assert call('start', input_topic='/camera')['state'] == 'running'
    node = bundle._plugins[0].node
    msg = CompressedImage(frame_bytes())
    msg.header.stamp = SimpleNamespace(sec=123, nanosec=456000000)
    node.subscriptions[0].callback(msg)
    node.tick()
    view, points = node.publishers
    assert view.topic == '/actucore/handdrive/view' and view.messages[-1]
    assert points.topic == '/actucore/handdrive/keypoints'
    sample = json.loads(points.messages[-1])
    assert sample['schema'] == 'motus.hand/1' and sample['obs_stamp_ms'] == 123456
    assert np.shape(sample['landmarks']) == (21, 3)
    assert sample['handedness'] == 'Right'
    assert [sample['flexions'][f] for f in ('thumb','index','middle','ring','pinky')] == pytest.approx(expected)
    call('pause'); node.tick()
    call('resume'); node.tick()
    assert len(points.messages) == 1  # resume must not replay the old frame
    assert call('stop')['state'] == 'idle' and node.destroyed


def test_rtmpose_imports_from_perception_and_declares_left(monkeypatch):
    from plugins import hand_runtime
    from plugins.hand_landmarks.vision import create_backend
    from utils import model_downloader
    monkeypatch.setattr(model_downloader, 'ensure_hand_model', lambda *a, **kw: {'x.engine': 'x.engine'})
    monkeypatch.setattr(hand_runtime, 'HandEngine', lambda path: object())
    monkeypatch.setattr(hand_runtime, 'assert_simcc', lambda session: None)
    xy = np.array([[10, 20, .9]] * 21)
    monkeypatch.setattr(hand_runtime, 'hands_in_frame', lambda *a, **kw: [(xy, .9)])
    model = create_backend('rtmpose', {'hand': 'Left', 'detection_hold_frames': 0})
    [detection] = model.detect(np.zeros((101, 101, 3)))
    assert detection.handedness == 'Left' and detection.handedness_source == 'configured'
    assert detection.landmarks[0].tolist() == pytest.approx([.1, .2, 0])
    monkeypatch.setattr(hand_runtime, 'hands_in_frame', lambda *a, **kw: [])
    assert model.detect(np.zeros((101, 101, 3))) == []


def test_output_uses_control_hand_flex_format_not_data_json():
    """The keypoint port must be a control-style stream, not a raw JSON blob."""
    from plugins.hand_landmarks.plugin import HandLandmarksPlugin, FLEX_FORMAT
    assert FLEX_FORMAT == "control/hand_flex"
    p = HandLandmarksPlugin({'hand': 'Right'}, None)
    outs = p._outputs()
    kf = [o for o in outs if o['format'] == FLEX_FORMAT]
    assert kf, f"no {FLEX_FORMAT} output: {outs}"
    assert not any(o['format'] == 'data/json' for o in outs)
    # control-style fields present in the wire payload
    tool = p.get_tools()[0]
    assert tool['topic_out'][1]['format'] == FLEX_FORMAT


def test_documented_canvas_port_contract_matches_topic_out():
    """The migration doc must name the same format the card advertises.

    A format change that leaves the doc behind makes the documented canvas
    wiring impossible (the operator connects a port that does not exist).
    """
    from pathlib import Path
    from plugins.hand_landmarks.plugin import HandLandmarksPlugin, FLEX_FORMAT

    doc = (Path(__file__).resolve().parents[1] / 'docs' / 'hand-landmarks.md').read_text()
    assert FLEX_FORMAT in doc, "migration doc must name the flexion port format"
    assert 'data/json' not in doc, "doc still advertises a port the card no longer has"
    p = HandLandmarksPlugin({'hand': 'Right'}, None)
    outs = p.get_tools()[0]['topic_out']
    assert any(o['format'] == FLEX_FORMAT for o in outs)
    assert not any(o['format'] == 'data/json' for o in outs)
