from __future__ import annotations

import argparse
import json
import pathlib
import sys
import uuid


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('fixture', type=pathlib.Path)
    parser.add_argument('--actucore-plugins', type=pathlib.Path)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    frame = json.loads(arguments.fixture.read_text(encoding='utf-8'))
    assert set(frame) == {
        'schema_version',
        'sequence',
        'client_monotonic_ns',
        'mode',
        'deadman',
        'clutch_sequence',
        'tracking',
        'head',
        'left_controller',
        'right_controller',
        'controllers',
    }
    assert frame['schema_version'] == 1
    assert frame['mode'] == 'live'
    assert frame['deadman'] is True
    assert frame['clutch_sequence'] == 1
    assert frame['tracking'] == {
        'head': True,
        'left_controller': True,
        'right_controller': True,
    }
    assert frame['controllers']['left'] == {
        'axes': [0, 0, 0, 0],
        'buttons': [0, 1],
    }
    assert frame['controllers']['right'] == frame['controllers']['left']

    if arguments.actucore_plugins is None:
        print('openxr-capture fixture: public JSON contract passed')
        return

    plugins_root = arguments.actucore_plugins.resolve()
    if not (plugins_root / 'teleop' / 'protocol.py').is_file():
        raise SystemExit(f'ActuCore root is invalid: {plugins_root}')
    sys.path.insert(0, str(plugins_root))
    from teleop.protocol import bind_rtc_frame_v1  # noqa: PLC0415

    session_id = str(uuid.UUID('d097eb8f-b386-455f-9e2b-23f1ad6a1ee3'))
    bound = bind_rtc_frame_v1(
        frame,
        authority={
            'boot_id': str(uuid.UUID('6e6973a9-5b32-4d10-b1b8-c0331800a4aa')),
            'session_id': session_id,
            'epoch': 1,
            'fence': 'f' * 32,
        },
        expected_mode='live',
    )
    assert bound['session_id'] == session_id
    assert bound['deadman'] is True
    assert bound['clutch_sequence'] == 1
    print('openxr-capture fixture: ActuCore strict contract passed')


if __name__ == '__main__':
    main()
