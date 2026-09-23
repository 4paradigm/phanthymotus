"""Check provisioned site files before advertising the optional teleop card.

This is deployment readiness, not model calibration or permission to move. It
opens no network/ROS endpoints and never creates credentials or calibration.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET


def required_site_config(config: dict) -> list[dict[str, str]]:
    issues = []

    def missing(field, code):
        # Never return exception messages: they may contain a secret or a site
        # filename. These diagnostics are exposed through unauthenticated MCP.
        issues.append({'field': field, 'code': code})

    def configured_path(value):
        if not isinstance(value, str) or not value or not Path(value).is_absolute():
            raise ValueError('absolute_path_required')
        return Path(value)

    # Core separately checks its TELEOP_MANAGEMENT_URL before forwarding this
    # capability. The server does not need (and cannot verify) that client env.
    try:
        path = configured_path(os.environ.get('TELEOP_MANAGEMENT_KEY_FILE'))
        if path.is_symlink() or not path.is_file():
            raise ValueError('regular_file_required')
        key = path.read_text().strip()
        if len(key) < 32 or not all('!' <= ch <= '~' for ch in key):
            raise ValueError('invalid_key')
    except (OSError, ValueError, UnicodeError):
        missing('TELEOP_MANAGEMENT_KEY_FILE', 'readable_ascii_key_required')

    capture = config.get('capture')
    if not isinstance(capture, dict):
        capture = {}
    tls_fields = ('public_wss_url', 'tls_cert_file', 'tls_key_file')
    complete_tls = True
    for field in tls_fields:
        if not isinstance(capture.get(field), str) or not capture[field]:
            missing('capture.' + field, 'required')
            complete_tls = False
    if complete_tls:
        try:
            from .capture_server import build_capture_ssl_context
            build_capture_ssl_context(capture)
        except ImportError:
            missing('runtime', 'teleop_dependencies_unavailable')
        except (OSError, ValueError, RuntimeError, TypeError):
            missing('capture.tls', 'invalid_certificate_key_or_public_url')

    try:
        state = configured_path(capture.get('state_file'))
        if state.is_symlink() or not state.parent.is_dir():
            raise ValueError('state_directory_required')
        # This verifies the actual container mount is writable (including a
        # read-only mount under uid 0) without touching saved pairing state.
        with tempfile.TemporaryFile(dir=state.parent):
            pass
        if state.exists():
            from .capture import CaptureManager
            # Construction only reads persisted state; it creates no tasks,
            # sockets or tickets. Reuse the real parser instead of weakening it.
            CaptureManager(None, None, None, state_file=state)
    except ImportError:
        if not any(item['field'] == 'runtime' for item in issues):
            missing('runtime', 'teleop_dependencies_unavailable')
    except (OSError, ValueError, TypeError, RuntimeError):
        missing('capture.state_file', 'writable_directory_and_valid_state_required')

    remote = config.get('control_backend') == 'motion_control' and config.get('robot_profile', 'tianyi2') != 'g1_23'
    if not remote:
        try:
            calibration = configured_path(config.get('calibration_path'))
            profile = json.loads(calibration.read_text())
            robot = config.get('robot_profile', 'tianyi2')
            expected = {'tianyi2': 'motus.tianyi-calibration.v1',
                        'g1_23': 'motus.g1-calibration.v1'}
            if robot not in expected or not isinstance(profile, dict) or profile.get('schema') != expected[robot]:
                raise ValueError('calibration_schema')
            model = Path(profile['urdf_path'])
            # Match the profile readers: only G1 resolves a relative model path
            # against the calibration file. Tianyi uses the process working dir.
            if robot == 'g1_23' and not model.is_absolute():
                model = calibration.parent / model
            model_bytes = model.read_bytes()
            if hashlib.sha256(model_bytes).hexdigest() != profile['urdf_sha256']:
                raise ValueError('calibration_model_changed')
            if ET.fromstring(model_bytes).tag != 'robot':
                raise ValueError('urdf_required')
        except (OSError, ValueError, TypeError, KeyError, ET.ParseError):
            missing('calibration_path', 'readable_profile_and_matching_model_required')
    if not issues:
        try:
            dependencies = ['numpy', 'scipy.spatial.transform', 'aiortc'] if remote else ['pinocchio', 'scipy.optimize', 'aiortc']
            if capture.get('discovery_enabled', True):
                dependencies.append('zeroconf')
            if config.get('robot_profile') == 'g1_23':
                dependencies.extend(('casadi', 'pinocchio.casadi'))
            for module in dependencies:
                importlib.import_module(module)
        except (ImportError, OSError):
            missing('runtime', 'teleop_dependencies_unavailable')
    return issues
