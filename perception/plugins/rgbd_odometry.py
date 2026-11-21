"""Generic RGB-D camera-motion card; robot geometry belongs to the producer."""
import json
import logging
import re
import threading
import time
from copy import deepcopy

from .rgbd_odometry_contract import start_topics, validate_start_inputs
from .rgbd_association import validate_controller_rotation
from .rgbd_odometry_runtime import RgbdSession, require_runtime
from .rgbd_recovery import ASSISTANCE_RETRY_S, TRACKING_RESET_S, assistance_mode
from utils.ros_lifecycle import dispose_node


_INTERFACE = {'schema': 'motus.odom/1', 'slam_pose': {'available': True, 'profile': 'motus_slam/1',
    'supported_profiles': ['motus_slam/1', 'motus_slam/2'],
    'time_bases': ['source_capture', 'capture_estimated', 'receipt_estimated']}}
_FRONTENDS = ('f2m_feature', 'f2f_optical_flow')
# Engineering limits, not tuned on hardware. A switch needs a
# visual-kind loss with fresh inputs, then waits out a cooldown; after two
# switches without tracking in between, ordinary recovery keeps the frontend.
_AUTO_COOLDOWN_S, _AUTO_MAX_UNTRACKED_SWITCHES = 20., 2


class RgbdOdometryPlugin:
    PREFIX = 'rgbd_odometry'

    def __init__(self, config, executor, *, input_registry=None):
        self._input_registry = input_registry
        self.executor = executor
        self.instances = {}
        self.lock = threading.RLock()
        self.recovery = {}
        self.frontends = {}
        self.external_streams = {}
        self.controller_profiles = {}
        self.retiring = {}
        self.recovery_node = self.recovery_timer = None
        self.closed = False

    def _ensure_supervisor(self):
        if self.recovery_node is None:
            from rclpy.node import Node
            self.recovery_node = Node('rgbd_odometry_recovery')
            self.recovery_timer = self.recovery_node.create_timer(.2, self._recover)
            self.executor.add_node(self.recovery_node)

    def _recover(self):
        # This node belongs to the bundle executor, never the session's IMU
        # executor: closing a retired session must not join its own callback.
        with self.lock:
            if self.closed:
                return
            now = time.monotonic()
            for instance, current in list(self.instances.items()):
                if instance in self.retiring:
                    continue  # explicit stop must finish incomplete worker cleanup
                session, signature, bindings = current
                state = self.recovery[instance]
                candidate = None
                try:
                    info = session.info()
                    target = (session.recovery_target(now, time.time_ns())
                              if hasattr(session, 'recovery_target') else None)
                    if info['status'] == 'tracking':
                        if state['healthy_since'] is None:
                            state['healthy_since'] = now
                        if now-state['healthy_since'] >= TRACKING_RESET_S:
                            state.update(failure_streak=0, next_attempt=None, backoff_s=1.5)
                        if instance in self.frontends:
                            self.frontends[instance]['untracked'] = 0
                    else:
                        state['healthy_since'] = None
                    restoring = target is not None and target['action'] == 'restore_assistance'
                    if restoring and info['status'] != 'lost' and session.first_tracking_at is not None:
                        # A replacement resets the odometry origin. Keep a
                        # published segment, including during bounded recovery.
                        restoring = False
                    if info['status'] != 'lost' and not restoring:
                        state['lost_since'] = None
                        continue
                    if info['status'] == 'lost' and state['lost_since'] is None:
                        state['lost_since'] = now
                        # A published run interrupted by missing camera input
                        # succeeded. Retry when both inputs return; a failed
                        # launch still earns its own backoff below.
                        if info['loss_kind'] == 'input' and session.first_tracking_at is not None:
                            state.update(failure_streak=0, next_attempt=None, backoff_s=1.5)
                    if ((not restoring and now-state['lost_since'] < .8) or
                            (state['next_attempt'] is not None and now < state['next_attempt']) or
                            (restoring and state['mode_retry_after'] is not None and now < state['mode_retry_after']) or
                            not session.has_fresh_inputs(time.time_ns())):
                        continue
                    state['last_attempt'] = now
                    state['attempts'] += 1
                    state['failure_streak'] += 1
                    state['backoff_s'] = min(30., 1.5*2**min(5, state['failure_streak']-1))
                    state['next_attempt'] = now+state['backoff_s']
                    if restoring:
                        state['mode_retry_after'] = now+ASSISTANCE_RETRY_S
                        state['restoration_attempts'] += 1
                        calibration, switch = target['calibration'], None
                    else:
                        calibration, switch = self._next_frontend(instance, session, info, now,
                            target['calibration'] if target is not None else None)
                    transition = dict(action=target['action'] if target else 'restart',
                        reason=target['reason'] if target else info['reason'], previous_segment=info['segment_id'],
                        from_mode=info.get('mode', {}).get('effective', assistance_mode(session.calibration)),
                        to_mode='visual' if calibration.get('_start_with_rotation_only') else assistance_mode(calibration))
                    requested = dict(getattr(session, 'requested_calibration', session.calibration),
                                     visual_frontend=calibration.get('visual_frontend', 'f2m_feature'))
                    candidate = RgbdSession(calibration, self.executor,
                                            self._outputs(instance)[0]['topic'], publish_output=False,
                                            requested_calibration=requested, mode_transition=transition,
                                            **({'input_registry': self._input_registry} if self._input_registry else {}))
                    if instance in self.external_streams:
                        candidate.external_stream = self.external_streams[instance]
                        if getattr(candidate, 'observations', None) is not None:
                            candidate.observations.visual_sink = candidate._observe_external_visual
                    # The old worker can finish its first tracking estimate
                    # while the candidate starts. Decide before it can publish again.
                    with session.lock:
                        keep_session = (session.info()['status'] != 'lost' and
                                        (not restoring or session.first_tracking_at is not None))
                        if not keep_session:
                            session.publish_output = False
                    if keep_session:
                        candidate.close(report_idle=False)
                        state['lost_since'] = None
                        continue
                    session.close(report_idle=False)
                    with candidate.lock:
                        self.instances[instance] = candidate, signature, bindings
                        candidate.publish_output = True
                    state.update(restarts=state['restarts']+1, previous_segment=info['segment_id'],
                                 reason=transition['reason'], error=None, lost_since=None, healthy_since=None)
                    state['transitions'].append(dict(transition, segment_id=candidate.segment))
                    del state['transitions'][:-12]
                    if switch:
                        frontend = self.frontends[instance]
                        frontend.update(active=switch, switches=frontend['switches']+1,
                                        untracked=frontend['untracked']+1, last_switch_reason=info['reason'],
                                        cooldown_until=now+_AUTO_COOLDOWN_S)
                except Exception as error:
                    reason = str(error)
                    if candidate is not None:
                        try:
                            candidate.close(report_idle=False)
                        except Exception as cleanup_error:
                            self.retiring[instance] = candidate
                            reason += '; candidate cleanup: ' + str(cleanup_error)
                    state['error'] = reason
                    logging.getLogger(__name__).warning('RGB-D recovery %s: %s', instance, reason)

    def _next_frontend(self, instance, session, info, now, calibration=None):
        """Choose the replacement's frontend; never run both, never splice origins."""
        frontend = self.frontends.get(instance)
        calibration = (getattr(session, 'restart_calibration', session.calibration)
                       if calibration is None else calibration)
        if not frontend:
            return calibration, None
        active, kind = frontend['active'], info.get('loss_kind')
        other = _FRONTENDS[1-_FRONTENDS.index(active)]
        native_lost = kind == 'relocalization' and session.gate.relocalization_native_loss_streak >= 3
        if kind != 'visual' and not native_lost:
            frontend['decision'] = f'kept {active}: loss kind {kind} is not visual degradation'
        elif frontend['cooldown_until'] is not None and now < frontend['cooldown_until']:
            frontend['decision'] = f'kept {active}: frontend switch cooldown'
        elif frontend['untracked'] >= _AUTO_MAX_UNTRACKED_SWITCHES:
            frontend['decision'] = (f'kept {active}: {_AUTO_MAX_UNTRACKED_SWITCHES} switches '
                                    'without tracking; switching suspended')
        else:
            frontend['decision'] = f'switched to {other}: visual degradation with fresh inputs'
            return dict(calibration, visual_frontend=other), other
        return calibration, None

    def _recovery_info(self, instance):
        state = self.recovery.get(instance)
        if state is None:
            return {}
        now = time.monotonic()
        internal = {'lost_since', 'last_attempt', 'healthy_since', 'next_attempt', 'mode_retry_after'}
        return dict({k: deepcopy(v) for k, v in state.items() if k not in internal},
                    retry_in_s=round(max(0., (state['next_attempt'] or now)-now), 2),
                    restoration_retry_in_s=round(max(0., (state['mode_retry_after'] or now)-now), 2))

    def _frontend_info(self, instance):
        frontend = self.frontends.get(instance)
        if not frontend:
            return None
        remaining = (0. if frontend['cooldown_until'] is None else
                     max(0., round(frontend['cooldown_until']-time.monotonic(), 1)))
        return dict(mode='auto', active=frontend['active'], switches=frontend['switches'],
                    last_switch_reason=frontend['last_switch_reason'], decision=frontend['decision'],
                    cooldown_remaining_s=remaining,
                    switching_suspended=frontend['untracked'] >= _AUTO_MAX_UNTRACKED_SWITCHES)

    def _stop_instance(self, instance):
        failures = []
        if instance in self.retiring:
            try:
                self.retiring[instance].close(report_idle=False)
                del self.retiring[instance]
            except Exception as error:
                failures.append('candidate cleanup: ' + str(error))
        if instance in self.external_streams:
            try:
                self.external_streams[instance].close()
                del self.external_streams[instance]
            except Exception as error:
                failures.append(str(error))
        if instance in self.instances:
            try:
                self.instances[instance][0].close()
                del self.instances[instance]
            except Exception as error:
                failures.append(str(error))
        if failures:
            raise RuntimeError('; '.join(failures))
        self.recovery.pop(instance, None)
        self.frontends.pop(instance, None)

    def get_tools(self):
        return [{'name': 'rgbd_odometry', 'type': 'processor', 'multiInstance': True,
                 'description': 'RGB-D 视觉里程计：彩色和实测深度估计相机位姿，接入相机 IMU 自动启用辅助',
                 'inputSchema': {'type': 'object', 'x-input-metadata-consumer': True, 'properties': {
                     'action': {'type': 'string', 'enum': ['start', 'stop', 'info']}}},
                 'configSchema': {'type': 'object', 'properties': {'controller_rotation': {
                     'type': ['object', 'null'], 'scope': 'instance', 'title': '控制器旋转先验',
                     'description': '旧版原始控制器话题的读时对齐与相机外参覆盖。只用于视觉配准初值。',
                     'properties': {'topic': {'type': 'string'}, 'clock_id': {'type': 'string'},
                                    'calibration_id': {'type': 'string'},
                                    'camera_from_body': {'type': 'array', 'items': {'type': 'number'},
                                                         'minItems': 4, 'maxItems': 4},
                                    'read_time_offset_ns': {'type': 'integer'}},
                     'required': ['topic', 'camera_from_body']}}},
                 'topic_in': [{'format': 'image/jpeg'}, {'format': 'image/depth-zlib'},
                              {'format': 'sensor/imu', 'optional': True},
                              {'format': 'state/odom', 'optional': True},
                              {'format': 'image/depth-f32-zlib', 'optional': True}],
                 'topic_out': self._outputs('default'), 'odom_interface': _INTERFACE}]

    @staticmethod
    def _outputs(instance):
        root = '/rgbd_odometry/i_' + instance.encode('utf-8').hex()
        return [{'topic': root + '/state/odom', 'format': 'state/odom'},
                {'topic': root + '/status', 'format': 'data/json'}]

    def dispatch(self, name, args):
        if name != 'rgbd_odometry':
            return None
        instance = args.get('instance_id') or 'default'
        if not isinstance(instance, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', instance):
            return {'state': 'error', 'message': 'invalid RGB-D instance ID'}
        action = args.get('action', 'info')
        with self.lock:
            current = self.instances.get(instance)
            try:
                topics = start_topics(args) if action == 'start' else None
                if action == 'start':
                    metadata = args.get('input_metadata')
                    metadata = metadata if isinstance(metadata, dict) else {}
                    formats = {topic: item.get('format') for topic, item in metadata.items()
                               if isinstance(item, dict)}
                    if isinstance(args.get('input_bindings'), dict):
                        formats.update(args['input_bindings'])
                    try:
                        calibration = validate_start_inputs(args.get('camera_info'), topics,
                            isinstance(topics, list) and len(topics) == 3, bindings=formats)
                    except ValueError as error:
                        logging.getLogger(__name__).warning('RGB-D %s start inputs: %s', instance, error)
                        raise
                    enabled = 'imu_topic' in calibration
                else:
                    enabled = bool(current and action != 'stop' and
                                   any(port['format'] == 'sensor/imu' for port in current[2]))
                # Saved canvases may still send config or old start options.
                settings = dict(visual_frontend='auto', max_features=1000, imu_assisted=enabled,
                                short_term_compensation=enabled, depth_geometry=enabled,
                                inertial_association=enabled)
                profile = self.controller_profiles.get(instance)
                if action == 'config':
                    if 'controller_rotation' in args:
                        profile = validate_controller_rotation(args['controller_rotation'])
                        if profile is None:
                            self.controller_profiles.pop(instance, None)
                        else:
                            self.controller_profiles[instance] = profile
                    settings['controller_rotation'] = deepcopy(profile)
                    return {'state': 'configured', 'config': settings}
                if action == 'start':
                    if self.closed:
                        raise RuntimeError('RGB-D plugin is shutting down')
                    if instance in self.retiring:
                        raise RuntimeError('finish failed RGB-D cleanup with stop before restarting')
                    if 'controller_rotation' in args:
                        profile = validate_controller_rotation(args['controller_rotation'])
                    if profile is not None:
                        if profile['topic'] in topics:
                            raise ValueError('controller rotation topic must be distinct from RGB-D and IMU inputs')
                        settings['controller_rotation'] = deepcopy(profile)
                    calibration.update(settings, visual_frontend=_FRONTENDS[0])
                    calibration['input_declarations'] = deepcopy(args.get('camera_info') or {})
                    image_topics = [calibration[k + '_topic'] for k in ('color', 'depth')]
                    from .camera_geometry import camera_parameter
                    geometry = {topic: {key: camera_parameter(args['camera_info'][topic], key)
                                        for key in ('width', 'height', 'K', 'D', 'distortion_model', 'format',
                                                    'color_from_depth', 'depth_scale_m', 'body_from_camera')}
                                for topic in image_topics}
                    signature = json.dumps([image_topics, calibration.get('imu_candidates'),
                                            calibration.get('external_candidates'), calibration.get('external_topic'),
                                            geometry, profile],
                                           sort_keys=True)
                    if current and current[1] != signature:
                        raise ValueError('stop RGB-D odometry before changing its input geometry or rotation alignment')
                    if not current:
                        require_runtime()
                        self._ensure_supervisor()
                        bindings = [{'topic': calibration['color_topic'], 'format': 'image/jpeg'},
                                    {'topic': calibration['depth_topic'], 'format': calibration['depth_format']}]
                        bindings.extend({'topic': topic, 'format': 'sensor/imu'}
                                        for topic in calibration.get('imu_candidates', ()))
                        if calibration.get('external_topic'):
                            bindings.append({'topic': calibration['external_topic'], 'format': 'state/odom'})
                        current = (RgbdSession(calibration, self.executor, self._outputs(instance)[0]['topic'],
                            **({'input_registry': self._input_registry} if self._input_registry else {})), signature, bindings)
                        if calibration.get('external_candidates'):
                            from .rgbd_external_input import ExternalOdometryInput
                            try:
                                stream = ExternalOdometryInput(self._outputs(instance)[0]['topic'],
                                    calibration, learning=args.get('external_motion_learning', True))
                            except Exception:
                                current[0].close(report_idle=False)
                                raise
                            self.external_streams[instance] = current[0].external_stream = stream
                            if getattr(current[0], 'observations', None) is not None:
                                current[0].observations.visual_sink = current[0]._observe_external_visual
                        controller = getattr(current[0], 'calibration', {}).get('controller_rotation')
                        if controller is not None:
                            bindings.append({'topic': controller['topic'], 'format': 'data/json'})
                        self.instances[instance] = current
                        self.recovery[instance] = dict(lost_since=None, last_attempt=None,
                            attempts=0, restarts=0, previous_segment=None, reason=None, error=None,
                            failure_streak=0, next_attempt=None, backoff_s=1.5, healthy_since=None,
                            mode_retry_after=None, restoration_attempts=0, transitions=[])
                        self.frontends[instance] = dict(active=calibration['visual_frontend'],
                            switches=0, untracked=0, last_switch_reason=None, decision=None,
                            cooldown_until=None)
                elif action == 'stop':
                    self._stop_instance(instance)
                    current = None
                elif action != 'info':
                    raise ValueError('unknown RGB-D odometry action')
                if current:
                    active_profile = getattr(current[0], 'requested_calibration', {}).get('controller_rotation')
                    if active_profile is not None:
                        settings['controller_rotation'] = deepcopy(active_profile)
                stream = self.external_streams.get(instance)
                info = current[0].info() if current else {'state': 'idle'}
                if stream is not None:
                    info['external_fusion'] = external = stream.info()
                    if 'stamp_ns' in external:
                        info.update(visual_status=info['status'], visual_segment_id=info['segment_id'],
                                    visual_reason=info['reason'], status=external['status'],
                                    reason=external['reason'], segment_id=external['segment_id'],
                                    pose_reference='body', last_measurement_ns=external['stamp_ns'],
                                    loss_kind='output' if external['status'] == 'lost' else None)
                return {**info,
                        'topic_out': self._outputs(instance), 'odom_interface': _INTERFACE,
                        'topic_in': current[2] if current else [],
                        'recovery': self._recovery_info(instance),
                        'frontend': self._frontend_info(instance),
                        'config': settings}
            except (ValueError, RuntimeError, ImportError, LookupError) as error:
                result = {'state': 'error', 'message': str(error)}
                if action == 'config':
                    result['adapter_ok'] = False
                return result

    def shutdown(self):
        with self.lock:
            self.closed = True
            if self.recovery_node is not None:
                self.recovery_node.destroy_timer(self.recovery_timer)
                dispose_node(self.executor, self.recovery_node, label='RGB-D recovery')
                self.recovery_node = self.recovery_timer = None
            failures = []
            for instance in set(self.instances) | set(self.retiring) | set(self.external_streams):
                try:
                    self._stop_instance(instance)
                except Exception as error:
                    failures.append(f'{instance}: {error}')
            if failures:
                raise RuntimeError('RGB-D shutdown incomplete: ' + '; '.join(failures))
