"""Generic RGB-D camera-motion card; robot geometry belongs to the producer."""
import json
import logging
import re
import threading
import time
from copy import deepcopy

from .rgbd_odometry_contract import validate_start_inputs
from .rgbd_odometry_runtime import RgbdSession, require_runtime
from utils.ros_lifecycle import dispose_node


_INTERFACE = {'schema': 'motus.odom/1', 'slam_pose': {'available': True, 'profile': 'motus_slam/1'}}


class RgbdOdometryPlugin:
    PREFIX = 'rgbd_odometry'

    def __init__(self, config, executor):
        self.executor = executor
        self.instances = {}
        self.defaults = {k: (config or {}).get(k, False) for k in
                         ('imu_assisted', 'short_term_compensation', 'depth_geometry')}
        self.defaults['visual_frontend'] = (config or {}).get('visual_frontend', 'f2m_feature')
        self.defaults['max_features'] = (config or {}).get('max_features', 1000)
        self.settings = {}
        self.lock = threading.RLock()
        self.recovery = {}
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
                    if info['status'] != 'lost':
                        state['lost_since'] = None
                        continue
                    if state['lost_since'] is None:
                        state['lost_since'] = now
                    if (now-state['lost_since'] < .8 or
                            (state['last_attempt'] is not None and now-state['last_attempt'] < 1.5) or
                            not session.has_fresh_inputs(time.time_ns())):
                        continue
                    state['last_attempt'] = now
                    state['attempts'] += 1
                    candidate = RgbdSession(session.calibration, self.executor,
                                            self._outputs(instance)[0]['topic'], publish_output=False)
                    session.close(report_idle=False)
                    with candidate.lock:
                        self.instances[instance] = candidate, signature, bindings
                        candidate.publish_output = True
                    state.update(restarts=state['restarts']+1, previous_segment=info['segment_id'],
                                 reason=info['reason'], error=None, lost_since=None)
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

    def _stop_instance(self, instance):
        failures = []
        if instance in self.retiring:
            try:
                self.retiring[instance].close(report_idle=False)
                del self.retiring[instance]
            except Exception as error:
                failures.append('candidate cleanup: ' + str(error))
        if instance in self.instances:
            try:
                self.instances[instance][0].close()
                del self.instances[instance]
            except Exception as error:
                failures.append(str(error))
        if failures:
            raise RuntimeError('; '.join(failures))
        self.recovery.pop(instance, None)

    def get_tools(self):
        return [{'name': 'rgbd_odometry', 'type': 'processor', 'multiInstance': True,
                 'description': 'RGB-D 视觉里程计：从配准彩色和实测深度估计相机位姿',
                 'inputSchema': {'type': 'object', 'properties': {
                     'action': {'type': 'string', 'enum': ['start', 'stop', 'info']}}},
                 'configSchema': {'type': 'object', 'properties': {
                     'imu_assisted': {'type': 'boolean', 'default': False, 'scope': 'instance',
                                      'description': '启用可选相机 IMU 辅助（需要第三路 IMU 输入）'},
                     'short_term_compensation': {'type': 'boolean', 'default': False, 'scope': 'instance',
                         'description': '启用 ESKF 短时补偿，需要 IMU 完整标定及低运动初始化'},
                     'depth_geometry': {'type': 'boolean', 'default': False, 'scope': 'instance',
                         'description': '启用深度几何配准辅助，需要先启用短时补偿'},
                     'visual_frontend': {'type': 'string', 'scope': 'instance',
                                         'default': self.defaults['visual_frontend'],
                                         'enum': ['f2m_feature', 'f2f_optical_flow'],
                                         'description': 'F2M 特征匹配（默认）或 F2F 光流；切换需停止实例'},
                     'max_features': {'type': 'integer', 'scope': 'instance',
                                      'default': self.defaults['max_features'],
                                      'minimum': 100, 'maximum': 5000,
                                      'description': '最大特征数，调整需停止实例'}}},
                 'topic_in': [{'format': 'image/jpeg'}, {'format': 'image/depth-zlib'},
                              {'format': 'sensor/imu', 'optional': True}],
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
                settings = {k: args.get(k, self.settings.get(instance, self.defaults).get(k, False))
                            for k in self.defaults}
                if any(type(settings[k]) is not bool for k in
                       ('imu_assisted', 'short_term_compensation', 'depth_geometry')):
                    raise ValueError('IMU, compensation and geometry flags must be boolean')
                if settings['visual_frontend'] not in ('f2m_feature', 'f2f_optical_flow'):
                    raise ValueError('unsupported visual_frontend: use f2m_feature or f2f_optical_flow')
                if (type(settings['max_features']) is not int or
                        not 100 <= settings['max_features'] <= 5000):
                    raise ValueError('max_features must be an integer in [100, 5000]')
                enabled = settings['imu_assisted']
                if settings['short_term_compensation'] and not enabled:
                    raise ValueError('short-term compensation requires imu_assisted')
                if settings['depth_geometry'] and not settings['short_term_compensation']:
                    raise ValueError('depth geometry requires short-term compensation')
                if action == 'config':
                    if current and settings != self.settings[instance]:
                        raise ValueError('stop RGB-D odometry before changing frontend or assistance settings')
                    self.settings[instance] = settings
                    return {'state': 'configured', 'config': settings.copy()}
                if action == 'start':
                    if self.closed:
                        raise RuntimeError('RGB-D plugin is shutting down')
                    if instance in self.retiring:
                        raise RuntimeError('finish failed RGB-D cleanup with stop before restarting')
                    calibration = validate_start_inputs(args.get('camera_info'), args.get('input_topics'), enabled)
                    calibration.update(settings)
                    geometry = deepcopy(args['camera_info'])
                    for declaration in geometry.values():
                        mount = (declaration.get('vendor') or {}).get('body_mount')
                        if isinstance(mount, dict):
                            mount.pop('status', None)
                            mount.pop('reason', None)
                    signature = json.dumps([args['input_topics'], geometry, settings], sort_keys=True)
                    if current and current[1] != signature:
                        raise ValueError('stop RGB-D odometry before changing its input geometry')
                    if not current:
                        require_runtime()
                        self._ensure_supervisor()
                        bindings = [{'topic': calibration[k + '_topic'], 'format': f}
                                    for k, f in (('color', 'image/jpeg'), ('depth', 'image/depth-zlib'))]
                        if enabled:
                            bindings.append({'topic': calibration['imu_topic'], 'format': 'sensor/imu'})
                        current = (RgbdSession(calibration, self.executor, self._outputs(instance)[0]['topic']), signature, bindings)
                        self.instances[instance] = current
                        self.settings[instance] = settings
                        self.recovery[instance] = dict(lost_since=None, last_attempt=None,
                            attempts=0, restarts=0, previous_segment=None, reason=None, error=None)
                elif action == 'stop':
                    self._stop_instance(instance)
                    current = None
                elif action != 'info':
                    raise ValueError('unknown RGB-D odometry action')
                return {**(current[0].info() if current else {'state': 'idle'}),
                        'topic_out': self._outputs(instance), 'odom_interface': _INTERFACE,
                        'topic_in': current[2] if current else [],
                        'recovery': {k: v for k, v in self.recovery.get(instance, {}).items()
                                     if k not in ('lost_since', 'last_attempt')},
                        'config': self.settings.get(instance, self.defaults).copy()}
            except (ValueError, RuntimeError, ImportError, LookupError) as error:
                return {'state': 'error', 'message': str(error)}

    def shutdown(self):
        with self.lock:
            self.closed = True
            if self.recovery_node is not None:
                self.recovery_node.destroy_timer(self.recovery_timer)
                dispose_node(self.executor, self.recovery_node, label='RGB-D recovery')
                self.recovery_node = self.recovery_timer = None
            failures = []
            for instance in set(self.instances) | set(self.retiring):
                try:
                    self._stop_instance(instance)
                except Exception as error:
                    failures.append(f'{instance}: {error}')
            if failures:
                raise RuntimeError('RGB-D shutdown incomplete: ' + '; '.join(failures))
