"""Generic RGB-D camera-motion card; robot geometry belongs to the producer."""
import json
import re
import threading

from .rgbd_odometry_contract import validate_start_inputs
from .rgbd_odometry_runtime import RgbdSession, require_runtime


_INTERFACE = {'schema': 'motus.odom/1', 'slam_pose': {'available': True, 'profile': 'motus_slam/1'}}


class RgbdOdometryPlugin:
    PREFIX = 'rgbd_odometry'

    def __init__(self, config, executor):
        self.executor = executor
        self.instances = {}
        self.imu_default = (config or {}).get('imu_assisted', False)
        self.settings = {}
        self.lock = threading.RLock()

    def get_tools(self):
        return [{'name': 'rgbd_odometry', 'type': 'processor', 'multiInstance': True,
                 'description': 'RGB-D 视觉里程计：从配准彩色和实测深度估计相机位姿',
                 'inputSchema': {'type': 'object', 'properties': {
                     'action': {'type': 'string', 'enum': ['start', 'stop', 'info', 'config']},
                     'instance_id': {'type': 'string'}, 'input_topics': {'type': 'array', 'items': {'type': 'string'}},
                     'camera_info': {'type': 'object'}, 'imu_assisted': {'type': 'boolean'}}},
                 'configSchema': {'type': 'object', 'properties': {
                     'imu_assisted': {'type': 'boolean', 'default': False, 'scope': 'instance',
                                      'description': '启用可选相机 IMU 姿态辅助（需要第三路 IMU 输入）'}}},
                 'topic_in': [{'format': 'image/jpeg'}, {'format': 'image/depth-zlib'},
                              {'format': 'sensor/imu', 'optional': True}],
                 'topic_out': self._outputs('default'), 'odom_interface': _INTERFACE}]

    @staticmethod
    def _outputs(instance):
        return [{'topic': '/rgbd_odometry/i_' + instance.encode('utf-8').hex() + '/state/odom', 'format': 'state/odom'}]

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
                enabled = args.get('imu_assisted', self.settings.get(instance, self.imu_default))
                if type(enabled) is not bool:
                    raise ValueError('imu_assisted must be a boolean')
                if action == 'config':
                    if current and enabled != self.settings[instance]:
                        raise ValueError('stop RGB-D odometry before changing IMU mode')
                    self.settings[instance] = enabled
                    return {'state': 'configured', 'config': {'imu_assisted': enabled}}
                if action == 'start':
                    calibration = validate_start_inputs(args.get('camera_info'), args.get('input_topics'), enabled)
                    signature = json.dumps([args['input_topics'], args['camera_info'], enabled], sort_keys=True)
                    if current and current[1] != signature:
                        raise ValueError('stop RGB-D odometry before changing its input geometry')
                    if not current:
                        require_runtime()
                        bindings = [{'topic': calibration[k + '_topic'], 'format': f}
                                    for k, f in (('color', 'image/jpeg'), ('depth', 'image/depth-zlib'))]
                        if enabled:
                            bindings.append({'topic': calibration['imu_topic'], 'format': 'sensor/imu'})
                        current = (RgbdSession(calibration, self.executor, self._outputs(instance)[0]['topic']), signature, bindings)
                        self.instances[instance] = current
                        self.settings[instance] = enabled
                elif action == 'stop':
                    if current:
                        current[0].close()
                        del self.instances[instance]
                    current = None
                elif action != 'info':
                    raise ValueError('unknown RGB-D odometry action')
                return {**(current[0].info() if current else {'state': 'idle'}),
                        'topic_out': self._outputs(instance), 'odom_interface': _INTERFACE,
                        'topic_in': current[2] if current else [],
                        'config': {'imu_assisted': self.settings.get(instance, self.imu_default)}}
            except (ValueError, RuntimeError, ImportError, LookupError) as error:
                return {'state': 'error', 'message': str(error)}

    def shutdown(self):
        with self.lock:
            failures = []
            for instance in list(self.instances):
                try:
                    self.instances[instance][0].close()
                    del self.instances[instance]
                except Exception as error:
                    failures.append(f'{instance}: {error}')
            if failures:
                raise RuntimeError('RGB-D shutdown incomplete: ' + '; '.join(failures))
