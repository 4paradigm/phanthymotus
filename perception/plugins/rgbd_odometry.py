"""Generic RGB-D camera-motion card; robot geometry belongs to the producer."""
import json
import re
import threading
from copy import deepcopy

from .rgbd_odometry_contract import validate_start_inputs
from .rgbd_odometry_runtime import RgbdSession, require_runtime


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

    def get_tools(self):
        return [{'name': 'rgbd_odometry', 'type': 'processor', 'multiInstance': True,
                 'description': 'RGB-D 视觉里程计：从配准彩色和实测深度估计相机位姿',
                 'inputSchema': {'type': 'object', 'properties': {
                     'action': {'type': 'string', 'enum': ['start', 'stop', 'info', 'config']},
                     'instance_id': {'type': 'string'}, 'input_topics': {'type': 'array', 'items': {'type': 'string'}},
                     'camera_info': {'type': 'object'}, 'imu_assisted': {'type': 'boolean'},
                     'short_term_compensation': {'type': 'boolean'}, 'depth_geometry': {'type': 'boolean'},
                     'visual_frontend': {'type': 'string', 'enum': ['f2m_feature', 'f2f_optical_flow']},
                     'max_features': {'type': 'integer', 'minimum': 100, 'maximum': 5000}}},
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
                        bindings = [{'topic': calibration[k + '_topic'], 'format': f}
                                    for k, f in (('color', 'image/jpeg'), ('depth', 'image/depth-zlib'))]
                        if enabled:
                            bindings.append({'topic': calibration['imu_topic'], 'format': 'sensor/imu'})
                        current = (RgbdSession(calibration, self.executor, self._outputs(instance)[0]['topic']), signature, bindings)
                        self.instances[instance] = current
                        self.settings[instance] = settings
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
                        'config': self.settings.get(instance, self.defaults).copy()}
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
