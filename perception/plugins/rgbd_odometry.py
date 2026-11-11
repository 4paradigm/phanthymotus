"""Generic RGB-D camera-motion card; robot geometry belongs to the producer."""
import json
import re
import threading

from .rgbd_odometry_contract import validate_inputs
from .rgbd_odometry_runtime import RgbdSession, require_runtime


_INTERFACE = {'schema': 'motus.odom/1', 'slam_pose': {'available': True, 'profile': 'motus_slam/1'}}


class RgbdOdometryPlugin:
    PREFIX = 'rgbd_odometry'

    def __init__(self, config, executor):
        self.executor = executor
        self.instances = {}
        self.lock = threading.RLock()

    def get_tools(self):
        return [{'name': 'rgbd_odometry', 'type': 'processor', 'multiInstance': True,
                 'description': 'RGB-D 视觉里程计：从配准彩色和实测深度估计相机位姿',
                 'inputSchema': {'type': 'object', 'properties': {
                     'action': {'type': 'string', 'enum': ['start', 'stop', 'info']},
                     'instance_id': {'type': 'string'}, 'input_topics': {'type': 'array', 'items': {'type': 'string'}},
                     'camera_info': {'type': 'object'}}},
                 'topic_in': [{'format': 'image/jpeg'}, {'format': 'image/depth-zlib'}],
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
                if action == 'start':
                    calibration = validate_inputs(args.get('camera_info'), args.get('input_topics'))
                    signature = json.dumps([args['input_topics'], args['camera_info']], sort_keys=True)
                    if current and current[1] != signature:
                        raise ValueError('stop RGB-D odometry before changing its input geometry')
                    if not current:
                        require_runtime()
                        current = (RgbdSession(calibration, self.executor, self._outputs(instance)[0]['topic']), signature)
                        self.instances[instance] = current
                elif action == 'stop':
                    if current:
                        current[0].close()
                        del self.instances[instance]
                    current = None
                elif action != 'info':
                    raise ValueError('unknown RGB-D odometry action')
                return {**(current[0].info() if current else {'state': 'idle'}),
                        'topic_out': self._outputs(instance), 'odom_interface': _INTERFACE}
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
