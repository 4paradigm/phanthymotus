"""Point-cloud mapping card backed by the native RTAB-Map process."""
from copy import deepcopy
import json
import math
from pathlib import Path
import re
import shutil
import threading
import uuid

import yaml

from .pointcloud_slam_input import resolve_inputs
from .pointcloud_slam_ros import RtabmapNode, discover_input_bindings
from .pointcloud_slam_rtabmap import require_runtime
from .pointcloud_slam_transforms import rigid
from .pointcloud_slam_view import SavedMapView
from utils.ros_lifecycle import dispose_node

_ARCHIVE_DIR = '/models/pointcloud_slam'
_DEFAULTS = {'robot': '其他', 'input_policy': 'strict', 'clock_relation_verified': False,
             'extrinsic_calibrated': False, 'max_gap_ms': 200, 'max_age_s': 2,
             'max_measurement_age_ms': 1500, 'max_future_skew_ms': 100,
             'max_pending': 10, 'max_cloud_points': 1000000,
             'voxel_size_m': .08, 'max_working_nodes': 300, 'max_update_ms': 100,
             'max_disk_bytes': 2147483648, 'reserve_bytes': 1073741824,
             'preview_interval_s': 1, 'preview_max_points': 40000}
_RETIRED = {'preview_enabled', 'preview_backend', 'preview_geometry', 'preview_relocalization',
            'preview_odom_sigmas', 'archive_name', 'restore_archive',
            'max_frames', 'max_points', 'persistent_history'}
_SENSOR_KEYS = {'cloud_frame', 'child_frame', 'clock_id', 'child_from_cloud'}
_ACTIONS = {
    'start_mapping': ('Start mapping', ['map_name']),
    'stop_mapping': ('Stop mapping and save', []),
    'list_maps': ('List maps', []), 'select_map': ('View saved map', ['map_id']),
    'rename_map': ('Rename saved map', ['map_id', 'map_name']),
    'close_map_view': ('Close map view', ['map_id']), 'delete_map': ('Delete saved map', ['map_id'])}


def _robot_config(config):
    robot = config.get('robot', '其他')
    if robot not in ('Bumi', 'G1', 'Tianyi', '其他'):
        raise ValueError('unknown robot selection')
    if robot == '其他':
        return config
    config = {k: v for k, v in config.items() if k not in _SENSOR_KEYS}
    if robot == 'Bumi':
        path = Path(__file__).parent.parent / 'configs' / 'pointcloud_slam_bumi.yaml'
        with path.open() as source:
            config.update(yaml.safe_load(source))
    else:
        # These drivers need compatible odometry before a sensor preset can be established.
        config.update(input_policy='strict', clock_relation_verified=False, extrinsic_calibrated=False)
    return config


def _config_properties():
    props = {key: {'type': 'integer' if type(value) is int else 'boolean' if type(value) is bool
                  else 'number' if isinstance(value, float) else 'string',
                  'default': value, 'scope': 'instance'} for key, value in _DEFAULTS.items()}
    props['input_policy']['enum'] = ['strict', 'experimental']
    props['robot'].update(title='机器人', enum=['Bumi', 'G1', 'Tianyi', '其他'],
        description='Bumi 使用当前已验证预设；G1、Tianyi 预留待接入里程计。选择“其他”时手动填写高级设置，预设在保存后生效。')
    props.update({key: {'type': 'string', 'scope': 'instance'}
                  for key in ('cloud_frame', 'child_frame', 'clock_id')})
    props['child_from_cloud'] = {'type': 'string', 'scope': 'instance',
        'title': 'Cloud to odometry reference transform', 'description': 'Rigid 4x4 transform, meters (JSON)'}
    props['max_update_ms']['description'] = 'RTAB-Map memory management time threshold; not a hard deadline'
    for key, prop in props.items():
        if key != 'robot':
            prop['advanced'] = True
    return props


class PointCloudSlamPlugin:
    PREFIX = 'pointcloud_slam'

    def __init__(self, config, executor):
        # 标点移至后续导航卡片；忽略旧画布中的标注平面，不阻止项目启动。
        self._config = _robot_config({**_DEFAULTS, **{k: v for k, v in (config or {}).items()
                                                     if k not in _RETIRED and k != 'map_plane'}})
        retired = sorted(_RETIRED.intersection(config or {}))
        self._config_error = ('Remove obsolete SLAM configuration: ' + ', '.join(retired)) if retired else None
        self._executor = executor
        self._node = self._pending = self._view = None
        self._view_id = self._session_dir = self._map_name = self._save_request = None
        self._project_inputs = None
        self._topics = {}
        self._resolved_inputs = None
        self._input_diagnostics = []
        self._legacy_core = True
        self._session_settings = {}
        self._last_stop = None
        self._save_status = {'state': 'idle'}
        self._capture_stop = threading.Event()
        self._project_stopping = False
        self._blocking_stops = self._selecting_views = 0
        self._views_idle = threading.Event()
        self._views_idle.set()
        self._lock = threading.RLock()
        self._operation_lock = threading.Lock()
        self._save_lock = threading.Lock()
        self._worker = None
        self._operation = {}
        self._closed_outputs = []

    def get_tools(self):
        props = _config_properties()
        for key, value in self._settings().items():
            if key in props and value is not None:
                props[key]['default'] = json.dumps(value) if key == 'child_from_cloud' else value
        verbs = ['start', 'stop', 'info', 'config', *_ACTIONS, 'operation_status']
        actions = {key: {'requires_project': key == 'start_mapping'} for key in _ACTIONS}
        actions['operation_status'] = {'internal': True}
        actions['delete_map']['confirm'] = '删除地图及其全部版本？'
        for key in ('stop_mapping', 'select_map'):
            actions[key].update(background=True, status_action='operation_status',
                pending_states=['working'], error_states=['error'], poll_interval_ms=500)
        schema = {'type': 'object', 'properties': {
            'action': {'type': 'string', 'enum': verbs,
                       'oneOf': [{'const': k, 'title': _ACTIONS[k][0] if k in _ACTIONS else k} for k in verbs]},
            'input_topics': {'type': 'array', 'items': {'type': 'string'}},
            'input_bindings': {'type': 'object'}, 'odom_interfaces': {'type': 'object'},
            'input_metadata': {'type': 'object'},
            **{key: {'type': 'string'} for key in ('map_id', 'map_name', 'request_id')},
            'background': {'type': 'boolean'}, 'revision': {'type': 'integer'}}, 'required': ['action'],
            'x-action-params': {k: {'params': v[1]} for k, v in _ACTIONS.items()},
            'x-map-manager': {'schema': 'motus.map-manager/1', 'actions': {
                k: k for k in ('list_maps', 'select_map', 'rename_map', 'delete_map', 'close_map_view')}},
            'x-action-lifecycle': {'schema': 'motus.action-lifecycle/1', 'config_actions': ['start'], 'actions': actions}}
        schema['properties']['map_id']['x-resource-options'] = {
            'action': 'list_maps', 'items': 'maps', 'value': 'id', 'label': 'name'}
        return [{'name': self.PREFIX, 'type': 'processor', 'multiInstance': False,
                 'description': 'RTAB-Map point-cloud mapping with existing odometry',
                 'inputSchema': schema, 'configSchema': {'type': 'object', 'properties': props},
                 'topic_in': [{'format': 'state/odom'}, {'format': 'sensor/pointcloud2'}],
                 'topic_out': [{'format': 'sensor/mapping', 'desc': 'RTAB-Map point cloud'},
                               {'format': 'data/json', 'desc': 'Mapping status'}]}]

    def start(self):
        pass  # Canvas start supplies the connected native inputs.

    def _outputs(self):
        if self._node:
            return self._node.outputs()
        if not self._project_inputs:
            return []
        cloud = next(t for t, f in self._topics.items() if f == 'sensor/pointcloud2')
        topic = cloud + '/slam_odom_preview'
        return [{'topic': topic, 'format': 'sensor/mapping'},
                {'topic': cloud + '/slam_odom_preview_status', 'format': 'data/json'}]

    def _dispose(self, node):
        if node is None:
            return
        outputs = node.outputs()
        node.close()
        dispose_node(self._executor, node, label=self.PREFIX)
        with self._operation_lock:
            self._closed_outputs.extend(o['topic'] for o in outputs)

    def _close_view(self):
        view, self._view = self._view, None
        self._view_id = None
        self._dispose(view)

    def _settings(self):
        return {**self._config, **(self._resolved_inputs or {}).get('settings', {})}

    def _validate(self, cfg):
        if cfg.get('robot') in ('G1', 'Tianyi') and not all(cfg.get(k) for k in
                ('cloud_frame', 'child_frame', 'clock_id')):
            raise ValueError(cfg['robot'] + ' preset awaits compatible state/odom and verified sensor configuration; '
                             'use 其他 for manually configured inputs')
        for key in ('cloud_frame', 'child_frame', 'clock_id'):
            if not isinstance(cfg.get(key), str) or not cfg[key].strip():
                raise ValueError(key + ' is required')
        if cfg.get('extrinsic_mode', 'static') != 'dynamic':
            rigid(cfg.get('child_from_cloud'))
        if cfg.get('input_policy') not in ('strict', 'experimental'):
            raise ValueError('input_policy must be strict or experimental')
        if cfg['input_policy'] == 'strict' and not all(cfg.get(k) is True for k in
                ('clock_relation_verified', 'extrinsic_calibrated')):
            raise ValueError('verified clock relation and calibrated extrinsic required')
        for key in ('max_pending', 'max_cloud_points', 'max_working_nodes', 'preview_max_points', 'max_disk_bytes'):
            if type(cfg.get(key)) is not int or cfg[key] <= 0:
                raise ValueError(key + ' must be a positive integer')
        if cfg['preview_max_points'] > 80000:
            raise ValueError('preview_max_points exceeds existing renderer capacity')
        if type(cfg.get('reserve_bytes')) is not int or cfg['reserve_bytes'] < 0:
            raise ValueError('reserve_bytes must be a nonnegative integer')
        for key in ('max_gap_ms', 'max_age_s', 'max_measurement_age_ms', 'voxel_size_m',
                    'max_update_ms', 'preview_interval_s', 'max_future_skew_ms'):
            value = cfg.get(key)
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(value) or value < 0 or (value == 0 and key != 'max_future_skew_ms')):
                raise ValueError(key + ' has an invalid value')

    def dispatch(self, name, args):
        try:
            result = self._dispatch_action(args.get('action', name), args)
        except (ImportError, ValueError, TypeError, KeyError, OSError, RuntimeError) as error:
            result = {'state': 'error', 'message': str(error)}
        with self._operation_lock:
            closed, self._closed_outputs = self._closed_outputs, []
        if closed and isinstance(result, dict):
            result = {**result, 'closed_topics': list(dict.fromkeys(closed))}
        return result

    def _dispatch_action(self, action, args):
        if action == 'operation_status':
            request = args.get('request_id')
            with self._operation_lock:
                for operation in (self._operation, self._save_status):
                    if request and operation.get('request_id') == request:
                        return dict(operation, state='working' if operation['state'] == 'saving' else operation['state'])
            from .pointcloud_slam_library import map_entries
            for entry in map_entries(_ARCHIVE_DIR):
                for saved in entry['saves']:
                    if saved['request_id'] == request:
                        return self._saved_result(entry, saved)
            return {'state': 'error', 'message': 'unknown map operation'}
        if args.get('background') is True and action in ('stop_mapping', 'select_map'):
            request = args.get('request_id')
            self._check_request(request)
            with self._operation_lock:
                current = self._operation
                identity = (action, args.get('map_id'))
                if current.get('request_id') == request:
                    if current['identity'] != identity:
                        raise ValueError('map request id reused')
                    if current['state'] != 'error':
                        return dict(current)
                if self._project_stopping or self._blocking_stops or self._save_status['state'] == 'saving' or current.get('state') == 'working':
                    raise ValueError('map operation already in progress')
                self._operation = {'state': 'working', 'request_id': request, 'identity': identity}
                def run():
                    try:
                        result = self._dispatch(action, {**args, 'background': False})
                    except Exception as error:
                        result = {'state': 'error', 'message': str(error)}
                    with self._operation_lock:
                        self._operation.update(result, request_id=request)
                self._worker = threading.Thread(target=run, name='rtabmap_operation', daemon=True)
                self._worker.start()
                return dict(self._operation)
        with self._operation_lock:
            busy = self._project_stopping or self._blocking_stops or self._save_status['state'] == 'saving'
            if ((busy or self._operation.get('state') == 'working') and action not in ('info', 'stop', 'list_maps')
                    and not (action == 'config' and (self._node or self._pending))):
                raise ValueError('map operation already in progress')
            if action == 'select_map':
                self._selecting_views += 1
                self._views_idle.clear()
        try:
            return self._dispatch(action, args)
        finally:
            if action == 'select_map':
                with self._operation_lock:
                    self._selecting_views -= 1
                    if not self._selecting_views:
                        self._views_idle.set()

    @staticmethod
    def _check_request(request):
        if not isinstance(request, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', request):
            raise ValueError('invalid map request id')

    def _dispatch(self, action, args):
        if action == 'stop':
            return self._stop_project()
        if action == 'stop_mapping':
            return self._stop_mapping(args.get('request_id'))
        if action == 'info':
            with self._operation_lock:
                save = deepcopy(self._save_status)
                stopping = self._project_stopping or self._blocking_stops
            if stopping:
                return {'state': 'idle', 'capture_stopped': True, 'backend': 'rtabmap',
                        'save_state': save['state'], 'save_result': save, 'topic_in': [], 'topic_out': []}
            with self._lock:
                return {'state': 'idle' if self._capture_stop.is_set() else
                        'running' if self._node else 'save_pending' if self._pending else
                        'ready' if self._project_inputs else 'idle', 'backend': 'rtabmap',
                        'capture_stopped': self._capture_stop.is_set(), 'map_name': self._map_name,
                        'selected_map_id': self._view_id, 'last_stop': deepcopy(self._last_stop),
                        'save_state': save['state'], 'save_result': save,
                        'input_validation': {key: self._settings().get(key) is True for key in
                            ('clock_relation_verified', 'extrinsic_calibrated')},
                        'input_compatibility': {'mode': 'legacy_core' if self._legacy_core else 'explicit',
                            'role_source': 'not_resolved' if not self._topics else
                                'native_ros_publishers' if self._legacy_core else 'input_bindings',
                            'producer_metadata_provided': self._resolved_inputs is not None},
                        'input_diagnostics': deepcopy(self._input_diagnostics),
                        'topic_in': [{'topic': t, 'format': f} for t, f in self._topics.items()],
                        'topic_out': self._outputs() + (self._view.outputs() if self._view else []),
                        'input_status': self._node.status() if self._node else {'odometry': 'waiting'}}
        with self._lock:
            if action == 'config':
                mapping = self._node is not None or self._pending is not None
                try:
                    retired = _RETIRED.intersection(args)
                    if retired:
                        raise ValueError('Remove obsolete SLAM configuration: ' + ', '.join(sorted(retired)))
                    updates = {k: v for k, v in args.items() if k in _config_properties() and v is not None and v != ''}
                    # Old saved projects have no selector; never inherit the prior project's robot.
                    updates.setdefault('robot', '其他')
                    base = dict(self._config)
                    if updates['robot'] == '其他' and base.get('robot') != '其他':
                        for key in ('input_policy', 'clock_relation_verified', 'extrinsic_calibrated'):
                            base[key] = _DEFAULTS[key]
                    merged = _robot_config({**base, **updates})
                    if isinstance(merged.get('child_from_cloud'), str):
                        merged['child_from_cloud'] = json.loads(merged['child_from_cloud'])
                    if 'child_from_cloud' in merged:
                        merged['child_from_cloud'] = rigid(merged['child_from_cloud']).tolist()
                    for key in ('clock_relation_verified', 'extrinsic_calibrated'):
                        if type(merged[key]) is not bool:
                            raise ValueError(key + ' must be boolean')
                    if all(merged.get(k) is not None for k in ('cloud_frame', 'child_frame', 'clock_id', 'child_from_cloud')):
                        self._validate({**merged, **(self._resolved_inputs or {}).get('settings', {})})
                except (ValueError, TypeError) as error:
                    if not mapping:
                        self._config_error = str(error)
                    raise
                if mapping:
                    if merged != self._config:
                        raise ValueError('stop mapping before changing configuration')
                    return {'state': 'configured', 'config': dict(self._config)}
                self._config = merged
                self._config_error = None
                return {'state': 'configured', 'config': dict(self._config)}
            if action == 'start':
                return self._prepare(args)
            if action == 'start_mapping':
                return self._start_mapping(args)
            return self._map_action(action, args)

    def _prepare(self, args):
        if self._config_error:
            raise ValueError('invalid configuration: ' + self._config_error)
        if self._node or self._pending:
            raise ValueError('finish mapping and the pending save before starting')
        if self._executor is None:
            raise ValueError('ROS executor unavailable')
        topics = args.get('input_topics')
        legacy = 'input_bindings' not in args
        bindings = discover_input_bindings(topics) if legacy else args.get('input_bindings')
        if (not isinstance(topics, list) or len(topics) != 2 or len(set(topics)) != 2 or
                not isinstance(bindings, dict) or set(bindings) != set(topics) or
                set(bindings.values()) != {'state/odom', 'sensor/pointcloud2'}):
            raise ValueError('one format-resolved odometry and one PointCloud2 input required')
        odom = next(t for t, f in bindings.items() if f == 'state/odom')
        if not legacy or 'odom_interfaces' in args:
            interfaces = args.get('odom_interfaces')
            interface = interfaces.get(odom) if isinstance(interfaces, dict) else None
            capability = interface.get('slam_pose') if isinstance(interface, dict) else None
            if (not isinstance(interface, dict) or not isinstance(capability, dict) or
                    interface.get('schema') != 'motus.odom/1' or capability.get('available') is not True or
                    capability.get('profile') != 'motus_slam/1'):
                raise ValueError('odometry source lacks declared SLAM pose capability')
        self._resolved_inputs = None
        self._input_diagnostics = []
        if 'input_metadata' in args:
            try:
                self._resolved_inputs = resolve_inputs(args['input_metadata'], bindings,
                    experimental=self._config['input_policy'] == 'experimental')
            except ValueError as error:
                self._input_diagnostics = deepcopy(getattr(error, 'diagnostics', []))
                raise
        self._validate(self._settings())
        require_runtime()
        with self._operation_lock:
            if self._project_stopping or self._blocking_stops:
                raise ValueError('project is stopping')
            self._capture_stop = threading.Event()
        self._topics = dict(bindings)
        self._legacy_core = legacy
        self._project_inputs = deepcopy(args)
        return {'state': 'ready', 'backend': 'rtabmap', 'topic_out': self._outputs()}

    def _start_mapping(self, args):
        from .pointcloud_slam_library import map_entries, validate_map_name
        if self._config_error:
            raise ValueError('invalid configuration: ' + self._config_error)
        if self._node or self._pending:
            raise ValueError('finish the pending map save first')
        if not self._project_inputs or self._capture_stop.is_set():
            raise ValueError('start the project before mapping')
        self._validate(self._settings())
        name = validate_map_name(args.get('map_name'))
        if any(e['name'] == name for e in map_entries(_ARCHIVE_DIR)):
            raise ValueError('map name already in use')
        with self._operation_lock:
            if self._capture_stop.is_set() or self._project_stopping or self._blocking_stops:
                return {'state': 'idle', 'capture_stopped': True}
            self._last_stop = None
        session = Path(_ARCHIVE_DIR) / 'sessions' / uuid.uuid4().hex
        session.mkdir(parents=True, exist_ok=False)
        settings = deepcopy(self._settings())
        settings['resolved_inputs'] = deepcopy((self._resolved_inputs or {}).get('inputs', {}))
        settings['input_validation'] = {'policy': settings['input_policy'],
            'clock_relation_verified': settings.get('clock_relation_verified') is True,
            'extrinsic_calibrated': settings.get('extrinsic_calibrated') is True}
        node = None
        try:
            node = RtabmapNode(next(t for t, f in self._topics.items() if f == 'state/odom'),
                next(t for t, f in self._topics.items() if f == 'sensor/pointcloud2'), settings,
                session_dir=str(session), input_contract=self._resolved_inputs, capture_stop=self._capture_stop)
            self._executor.add_node(node)
            node.start()
        except Exception:
            if node:
                try:
                    self._dispose(node)
                except Exception:
                    pass
            # Keep a partial database after failed native startup for diagnosis.
            raise
        self._session_dir, self._session_settings = session, settings
        self._node, self._map_name = node, name
        if self._capture_stop.is_set():
            node.freeze_capture()
            return {'state': 'idle', 'capture_stopped': True}
        return {'state': 'running', 'map_name': name, 'backend': 'rtabmap', 'topic_out': self._outputs()}

    @staticmethod
    def _saved_result(entry, saved):
        return {'state': 'saved', 'map_id': entry['id'], 'map_name': entry['name'],
                'revision': saved['revision'], 'request_id': saved['request_id']}

    def _stop_mapping(self, request=None):
        from .pointcloud_slam_library import map_entries, save_rtabmap_revision
        if request is not None:
            self._check_request(request)
        with self._save_lock:
            with self._lock:
                if self._node is None and self._pending is None:
                    if request:
                        for entry in map_entries(_ARCHIVE_DIR):
                            for saved in entry['saves']:
                                if saved['request_id'] == request:
                                    return self._saved_result(entry, saved)
                    return deepcopy(self._last_stop) or {
                        'state': 'ready' if self._project_inputs else 'idle', 'request_id': request}
                if self._node:
                    self._node.freeze_capture()
                    self._pending, self._node = self._node, None
                    self._save_request = request or uuid.uuid4().hex
                elif request is not None and request != self._save_request:
                    raise ValueError('pending save retains its original request id')
                pending = self._pending
                save_request = self._save_request
            with self._operation_lock:
                self._save_status = {'state': 'saving', 'request_id': save_request}
            try:
                pending.close()
                artifacts = pending.artifacts()
                if artifacts['segments']:
                    map_id, revision, _ = save_rtabmap_revision(_ARCHIVE_DIR, artifacts,
                        self._session_settings, self._map_name, None, save_request)
                    result = {'state': 'saved', 'map_id': map_id, 'map_name': self._map_name,
                              'revision': revision, 'request_id': save_request}
                else:
                    result = {'state': 'ready', 'saved': False, 'message': 'No map nodes persisted',
                              'request_id': save_request}
                self._dispose(pending)
                shutil.rmtree(self._session_dir, ignore_errors=True)
                with self._lock:
                    self._pending = self._session_dir = self._map_name = self._save_request = None
                    self._last_stop = result
            except Exception as error:
                with self._operation_lock:
                    self._save_status = {'state': 'error', 'message': str(error), 'request_id': save_request}
                raise
            with self._operation_lock:
                self._save_status = dict(result, state='saved' if result['state'] == 'saved' else 'idle')
            return result

    def _stop_ack(self):
        return {'state': 'idle', 'capture_stopped': True,
                'save_state': self._save_status['state'], 'save_result': dict(self._save_status)}

    def _stop_project(self):
        with self._operation_lock:
            already_stopped = self._capture_stop.is_set()
            self._capture_stop.set()
            if self._node:
                self._node.freeze_capture()
            if self._project_stopping or self._blocking_stops or (already_stopped and
                    self._save_status['state'] != 'error' and self._view is None and
                    not self._selecting_views and self._operation.get('state') != 'working'):
                return self._stop_ack()
            self._project_stopping = True
            previous = self._worker
            manual_request = (self._operation.get('request_id')
                if self._operation.get('state') == 'working' and
                   self._operation.get('identity', (None,))[0] == 'stop_mapping' else None)
            request = (self._save_request or manual_request or
                       (self._last_stop or {}).get('request_id') or uuid.uuid4().hex)
            self._save_status = {'state': 'saving', 'request_id': request}
            def finish():
                try:
                    if previous:
                        previous.join()
                    self._views_idle.wait()
                    with self._lock:
                        self._close_view()
                        self._project_inputs = None
                        self._topics = {}
                    result = self._stop_mapping(request)
                    with self._operation_lock:
                        self._save_status = dict(result, state='saved' if result['state'] == 'saved' else 'idle')
                except Exception as error:
                    with self._operation_lock:
                        self._save_status = {**self._save_status, 'state': 'error', 'message': str(error)}
                finally:
                    with self._operation_lock:
                        self._project_stopping = False
            self._worker = threading.Thread(target=finish, name='rtabmap_project_save', daemon=True)
            self._worker.start()
            return self._stop_ack()

    def stop(self):
        with self._operation_lock:
            self._blocking_stops += 1
            self._capture_stop.set()
            if self._node:
                self._node.freeze_capture()
            worker = self._worker
        try:
            if worker:
                worker.join()
            self._views_idle.wait()
            with self._lock:
                self._close_view()
                self._project_inputs = None
                self._topics = {}
            with self._operation_lock:
                error = dict(self._save_status) if self._save_status['state'] == 'error' else None
            if error:
                return error
            self._stop_mapping()
            return {'state': 'idle'}
        except (OSError, ValueError, RuntimeError) as error:
            return {'state': 'error', 'message': str(error)}
        finally:
            with self._operation_lock:
                self._blocking_stops -= 1

    def shutdown(self):
        result = self.stop()
        if result['state'] == 'error':
            raise RuntimeError(result['message'])

    def _map_action(self, action, args):
        from . import pointcloud_slam_library as library
        map_id, revision = args.get('map_id'), args.get('revision')
        if action == 'list_maps':
            return {'state': 'ok', 'maps': [{k: e[k] for k in ('id', 'name', 'revision')}
                                           for e in library.map_entries(_ARCHIVE_DIR)]}
        if action == 'select_map':
            if self._executor is None:
                raise ValueError('ROS executor unavailable')
            entry = next((e for e in library.map_entries(_ARCHIVE_DIR) if e['id'] == map_id), None)
            if entry is None:
                raise ValueError('unknown map')
            revision = revision or entry['revision']
            if self._view_id != (map_id, revision):
                snapshots = library.load_map_snapshots(_ARCHIVE_DIR, map_id, revision,
                    max_points=self._config['preview_max_points'])
                view = SavedMapView(map_id, revision, entry['name'], snapshots,
                                    interval_s=self._config['preview_interval_s'])
                try:
                    self._executor.add_node(view)
                    view.start()
                except Exception:
                    self._dispose(view)
                    raise
                self._close_view()
                self._view, self._view_id = view, (map_id, revision)
            return {'state': 'map_selected', 'map_id': map_id, 'map_name': entry['name'],
                    'revision': revision, 'tracking': False,
                    'snapshot_available': bool(self._view.outputs()), 'topic_out': self._view.outputs()}
        if action == 'close_map_view':
            if self._view_id and self._view_id[0] == map_id:
                self._close_view()
            return {'state': 'ok'}
        if action == 'rename_map':
            entry = library.rename_map(_ARCHIVE_DIR, map_id, args.get('map_name'))
            if self._last_stop and self._last_stop.get('map_id') == map_id:
                self._last_stop['map_name'] = entry['name']
            if self._view_id and self._view_id[0] == map_id:
                self._view.rename(entry['name'])
            return {'state': 'renamed', 'map_id': map_id, 'map_name': entry['name'],
                    'topic_out': self._view.outputs() if self._view else []}
        if action == 'delete_map':
            if self._view_id and self._view_id[0] == map_id:
                self._close_view()
            library.delete_map(_ARCHIVE_DIR, map_id)
            if self._last_stop and self._last_stop.get('map_id') == map_id:
                self._last_stop = None
            return {'state': 'deleted', 'map_id': map_id}
        return None
