"""Opt-in experimental input card with clearly labelled odometry preview."""
import json
import math
import os
import re
import shutil
import threading
import uuid
from copy import deepcopy

from .pointcloud_slam_ros import SlamInputNode
from .pointcloud_slam_stream import SlamStream, _rigid
from utils.ros_lifecycle import dispose_node

_ARCHIVE_DIR = '/models/pointcloud_slam'

def _archive_path(name):
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', name):
        raise ValueError('archive name must contain only letters, digits, _ or -')
    return os.path.join(_ARCHIVE_DIR, name + '.json')


# Project start/stop supply wiring; only the following actions appear in the map menu.
_MAP_ACTIONS = {
    'start_mapping': ('Start mapping', ['map_name']),
    'stop_mapping': ('Stop mapping and save', []),
    'list_maps': ('List maps', []),
    'select_map': ('View saved map', ['map_id']),
    'rename_map': ('Rename saved map', ['map_id', 'map_name']),
    'close_map_view': ('Close map view', ['map_id']),
    'delete_map': ('Delete saved map', ['map_id']),
    'tag_place': ('Tag place', ['map_id', 'revision', 'segment_id', 'tag_name', 'x', 'y', 'yaw', 'tag_id']),
    'list_tags': ('List tags', ['map_id', 'revision']),
    'untag_place': ('Remove tag', ['map_id', 'revision', 'tag_id']),
}

TOOLS = [{'name': 'pointcloud_slam', 'type': 'processor', 'multiInstance': False,
          'description': 'Experimental point-cloud mapping from odometry and depth',
          'inputSchema': {'type': 'object', 'properties': {
              'action': {'type': 'string', 'enum': ['start', 'stop', 'info', 'config']},
              'input_topics': {'type': 'array', 'items': {'type': 'string'}},
              'input_bindings': {'type': 'object'}, 'odom_interfaces': {'type': 'object'},
              'input_metadata': {'type': 'object'}}, 'required': ['action']},
          'configSchema': {'type': 'object', 'properties': {
              'cloud_frame': {'type': 'string', 'description': 'Point cloud frame'},
              'child_frame': {'type': 'string', 'description': 'Pose reference frame, supports {segment_id}'},
              'clock_id': {'type': 'string', 'description': 'Input clock identifier'},
              'clock_relation_verified': {'type': 'boolean', 'default': False, 'description': 'Enable after verifying the clock relation'},
              'extrinsic_calibrated': {'type': 'boolean', 'default': False, 'description': 'Enable after measuring the extrinsic'},
              'child_from_cloud': {'type': 'string', 'title': 'Camera to odometry extrinsic',
                                   'description': '4×4 cloud-to-pose transform with translation in meters'},
              'max_gap_ms': {'type': 'number', 'description': 'Maximum pose interpolation interval (ms)'},
              'max_age_s': {'type': 'number', 'description': 'Receipt timeout (s)'},
              'max_measurement_age_ms': {'type': 'number', 'description': 'Maximum measurement age (ms)'},
              'max_future_skew_ms': {'type': 'number', 'description': 'Allowed future clock skew (ms)'},
              'max_pending': {'type': 'integer', 'description': 'Pending cloud limit'},
              'max_cloud_points': {'type': 'integer', 'description': 'Point limit per cloud'},
              'max_frames': {'type': 'integer', 'description': 'Keyframe limit'},
              'max_points': {'type': 'integer', 'description': 'Total keyframe point limit'},
              'preview_enabled': {'type': 'boolean', 'default': False, 'description': 'Enable experimental odometry preview'},
              'preview_backend': {'type': 'string', 'enum': ['raw', 'gtsam', 'geometry_experimental'], 'description': 'Preview backend'},
              'preview_geometry': {'type': 'string', 'description': 'Experimental geometry parameters (JSON)'},
              'preview_relocalization': {'type': 'boolean', 'default': False, 'description': 'Enable experimental submap relocalization preview'},
              'archive_name': {'type': 'string', 'description': 'Legacy archive filename'},
              'restore_archive': {'type': 'boolean', 'default': False, 'description': 'Restore the legacy archive on startup'},
              'preview_interval_s': {'type': 'number', 'description': 'Preview interval (s)'},
              'preview_max_points': {'type': 'integer', 'description': 'Preview point limit'},
              'preview_odom_sigmas': {'type': 'string', 'description': 'GTSAM six-dimensional noise array (JSON)'},
          }},
          'topic_in': [{'format': 'state/odom'}, {'format': 'sensor/pointcloud2'}],
          'topic_out': [{'format': 'sensor/slam-preview', 'desc': 'Experimental map preview'},
                        {'format': 'data/json', 'desc': 'Status'},
                        {'format': 'data/json', 'desc': 'Experimental map pose'},
                        {'format': 'data/json', 'desc': 'Experimental map to odometry transform'}]}]

for _setting in TOOLS[0]['configSchema']['properties'].values():
    _setting['scope'] = 'instance'


class PointCloudSlamPlugin:
    PREFIX = 'pointcloud_slam'

    def __init__(self, config, executor):
        self._config = dict(config or {})
        self._executor = executor
        self._node = None
        self._topics = {}
        self._resolved_inputs = None
        self._input_diagnostics = []
        self._project_inputs = None
        self._pending = None
        self._map_name = None
        self._save_request = None
        self._last_stop = None
        self._view = None
        self._view_id = None
        self._closed_outputs = []
        self._operation_lock = threading.Lock()
        self._operation = {}
        self._worker = None
        self._session_settings = {}
        self._strategy = {}
        self._lock = threading.RLock()

    def get_tools(self):
        if self._config.get('persistent_history'):
            tools = deepcopy(TOOLS)
            extrinsic = tools[0]['configSchema']['properties']['child_from_cloud']
            if self._config.get('child_from_cloud') is not None:
                extrinsic['default'] = json.dumps(self._config['child_from_cloud'])
            tools[0]['configSchema']['properties'] = {'child_from_cloud': extrinsic}
            schema = tools[0]['inputSchema']
            schema['properties'].update({
              'map_name': {'type': 'string'}, 'map_id': {'type': 'string'},
              'request_id': {'type': 'string'}, 'background': {'type': 'boolean'}, 'revision': {'type': 'integer'},
              'tag_id': {'type': 'string'}, 'tag_name': {'type': 'string'}, 'segment_id': {'type': 'string'},
              'x': {'type': 'number'}, 'y': {'type': 'number'}, 'yaw': {'type': 'number', 'description': 'Radians'}})
            schema['properties']['action']['enum'] += [*_MAP_ACTIONS, 'operation_status']
            schema['x-action-params'] = {action: {'params': params}
                                         for action, (_, params) in _MAP_ACTIONS.items()}
            schema['properties']['map_id']['x-resource-options'] = {
                'action': 'list_maps', 'items': 'maps', 'value': 'id', 'label': 'name'}
            schema['x-map-manager'] = {'schema': 'motus.map-manager/1', 'actions': {
                key: key for key in ('list_maps', 'select_map', 'rename_map', 'delete_map', 'close_map_view')}}
            actions = {key: {'requires_project': key == 'start_mapping'} for key in _MAP_ACTIONS}
            actions['operation_status'] = {'internal': True}
            actions['delete_map']['confirm'] = '删除地图及其全部版本和地点？'
            for key in ('stop_mapping', 'select_map'):
                actions[key].update(background=True, status_action='operation_status',
                                    pending_states=['working'], error_states=['error'], poll_interval_ms=500)
            schema['x-action-lifecycle'] = {'schema': 'motus.action-lifecycle/1',
                                          'config_actions': ['start'], 'actions': actions}
            schema['x-stop'] = {'timeout_s': 300}

            schema['properties']['action']['oneOf'] = [
                {'const': action, 'title': _MAP_ACTIONS[action][0] if action in _MAP_ACTIONS else action}
                for action in schema['properties']['action']['enum']]
            return tools
        if self._config.get('input_policy') == 'experimental':
            tools = deepcopy(TOOLS)
            props = tools[0]['configSchema']['properties']
            tools[0]['configSchema']['properties'] = {
                k: {**props[k], 'default': self._config.get(k, props[k].get('default'))}
                for k in ('preview_enabled', 'archive_name', 'restore_archive')}
            extrinsic = props['child_from_cloud']
            if self._config.get('child_from_cloud') is not None:
                extrinsic['default'] = json.dumps(self._config['child_from_cloud'])
            tools[0]['configSchema']['properties']['child_from_cloud'] = extrinsic
            return tools
        return TOOLS

    def _input_validation(self):
        settings = (self._resolved_inputs or {}).get('settings', self._config)
        return {'policy': self._config.get('input_policy', 'strict'),
                'clock_relation_verified': settings.get('clock_relation_verified') is True,
                'extrinsic_calibrated': settings.get('extrinsic_calibrated') is True}

    def _stream(self):
        keys = ('cloud_frame', 'child_frame', 'clock_id', 'child_from_cloud',
                'max_gap_ms', 'max_age_s', 'max_pending', 'max_cloud_points',
                'max_frames', 'max_points', 'max_measurement_age_ms', 'max_future_skew_ms')
        settings = {**self._config, **(self._resolved_inputs or {}).get('settings', {})}
        return SlamStream(**{key: settings[key] for key in keys},
                          extrinsic_mode=settings.get('extrinsic_mode', 'static'),
                          input_contract=self._resolved_inputs)

    def _geometry_options(self, args):
        options = deepcopy(self._config.get('preview_geometry'))
        if not isinstance(options, dict):
            raise ValueError('explicit geometry settings required')
        from .camera_info import for_topic
        from .pointcloud_slam_observation import resolve_observation, resolve_scoring
        topic = next(t for t, f in args['input_bindings'].items() if f == 'sensor/pointcloud2')
        declaration = deepcopy((self._resolved_inputs or {}).get('inputs', {}).get(topic))
        if declaration is not None:
            declaration['extrinsic'] = deepcopy(self._resolved_inputs['extrinsic'])
        camera = for_topic(args.get('camera_info'), topic)
        settings = {**self._config, **(self._resolved_inputs or {}).get('settings', {})}
        profile = options.get('legacy_profile')
        if declaration is None and profile == 'visible_optical_depth':
            declaration = {'observation_model': 'pinhole', 'source_id': camera.get('id'),
                           'frame_id': settings.get('cloud_frame')}
        if declaration is not None:
            options['observation'] = resolve_observation(declaration, camera, settings['child_from_cloud'])
            if profile is None:
                score = resolve_scoring(options.get('scoring'), options['observation'])
                gates = options.get('thresholds', {})
                if (gates.get('score_contract') != score or 'max_range_stratified_rmse_m' not in gates or
                        'max_forward_depth_score_m' in gates or 'max_rmse_m' in gates):
                    raise ValueError('new score requires matching threshold contract')
        return options

    def start(self):
        pass  # Canvas start supplies the two resolved inputs.

    def _dispose(self, node):
        if node is not None:
            preview = node._preview or {}
            self._closed_outputs.extend(t for t in (preview.get('topic'), preview.get('status_topic')) if t)
            if preview.get('backend') == 'geometry_experimental' and preview.get('topic'):
                self._closed_outputs.extend([preview['topic'] + '/pose', preview['topic'] + '/map_to_odom'])
            self._closed_outputs.extend(node._history_publishers)
            node.stop_preview()
            dispose_node(self._executor, node, label='pointcloud_slam')

    def _close_view(self):
        self._dispose(self._view)
        self._view = self._view_id = None

    @staticmethod
    def _saved_result(entry, saved):
        result = {'state': 'saved', 'map_id': entry['id'], 'map_name': entry['name'],
                  'revision': saved['revision'], 'request_id': saved['request_id']}
        if 'keyframes' in saved:
            result['keyframes'] = saved['keyframes']
        return result

    def _stop_mapping(self, request=None):
        from .pointcloud_slam_library import save_revision
        if self._node is None and self._pending is None:
            return self._last_stop or {'state': 'ready' if self._project_inputs else 'idle'}
        if self._node is not None:
            self._pending = self._node
            self._dispose(self._node)
            self._node = None
            self._save_request = request or uuid.uuid4().hex
        node = self._pending
        with node._lock:
            records = list(node.stream.store.records)
            history = dict(node._history_snapshots)
            archived = getattr(node.stream.store, 'archive_metadata', {})
            self._session_settings['segment_metadata'] = deepcopy(archived.get('segment_metadata', {}))
        if records:
            map_id, revision, _ = save_revision(_ARCHIVE_DIR, records, history, self._session_settings,
                                                self._map_name, None, self._save_request)
            result = {'state': 'saved', 'map_id': map_id, 'map_name': self._map_name,
                      'revision': revision, 'keyframes': len(records), 'request_id': self._save_request}
        else:
            result = {'state': 'ready', 'message': 'No keyframes captured', 'saved': False}
        shutil.rmtree(node.stream.store.directory, ignore_errors=True)
        self._pending = self._map_name = self._save_request = None
        self._last_stop = result
        return result

    def stop(self):
        # A submitted view or save must finish before project teardown.
        with self._operation_lock:
            worker = self._worker
        if worker:
            worker.join()
        with self._lock:
            self._close_view()
            self._project_inputs = None
            self._topics = {}
            if self._config.get('persistent_history'):
                try:
                    result = self._stop_mapping()
                except (ValueError, TypeError, KeyError, OSError) as error:
                    return {'state': 'error', 'message': str(error)}
                return {**result, 'state': 'idle'}
            self._dispose(self._node)
            self._node = None
            return {'state': 'idle'}

    def shutdown(self):
        result = self.stop()
        if result['state'] == 'error':
            raise RuntimeError(result['message'])

    def dispatch(self, name, args):
        result = self._dispatch_action(name, args)
        if isinstance(result, dict):
            with self._lock:
                closed, self._closed_outputs = self._closed_outputs, []
            if closed:
                result = {**result, 'closed_topics': list(dict.fromkeys(closed))}
        return result

    def _dispatch_action(self, name, args):
        action = args.get('action', name)
        if action == 'operation_status':
            request = args.get('request_id')
            with self._operation_lock:
                if request and self._operation.get('request_id') == request:
                    return dict(self._operation)
            from .pointcloud_slam_library import map_entries
            for entry in map_entries(_ARCHIVE_DIR):
                for saved in entry['saves']:
                    if saved['request_id'] == request:
                        return self._saved_result(entry, saved)
            return {'state': 'error', 'message': 'unknown map operation'}
        if args.get('background') is True and action in ('stop_mapping', 'select_map'):
            request = args.get('request_id')
            if not isinstance(request, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', request):
                return {'state': 'error', 'message': 'request id required'}
            with self._operation_lock:
                current = self._operation
                identity = (action, args.get('map_name'), args.get('map_id'))
                if current.get('request_id') == request:
                    if current['identity'] != identity:
                        return {'state': 'error', 'message': 'map request id reused'}
                    if current.get('state') != 'error':
                        return dict(current)
                if current.get('state') == 'working':
                    return {'state': 'error', 'message': 'map operation already in progress'}
                self._operation = {'state': 'working', 'request_id': request, 'identity': identity}
                def run():
                    try:
                        result = self._dispatch(action, {**args, 'background': False})
                    except Exception as error:
                        result = {'state': 'error', 'message': str(error)}
                    with self._operation_lock:
                        self._operation.update(result, request_id=request)
                self._worker = threading.Thread(target=run, name='slam_map_operation', daemon=True)
                self._worker.start()
                return dict(self._operation)
        with self._operation_lock:
            if self._operation.get('state') == 'working' and action not in ('info', 'stop', 'list_maps'):
                return {'state': 'error', 'message': 'map operation already in progress'}
        return self._dispatch(action, args)

    def _dispatch(self, action, args):
        if action in _MAP_ACTIONS:
            if not self._config.get('persistent_history'):
                return {'state': 'error', 'message': 'Named maps require persistent history'}
            with self._lock:
                try:
                    return self._map_action(action, args)
                except (ImportError, KeyError, TypeError, ValueError, RuntimeError, OSError) as error:
                    return {'state': 'error', 'message': str(error)}
        if action == 'config':
            with self._lock:
                if self._node is not None:
                    return {'state': 'error', 'message': 'stop the card before changing configuration'}
                updates = {k: v for k, v in args.items()
                           if k in self.get_tools()[0]['configSchema']['properties'] and v is not None and v != ''}
                try:
                    for key in ('child_from_cloud', 'preview_odom_sigmas', 'preview_geometry'):
                        if isinstance(updates.get(key), str):
                            updates[key] = json.loads(updates[key])
                    if 'child_from_cloud' in updates:
                        updates['child_from_cloud'] = _rigid(updates['child_from_cloud']).tolist()
                    if 'preview_geometry' in updates and not isinstance(updates['preview_geometry'], dict):
                        raise ValueError('preview_geometry must be an object')
                    if 'archive_name' in updates:
                        _archive_path(updates['archive_name'])
                    for key in ('clock_relation_verified', 'extrinsic_calibrated', 'preview_enabled', 'preview_relocalization', 'restore_archive'):
                        if key in updates and type(updates[key]) is not bool:
                            raise ValueError(key + ' must be boolean')
                    for key in ('max_pending', 'max_cloud_points', 'max_frames', 'max_points', 'preview_max_points'):
                        if key in updates and (type(updates[key]) is not int or updates[key] <= 0):
                            raise ValueError(key + ' must be a positive integer')
                    for key in ('max_gap_ms', 'max_age_s', 'max_measurement_age_ms', 'max_future_skew_ms', 'preview_interval_s'):
                        if key in updates:
                            value = updates[key]
                            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                                    not math.isfinite(value) or value < 0 or
                                    (value == 0 and key != 'max_future_skew_ms')):
                                raise ValueError(key + ' has an invalid value')
                    merged = {**self._config, **updates}
                    keys = ('cloud_frame', 'child_frame', 'clock_id', 'child_from_cloud',
                            'max_gap_ms', 'max_age_s', 'max_pending', 'max_cloud_points',
                            'max_frames', 'max_points', 'max_measurement_age_ms', 'max_future_skew_ms')
                    if all(key in merged for key in keys):
                        SlamStream(**{key: merged[key] for key in keys})
                except (ValueError, TypeError) as error:
                    return {'state': 'error', 'message': str(error)}
                self._config.update(updates)
                return {'state': 'configured', 'config': dict(self._config)}
        if action == 'stop':
            return self.stop()
        if action == 'info':
            with self._lock:
                node = self._node
                input_status = node.status() if node else {'odometry': 'waiting'}
                return {'state': ('running' if node else 'save_pending' if self._pending else
                                  'ready' if self._project_inputs else 'idle'),
                        'map_name': self._map_name, 'selected_map_id': self._view_id,
                        'last_stop': self._last_stop,
                        'input_validation': self._input_validation(),
                        'resolved_inputs': deepcopy((self._resolved_inputs or {}).get('inputs', {})),
                        **deepcopy(self._strategy),
                        'input_diagnostics': deepcopy(self._input_diagnostics),
                        'desc': 'Experimental odometry and point-cloud preview',
                        'topic_in': [{'topic': topic, 'format': fmt}
                                     for topic, fmt in self._topics.items()],
                        'topic_out': self._live_outputs() + (self._view.history_topics() if self._view else []),
                        'archive_metadata': deepcopy(getattr(self._view.stream.store, 'archive_metadata', None)) if self._view else None,
                        'input_status': input_status,
                        'last_error': input_status.get('last_error', '')}
        if action != 'start':
            return None
        with self._lock:
            if self._node is not None:
                return {'state': 'error', 'message': 'already running'}
            if self._executor is None:
                return {'state': 'error', 'message': 'ROS executor unavailable'}
            bindings, topics = args.get('input_bindings'), args.get('input_topics')
            if (not isinstance(bindings, dict) or not isinstance(topics, list) or
                    len(topics) != 2 or any(not isinstance(t, str) or not t.strip() for t in topics) or
                    any(not isinstance(f, str) for f in bindings.values()) or
                    len(set(topics)) != 2 or
                    set(bindings) != set(topics) or
                    set(bindings.values()) != {'state/odom', 'sensor/pointcloud2'}):
                return {'state': 'error', 'message': 'two format-resolved odometry and cloud inputs required'}
            odom_topic = next(t for t, f in bindings.items() if f == 'state/odom')
            interfaces = args.get('odom_interfaces')
            interface = interfaces.get(odom_topic) if isinstance(interfaces, dict) else None
            capability = interface.get('slam_pose') if isinstance(interface, dict) else None
            if (not isinstance(interface, dict) or interface.get('schema') != 'motus.odom/1' or
                    not isinstance(capability, dict) or capability.get('available') is not True or
                    capability.get('profile') != 'motus_slam/1'):
                return {'state': 'error', 'message': 'odometry source lacks declared SLAM pose capability'}
            self._resolved_inputs = None
            self._input_diagnostics = []
            if 'input_metadata' in args:
                from .pointcloud_slam_input import resolve_inputs, InputContractError
                try:
                    self._resolved_inputs = resolve_inputs(args['input_metadata'], bindings,
                        experimental=self._config.get('input_policy') == 'experimental')
                except InputContractError as error:
                    self._resolved_inputs = None
                    self._input_diagnostics = error.diagnostics
                    return {'state': 'error', 'message': str(error),
                            'input_diagnostics': deepcopy(self._input_diagnostics)}
            validation = self._input_validation()
            if (self._config.get('input_policy') != 'experimental' and
                    (validation['clock_relation_verified'] is not True or
                     validation['extrinsic_calibrated'] is not True)):
                return {'state': 'error', 'message': 'verified clock relation and measured extrinsic required'}
            if (self._config.get('max_measurement_age_ms') is None or
                    self._config.get('max_future_skew_ms') is None):
                return {'state': 'error', 'message': 'measurement age and clock skew limits required'}
            if (self._config.get('preview_backend') == 'geometry_experimental' and
                    (self._config.get('preview_enabled') is True or self._config.get('persistent_history'))):
                try:
                    self._geometry_options(args)
                except (KeyError, TypeError, ValueError) as error:
                    return {'state': 'error', 'message': str(error)}
            if self._config.get('persistent_history'):
                self._project_inputs = deepcopy(args)
                self._topics = dict(bindings)
                return {'state': 'ready', 'topic_out': self._live_outputs()}
            return self._start_inputs(args)

    def _live_outputs(self):
        if self._node:
            preview = self._node._preview
        elif self._project_inputs:
            cloud = next(t for t, f in self._topics.items() if f == 'sensor/pointcloud2')
            preview = {'topic': cloud + '/slam_odom_preview', 'status_topic': cloud + '/slam_odom_preview_status',
                       'backend': self._config.get('preview_backend')}
        else:
            preview = None
        if not preview:
            return []
        topic = preview['topic']
        plane = self._session_settings.get('map_plane')
        outputs = [{'topic': topic, 'format': 'sensor/slam-preview', 'metadata': {'map_view': {
                        'schema': 'motus.map-view/1', 'scope': 'live', 'map_id': None,
                        'name': self._map_name, 'revision': None, 'segment_id': None,
                        'frame_id': None, 'plane': plane if plane == 'xy' else None, 'actions': {}}}},
                   {'topic': preview['status_topic'], 'format': 'data/json'}]
        if preview['backend'] == 'geometry_experimental':
            outputs.extend([{'topic': topic + '/pose', 'format': 'data/json'},
                            {'topic': topic + '/map_to_odom', 'format': 'data/json'}])
        if self._node:
            outputs.extend(self._node.history_topics())
        return outputs

    def _start_inputs(self, args):
        bindings = args['input_bindings']
        odom_topic = next(t for t, f in bindings.items() if f == 'state/odom')
        session_directory = None
        settings = {**self._config, **(self._resolved_inputs or {}).get('settings', {})}
        try:
            stream = self._stream()
            links, loops, history = [], {}, {}
            persistent = self._config.get('persistent_history')
            if persistent:
                if (self._config.get('preview_backend') != 'geometry_experimental' or
                        self._config.get('preview_relocalization')):
                    raise ValueError('Paged history requires bounded geometry without cross-segment relocalization')
                if self._config.get('restore_archive') is True:
                    from .pointcloud_slam_paged_archive import load_paged_archive
                    stream.store, history = load_paged_archive(_archive_path(self._config.get('archive_name')),
                        settings, max_frames=self._config['max_frames'], max_points=self._config['max_points'])
                    session_directory = stream.store.directory
                else:
                    from .pointcloud_slam_paged_store import PagedKeyframeStore
                    session_directory = os.path.join(_ARCHIVE_DIR, 'sessions', uuid.uuid4().hex)
                    stream.store = PagedKeyframeStore(session_directory,
                        max_frames=self._config['max_frames'], max_points=self._config['max_points'], **persistent)
            elif self._config.get('restore_archive') is True:
                from .pointcloud_slam_archive import load_archive
                stream.store, links, loops = load_archive(_archive_path(self._config.get('archive_name')),
                    settings, max_frames=self._config['max_frames'], max_points=self._config['max_points'])
            cloud_topic = next(t for t, f in bindings.items() if f == 'sensor/pointcloud2')
            preview = None
            if persistent or self._config.get('preview_enabled') is True:
                preview = {'topic': cloud_topic + '/slam_odom_preview',
                           'input_validation': self._input_validation(),
                           'archive_metadata': deepcopy(getattr(stream.store, 'archive_metadata', None)),
                           'status_topic': cloud_topic + '/slam_odom_preview_status',
                           'persistent_history': bool(persistent),
                           'backend': self._config.get('preview_backend', 'raw'),
                           'interval_s': self._config.get('preview_interval_s', 1.),
                           'max_points': self._config.get('preview_max_points', 40000),
                           'sigmas': self._config.get('preview_odom_sigmas'),
                           'geometry': self._geometry_options(args) if self._config.get('preview_backend') == 'geometry_experimental' else None,
                           'relocalization': self._config.get('preview_relocalization', False)}
            node = SlamInputNode(odom_topic, cloud_topic, stream, preview=preview)
            self._strategy = node._loop_search.strategy_snapshot() if node._loop_search else {}
            if self._strategy:
                self._strategy['resource_profile'].update({key: settings.get(key) for key in
                    ('max_frames', 'max_points', 'max_pending', 'max_cloud_points', 'preview_max_points')})
                if persistent:
                    self._strategy['resource_profile'].update(window_frames=settings['max_frames'],
                        window_points=settings['max_points'], persistent_history=deepcopy(persistent))
                settings.update(deepcopy(self._strategy))
            settings['resolved_inputs'] = deepcopy((self._resolved_inputs or {}).get('inputs', {}))
            if preview:
                preview['strategy'] = deepcopy(self._strategy)
            node._segment_links, node._segment_loops = links, loops
            node._history_snapshots = history
            try:
                self._executor.add_node(node)
                node.start_preview()
            except Exception:
                node.stop_preview()
                dispose_node(self._executor, node, label='pointcloud_slam')
                raise
        except (ImportError, KeyError, TypeError, ValueError, RuntimeError, OSError) as error:
            if session_directory:
                shutil.rmtree(session_directory, ignore_errors=True)
            return {'state': 'error', 'message': str(error)}
        self._session_settings = deepcopy(settings)
        self._topics = dict(bindings)
        self._node = node
        return {'state': 'running', 'message': 'Experimental inputs started'}


    def _map_action(self, action, args):
        from . import pointcloud_slam_library as library
        map_id, revision = args.get('map_id'), args.get('revision')
        if action == 'start_mapping':
            if self._node or self._pending:
                raise ValueError('Stop mapping and finish the pending save first')
            if not self._project_inputs:
                raise ValueError('Start the project before mapping')
            name = library.validate_map_name(args.get('map_name'))
            if any(entry['name'] == name for entry in library.map_entries(_ARCHIVE_DIR)):
                raise ValueError('Map name already in use')
            result = self._start_inputs(self._project_inputs)
            if result['state'] == 'running':
                self._map_name = name
                self._last_stop = None
                result.update(map_name=name, topic_out=self._live_outputs())
            return result
        if action == 'stop_mapping':
            request = args.get('request_id')
            if request is not None:
                if not isinstance(request, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', request):
                    raise ValueError('Invalid stop request id')
                for entry in library.map_entries(_ARCHIVE_DIR):
                    for saved in entry['saves']:
                        if saved['request_id'] == request and not self._pending:
                            if self._last_stop and self._last_stop.get('request_id') == request:
                                return self._last_stop
                            return self._saved_result(entry, saved)
            return self._stop_mapping(request)
        if action == 'list_maps':
            return {'state': 'ok', 'maps': [{k: e[k] for k in ('id', 'name', 'revision', 'tag_count')}
                                              for e in library.map_entries(_ARCHIVE_DIR)]}
        if action == 'close_map_view':
            if self._view_id == map_id:
                self._close_view()
            return {'state': 'ok', 'selected_map_id': self._view_id}
        if action == 'select_map':
            from .pointcloud_slam_paged_archive import load_paged_archive
            entry = next((e for e in library.map_entries(_ARCHIVE_DIR) if e['id'] == map_id), None)
            if not entry:
                raise ValueError('Unknown map')
            if self._executor is None:
                raise ValueError('ROS executor unavailable')
            if self._view_id != map_id:
                from .pointcloud_slam_paged_archive import inspect_paged_archive
                path = os.path.join(_ARCHIVE_DIR, 'maps', map_id, str(entry['revision']) + '.json')
                cfg, archived = inspect_paged_archive(path)
                resources = archived.get('resource_profile') or {}
                max_frames, max_points = resources.get('max_frames', 300), resources.get('max_points', 6000000)
                stream = SlamStream(cloud_frame=cfg['cloud_frame'], child_frame=cfg['child_frame'],
                    clock_id=cfg['clock_id'], child_from_cloud=cfg['child_from_cloud'],
                    extrinsic_mode=cfg.get('extrinsic_mode', 'static'),
                    max_gap_ms=200, max_age_s=2, max_pending=1, max_cloud_points=1000000,
                    max_frames=max(2, max_frames), max_points=max_points)
                stream.store, history = load_paged_archive(path, {}, max_frames=max(2, max_frames),
                                                          max_points=max_points, read_only=True)
                preview = {'topic': '/pointcloud_slam/maps/m' + map_id,
                           'status_topic': '/pointcloud_slam/map_view_status', 'backend': 'raw',
                           'persistent_history': True, 'interval_s': 1.,
                           'max_points': cfg.get('preview_max_points', 40000),
                           'input_validation': deepcopy(archived['evidence']),
                           'archive_metadata': archived,
                           'map_view': {'schema': 'motus.map-view/1', 'scope': 'history',
                               'map_id': map_id, 'name': entry['name'], 'revision': entry['revision'],
                               'actions': {key:key for key in ('list_tags','tag_place','untag_place')}}}
                node = SlamInputNode(None, None, stream, preview=preview)
                node._history_snapshots = history
                try:
                    self._executor.add_node(node)
                    node.start_preview()
                except Exception:
                    self._dispose(node)
                    raise
                self._close_view()
                self._view, self._view_id = node, map_id
            return {'state': 'map_selected', 'map_id': map_id, 'map_name': entry['name'],
                    'revision': entry['revision'], 'tracking': False,
                    'archive_metadata': deepcopy(self._view.stream.store.archive_metadata),
                    'topic_out': self._view.history_topics()}
        if action == 'rename_map':
            entry = library.rename_map(_ARCHIVE_DIR, map_id, args.get('map_name'))
            if self._last_stop and self._last_stop.get('map_id') == map_id:
                self._last_stop['map_name'] = entry['name']
            outputs = []
            if self._view_id == map_id:
                self._view._preview['map_view']['name'] = entry['name']
                outputs = self._view.history_topics()
            return {'state': 'renamed', 'map_id': map_id, 'map_name': entry['name'],
                    'revision': entry['revision'], 'topic_out': outputs}
        if action == 'delete_map':
            if self._view_id == map_id:
                self._close_view()
            library.delete_map(_ARCHIVE_DIR, map_id)
            if self._last_stop and self._last_stop.get('map_id') == map_id:
                self._last_stop = None
            return {'state': 'deleted', 'map_id': map_id}
        if action == 'tag_place':
            if library.revision_plane(_ARCHIVE_DIR, map_id, revision, args.get('segment_id')) != 'xy':
                raise ValueError('tag picking requires a declared archived xy plane')
            values = (args.get('tag_name'), args.get('x'), args.get('y'), args.get('yaw'))
            if args.get('tag_id'):
                tag = library.update_tag(_ARCHIVE_DIR, map_id, revision, args['tag_id'], *values,
                                         segment_id=args.get('segment_id'))
            else:
                tag = library.create_tag(_ARCHIVE_DIR, map_id, revision, *values, segment_id=args.get('segment_id'))
            return {'state': 'ok', 'tag': tag}
        if action == 'list_tags':
            return {'state': 'ok', 'tags': library.list_tags(_ARCHIVE_DIR, map_id, revision),
                    'segments': library.revision_segments(_ARCHIVE_DIR, map_id, revision)}
        if action == 'untag_place':
            library.delete_tag(_ARCHIVE_DIR, map_id, revision, args.get('tag_id'))
            return {'state': 'ok'}
