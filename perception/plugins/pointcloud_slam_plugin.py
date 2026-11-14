"""Point-cloud mapping card backed by the native RTAB-Map process."""
from copy import deepcopy
import logging
import json
import math
from pathlib import Path
import re
import shutil
import threading
import uuid

from .pointcloud_slam_config import discover_runtime_geometry, reconcile_explicit_extrinsic
from .pointcloud_slam_contract import SpatialSlamError, validate_slam_capability
from .pointcloud_slam_input import InputContractError, resolve_inputs
from .pointcloud_slam_preview import PREVIEW_TOPIC
from .pointcloud_slam_ros import RtabmapNode, discover_input_bindings
from .pointcloud_slam_rtabmap import require_runtime
from .pointcloud_slam_transforms import rigid
from .pointcloud_slam_view import SavedMapView
from utils.ros_lifecycle import dispose_node

log = logging.getLogger(__name__)
_ARCHIVE_DIR = '/models/pointcloud_slam'
_DEFAULTS = {'max_gap_ms': 200, 'max_age_s': 2,
             'max_measurement_age_ms': 1500, 'max_future_skew_ms': 100,
             'max_pending': 10, 'max_cloud_points': 1000000,
             'voxel_size_m': .08, 'max_working_nodes': 3000, 'max_update_ms': 3000,
             'max_disk_bytes': 2147483648, 'reserve_bytes': 1073741824,
             'preview_interval_s': 1, 'preview_max_points': 40000}
_RETIRED = {'robot', 'input_policy', 'clock_relation_verified', 'extrinsic_calibrated',
            'preview_enabled', 'preview_backend', 'preview_geometry', 'preview_relocalization',
            'preview_odom_sigmas', 'archive_name', 'restore_archive',
            'max_frames', 'max_points', 'persistent_history'}
_ACTIONS = {
    'start_mapping': ['map_name'],
    'stop_mapping': [],
    'abandon_pending': [],
    'list_maps': [],
    'select_map': ['map_id'],
    'rename_map': ['map_id', 'map_name'],
    'delete_map': ['map_id'],
}


def _config_properties():
    """Only the optional manual mount is configurable on the canvas.

    Frames and clocks come from connected publishers, not a model selector.
    Algorithm budgets are deployment configuration, not hardware-model presets.
    """
    return {
        'child_from_cloud': {'type': 'string', 'scope': 'instance',
                             'title': 'Cloud to odometry reference transform',
                             'description': 'Rigid 4x4 transform, meters (JSON)'},
    }


class PointCloudSlamPlugin:
    PREFIX = 'pointcloud_slam'

    def __init__(self, config, executor):
        # 标点移至后续导航卡片；忽略旧画布中的标注平面，不阻止项目启动。
        # Retired/unknown deployment keys are ignored, never fatal: old config
        # rows and old canvas documents must not block a project start. The
        # filtered deployment config is kept separately so config-action
        # updates re-base on it instead of on stale canvas fields.
        deployment = {k: v for k, v in (config or {}).items()
                      if k not in _RETIRED and k != 'map_plane'}
        self._deployment_config = deepcopy(deployment)
        self._config = {**_DEFAULTS, **deployment}
        self._executor = executor
        self._node = self._pending = self._view = None
        self._view_id = self._session_dir = self._map_name = self._save_request = None
        self._project_inputs = None
        self._topics = {}
        self._resolved_inputs = None
        self._runtime_inputs = self._runtime_provenance = None
        self._input_diagnostics = []
        self._legacy_core = True
        self._session_settings = {}
        self._last_stop = None
        self._last_recovery = None
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

    def get_tools(self):
        props = _config_properties()
        for key, value in self._settings().items():
            if key in props and value is not None:
                props[key]['default'] = json.dumps(value) if key == 'child_from_cloud' else value
        verbs = ['start', 'stop', 'info', 'config', *_ACTIONS, 'operation_status']
        actions = {key: {'requires_project': key == 'start_mapping'} for key in _ACTIONS}
        actions['operation_status'] = {'internal': True}
        actions['delete_map']['confirm'] = '删除地图及其全部版本？'
        actions['abandon_pending']['confirm'] = '保存已经失败。解除阻塞并保留原始会话文件供人工恢复？该地图不会列入已保存地图。'
        for key in ('stop_mapping', 'select_map'):
            actions[key].update(background=True, status_action='operation_status',
                pending_states=['working'], error_states=['error'], poll_interval_ms=500)
        schema = {'type': 'object', 'properties': {
            'action': {'type': 'string', 'enum': verbs,
                       'oneOf': [{'const': k, 'title': k} for k in verbs]},
            'input_topics': {'type': 'array', 'items': {'type': 'string'}},
            'input_bindings': {'type': 'object'}, 'odom_interfaces': {'type': 'object'},
            'input_metadata': {'type': 'object'},
            **{key: {'type': 'string'} for key in ('map_id', 'map_name', 'request_id')},
            'background': {'type': 'boolean'}, 'revision': {'type': 'integer'}}, 'required': ['action'],
            'x-action-params': {k: {'params': params} for k, params in _ACTIONS.items()},
            'x-slam-input-contract': 'motus.slam_input/1',
            'x-map-manager': {'schema': 'motus.map-manager/1', 'actions': {
                k: k for k in ('list_maps', 'select_map', 'rename_map', 'delete_map')}},
            'x-action-lifecycle': {'schema': 'motus.action-lifecycle/1', 'config_actions': ['start'], 'actions': actions}}
        schema['properties']['map_id']['x-resource-options'] = {
            'action': 'list_maps', 'items': 'maps', 'value': 'id', 'label': 'name'}
        return [{'name': self.PREFIX, 'type': 'processor', 'multiInstance': False,
                 'description': 'RTAB-Map point-cloud mapping with existing odometry',
                 'inputSchema': schema, 'configSchema': {'type': 'object', 'properties': props},
                 'topic_in': [{'format': 'state/odom'}, {'format': 'sensor/pointcloud2'}],
                 'topic_out': self._outputs()}]

    def start(self):
        pass  # Canvas start supplies the connected native inputs.

    def _outputs(self):
        # Stable even before project start and after it stops. The saved-map
        # viewer and native mapping node take turns publishing this one topic.
        return [{'topic': PREVIEW_TOPIC, 'format': 'sensor/mapping',
                 'desc': 'Live or saved RTAB-Map preview'}]

    def _dispose(self, node):
        if node is None:
            return
        node.close()
        dispose_node(self._executor, node, label=self.PREFIX)

    def _close_view(self):
        view, self._view = self._view, None
        self._view_id = None
        self._dispose(view)

    def _settings(self):
        return {**self._config, **(self._runtime_inputs or {}), **(self._resolved_inputs or {}).get('settings', {})}

    def _validate(self, cfg):
        """Validate the internal settings that are present; never require facts.

        Canvas configuration may be partial: frames/clock/extrinsic arrive from
        live connected inputs, deployment config or producer declarations; their
        absence is reported when mapping starts instead of while saving config.
        """
        if cfg.get('extrinsic_mode', 'static') not in ('static', 'dynamic'):
            raise ValueError('extrinsic_mode must be static or dynamic')
        if cfg.get('extrinsic_mode', 'static') != 'dynamic' and cfg.get('child_from_cloud') is not None:
            rigid(cfg['child_from_cloud'])
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

    def _require_mapping_inputs(self, cfg):
        """The mapping session needs concrete frames and an extrinsic; fail clearly at start."""
        for key in ('cloud_frame', 'child_frame', 'clock_id'):
            value = cfg.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(key + ' is required before mapping starts: connect capture-timed '
                                 'PointCloud2 and tracking motus.odom/1 or supply explicit metadata')
        if cfg.get('extrinsic_mode', 'static') != 'dynamic' and cfg.get('child_from_cloud') is None:
            raise ValueError('child_from_cloud is required before mapping starts: supply the '
                             'measured card transform or a calibrated upstream mount')

    def dispatch(self, name, args):
        try:
            return self._dispatch_action(args.get('action', name), args)
        except (ImportError, ValueError, TypeError, KeyError, OSError, RuntimeError) as error:
            return {'state': 'error', 'message': str(error)}

    def _dispatch_action(self, action, args):
        if action == 'operation_status':
            request = args.get('request_id')
            with self._operation_lock:
                for operation in (self._operation, self._save_status):
                    if request and operation.get('request_id') == request:
                        return dict(operation, state='working' if operation['state'] == 'saving' else operation['state'])
            if request and self._last_recovery and self._last_recovery.get('request_id') == request:
                return {'state': 'error', 'request_id': request,
                        'message': 'Map save failed; session retained at ' + self._last_recovery['path']}
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
        if action == 'start':
            return self._prepare_after_prior_mapping(args)
        if action == 'abandon_pending':
            return self._abandon_pending()
        if action == 'stop_mapping':
            return self._stop_mapping(args.get('request_id'))
        if action == 'info':
            with self._operation_lock:
                save = deepcopy(self._save_status)
                stopping = self._project_stopping or self._blocking_stops
            if stopping:
                return {'state': 'idle', 'capture_stopped': True, 'backend': 'rtabmap',
                        'save_state': save['state'], 'save_result': save,
                        'last_recovery': deepcopy(self._last_recovery),
                        'topic_in': [], 'topic_out': self._outputs()}
            with self._lock:
                return {'state': 'idle' if self._capture_stop.is_set() else
                        'running' if self._node else 'save_pending' if self._pending else
                        'ready' if self._project_inputs else 'idle', 'backend': 'rtabmap',
                        'capture_stopped': self._capture_stop.is_set(), 'map_name': self._map_name,
                        'selected_map_id': self._view_id, 'last_stop': deepcopy(self._last_stop),
                        'save_state': save['state'], 'save_result': save,
                        'last_recovery': deepcopy(self._last_recovery),
                        'input_compatibility': {'mode': 'legacy_core' if self._legacy_core else 'explicit',
                            'role_source': 'not_resolved' if not self._topics else
                                'native_ros_publishers' if self._legacy_core else 'input_bindings',
                            'producer_metadata_provided': self._resolved_inputs is not None},
                        'input_diagnostics': deepcopy(self._input_diagnostics),
                        'input_provenance': deepcopy(self._runtime_provenance),
                        'topic_in': [{'topic': t, 'format': f} for t, f in self._topics.items()],
                        'topic_out': self._outputs(),
                        'input_status': self._node.status() if self._node else {'odometry': 'waiting'}}
        with self._lock:
            if action == 'config':
                mapping = self._node is not None or self._pending is not None
                # Only the one canvas field is configurable; every other key
                # (stale canvas rows, including a saved `robot`, and old config
                # rows) is ignored rather than fatal. A legacy robot selector is ignored rather than used to choose
                # any robot-specific SLAM behavior.
                props = _config_properties()
                updates = {k: v for k, v in args.items() if k in props and v is not None and v != ''}
                merged = {**_DEFAULTS, **self._deployment_config, **updates}
                if isinstance(merged.get('child_from_cloud'), str):
                    merged['child_from_cloud'] = json.loads(merged['child_from_cloud'])
                if merged.get('child_from_cloud') is not None:
                    merged['child_from_cloud'] = rigid(merged['child_from_cloud']).tolist()
                self._validate({**merged, **(self._resolved_inputs or {}).get('settings', {})})
                if mapping:
                    if merged != self._config:
                        raise ValueError('stop mapping before changing configuration')
                    return {'state': 'configured', 'config': dict(self._config)}
                self._config = merged
                self._runtime_inputs = self._runtime_provenance = None
                return {'state': 'configured', 'config': dict(self._config)}
            if action == 'start_mapping':
                return self._start_mapping(args)
            return self._map_action(action, args)

    def _prepare_after_prior_mapping(self, args):
        # Intelligent control can restart without issuing the previous card's
        # stop action. A new project start must finalize that previous session,
        # not leave the canvas permanently blocked on the old node/pending map.
        with self._lock:
            previous = self._node or self._pending
        if previous is not None:
            log.warning('[pointcloud_slam] new project start finalizing previous map session')
            try:
                self._stop_mapping()  # commits native DB; never discards it
            except Exception as error:
                raise RuntimeError('previous map save failed: ' + str(error) +
                                   '; retry stop_mapping or use abandon_pending '
                                   'to preserve recovery files and release the card') from error
        with self._lock:
            return self._prepare(args)

    def _prepare(self, args):
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
        self._resolved_inputs = None
        self._runtime_inputs = self._runtime_provenance = None
        self._input_diagnostics = []

        if 'odom_interfaces' not in args:
            # No declaration is recoverable while an upstream producer starts.
            # Actual mapping still requires valid motus_slam/1 tracking samples.
            self._input_diagnostics.append({
                'topic': odom, 'field': 'odom_interface',
                'reason': 'SLAM capability declaration is missing; waiting for upstream info()'})
        else:
            interfaces = args['odom_interfaces']
            if not isinstance(interfaces, dict):
                raise ValueError('odom_interfaces must be a topic-keyed object')
            if odom not in interfaces:
                self._input_diagnostics.append({
                    'topic': odom, 'field': 'odom_interface',
                    'reason': 'SLAM capability declaration is missing for the connected odometry topic'})
            else:
                try:
                    validate_slam_capability(interfaces[odom])
                except SpatialSlamError as error:
                    raise ValueError(f'{odom}: invalid SLAM odometry capability: {error}') from error

        if 'input_metadata' in args:
            provided = args.get('input_metadata')
            # Core calls info() while starting an upstream camera/VO. A missing
            # producer declaration is not a contradictory claim. In particular,
            # collect_slam_inputs() may legitimately supply an empty dict when
            # both ROS topics exist but the first tracking sample does not.
            if (not isinstance(provided, dict) or
                    any(not isinstance(provided.get(topic), dict) for topic in bindings)):
                missing = [topic for topic in bindings if not isinstance(
                    provided.get(topic) if isinstance(provided, dict) else None, dict)]
                self._input_diagnostics.append({
                    'topic': ', '.join(missing), 'field': 'input_metadata',
                    'reason': 'waiting for producer descriptions; runtime geometry will be probed'})
            else:
                try:
                    self._resolved_inputs = resolve_inputs(provided, bindings)
                    reconcile_explicit_extrinsic(self._config.get('child_from_cloud'),
                        self._resolved_inputs['settings'].get('child_from_cloud'))
                except InputContractError as error:
                    self._input_diagnostics.extend(deepcopy(error.diagnostics))
                    # Missing startup information is recoverable; a producer
                    # explicitly declaring a wrong format, clock or incompatible
                    # mount is not. Avoid treating a *derived* mismatch as
                    # evidence when the source has not reported its frame yet.
                    odom_decl = provided.get(odom) or {}
                    cloud_topic = next(t for t, f in bindings.items() if f == 'sensor/pointcloud2')
                    cloud_decl = provided.get(cloud_topic) or {}
                    clock_known = all(isinstance(d.get('clock_id'), str) and
                                      d['clock_id'] and d['clock_id'] != 'unknown'
                                      for d in (odom_decl, cloud_decl))
                    frames_known = (isinstance(odom_decl.get('child_frame_id'), str) and
                                    bool(odom_decl['child_frame_id'].strip()) and
                                    isinstance(cloud_decl.get('frame_id'), str) and
                                    bool(cloud_decl['frame_id'].strip()))
                    sources_known = all(isinstance(d.get('source_id'), str) and
                                        bool(d['source_id'].strip()) and
                                        d['source_id'] != 'unknown'
                                        for d in (odom_decl, cloud_decl))
                    fatal_fields = {'schema', 'topic/format', 'kind',
                        'cloud_semantics', 'timing_model', 'observation_model',
                        'extrinsic.mode', 'extrinsic.pose_reference'}
                    fatal = any(
                        d.get('field') in fatal_fields or
                        'conflicting' in d.get('reason', '') or
                        (d.get('field') == 'clock.relation_id' and sources_known) or
                        (d.get('field') == 'clock_id' and
                         'differs' in d.get('reason', '') and clock_known) or
                        (d.get('field') == 'extrinsic' and
                         'must explicitly map' in d.get('reason', '') and frames_known)
                        for d in error.diagnostics)
                    if fatal:
                        raise
                except ValueError:
                    # A manual transform conflicting with upstream geometry is
                    # a real configuration conflict, never silently overridden.
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
        if self._node or self._pending:
            raise ValueError('finish the pending map save first')
        if not self._project_inputs or self._capture_stop.is_set():
            raise ValueError('start the project before mapping')
        # Startup declarations may be incomplete. Learn missing geometry from
        # fresh ROS pointcloud and tracking odometry, never a model preset.
        if self._resolved_inputs is None and any(
                self._config.get(key) is None for key in
                ('cloud_frame', 'child_frame', 'clock_id', 'child_from_cloud')):
            odom_topic = next(t for t, f in self._topics.items() if f == 'state/odom')
            cloud_topic = next(t for t, f in self._topics.items() if f == 'sensor/pointcloud2')
            self._runtime_inputs, self._runtime_provenance = discover_runtime_geometry(
                self._executor, odom_topic, cloud_topic,
                manual=self._config.get('child_from_cloud'),
                max_gap_ms=self._config['max_gap_ms'])
        self._require_mapping_inputs(self._settings())
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
        settings['input_provenance'] = deepcopy(self._runtime_provenance)
        node = None
        try:
            node = RtabmapNode(next(t for t, f in self._topics.items() if f == 'state/odom'),
                next(t for t, f in self._topics.items() if f == 'sensor/pointcloud2'), settings,
                session_dir=str(session), input_contract=self._resolved_inputs, capture_stop=self._capture_stop)
            # A new mapping session replaces the saved view, never competes for
            # the single mapping topic.
            self._close_view()
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

    def _abandon_pending(self):
        """Release an unrecoverable failed save, preserving its raw session.

        Explicit confirmation is required: this intentionally does not claim
        the map is saved and never deletes a native database.
        """
        with self._save_lock:
            with self._operation_lock:
                if (self._save_status.get('state') != 'error' or
                        self._project_stopping or self._blocking_stops or
                        self._operation.get('state') == 'working'):
                    raise ValueError('abandon_pending requires a failed, completed map save')
            with self._lock:
                pending = self._pending
                if pending is None or self._node is not None:
                    raise ValueError('no failed pending map to release')
                for segment in getattr(pending, '_segments', []):
                    process = segment.get('process')
                    if process is not None and process.poll() is None:
                        raise ValueError('native mapping process is still alive; retry stop_mapping first')
                pending.freeze_capture()
                recovery = {'path': str(self._session_dir), 'map_name': self._map_name,
                            'request_id': self._save_request,
                            'error': self._save_status.get('message')}
                self._last_recovery = recovery
                self._pending = self._session_dir = self._map_name = self._save_request = None
            dispose_node(self._executor, pending, label=self.PREFIX)
            with self._operation_lock:
                self._save_status = {'state': 'idle'}
        log.warning('[pointcloud_slam] failed map retained for recovery at %s', recovery['path'])
        return {'state': 'recovery_required', 'recovery': recovery,
                'message': 'Failed map save was released; raw files are preserved for recovery'}

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
            if self._node is not None or self._pending is not None:
                raise ValueError('stop mapping and finish saving before viewing a saved map')
            entry = next((e for e in library.map_entries(_ARCHIVE_DIR) if e['id'] == map_id), None)
            if entry is None:
                raise ValueError('unknown map')
            revision = revision or entry['revision']
            if self._view_id != (map_id, revision):
                snapshots = library.load_map_snapshots(_ARCHIVE_DIR, map_id, revision,
                    max_points=self._config['preview_max_points'])
                if not snapshots:
                    raise ValueError('saved map has no viewable snapshot')
                view = SavedMapView(map_id, revision, entry['name'], snapshots,
                                    interval_s=self._config['preview_interval_s'])
                # A new publisher must not start before the old one is closed.
                self._close_view()
                try:
                    self._executor.add_node(view)
                    view.start()
                except Exception:
                    self._dispose(view)
                    raise
                self._view, self._view_id = view, (map_id, revision)
            return {'state': 'map_selected', 'map_id': map_id, 'map_name': entry['name'],
                    'revision': revision, 'tracking': False, 'snapshot_available': True,
                    'preview_segment_id': self._view.selected_segment_id,
                    'preview_segment_count': self._view.segment_count,
                    'topic_out': self._outputs()}
        if action == 'rename_map':
            entry = library.rename_map(_ARCHIVE_DIR, map_id, args.get('map_name'))
            if self._last_stop and self._last_stop.get('map_id') == map_id:
                self._last_stop['map_name'] = entry['name']
            if self._view_id and self._view_id[0] == map_id:
                self._view.rename(entry['name'])
            return {'state': 'renamed', 'map_id': map_id, 'map_name': entry['name'],
                    'topic_out': self._outputs()}
        if action == 'delete_map':
            if self._view_id and self._view_id[0] == map_id:
                self._close_view()
            library.delete_map(_ARCHIVE_DIR, map_id)
            if self._last_stop and self._last_stop.get('map_id') == map_id:
                self._last_stop = None
            return {'state': 'deleted', 'map_id': map_id}
        return None
