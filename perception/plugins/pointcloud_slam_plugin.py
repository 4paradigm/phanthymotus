"""Opt-in experimental input card with clearly labelled odometry preview."""
import json
import math
import os
import re
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


TOOLS = [{'name': 'pointcloud_slam', 'type': 'processor', 'multiInstance': False,
          'description': '实验性点云与里程计输入；可选未验证里程计预览，尚无真实回环',
          'inputSchema': {'type': 'object', 'properties': {
              'action': {'type': 'string', 'enum': ['start', 'stop', 'info', 'config', 'save_map', 'list_maps', 'load_map', 'unload_map', 'operation_status']},
              'map_name': {'type': 'string'}, 'map_id': {'type': 'string'},
              'request_id': {'type': 'string'}, 'background': {'type': 'boolean'}, 'discard_unsaved': {'type': 'boolean'},
              'input_topics': {'type': 'array', 'items': {'type': 'string'}},
              'input_bindings': {'type': 'object'}, 'odom_interfaces': {'type': 'object'}}, 'required': ['action']},
          'configSchema': {'type': 'object', 'properties': {
              'cloud_frame': {'type': 'string', 'description': '点云坐标系'},
              'child_frame': {'type': 'string', 'description': '位姿参考点坐标系；支持 {segment_id} 模板'},
              'clock_id': {'type': 'string', 'description': '已验证的输入时钟'},
              'clock_relation_verified': {'type': 'boolean', 'default': False, 'description': '仅在独立验证时钟关系后启用'},
              'extrinsic_calibrated': {'type': 'boolean', 'default': False, 'description': '仅在实测外参后启用'},
              'child_from_cloud': {'type': 'string', 'description': '实测 T_child_cloud：4×4 JSON 数组'},
              'max_gap_ms': {'type': 'number', 'description': '最大位姿插值间隔（ms）'},
              'max_age_s': {'type': 'number', 'description': '接收超时（秒）'},
              'max_measurement_age_ms': {'type': 'number', 'description': '最大采集时间年龄（ms）'},
              'max_future_skew_ms': {'type': 'number', 'description': '允许未来时钟偏差（ms）'},
              'max_pending': {'type': 'integer', 'description': '待匹配点云上限'},
              'max_cloud_points': {'type': 'integer', 'description': '单帧点数上限'},
              'max_frames': {'type': 'integer', 'description': '关键帧数量上限'},
              'max_points': {'type': 'integer', 'description': '关键帧总点数上限'},
              'preview_enabled': {'type': 'boolean', 'default': False, 'description': '启用未验证里程计预览'},
              'preview_backend': {'type': 'string', 'enum': ['raw', 'gtsam', 'geometry_experimental'], 'description': '预览后端；几何候选仍未验证'},
              'preview_geometry': {'type': 'string', 'description': '实验性几何检索、配准和图参数的 JSON 对象'},
              'preview_relocalization': {'type': 'boolean', 'default': False, 'description': '实验性双子图重定位预览，结果未经验证'},
              'archive_name': {'type': 'string', 'description': '地图存档名称，仅字母、数字、下划线和连字符'},
              'restore_archive': {'type': 'boolean', 'default': False, 'description': '启动时显式恢复存档，仍须重新定位'},
              'preview_interval_s': {'type': 'number', 'description': '预览间隔（秒）'},
              'preview_max_points': {'type': 'integer', 'description': '预览点数上限'},
              'preview_odom_sigmas': {'type': 'string', 'description': 'GTSAM 六维噪声 JSON 数组'},
          }},
          'topic_in': [{'format': 'state/odom'}, {'format': 'sensor/pointcloud2'}],
          'topic_out': [{'format': 'sensor/slam-preview', 'desc': '未验证地图预览'},
                        {'format': 'data/json', 'desc': '状态'},
                        {'format': 'data/json', 'desc': '实验地图位姿'},
                        {'format': 'data/json', 'desc': '实验 map→odom'}]}]

for _setting in TOOLS[0]['configSchema']['properties'].values():
    _setting['scope'] = 'instance'


class PointCloudSlamPlugin:
    PREFIX = 'pointcloud_slam'

    def __init__(self, config, executor):
        self._config = dict(config or {})
        self._executor = executor
        self._node = None
        self._topics = {}
        self._paused = None
        self._map_id = None
        self._read_only = False
        self._saved_cutoff = None
        self._saving = False
        self._operation_lock = threading.Lock()
        self._operation = {}
        self._session_settings = {}
        self._session_topics = {}
        self._lock = threading.RLock()

    def get_tools(self):
        if self._config.get('persistent_history'):
            tools = deepcopy(TOOLS)
            tools[0]['configSchema']['properties'] = {}
            schema = tools[0]['inputSchema']
            schema['x-action-params'] = {action: {'params': params} for action, params in
                [('save_map', ['map_name']), ('list_maps', []), ('load_map', ['map_name']),
                 ('unload_map', ['discard_unsaved'])]}
            schema['properties']['action']['oneOf'] = [
                {'const': action, 'title': title} for action, title in
                [('save_map', '保存地图'), ('list_maps', '已保存地图'), ('load_map', '打开地图'),
                 ('unload_map', '关闭地图')]]
            return tools
        if self._config.get('input_policy') == 'experimental':
            tools = deepcopy(TOOLS)
            props = tools[0]['configSchema']['properties']
            tools[0]['configSchema']['properties'] = {
                k: {**props[k], 'default': self._config.get(k, props[k].get('default'))}
                for k in ('preview_enabled', 'archive_name', 'restore_archive')}
            return tools
        return TOOLS

    def _input_validation(self):
        return {'policy': self._config.get('input_policy', 'strict'),
                'clock_relation_verified': self._config.get('clock_relation_verified') is True,
                'extrinsic_calibrated': self._config.get('extrinsic_calibrated') is True}

    def _stream(self):
        keys = ('cloud_frame', 'child_frame', 'clock_id', 'child_from_cloud',
                'max_gap_ms', 'max_age_s', 'max_pending', 'max_cloud_points',
                'max_frames', 'max_points', 'max_measurement_age_ms', 'max_future_skew_ms')
        return SlamStream(**{key: self._config[key] for key in keys})

    @staticmethod
    def _cutoff(node):
        return (node.stream.store._frame_count,
                dict(node._history_snapshots))

    def start(self):
        pass  # Canvas start supplies the two resolved inputs.

    def stop(self):
        with self._lock:
            node, self._node = self._node, None
            self._topics = {}
            if self._read_only:
                self._read_only, self._map_id = False, None
            elif node is not None:
                self._paused = node if self._session_settings.get('persistent_history') else None
            if node is not None:
                node.stop_preview()
                dispose_node(self._executor, node, label='pointcloud_slam')

    def shutdown(self):
        self.stop()

    def dispatch(self, name, args):
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
                        return {'state': 'saved', 'map_id': entry['id'], 'revision': saved['revision']}
            return {'state': 'error', 'message': 'unknown map operation'}
        if args.get('background') is True and action in ('save_map', 'load_map'):
            request = args.get('request_id')
            if not isinstance(request, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', request):
                return {'state': 'error', 'message': 'request id required'}
            with self._operation_lock:
                current = self._operation
                identity = (action, args.get('map_name'), args.get('map_id'))
                if current.get('request_id') == request:
                    if current['identity'] != identity:
                        return {'state': 'error', 'message': 'map request id reused'}
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
                        self._operation.update(result)
                threading.Thread(target=run, name='slam_map_operation', daemon=True).start()
                return dict(self._operation)
        with self._operation_lock:
            if self._operation.get('state') == 'working' and action not in ('info', 'stop', 'list_maps'):
                return {'state': 'error', 'message': 'map operation already in progress'}
        return self._dispatch(action, args)

    def _dispatch(self, action, args):
        if action == 'list_maps':
            from .pointcloud_slam_library import map_entries
            return {'state': 'ok', 'maps': [{k: e[k] for k in ('id', 'name', 'revision')}
                                             for e in map_entries(_ARCHIVE_DIR)]}
        if action == 'unload_map':
            with self._lock:
                if self._saving or (self._node is not None and not self._read_only):
                    return {'state': 'error', 'message': 'stop mapping before unloading'}
                if self._paused and self._cutoff(self._paused) != self._saved_cutoff and args.get('discard_unsaved') is not True:
                    return {'state': 'error', 'message': 'save or explicitly discard unsaved frames'}
                self.stop()
                self._paused = self._map_id = self._saved_cutoff = None
                self._read_only = False
                return {'state': 'idle'}
        if action == 'load_map':
            from .pointcloud_slam_library import map_entries
            from .pointcloud_slam_paged_archive import load_paged_archive
            with self._lock:
                if self._saving or (self._node and not self._read_only):
                    return {'state': 'error', 'message': 'stop mapping before opening a map'}
                if self._paused and self._cutoff(self._paused) != self._saved_cutoff:
                    return {'state': 'error', 'message': 'save or discard unsaved frames first'}
                try:
                    entry = next((e for e in map_entries(_ARCHIVE_DIR) if e['id'] == args.get('map_id') or e['name'] == args.get('map_name')), None)
                    if not entry:
                        raise ValueError('unknown map')
                    cfg = self._config
                    path = os.path.join(_ARCHIVE_DIR, 'maps', entry['id'], str(entry['revision']) + '.json')
                    stream = self._stream()
                    stream.store, history = load_paged_archive(path, cfg, max_frames=cfg['max_frames'],
                                                              max_points=cfg['max_points'], read_only=True)
                    preview = {'topic': '/pointcloud_slam/maps/' + 'm' + entry['id'],
                               'status_topic': '/pointcloud_slam/map_view_status', 'backend': 'raw',
                               'persistent_history': True, 'interval_s': 1.,
                               'max_points': cfg.get('preview_max_points', 40000),
                               'input_validation': self._input_validation()}
                    node = SlamInputNode(None, None, stream, preview=preview)
                    node._history_snapshots = history
                    try:
                        self._executor.add_node(node)
                        node.start_preview()
                    except Exception:
                        node.stop_preview()
                        dispose_node(self._executor, node, label='pointcloud_slam')
                        raise
                    if self._read_only:
                        self.stop()
                    self._paused = self._saved_cutoff = None
                    self._node, self._map_id, self._read_only = node, entry['id'], True
                    return {'state': 'map_loaded', 'tracking': False, 'topic_out': node.history_topics()}
                except (KeyError, TypeError, ValueError, RuntimeError, OSError) as error:
                    return {'state': 'error', 'message': str(error)}
        if action == 'save_map' and 'map_name' in args:
            from .pointcloud_slam_library import save_revision
            with self._lock:
                node = self._node or self._paused
                if self._saving or self._read_only or not node or not self._session_settings.get('persistent_history'):
                    return {'state': 'error', 'message': 'no writable paged session or save in progress'}
                with node._lock:
                    records = list(node.stream.store.records)
                    history = dict(node._history_snapshots)
                    cutoff = self._cutoff(node)
                self._saving = True
            try:
                map_id, revision, replay = save_revision(_ARCHIVE_DIR, records, history, self._session_settings,
                    args['map_name'], self._map_id, args.get('request_id', uuid.uuid4().hex))
                with self._lock:
                    if not replay:
                        self._map_id, self._saved_cutoff = map_id, cutoff
                return {'state': 'saved', 'map_id': map_id, 'revision': revision}
            except (ValueError, TypeError, KeyError, OSError) as error:
                return {'state': 'error', 'message': str(error)}
            finally:
                with self._lock:
                    self._saving = False
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
            self.stop()
            return {'state': 'idle'}
        if action == 'save_map':
            with self._lock:
                try:
                    if self._node is None or self._read_only:
                        raise ValueError('no active writable map')
                    path = _archive_path(self._config.get('archive_name'))
                    from .pointcloud_slam_archive import save_archive
                    os.makedirs(_ARCHIVE_DIR, exist_ok=True)
                    with self._node._lock:
                        if self._config.get('persistent_history'):
                            from .pointcloud_slam_paged_archive import save_paged_archive
                            save_paged_archive(path, self._node.stream.store,
                                               self._node._history_snapshots, self._config)
                        else:
                            save_archive(path, self._node.stream.store, self._node._segment_links,
                                         self._node._segment_loops, self._config)
                    return {'state': 'saved', 'path': path}
                except (ValueError, TypeError, OSError, KeyError) as error:
                    return {'state': 'error', 'message': str(error)}
        if action == 'info':
            with self._lock:
                node = self._node
                input_status = node.status() if node else {'odometry': 'waiting'}
                outputs = ([{'topic': node._preview['topic'], 'format': 'sensor/slam-preview'},
                            {'topic': node._preview['status_topic'], 'format': 'data/json'}]
                           if node and node._preview else [])
                if node and node._preview and node._preview['backend'] == 'geometry_experimental':
                    outputs += [{'topic': node._preview['topic'] + '/pose', 'format': 'data/json'},
                                {'topic': node._preview['topic'] + '/map_to_odom', 'format': 'data/json'}]
                if node:
                    if self._read_only:
                        outputs = node.history_topics()
                    else:
                        outputs += node.history_topics()
                return {'state': ('map_loaded' if self._read_only else 'running' if node else 'paused' if self._paused else 'idle'),
                        'map_id': self._map_id,
                        'input_validation': self._input_validation(),
                        'desc': '实验输入与未验证里程计预览；无真实回环',
                        'topic_in': [{'topic': topic, 'format': fmt}
                                     for topic, fmt in self._topics.items()],
                        'topic_out': outputs,
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
            if (self._config.get('input_policy') != 'experimental' and
                    (self._config.get('clock_relation_verified') is not True or
                     self._config.get('extrinsic_calibrated') is not True)):
                return {'state': 'error', 'message': 'verified clock relation and measured extrinsic required'}
            if (self._config.get('max_measurement_age_ms') is None or
                    self._config.get('max_future_skew_ms') is None):
                return {'state': 'error', 'message': 'measurement age and clock skew limits required'}
            try:
                stream = self._stream()
                links, loops, history = [], {}, {}
                persistent = self._config.get('persistent_history')
                if persistent and (self._config.get('preview_backend') != 'geometry_experimental' or
                                   self._config.get('preview_relocalization')):
                    raise ValueError('paged history requires bounded geometry without cross-segment relocalization')
                if self._read_only or self._saving:
                    raise ValueError('unload the read-only map or wait for the save')
                if self._paused:
                    if self._session_settings != self._config or self._session_topics != bindings:
                        raise ValueError('unload the previous map before changing its inputs or settings')
                    stream.store = self._paused.stream.store
                    history = self._paused._history_snapshots
                elif self._config.get('restore_archive') is True:
                    if persistent:
                        from .pointcloud_slam_paged_archive import load_paged_archive
                        stream.store, history = load_paged_archive(_archive_path(self._config.get('archive_name')),
                            self._config, max_frames=self._config['max_frames'], max_points=self._config['max_points'])
                    else:
                        from .pointcloud_slam_archive import load_archive
                        stream.store, links, loops = load_archive(_archive_path(self._config.get('archive_name')),
                            self._config, max_frames=self._config['max_frames'], max_points=self._config['max_points'])
                elif persistent:
                    from .pointcloud_slam_paged_store import PagedKeyframeStore
                    stream.store = PagedKeyframeStore(os.path.join(_ARCHIVE_DIR, 'sessions', uuid.uuid4().hex),
                        max_frames=self._config['max_frames'], max_points=self._config['max_points'], **persistent)
                cloud_topic = next(t for t, f in bindings.items() if f == 'sensor/pointcloud2')
                preview = None
                if persistent or self._config.get('preview_enabled') is True:
                    preview = {'topic': cloud_topic + '/slam_odom_preview',
                               'input_validation': self._input_validation(),
                               'status_topic': cloud_topic + '/slam_odom_preview_status',
                               'persistent_history': bool(persistent),
                               'backend': self._config.get('preview_backend', 'raw'),
                               'interval_s': self._config.get('preview_interval_s', 1.),
                               'max_points': self._config.get('preview_max_points', 40000),
                               'sigmas': self._config.get('preview_odom_sigmas'),
                               'geometry': self._config.get('preview_geometry'),
                               'relocalization': self._config.get('preview_relocalization', False)}
                node = SlamInputNode(odom_topic, cloud_topic, stream, preview=preview)
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
                return {'state': 'error', 'message': str(error)}
            self._paused = None
            self._session_settings = deepcopy(self._config)
            self._session_topics = dict(bindings)
            self._topics = dict(bindings)
            self._node = node
            return {'state': 'running', 'message': '实验输入已启动；预览未验证，无真实回环'}
