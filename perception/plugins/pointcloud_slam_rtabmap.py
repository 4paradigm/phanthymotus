"""Supervise the pinned headless RTAB-Map ROS node and its native SQLite DB."""
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess


def resolve_executable():
    try:
        from ament_index_python.packages import get_package_prefix
        executable = Path(get_package_prefix('rtabmap_slam')) / 'lib/rtabmap_slam/rtabmap'
    except (ImportError, LookupError) as error:
        raise RuntimeError('RTAB-Map runtime unavailable: install the pinned rtabmap_slam ROS package') from error
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise RuntimeError('RTAB-Map rtabmap_slam executable unavailable: ' + str(executable))
    return executable


def require_runtime():
    """Check dependencies without creating a ROS node, database or process."""
    resolve_executable()
    try:
        from nav_msgs.msg import Odometry  # noqa: F401
        from geometry_msgs.msg import TransformStamped  # noqa: F401
        from tf2_msgs.msg import TFMessage  # noqa: F401
        from rtabmap_msgs.msg import Info, MapGraph  # noqa: F401
    except ImportError as error:
        raise RuntimeError('RTAB-Map ROS message dependencies unavailable: ' + str(error)) from error


def mapping_parameters(database_path, namespace, settings):
    """Production scan-only parameters, shared with the real-image smoke gate.

    settings supplies max_pending, voxel_size_m, max_working_nodes and
    max_update_ms. RTAB-Map algorithm parameters are strings; ROS node
    parameters retain their native bool/int/float types.
    """
    frame = namespace.rstrip('/').lstrip('/')
    return {'database_path': str(Path(database_path).resolve()),
        'frame_id': frame + '/base', 'map_frame_id': frame + '/map', 'odom_frame_id': '',
        'publish_tf': False, 'subscribe_odom': True, 'subscribe_scan_cloud': True,
        'subscribe_scan': False, 'subscribe_depth': False, 'subscribe_rgb': False,
        'subscribe_rgbd': False, 'subscribe_stereo': False, 'subscribe_odom_info': False,
        'approx_sync': False, 'topic_queue_size': settings['max_pending'],
        'sync_queue_size': settings['max_pending'], 'qos_odom': 1, 'qos_scan': 1,
        'wait_for_transform': .2, 'latch': True,
        'Reg/Strategy': '1', 'Icp/Strategy': '0', 'Optimizer/Strategy': '0',
        'Icp/VoxelSize': str(settings['voxel_size_m']), 'Grid/Sensor': '0',
        'Grid/CellSize': str(settings['voxel_size_m']), 'Grid/3D': 'true',
        'RGBD/ProximityBySpace': 'true', 'RGBD/NeighborLinkRefining': 'true',
        'Rtabmap/MemoryThr': str(settings['max_working_nodes']),
        'Rtabmap/TimeThr': str(settings['max_update_ms']),
        'DbSqlite3/JournalMode': '0', 'DbSqlite3/Synchronous': '1'}


class RtabmapProcess:
    """One external VO segment owns one process and one fresh native database.

    SIGINT is sent once: RTAB-Map saves memory during its node destructor.
    A timeout or abnormal exit leaves the session directory intact for retry
    or recovery and can never be mistaken for a saved map.
    """
    def __init__(self, database_path, namespace, settings, *, executable=None):
        self.database_path = Path(database_path).resolve()
        self.parameter_path = self.database_path.with_suffix('.params.json')
        self.namespace = namespace.rstrip('/')
        self._settings = dict(settings)
        self._executable = executable
        self._process = None
        self._log = None
        self._stop_sent = False
        self._closed = False
        self.mapped_nodes = None
        self._forced_stop = False

    def start(self):
        if self._process is not None:
            raise RuntimeError('RTAB-Map process already started')
        executable = self._executable or resolve_executable()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        if self.database_path.exists():
            raise RuntimeError('Refusing to overwrite native RTAB-Map database')
        params = mapping_parameters(self.database_path, self.namespace, self._settings)
        self.parameter_path.write_text(json.dumps({'/**': {'ros__parameters': params}}))
        self._log = self.database_path.with_suffix('.log').open('ab', buffering=0)
        command = [str(executable), '--ros-args', '--params-file', str(self.parameter_path),
            '-r', '__ns:=' + self.namespace, '-r', '__node:=rtabmap',
            '-r', '/tf:=' + self.namespace + '/tf',
            '-r', '/tf_static:=' + self.namespace + '/tf_static']
        try:
            self._process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                stdout=self._log, stderr=subprocess.STDOUT, start_new_session=True)
        except Exception:
            self._log.close()
            raise

    def poll(self):
        return self._process.poll() if self._process is not None else None

    def close(self, *, timeout_s=60.):
        if self._closed:
            return self.database_path
        if self._process is None:
            raise RuntimeError('RTAB-Map process was not started')
        if self._forced_stop:
            if self._process.poll() is None:
                self._process.kill()
                self._process.wait(timeout=1.)
            raise RuntimeError('RTAB-Map required a forced stop; native database retained for recovery')
        code = self._process.poll()
        if code is not None and not self._stop_sent:
            if self._log is not None:
                self._log.close()
            raise RuntimeError('RTAB-Map exited before graceful stop (exit ' + str(code) + ')')
        if code is None and not self._stop_sent:
            self._stop_sent = True
            try:
                os.killpg(self._process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
        try:
            code = self._process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired as error:
            self._forced_stop = True
            for stop_signal in (signal.SIGTERM, signal.SIGKILL):
                if self._process.poll() is not None:
                    break
                try:
                    os.killpg(self._process.pid, stop_signal)
                except ProcessLookupError:
                    pass
                try:
                    self._process.wait(timeout=1.)
                except subprocess.TimeoutExpired:
                    continue
            if self._log is not None:
                self._log.close()
            raise RuntimeError('RTAB-Map graceful stop timed out; process forced to stop, native database retained') from error
        if self._log is not None:
            self._log.close()
        if code != 0:
            raise RuntimeError('RTAB-Map failed to finalize database (exit ' + str(code) + ')')
        try:
            if not self.database_path.is_file() or not self.database_path.stat().st_size:
                raise ValueError('native database is missing or empty')
            with sqlite3.connect(self.database_path.as_uri() + '?mode=ro', uri=True) as db:
                if db.execute('PRAGMA quick_check').fetchone() != ('ok',):
                    raise ValueError('native database integrity check failed')
                self.mapped_nodes = db.execute('SELECT COUNT(*) FROM Node').fetchone()[0]
        except (OSError, sqlite3.Error, ValueError) as error:
            raise RuntimeError('RTAB-Map database validation failed: ' + str(error)) from error
        self._closed = True
        return self.database_path
