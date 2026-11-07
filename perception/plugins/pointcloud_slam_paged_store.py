"""Durable local keyframes; paging does not establish global SLAM correctness."""
from dataclasses import asdict
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

import numpy as np

from .pointcloud_slam_keyframes import CapacityError, KeyframeStore


class StorageError(CapacityError):
    """Stop fusion after a persistence failure without changing source tracking."""


class DiskPoints:
    """Open one immutable file per read; never retain a map/FD for all history."""
    def __init__(self, path, count):
        self.path, self.count = Path(path), count
        self.shape = (count, 3)

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        try:
            data = np.load(self.path, mmap_mode='r', allow_pickle=False)
            try:
                if data.shape != self.shape or data.dtype != np.float64:
                    raise ValueError('point file shape/type mismatch')
                return np.array(data[index], copy=True)
            finally:
                data._mmap.close()
        except (OSError, ValueError) as error:
            raise StorageError('map point file unreadable: ' + str(error)) from error

    def __array__(self, dtype=None):
        return np.asarray(self[:], dtype=dtype)

    def __matmul__(self, other):
        return self[:] @ other

    def tolist(self):
        return self[:].tolist()


class PagedKeyframeStore(KeyframeStore):
    def __init__(self, directory, *, max_frames, max_points, max_disk_bytes,
                 reserve_bytes, min_translation_m=0., min_rotation_deg=0.):
        super().__init__(max_frames=max_frames, max_points=max_points)
        if (any(type(x) is not int or x < 0 for x in (max_disk_bytes, reserve_bytes)) or
                max_disk_bytes <= 0 or max_frames < 2 or
                any(isinstance(x, bool) or not isinstance(x, (int, float)) or
                    not math.isfinite(x) or x < 0 for x in (min_translation_m, min_rotation_deg))):
            raise ValueError('invalid persistent history budget or keyframe threshold')
        self.directory = Path(directory) if directory is not None else None
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=False)
        self.max_disk_bytes, self.reserve_bytes = max_disk_bytes, reserve_bytes
        self.min_translation_m, self.min_rotation_deg = min_translation_m, min_rotation_deg
        self.disk_bytes, self.storage_error, self.skipped_stationary = 0, '', 0
        self.records = []
        self._restore_ref = None

    def _append_record(self, record):
        # Roll back a partial journal append; a failed rollback still latches a
        # storage fault and prevents any further writes to this session.
        path = self.directory / 'frames.jsonl'
        with path.open('ab') as out:
            offset = out.tell()
            try:
                encoded = (json.dumps(record, allow_nan=False, separators=(',', ':'))+'\n').encode()
                out.write(encoded)
                out.flush(); os.fsync(out.fileno())
                directory_fd = os.open(self.directory, os.O_RDONLY)
                try: os.fsync(directory_fd)
                finally: os.close(directory_fd)
            except OSError:
                out.truncate(offset); out.flush(); os.fsync(out.fileno())
                raise
        return len(encoded)

    def _retain(self, pose, points, stamp, capture_context=None):
        if self._restore_ref is not None:
            return self._restore_ref
        if self.directory is None:
            raise StorageError('map is read only')
        if self.storage_error:
            raise StorageError(self.storage_error)
        if self._segments and pose.segment_id == self._segments[-1][0]:
            previous = self._segments[-1][1][-1].pose
            distance = np.linalg.norm(np.asarray(pose.position)-previous.position)
            a, b = np.asarray(pose.quaternion), np.asarray(previous.quaternion)
            angle = math.degrees(2*math.acos(np.clip(abs(np.dot(a,b))/(np.linalg.norm(a)*np.linalg.norm(b)),0,1)))
            if distance < self.min_translation_m and angle < self.min_rotation_deg:
                self.skipped_stationary += 1
                return None
        path = self.directory / ('%08d.npy' % self._frame_count)
        temp = path.with_suffix('.tmp')
        try:
            required = points.nbytes + 4096
            if (len(points) > self._max_points or self.disk_bytes + required > self.max_disk_bytes or
                    shutil.disk_usage(self.directory).free < required + self.reserve_bytes):
                raise OSError('persistent map disk budget or free-space reserve reached')
            with temp.open('xb') as out:
                np.save(out, points, allow_pickle=False); out.flush(); os.fsync(out.fileno())
            os.replace(temp, path)
            directory_fd = os.open(self.directory, os.O_RDONLY)
            try: os.fsync(directory_fd)
            finally: os.close(directory_fd)
            file_size = path.stat().st_size
            record = {'pose': asdict(pose), 'stamp_ns': stamp, 'file': path.name,
                      'count': len(points), 'session': self.directory.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            if capture_context is not None:
                record['capture_context'] = deepcopy(capture_context)
            journal_size = self._append_record(record)
        except OSError as error:
            self.storage_error = 'map storage fault: ' + str(error)
            for item in (temp, path):
                try: item.unlink(missing_ok=True)
                except OSError: pass
            raise StorageError(self.storage_error) from error
        self.records.append(record)
        self.disk_bytes += file_size + journal_size
        return DiskPoints(path, len(points))
