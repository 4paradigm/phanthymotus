"""Read bounded numeric demonstration data; no pickle, training or device runtime."""
from __future__ import annotations

import json
import math
from pathlib import Path
import zipfile

from . import media, tracks


def _clock(length, fps, start_index=0):
    fps = tracks.number(fps, '数据帧率')
    if not .1 <= fps <= 240:
        raise ValueError('没有可靠时间戳时必须提供0.1–240之间的源数据帧率')
    return [(i + start_index)/fps for i in range(length)]


def _array(value, *, max_values=2_000_000):
    import numpy as np
    if value.dtype.kind not in 'fiu' or math.prod(value.shape) > max_values:
        raise ValueError('只接受有界数值数组')
    if getattr(value, 'is_virtual', False) or getattr(value, 'external', None):
        raise ValueError('不接受引用外部数据的数组')
    return np.asarray(value[:])


def _point(values, confidence=1.):
    import numpy as np
    if not np.isfinite(values).all() or not math.isfinite(float(confidence)) or confidence <= 0:
        return None
    return [float(v) for v in values] + [min(1., float(confidence))]


def _zip_check(path):
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) > 20000 or sum(m.file_size for m in members) > media.MAX_BYTES:
            raise ValueError('解压后的数据超过64 MiB，请先导出单个短episode')
        for member in members:
            if member.filename.startswith('/') or '..' in Path(member.filename).parts:
                raise ValueError('压缩包包含非法路径')
            if member.filename.endswith('.npy'):
                import numpy as np
                with archive.open(member) as source:
                    version = np.lib.format.read_magic(source)
                    if version == (1, 0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(source)
                    elif version == (2, 0):
                        shape, _, dtype = np.lib.format.read_array_header_2_0(source)
                    else:
                        raise ValueError('不支持此NPY版本，请导出普通数值数组')
                    count = math.prod(shape)
                    if dtype.kind not in 'fiu' or count > 2_000_000 or count*dtype.itemsize > member.file_size-source.tell():
                        raise ValueError('NPZ数组类型、长度或大小无效')
            if member.filename.endswith('.zarray'):
                metadata = json.loads(archive.read(member))
                import numpy as np
                if (np.dtype(metadata['dtype']).kind not in 'fiu' or math.prod(metadata['shape']) > 2_000_000
                        or math.prod(metadata['chunks'])*np.dtype(metadata['dtype']).itemsize > media.MAX_BYTES):
                    raise ValueError('Zarr数组不是有界数值；请去掉图像数组，视频单独导入')
                compressor = metadata.get('compressor')
                if (compressor and compressor.get('id') not in ('blosc', 'zlib', 'gzip', 'zstd', 'lz4')) or metadata.get('filters'):
                    raise ValueError('不支持该Zarr编码，请导出数值NPZ或JSON')


def read(path, format_name, *, fps=None, episode=0, coordinate_frame='', start_index=0, end_index=None):
    path = Path(path)
    if path.stat().st_size > media.MAX_BYTES:
        raise ValueError('数据文件超过64 MiB')
    if format_name == 'tracks_json':
        value = json.loads(path.read_text(encoding='utf-8'))
        return tracks.validate(value)
    if format_name == 'egodex_hdf5':
        return _egodex(path, fps, start_index, end_index)
    if format_name in ('umi_npz', 'umi_zarr'):
        if not coordinate_frame.strip():
            raise ValueError('请注明UMI处理后的坐标系；不会假设它已是机器人坐标')
        return _umi(path, format_name, fps, episode, coordinate_frame, start_index, end_index)
    raise ValueError('支持tracks_json、egodex_hdf5、umi_npz、umi_zarr')


def _egodex(path, fps, first, last):
    try:
        import h5py
    except ImportError:
        raise ValueError('读取EgoDex需要可选依赖h5py；其他功能仍可使用') from None
    import numpy as np
    with h5py.File(path, 'r') as root:
        # Hard links only: uploaded HDF5 must not load another server file.
        for key in ('transforms', 'confidences', 'camera'):
            if key in root and not isinstance(root.get(key, getlink=True), h5py.HardLink):
                raise ValueError('HDF5不接受外部或软链接')
        group = root.get('transforms')
        if group is None:
            raise ValueError('EgoDex缺少transforms')
        preferred = ['camera', 'leftHand', 'rightHand']
        names = preferred + sorted(n for n in group if n not in preferred)
        names = [n for n in names if n in group]
        if not names or len(names) > tracks.MAX_POINTS:
            raise ValueError('EgoDex关节点数量无效')
        matrices, confidences, length = {}, {}, None
        first = int(first)
        for name in names:
            if not isinstance(group.get(name, getlink=True), h5py.HardLink):
                raise ValueError('HDF5关节点不能引用外部文件')
            dataset = group[name]
            if dataset.dtype.kind not in 'fiu' or len(dataset.shape) != 3 or dataset.shape[1:] != (4, 4) or dataset.is_virtual or dataset.external:
                raise ValueError('EgoDex变换必须为内部N×4×4数值数组')
            stop = dataset.shape[0] if last is None else int(last)
            if not 0 <= first < stop <= dataset.shape[0] or stop-first > tracks.MAX_FRAMES:
                raise ValueError('请选择最多3000帧的EgoDex区间')
            if length is not None and length != stop-first:
                raise ValueError('EgoDex各关节点帧数不一致')
            length = stop-first
            matrices[name] = dataset[first:stop]
            conf = root.get('confidences')
            if conf is not None and name in conf:
                if not isinstance(conf.get(name, getlink=True), h5py.HardLink):
                    raise ValueError('置信度不能引用外部文件')
                values = _array(conf[name])
                if values.ndim != 1 or len(values) != dataset.shape[0]:
                    raise ValueError('置信度帧数不匹配')
                confidences[name] = values[first:stop]
        times = _clock(length, fps, first)
        frames = []
        for i, stamp in enumerate(times):
            points, rotations = [], []
            for name in names:
                matrix = matrices[name][i]
                score = confidences[name][i] if name in confidences else 1.
                valid = np.isfinite(matrix).all() and np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-3)
                p = _point(matrix[:3, 3], score) if valid else None
                points.append(p)
                rotations.append(matrix[:3, :3].reshape(-1).tolist() if p is not None else None)
            frames.append({'time_s': stamp, 'points': points, 'rotations': rotations})
        meta = {'format': 'EgoDex', 'source_fps': float(fps), 'source_frame_start': first,
                'time_source': 'declared_fps', 'rotation_format': 'matrix3x3',
                'origin': 'ARKit recording-session origin; not shared across episodes',
                'missing_confidence': '1 means present, not measured confidence'}
        camera = root.get('camera')
        if camera is not None and 'intrinsic' in camera:
            if not isinstance(camera.get('intrinsic', getlink=True), h5py.HardLink):
                raise ValueError('相机内参不能引用外部文件')
            intrinsic = _array(camera['intrinsic'])
            if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
                raise ValueError('相机内参无效')
            meta['intrinsic'] = intrinsic.tolist()
    return tracks.validate({'kind': 'trajectory3d', 'units': 'm', 'coordinate_frame': 'ARKit_session',
                            'names': names, 'frames': frames, 'metadata': meta})


def _umi(path, format_name, fps, episode, coordinate_frame, first, last):
    import numpy as np
    _zip_check(path)
    store = None
    try:
        if format_name == 'umi_npz':
            store = np.load(path, allow_pickle=False)
            arrays = {k: store[k] for k in store.files}
            data = arrays
            ends = arrays.get('episode_ends')
        else:
            try:
                import zarr
            except ImportError:
                raise ValueError('读取UMI Zarr需要可选依赖zarr<3；也可导入数值NPZ/JSON') from None
            if not hasattr(zarr, 'ZipStore'):
                raise ValueError('当前读取器需要zarr 2.x，未自动升级运行环境')
            store = zarr.ZipStore(str(path), mode='r')
            root = zarr.open_group(store, mode='r')
            data = root['data']
            ends = _array(root['meta']['episode_ends'])
        names = sorted(k[:-8] for k in data if k.endswith('_eef_pos'))
        if not 1 <= len(names) <= 4:
            raise ValueError('UMI数据需要robotN_eef_pos，最多4个末端')
        total = data[names[0]+'_eef_pos'].shape[0]
        if ends is not None:
            ends = np.asarray(ends)
            if ends.ndim != 1 or ends.dtype.kind not in 'iu' or not len(ends) or np.any(np.diff(ends) <= 0) or ends[-1] != total or not 0 <= int(episode) < len(ends):
                raise ValueError('UMI episode边界无效')
            base = int(ends[int(episode)-1]) if int(episode) else 0
            limit = int(ends[int(episode)])
        else:
            if int(episode):
                raise ValueError('此NPZ只有一个episode')
            base, limit = 0, total
        start = base + int(first)
        stop = limit if last is None else base + int(last)
        if not base <= start < stop <= limit or stop-start > tracks.MAX_FRAMES:
            raise ValueError('请选择最多3000帧的UMI区间')
        parsed = {}
        for name in names:
            parsed[name] = {}
            for field, width in (('eef_pos', 3), ('eef_rot_axis_angle', 3), ('gripper_width', 1)):
                key = name + '_' + field
                if key not in data:
                    raise ValueError('UMI缺少字段：' + key)
                arr = data[key]
                if arr.dtype.kind not in 'fiu' or arr.shape not in ((total, width), (total,)) or (arr.ndim == 1 and width != 1):
                    raise ValueError('UMI数组形状或类型错误：' + key)
                parsed[name][field] = np.asarray(arr[start:stop]).reshape(-1, width)
        times = _clock(stop-start, fps, int(first))
        frames = []
        for i, stamp in enumerate(times):
            points, rotations, widths = [], [], []
            for name in names:
                values = parsed[name]
                p = _point(values['eef_pos'][i])
                rotation = values['eef_rot_axis_angle'][i]
                width = float(values['gripper_width'][i, 0])
                if not np.isfinite(rotation).all() or not math.isfinite(width):
                    p = None
                points.append(p)
                rotations.append(rotation.tolist() if p is not None else None)
                widths.append(width if p is not None else None)
            frames.append({'time_s': stamp, 'points': points, 'rotations': rotations, 'widths': widths})
        return tracks.validate({'kind': 'trajectory3d', 'units': 'm', 'coordinate_frame': coordinate_frame,
            'names': names, 'frames': frames, 'metadata': {'format': format_name, 'episode': int(episode),
            'time_source': 'declared_fps', 'source_fps': float(fps), 'source_frame_start': int(first),
            'rotation_format': 'axis_angle_radians', 'identity': 'robot index from source; not inferred left/right'}})
    finally:
        if store is not None:
            store.close()
