"""Exploratory geometry retrieval on recorded optical-frame point clouds.

This tool proposes candidates only. It neither validates geometry nor accepts a
loop edge. Optional labels must be independently supplied, never inferred from
odometry or from registration scores.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

import numpy as np


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def scan_context_ring(points, rings=20, sectors=60, radius_m=6.):
    """Compact ring key from a Scan Context-style optical-frame height grid.

    Optical +z is forward and -y is up. The offset keeps observed heights
    positive while unobserved cells are zero. The ring key is a baseline, not a
    validated implementation of the full Scan Context retrieval pipeline.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points must be an Nx3 array')
    points = points[np.isfinite(points).all(axis=1)]
    r = np.hypot(points[:, 0], points[:, 2])
    keep = (r > 0.) & (r <= radius_m)
    points, r = points[keep], r[keep]
    if not len(points):
        raise ValueError('no valid points in descriptor range')
    ri = np.minimum((r / radius_m * rings).astype(int), rings - 1)
    angle = np.mod(np.arctan2(-points[:, 0], points[:, 2]), 2 * np.pi)
    si = np.minimum((angle / (2 * np.pi) * sectors).astype(int), sectors - 1)
    grid = np.zeros((rings, sectors))
    np.maximum.at(grid, (ri, si), np.maximum(-points[:, 1] + 2.5, .01))
    return grid.mean(axis=1)


def rank_candidates(descriptors, stamps_s, query, min_separation_s, top_k):
    values = np.asarray(descriptors, dtype=float)
    if (values.ndim != 2 or len(values) != len(stamps_s) or not np.isfinite(values).all()
            or not np.isfinite(np.asarray(stamps_s, dtype=float)).all()):
        raise ValueError('invalid descriptors or timestamps')
    if not 0 <= query < len(values) or top_k < 1 or min_separation_s < 0:
        raise ValueError('invalid query or retrieval parameters')
    scores = []
    for i, row in enumerate(values):
        if i == query or abs(stamps_s[query] - stamps_s[i]) < min_separation_s:
            continue
        scores.append({'index': i, 'distance': float(np.linalg.norm(row - values[query]))})
    return sorted(scores, key=lambda item: (item['distance'], item['index']))[:top_k]


def evaluate_labels(rankings, labels, top_k):
    seen, hits = set(), 0
    for entry in labels:
        query, positives = entry['query'], entry['positive']
        if (query in seen or query not in rankings or not isinstance(positives, list)
                or not positives or query in positives):
            raise ValueError('invalid or duplicate query labels')
        seen.add(query)
        hits += bool(set(positives) & {item['index'] for item in rankings[query][:top_k]})
    return {'labelled_queries': len(seen), 'hits': hits, 'top_k': top_k,
            'recall': hits / len(seen) if seen else None}


def _read_cloud(file, entry):
    offset, length = entry['offset'], entry['bytes']
    if (type(offset) is not int or type(length) is not int or offset < 0 or
            length <= 0 or length % 12 or offset + length > file.seek(0, 2)):
        raise ValueError('invalid cloud offset or size')
    file.seek(offset)
    return np.frombuffer(file.read(length), dtype='<f4').reshape(-1, 3).copy()


def _descriptors(points, o3d, voxel):
    valid = points[np.isfinite(points).all(axis=1)]
    ring = scan_context_ring(valid)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(valid.astype(np.float64)))
    cloud = cloud.voxel_down_sample(voxel)
    if len(cloud.points) < 10:
        raise ValueError('insufficient cloud points')
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2 * voxel, max_nn=30))
    feature = o3d.pipelines.registration.compute_fpfh_feature(
        cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=5 * voxel, max_nn=100))
    fpfh = np.mean(np.asarray(feature.data), axis=1)
    if not np.isfinite(fpfh).all() or np.linalg.norm(fpfh) == 0:
        raise ValueError('invalid FPFH descriptor')
    return ring / np.linalg.norm(ring), fpfh / np.linalg.norm(fpfh)


def main():
    started = time.perf_counter()
    tool_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path, help='directory containing manifest.json, cloud.jsonl and cloud.xyz')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--min-separation-s', type=float, default=10.)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--voxel-m', type=float, default=.08)
    parser.add_argument('--labels', type=Path, help='JSON list of {query: cloud index, positive: [cloud indices]}')
    parser.add_argument('--submap-frames', type=int, default=1, help='odd number of neighboring frames')
    parser.add_argument('--extrinsic', type=Path, help='JSON with child_frame_id, cloud_frame_id, provenance and child_from_cloud matrix')
    parser.add_argument('--max-gap-ms', type=float, help='experimental maximum pose interpolation gap')
    parser.add_argument('--assume-shared-clock', action='store_true', help='explicitly assume cloud and odometry share a clock')
    args = parser.parse_args()
    if args.stride < 1 or args.top_k < 1 or not math.isfinite(args.voxel_m) or args.voxel_m <= 0 or not math.isfinite(args.min_separation_s) or args.min_separation_s < 0:
        parser.error('invalid stride, separation, top-k or voxel size')
    if args.submap_frames < 1 or args.submap_frames % 2 != 1:
        parser.error('submap-frames must be a positive odd number')
    if args.submap_frames > 1 and (not args.extrinsic or args.max_gap_ms is None or
                                   not math.isfinite(args.max_gap_ms) or args.max_gap_ms <= 0 or
                                   not args.assume_shared_clock):
        parser.error('submaps require --extrinsic, positive --max-gap-ms and --assume-shared-clock')
    import open3d as o3d
    root = args.recording
    manifest_bytes = (root / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get('schema') != 'bumi-pointcloud-slam-record/1' or manifest.get('cloud_encoding') != 'xyz-float32-le':
        raise ValueError('unsupported recording format')
    entries = [json.loads(line) for line in (root / 'cloud.jsonl').read_text().splitlines()]
    requested = list(range(0, len(entries), args.stride))
    if not requested:
        raise ValueError('recording contains no clouds')
    poses, extrinsic_info = None, None
    if args.submap_frames > 1:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from plugins.pointcloud_slam_alignment import align_pose
        from plugins.pointcloud_slam_odom import decode_bumi_odom
        from tools.pointcloud_slam_submap import _rigid, compose_submap
        extrinsic_info = json.loads(args.extrinsic.read_text())
        if (extrinsic_info.get('cloud_frame_id') != manifest.get('cloud_frame') or
                not extrinsic_info.get('child_frame_id') or not extrinsic_info.get('provenance')):
            raise ValueError('extrinsic must name matching cloud frame, child frame and provenance')
        _rigid(extrinsic_info['child_from_cloud'])
        samples = []
        for line in (root / 'odom.jsonl').read_text().splitlines():
            try:
                samples.append(decode_bumi_odom(json.loads(line)))
            except ValueError:
                samples.append(None)
        poses = []
        for entry in entries:
            try:
                pose = align_pose(samples, entry['sec'] * 1_000_000_000 + entry['nanosec'],
                                  max_gap_ms=args.max_gap_ms)
                poses.append(pose if pose.child_frame_id == extrinsic_info['child_frame_id'] else None)
            except ValueError:
                poses.append(None)
    selected, skipped, rings, fpfhs, stamps = [], [], [], [], []
    with (root / 'cloud.xyz').open('rb') as binary:
        for index in requested:
            entry = entries[index]
            half = args.submap_frames // 2
            indices = range(index - half, index + half + 1)
            if indices.start < 0 or indices.stop > len(entries) or (poses is not None and any(poses[i] is None for i in indices)):
                skipped.append(index)
                continue
            if any(entries[i].get('frame_id') != manifest.get('cloud_frame') for i in indices):
                raise ValueError('cloud frame differs from recording manifest')
            clouds = [_read_cloud(binary, entries[i]) for i in indices]
            clouds = [p[np.isfinite(p).all(axis=1)] for p in clouds]
            if poses is not None:
                try:
                    points = compose_submap(clouds, [poses[i] for i in indices],
                                            extrinsic_info['child_from_cloud'], anchor=half)
                except ValueError:
                    skipped.append(index)
                    continue
            else:
                points = clouds[0]
            ring, fpfh = _descriptors(points, o3d, args.voxel_m)
            selected.append(index)
            rings.append(ring)
            fpfhs.append(fpfh)
            stamps.append(entry['sec'] + entry['nanosec'] / 1e9)
    if not selected:
        raise ValueError('no usable clouds or submaps')
    labels = json.loads(args.labels.read_text()) if args.labels else []
    local_index = {value: i for i, value in enumerate(selected)}
    translated = []
    for entry in labels:
        if entry['query'] not in local_index or any(i not in local_index for i in entry['positive']):
            raise ValueError('label refers to a frame excluded from this run')
        translated.append({'query': local_index[entry['query']],
                           'positive': [local_index[i] for i in entry['positive']]})
    out = {'recording_schema': manifest['schema'], 'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
           'tool_sha256': tool_sha256,
           'cloud_index_sha256': _sha256_file(root / 'cloud.jsonl'),
           'cloud_data_sha256': _sha256_file(root / 'cloud.xyz'),
           'odom_data_sha256': _sha256_file(root / 'odom.jsonl') if poses is not None else None,
           'sampled_frame_indices': selected, 'voxel_m': args.voxel_m,
           'submap_frames': args.submap_frames, 'skipped_frame_indices': skipped,
           'max_gap_ms': args.max_gap_ms, 'assume_shared_clock': args.assume_shared_clock,
           'extrinsic': extrinsic_info,
           'min_separation_s': args.min_separation_s, 'top_k': args.top_k,
           'label_source': str(args.labels) if args.labels else None,
           'status': 'exploratory retrieval only; no geometric validation or loop edges', 'methods': {}}
    for name, features in [('scan_context_ring', rings), ('mean_fpfh', fpfhs)]:
        rankings = {i: rank_candidates(features, stamps, i, args.min_separation_s, args.top_k)
                    for i in range(len(selected))}
        metrics = evaluate_labels(rankings, translated, args.top_k)
        out['methods'][name] = {'metrics': metrics, 'rankings': {
            str(selected[q]): [{'index': selected[item['index']], 'distance': item['distance']}
                               for item in ranked] for q, ranked in rankings.items()}}
    out['elapsed_s'] = time.perf_counter() - started
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out['peak_process_rss_bytes'] = max_rss if sys.platform == 'darwin' else max_rss * 1024
    print(json.dumps(out, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
