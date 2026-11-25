"""Run real engines against imported independent labels; never generate truth.

Static images get two identical observations for the Card's confirmation rule.
This is explicitly not a temporal experiment. Each sequence gets a new engine,
including fresh MediaPipe VIDEO state. Initialization/decode are timed separately.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import cv2
import numpy as np
import psutil

from natural_benchmark import load_package, stats
from dataset_adapters import relative_path, sha256


def observation(face):
    return {'box': face.box.tolist(), 'points': face.points.tolist(),
            'topology': getattr(face, 'topology', 'osf66'),
            'confidence': float(face.confidence), 'quality': float(face.quality),
            'landmarks': face.landmarks.tolist() if face.landmarks is not None else None}


def installed_version(name):
    try: return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: return None


def run(labels, root, package, models, model, output, extra=None):
    cv2.setNumThreads(2)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    engine_module, contracts = load_package(Path(package).resolve(), 'dataset_engine')
    root = Path(root).resolve(); models = Path(models).resolve()
    config = {'detector_model': str(models/'yunet.onnx'), 'renderer': 'cpu',
              'max_faces': 8, 'detect_side': 640, 'max_age_ms': 2000}
    if model in ('osf0', 'osf3', 'hybrid'):
        config['landmark_model'] = str(models/(('osf0' if model == 'hybrid' else model)+'.onnx'))
    if model == 'mediapipe': config['detector_backend'] = 'mediapipe'
    if model == 'hybrid': config['landmarker_backend'] = 'hybrid'
    if model in ('mediapipe', 'hybrid'):
        config['face_landmarker_model'] = str(models/'face_landmarker.task')
    config.update(extra or {})
    process = psutil.Process(); begin = time.monotonic(); cpu0 = sum(process.cpu_times()[:2])
    times = []; init_times = []; decode_times = []; stages = {}; rss = []; tree_rss = []; input_hashes = {}
    nframes = 0; nobs = 0
    for clip, annotations in labels['clips'].items():
        relative_path(clip)
        start = time.perf_counter(); engine = engine_module.Engine(config)
        init_times.append((time.perf_counter()-start)*1000)
        observed = {}; original = engine.detector.detect
        def detect(image):
            faces = original(image); observed['detector'] = faces
            observed['refined'] = faces
            return faces
        engine.detector.detect = detect
        if engine.landmarker:
            refine_original = engine.landmarker.refine
            def refine(image, faces):
                faces = refine_original(image, faces)
                observed['refined'] = faces
                return faces
            engine.landmarker.refine = refine
        metadata = labels['sequence_metadata'][clip]; cap = None
        if 'video' in metadata:
            path = root/relative_path(metadata['video'])
            input_hashes[metadata['video']] = sha256(path); cap = cv2.VideoCapture(str(path))
            if not cap.isOpened(): raise RuntimeError('source video failed to open')
            source_index = -1
        try:
            with (output/(clip+'.jsonl')).open('x') as stream:
                for annotation in annotations:
                    t = time.perf_counter()
                    if cap is not None:
                        wanted = annotation['decode_index']
                        if wanted <= source_index: raise ValueError('non-increasing source frame order')
                        while source_index < wanted:
                            ok, image = cap.read(); source_index += 1
                            if not ok: raise RuntimeError('video ended before original annotated frame')
                    else:
                        path = root/relative_path(annotation['image']); image = cv2.imread(str(path))
                        input_hashes[annotation['image']] = sha256(path)
                    if image is None: raise ValueError('source image decode failed')
                    source_image = metadata.get('source_image')
                    if source_image and image.shape[:2] != (source_image['height'], source_image['width']):
                        raise ValueError('decoded image size differs from original annotation metadata')
                    decode_times.append((time.perf_counter()-t)*1000)
                    # Resetting per static sequence avoids hidden cross-image state.
                    repeats = 1 if metadata['temporal'] else 2
                    for repeat in range(repeats):
                        timestamp = annotation['timestamp'] if metadata['temporal'] else repeat/30
                        frame = contracts.Frame(image, timestamp, annotation['frame_id']+repeat)
                        result = engine.process(frame); nobs += 1
                        if result is None: raise RuntimeError('offline frame dropped; report run failure')
                        times.append(result.metadata['processing_ms'])
                        for key, value in result.metadata['stages'].items(): stages.setdefault(key, []).append(value)
                    row = {'frame_id': annotation['frame_id'], 'timestamp': annotation['timestamp'],
                           'static_repeated_observations': repeats if not metadata['temporal'] else None,
                           'detector_observations': [observation(f) for f in observed.get('detector', [])],
                           'detections': [observation(f) for f in observed.get('refined', [])], **result.metadata,
                           'boxes': {t.id: t.face.box.tolist() for t in engine.tracker.tracks if t.visible},
                           'raw_anchors': {t.id: t.face.points.tolist() for t in engine.tracker.tracks if t.visible},
                           'glasses_eyes': {}}
                    # Source identifiers override repeated-observation engine IDs.
                    row['frame_id'] = annotation['frame_id']; row['timestamp'] = annotation['timestamp']
                    for track in engine.tracker.tracks:
                        if not track.visible: continue
                        commands = engine.effects.plugins['glasses'].commands(track, timestamp, row['poses'][track.id])
                        if commands:
                            row['glasses_eyes'][track.id] = (
                                np.array([[62., 44.], [178., 44.]]) @ commands[0].matrix[:, :2].T
                                + commands[0].matrix[:, 2]).tolist()
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                    nframes += 1; rss.append(process.memory_info().rss/1048576)
                    tree_rss.append(sum(p.memory_info().rss for p in [process, *process.children(recursive=True)])/1048576)
                    if nframes % 50 == 0: print(model, nframes, flush=True)
        finally:
            if cap is not None: cap.release()
            engine.close()
    elapsed = time.monotonic()-begin
    summary = {'schema': 'face_effects_dataset_benchmark/1', 'model': model, 'config': config,
               'annotation_provenance': labels['provenance'], 'source_input_sha256': input_hashes,
               'source_sha256': {str(p.relative_to(package)): sha256(p) for p in sorted(Path(package).rglob('*.py'))},
               'model_sha256': {Path(p).name: sha256(p) for key, p in config.items() if key.endswith('_model')},
               'frames': nframes, 'processing_observations': nobs, 'wall_seconds': elapsed,
               'processing_ms': stats(times), 'initialization_ms': stats(init_times),
               'decode_ms': stats(decode_times), 'stages_ms': {k: stats(v) for k, v in stages.items()},
               'cpu_core_percent': (sum(process.cpu_times()[:2])-cpu0)/elapsed*100,
               'cpu_parent_and_children_core_percent': (sum(sum(p.cpu_times()[:2]) for p in [process, *process.children(recursive=True)])-cpu0)/elapsed*100,
               'rss_peak_mib': max(rss),
               'rss_parent_and_children_peak_mib': max(tree_rss),
               'environment': {'python': sys.version, 'machine': platform.machine(),
                   'platform': platform.platform(), 'numpy': np.__version__, 'opencv': cv2.__version__,
                   'opencv_threads': cv2.getNumThreads(),
                   'packages': {p: installed_version(p) for p in ('onnxruntime', 'mediapipe', 'psutil')}},
               'limits': ['static repeat is confirmation only, not jitter/lag/ID evidence',
                          'processing excludes initialization, input decode and observation serialization',
                          'throughput capacity is not live camera/display FPS',
                          'CPU/RSS include initialization and evidence collection; aggregate RSS sums shared pages, not PSS']}
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('labels', 'images', 'package', 'models', 'output'): p.add_argument('--'+key, required=True)
    p.add_argument('--model', choices=['osf0', 'osf3', 'yunet', 'mediapipe', 'hybrid'], required=True)
    p.add_argument('--extra-config', default='{}', help='explicit candidate config for reproducible A/B')
    a = p.parse_args()
    run(json.loads(Path(a.labels).read_text()), a.images, a.package, a.models, a.model, a.output, json.loads(a.extra_config))


if __name__ == '__main__': main()
