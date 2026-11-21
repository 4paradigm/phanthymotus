from dataclasses import dataclass
import time
import cv2
import numpy as np
from .backends import YuNet, OpenSeeFace, SingleFaceDetail
from .effects import Effects
from .tracking import Tracker, OpticalFlow
from .pose import estimate_pose
from .render import create_renderer
from .continuity import SceneContinuity, appearance


@dataclass
class Result:
    image: np.ndarray
    metadata: dict


class Engine:
    def __init__(self, config, detector=None, landmarker=None, renderer=None):
        self.cfg = dict(config)
        self.detector = self.landmarker = self.effects = self.renderer = None
        self.closed = False
        self.interval = int(config.get("detect_interval", 1))
        self.bridge_enabled = config.get('flow_bridge', True)
        if type(self.bridge_enabled) is not bool:
            raise ValueError('flow_bridge must be boolean')
        self.bridge_misses = 0
        self.last_single_model_time = None
        if not 1 <= self.interval <= 5:
            raise ValueError("detect_interval must be 1..5")
        self.max_age = float(config.get("max_age_ms", 250))/1000
        if not .01 <= self.max_age <= 2:
            raise ValueError("max_age_ms must be 10..2000")
        try:
            backend = config.get("detector_backend", "yunet")
            if backend not in ("yunet", "mediapipe"):
                raise ValueError("detector_backend must be yunet/mediapipe")
            if backend == "mediapipe" and config.get("provider", "cpu") != "cpu":
                raise ValueError("MediaPipe backend supports provider=cpu only")
            if config.get("landmarker_backend", "osf") not in ("osf", "hybrid"):
                raise ValueError("landmarker_backend must be osf/hybrid")
            if config.get('single_face_landmark_model') and (backend != 'yunet' or config.get('landmarker_backend','osf') != 'osf'):
                raise ValueError('single-face detail model requires YuNet + OSF')
            if backend == "yunet" and config.get("landmarker_backend") == "hybrid" and not config.get("landmark_model"):
                raise ValueError("hybrid requires a configured OSF fallback model")
            if detector is None and backend == "mediapipe":
                from .mediapipe_backend import MediaPipe
                detector = MediaPipe(config.get("face_landmarker_model", ""),
                                     config.get("max_faces", 4), config.get("face_landmarker_sha256"))
            self.detector = detector or YuNet(config.get("detector_model", ""),
                                               max_faces=config.get("max_faces", 4),
                                               max_side=config.get("detect_side", 640),
                                               upscale=config.get("detect_upscale", False),
                                               sha256=config.get("detector_sha256"))
            self.landmarker = landmarker
            if self.landmarker is None and config.get("landmark_model") and backend == "yunet":
                self.landmarker = OpenSeeFace(config["landmark_model"],
                                               config.get("provider", "cpu"),
                                               config.get("allow_cpu_fallback", True),
                                               config.get("landmark_sha256"),
                                               batch=config.get("batch_landmarks", True),
                                               threads=config.get('ort_threads',2),spinning=config.get('ort_spinning',True))
                if config.get('single_face_landmark_model'):
                    detail = OpenSeeFace(config['single_face_landmark_model'],
                        config.get('provider','cpu'),config.get('allow_cpu_fallback',True),
                        config.get('single_face_landmark_sha256'),batch=config.get('batch_landmarks',True),
                        threads=config.get('single_face_ort_threads',1),
                        spinning=config.get('single_face_ort_spinning',False))
                    self.landmarker = SingleFaceDetail(self.landmarker,detail)
                if config.get("landmarker_backend", "osf") == "hybrid":
                    from .mediapipe_backend import HybridLandmarker
                    self.landmarker = HybridLandmarker(self.landmarker, config.get("face_landmarker_model", ""),
                                                        config.get("mesh_min_face_pixels", 120),
                                                        config.get("face_landmarker_sha256"))
            self.effects = Effects(config.get("effects_manifest"))
            self.effects.select(config.get("effects", list(self.effects.plugins)))
            if renderer is None:
                self.renderer, self.render_fallback = create_renderer(config.get("renderer", "cpu"))
            else:
                self.renderer, self.render_fallback = renderer, ""
        except Exception:
            self.close()
            raise
        self.tracker = Tracker(config.get("max_faces", 4), smooth=config.get("smooth", True),
                               stabilization=config.get("stabilization", "similarity"))
        self.continuity = SceneContinuity()
        self.flow = OpticalFlow()
        self.last_timestamp = self.last_id = self.size = None
        self.count = 0
        self.dropped = 0

    def process(self, frame):
        if self.closed:
            raise RuntimeError("engine closed")
        begin = time.monotonic()
        if begin-frame.received > self.max_age:
            self.dropped += 1
            return None
        if (self.last_timestamp is not None and
                (frame.timestamp <= self.last_timestamp or frame.frame_id <= self.last_id)):
            self.dropped += 1
            return None
        t = time.perf_counter()
        scene_cut = self.continuity.update(frame.image) if self.cfg.get("scene_cuts", True) else False
        reset = (frame.size != self.size or self.last_timestamp is None or scene_cut
                 or frame.metadata.get("timestamp_repaired", False)
                 or frame.timestamp-self.last_timestamp > .5)
        if reset:
            self.tracker.reset()
            self.flow.reset()
            self.bridge_misses = 0
            self.last_single_model_time = None
            reset_backend = getattr(self.detector, "reset", None)
            if reset_backend and self.last_timestamp is not None:
                reset_backend()
            self.count = 0
        self.size = frame.size
        self.last_timestamp, self.last_id = frame.timestamp, frame.frame_id
        stages = {}
        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY) if self.interval > 1 else None
        stages["preprocess_ms"] = (time.perf_counter()-t)*1000
        use_detection = self.count % self.interval == 0 or not self.flow.faces
        prepare = getattr(self.detector, "set_timestamp", None)
        if prepare:
            prepare(frame.timestamp)
        t = time.perf_counter()
        faces = self.detector.detect(frame.image) if use_detection else self.flow.advance(gray)
        proposal_count = len(faces) if use_detection else None
        stages["detection_ms"] = (time.perf_counter()-t)*1000 if use_detection else 0.
        stages["flow_ms"] = 0. if use_detection else (time.perf_counter()-t)*1000
        t = time.perf_counter()
        if use_detection and self.landmarker:
            faces = self.landmarker.refine(frame.image, faces)
        stages["landmark_ms"] = (time.perf_counter()-t)*1000
        faces = [f for f in faces if f.valid()]
        model_face_count = len(faces) if use_detection else 0
        bridged = False
        bridge_start = time.perf_counter()
        # Genuine frame-to-frame pixel tracking only. No old sticker is held on
        # a missing frame; a bridge requires checked forward/backward LK + RANSAC.
        if self.bridge_enabled and self.interval == 1:
            if faces:
                self.bridge_misses = 0
                self.last_single_model_time = frame.timestamp if len(faces) == 1 else None
            elif (proposal_count == 0 and len(self.flow.faces) == 1
                  and len(self.tracker.tracks) == 1 and self.tracker.tracks[0].visible
                  and self.tracker.tracks[0].hits >= 2 and self.bridge_misses < 2
                  and self.last_single_model_time is not None
                  and 0 < frame.timestamp-self.last_single_model_time <= .1):
                gray = cv2.cvtColor(frame.image,cv2.COLOR_BGR2GRAY)
                faces = self.flow.advance(gray)
                self.bridge_misses += 1
                bridged = bool(faces)
                for face in faces:face.landmark_source = 'flow_bridge'
            if len(faces) == 1:
                if gray is None:gray = cv2.cvtColor(frame.image,cv2.COLOR_BGR2GRAY)
                self.flow.seed(gray,faces)
            else:self.flow.reset()
        stages['flow_ms'] += (time.perf_counter()-bridge_start)*1000
        if use_detection and self.interval > 1:
            self.flow.seed(gray, faces)
        t = time.perf_counter()
        if self.cfg.get("association_appearance", True):
            for face in faces:
                face.appearance = appearance(frame.image, face)
        stages["appearance_ms"] = (time.perf_counter()-t)*1000
        t = time.perf_counter()
        tracks = self.tracker.update(faces, frame.timestamp, frame.size)
        stages["tracking_ms"] = (time.perf_counter()-t)*1000
        t = time.perf_counter()
        poses = {track.id: estimate_pose(track.face, frame.size, self.cfg.get("camera_matrix")) for track in tracks}
        commands = self.effects.commands(tracks, frame.timestamp, poses)
        stages["pose_effect_ms"] = (time.perf_counter()-t)*1000
        t = time.perf_counter()
        output = self.renderer.render(frame.image, commands)
        stages["render_ms"] = (time.perf_counter()-t)*1000
        self.count += 1
        end = time.monotonic()
        # Never publish an old result merely because it was fresh before inference.
        if end-frame.received > self.max_age:
            self.dropped += 1
            return None
        metadata = {"schema": "motus.face_effects/1", "frame_id": frame.frame_id,
                    "source_timestamp": frame.timestamp, "size": list(frame.size),
                    "detected": len(faces), "tracked": len(tracks),
                    "detector_proposals": proposal_count, "model_face_count": model_face_count,
                    "flow_bridge": bridged,
                    "ids": [t.id for t in tracks], "poses": poses,
                    "anchors": {t.id: t.points.tolist() for t in tracks},
                    "landmark_sources": {t.id:t.face.landmark_source for t in tracks},
                    "effects": sorted(self.effects.active), "renderer": self.renderer.name,
                    "render_fallback": self.render_fallback,
                    "detector_runtime": getattr(self.detector, "runtime", ""),
                    "providers": getattr(self.landmarker or self.detector, "providers", ["opencv-cpu"]),
                    "inference_fallback": getattr(self.landmarker, "fallback_reason", ""),
                    "used_detection": use_detection, "stages": stages,
                    "processing_ms": (end-begin)*1000,
                    "receive_to_render_ms": (end-frame.received)*1000,
                    "source_to_display_ms": None, "ambiguous_associations": self.tracker.ambiguities,
                    "dropped": self.dropped, "scene_reset": reset, "scene_cut": scene_cut,
                    "source_metadata": frame.metadata}
        metadata['inference_diagnostics'] = dict(getattr(self.landmarker, 'last_profile', {}))
        return Result(output, metadata)

    def close(self):
        if self.closed:
            return
        self.closed = True
        # Release all resources even if one backend raises.
        errors = []
        for resource in (self.renderer, self.effects, self.landmarker, self.detector):
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:
                    errors.append(type(exc).__name__)
        if errors:
            raise RuntimeError("resource close failed: " + ", ".join(errors))
