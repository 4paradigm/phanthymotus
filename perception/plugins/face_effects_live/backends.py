"""YuNet detection and optional OpenSeeFace-66 landmark inference.

OSF preprocessing and heatmap/offset decoding adapted from Emiliana's tracker.py;
copyright (c) 2019 Emiliana, BSD-2-Clause. See THIRD_PARTY.md and licenses/.
"""
from pathlib import Path
import importlib.util
import hashlib
import time
import uuid
import numpy as np
import cv2
from .contracts import Face
from .osf_decode import decode_heatmaps
from .osf_input import prepare_crop


def capabilities():
    ort = importlib.util.find_spec("onnxruntime") is not None
    providers = []
    if ort:
        import onnxruntime
        providers = onnxruntime.get_available_providers()
    return {"opencv": cv2.__version__, "yunet": hasattr(cv2, "FaceDetectorYN"),
            "onnxruntime": ort, "providers_advertised": providers,
            "moderngl_installed": importlib.util.find_spec("moderngl") is not None,
            "mediapipe_installed": importlib.util.find_spec("mediapipe") is not None,
            "note": "provider listing is not a successful device/session probe"}


def model_path(path, expected_sha=None):
    p = Path(path).expanduser().resolve()
    if not p.is_file() or not 1024 < p.stat().st_size < 150_000_000:
        raise ValueError("missing or invalid local model file")
    if expected_sha and hashlib.sha256(p.read_bytes()).hexdigest() != expected_sha:
        raise ValueError("model SHA256 mismatch")
    return str(p)


class YuNet:
    def __init__(self, path, max_side=640, max_faces=8, score=0.7, sha256=None, upscale=False):
        self.max_side, self.max_faces = int(max_side), int(max_faces)
        if type(upscale) is not bool:
            raise ValueError("upscale must be boolean")
        self.upscale = upscale
        if not 160 <= self.max_side <= 1920 or not 1 <= self.max_faces <= 8:
            raise ValueError("max_side 160..1920; max_faces 1..8")
        self.detector = cv2.FaceDetectorYN.create(
            model_path(path, sha256), "", (320, 320), float(score), 0.3, 5000)

    def detect(self, image):
        h, w = image.shape[:2]
        # Optional small-face quality profile. Bounded at 2x; never changes the
        # source frame, effect coordinates, or default CPU performance profile.
        scale = min(2. if self.upscale else 1., self.max_side / max(w, h))
        # YuNet requires multiples of 32 for some model/OpenCV versions.
        sw, sh = max(32, round(w * scale / 32) * 32), max(32, round(h * scale / 32) * 32)
        small = cv2.resize(image, (sw, sh))
        self.detector.setInputSize((sw, sh))
        _, rows = self.detector.detect(small)
        result = []
        if rows is not None:
            for row in sorted(rows, key=lambda r: -float(r[-1]))[:self.max_faces]:
                factors = np.array([w / sw, h / sh], dtype=np.float64)
                box = row[:4].astype(np.float64) * np.tile(factors, 2)
                points = row[4:14].reshape(5, 2).astype(np.float64) * factors
                points[:2] = points[:2][np.argsort(points[:2, 0])]
                points[3:] = points[3:][np.argsort(points[3:, 0])]
                face = Face(box, points, float(row[-1]))
                if face.valid():
                    result.append(face)
        return result

    def close(self):
        self.detector = None


class OpenSeeFace:
    """Supports lm_model0..4_opt.onnx (224 input, 198x28x28 output)."""
    def __init__(self, path, provider="cpu", allow_cpu_fallback=True, sha256=None, batch=True,
                 threads=2, spinning=True):
        import onnxruntime as ort
        from plugins.ort_worker import get_worker
        self.session = self.worker = self.session_key = None
        if type(batch) is not bool:
            raise ValueError('batch must be boolean')
        if type(threads) is not int or not 1 <= threads <= 4 or type(spinning) is not bool:
            raise ValueError('ORT threads 1..4 and boolean spinning required')
        self.batch = batch
        self.last_profile = {}
        if provider not in ("cpu", "cuda", "tensorrt"):
            raise ValueError("provider must be cpu/cuda/tensorrt")
        desired = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider",
                   "tensorrt": "TensorrtExecutionProvider"}[provider]
        available = ort.get_available_providers()
        self.fallback_reason = ""
        if desired not in available:
            if not allow_cpu_fallback:
                raise RuntimeError("requested inference provider unavailable")
            self.fallback_reason = f"{desired} unavailable; using CPU"
            desired = "CPUExecutionProvider"
        # Perception already owns a spawn-only ORT host. Never create a second
        # runtime session beside sherpa-onnx in the Card/ROS process. The proxy
        # transports tensors; it is not a zero-copy path.
        options = {"intra_op_num_threads": threads, "log_severity_level": 3}
        if not spinning:
            options['session_config'] = {'session.intra_op.allow_spinning':'0',
                                         'session.inter_op.allow_spinning':'0'}
        providers = [desired]
        if allow_cpu_fallback and desired != "CPUExecutionProvider":
            providers.append("CPUExecutionProvider")
        path = model_path(path, sha256)
        self.worker = get_worker()
        self.session_key = "face_effects_live." + uuid.uuid4().hex
        self._postprocess = {"module": "plugins.face_effects_live.osf_decode", "func": "decode_heatmaps"}
        try:
            self.session = self.worker.load(self.session_key, path, providers, options, postprocess=self._postprocess)
        except Exception:
            if not allow_cpu_fallback or desired == "CPUExecutionProvider":
                self.close()
                raise
            self.fallback_reason = "accelerated session initialization failed; using CPU"
            try:
                self.session = self.worker.load(self.session_key, path, ["CPUExecutionProvider"], options, postprocess=self._postprocess)
            except Exception:
                self.close()
                raise
        actual = self.session.get_providers()
        if desired not in actual and not allow_cpu_fallback:
            self.close()
            raise RuntimeError("runtime silently disabled requested provider")
        self.providers = actual
        self._reload = (path, actual, options)
        if desired not in actual:
            self.fallback_reason = "runtime disabled requested provider; using CPU"
        self.input = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        if len(shape) != 4 or shape[1:] != [3, 224, 224] or (isinstance(shape[0], int) and shape[0] != 1):
            self.close()
            raise ValueError("unsupported OpenSeeFace input schema")

    def refine(self, image, faces):
        h, w = image.shape[:2]
        pending = []
        batched = getattr(self.session, "run_many", None) if self.batch else None
        for face in faces:
            x, y, bw, bh = face.box
            x1, y1 = max(0, int(x - .1*bw)), max(0, int(y - .125*bh))
            x2, y2 = min(w, int(x + 1.1*bw)), min(h, int(y + 1.125*bh))
            if x2-x1 < 4 or y2-y1 < 4:
                continue
            crop = image[y1:y2, x1:x2]
            # Compact original pixels for small/medium faces. A large crop is
            # normalized in the parent so transport never exceeds its old tensor.
            feeds = ({'bgr_crop':np.ascontiguousarray(crop)}
                     if batched is not None and crop.nbytes < 3*224*224*4
                     else {self.input:prepare_crop(crop)})
            pending.append((face, x1, y1, x2, y2, feeds))
        if not pending:
            self.last_profile = {}
            return []
        feeds = [row[-1] for row in pending]
        started = time.perf_counter()
        if batched is not None:
            measured = batched(None, feeds, preprocess={
                'module':'plugins.face_effects_live.osf_input','func':'prepare_feeds',
                'kwargs':{'input_name':self.input}}, profile=True)
            replies = measured.pop('outputs')
            self.last_profile = measured
        else:
            replies = [self.session.run(None, feed) for feed in feeds]
            self.last_profile = {}
        elapsed = (time.perf_counter()-started)*1000
        self.last_profile.update(round_trip_ms=elapsed,
            request_array_bytes=sum(a.nbytes for feed in feeds for a in feed.values()),
            faces=len(feeds), transport='bounded_crop_rpc' if batched is not None else 'legacy_tensor_rpc')
        if batched is not None:
            compute = sum(self.last_profile[k] for k in ('worker_prepare_ms','worker_inference_ms','worker_post_ms'))
            self.last_profile['queue_copy_scheduling_ms'] = max(0., elapsed-compute)
        if not isinstance(replies, (list, tuple)) or len(replies) != len(pending):
            raise ValueError("incomplete OpenSeeFace batch reply")
        result = []; reload_postprocess = False
        for (face, x1, y1, x2, y2, _), reply in zip(pending, replies):
            if (isinstance(reply, tuple) and len(reply) == 2
                    and np.shape(reply[0]) == (66, 2) and np.shape(reply[1]) == (66,)):
                coordinates, conf = reply
            else:
                # The existing host's crash-reload path restores sessions without
                # their optional postprocessor. Decode that first raw reply with
                # the identical function, then restore the compact transport.
                # Keep the shared-host restart contract; no in-process fallback.
                coordinates, conf = decode_heatmaps(reply)
                reload_postprocess = True
            if not np.isfinite(coordinates).all() or not np.isfinite(conf).all():
                raise ValueError("invalid OpenSeeFace decoded values")
            xs = x1 + coordinates[:, 0] * (x2-x1)/224
            ys = y1 + coordinates[:, 1] * (y2-y1)/224
            lms = np.column_stack((xs, ys))
            quality = float(conf.mean())
            if quality < .45:
                continue  # no convincing landmarks: no effect, not fabricated points
            # OSF-66 is NOT iBUG-68. Upstream's mouth_corner features use 58/62;
            # 48/54 lie on the lip curves and create a tilted, asymmetric mouth.
            points = np.array([lms[36:42].mean(0), lms[42:48].mean(0), lms[30], lms[58], lms[62]])
            points[:2] = points[:2][np.argsort(points[:2, 0])]
            points[3:] = points[3:][np.argsort(points[3:, 0])]
            result.append(Face(face.box.copy(), points, face.confidence, lms, quality))
        if reload_postprocess:
            path, providers, options = self._reload
            self.session = self.worker.load(self.session_key, path, providers, options, postprocess=self._postprocess)
        return result

    def close(self):
        # Own session only: the shared worker can still serve other Cards.
        if self.worker is not None and self.session_key is not None:
            self.worker.unload(self.session_key)
        self.session = None
        self.worker = self.session_key = None


class SingleFaceDetail:
    """Guard a single-face detail proposal with the unchanged fast model.

    Both paths evaluate all proposals; this never truncates a crowded frame to
    manufacture throughput. The optional model is explicit, never downloaded or
    guessed from a host path. Small/clipped faces stay on the fast model. On a
    sufficiently resolved face, both models must agree geometrically before the
    detail points may replace the fast result. These are inference safety gates,
    never changes to dataset labels or accuracy scoring thresholds.
    """
    def __init__(self, fast, detail):
        self.fast, self.detail = fast, detail
        self.providers = list(dict.fromkeys([*fast.providers, *detail.providers]))
        self.fallback_reason = '; '.join(x for x in (fast.fallback_reason,detail.fallback_reason) if x)
        self.last_profile = {}

    def refine(self, image, faces):
        result = self.fast.refine(image, faces)
        profile = dict(self.fast.last_profile)
        tier = 'fast_multi' if len(faces) != 1 else 'fast_small_or_clipped'
        accepted = False
        if len(faces) == len(result) == 1:
            x,y,w,h = result[0].box
            height,width = image.shape[:2]
            # Expanded ROI is at least about half of the 224 model input at
            # 96px face width. Do not spend a larger model hallucinating detail
            # in heavily enlarged or clipped faces.
            eligible = min(w,h) >= 96 and x >= 0 and y >= 0 and x+w <= width and y+h <= height
            if eligible:
                proposed = self.detail.refine(image, faces)
                for key,value in self.detail.last_profile.items():
                    if isinstance(value,(int,float)) and isinstance(profile.get(key),(int,float)):
                        profile[key] += value
                tier = 'fast_after_detail_missing'
                if len(proposed) == 1:
                    eye_distance = np.linalg.norm(result[0].points[1]-result[0].points[0])
                    differences = np.linalg.norm(result[0].points-proposed[0].points,axis=1)
                    ratio = float(differences.mean()/max(eye_distance,1e-9))
                    profile['detail_disagreement_interocular'] = ratio
                    accepted = bool(eye_distance > 0 and ratio <= .04 and differences.max() <= .08*eye_distance)
                    tier = 'detail_consistent' if accepted else 'fast_after_detail_disagreement'
                    if accepted:result = proposed
        # A change in decision reason is not a change in model. Preserve the
        # filter across fast-small / fast-disagreement / fast-multi decisions.
        source = 'osf3_detail' if accepted else 'osf0_fast'
        for face in result:
            face.landmark_source = source
        self.last_profile = {**profile,'quality_tier':tier,'detail_accepted':accepted}
        return result

    def close(self):
        try:self.detail.close()
        finally:self.fast.close()
