"""Optional official MediaPipe Face Landmarker CPU backend; local model only."""
import cv2
import numpy as np
from .backends import model_path
from .contracts import Face


class MediaPipe:
    def __init__(self, path, max_faces=4, sha256=None, mode="video"):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        self.mp = mp
        self.mode = mode
        self.timestamp_ms = -1
        self.requested_timestamp = None
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path(path, sha256),
                                             delegate=python.BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO if mode == "video" else vision.RunningMode.IMAGE, num_faces=max_faces,
            min_face_detection_confidence=.5, min_face_presence_confidence=.5,
            min_tracking_confidence=.5, output_face_blendshapes=False,
            output_facial_transformation_matrixes=False)
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.options = options
        self.factory = vision.FaceLandmarker.create_from_options
        self.providers = ["mediapipe-cpu"]

    def set_timestamp(self, timestamp):
        self.requested_timestamp = timestamp

    def detect(self, image):
        requested = round(self.requested_timestamp*1000) if self.requested_timestamp is not None else self.timestamp_ms+33
        self.timestamp_ms = max(self.timestamp_ms+1, requested)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        packet = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        result = (self.landmarker.detect_for_video(packet, self.timestamp_ms) if self.mode == "video"
                  else self.landmarker.detect(packet))
        h, w = image.shape[:2]
        faces = []
        for landmarks in result.face_landmarks:
            xy = np.array([(p.x*w, p.y*h) for p in landmarks], dtype=np.float64)
            minimum, maximum = xy.min(0), xy.max(0)
            points = np.array([(xy[33]+xy[133])/2, (xy[362]+xy[263])/2,
                               xy[1], xy[61], xy[291]])
            points[:2] = points[:2][np.argsort(points[:2, 0])]
            points[3:] = points[3:][np.argsort(points[3:, 0])]
            # Tasks does not expose per-point confidence; 1 is a presence sentinel,
            # never an accuracy score or a ranking against OSF confidence.
            face = Face(np.r_[minimum, maximum-minimum], points, 1., xy, 1., topology="mediapipe478")
            if face.valid():
                faces.append(face)
        return faces

    def close(self):
        self.landmarker.close()

    def reset(self):
        self.landmarker.close()
        self.landmarker = self.factory(self.options)


class HybridLandmarker:
    """YuNet proposals, OSF for small faces, optional independent MP image ROIs.

    An IMAGE graph avoids sharing a hidden single-face temporal state across
    different people. Each successful result retains its explicit topology.
    """
    def __init__(self, osf, path, min_face_pixels=120, sha256=None):
        self.osf = osf
        self.mesh = None
        try:
            self.mesh = MediaPipe(path, max_faces=1, sha256=sha256, mode="image")
        except Exception:
            self.osf.close()
            raise
        self.min_face_pixels = min_face_pixels
        self.providers = [*osf.providers, "mediapipe-cpu-roi"]
        self.fallback_reason = osf.fallback_reason

    def refine(self, image, faces):
        result = []
        height, width = image.shape[:2]
        for face in faces:
            x, y, w, h = face.box
            candidates = []
            if min(w, h) >= self.min_face_pixels:
                x1, y1 = max(0, int(x-.35*w)), max(0, int(y-.25*h))
                x2, y2 = min(width, int(x+1.35*w)), min(height, int(y+1.25*h))
                if x2 > x1 and y2 > y1:
                    candidates = self.mesh.detect(image[y1:y2, x1:x2])
                    if candidates:
                        center = face.points[:2].mean(0)
                        refined = min(candidates, key=lambda f: np.linalg.norm(f.points[:2].mean(0)+[x1,y1]-center))
                        refined.points += (x1, y1)
                        refined.landmarks += (x1, y1)
                        # Detector box remains the association reference.
                        refined.box = face.box.copy()
                        refined.confidence = face.confidence
                        result.append(refined)
            if not candidates:
                result.extend(self.osf.refine(image, [face]))
        return result

    def close(self):
        try:
            if self.mesh:
                self.mesh.close()
        finally:
            self.osf.close()
