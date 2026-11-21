"""Effect plugins produce affine sprite commands, shared by CPU and GL."""
from dataclasses import dataclass
from pathlib import Path
import json
import math
import cv2
import numpy as np


@dataclass
class Sprite:
    texture: np.ndarray  # premultiplied RGBA float32
    matrix: np.ndarray  # local sprite pixels -> image pixels
    opacity: float = 1.


class Effect:
    def __init__(self, cfg, root):
        self.cfg = cfg
        self.name = cfg["id"]
        asset = (root / cfg["asset"]).resolve()
        if not asset.is_relative_to(root.resolve()):
            raise ValueError("effect asset escapes resource directory")
        image = cv2.imread(str(asset), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 3 or image.shape[2] != 4 or max(image.shape[:2]) > 2048:
            raise ValueError("effect requires a local RGBA PNG <=2048px")
        rgba = image[:, :, [2, 1, 0, 3]].astype(np.float32)/255
        rgba[:, :, :3] *= rgba[:, :, 3:4]
        self.texture = rgba

    def commands(self, track, timestamp, pose):
        eyes = track.points[:2]
        axis = eyes[1]-eyes[0]
        distance = float(np.linalg.norm(axis))
        if distance < 8:
            return []
        axis /= distance
        normal = np.array([-axis[1], axis[0]])
        # A profile compresses the eye baseline; forehead ornaments should keep
        # a bounded head extent instead of collapsing with the hidden far eye.
        extent = max(distance, .36*float(track.face.box[2]))
        center = eyes.mean(0) + normal * extent * self.cfg.get("offset", 0.)
        width = extent * self.cfg.get("width", 2.4)
        height, tw = self.texture.shape[:2]
        scale = width/tw
        matrix = np.column_stack((axis*scale, normal*scale,
                                  center-axis*width/2-normal*height*scale/2))
        return [Sprite(self.texture, matrix)]

    def close(self):
        self.texture = None


class Glasses(Effect):
    def commands(self, track, timestamp, pose):
        # Bounded 2D accessory, deliberately fades on uncertain/extreme profiles.
        yaw = abs(pose["yaw_proxy"])
        if yaw >= .5:
            return []
        eyes = track.points[:2]
        axis = eyes[1]-eyes[0]
        distance = float(np.linalg.norm(axis))
        if distance < 8 or distance/max(float(track.face.box[2]),1.) < .14:
            return []
        axis /= distance
        normal = np.array([-axis[1], axis[0]])
        nose = track.points[2]-eyes.mean(0)
        vertical = float(np.dot(nose, normal))
        if not .08*distance < vertical < 1.4*distance:
            return []
        # Per-resource virtual nose anchor may lie outside the PNG. This makes
        # eye placement exact and permits bounded perspective-like nose shear.
        th, tw = self.texture.shape[:2]
        anchors = np.asarray(self.cfg.get("anchors", [[62/240,44/90],[178/240,44/90],[.5,105/90]]),float)*[tw,th]
        span = anchors[1,0]-anchors[0,0]
        rise = anchors[2,1]-anchors[:2,1].mean()
        sx = distance/span
        sy = float(np.clip(vertical/rise, .6*sx, 1.6*sx))
        shear = float(np.clip(np.dot(nose,axis)/rise, -.7*sx, .7*sx))
        linear = np.column_stack((axis*sx, normal*sy+axis*shear))
        translation = eyes[0]-linear@anchors[0]
        matrix = np.column_stack((linear, translation))
        return [Sprite(self.texture, matrix, min(1., max(0., (.5-yaw)/.2)))]


class Orbit(Effect):
    def commands(self, track, timestamp, pose):
        eyes = track.points[:2]
        distance = max(float(np.linalg.norm(eyes[1]-eyes[0])), .36*float(track.face.box[2]))
        if distance < 8:
            return []
        center = eyes.mean(0)
        phase = (timestamp % 10)*math.tau/2.5
        result = []
        for k in range(5):  # draw calls, never per-pixel Python loops
            theta = phase + k*math.tau/5
            point = center + distance*np.array([1.4*math.cos(theta), -1.+.6*math.sin(theta)])
            scale = distance*.28/self.texture.shape[1]
            matrix = np.array([[scale, 0, point[0]-self.texture.shape[1]*scale/2],
                               [0, scale, point[1]-self.texture.shape[0]*scale/2]])
            result.append(Sprite(self.texture, matrix, .65+.35*math.sin(theta)**2))
        return result


PLUGINS = {"sticker": Effect, "glasses": Glasses, "orbit": Orbit}


class Effects:
    def __init__(self, manifest=None):
        manifest = Path(manifest) if manifest else Path(__file__).with_name("assets") / "effects.json"
        config = json.loads(manifest.read_text())
        self.plugins = {}
        try:
            for cfg in config["effects"]:
                if cfg["id"] in self.plugins or cfg["plugin"] not in PLUGINS:
                    raise ValueError("duplicate/unknown effect")
                self.plugins[cfg["id"]] = PLUGINS[cfg["plugin"]](cfg, manifest.parent)
        except Exception:
            self.close()
            raise
        self.active = set(self.plugins)

    def select(self, names):
        if isinstance(names, str) or not set(names) <= self.plugins.keys():
            raise ValueError("unknown effects list")
        self.active = set(names)

    def commands(self, tracks, timestamp, poses):
        return [cmd for t in tracks for name, plugin in self.plugins.items()
                if name in self.active for cmd in plugin.commands(t, timestamp, poses[t.id])]

    def close(self):
        for p in self.plugins.values():
            p.close()
        self.plugins.clear()
