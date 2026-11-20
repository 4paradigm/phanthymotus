"""Read-only perspective framing of the robot's visible articulated geometry."""
from __future__ import annotations

import itertools
import math


class RobotCameraFit:
    """Project local geom boxes directly; never inflate a whole robot twice.

    MuJoCo geom_aabb contains a local center and half-size. Its eight corners
    enclose the actual mesh/primitive and follow geom_xmat/geom_xpos after
    kinematics. Static terrain and collision proxies hidden by FrameRenderer
    are not part of the subject. No model or physics state is changed here.
    """

    def __init__(self, model, geoms):
        import mujoco
        import numpy as np

        self.model = model
        geoms = np.asarray(geoms, dtype=int)
        visible = model.geom_rgba[geoms, 3] > 0
        meshes = model.geom_type == int(mujoco.mjtGeom.mjGEOM_MESH)
        # Match FrameRenderer's visual-mesh / collision-proxy convention.
        if np.any(meshes & (model.geom_group == 2)) and not np.any(meshes & (model.geom_group == 3)):
            visible &= model.geom_group[geoms] != 3
        self.geoms = geoms[visible]
        if not len(self.geoms):
            raise ValueError("机器人没有可取景的可见几何体")
        boxes = model.geom_aabb[self.geoms]
        signs = np.array(list(itertools.product((-1., 1.), repeat=3)))
        self.corners = boxes[:, None, :3] + boxes[:, None, 3:] * signs

    def points(self, data):
        import numpy as np

        rotations = data.geom_xmat[self.geoms].reshape(-1, 3, 3)
        return (np.einsum("gij,gkj->gki", rotations, self.corners)
                + data.geom_xpos[self.geoms, None, :]).reshape(-1, 3)

    def view(self, data, width, height, overrides=None):
        import numpy as np

        overrides = overrides or {}
        azimuth, elevation = overrides.get("azimuth", 130.), overrides.get("elevation", -25.)
        az, el = math.radians(azimuth), math.radians(elevation)
        # Same orthonormal frame as MuJoCo's free camera: the camera position
        # is lookat - distance*forward, and depth is distance + point.forward.
        right = np.array([math.sin(az), -math.cos(az), 0.])
        up = np.array([-math.cos(az) * math.sin(el), -math.sin(az) * math.sin(el), math.cos(el)])
        forward = np.array([math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)])
        axes = np.column_stack((right, up, forward))
        projected = self.points(data) @ axes
        center = (projected.min(axis=0) + projected.max(axis=0)) / 2
        relative = projected - center
        vertical = math.tan(math.radians(float(self.model.vis.global_.fovy)) / 2)
        horizontal = vertical * width / height
        # Fit each corner at its own depth, retaining an 8% border on each
        # side. A wide pose can be width-limited; never crop it to fill height.
        fill = .84
        requirements = np.maximum(abs(relative[:, 0]) / (horizontal * fill),
                                  abs(relative[:, 1]) / (vertical * fill)) - relative[:, 2]
        near = float(self.model.vis.map.znear * self.model.stat.extent)
        distance = max(.1, float(requirements.max()), float(near - relative[:, 2].min()) + .01)
        return {"lookat": (axes @ center).tolist(), "distance": distance,
                "azimuth": azimuth, "elevation": elevation, **overrides}
