"""Bounded RGB-D observation assistance using only NumPy."""

import numpy as np


def _valid_depth(depth):
    return (depth >= 200) & (depth <= 8000)


def _bounded_transform(transform):
    return (transform.shape == (4, 4) and np.isfinite(transform).all()
            and np.allclose(transform[3], [0, 0, 0, 1], atol=1e-7)
            and np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5)
            and abs(np.linalg.det(transform[:3, :3])-1) < 1e-5
            and np.linalg.norm(transform[:3, 3]) <= .3
            and np.trace(transform[:3, :3]) >= 1+2*np.cos(np.deg2rad(20)))


def _increment(delta):
    """Left SE(3) increment, with translation followed by rotation tangent."""
    vector = delta[3:]
    theta = np.linalg.norm(vector)
    x, y, z = vector
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    a = np.sinc(theta/np.pi)
    b = .5 if theta < 1e-8 else (1-np.cos(theta))/theta**2
    c = 1/6 if theta < 1e-8 else (theta-np.sin(theta))/theta**3
    result = np.eye(4)
    result[:3, :3] += a*skew+b*(skew @ skew)
    result[:3, 3] = (np.eye(3)+b*skew+c*(skew @ skew)) @ delta[:3]
    return result


class DepthRegistration:
    """Projective point-to-plane ICP for small, inertially initialized motion.

    Axial uint16 millimetre images share calibrated intrinsics. Returned
    T_reference_current maps current optical coordinates to the reference.
    Covariance uses a left tangent at the reference: xyz metres then radians.
    Single planes and any poorly observed six-axis solution fail closed.
    A bounded coarse multi-start check rejects competing pose basins; it is
    not a proof of global uniqueness for arbitrary repeated geometry.
    """

    def __init__(self, calibration):
        self.K = np.asarray(calibration, dtype=float)
        if (self.K.shape != (3, 3) or not np.isfinite(self.K).all()
                or self.K[0, 0] <= 0 or self.K[1, 1] <= 0
                or not np.allclose(self.K[2], [0, 0, 1])
                or self.K[0, 1] != 0 or self.K[1, 0] != 0):
            raise ValueError('Depth registration requires finite pinhole intrinsics')

    def _points(self, depth):
        y, x = np.indices(depth.shape)
        z = depth.astype(float)*.001
        return np.stack(((x-self.K[0, 2])*z/self.K[0, 0],
                         (y-self.K[1, 2])*z/self.K[1, 1], z), -1)

    def _correspondences(self, points, target, normals, transform):
        moved = points @ transform[:3, :3].T+transform[:3, 3]
        height, width = target.shape[:2]
        valid = moved[:, 2] > .2
        # The pose is bounded, but points may still project behind the camera.
        z = np.maximum(moved[:, 2], .001)
        u = np.rint(self.K[0, 0]*moved[:, 0]/z+self.K[0, 2]).astype(int)
        v = np.rint(self.K[1, 1]*moved[:, 1]/z+self.K[1, 2]).astype(int)
        valid &= (u >= 0) & (u < width) & (v >= 0) & (v < height)
        moved, u, v = moved[valid], u[valid], v[valid]
        reference, normal = target[v, u], normals[v, u]
        difference = moved-reference
        residual = np.sum(normal*difference, axis=1)
        valid = ((np.linalg.norm(normal, axis=1) > .9)
                 & (np.linalg.norm(difference, axis=1) < .12)
                 & (np.abs(residual) < .04))
        moved, normal, residual = moved[valid], normal[valid], residual[valid]
        jacobian = np.concatenate((normal, np.cross(moved, normal)), axis=1)
        return jacobian, residual

    def _fit(self, points, target, normals, transform, iterations):
        for _ in range(iterations):
            jacobian, residual = self._correspondences(points, target, normals, transform)
            if len(residual) < 100 or len(residual)/len(points) < .35:
                return None
            # Huber weighting suppresses mismatched surfaces without declaring
            # those pixels independent precision measurements.
            weights = np.minimum(1., .01/np.maximum(np.abs(residual), 1e-10))
            hessian = jacobian.T @ (weights[:, None]*jacobian)
            eigenvalues = np.linalg.eigvalsh(hessian)
            if eigenvalues[0] <= eigenvalues[-1]*1e-4:
                return None
            delta = np.linalg.solve(hessian, -jacobian.T @ (weights*residual))
            transform = _increment(delta) @ transform
            if not _bounded_transform(transform):
                return None
            if np.linalg.norm(delta[:3]) < 1e-5 and np.linalg.norm(delta[3:]) < 1e-5:
                break
        return transform

    def _ambiguous(self, points, target, normals, transform, inliers, rmse):
        # At most 19 starts, five iterations each, roughly 600 source points.
        # Multiple metric scales avoid assuming a known repetition period.
        starts = [np.eye(4)]
        for axis in range(3):
            for distance in (-.1, -.04, .04, .1):
                seed = transform.copy()
                seed[axis, 3] += distance
                starts.append(seed)
            for angle in (-np.deg2rad(5.), np.deg2rad(5.)):
                delta = np.zeros(6)
                delta[3+axis] = angle
                starts.append(_increment(delta) @ transform)
        coarse = points[::max(1, int(np.ceil(len(points)/600)))]
        for seed in starts:
            if not _bounded_transform(seed):
                continue
            alternative = self._fit(coarse, target, normals, seed, 5)
            if alternative is None:
                continue
            translation = np.linalg.norm(alternative[:3, 3]-transform[:3, 3])
            rotation = alternative[:3, :3] @ transform[:3, :3].T
            if translation < .02 and np.trace(rotation) > 1+2*np.cos(np.deg2rad(2.)):
                continue
            # Recheck contenders on all samples, so a coarse-grid alias cannot
            # manufacture evidence for an alternative. Require similar support.
            _, residual = self._correspondences(points, target, normals, alternative)
            if (len(residual) >= max(100, .85*inliers)
                    and np.sqrt(np.mean(residual**2)) <= max(.0015, 1.25*rmse)):
                return True
        return False

    def register(self, current_depth, reference_depth, initial_transform):
        current, reference = np.asarray(current_depth), np.asarray(reference_depth)
        try:
            transform = np.array(initial_transform, dtype=float, copy=True)
        except (TypeError, ValueError):
            return None
        if (current.ndim != 2 or current.shape != reference.shape
                or min(current.shape) < 8 or current.dtype != np.uint16
                or reference.dtype != np.uint16 or not _bounded_transform(transform)):
            return None
        valid_current, valid_reference = _valid_depth(current), _valid_depth(reference)
        if valid_current.mean() < .15 or valid_reference.mean() < .15:
            return None
        target = self._points(reference)
        normals = np.zeros_like(target)
        interior = (valid_reference[1:-1, 1:-1] & valid_reference[1:-1, :-2]
                    & valid_reference[1:-1, 2:] & valid_reference[:-2, 1:-1]
                    & valid_reference[2:, 1:-1])
        center = target[1:-1, 1:-1, 2]
        for neighbor in (target[1:-1, :-2, 2], target[1:-1, 2:, 2],
                         target[:-2, 1:-1, 2], target[2:, 1:-1, 2]):
            interior &= np.abs(neighbor-center) < .05
        cross = np.cross(target[1:-1, 2:]-target[1:-1, :-2],
                         target[2:, 1:-1]-target[:-2, 1:-1])
        length = np.linalg.norm(cross, axis=2)
        interior &= length > 1e-10
        normals[1:-1, 1:-1][interior] = cross[interior]/length[interior, None]
        stride = max(1, int(np.ceil(np.sqrt(current.size/4800))))
        points = self._points(current)[::stride, ::stride]
        points = points[valid_current[::stride, ::stride]]
        if len(points) < 100:
            return None
        transform = self._fit(points, target, normals, transform, 15)
        if transform is None:
            return None
        jacobian, residual = self._correspondences(points, target, normals, transform)
        if len(residual) < 100 or len(residual)/len(points) < .35:
            return None
        rmse = float(np.sqrt(np.mean(residual**2)))
        hessian = jacobian.T @ jacobian
        eigenvalues = np.linalg.eigvalsh(hessian)
        if rmse >= .02 or eigenvalues[0] <= eigenvalues[-1]*1e-4:
            return None
        if self._ambiguous(points, target, normals, transform, len(residual), rmse):
            return None
        # Do not let thousands of correlated depth pixels imply submillimetre
        # pose certainty. Retain inverse-Hessian shape plus explicit floors.
        covariance = max(rmse**2, .005**2)*np.linalg.inv(hessian)
        covariance += np.diag([.01**2]*3+[np.deg2rad(1.)**2]*3)
        return {'transform': transform, 'covariance': (covariance+covariance.T)*.5,
                'inliers': len(residual), 'rmse': rmse,
                'overlap': len(residual)/len(points),
                'condition': float(eigenvalues[-1]/eigenvalues[0])}


def observation_quality(color, depth):
    """Basic RGB-D quality gate; native inliers/covariance remain caller gates.

    Color is uint8 RGB/BGR or grayscale (channel order is immaterial here).
    Larger weights conservatively inflate accepted visual covariance. These
    image heuristics and model floors require real sensor validation.
    """
    image, depth = np.asarray(color), np.asarray(depth)
    valid_depth = (depth.ndim == 2 and depth.size and depth.dtype == np.uint16
                   and depth.shape == image.shape[:2])
    fraction = float(_valid_depth(depth).mean()) if valid_depth else 0.
    contrast, sharpness = 0., 0.
    valid_image = (image.dtype == np.uint8 and image.ndim in (2, 3)
                   and image.shape[0] >= 3 and image.shape[1] >= 3
                   and (image.ndim == 2 or image.shape[2] in (3, 4)))
    if valid_image:
        gray = image.astype(float) if image.ndim == 2 else image[..., :3].mean(axis=2)
        contrast = float(gray.std())
        laplacian = (gray[2:, 1:-1]+gray[:-2, 1:-1]+gray[1:-1, 2:]
                     + gray[1:-1, :-2]-4*gray[1:-1, 1:-1])
        sharpness = float(laplacian.var())
    reason = ('invalid_image' if not valid_image else 'low_contrast' if contrast < 8
              else 'blur_or_low_texture' if sharpness < 12
              else 'insufficient_depth' if fraction < .15 else 'usable')
    weight = max(1., (25/max(contrast, 1.))**2,
                 100/max(sharpness, 1.), .6/max(fraction, .01))
    return {'visual_usable': reason == 'usable', 'weight': float(min(weight, 100.)),
            'depth_fraction': fraction, 'contrast': contrast, 'sharpness': sharpness,
            'reason': reason}
