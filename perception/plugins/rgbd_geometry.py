"""Bounded RGB-D observation assistance using only NumPy."""

import threading

import numpy as np


# Compare translation and rotation in units of the existing model floors.
# Raw Hessian eigenvalues otherwise mix metres and radians.
_TANGENT_SCALE = np.array([.01]*3+[np.deg2rad(1.)]*3)
_TANGENT_FLOOR = np.diag(_TANGENT_SCALE**2)


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


def _skew(vector):
    x, y, z = vector
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def _rotation_log(rotation):
    # Both poses are bounded to 20 degrees, so their difference is far from pi.
    angle = np.arccos(np.clip((np.trace(rotation)-1.)/2., -1., 1.))
    vector = np.array([rotation[2, 1]-rotation[1, 2], rotation[0, 2]-rotation[2, 0],
                       rotation[1, 0]-rotation[0, 1]])
    return vector*(.5 if angle < 1e-7 else angle/(2*np.sin(angle)))


def _rotation_jacobian(vector):
    """Left SO(3) derivative of an additive, seed-anchored rotation vector."""
    angle = np.linalg.norm(vector)
    cross = _skew(vector)
    b = .5-angle**2/24. if angle < 1e-5 else (1-np.cos(angle))/angle**2
    c = 1/6-angle**2/120. if angle < 1e-5 else (angle-np.sin(angle))/angle**3
    return np.eye(3)+b*cross+c*(cross@cross)


def _pose_modes(jacobian, translation, weights=None):
    """Observable camera-origin pose directions, normalized by physical scales."""
    pose_jacobian = jacobian.copy()
    pose_jacobian[:, 3:] -= np.cross(translation, jacobian[:, :3])
    scaled = pose_jacobian*_TANGENT_SCALE
    weighted = scaled if weights is None else weights[:, None]*scaled
    hessian = np.einsum('ni,nj->ij', scaled, weighted)
    values, vectors = np.linalg.eigh((hessian+hessian.T)*.5)
    if not np.isfinite(values).all():
        raise ValueError('nonfinite depth observability')
    retained = values > max(1e-12, values[-1]*1e-4)
    return pose_jacobian, values, vectors[:, retained], retained


class DepthRegistration:
    """Projective point-to-plane ICP for small, inertially initialized motion.

    Axial uint16 millimetre images share calibrated intrinsics. Returned
    T_reference_current maps current optical coordinates to the reference.
    Measurements use a left tangent at the reference: xyz metres then radians.
    Partial geometry returns only its observable projection and covariance;
    unobserved pose coordinates retain the supplied inertial seed.
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
        seed = transform.copy()
        basis = coefficients = None
        for _ in range(iterations):
            jacobian, residual = self._correspondences(points, target, normals, transform)
            if len(residual) < 100 or len(residual)/len(points) < .35:
                return None
            # Huber weighting suppresses mismatched surfaces without declaring
            # those pixels independent precision measurements.
            weights = np.minimum(1., .01/np.maximum(np.abs(residual), 1e-10))
            pose_jacobian, _, observed, _ = _pose_modes(jacobian, transform[:3, 3], weights)
            if observed.shape[1] == 0:
                return None
            if basis is None:
                basis = observed
                coefficients = np.zeros(basis.shape[1])
            elif observed.shape[1] != basis.shape[1]:
                # Changing support cannot silently add/remove a measured mode
                # after that mode has already moved in a different gauge.
                return None
            anchored = _TANGENT_SCALE*(basis@coefficients)
            pose_jacobian[:, 3:] = pose_jacobian[:, 3:]@_rotation_jacobian(anchored[3:])
            reduced = (pose_jacobian*_TANGENT_SCALE)@basis
            # Avoid threaded BLAS dispatch for these small reductions.
            hessian = np.einsum('ni,nj->ij', reduced, weights[:, None]*reduced)
            values = np.linalg.eigvalsh(hessian)
            if values[0] <= max(1e-12, values[-1]*1e-4):
                return None
            step = np.linalg.solve(hessian, -np.einsum('ni,n->i', reduced, weights*residual))
            coefficients += step
            anchored = _TANGENT_SCALE*(basis@coefficients)
            transform = seed.copy()
            transform[:3, :3] = _increment(np.r_[np.zeros(3), anchored[3:]])[:3, :3]@seed[:3, :3]
            transform[:3, 3] += anchored[:3]
            if not _bounded_transform(transform):
                return None
            delta = _TANGENT_SCALE*(basis@step)
            if np.linalg.norm(delta[:3]) < 1e-5 and np.linalg.norm(delta[3:]) < 1e-5:
                break
        return transform, basis.shape[1]

    def _ambiguous(self, points, target, normals, transform, inliers, rmse, basis):
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
            alternative = alternative[0]
            rotation = alternative[:3, :3] @ transform[:3, :3].T
            difference = np.r_[alternative[:3, 3]-transform[:3, 3], _rotation_log(rotation)]
            observed = _TANGENT_SCALE*(basis@(basis.T@(difference/_TANGENT_SCALE)))
            if np.linalg.norm(observed[:3]) < .02 and np.linalg.norm(observed[3:]) < np.deg2rad(2.):
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
        # A wider angular baseline prevents millimetre depth quantization from
        # manufacturing extra normal directions on a tilted plane. Every pixel
        # along the stencil must still be valid and on the same smooth surface.
        radius = min(max(2, int(np.ceil(max(self.K[0, 0], self.K[1, 1])/150.))),
                     (min(reference.shape)-1)//2)
        height, width = reference.shape
        rows, columns = slice(radius, height-radius), slice(radius, width-radius)
        interior = valid_reference[rows, columns].copy()
        center = target[rows, columns, 2]
        for offset in range(-radius, radius+1):
            for row, column in ((slice(radius+offset, height-radius+offset), columns),
                                (rows, slice(radius+offset, width-radius+offset))):
                interior &= valid_reference[row, column] & (np.abs(target[row, column, 2]-center) < .05)
        cross = np.cross(target[rows, 2*radius:]-target[rows, :-2*radius],
                         target[2*radius:, columns]-target[:-2*radius, columns])
        length = np.linalg.norm(cross, axis=2)
        interior &= length > 1e-10
        normals[rows, columns][interior] = cross[interior]/length[interior, None]
        stride = max(1, int(np.ceil(np.sqrt(current.size/4800))))
        points = self._points(current)[::stride, ::stride]
        points = points[valid_current[::stride, ::stride]]
        if len(points) < 100:
            return None
        fitted = self._fit(points, target, normals, transform, 15)
        if fitted is None:
            return None
        transform, rank = fitted
        jacobian, residual = self._correspondences(points, target, normals, transform)
        if len(residual) < 100 or len(residual)/len(points) < .35:
            return None
        rmse = float(np.sqrt(np.mean(residual**2)))
        hessian = jacobian.T @ jacobian
        _, eigenvalues, basis, retained = _pose_modes(jacobian, transform[:3, 3])
        if rmse >= .02 or basis.shape[1] != rank:
            return None
        if self._ambiguous(points, target, normals, transform, len(residual), rmse, basis):
            return None
        # Do not let thousands of correlated depth pixels imply submillimetre
        # pose certainty. Retain inverse-Hessian shape plus explicit floors.
        tangent = np.eye(6)
        tangent[:3, 3:] = _skew(transform[:3, 3])
        inverse = np.eye(6)
        inverse[:3, 3:] = -tangent[:3, 3:]
        projection = (basis.T/_TANGENT_SCALE)@inverse
        lift = tangent@(_TANGENT_SCALE[:, None]*basis)
        covariance = (max(rmse**2, .005**2)*np.diag(1/eigenvalues[retained]) +
                      projection@_TANGENT_FLOOR@projection.T)
        result = dict(transform=transform, rank=rank, projection=projection,
                      observable_lift=lift, projected_covariance=(covariance+covariance.T)*.5,
                      eigenvalues=eigenvalues.tolist(), inliers=len(residual), rmse=rmse,
                      overlap=len(residual)/len(points),
                      condition=float(eigenvalues[-1]/eigenvalues[retained][0]))
        if rank == 6:
            covariance = max(rmse**2, .005**2)*np.linalg.inv(hessian)+_TANGENT_FLOOR
            result['covariance'] = (covariance+covariance.T)*.5
        return result


class GeometryWorker:
    """One registration thread: one running, one pending, one finished result.

    Keeps ICP off the caller's estimator lock. A newer submission replaces an
    unstarted one; results are claimed only by their exact token, so a caller
    that moved on (or closed) never receives a stale correction.
    """

    def __init__(self, registration):
        self.registration = registration
        self.condition = threading.Condition()
        self.pending = self.running = self.finished = None
        self.dropped = 0
        self.closed = False
        self.thread = threading.Thread(target=self._run, name='rgbd_geometry', daemon=True)
        self.thread.start()

    def submit(self, token, current, reference, initial):
        with self.condition:
            if self.closed:
                return False
            if self.pending is not None:
                self.dropped += 1
            self.pending = (token, current, reference, initial)
            self.condition.notify()
            return True

    def take(self, token):
        with self.condition:
            if self.finished is not None and self.finished[0] == token:
                _, result, error = self.finished
                self.finished = None
                return 'done', (result, error)
            if (self.running == token or
                    (self.pending is not None and self.pending[0] == token)):
                return 'pending', None
            return 'missing', None

    def close(self, timeout=5.):
        with self.condition:
            self.closed = True
            self.pending = self.finished = None
            self.condition.notify()
        self.thread.join(timeout=timeout)
        return not self.thread.is_alive()

    def _run(self):
        while True:
            with self.condition:
                while self.pending is None and not self.closed:
                    self.condition.wait()
                if self.closed:
                    return
                token, current, reference, initial = self.pending
                self.pending, self.running = None, token
            result = error = None
            try:
                result = self.registration.register(current, reference, initial)
            except Exception as failure:  # reported to the filter, never raised in its thread
                error = f'{type(failure).__name__}: {failure}'
            with self.condition:
                self.running = None
                if not self.closed:
                    self.finished = (token, result, error)


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
        if image.ndim == 2:
            gray = image.astype(float)
        else:
            channel_sum = np.add(image[..., 0], image[..., 1], dtype=np.uint16)
            channel_sum += image[..., 2]
            gray = channel_sum / 3.
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
