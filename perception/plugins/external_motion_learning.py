"""Calibrate body-frame external odometry increments against native visual motion."""
from collections import deque
from copy import copy
from time import perf_counter_ns

import numpy as np


_PARAMETER_SCALE = np.array([.25] * 6 + [.20] * 3 + [.35] * 3)
_MEASUREMENT_NOISE = np.array([.025] * 3 + [.05] * 3)
_WINDOW_NS, _SOLVE_INTERVAL_NS = 10_000_000_000, 500_000_000
_MAX_INTERVALS, _HUBER = 64, 2.5


def _skew(v):
    x, y, z = np.moveaxis(v, -1, 0)
    zero = np.zeros_like(x)
    return np.stack((zero, -z, y, z, zero, -x, -y, x, zero), axis=-1).reshape(v.shape[:-1] + (3, 3))


def se3_exp(increment):
    """Body displacement from a twist integral in [linear, angular] order."""
    v, w = np.split(np.asarray(increment, dtype=float), 2, axis=-1)
    angle = np.linalg.norm(w, axis=-1)
    small = angle < 1e-3
    safe = np.where(small, 1., angle)
    squared = angle**2
    sin_ratio = np.sinc(angle / np.pi)
    cos_ratio = .5 * np.sinc(angle / (2. * np.pi))**2
    jacobian_ratio = np.where(small, 1. / 6. - squared / 120., (safe - np.sin(safe)) / safe**3)
    skew = _skew(w)
    second = skew @ skew
    result = np.broadcast_to(np.eye(4), w.shape[:-1] + (4, 4)).copy()
    result[..., :3, :3] = np.eye(3) + sin_ratio[..., None, None] * skew + cos_ratio[..., None, None] * second
    first_cross = np.cross(w, v)
    result[..., :3, 3] = (v + cos_ratio[..., None] * first_cross +
                          jacobian_ratio[..., None] * np.cross(w, first_cross))
    return result


def se3_right_jacobian(increment):
    """Map additive [linear, angular] perturbations to the local tangent of Exp."""
    vectors = np.asarray(increment, dtype=float).reshape(2, 3)
    v, w = vectors
    squared = w @ w
    if squared < 1e-4:
        # b = (1-cos(theta))/theta², c = (theta-sin(theta))/theta³.
        # db and dc are the derivatives with respect to theta, divided by theta.
        b = .5 - squared / 24. + squared**2 / 720.
        c = 1. / 6. - squared / 120. + squared**2 / 5040.
        db = -1. / 12. + squared / 180. - squared**2 / 6720.
        dc = -1. / 60. + squared / 1260. - squared**2 / 60480.
    else:
        angle = np.sqrt(squared)
        sine_ratio = np.sin(angle) / angle
        b = 2. * (np.sin(.5 * angle) / angle)**2
        c = (1. - sine_ratio) / squared
        db = (sine_ratio - 2. * b) / squared
        dc = (b - 3. * c) / squared
    linear, angular = _skew(vectors)
    second = angular @ angular
    rotation = np.eye(3) - b * angular + c * second
    # ad([v,w]) has repeated diagonal blocks [w]x and upper block [v]x.
    # Applying the Jacobian series to it gives Jr_SO3 on the diagonal
    # and its directional derivative along v in the upper right block.
    coupling = (-b * linear + c * (angular @ linear + linear @ angular)
                + (w @ v) * (-db * angular + dc * second))
    jacobian = np.zeros((6, 6))
    jacobian[:3, :3] = jacobian[3:, 3:] = rotation
    jacobian[:3, 3:] = coupling
    return jacobian


def se3_log(transform):
    """Principal SE(3) logarithm with the inverse rotational translation Jacobian."""
    transform = np.asarray(transform, dtype=float)
    rotation, translation = transform[..., :3, :3], transform[..., :3, 3]
    axis = np.stack((rotation[..., 2, 1] - rotation[..., 1, 2],
                     rotation[..., 0, 2] - rotation[..., 2, 0],
                     rotation[..., 1, 0] - rotation[..., 0, 1]), axis=-1) * .5
    sine = np.linalg.norm(axis, axis=-1)
    cosine = (np.trace(rotation, axis1=-2, axis2=-1) - 1.) * .5
    angle = np.arctan2(sine, cosine)
    safe_sine = np.where(sine < 1e-9, 1., sine)
    w = axis * np.where(angle < 1e-6, 1. + angle**2 / 6., angle / safe_sine)[..., None]
    near_pi = angle > np.pi - 1e-5
    if np.any(near_pi):
        singular = rotation[near_pi]
        _, vectors = np.linalg.eigh((singular + np.swapaxes(singular, -1, -2)) * .5)
        direction = vectors[..., -1]
        sign = np.where(np.sum(direction * axis[near_pi], axis=-1) < 0., -1., 1.)
        w[near_pi] = direction * (angle[near_pi] * sign)[..., None]
    small = angle < 1e-3
    safe_angle = np.where(small, 1., angle)
    ratio = np.where(small, 1. / 12. + angle**2 / 720.,
                     (1. - .5 * safe_angle / np.tan(.5 * safe_angle)) / safe_angle**2)
    first_cross = np.cross(w, translation)
    v = translation - .5 * first_cross + ratio[..., None] * np.cross(w, first_cross)
    return np.concatenate((v, w), axis=-1)


def _velocity(velocity, observed):
    u = np.asarray(velocity, dtype=float)
    mask = np.ones(6, dtype=bool) if observed is None else np.asarray(observed, dtype=bool)
    if u.shape != (6,) or mask.shape != (6,) or not np.isfinite(u[mask]).all():
        raise ValueError('external body velocity requires six finite observed components')
    return np.where(mask, u, 0.), mask


class ExternalMotionLearner:
    """A bounded sequential estimator. Deep copies preserve its replay checkpoint."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.parameters = np.zeros(12)
        self.covariance = np.diag(_PARAMETER_SCALE**2)
        self.window = deque(maxlen=40)
        self.last_solve_ns = self.effective_after_ns = None
        self.rank = 0
        self.observable_directions = np.empty((0, 12))
        self.last_solve_ms = 0.

    def checkpoint(self):
        state = copy(self)
        # Observations and parameter arrays are replaced by the estimator.
        state.window = self.window.copy()
        return state

    def corrected_velocity(self, velocity, observed=None):
        raw, mask = _velocity(velocity, observed)
        return (raw * np.exp(self.parameters[:6]) + self.parameters[6:]) * mask

    def correct(self, velocity, dt_seconds, observed=None):
        return se3_exp(self.corrected_velocity(velocity, observed) * dt_seconds)

    def velocity_parameter_jacobian(self, velocity, observed=None):
        raw, mask = _velocity(velocity, observed)
        return np.c_[np.diag(raw * np.exp(self.parameters[:6])), np.diag(mask.astype(float))]

    def add(self, start_ns, end_ns, intervals, visual_delta, noise=None, *, solve=None):
        """Add a closed interval; raw body velocity pieces must cover it exactly."""
        if (not self.enabled or start_ns >= end_ns or end_ns-start_ns > _WINDOW_NS or
                self.window and start_ns < self.window[-1][1]):
            return False
        pieces = []
        for velocity, duration, observed in intervals:
            if not np.isfinite(duration) or duration <= 0 or len(pieces) >= _MAX_INTERVALS:
                return False
            raw, mask = _velocity(velocity, observed)
            pieces.append((raw, mask, duration))
        if not pieces or abs(sum(piece[2] for piece in pieces) - (end_ns-start_ns)/1e9) > 1e-6:
            return False
        axes = np.logical_and.reduce([piece[1] for piece in pieces])
        target = np.asarray(visual_delta, dtype=float)
        sigma = _MEASUREMENT_NOISE if noise is None else np.asarray(noise, dtype=float)
        if (not axes.any() or target.shape != (4, 4) or not np.isfinite(target).all() or
                sigma.shape != (6,) or not np.isfinite(sigma).all() or np.any(sigma <= 0)):
            return False
        missing_rotation = ~axes[3:]
        if missing_rotation.any():
            # A projected net rotation can acquire an apparent measured-axis
            # component from noncommuting turns on omitted axes. Train only
            # when the visual endpoint is consistent with the reported angular
            # subspace to one noise sigma, capped at nominal angular precision.
            # Endpoints alone cannot detect an omitted excursion that closes.
            rotation = se3_log(target)[3:]
            tolerance = np.minimum(sigma[3:], _MEASUREMENT_NOISE[3:])
            if np.linalg.norm(rotation[missing_rotation] / tolerance[missing_rotation]) > 1.:
                return False
        self.window.append((start_ns, end_ns, tuple(pieces), axes, target.copy(),
                            np.maximum(sigma, _MEASUREMENT_NOISE)))
        while self.window[0][0] < end_ns-_WINDOW_NS:
            self.window.popleft()
        if len(self.window) < 3 or (self.last_solve_ns is not None and
                                    end_ns-self.last_solve_ns < _SOLVE_INTERVAL_NS):
            return False
        self.last_solve_ns = end_ns
        if solve is None:
            return self.fit()
        result = solve(self)
        if result is None:
            return False
        self.parameters, self.covariance = result.parameters, result.covariance
        self.rank, self.observable_directions = result.rank, result.observable_directions
        self.last_solve_ms, self.effective_after_ns = result.last_solve_ms, result.effective_after_ns
        return self.effective_after_ns == end_ns

    def fit(self):
        start_wall = perf_counter_ns()
        try:
            updated = self._solve()
        except (np.linalg.LinAlgError, FloatingPointError):
            updated = False
        self.last_solve_ms = (perf_counter_ns() - start_wall) / 1e6
        if updated:
            self.effective_after_ns = self.last_solve_ns
        return updated

    def info(self):
        return dict(enabled=self.enabled, scale=np.exp(self.parameters[:6]).tolist(),
                    bias=self.parameters[6:].tolist(), rank=self.rank, samples=len(self.window),
                    window_begin_ns=self.window[0][0] if self.window else None,
                    effective_after_ns=self.effective_after_ns, solve_ms=self.last_solve_ms,
                    observable_directions=self.observable_directions.tolist(),
                    parameter_sigma=np.sqrt(np.diag(self.covariance)).tolist())

    def _residual_model(self):
        """Batch all finite differences while preserving each sample's ordered product."""
        count = len(self.window)
        steps = max(len(row[2]) for row in self.window)
        raw, mask = np.zeros((steps, count, 6)), np.zeros((steps, count, 6), dtype=bool)
        durations = np.zeros((steps, count, 1))
        axes = np.stack([row[3] for row in self.window])
        target = np.stack([row[4] for row in self.window])
        sigmas = np.stack([row[5] for row in self.window])
        for sample, row in enumerate(self.window):
            for step, (velocity, observed, duration) in enumerate(row[2]):
                raw[step, sample], mask[step, sample], durations[step, sample, 0] = velocity, observed, duration
        partial = ~np.all(axes, axis=1)
        partial_target = se3_log(target[partial])

        def evaluate(normalized):
            parameters = normalized * _PARAMETER_SCALE
            velocities = (np.exp(parameters[:, None, None, :6]) * raw +
                          parameters[:, None, None, 6:]) * mask
            pose = np.eye(4)
            for increment in np.swapaxes(se3_exp(velocities * durations), 0, 1):
                pose = pose @ increment
            inverse_rotation = np.swapaxes(pose[..., :3, :3], -1, -2)
            relative = np.empty_like(pose)
            relative[..., :3, :3] = inverse_rotation @ target[:, :3, :3]
            relative[..., :3, 3] = (inverse_rotation @
                                    (target[:, :3, 3] - pose[..., :3, 3])[..., None])[..., 0]
            relative[..., 3, :] = [0., 0., 0., 1.]
            residual = se3_log(relative)
            # With a partial twist, compare only the reported body components.
            # Unreported motion must not turn into residuals on absent axes.
            residual[:, partial] = partial_target - se3_log(pose[:, partial])
            return np.where(axes, residual / sigmas, 0.)

        return evaluate

    def _solve(self):
        evaluate = self._residual_model()
        point = self.parameters / _PARAMETER_SCALE
        difference = 1e-5
        perturbations = np.r_[np.zeros((1, 12)), difference * np.eye(12)]
        for _ in range(5):
            values = evaluate(point + perturbations)
            residual = values[0]
            weights = np.sqrt(np.minimum(1., _HUBER / np.maximum(np.linalg.norm(residual, axis=1), 1e-12)))
            jacobian = (values[1:] - residual) / difference
            jacobian *= weights[None, :, None]
            jacobian = jacobian.reshape(12, -1).T
            scale, bias = jacobian[:, :6], jacobian[:, 6:]
            # The norm perpendicular to the other column measures the independent
            # effect of a one-prior-sigma change, in window noise sigmas. Keep
            # bias fixed at steady speed; near zero, bias alone can be measured.
            scale_norm = np.linalg.norm(scale, axis=0)
            bias_norm = np.linalg.norm(bias, axis=0)
            product = scale_norm * bias_norm
            cosine = np.sum(scale * bias, axis=0) / np.where(product > 0., product, 1.)
            sine = np.sqrt(np.maximum(0., 1. - cosine**2))
            fit_bias = (scale_norm == 0.) | (bias_norm * sine >= 1.)
            fit_scale = ~fit_bias | (scale_norm * sine >= 1.)
            design = np.column_stack((scale * fit_scale, bias * fit_bias))
            left, singular, right = np.linalg.svd(design, full_matrices=False)
            rank = np.count_nonzero(singular > max(.02, singular[0] * 1e-3))
            if not rank:
                self.rank = 0
                self.observable_directions = np.empty((0, 12))
                return False
            basis, scales = right[:rank].T, singular[:rank]
            weighted = (weights[:, None] * residual).ravel()
            step = -basis @ ((left[:, :rank].T @ weighted) / scales)
            step /= max(1., np.linalg.norm(step))
            if np.linalg.norm(step) <= 1e-6:
                break
            objective = weighted @ weighted
            accepted = False
            for fraction in (1., .5, .25):
                trial = point + fraction * step
                if np.any(np.abs(trial[:6] * _PARAMETER_SCALE[:6]) > np.log(4.)):
                    continue
                cost = np.linalg.norm(evaluate(trial[None, :])[0] * weights[:, None])**2
                if cost < objective:
                    point = trial
                    accepted = True
                    break
            if not accepted:
                break
        # Use only this window's information. Shared visual endpoints and model
        # mismatch make the covariance an engineering estimate.
        # Propagate the uncertainty of held parameters through the conditional
        # fit, e.g. uncertain bias also limits scale learned at a single speed.
        measured = left[:, :rank].T @ jacobian
        null = np.eye(12) - (basis / scales) @ measured
        factors = _PARAMETER_SCALE[:, None] * _PARAMETER_SCALE
        prior = self.covariance / factors
        observable = (basis * (2. / scales**2)) @ basis.T
        self.covariance = (null @ prior @ null.T + observable) * factors
        self.rank = int(rank)
        measured_basis, _ = np.linalg.qr(measured.T)
        self.observable_directions = (measured_basis * _PARAMETER_SCALE[:, None]).T
        self.parameters = point * _PARAMETER_SCALE
        return True
