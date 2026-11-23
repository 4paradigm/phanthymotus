"""Bounded capture-time inertial assistance; predictions never claim SLAM tracking."""
from collections import deque
from copy import deepcopy

import numpy as np

from .rgbd_imu import MAX_IMU_GAP_NS, ImuAttitude, _vector, rotation_from_quaternion, translation_extrinsics
from .rgbd_odometry_contract import SIGMA_LIMIT_M, make_sample

MAX_FRAME_AGE_NS = 1_000_000_000
MAX_VISUAL_RESULT_AGE_NS = 1_500_000_000


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def exp_rotation(v):
    v = np.asarray(v, dtype=float)
    angle = np.linalg.norm(v)
    a = skew(v)
    return np.eye(3) + (1.-angle*angle/6. if angle < 1e-7 else np.sin(angle)/angle)*a + (
        .5-angle*angle/24. if angle < 1e-7 else (1.-np.cos(angle))/(angle*angle))*(a@a)


def log_rotation(r):
    angle = np.arccos(np.clip((np.trace(r)-1.)/2., -1., 1.))
    axis = np.array([r[2, 1]-r[1, 2], r[0, 2]-r[2, 0], r[1, 0]-r[0, 1]])
    if angle < 1e-7:
        return axis*.5
    if np.pi-angle < 1e-5:
        _, vectors = np.linalg.eigh((r+r.T)/2.)
        return vectors[:, -1]*angle
    return axis*(angle/(2.*np.sin(angle)))


def quaternion(r):
    # Symmetric quaternion matrix also handles rotations close to pi.
    k = np.array([[r[0, 0]-r[1, 1]-r[2, 2], r[0, 1]+r[1, 0], r[0, 2]+r[2, 0], r[2, 1]-r[1, 2]],
                  [r[0, 1]+r[1, 0], r[1, 1]-r[0, 0]-r[2, 2], r[1, 2]+r[2, 1], r[0, 2]-r[2, 0]],
                  [r[0, 2]+r[2, 0], r[1, 2]+r[2, 1], r[2, 2]-r[0, 0]-r[1, 1], r[1, 0]-r[0, 1]],
                  [r[2, 1]-r[1, 2], r[0, 2]-r[2, 0], r[1, 0]-r[0, 1], np.trace(r)]])
    _, vectors = np.linalg.eigh(k)
    q = vectors[:, -1]
    return (q if q[3] >= 0 else -q).tolist()


def pose_values(pose):
    p = _vector(pose['position'])
    q = np.asarray(pose['quaternion'], dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(q@q-1.) > .002:
        raise ValueError('visual quaternion must be normalized')
    return p, rotation_from_quaternion(q/np.linalg.norm(q))


def observation_covariance(pose):
    c = np.asarray(pose['covariance'], dtype=float).reshape(6, 6)
    if (not np.isfinite(c).all() or not np.allclose(c, c.T, atol=1e-8) or
            np.linalg.eigvalsh(c).min() < -1e-10 or np.max(np.diag(c)[:3]) > .04 or
            np.max(np.diag(c)[3:]) > .12):
        raise ValueError('invalid or excessive visual covariance')
    # Model floors, not robot-calibrated accuracy. Avoid zero-noise observations.
    return c + np.diag([.005**2]*3 + [np.deg2rad(1.)**2]*3)


def geometry_observation(reference, result):
    """Compose camera pose and left-SE(3) registration uncertainty in world axes."""
    ref_p, ref_r = pose_values(reference)
    transform = result['transform']
    translation = transform[:3, 3]
    measured = dict(position=(ref_p+ref_r@translation).tolist(),
                    quaternion=quaternion(ref_r@transform[:3, :3]))
    relative = np.zeros((6, 6))
    relative[:3, :3], relative[3:, 3:] = ref_r, ref_r
    relative[:3, 3:] = -ref_r@skew(translation)
    anchor = np.eye(6)
    anchor[:3, 3:] = -skew(ref_r@translation)
    covariance = (relative@result['covariance']@relative.T +
                  anchor@observation_covariance(reference)@anchor.T)
    return measured, (covariance+covariance.T)/2.


def geometry_projection(reference, result):
    """Compose a reduced depth factor without filling unmeasured pose axes.

    Returned projection/lift use world camera position/rotation errors. Reference
    uncertainty is projected with its full cross covariance before fusion's
    covariance-intersection treatment of the shared anchor.
    """
    from .rgbd_geometry import _bounded_transform
    ref_p, ref_r = pose_values(reference)
    transform = np.asarray(result['transform'], dtype=float)
    rank = result['rank']
    projection = np.asarray(result['projection'], dtype=float)
    lift = np.asarray(result['observable_lift'], dtype=float)
    covariance = np.asarray(result['projected_covariance'], dtype=float)
    if (not isinstance(rank, (int, np.integer)) or not 1 <= rank <= 6
            or not _bounded_transform(transform) or projection.shape != (rank, 6)
            or lift.shape != (6, rank) or covariance.shape != (rank, rank)
            or not np.isfinite(projection).all() or not np.isfinite(lift).all()
            or not np.isfinite(covariance).all()
            or not np.allclose(projection@lift, np.eye(rank), atol=1e-7)
            or not np.allclose(covariance, covariance.T, atol=1e-8)
            or np.linalg.eigvalsh(covariance).min() <= 0):
        raise ValueError('invalid projected depth measurement')
    translation = transform[:3, 3]
    measured = dict(position=(ref_p+ref_r@translation).tolist(),
                    quaternion=quaternion(ref_r@transform[:3, :3]))
    relative = np.zeros((6, 6))
    relative[:3, :3], relative[3:, 3:] = ref_r, ref_r
    relative[:3, 3:] = -ref_r@skew(translation)
    reference_basis = np.zeros((6, 6))
    reference_basis[:3, :3], reference_basis[3:, 3:] = ref_r.T, ref_r.T
    inverse = np.eye(6)
    inverse[:3, 3:] = skew(translation)
    inverse = inverse@reference_basis
    anchor_projection = projection@reference_basis
    covariance = covariance + anchor_projection@observation_covariance(reference)@anchor_projection.T
    return measured, (covariance+covariance.T)*.5, projection@inverse, relative@lift


class InertialFilter:
    """Right attitude-error ESKF; position/velocity refer to accel sensor origin."""
    def __init__(self, stamp, pose, velocity, force, gyro, lever):
        camera, self.R = pose_values(pose)
        self.lever = _vector(lever)
        self.p, self.v = camera+self.R@self.lever, _vector(velocity)
        force = _vector(force)
        self.g = -self.R@force/np.linalg.norm(force)*9.80665
        self.bg = _vector(gyro)
        self.ba = force + self.R.T@self.g
        self.stamp = stamp
        self.rejection = None
        self.P = np.diag([.005**2]*3 + [.03**2]*3 + [.02**2]*3 + [.01**2]*3 + [.1**2]*3)

    def propagate(self, dt, gyro, force):
        if not 0 < dt <= .1:
            raise ValueError('inertial interval outside 0..100 ms')
        w, a = _vector(gyro)-self.bg, _vector(force)-self.ba
        mid = self.R@exp_rotation(w*dt/2.)
        acceleration = mid@a+self.g
        self.p += self.v*dt + .5*acceleration*dt*dt
        self.v += acceleration*dt
        f = np.zeros((15, 15))
        f[:3, 3:6] = np.eye(3)
        f[3:6, 6:9], f[3:6, 12:15] = -mid@skew(a), -mid
        f[6:9, 6:9], f[6:9, 9:12] = -skew(w), -np.eye(3)
        transition = np.eye(15)+f*dt+.5*(f@f)*dt*dt
        noise = np.zeros((15, 12))
        noise[3:6, :3], noise[6:9, 3:6] = mid, -np.eye(3)
        noise[9:12, 6:9], noise[12:15, 9:12] = np.eye(3), np.eye(3)
        # Conservative engineering priors until device noise is measured.
        q = noise@np.diag([.15**2]*3 + [.01**2]*3 + [.001**2]*3 + [.02**2]*3)@noise.T
        # Midpoint discretization includes position/velocity noise correlation.
        half = np.eye(15)+f*dt/2.
        self.P = transition@self.P@transition.T + half@q@half.T*dt
        self.P = (self.P+self.P.T)/2.
        self.R = self.R@exp_rotation(w*dt)
        self.stamp += round(dt*1e9)

    def measurement_matrix(self):
        h = np.zeros((6, 15))
        h[:3, :3], h[:3, 6:9], h[3:, 6:9] = np.eye(3), self.R@skew(self.lever), np.eye(3)
        return h

    def correct(self, pose, covariance, *, correlated=False, projection=None, observable_lift=None):
        position, rotation = pose_values(pose)
        residual = np.r_[position-(self.p-self.R@self.lever), log_rotation(self.R.T@rotation)]
        h = self.measurement_matrix()
        if projection is not None:
            # Projection/lift act on WORLD camera position and attitude errors;
            # covariance already describes only these k observed coordinates.
            projection = np.asarray(projection, dtype=float)
            lift = np.asarray(observable_lift, dtype=float)
            covariance = np.asarray(covariance, dtype=float)
            rank = projection.shape[0] if projection.ndim == 2 else 0
            if (not 1 <= rank <= 6 or projection.shape != (rank, 6) or lift.shape != (6, rank)
                    or covariance.shape != (rank, rank) or not np.isfinite(projection).all()
                    or not np.isfinite(lift).all() or not np.isfinite(covariance).all()
                    or not np.allclose(projection@lift, np.eye(rank), atol=1e-7)
                    or not np.allclose(covariance, covariance.T, atol=1e-8)
                    or np.linalg.eigvalsh(covariance).min() <= 0):
                self.rejection = 'invalid_projected_measurement'
                return False
            world = np.eye(6); world[3:, 3:] = self.R
            local_projection = projection@world
            residual = local_projection@residual
            h = local_projection@h
            absolute = lift@residual
        else:
            rank = 6
            absolute = residual
            # Incoming rotation covariance is in world axes; residual uses local axes.
            basis = np.eye(6)
            basis[3:, 3:] = self.R.T
            covariance = basis@covariance@basis.T
        # Absolute gates are independent of possibly over-large model covariance.
        if np.linalg.norm(absolute[:3]) > .15 or np.linalg.norm(absolute[3:]) > np.deg2rad(15.):
            self.rejection = 'absolute_residual_gate'
            return False
        innovation = h@self.P@h.T+covariance
        if residual@np.linalg.solve(innovation, residual) > (10.83, 13.82, 16.27, 18.47, 20.52, 22.46)[rank-1]:
            self.rejection = 'innovation_gate'
            return False
        self.rejection = None
        self._update(h, residual, covariance, correlated=correlated)
        return True

    def zero_velocity(self):
        h = np.zeros((3, 15))
        h[:, 3:6] = np.eye(3)
        self._update(h, -self.v, np.eye(3)*.015**2)

    def _update(self, h, residual, covariance, *, correlated=False):
        # Covariance intersection for depth corrections sharing a reference:
        # do not repeatedly average down a common reference/calibration error.
        prior = self.P/.9 if correlated else self.P
        covariance = covariance/.1 if correlated else covariance
        innovation = h@prior@h.T+covariance
        # A matrix RHS dispatches threaded BLAS even for this 6x6 system on
        # Jetson, blocking IMU callbacks for tens of milliseconds. Solve the
        # independent columns with vector RHSs; retain the same equations and
        # leave other plugins' process-wide BLAS thread settings untouched.
        gain = np.array([np.linalg.solve(innovation, column) for column in (h@prior).T])
        delta = gain@residual
        self.p += delta[:3]
        self.v += delta[3:6]
        self.R = self.R@exp_rotation(delta[6:9])
        self.bg += delta[9:12]
        self.ba += delta[12:15]
        identity = np.eye(15)-gain@h
        self.P = identity@prior@identity.T+gain@covariance@gain.T
        reset = np.eye(15)
        reset[6:9, 6:9] -= .5*skew(delta[6:9])
        self.P = reset@self.P@reset.T
        self.P = (self.P+self.P.T)/2.

    def camera_pose(self):
        h = self.measurement_matrix()
        covariance = h@self.P@h.T
        basis = np.eye(6)
        basis[3:, 3:] = self.R
        covariance = basis@covariance@basis.T
        return dict(position=(self.p-self.R@self.lever).tolist(), quaternion=quaternion(self.R),
                    covariance=covariance.ravel().tolist())


INERTIAL_LIMIT_NS = 1_000_000_000
MAX_BRIDGE_NS = 2_000_000_000
FRAME_SILENCE_NS = 500_000_000
# Off-lock registration has its own bounded wait after visual resolution.
# CaptureObservations holds subsequent filter updates within this deadline.
GEOMETRY_DEADLINE_NS = 250_000_000


class ShortTermFusion:
    """Frame-time fusion. Caller serializes exact-stamp visual results before use.

    Per frame: hard time/frame/data errors end the segment; otherwise IMU
    prediction is followed by an attempted legal visual correction, and only
    then are predicted (no correction) or corrected uncertainties judged.
    """
    def __init__(self, segment, frame, clock, K, depth_geometry=False, *, inertial_association=False):
        from .rgbd_geometry import DepthRegistration
        self.segment, self.frame, self.clock = segment, frame, clock
        self.imu = ImuAttitude(frame, clock)
        self.samples = deque(maxlen=1024)
        self.imu_coverage = deque(maxlen=self.samples.maxlen)
        self.warmup = deque(maxlen=30)
        self.visual_fallback_stamp = None
        self.filter = None
        self.lever = self.translations = None
        self.geometry = DepthRegistration(K) if depth_geometry else None
        # Native association may use the same IMU even when a particular frame
        # has no hint. Keep its unknown correlation conservative for the segment.
        self.inertial_association = bool(inertial_association)
        self.reference = None
        self.state, self.reason = 'initializing', 'waiting for 0.5 s low-motion visual/IMU warmup'
        self.loss_kind = None
        self.last_frame = self.last_visual = self.last_confident = 0
        self.last_processed_ns = self.last_visual_wall = 0
        self.good = self.visual_good = 0
        self.aid = 'none'
        self.quality = {}
        self.frames = deque(maxlen=16)
        self.last_tracking = None
        self.recovery_policy = self.pause_context = None
        self.input_period_ns = self.processing_p95_ns = None
        self.result_period_ns = self.native_processing_p95_ns = None
        self.bridge_budget_ns = INERTIAL_LIMIT_NS
        # Optional off-lock registration (GeometryWorker); None runs inline.
        self.geometry_runner = self.geometry_job = None
        self.revision = 0
        self.zero_velocity_updates = 0

    def _start_pause(self):
        if self.pause_context is None and self.last_tracking is not None and self.recovery_policy is not None:
            import time
            started = time.monotonic_ns()
            bridge = self.recovery_policy.begin(self.input_period_ns, self.processing_p95_ns, started)
            self.bridge_budget_ns = min(MAX_BRIDGE_NS, bridge.budget_ns)
            # Native result cadence budgets measured recovery. Pure inertial
            # propagation keeps its original capture-rate bound.
            tails = [v for v in (self.processing_p95_ns, self.native_processing_p95_ns) if v is not None]
            self.pause_context = self.recovery_policy.begin(self._frame_period_ns(), max(tails, default=None), started)

    def _frame_period_ns(self):
        periods = [v for v in (self.input_period_ns, self.result_period_ns) if v is not None]
        return max(periods, default=None)

    def fault(self, reason, kind='fusion'):
        if self.state != 'lost':
            self.loss_kind = kind
        self.state, self.reason = 'lost', self.reason if self.state == 'lost' else str(reason)

    def observe_imu(self, packet, now_ns):
        if self.state == 'lost':
            return
        try:
            sample = self.imu.observe(packet, now_ns)
            translations = translation_extrinsics(packet)
            self.translations = translations
            self.lever = translations[3:].copy()
            if self.filter is not None:
                self.filter.lever = self.lever.copy()
            if sample is not None:
                stamp = sample['stamp_ns']
                source = None
                if self.imu.capture_source is not None:
                    source = (self.imu.capture_source, tuple(translations))
                    if self.imu_coverage and source == self.imu_coverage[-1][1]:
                        source = self.imu_coverage[-1][1]
                if self.imu_coverage and stamp == self.imu_coverage[-1][0] and source == self.imu_coverage[-1][1]:
                    return
                if self.imu_coverage and stamp <= self.imu_coverage[-1][0]:
                    self.imu_coverage.clear()
                    self.samples.clear()
                self.imu_coverage.append((stamp, source))
                self.samples.append((stamp, np.array(sample['angular_velocity']),
                                     np.array(sample['linear_acceleration'])))
        except (ValueError, TypeError, KeyError) as error:
            self.fault(error)
            raise ValueError(str(error)) from error

    def _frame_silence_ns(self):
        # The watchdog and capture-order path use the same measured cadence.
        return max(FRAME_SILENCE_NS, min(INERTIAL_LIMIT_NS, 2*(self._frame_period_ns() or 0)))

    def status(self, now_ns):
        """Wall-clock health for watchdog/info; never judges covariance or
        resets recovery evidence, which belong to capture-time frame processing."""
        if self.state != 'lost' and self.imu.status(now_ns) == 'lost':
            self.fault('IMU stale or invalid: '+str(self.imu.reason))
        if self.filter is not None and self.state != 'lost':
            if (self.state == 'tracking' and self.last_processed_ns and
                    now_ns-self.last_processed_ns > self._frame_silence_ns()):
                self.state, self.reason = 'coasting', 'camera frames delayed'
                self._start_pause()
            if self.last_visual_wall and now_ns-self.last_visual_wall > self.bridge_budget_ns:
                self.fault('visual correction absent beyond the inertial bridge')
        return self.state

    def _expire(self, stamp):
        """Capture-time bounds, checked before this frame can refresh them."""
        if self.state != 'lost' and self.imu.status(stamp) == 'lost':
            self.fault('IMU stale or invalid: '+str(self.imu.reason))
        if self.filter is not None and self.state != 'lost':
            if self.last_frame and stamp-self.last_frame > self._frame_silence_ns():
                # A normal 4 Hz capture (plus jitter) is not a failed visual
                # correction. Only actual frame silence resets evidence here;
                # missing/rejected corrections reset it in observe_frame.
                self.state, self.reason, self.good = 'coasting', 'visual correction delayed', 0
                self._start_pause()
            if stamp-self.last_visual > self.bridge_budget_ns:
                self.fault('visual correction absent beyond the inertial bridge')

    def position_sigma(self):
        covariance = np.array(self.filter.camera_pose()['covariance']).reshape(6, 6)
        return float(np.sqrt(max(0., np.linalg.eigvalsh(covariance[:3, :3]).max())))

    def _initialize(self, stamp, pose):
        self.warmup.append((stamp, deepcopy(pose)))
        # Use the shortest measured window spanning half a second and at least
        # three poses. A low frame rate must not force a stationary IMU to wait forever.
        while self.warmup and stamp-self.warmup[0][0] > 1_500_000_000:
            self.warmup.popleft()
        while len(self.warmup) > 3 and stamp-self.warmup[1][0] >= 500_000_000:
            self.warmup.popleft()
        if len(self.warmup) < 3 or stamp-self.warmup[0][0] < 500_000_000:
            return False
        first_stamp, first_pose = self.warmup[0]
        positions = np.array([p['position'] for _, p in self.warmup])
        first_rotation = pose_values(first_pose)[1]
        angular_motion = max(np.linalg.norm(log_rotation(first_rotation.T@pose_values(p)[1]))
                             for _, p in self.warmup)
        velocity = (positions[-1]-positions[0])/((stamp-first_stamp)/1e9)
        if (angular_motion > .005 or np.linalg.norm(velocity) > .05 or
                np.max(np.linalg.norm(positions-positions[-1], axis=1)) > .025):
            self.visual_fallback_stamp = stamp
            return False
        if (not self.samples or self.samples[0][0] > first_stamp or self.samples[-1][0] < stamp):
            return False
        entries = [s for s in self.samples if first_stamp <= s[0] <= stamp]
        if len(entries) < 20:
            # Both boundaries have arrived. This window cannot initialize the filter.
            self.visual_fallback_stamp = stamp
            return False
        if self.imu_coverage and not self._imu_covers(first_stamp, stamp):
            return False
        # Include the measured samples bracketing both interpolation boundaries.
        times = np.array([s[0]-first_stamp for s in self.samples], dtype=float)
        begin = np.searchsorted(times, 0, side='right')-1
        end = np.searchsorted(times, stamp-first_stamp)
        if np.max(np.diff(times[begin:end+1])) > MAX_IMU_GAP_NS:
            return False
        # Interpolate the independently sampled IMU at both visual boundaries.
        values = np.array([np.r_[s[1], s[2]] for s in self.samples])
        boundaries = [np.array([np.interp(t-first_stamp, times, values[:, j]) for j in range(6)])
                      for t in (first_stamp, stamp)]
        entries = [(first_stamp, boundaries[0][:3], boundaries[0][3:])] + entries + [
                   (stamp, boundaries[1][:3], boundaries[1][3:])]
        gyro, accel = np.array([s[1] for s in entries]), np.array([s[2] for s in entries])
        if (np.max(np.linalg.norm(gyro, axis=1)) > .1 or np.max(np.std(accel, axis=0)) > .15 or
                abs(np.linalg.norm(accel.mean(axis=0))-9.80665) > .3):
            return False
        self.filter = InertialFilter(stamp, pose, velocity, accel.mean(axis=0), gyro.mean(axis=0), self.lever)
        self.good = 1
        return True

    def _propagate(self, stamp, estimate=None):
        estimate = self.filter if estimate is None else estimate
        if stamp < estimate.stamp or not self.samples or stamp > self.samples[-1][0]:
            raise ValueError('frame lacks capture-time IMU coverage')
        if self.imu_coverage and not self._imu_covers(estimate.stamp, stamp, max_gap_ns=None):
            raise ValueError('IMU source or numeric motion model changed inside propagation')
        entries = list(self.samples)
        if entries[0][0] > estimate.stamp:
            raise ValueError('IMU history no longer covers estimator state')
        for i in range(len(entries)-1):
            left, right = entries[i], entries[i+1]
            start, end = max(estimate.stamp, left[0]), min(stamp, right[0])
            if end <= start:
                continue
            # Accepted capture gaps can exceed the integrator's 100 ms step.
            # Interpolate only between real samples; never extend coverage.
            while start < end:
                step_end = min(end, start+100_000_000)
                fraction = (start-left[0]+(step_end-start)/2.)/(right[0]-left[0])
                gyro = left[1]*(1-fraction)+right[1]*fraction
                accel = left[2]*(1-fraction)+right[2]*fraction
                estimate.propagate((step_end-start)/1e9, gyro, accel)
                start = step_end
            if estimate.stamp == stamp:
                break
        if estimate.stamp != stamp:
            raise ValueError('IMU history has a propagation gap')

    def _imu_covers(self, start, end, *, max_gap_ns=MAX_IMU_GAP_NS):
        """Check the capture bracket and source continuity for the whole interval."""
        if start > end:
            return False
        right = source = None
        for stamp, identity in reversed(self.imu_coverage):
            if stamp >= end:
                source = identity
            elif (right is None or source is None or identity != source or stamp >= right or
                  (max_gap_ns is not None and right-stamp > max_gap_ns)):
                return False
            if stamp <= start:
                return identity is not None
            right = stamp
        return False

    def _stationary_update(self, stamp, measured_pose, ref_stamp, reference_pose):
        dt = (stamp-ref_stamp)/1e9
        if not .1 <= dt <= 1.:
            return False
        for pose in (measured_pose, reference_pose):
            cov = np.asarray(pose['covariance'], dtype=float).reshape(6, 6)
            if np.max(np.diag(cov)[:3]) > .025**2 or np.max(np.diag(cov)[3:]) > .05**2:
                return False
        p, r = pose_values(measured_pose)
        ref_p, ref_r = pose_values(reference_pose)
        if np.linalg.norm(p-ref_p) > min(.008, .02*dt) or np.linalg.norm(log_rotation(ref_r.T@r)) > .01:
            return False
        samples = [s for s in self.samples if ref_stamp <= s[0] <= stamp]
        # The measured visual pair establishes stationarity; the IMU corroborates it.
        if len(samples) < 5 or not self._imu_covers(ref_stamp, stamp):
            return False
        gravity_force = -r.T@self.filter.g
        if any(np.linalg.norm(w-self.filter.bg) > .045 or
               np.linalg.norm(a-self.filter.ba-gravity_force) > .25 for _, w, a in samples):
            return False
        self.filter.zero_velocity()
        self.zero_velocity_updates += 1
        return True

    def observe_frame(self, stamp, color, depth, visual_pose, now_ns, reason=None, *, arrival_ns=None):
        from .rgbd_geometry import observation_quality
        # CaptureObservations waits for the previous geometry decision first.
        # Direct callers cannot apply an older result after advancing state.
        self.poll_geometry(now_ns, force=True)
        diagnostic = dict(capture_ns=stamp, arrival_ns=arrival_ns, processed_ns=now_ns,
                          latency_ms=round((now_ns-stamp)/1e6, 1), predicted_sigma_m=None,
                          corrected_sigma_m=None, visual='missing' if visual_pose is None else 'offered',
                          visual_reject_reason=reason if visual_pose is None else None)
        self._expire(stamp)
        if self.state == 'lost':
            return self._finish(stamp, diagnostic)
        try:
            age = now_ns-stamp
            max_age = MAX_VISUAL_RESULT_AGE_NS if visual_pose is not None else MAX_FRAME_AGE_NS
            if stamp <= self.last_frame or not -50_000_000 <= age <= max_age:
                self.fault('fusion frame time is stale, repeated or reset', 'clock')
                return self._finish(stamp, diagnostic)
            self.last_frame, self.last_processed_ns = stamp, now_ns
            self.quality = observation_quality(color, depth)
            covariance = None
            if visual_pose is not None:
                if visual_pose.get('frame_id') != self.segment+'_odom' or visual_pose.get('child_frame_id') != self.frame:
                    self.fault('visual frame or segment changed', 'frame')
                    return self._finish(stamp, diagnostic)
                pose_values(visual_pose)
                covariance = observation_covariance(visual_pose)*self.quality['weight']*visual_pose.get('quality_weight', 1.)
                if not self.quality['visual_usable']:
                    visual_pose = None
                    diagnostic.update(visual='rejected', visual_reject_reason='image_' + self.quality['reason'])
            if self.filter is None:
                if visual_pose is not None and self._initialize(stamp, visual_pose):
                    self.last_visual = self.last_confident = stamp
                    self.last_visual_wall = now_ns
                    self.reference = (stamp, depth.copy(), deepcopy(visual_pose))
                return self._finish(stamp, diagnostic)
            self._propagate(stamp)
            self.revision += 1
            diagnostic['predicted_sigma_m'] = self.position_sigma()
            # A legal visual correction is attempted before predicted
            # covariance can end the segment; gates and time bounds still apply.
            accepted = visual_pose is not None and self.filter.correct(
                visual_pose, covariance, correlated=self.inertial_association)
            if visual_pose is not None and not accepted:
                diagnostic.update(visual='rejected', visual_reject_reason=self.filter.rejection)
            self.aid = 'visual' if accepted else 'inertial'
            if accepted:
                self.revision += 1
                diagnostic['visual'] = 'accepted'
                if self.reference is not None:
                    ref_stamp, _, ref_pose = self.reference
                    diagnostic['zero_velocity'] = self._stationary_update(stamp, visual_pose, ref_stamp, ref_pose)
                self.last_visual, self.last_visual_wall = stamp, now_ns
                diagnostic['corrected_sigma_m'] = self.position_sigma()
                self._accept_correction(stamp, visual_pose, depth)
            else:
                self.state, self.reason = 'coasting', reason or 'visual observation rejected'
                if age <= MAX_FRAME_AGE_NS:
                    self._assist_with_geometry(stamp, depth, now_ns, diagnostic)
                diagnostic['corrected_sigma_m'] = self.position_sigma() if self.aid == 'depth_geometry' else None
                if self.geometry_job is None and self.aid != 'depth_geometry':
                    self.good = 0
                # A pending registration defers this judgement to its result or
                # deadline; subsequent captures remain queued until it resolves.
                if self.geometry_job is None and self.position_sigma() > SIGMA_LIMIT_M:
                    self.fault(f'predicted position uncertainty exceeds {SIGMA_LIMIT_M:.2f} m without an accepted visual correction')
        except (ValueError, TypeError, KeyError, np.linalg.LinAlgError) as error:
            self.fault(error)
        return self._finish(stamp, diagnostic)

    def _accept_correction(self, stamp, visual_pose=None, depth=None):
        corrected = self.position_sigma()
        if corrected > SIGMA_LIMIT_M:
            self.good = 0
            self.state, self.reason = 'recovering', f'corrected position uncertainty exceeds {SIGMA_LIMIT_M:.2f} m'
            if stamp-self.last_confident > INERTIAL_LIMIT_NS:
                self.fault(f'corrected position uncertainty exceeded {SIGMA_LIMIT_M:.2f} m for more than 1 s')
            return
        self.last_confident = stamp
        if self.good == 0 or visual_pose is None:
            self.visual_good = 0
        if visual_pose is not None:
            self.visual_good += 1
        self.good += 1
        if self.state == 'tracking' or self.good >= 3:
            if (self.state != 'tracking' and self.last_tracking is not None and
                    stamp-self.last_tracking[0] > (self.pause_context.budget_ns if self.pause_context else INERTIAL_LIMIT_NS) and
                    not self._verified_late_visual(stamp, visual_pose)):
                self.fault(f'fusion recovery exceeded the {(self.pause_context.budget_ns if self.pause_context else INERTIAL_LIMIT_NS)/1e9:g} s mapping pause; restart required')
            else:
                self.state, self.reason = 'tracking', None
                if visual_pose is not None:
                    self.reference = (stamp, depth.copy(), deepcopy(visual_pose))
                self.last_tracking = (stamp, self.filter.camera_pose(), deepcopy(visual_pose))
        else:
            self.state, self.reason = 'recovering', 'waiting for three consistent observations'

    def _verified_late_visual(self, stamp, visual_pose):
        # The mapping wait can expire before three measured results finish.
        # Only a near-stationary measured recovery may keep the old origin.
        if (self.pause_context is None or self.visual_good < 3 or
                stamp-self.last_tracking[0] > MAX_BRIDGE_NS or self.last_tracking[2] is None):
            return False
        for current, anchor in ((visual_pose, self.last_tracking[2]),
                                (self.filter.camera_pose(), self.last_tracking[1])):
            p, r = pose_values(current)
            anchor_p, anchor_r = pose_values(anchor)
            if np.linalg.norm(p-anchor_p) > .05 or np.linalg.norm(log_rotation(anchor_r.T@r)) > np.deg2rad(5.):
                return False
        return True

    def _assist_with_geometry(self, stamp, depth, now_ns, diagnostic):
        if not (self.geometry and self.reference and stamp-self.reference[0] <= 1_000_000_000):
            return
        ref_stamp, ref_depth, ref_pose = self.reference
        ref_p, ref_r = pose_values(ref_pose)
        p, r = pose_values(self.filter.camera_pose())
        initial = np.eye(4)
        if self._imu_covers(ref_stamp, stamp):
            initial[:3, :3], initial[:3, 3] = ref_r.T@r, ref_r.T@(p-ref_p)
        # Only a verified observation of this exact capture may authorize mapping.
        # Keep the visual reference and its original one-second lifetime.
        self.quality['geometry'] = dict(attempted=True, accepted=False, reason='registration_rejected')
        if self.geometry_runner is not None:
            token = (self.segment, stamp, self.revision)
            if self.geometry_runner.submit(token, depth, ref_depth, initial):
                self.quality['geometry']['reason'] = diagnostic['geometry'] = 'pending'
                self.geometry_job = dict(token=token, revision=self.revision, ref_pose=ref_pose,
                                         deadline=now_ns+GEOMETRY_DEADLINE_NS, diagnostic=diagnostic)
            return
        try:
            result = self.geometry.register(depth, ref_depth, initial)
        except (ValueError, TypeError, ArithmeticError, np.linalg.LinAlgError) as error:
            # An assistance failure skips assistance; it is not a fusion fault.
            self.quality['geometry']['reason'] = 'registration_error: ' + str(error)[:80]
            return
        self._apply_geometry(ref_pose, result)

    def poll_geometry(self, now_ns, *, force=False):
        """Claim an off-lock registration for the exact filter state it used.

        `force` (next frame) resolves the job without judging uncertainty; the
        frame does that after its own visual attempt. A watchdog poll applies a
        ready result or expires the job, then applies the deferred limit.
        """
        job = self.geometry_job
        if job is None or self.geometry_runner is None:
            return
        status, payload = self.geometry_runner.take(job['token'])
        if status == 'pending' and not force and now_ns <= job['deadline']:
            return
        self.geometry_job = None
        geometry = self.quality.setdefault('geometry', dict(attempted=True, accepted=False))
        self.status(now_ns)
        if now_ns > job['deadline']:
            outcome = 'timeout_discarded'
        elif status != 'done':
            outcome = 'stale_result_discarded' if force else 'timeout_discarded'
        elif self.state == 'lost' or self.filter is None or self.revision != job['revision']:
            outcome = 'stale_result_discarded'
        elif payload[1] is not None:
            outcome = geometry['reason'] = 'registration_error: ' + payload[1][:80]
        elif payload[0] is None:
            outcome = geometry['reason'] = 'registration_rejected'
        else:
            self._apply_geometry(job['ref_pose'], payload[0])
            outcome = 'accepted' if geometry.get('accepted') else 'fusion_gate_rejected'
            if geometry.get('accepted'):
                self.revision += 1
                job['diagnostic']['corrected_sigma_m'] = self.position_sigma()
        if outcome.endswith('discarded'):
            geometry['reason'] = outcome
        if outcome != 'accepted':
            self.good = 0
        job['diagnostic'].update(geometry=outcome, state=self.state, aid=self.aid)
        if (not force and self.state == 'coasting' and self.filter is not None and
                self.position_sigma() > SIGMA_LIMIT_M):
            self.fault(f'predicted position uncertainty exceeds {SIGMA_LIMIT_M:.2f} m without an accepted visual correction')

    def _apply_geometry(self, ref_pose, result):
        if result is None:
            return
        try:
            rank = result.get('rank', 6)  # Existing full-rank observations remain valid.
            if not isinstance(rank, (int, np.integer)) or not 1 <= rank <= 6:
                raise ValueError('invalid depth measurement rank')
            self.quality['geometry'].update(
                inliers=result['inliers'], rmse=result['rmse'],
                overlap=result['overlap'], condition=result['condition'],
                rank=rank, full_pose_supported=False)
            projected = (geometry_projection(ref_pose, result)
                         if rank < 6 or 'projection' in result else None)
            if rank < 6:
                measured, covariance, projection, lift = projected
                accepted = self.filter.correct(measured, covariance, correlated=True,
                                               projection=projection, observable_lift=lift)
            else:
                from .rgbd_geometry import _bounded_transform
                relative_covariance = np.asarray(result['covariance'], dtype=float)
                if (not _bounded_transform(np.asarray(result['transform'], dtype=float))
                        or relative_covariance.shape != (6, 6) or not np.isfinite(relative_covariance).all()
                        or not np.allclose(relative_covariance, relative_covariance.T, atol=1e-8)
                        or np.linalg.eigvalsh(relative_covariance).min() <= 0):
                    raise ValueError('invalid full depth measurement')
                if projected is not None:
                    lift = np.asarray(result['observable_lift'], dtype=float)
                    reconstructed = lift@np.asarray(result['projected_covariance'], dtype=float)@lift.T
                    if not np.allclose(reconstructed, relative_covariance, atol=1e-10, rtol=1e-6):
                        raise ValueError('inconsistent full depth covariance')
                self.quality['geometry']['full_pose_supported'] = True
                measured, covariance = geometry_observation(ref_pose, result)
                accepted = self.filter.correct(measured, covariance, correlated=True)
        except (ValueError, TypeError, KeyError, ArithmeticError, np.linalg.LinAlgError) as error:
            self.quality['geometry']['reason'] = 'registration_error: ' + str(error)[:80]
            return
        if accepted:
            self.aid = 'depth_geometry'
            self.quality['geometry'].update(accepted=True, reason=None)
            if rank == 6:
                if self.reference is not None:
                    stationary_pose = dict(measured, covariance=covariance.ravel().tolist())
                    self.quality['geometry']['zero_velocity'] = self._stationary_update(
                        self.filter.stamp, stationary_pose, self.reference[0], ref_pose)
                self._accept_correction(self.filter.stamp)
            else:
                # A useful partial correction cannot certify a complete pose,
                # count toward recovery, or renew the original visual anchor.
                self.good = 0
                self.state, self.reason = 'coasting', f'depth geometry constrains {rank} of 6 pose directions'
        else:
            self.quality['geometry']['reason'] = 'fusion_gate_rejected'

    def _finish(self, stamp, diagnostic):
        if self.state in ('coasting', 'recovering') and self.last_tracking is not None and self.recovery_policy is not None:
            self._start_pause()
        elif self.state == 'tracking':
            self.pause_context = None
            self.bridge_budget_ns = INERTIAL_LIMIT_NS
        diagnostic.update(state=self.state, aid=self.aid)
        self.frames.append(diagnostic)
        return self.output(stamp, allow_prediction=diagnostic['processed_ns']-stamp <= MAX_FRAME_AGE_NS)

    def tracking_offset(self):
        """Fused-vs-visual offset at the last tracking capture, or None."""
        if self.last_tracking is None or self.last_tracking[2] is None:
            return None
        _, fused, visual = self.last_tracking
        (fp, fr), (vp, vr) = pose_values(fused), pose_values(visual)
        return float(np.linalg.norm(fp-vp)), float(np.linalg.norm(log_rotation(fr.T@vr)))

    def watchdog_output(self, now_ns):
        self.status(now_ns)
        stamp = self.last_frame or now_ns
        if self.state != 'coasting' or self.filter is None or not self.last_tracking or not self.samples:
            return self.output(stamp)
        latest = self.samples[-1][0]
        if not (self.filter.stamp <= latest <= self.last_visual+self.bridge_budget_ns and
                0 <= now_ns-latest <= MAX_IMU_GAP_NS):
            return self.output(stamp, allow_prediction=False)
        if latest == self.filter.stamp:
            return self.output(stamp)
        if not self._imu_covers(self.last_tracking[0], latest):
            return self.output(stamp, allow_prediction=False)
        estimate = deepcopy(self.filter)
        try:
            self._propagate(latest, estimate)
        except ValueError:
            return self.output(stamp, allow_prediction=False)
        return self.output(latest, estimate=estimate)

    def output(self, stamp, *, estimate=None, allow_prediction=True):
        estimate = self.filter if estimate is None else estimate
        pose = estimate.camera_pose() if estimate is not None and self.state != 'lost' else None
        if pose is not None:
            pose.update(frame_id=self.segment+'_odom', child_frame_id=self.frame)
        sample = make_sample(self.state, stamp, self.segment, self.clock,
                             pose if self.state == 'tracking' else None, reason=self.reason)
        sample['vendor']['source'] = 'rgbd_eskf'
        sample['vendor']['fusion'] = self.info(stamp)
        fusion = sample['vendor']['fusion']
        fusion['prediction'] = None
        if (pose is not None and allow_prediction and self.state in ('coasting', 'recovering') and
                self.last_tracking is not None and self._imu_covers(self.last_tracking[0], estimate.stamp)):
            covariance = np.asarray(pose['covariance']).reshape(6, 6)[:3, :3]
            fusion['prediction_stamp_ns'] = estimate.stamp
            if np.linalg.eigvalsh(covariance).max() <= SIGMA_LIMIT_M**2:
                fusion['prediction'] = pose
                fusion['prediction_anchor_ns'] = self.last_tracking[0]
        return sample

    def info(self, now_ns, *, history=False):
        value = dict(enabled=True, status=self.status(now_ns), reason=self.reason,
                     aid=self.aid, quality=deepcopy(self.quality), last_visual_ns=self.last_visual,
                     last_measurement_ns=self.last_frame, depth_geometry=self.geometry is not None,
                     prediction_stamp_ns=self.filter.stamp if self.filter is not None else None,
                     visual_corrections_correlated=self.inertial_association,
                     position_sigma_m=self.position_sigma() if self.filter is not None else None,
                     sigma_limit_m=SIGMA_LIMIT_M,
                     uncertainty_model='engineering_prior_unvalidated',
                     translation_integrated=True, mapping_allowed=self.state == 'tracking',
                     zero_velocity_updates=self.zero_velocity_updates,
                     imu=self.imu.info(now_ns),
                     last_frame=deepcopy(self.frames[-1]) if self.frames else None)
        if history:
            value['frames'] = deepcopy(list(self.frames))
        return value
