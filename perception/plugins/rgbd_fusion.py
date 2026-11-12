"""Bounded capture-time inertial assistance; predictions never claim SLAM tracking."""
from collections import deque
from copy import deepcopy

import numpy as np

from .rgbd_imu import ImuAttitude, _vector, rotation_from_quaternion
from .rgbd_odometry_contract import make_sample


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

    def correct(self, pose, covariance, *, correlated=False):
        position, rotation = pose_values(pose)
        residual = np.r_[position-(self.p-self.R@self.lever), log_rotation(self.R.T@rotation)]
        # Absolute gates are independent of possibly over-large model covariance.
        if np.linalg.norm(residual[:3]) > .15 or np.linalg.norm(residual[3:]) > np.deg2rad(15.):
            self.rejection = 'absolute_residual_gate'
            return False
        h = self.measurement_matrix()
        # Incoming rotation covariance is in world axes; residual uses local axes.
        basis = np.eye(6)
        basis[3:, 3:] = self.R.T
        covariance = basis@covariance@basis.T
        innovation = h@self.P@h.T+covariance
        if residual@np.linalg.solve(innovation, residual) > 22.46:
            self.rejection = 'innovation_gate'
            return False
        self.rejection = None
        # Covariance intersection for depth corrections sharing a reference:
        # do not repeatedly average down a common reference/calibration error.
        prior = self.P/.9 if correlated else self.P
        covariance = covariance/.1 if correlated else covariance
        innovation = h@prior@h.T+covariance
        gain = np.linalg.solve(innovation, h@prior).T
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
        return True

    def camera_pose(self):
        h = self.measurement_matrix()
        covariance = h@self.P@h.T
        basis = np.eye(6)
        basis[3:, 3:] = self.R
        covariance = basis@covariance@basis.T
        return dict(position=(self.p-self.R@self.lever).tolist(), quaternion=quaternion(self.R),
                    covariance=covariance.ravel().tolist())


SIGMA_LIMIT_M = .10
INERTIAL_LIMIT_NS = 1_000_000_000
# Wall-clock polls lag capture time by native latency. Frames resolve within the
# 250 ms join deadline, so 500 ms without any processed frame (>= 4 Hz) means
# frames stopped, not that the pipeline is merely latent.
FRAME_SILENCE_NS = 500_000_000
# Off-lock registration must finish within the visual join deadline; later
# results describe a superseded filter state and are discarded.
GEOMETRY_DEADLINE_NS = 250_000_000


class ShortTermFusion:
    """Frame-time fusion. Caller serializes exact-stamp visual results before use.

    Per frame: hard time/frame/data errors end the segment; otherwise IMU
    prediction is followed by an attempted legal visual correction, and only
    then are predicted (no correction) or corrected uncertainties judged.
    """
    def __init__(self, segment, frame, clock, K, depth_geometry=False):
        from .rgbd_geometry import DepthRegistration
        self.segment, self.frame, self.clock = segment, frame, clock
        self.imu = ImuAttitude(frame, clock)
        self.samples = deque(maxlen=1024)
        self.warmup = deque(maxlen=30)
        self.filter = None
        self.lever = self.translations = self.source_provenance = None
        self.geometry = DepthRegistration(K) if depth_geometry else None
        self.reference = None
        self.state, self.reason = 'initializing', 'waiting for 0.5 s low-motion visual/IMU warmup'
        self.last_frame = self.last_visual = self.last_confident = 0
        self.last_processed_ns = self.last_visual_wall = 0
        self.good = 0
        self.aid = 'none'
        self.quality = {}
        self.frames = deque(maxlen=16)
        self.last_tracking = None
        # Optional off-lock registration (GeometryWorker); None runs inline.
        self.geometry_runner = self.geometry_job = None
        self.revision = 0

    def fault(self, reason):
        self.state, self.reason = 'lost', self.reason if self.state == 'lost' else str(reason)

    def observe_imu(self, packet, now_ns):
        if self.state == 'lost':
            return
        try:
            if packet.get('translation_source') not in ('tf', 'sdk_factory_extrinsics'):
                raise ValueError('calibrated IMU translation is required for compensation')
            translations = np.r_[_vector(packet.get('camera_from_gyro_translation')),
                                  _vector(packet.get('camera_from_accel_translation'))]
            if self.translations is not None and not np.allclose(translations, self.translations, atol=1e-7, rtol=0):
                raise ValueError('IMU translation changed; restart segment')
            provenance = (packet['translation_source'], deepcopy(packet.get('clock_mapping')))
            if self.source_provenance is not None and provenance != self.source_provenance:
                raise ValueError('IMU calibration source or clock mapping changed; restart segment')
            sample = self.imu.observe(packet, now_ns)
            self.source_provenance = provenance
            self.translations = translations
            self.lever = translations[3:].copy()
            if sample is not None:
                self.samples.append((sample['stamp_ns'], np.array(sample['angular_velocity']),
                                     np.array(sample['linear_acceleration'])))
        except (ValueError, TypeError, KeyError) as error:
            self.fault(error)
            raise ValueError(str(error)) from error

    def status(self, now_ns):
        """Wall-clock health for watchdog/info; never judges covariance or
        resets recovery evidence, which belong to capture-time frame processing."""
        if self.state != 'lost' and self.imu.status(now_ns) == 'lost':
            self.fault('IMU stale or invalid: '+str(self.imu.reason))
        if self.filter is not None and self.state != 'lost':
            if self.last_visual_wall and now_ns-self.last_visual_wall > INERTIAL_LIMIT_NS:
                self.fault('visual correction absent for more than 1 s')
            elif (self.state == 'tracking' and self.last_processed_ns and
                    now_ns-self.last_processed_ns > FRAME_SILENCE_NS):
                self.state, self.reason = 'coasting', 'camera frames delayed'
        return self.state

    def _expire(self, stamp):
        """Capture-time bounds, checked before this frame can refresh them."""
        if self.state != 'lost' and self.imu.status(stamp) == 'lost':
            self.fault('IMU stale or invalid: '+str(self.imu.reason))
        if self.filter is not None and self.state != 'lost':
            if stamp-self.last_visual > INERTIAL_LIMIT_NS:
                self.fault('visual correction absent for more than 1 s')
            elif stamp-self.last_visual > 250_000_000:
                self.state, self.reason, self.good = 'coasting', 'visual correction delayed', 0

    def position_sigma(self):
        covariance = np.array(self.filter.camera_pose()['covariance']).reshape(6, 6)
        return float(np.sqrt(max(0., np.linalg.eigvalsh(covariance[:3, :3]).max())))

    def _initialize(self, stamp, pose):
        self.warmup.append((stamp, deepcopy(pose)))
        while self.warmup and stamp-self.warmup[0][0] > 700_000_000:
            self.warmup.popleft()
        if len(self.warmup) < 3 or stamp-self.warmup[0][0] < 500_000_000:
            return False
        first_stamp, first_pose = self.warmup[0]
        if not self.samples or self.samples[0][0] > first_stamp or self.samples[-1][0] < stamp:
            return False
        entries = [s for s in self.samples if first_stamp <= s[0] <= stamp]
        if len(entries) < 20:
            return False
        # Interpolate the independently sampled IMU at both visual boundaries.
        times = np.array([s[0]-first_stamp for s in self.samples], dtype=float)
        values = np.array([np.r_[s[1], s[2]] for s in self.samples])
        boundaries = [np.array([np.interp(t-first_stamp, times, values[:, j]) for j in range(6)])
                      for t in (first_stamp, stamp)]
        entries = [(first_stamp, boundaries[0][:3], boundaries[0][3:])] + entries + [
                   (stamp, boundaries[1][:3], boundaries[1][3:])]
        positions = np.array([p['position'] for _, p in self.warmup])
        gyro, accel = np.array([s[1] for s in entries]), np.array([s[2] for s in entries])
        first_rotation = pose_values(first_pose)[1]
        angular_motion = max(np.linalg.norm(log_rotation(first_rotation.T@pose_values(p)[1]))
                             for _, p in self.warmup)
        velocity = (positions[-1]-positions[0])/((stamp-first_stamp)/1e9)
        if (angular_motion > .005 or np.linalg.norm(velocity) > .05 or np.max(np.linalg.norm(positions-positions[-1], axis=1)) > .025 or
                np.max(np.linalg.norm(gyro, axis=1)) > .1 or np.max(np.std(accel, axis=0)) > .15 or
                abs(np.linalg.norm(accel.mean(axis=0))-9.80665) > .3):
            return False
        self.filter = InertialFilter(stamp, pose, velocity, accel.mean(axis=0), gyro.mean(axis=0), self.lever)
        self.good = 1
        return True

    def _propagate(self, stamp):
        if stamp < self.filter.stamp or not self.samples or stamp > self.samples[-1][0]:
            raise ValueError('frame lacks capture-time IMU coverage')
        entries = list(self.samples)
        if entries[0][0] > self.filter.stamp:
            raise ValueError('IMU history no longer covers estimator state')
        for i in range(len(entries)-1):
            left, right = entries[i], entries[i+1]
            start, end = max(self.filter.stamp, left[0]), min(stamp, right[0])
            if end <= start:
                continue
            fraction = ((start+end)/2.-left[0])/(right[0]-left[0])
            gyro = left[1]*(1-fraction)+right[1]*fraction
            accel = left[2]*(1-fraction)+right[2]*fraction
            self.filter.propagate((end-start)/1e9, gyro, accel)
            if self.filter.stamp == stamp:
                break
        if self.filter.stamp != stamp:
            raise ValueError('IMU history has a propagation gap')

    def observe_frame(self, stamp, color, depth, visual_pose, now_ns, reason=None, *, arrival_ns=None):
        from .rgbd_geometry import observation_quality
        # A registration for an earlier state applies only before this frame
        # advances the filter; otherwise it is discarded, never replayed.
        self.poll_geometry(now_ns, force=True)
        diagnostic = dict(capture_ns=stamp, arrival_ns=arrival_ns, processed_ns=now_ns,
                          latency_ms=round((now_ns-stamp)/1e6, 1), predicted_sigma_m=None,
                          corrected_sigma_m=None, visual='missing' if visual_pose is None else 'offered',
                          visual_reject_reason=reason if visual_pose is None else None)
        self._expire(stamp)
        if self.state == 'lost':
            return self._finish(stamp, diagnostic)
        try:
            if stamp <= self.last_frame or not -50_000_000 <= now_ns-stamp <= 1_000_000_000:
                raise ValueError('fusion frame time is stale, repeated or reset')
            self.last_frame, self.last_processed_ns = stamp, now_ns
            self.quality = observation_quality(color, depth)
            covariance = None
            if visual_pose is not None:
                if visual_pose.get('frame_id') != self.segment+'_odom' or visual_pose.get('child_frame_id') != self.frame:
                    raise ValueError('visual frame or segment changed')
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
            accepted = visual_pose is not None and self.filter.correct(visual_pose, covariance)
            if visual_pose is not None and not accepted:
                diagnostic.update(visual='rejected', visual_reject_reason=self.filter.rejection)
            self.aid = 'visual' if accepted else 'inertial'
            if accepted:
                self.revision += 1
                diagnostic['visual'] = 'accepted'
                self.last_visual, self.last_visual_wall = stamp, now_ns
                corrected = diagnostic['corrected_sigma_m'] = self.position_sigma()
                if corrected > SIGMA_LIMIT_M:
                    self.good = 0
                    self.state, self.reason = 'recovering', 'corrected position uncertainty exceeds 0.10 m'
                    if stamp-self.last_confident > INERTIAL_LIMIT_NS:
                        self.fault('corrected position uncertainty exceeded 0.10 m for more than 1 s')
                else:
                    self.last_confident = stamp
                    self.good += 1
                    if self.state == 'tracking' or self.good >= 3:
                        self.state, self.reason = 'tracking', None
                        self.reference = (stamp, depth.copy(), deepcopy(visual_pose))
                        self.last_tracking = (stamp, self.filter.camera_pose(), deepcopy(visual_pose))
                    else:
                        self.state, self.reason = 'recovering', 'waiting for three consistent visual observations'
            else:
                self.good = 0
                self.state, self.reason = 'coasting', reason or 'visual observation rejected'
                self._assist_with_geometry(stamp, depth, now_ns, diagnostic)
                diagnostic['corrected_sigma_m'] = self.position_sigma() if self.aid == 'depth_geometry' else None
                # A pending registration defers this judgement to its result or
                # deadline; the next frame re-judges after its own prediction.
                if self.geometry_job is None and self.position_sigma() > SIGMA_LIMIT_M:
                    self.fault('predicted position uncertainty exceeds 0.10 m without an accepted visual correction')
        except (ValueError, TypeError, KeyError, np.linalg.LinAlgError) as error:
            self.fault(error)
        return self._finish(stamp, diagnostic)

    def _assist_with_geometry(self, stamp, depth, now_ns, diagnostic):
        if not (self.geometry and self.reference and stamp-self.reference[0] <= 1_000_000_000):
            return
        ref_stamp, ref_depth, ref_pose = self.reference
        ref_p, ref_r = pose_values(ref_pose)
        p, r = pose_values(self.filter.camera_pose())
        initial = np.eye(4)
        initial[:3, :3], initial[:3, 3] = ref_r.T@r, ref_r.T@(p-ref_p)
        # Geometry only corrects a bounded prediction; it never
        # promotes coasting to tracking or authorizes map integration.
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
        if status != 'done':
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
        job['diagnostic']['geometry'] = outcome
        if (not force and self.state == 'coasting' and self.filter is not None and
                self.position_sigma() > SIGMA_LIMIT_M):
            self.fault('predicted position uncertainty exceeds 0.10 m without an accepted visual correction')

    def _apply_geometry(self, ref_pose, result):
        if result is None:
            return
        self.quality['geometry'].update(
            inliers=result['inliers'], rmse=result['rmse'],
            overlap=result['overlap'], condition=result['condition'])
        measured, geom_cov = geometry_observation(ref_pose, result)
        if self.filter.correct(measured, geom_cov, correlated=True):
            self.aid = 'depth_geometry'
            self.quality['geometry'].update(accepted=True, reason=None)
        else:
            self.quality['geometry']['reason'] = 'fusion_gate_rejected'

    def _finish(self, stamp, diagnostic):
        diagnostic.update(state=self.state, aid=self.aid)
        self.frames.append(diagnostic)
        return self.output(stamp)

    def tracking_offset(self):
        """Fused-vs-visual offset at the last tracking capture, or None."""
        if self.last_tracking is None:
            return None
        _, fused, visual = self.last_tracking
        (fp, fr), (vp, vr) = pose_values(fused), pose_values(visual)
        return float(np.linalg.norm(fp-vp)), float(np.linalg.norm(log_rotation(fr.T@vr)))

    def output(self, stamp):
        pose = self.filter.camera_pose() if self.filter is not None and self.state != 'lost' else None
        if pose is not None:
            pose.update(frame_id=self.segment+'_odom', child_frame_id=self.frame)
        sample = make_sample(self.state, stamp, self.segment, self.clock,
                             pose if self.state == 'tracking' else None, reason=self.reason)
        sample['vendor']['source'] = 'rgbd_eskf'
        sample['vendor']['fusion'] = self.info(stamp)
        sample['vendor']['fusion']['prediction'] = pose if self.state in ('coasting', 'recovering') else None
        return sample

    def info(self, now_ns, *, history=False):
        value = dict(enabled=True, status=self.status(now_ns), reason=self.reason,
                     aid=self.aid, quality=deepcopy(self.quality), last_visual_ns=self.last_visual,
                     last_measurement_ns=self.last_frame, depth_geometry=self.geometry is not None,
                     prediction_stamp_ns=self.filter.stamp if self.filter is not None else None,
                     position_sigma_m=self.position_sigma() if self.filter is not None else None,
                     sigma_limit_m=SIGMA_LIMIT_M,
                     uncertainty_model='engineering_prior_unvalidated',
                     translation_integrated=True, mapping_allowed=self.state == 'tracking',
                     imu=self.imu.info(now_ns),
                     last_frame=deepcopy(self.frames[-1]) if self.frames else None)
        if history:
            value['frames'] = deepcopy(list(self.frames))
        return value
