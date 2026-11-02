"""Small, bounded yaw-response checks with no ROS or chassis dependency.

The caller owns command delivery, odometry freshness and chassis arbitration.
This module adds independent time/sample/motion bounds and always returns a
zero twist on failure, cancellation or completion. It derives conservative
gains from two short pulses; it does not identify a complete plant or tune PID.

The two pulses reverse directly.  An intermediate explicit zero would make the
Tianyi Slamware adapter cancel its velocity action; reacquiring that action less
than a second later is not a continuous servo update and can leave the second
pulse unexecuted.  The downstream ControlSink still ramps the sign change
through zero, while the requested command remains non-zero and therefore keeps
the same SDK action.  A real zero is sent only before motion and at the end.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import statistics
import tempfile
import time


log = logging.getLogger(__name__)
SCHEMA = "navi.yaw-calibration/1"
MAX_COMMAND_GAP_S = 0.3
# REST odometry and command dispatch are independent clocks. A slower fresh
# sensor sample does not mean that the low-speed command heartbeat stopped.
MAX_SAMPLE_AGE_S = 0.65
# The measured-rate and angle guards are derived per session from the command
# selected for that chassis.  At the reference 0.5 rad/s they remain exactly
# the original 0.75 / 1.5 / 2.8 limits; an R1 needs 1.0 rad/s merely to leave
# its standstill yaw deadband, so fixed 0.5-specific guards would reject a
# healthy response before it could be measured.
_REFERENCE_SPEED = 0.5
_MAX_CALIBRATION_SPEED = 1.0
_MAX_RESPONSE_RATIO = 1.5
_ANGLE_MARGIN = 1.2
_BASE_MAX_SINGLE_ANGLE_RAD = 1.5
_BASE_MAX_TOTAL_ANGLE_RAD = 2.8
MAX_SESSION_S = 12.0
SETTLED_WZ = 0.03
_EPSILON = 1e-9
_SESSION_DURATION_S = 10.0
_SETTLE_S = 1.0
_BASE_LEFT_S = 2.0
_BASE_RIGHT_S = 2.5
_MIN_LEFT_S = 1.75
_MIN_RIGHT_S = 2.0
_STABLE_WINDOW_S = 1.25
_REST_S = 1.5
_REVERSAL_TRANSIENT_S = 1.2
# Tianyi's SDK action plus the chassis acceleration limiter can take roughly
# 0.75 s before odometry first clears the 15% response threshold, even on the
# initial pulse.  Use the same bounded limit as the stored delay profile for
# both directions; the independent 300 ms command-heartbeat and 650 ms fresh-
# observation guards still reject an actual control or feedback outage.
_ONSET_LIMIT_S = (1.25, 1.25)
_PROFILE_BOUNDS = {"k_yaw": (0.2, 1.2), "yaw_damping": (0.0, 0.5),
                   "response_gain": (0.15, 2.5), "delay_s": (0.0, 1.25),
                   "settling_s": (0.0, 1.5), "left_gain": (0.15, 2.5),
                   "right_gain": (0.15, 2.5)}
_NOMINAL_YAW_P = 0.6


def _calibrated_p(response_gain, delay_s, settling_s):
    """Compensate plant gain, then conservatively discount slow response.

    The old formula started from 1.2 and multiplied delay by four, which drove
    the measured Tianyi profile to the 0.2 floor even though its left/right
    gains agreed.  The runtime nominal is now 0.6: correct that value for the
    measured plant gain, then apply a bounded, gentler delay/settling penalty.
    """
    effective_gain = max(0.5, min(2.5, response_gain))
    # Delay matters, but it must not overwhelm the measured gain: doing so made
    # a weak-but-well-behaved chassis less responsive than the uncalibrated
    # 0.6 default. Keep a modest margin for its phase lag instead.
    penalty = 1.0 + 0.2 * delay_s + 0.1 * settling_s
    return max(0.2, min(1.2, (_NOMINAL_YAW_P / effective_gain) / penalty))


def _number(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _identity(robot_key):
    if not isinstance(robot_key, str) or not robot_key.strip() or len(robot_key) > 1024:
        raise ValueError("robot_key must be a nonempty robot identity")
    return robot_key


def validate_profile(profile, robot_key):
    """Return a normalized profile, rejecting wrong-robot or unsafe values."""
    _identity(robot_key)
    if not isinstance(profile, dict) or profile.get("schema") != SCHEMA:
        raise ValueError("invalid yaw calibration schema")
    if profile.get("robot_key") != robot_key:
        raise ValueError("yaw calibration belongs to a different robot")
    result = {"schema": SCHEMA, "robot_key": robot_key}
    created = profile.get("created_at_ms")
    if isinstance(created, bool) or not isinstance(created, int) or created <= 0:
        raise ValueError("created_at_ms must be a positive integer")
    result["created_at_ms"] = created
    for name, (lower, upper) in _PROFILE_BOUNDS.items():
        value = _number(profile.get(name), name)
        if not lower <= value <= upper:
            raise ValueError(f"{name} is outside calibration bounds [{lower}, {upper}]")
        result[name] = value
    # k_yaw is derived from the measured fields. Recompute it when loading an
    # older profile so a robot does not remain pinned to the obsolete 0.2-floor
    # formula merely because the measurement file already exists.
    result["k_yaw"] = _calibrated_p(result["response_gain"], result["delay_s"],
                                    result["settling_s"])
    # v1 profiles created by earlier images may contain a non-zero D-like
    # angular-rate damping term.  Accept the file for its measured P, but never
    # reactivate that retired feedback path.
    result["yaw_damping"] = 0.0
    noisy_odom = profile.get("noisy_odom", False)
    if type(noisy_odom) is not bool:
        raise ValueError("noisy_odom must be a boolean")
    result["noisy_odom"] = noisy_odom
    low, high = sorted((result["left_gain"], result["right_gain"]))
    if high > 2.5 * low:
        raise ValueError("yaw response is too asymmetric")
    if not math.isclose(result["response_gain"],
                        (result["left_gain"] + result["right_gain"]) / 2,
                        rel_tol=1e-6, abs_tol=1e-9):
        raise ValueError("response_gain does not match the two measured directions")
    return result


def profile_path(directory, robot_key):
    """A robot identity never becomes an untrusted path component."""
    identity = _identity(robot_key)
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return Path(directory) / f"yaw-{digest}.json"


def load_profile(directory, robot_key):
    """Missing profiles return None; invalid files raise a reportable error."""
    path = profile_path(directory, robot_key)
    try:
        with path.open("r", encoding="utf-8") as stream:
            profile = json.load(stream)
    except FileNotFoundError:
        return None
    return validate_profile(profile, robot_key)


def save_profile(directory, robot_key, profile):
    """Validate, fsync and atomically replace one robot's local JSON profile.

    A caller must apply the profile only after this function succeeds. Rename
    is the atomic commit point. A subsequent directory-fsync failure is logged
    as a durability warning, since the replacement has already taken effect.
    No SDK, network service or global card configuration is touched here.
    """
    normalized = validate_profile(profile, robot_key)
    path = profile_path(directory, robot_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        descriptor, temporary = tempfile.mkstemp(prefix=".yaw-", suffix=".tmp", dir=path.parent)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(normalized, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError as error:
            log.warning("Yaw calibration committed at %s, but directory fsync failed: %s", path, error)
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    return path


@dataclass(frozen=True)
class CalibrationResult:
    values: list[float]
    status: str
    reason: str
    phase: str
    elapsed_s: float
    samples: int
    profile: dict | None = None
    diagnostics: dict | None = None


class CalibrationFitError(ValueError):
    """A fit refusal that carries the bounded evidence used to refuse it."""

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


class YawCalibration:
    """One wall-clock session: settle, +yaw, longer direct -yaw, rest (10 s).

    Longer bounded pulses yield several stable samples at 4 Hz; commands still
    have their own 300 ms dispatch guard. Missing samples never become new
    evidence, and the measured-angle limits can terminate a pulse early.

    ``sample_stamp`` is in monotonic seconds using the same clock as ``now``.
    When omitted, a new unique sample is assumed to have just been received.
    Keys must identify distinct odometry samples, not controller tick numbers.
    """

    duration_s = _SESSION_DURATION_S

    def __init__(self, *, now, robot_key, speed=0.5):
        self.robot_key = _identity(robot_key)
        self.speed = _number(speed, "calibration speed")
        if not 0.05 <= self.speed <= _MAX_CALIBRATION_SPEED:
            raise ValueError("calibration speed must be between 0.05 and 1.0 rad/s")

        # Keep the proven Tianyi timings unchanged at and below 0.5 rad/s.  A
        # faster chassis gets shorter pulses, but never so short that the last
        # 1.25 s window lacks several distinct R1 10 Hz odometry samples.  The
        # rest phase absorbs the saved time, so every session still ends at
        # ten seconds and spends longer, not shorter, proving that it stopped.
        ratio = min(1.0, _REFERENCE_SPEED / self.speed)
        left_s = max(_MIN_LEFT_S, _BASE_LEFT_S * ratio)
        right_s = max(_MIN_RIGHT_S, _BASE_RIGHT_S * ratio)
        left_begin = _SETTLE_S
        left_end = left_begin + left_s
        right_end = left_end + right_s
        self._pulses = ((left_begin, left_end, 1.0),
                        (left_end, right_end, -1.0))
        self._phases = (("settle", 0.0, left_begin, 0.0),
                        ("left", left_begin, left_end, 1.0),
                        ("right", left_end, right_end, -1.0),
                        ("rest_right", right_end, self.duration_s, 0.0))

        self.max_observed_wz = _MAX_RESPONSE_RATIO * self.speed
        planned_angles = [self.speed * (end - begin)
                          for begin, end, _direction in self._pulses]
        self.max_single_angle = max(_BASE_MAX_SINGLE_ANGLE_RAD,
                                    _ANGLE_MARGIN * max(planned_angles))
        self.max_total_angle = max(_BASE_MAX_TOTAL_ANGLE_RAD,
                                   _ANGLE_MARGIN * sum(planned_angles))
        self.started_at = self._last_step = _number(now, "now")
        self._last_sample_stamp = None
        self._last_wz = 0.0
        self._seen_keys = set()
        self._samples = []  # (monotonic offset, measured yaw rate)
        self._single_angle = self._total_angle = 0.0
        self._angle_group = None
        self._terminal = None

    def _result(self, now, status, reason, phase, wz=0.0, profile=None,
                diagnostics=None):
        return CalibrationResult([0.0, 0.0, 0.0, 0.0, 0.0, wz], status, reason,
                                 phase, max(0.0, now - self.started_at),
                                 len(self._samples), profile, diagnostics)

    def _fail(self, now, reason, diagnostics=None):
        self._terminal = self._result(now, "failed", reason, "failed",
                                      diagnostics=diagnostics)
        return self._terminal

    def cancel(self, reason="yaw calibration cancelled"):
        if self._terminal is None:
            self._terminal = self._result(self._last_step, "cancelled", reason, "cancelled")
        return self._terminal

    def step(self, now, measured_wz, sample_key, sample_stamp=None):
        if self._terminal is not None:
            return self._terminal
        try:
            now = _number(now, "now")
            stamp = now if sample_stamp is None else _number(sample_stamp, "sample_stamp")
            measured = _number(measured_wz, "measured_wz")
        except ValueError as error:
            return self._fail(self._last_step, str(error))
        if now < self._last_step:
            return self._fail(self._last_step, "calibration clock moved backwards")
        if now - self._last_step > MAX_COMMAND_GAP_S + _EPSILON:
            return self._fail(now, "calibration command gap exceeded 300 ms")
        self._last_step = now
        if stamp > now + 0.05 or now - stamp > MAX_SAMPLE_AGE_S + _EPSILON:
            return self._fail(now, "yaw observation is stale or future-dated")
        if abs(measured) > self.max_observed_wz:
            return self._fail(
                now,
                f"measured yaw rate exceeded {self.max_observed_wz:g} rad/s")
        if sample_key is None:
            return self._fail(now, "yaw observation has no distinct sample key")
        try:
            is_new = sample_key not in self._seen_keys
        except TypeError:
            return self._fail(now, "yaw observation sample key is not hashable")
        if is_new:
            if self._last_sample_stamp is not None and stamp <= self._last_sample_stamp:
                return self._fail(now, "yaw observation timestamp did not advance")
            offset = stamp - self.started_at
            if offset < 0.0:
                # A fresh cached odometry sample can predate this request by
                # a fraction of one camera/control period. Stay at zero until
                # a post-start sample arrives; it is not calibration evidence.
                return self._result(now, "running", "waiting for a post-start yaw observation", "settle")
            if self._last_sample_stamp is not None:
                interval = stamp - self._last_sample_stamp
                if interval > MAX_SAMPLE_AGE_S + _EPSILON:
                    return self._fail(now, "no fresh yaw observation for 650 ms")
                # Count the full measured arc, including residual rotation in
                # rest phases, rather than only the commanded pulse duration.
                angle = (abs(self._last_wz) + abs(measured)) * 0.5 * interval
                group = 1 if offset < self._pulses[1][0] else 2
                if group != self._angle_group:
                    # This interval straddles the command sign change. Count it
                    # in the total physical-motion budget, but not wholly as
                    # the new direction's arc: doing so double-attributes the
                    # previous pulse and can trip the independent per-side
                    # safety budget during a nominal direct reversal.
                    self._single_angle = 0.0
                    self._angle_group = group
                else:
                    self._single_angle += angle
                self._total_angle += angle
                if (self._single_angle > self.max_single_angle + _EPSILON
                        or self._total_angle > self.max_total_angle + _EPSILON):
                    return self._fail(now, "measured rotation exceeded calibration angle limit")
            self._seen_keys.add(sample_key)
            self._samples.append((offset, measured))
            self._last_sample_stamp, self._last_wz = stamp, measured
        elif self._last_sample_stamp is None or now - self._last_sample_stamp > MAX_SAMPLE_AGE_S + _EPSILON:
            return self._fail(now, "no fresh yaw observation for 650 ms")

        elapsed = now - self.started_at
        if elapsed > MAX_SESSION_S:
            return self._fail(now, "yaw calibration exceeded its twelve-second duration bound")
        if elapsed >= self.duration_s - _EPSILON:
            if not self._settled_recently(self._pulses[-1][1]):
                return self._fail(now, "chassis has not stopped at the end of calibration")
            try:
                profile, diagnostics = self._profile()
            except CalibrationFitError as error:
                return self._fail(now, str(error), error.diagnostics)
            except ValueError as error:
                return self._fail(now, str(error))
            self._terminal = self._result(now, "completed", "yaw response measured; save before applying",
                                          "completed", profile=profile,
                                          diagnostics=diagnostics)
            return self._terminal
        for phase, begin, end, direction in self._phases:
            if begin - _EPSILON <= elapsed < end - _EPSILON:
                if phase == "settle" and abs(measured) > SETTLED_WZ:
                    return self._fail(now, "chassis was already turning during initial settle")
                if phase == "left" and not self._samples_before(self._pulses[0][0]):
                    return self._fail(now, "insufficient initial stationary yaw observations")
                direction_guard_s = (_REVERSAL_TRANSIENT_S if phase == "right" else 0.15)
                if (direction and elapsed - begin >= direction_guard_s
                        and measured * direction < -max(0.03, self.speed * 0.2)):
                    return self._fail(now, "measured yaw direction disagrees with calibration command")
                return self._result(now, "running", "measuring yaw response", phase, direction * self.speed)
        return self._fail(now, "invalid calibration phase")

    def _samples_before(self, offset):
        return len([at for at, wz in self._samples
                    if at < offset and abs(wz) <= SETTLED_WZ]) >= 2

    def _settled_recently(self, rest_begin):
        recent = [(at, wz) for at, wz in self._samples if at >= rest_begin - _EPSILON][-2:]
        return len(recent) == 2 and all(abs(wz) <= SETTLED_WZ for at, wz in recent)

    def _profile(self):
        gains, delays = [], []
        filtered_outlier = False
        response_floor = max(0.01, self.speed * 0.15)
        diagnostics = {"commanded_wz_rad_s": self.speed,
                       "required_response_rad_s": response_floor,
                       "directions": {}}
        windows = []
        for name, (begin, end, direction) in zip(("left", "right"), self._pulses):
            stable = [direction * wz for at, wz in self._samples
                      if end - _STABLE_WINDOW_S - _EPSILON <= at < end - _EPSILON]

            def metrics(values):
                response = statistics.median(values) if values else None
                spread = (statistics.median(abs(value - response) for value in values)
                          if values else None)
                tolerance = (max(0.03, response * 0.25)
                             if response is not None else None)
                valid = bool(values) and response >= response_floor
                if valid:
                    valid = (spread <= max(0.02, response * 0.2)
                             and all(value >= response_floor
                                     and abs(value - response) <= tolerance
                                     for value in values))
                return response, spread, tolerance, valid

            raw_tail = stable[-3:]
            tail = raw_tail
            response, spread, tolerance, valid = metrics(tail)
            fit_method = "strict_tail3"
            excluded = None
            # A low-rate REST feed can occasionally contribute one positive
            # spike at the pulse boundary.  Only recover when the last four
            # readings are all genuine same-direction responses and dropping
            # exactly one leaves three readings that pass the unchanged strict
            # plateau test.  Zeros, reversals, alternating noise and two bad
            # points therefore remain failures.
            fit_window = stable[-4:] if len(stable) >= 4 else list(raw_tail)
            if (not valid and len(fit_window) == 4
                    and all(value >= response_floor for value in fit_window)):
                candidates = []
                for index in range(4):
                    candidate = fit_window[:index] + fit_window[index + 1:]
                    fit = metrics(candidate)
                    if fit[3]:
                        normalized_spread = fit[1] / max(fit[0], response_floor)
                        candidates.append((normalized_spread, index, candidate, fit))
                if candidates:
                    _, index, tail, fit = min(candidates, key=lambda item: (item[0], item[1]))
                    response, spread, tolerance, valid = fit
                    fit_method = "tail4_drop_one"
                    excluded = {"index": index,
                                "wz_rad_s": round(fit_window[index], 4)}
                    filtered_outlier = True
            diagnostics["directions"][name] = {
                "samples": len(stable),
                "window_s": [round(end - _STABLE_WINDOW_S, 3), round(end, 3)],
                "signed_wz_rad_s": [round(value, 4) for value in stable[-8:]],
                "tail_wz_rad_s": [round(value, 4) for value in raw_tail],
                "fit_window_wz_rad_s": [round(value, 4) for value in fit_window],
                "used_tail_wz_rad_s": [round(value, 4) for value in tail],
                "fit_method": fit_method,
                "excluded_outlier": excluded,
                "median_response_rad_s": (round(response, 4)
                                            if response is not None else None),
                "median_abs_deviation_rad_s": (round(spread, 4)
                                                 if spread is not None else None),
                "tail_tolerance_rad_s": (round(tolerance, 4)
                                           if tolerance is not None else None),
            }
            windows.append((name, begin, end, direction, stable, tail,
                            response, spread, tolerance, valid))
        diagnostics["outlier_filtered"] = filtered_outlier
        for pulse_index, (name, begin, end, direction, stable, tail,
                          response, spread, tail_tolerance, valid) in enumerate(windows):
            if len(stable) < 3:
                raise CalibrationFitError(
                    f"{name} has fewer than three distinct stable yaw samples", diagnostics)
            # At 4 Hz the beginning of this window can still contain the
            # acceleration/reversal transition.  Fit the last three readings,
            # and require all three to agree with that plateau.  This
            # ignores at most an early transition point without accepting a
            # tail that is still oscillating or has already dropped to zero.
            if response is None or response < response_floor:
                raise CalibrationFitError(
                    f"{name} median yaw response is below {response_floor:.3f} rad/s",
                    diagnostics)
            if not valid:
                raise CalibrationFitError(
                    f"{name} tail yaw response was not stable enough to calibrate",
                    diagnostics)
            gains.append(response / self.speed)
            threshold = max(0.01, self.speed * 0.15)
            onset = next((at - begin for at, wz in self._samples
                          if begin - _EPSILON <= at < end - _EPSILON
                          and direction * wz >= threshold), None)
            if onset is None or onset > _ONSET_LIMIT_S[pulse_index] + _EPSILON:
                raise CalibrationFitError(
                    f"{name} yaw response onset was missing or too delayed", diagnostics)
            delays.append(max(0.0, onset))
        # Only the final pulse is followed by a real zero.  Measuring a
        # "settling" time at the sign-change would require cancelling and
        # reacquiring Tianyi's SDK action, which is precisely what this
        # calibration must avoid.  Reversal onset is already captured by the
        # second direction's delay; settling is measured after the final zero.
        final_end = self._pulses[-1][1]
        threshold = max(0.01, self.speed * 0.15)
        rest = [(at - final_end, abs(wz)) for at, wz in self._samples
                if final_end - _EPSILON <= at <= final_end + _REST_S + _EPSILON]
        # Require two consecutive near-zero readings. A single zero amidst
        # residual motion cannot claim that the chassis has settled.
        settled = next((rest[index][0] for index in range(len(rest) - 1)
                        if rest[index][1] <= threshold and rest[index + 1][1] <= threshold), None)
        if settled is None:
            raise CalibrationFitError(
                "yaw response did not settle during the final rest phase", diagnostics)
        response_gain = statistics.mean(gains)
        delay, settle = max(delays), max(0.0, settled)
        k_yaw = _calibrated_p(response_gain, delay, settle)
        profile = {"schema": SCHEMA, "robot_key": self.robot_key,
                   "created_at_ms": int(time.time() * 1000), "k_yaw": k_yaw,
                   "yaw_damping": 0.0, "response_gain": response_gain,
                   "delay_s": delay, "settling_s": settle,
                   "left_gain": gains[0], "right_gain": gains[1],
                   "noisy_odom": filtered_outlier}
        try:
            return validate_profile(profile, self.robot_key), diagnostics
        except ValueError as error:
            raise CalibrationFitError(str(error), diagnostics) from error
