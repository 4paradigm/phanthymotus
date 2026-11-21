"""One Euro smoothing, using source time and normalized coordinates."""
import math
import numpy as np


class OneEuro:
    def __init__(self, min_cutoff=2.0, beta=8.0, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.raw = self.value = self.derivative = self.timestamp = None

    @staticmethod
    def alpha(cutoff, dt):
        return 1 / (1 + 1 / (2 * math.pi * cutoff * dt))

    def __call__(self, value, timestamp):
        value = np.asarray(value, dtype=np.float64)
        if (self.timestamp is None or timestamp <= self.timestamp
                or timestamp - self.timestamp > 0.5):
            self.raw, self.value = value.copy(), value.copy()
            self.derivative = np.zeros_like(value)
        else:
            dt = timestamp - self.timestamp
            derivative = (value - self.raw) / dt
            a = self.alpha(self.d_cutoff, dt)
            self.derivative += a * (derivative - self.derivative)
            a = self.alpha(self.min_cutoff + self.beta * np.abs(self.derivative), dt)
            self.value += a * (value - self.value)
            self.raw = value.copy()
        self.timestamp = timestamp
        return self.value.copy()


class SimilarityFilter:
    """Stabilize center, roll and log scale without bending the eye pair.

    Translation gain uses face size so a small distant face does not acquire a
    much larger motion lag merely because coordinates are screen-normalized.
    Local expression residuals remain in the current observation.
    """
    def __init__(self):
        self.center = OneEuro()
        self.angle = OneEuro(beta=2.)
        self.scale = OneEuro(beta=2.)
        self.raw_angle = self.unwrapped_angle = None

    def __call__(self, points, timestamp, size, box):
        delta = points[1]-points[0]
        distance = max(float(np.linalg.norm(delta)), 1.)
        center = points[:2].mean(0)
        angle = math.atan2(delta[1], delta[0])
        if self.raw_angle is None:
            self.unwrapped_angle = angle
        else:
            change = angle-self.raw_angle
            self.unwrapped_angle += math.atan2(math.sin(change), math.cos(change))
        self.raw_angle = angle
        extent = max(16., float(max(box[2:])))
        self.center.beta = 8.*max(size)/extent
        filtered_center = self.center(center/size, timestamp)*size
        filtered_angle = float(self.angle([self.unwrapped_angle], timestamp)[0])
        filtered_distance = math.exp(float(self.scale([math.log(distance/max(size))], timestamp)[0]))*max(size)
        change = filtered_angle-self.unwrapped_angle
        c, s = math.cos(change), math.sin(change)
        rotation = np.array([[c,-s],[s,c]])
        return (points-center) @ rotation.T * (filtered_distance/distance) + filtered_center
