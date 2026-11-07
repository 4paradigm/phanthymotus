"""Sensor visibility scores with explicit support sets and legacy compatibility."""
from copy import deepcopy
import math
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from .pointcloud_slam_observation import legacy_scoring


class VisibilityView:
    def __init__(self, points, camera, optical_from_child, voxel, *, scoring=None):
        self.camera = camera
        self.optical_from_child = optical_from_child
        self.scoring = deepcopy(scoring)
        self.score_metadata = deepcopy(scoring) if scoring is not None else legacy_scoring(visible=True)
        self.pixel_bin = scoring['visibility']['pixel_bin'] if scoring is not None else 4
        points = self._prepare(points, voxel)
        width, height = camera[:2]
        self.depth = np.full(((height+self.pixel_bin-1)//self.pixel_bin,
                              (width+self.pixel_bin-1)//self.pixel_bin), np.inf)
        u, v, z, inside = self.project(points)
        np.minimum.at(self.depth, (v[inside], u[inside]), z[inside])

    def _prepare(self, points, voxel):
        points = np.asarray(points, dtype=float)
        if points.size == 0:
            points = points.reshape(0, 3)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('expected Nx3 cloud')
        points = points[np.isfinite(points).all(axis=1)]
        if self.scoring is not None:
            points = points[self.project(points)[3]]
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        self.points = np.asarray(cloud.voxel_down_sample(voxel).points).copy()
        if self.scoring is not None:
            self.points = self.points[self.project(self.points)[3]]
        self.tree = cKDTree(self.points)
        self.bands = np.digitize(self.range_values(self.points), self.score_metadata['band_edges_m'])
        return points

    def range_values(self, points):
        return self.project(points)[2]

    def project(self, points):
        width, height, fx, fy, cx, cy = self.camera
        t = self.optical_from_child
        optical = points @ t[:3, :3].T + t[:3, 3]
        z = optical[:, 2]
        x = fx * optical[:, 0] / np.maximum(z, 1e-6) + cx
        y = fy * optical[:, 1] / np.maximum(z, 1e-6) + cy
        inside = (z > 0) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
        return (np.floor(np.clip(x, 0, width-1)/self.pixel_bin).astype(int),
                np.floor(np.clip(y, 0, height-1)/self.pixel_bin).astype(int), z, inside)

    def visible(self, projected, limit):
        u, v, z, inside = self.project(projected)
        depth = self.depth[v, u]
        return inside & np.isfinite(depth) & (z <= depth + limit)

    def score(self, target, transform, limit):
        projected = self.points @ transform[:3, :3].T + transform[:3, 3]
        keep = target.visible(projected, limit)
        bands = self.bands[keep]
        counts = np.bincount(bands, minlength=len(self.score_metadata['band_edges_m'])+1)
        distances = target.tree.query(projected)[0]
        modern = self.scoring is not None
        matched = (distances <= limit) & keep if modern else distances <= limit
        pairs = int(np.count_nonzero(matched))
        support = int(np.count_nonzero(keep))
        denominator = support if modern else len(projected)
        required = self.score_metadata['required_bands']
        error = None
        if np.all(counts[required]):
            distance = distances[keep]
            squared = np.minimum(distance, limit)**2
            totals = np.bincount(bands, weights=squared, minlength=len(counts))
            error = float(np.sqrt(np.mean(totals[required]/counts[required])))
        result = dict(coverage=support/len(projected) if len(projected) else 0.,
                      counts=counts.tolist(), error_m=error, correspondences=pairs,
                      fitness=pairs/denominator if denominator else 0.,
                      coverage_numerator=support, coverage_denominator=len(projected),
                      fitness_numerator=pairs, fitness_denominator=denominator,
                      score_metadata=deepcopy(self.score_metadata),
                      reason=None if error is not None else 'missing_range_support')
        if modern:
            result['range_stratified_rmse_m'] = error
        return result


class LidarVisibilityView(VisibilityView):
    def __init__(self, points, observation, voxel, scoring):
        self.parameters = observation['parameters']
        self.sensor_from_child = np.asarray(observation['sensor_from_child'])
        self.scoring = deepcopy(scoring)
        self.score_metadata = deepcopy(scoring)
        p = self.parameters; v = scoring['visibility']
        self.width = math.ceil(p['azimuth_span_rad']/v['azimuth_bin_rad'])
        self.height = math.ceil((p['elevation_max_rad']-p['elevation_min_rad'])/v['elevation_bin_rad'])
        self.full_circle = math.isclose(p['azimuth_span_rad'], 2*math.pi, abs_tol=1e-9)
        self.azimuth_bin = p['azimuth_span_rad']/self.width
        self.elevation_bin = (p['elevation_max_rad']-p['elevation_min_rad'])/self.height
        points = self._prepare(points, voxel)
        u, v, ranges, inside = self.project(points)
        self.depth = {}
        for x, y, r in zip(u[inside], v[inside], ranges[inside]):
            self.depth[x, y] = min(self.depth.get((x, y), np.inf), r)

    def project(self, points):
        t = self.sensor_from_child
        sensor = points @ t[:3, :3].T + t[:3, 3]
        radius = np.linalg.norm(sensor, axis=1)
        azimuth = np.arctan2(sensor[:, 1], sensor[:, 0])
        elevation = np.arctan2(sensor[:, 2], np.linalg.norm(sensor[:, :2], axis=1))
        p = self.parameters
        offset = (azimuth-p['azimuth_min_rad']) % (2*math.pi)
        inside = ((radius >= p['min_range_m']) & (radius <= p['max_range_m']) &
                  (elevation >= p['elevation_min_rad']) & (elevation <= p['elevation_max_rad']))
        if not self.full_circle:
            inside &= offset <= p['azimuth_span_rad']
        for start, end in p['blind_azimuth_intervals_rad']:
            inside &= (azimuth-start) % (2*math.pi) >= (end-start) % (2*math.pi)
        u = np.floor(offset/self.azimuth_bin + (.5 if self.full_circle else 0)).astype(np.int64)
        u = u % self.width if self.full_circle else np.clip(u, 0, self.width-1)
        v = np.floor((elevation-p['elevation_min_rad'])/self.elevation_bin).astype(np.int64)
        return u, np.clip(v, 0, self.height-1), radius, inside

    def visible(self, projected, limit):
        u, v, radius, inside = self.project(projected)
        depth = np.array([self.depth.get((x, y), np.inf) for x, y in zip(u, v)])
        return inside & np.isfinite(depth) & (radius <= depth+limit)


def split_decision(score, min_coverage, max_error, min_count):
    required = score.get('score_metadata', {}).get('required_bands', range(len(score['counts'])))
    return dict(coverage=score['coverage'] >= min_coverage,
                range_support=min(score['counts'][i] for i in required) >= min_count,
                visible_error=score['error_m'] is not None and score['error_m'] <= max_error)

class ReferenceWindow(VisibilityView):
    """A deduplicated reference in the first view's coordinates with real camera visibility."""

    def __init__(self, raw_points, views, poses, voxel):
        self.views = views
        self.scoring = deepcopy(views[0].scoring)
        self.score_metadata = deepcopy(views[0].score_metadata)
        if any(view.score_metadata != self.score_metadata for view in views):
            raise ValueError('reference observations require one scoring contract')
        self.from_anchor = [np.linalg.inv(pose) @ poses[0] for pose in poses]
        chunks, depths = [], []
        for points, view, transform in zip(raw_points, views, self.from_anchor):
            if self.scoring is not None:
                points = points[view.project(points)[3]]
            inverse = np.linalg.inv(transform)
            chunks.append(points @ inverse[:3, :3].T + inverse[:3, 3])
            depths.append(view.range_values(points))
        points = np.concatenate(chunks)
        if not len(points):
            self.points = points
            self.tree = cKDTree(points)
            self._ranges = np.empty(0)
            self.bands = np.empty(0, dtype=int)
            return
        origin = points.min(axis=0) - voxel/2
        keys = np.floor((points-origin)/voxel).astype(np.int64)
        _, groups, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        self.points = np.zeros((len(counts), 3))
        np.add.at(self.points, groups, points)
        self.points /= counts[:, None]
        self.tree = cKDTree(self.points)
        # Each centroid inherits the acquisition depth of its nearest contributor.
        distance = np.linalg.norm(points-self.points[groups], axis=1)
        ordered = np.lexsort((np.arange(len(points)), distance, groups))
        representatives = ordered[np.r_[True, np.diff(groups[ordered]) != 0]]
        self._ranges = np.concatenate(depths)[representatives]
        self.bands = np.digitize(self._ranges, self.score_metadata['band_edges_m'])

    def range_values(self, points):
        return self._ranges[self.tree.query(points)[1]]

    def visible(self, projected, limit):
        keep = np.zeros(len(projected), dtype=bool)
        for view, transform in zip(self.views, self.from_anchor):
            local = projected @ transform[:3, :3].T + transform[:3, 3]
            keep |= view.visible(local, limit)
        return keep
