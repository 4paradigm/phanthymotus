"""Offline pinhole visibility scoring for archived, camera-origin point clouds."""
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


class VisibilityView:
    def __init__(self, points, camera, optical_from_child, voxel):
        self.camera = camera
        self.optical_from_child = optical_from_child
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        self.points = np.asarray(cloud.voxel_down_sample(voxel).points).copy()
        self.tree = cKDTree(self.points)
        width, height = camera[:2]
        self.depth = np.full(((height + 3)//4, (width + 3)//4), np.inf)
        u, v, z, inside = self.project(points)
        np.minimum.at(self.depth, (v[inside], u[inside]), z[inside])
        self.bands = np.digitize(self.project(self.points)[2], [2., 4.])

    def project(self, points):
        width, height, fx, fy, cx, cy = self.camera
        t = self.optical_from_child
        optical = points @ t[:3, :3].T + t[:3, 3]
        z = optical[:, 2]
        x = fx * optical[:, 0] / np.maximum(z, 1e-6) + cx
        y = fy * optical[:, 1] / np.maximum(z, 1e-6) + cy
        inside = (z > 0) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
        return (np.floor(np.clip(x, 0, width-1)/4).astype(int),
                np.floor(np.clip(y, 0, height-1)/4).astype(int), z, inside)

    def visible(self, projected, limit):
        u, v, z, inside = self.project(projected)
        depth = self.depth[v, u]
        return inside & np.isfinite(depth) & (z <= depth + limit)

    def score(self, target, transform, limit):
        projected = self.points @ transform[:3, :3].T + transform[:3, 3]
        keep = target.visible(projected, limit)
        bands = self.bands[keep]
        counts = np.bincount(bands, minlength=3)
        distances = target.tree.query(projected)[0]
        pairs = int(np.count_nonzero(distances <= limit))
        error = None
        if np.all(counts):
            distance = distances[keep]
            squared = np.minimum(distance, limit)**2
            error = float(np.sqrt(np.mean(np.bincount(bands, weights=squared)/counts)))
        return dict(coverage=float(np.mean(keep)), counts=counts.tolist(), error_m=error,
                    correspondences=pairs, fitness=pairs/len(projected))


def split_decision(score, min_coverage, max_error, min_count):
    return dict(coverage=score['coverage'] >= min_coverage,
                range_support=min(score['counts']) >= min_count,
                visible_error=score['error_m'] is not None and score['error_m'] <= max_error)

class ReferenceWindow(VisibilityView):
    """A deduplicated reference in the first view's coordinates with real camera visibility."""

    def __init__(self, raw_points, views, poses, voxel):
        self.views = views
        self.from_anchor = [np.linalg.inv(pose) @ poses[0] for pose in poses]
        chunks, depths = [], []
        for points, view, transform in zip(raw_points, views, self.from_anchor):
            inverse = np.linalg.inv(transform)
            chunks.append(points @ inverse[:3, :3].T + inverse[:3, 3])
            depths.append(view.project(points)[2])
        points = np.concatenate(chunks)
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
        self.bands = np.digitize(np.concatenate(depths)[representatives], [2., 4.])

    def visible(self, projected, limit):
        keep = np.zeros(len(projected), dtype=bool)
        for view, transform in zip(self.views, self.from_anchor):
            local = projected @ transform[:3, :3].T + transform[:3, 3]
            keep |= view.visible(local, limit)
        return keep
