"""Numeric camera models shared by RGB-D odometry, point clouds and SLAM."""
import zlib

import numpy as np

from .pointcloud_slam_transforms import rigid


def camera_parameter(declaration, key, default=None):
    value = declaration.get(key)
    if value is not None:
        return value
    vendor = declaration.get('vendor')
    return vendor.get(key, default) if isinstance(vendor, dict) else default


def depth_scale(declaration):
    default = 1. if declaration.get('format') in ('image/depth-f32', 'image/depth-f32-zlib') else .001
    scale = camera_parameter(declaration, 'depth_scale_m', default)
    if (isinstance(scale, bool) or not isinstance(scale, (int, float)) or
            not np.isfinite(scale) or not 0 < scale <= np.finfo(np.float32).max):
        raise ValueError('depth_scale_m must be finite and positive')
    return scale


def zlib_decompress(data, cap=1 << 20):
    stream = zlib.decompressobj()
    raw = stream.decompress(data, cap)
    if stream.unconsumed_tail or stream.unused_data or not stream.eof:
        raise ValueError('invalid zlib stream or decompressed output exceeds the depth-frame cap')
    return raw


def decode_depth_zlib(data, width, height, *, floating=False):
    if any(type(n) is not int or not 0 < n <= 4096 for n in (width, height)):
        raise ValueError('depth dimensions must be positive integers up to 4096')
    dtype = np.dtype('<f4' if floating else '<u2')
    size = width * height * dtype.itemsize
    try:
        raw = zlib_decompress(data, cap=size + 1)
    except zlib.error as error:
        raise ValueError('invalid compressed depth') from error
    if len(raw) != size:
        raise ValueError('depth size differs from camera geometry')
    return np.frombuffer(raw, dtype=dtype).reshape(height, width)


class CameraModel:
    """One optical plane; K also defines its undistorted pinhole plane."""

    def __init__(self, declaration):
        self.width, self.height = declaration.get('width'), declaration.get('height')
        if any(type(n) is not int or not 0 < n <= 4096 for n in (self.width, self.height)):
            raise ValueError('camera dimensions must be positive integers up to 4096')
        self.K = np.asarray(declaration.get('K'), dtype=float).reshape(3, 3).copy()
        fx, fy = self.K[0, 0], self.K[1, 1]
        if (not np.isfinite(self.K).all() or min(fx, fy) <= 0 or self.K[1, 0] != 0 or
                not np.array_equal(self.K[2], [0., 0., 1.])):
            raise ValueError('camera K must be a finite upper-triangular intrinsic matrix with positive focal lengths')
        coefficients = declaration.get('D')
        self.D = np.asarray([] if coefficients is None else coefficients, dtype=float).copy()
        model = declaration.get('distortion_model') or 'pinhole'
        self.fisheye = model in ('equidistant', 'fisheye')
        allowed = {'plumb_bob': (0, 4, 5), 'rational_polynomial': (8, 12, 14),
                   'equidistant': (4,), 'fisheye': (4,)}
        if self.D.ndim != 1 or not np.isfinite(self.D).all():
            raise ValueError('camera D must contain finite distortion coefficients')
        self.distorted = self.fisheye or bool(np.any(self.D))
        if (model in allowed and self.D.size not in allowed[model]) or (
                model not in allowed and self.distorted):
            raise ValueError('camera distortion coefficient count or model is unsupported')
        if model not in allowed and model not in ('pinhole', 'none', 'unknown'):
            raise ValueError('unsupported camera distortion model')
        self.rectified_K = self.K.copy()
        self.rectified_K[0, 1] = 0.
        self._color_maps = None
        self._full_rays = self._decimated_rays = None

    def resized(self, width, height):
        if (width, height) == (self.width, self.height):
            return self
        k = self.K.copy()
        k[0] *= width / self.width
        k[1] *= height / self.height
        model = 'equidistant' if self.fisheye else 'plumb_bob' if self.D.size <= 5 else 'rational_polynomial'
        return CameraModel(dict(width=width, height=height, K=k, D=self.D if self.distorted else [],
                                distortion_model=model if self.distorted else 'pinhole'))

    def rectify_color(self, image):
        if image.shape != (self.height, self.width, 3):
            raise ValueError('RGB dimensions differ from camera geometry')
        if not self.distorted and self.K[0, 1] == 0.:
            return image
        import cv2
        if self._color_maps is None:
            init_map = (cv2.fisheye.initUndistortRectifyMap if self.fisheye else cv2.initUndistortRectifyMap)
            k = self.K.copy()
            skew = k[0, 1]
            k[0, 1] = 0.
            x, y = init_map(k, self.D if self.D.size else None, np.eye(3), self.rectified_K,
                            (self.width, self.height), cv2.CV_32FC1)
            if skew:
                x += skew * (y - k[1, 2]) / k[1, 1]
            self._color_maps = x, y
        return cv2.remap(image, *self._color_maps, interpolation=cv2.INTER_LINEAR)

    def _rays(self, stride):
        cached = self._full_rays if stride == 1 else self._decimated_rays
        if cached is not None and cached[0] == stride:
            return cached[1]
        v, u = np.indices(((self.height + stride - 1) // stride,
                           (self.width + stride - 1) // stride), dtype=np.float32)
        u *= stride
        v *= stride
        fx, skew, cx = self.K[0]
        fy, cy = self.K[1, 1:]
        y = (v - cy) / fy
        if self.distorted:
            import cv2
            undistort = cv2.fisheye.undistortPoints if self.fisheye else cv2.undistortPoints
            k = self.K.copy()
            k[0, 1] = 0.
            uv = np.stack((u - skew*y, v), axis=-1).reshape(-1, 1, 2)
            rays = undistort(uv, k, self.D).reshape(*u.shape, 2)
            rays[(~np.isfinite(rays)).any(axis=-1) | (np.abs(rays) >= 1e6).any(axis=-1)] = np.nan
        else:
            rays = np.stack(((u-cx-skew*y)/fx, y), axis=-1)
        if stride == 1:
            self._full_rays = stride, rays
        else:
            self._decimated_rays = stride, rays
        return rays

    def backproject(self, depth_m, stride=1, minimum=0., maximum=np.inf):
        if depth_m.shape != (self.height, self.width):
            raise ValueError('depth dimensions differ from camera geometry')
        z = depth_m[::stride, ::stride]
        rays = self._rays(stride)
        valid = np.isfinite(z) & (z > 0.) & (z >= minimum) & (z <= maximum)
        valid &= np.isfinite(rays).all(axis=-1)
        z = z[valid]
        return np.column_stack((rays[valid] * z[:, None], z)).astype(np.float32, copy=False)


def nearest_depth(u, v, z, width, height):
    """Drop holes and keep the nearest target-frame optical Z per pixel."""
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height) & (z > 0) & (z <= np.finfo(np.float32).max)
    indices = v[valid].astype(np.intp)*width + u[valid].astype(np.intp)
    depth = np.full(width*height, np.inf, dtype=np.float32)
    np.minimum.at(depth, indices, z[valid].astype(np.float32))
    depth[~np.isfinite(depth)] = 0.
    return depth.reshape(height, width)


def project_depth(points, camera_from_cloud, K, width, height):
    transform = rigid(camera_from_cloud)
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    moved = points @ transform[:3, :3].T + transform[:3, 3]
    moved = moved[np.isfinite(moved).all(axis=1) & (moved[:, 2] > 0)]
    pixels = moved @ np.asarray(K, dtype=float).reshape(3, 3).T
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        uv = np.rint(pixels[:, :2]/pixels[:, 2:3])
    return nearest_depth(uv[:, 0], uv[:, 1], moved[:, 2], width, height)


def camera_transform(value, field='color_from_depth'):
    matrix = rigid(value)
    r = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0., 0., 0., 1.], atol=1e-8, rtol=0.) or
            not np.allclose(r.T @ r, np.eye(3), atol=1e-4, rtol=0.) or np.linalg.det(r) <= 0.):
        raise ValueError(f'{field} must be a rigid homogeneous transform')
    return matrix


class RgbdPreprocessor:
    def __init__(self, color, depth):
        self.color, self.depth = CameraModel(color), CameraModel(depth)
        self.width, self.height, self.K = self.color.width, self.color.height, self.color.rectified_K
        self.scale = depth_scale(depth)
        transform = camera_parameter(depth, 'color_from_depth', camera_parameter(color, 'color_from_depth'))
        # The legacy RGB-D input format represents coincident optical planes.
        self.color_from_depth = np.eye(4) if transform is None else camera_transform(transform)
        self._aligned = (not self.depth.distorted and (self.width, self.height) ==
                         (self.depth.width, self.depth.height) and np.array_equal(self.K, self.depth.K) and
                         np.array_equal(self.color_from_depth, np.eye(4)))
        self._projection = None

    def rectify_color(self, image):
        return self.color.rectify_color(image)

    def align_depth(self, raw_depth):
        if raw_depth.shape != (self.depth.height, self.depth.width):
            raise ValueError('depth dimensions differ from camera geometry')
        with np.errstate(invalid='ignore', over='ignore'):
            depth = raw_depth.astype(np.float32) * self.scale
        if self._aligned:
            depth[~np.isfinite(depth) | (depth <= 0.)] = 0.
            return depth
        if self._projection is None:
            # Precompute target homogeneous ray directions once for this geometry.
            rays = self.depth._rays(1).astype(float)
            rotation = self.K @ self.color_from_depth[:3, :3]
            directions = tuple(row[0]*rays[:, :, 0]+row[1]*rays[:, :, 1]+row[2] for row in rotation)
            self._projection = directions, self.K @ self.color_from_depth[:3, 3]
        (x, y, z), t = self._projection
        valid = np.isfinite(depth) & (depth > 0.)
        measured = depth[valid]
        optical_z = z[valid]*measured + t[2]
        with np.errstate(invalid='ignore', divide='ignore'):
            u = np.rint((x[valid]*measured+t[0])/optical_z)
            v = np.rint((y[valid]*measured+t[1])/optical_z)
        return nearest_depth(u, v, optical_z, self.width, self.height)
