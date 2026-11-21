import zlib

import cv2
import numpy as np
import pytest

from plugins.camera_geometry import CameraModel, RgbdPreprocessor, decode_depth_zlib
from plugins.camera_info import inherit


def declaration(width=5, height=3, **changes):
    return dict(dict(width=width, height=height, K=[2., 0., 2., 0., 2., 1., 0., 0., 1.]), **changes)


def test_identity_geometry_keeps_jpeg_pixels_and_accepts_legacy_format():
    image = np.zeros((3, 5, 3), dtype=np.uint8)
    raw = np.full((3, 5), 1000, dtype=np.uint16)
    prepared = RgbdPreprocessor(declaration(), declaration())
    assert prepared.rectify_color(image) is image
    np.testing.assert_array_equal(prepared.align_depth(raw), np.ones((3, 5)))


def test_reprojection_keeps_target_z_and_nearest_in_collision():
    matrix = np.eye(4)
    matrix[2, 3] = 3.
    source = declaration(2, 1, K=[2., 0., 0., 0., 2., 0., 0., 0., 1.],
                         vendor={'color_from_depth': matrix.tolist(), 'depth_scale_m': .25})
    target = declaration(3, 1, K=[2., 0., 0., 0., 2., 0., 0., 0., 1.])
    raw = np.array([[4, 8]], dtype=np.uint16)
    depth = RgbdPreprocessor(target, source).align_depth(raw)
    np.testing.assert_array_equal(depth, [[4., 0., 0.]])


def test_identity_extrinsic_with_different_k_projects_without_resize():
    source = declaration(3, 1, K=[2., 0., 1., 0., 2., 0., 0., 0., 1.])
    target = declaration(5, 1, K=[2., 0., 2., 0., 2., 0., 0., 0., 1.])
    depth = RgbdPreprocessor(target, source).align_depth(np.array([[0, 750, 0]], dtype=np.uint16))
    np.testing.assert_allclose(depth, [[0., 0., .75, 0., 0.]], rtol=1e-6)


def test_fisheye_zero_coefficients_still_change_rays_and_pixels():
    model = CameraModel(declaration(distortion_model='equidistant', D=[0.]*4))
    measured = np.array([[0., 0., 0., 0., 2.], [0.]*5, [0.]*5], dtype=np.float32)
    expected = cv2.fisheye.undistortPoints(np.array([[[4., 0.]]]), model.K, np.zeros(4)).reshape(2)
    point, = model.backproject(measured)
    np.testing.assert_allclose(point, [*(expected*2.), 2.], rtol=1e-6)
    image = np.ones((3, 5, 3), dtype=np.uint8)
    assert model.rectify_color(image) is not image


def test_metre_float_depth_and_overflow_are_handled_at_input():
    raw = np.array([[.75, 0., np.inf]], dtype='<f4')
    blob = zlib.compress(raw.tobytes())
    actual = decode_depth_zlib(blob, 3, 1, floating=True)
    depth = RgbdPreprocessor(declaration(3, 1), declaration(3, 1, format='image/depth-f32-zlib')).align_depth(actual)
    np.testing.assert_array_equal(depth, [[.75, 0., 0.]])
    with pytest.raises(ValueError):
        decode_depth_zlib(blob, 2, 1, floating=True)


def test_legacy_coincident_axes_are_independent_of_frame_labels():
    model = RgbdPreprocessor(declaration(vendor={'frame_id': 'rgb'}),
                             declaration(vendor={'frame_id': 'depth'}))
    np.testing.assert_array_equal(model.color_from_depth, np.eye(4))


def test_non_rigid_numeric_extrinsic_is_rejected():
    matrix = np.eye(4)
    matrix[0, 0] = 2.
    with pytest.raises(ValueError, match='color_from_depth'):
        RgbdPreprocessor(declaration(), declaration(color_from_depth=matrix))


def test_numeric_camera_parameters_inherit_without_identity():
    metadata = declaration(depth_scale_m=.002, color_from_depth=np.eye(4).tolist())
    derived, = inherit(metadata, topic='/cloud', fmt='sensor/pointcloud2', stage='pointcloud')
    assert derived['K'] == metadata['K']
    assert derived['depth_scale_m'] == .002
    assert derived['color_from_depth'] == metadata['color_from_depth']
