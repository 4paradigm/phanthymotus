"""motus.camera/1 for the browser camera: only when a FOV was typed in."""
import io
import math
import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def jpeg(width, height, marker=0xC0, extra_segments=True):
    """A byte string with a valid SOF header — not a decodable image."""
    out = io.BytesIO()
    out.write(b'\xff\xd8')
    if extra_segments:
        # APP0/JFIF then DQT, the two a browser's encoder puts before SOF.
        out.write(b'\xff\xe0' + struct.pack('>H', 16) + b'JFIF\x00' + b'\x00' * 9)
        out.write(b'\xff\xdb' + struct.pack('>H', 67) + b'\x00' * 65)
    out.write(bytes([0xff, marker]) + struct.pack('>HBHHB', 17, 8, height, width, 3))
    out.write(b'\x00' * 6 + b'\xff\xd9')
    return out.getvalue()


import remote_camera


class JpegSizeTest(unittest.TestCase):
    def test_reads_size_past_app_and_quantisation_segments(self):
        self.assertEqual(remote_camera.jpeg_size(jpeg(1280, 720)), (1280, 720))
        self.assertEqual(remote_camera.jpeg_size(jpeg(640, 480, marker=0xC2)), (640, 480))

    def test_rejects_non_jpeg_and_truncated_input(self):
        for bad in (b'', b'not a jpeg', b'\xff\xd8', jpeg(640, 480)[:12]):
            self.assertIsNone(remote_camera.jpeg_size(bad))

    def test_does_not_mistake_a_huffman_table_for_a_frame_header(self):
        # 0xC4 is DHT and sits inside the SOF marker range; reading it as SOF
        # yields a plausible-looking size from table bytes.
        self.assertIsNone(remote_camera.jpeg_size(
            b'\xff\xd8' + b'\xff\xc4' + struct.pack('>H', 20) + b'\x11' * 18))


class DeclarationTest(unittest.TestCase):
    def setUp(self):
        self._saved = remote_camera._frame_size
        remote_camera._frame_size = (1280, 720)

    def tearDown(self):
        remote_camera._frame_size = self._saved

    def test_no_fov_declares_nothing(self):
        # The point of the whole change: silence, not a guessed focal length.
        for cfg in ({}, {'camera_fov_deg': ''}, {'camera_fov_deg': None},
                    {'camera_fov_deg': 'wide'}, {'camera_fov_deg': 0},
                    {'camera_fov_deg': 180}, {'camera_fov_deg': -70}):
            self.assertEqual(remote_camera.declaration(cfg), [], cfg)

    def test_no_frame_yet_declares_nothing(self):
        remote_camera._frame_size = None
        self.assertEqual(remote_camera.declaration({'camera_fov_deg': 70}), [])

    def test_k_matches_the_typed_field_of_view(self):
        d = remote_camera.declaration({'camera_fov_deg': 70})[0]
        fx, _, cx, _, fy, cy, _, _, one = d['K']
        self.assertAlmostEqual(fx, 640 / math.tan(math.radians(35)), places=4)
        self.assertEqual(fx, fy)                      # square pixels
        self.assertEqual((cx, cy, one), (639.5, 359.5, 1.0))
        self.assertAlmostEqual(d['half_fov_rad'], math.radians(35), places=5)
        # The angle a consumer derives back from K must be the one typed in.
        self.assertAlmostEqual(math.atan((d['width'] / 2) / fx),
                               math.radians(35), places=5)

    def test_declares_the_real_frame_size_not_the_configured_one(self):
        remote_camera._frame_size = (800, 600)
        d = remote_camera.declaration({'camera_fov_deg': 70, 'width': 640})[0]
        self.assertEqual((d['width'], d['height']), (800, 600))

    def test_shape_follows_the_spec(self):
        d = remote_camera.declaration({'camera_fov_deg': 70})[0]
        self.assertEqual(d['schema'], 'motus.camera/1')
        self.assertEqual(d['topic'], '/remote_control/camera')
        # source is required, and a typed-in number is `manual`, not `measured`.
        self.assertEqual(d['source'], 'manual')
        self.assertIsNone(d['D'])
        self.assertIsNone(d['half_fov_v_rad'])
        self.assertEqual(d['pipeline'], ['agentcore/remote_camera'])

    def test_k_passes_what_a_metric_depth_consumer_checks(self):
        """The contract DepthART's `camera_matrix` applies, asserted here.

        Deliberately a local copy of the checks and not an import of
        perception's parser: the spec is a document, not a shared package (see
        `phanthymotus-driver/README_dev.md` § Camera Parameters), and each side
        runs its own contract tests so the two repos can release apart. The
        import would also drag perception's top-level `plugins` package into
        this suite, which is the collision actucore and perception already
        cannot share a pytest process over.
        """
        d = remote_camera.declaration({'camera_fov_deg': 70})[0]
        K = d['K']
        self.assertEqual(len(K), 9)
        self.assertTrue(all(isinstance(v, float) and math.isfinite(v) for v in K))
        self.assertGreater(K[0], 0)                  # fx
        self.assertGreater(K[4], 0)                  # fy
        self.assertEqual(K[6:], [0.0, 0.0, 1.0])     # bottom row
        self.assertEqual(K[3], 0.0)                  # no skew
        self.assertGreater(d['width'], 0)
        self.assertGreater(d['height'], 0)

if __name__ == '__main__':
    unittest.main()
