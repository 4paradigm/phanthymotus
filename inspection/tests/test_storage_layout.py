from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


INSPECTION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INSPECTION_ROOT))

from storage.layout import detect_device_id, instance_storage_slug, source_storage_slug  # noqa: E402


class StorageLayoutTest(unittest.TestCase):
    def test_jetson_serial_becomes_automatic_device_id(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            serial = Path(tempdir) / "device-tree" / "serial-number"
            serial.parent.mkdir()
            serial.write_bytes(b"1424525045894\x00")
            self.assertEqual(
                "jetson-1424525045894",
                detect_device_id(serial_paths=(serial,)),
            )

    def test_canvas_random_id_is_replaced_by_readable_topic_slug(self) -> None:
        topic = "/phanthymotus_g1_driver/ext_camera/card-mrvusdyxxjln/rgb"

        self.assertEqual("ext-camera-rgb", source_storage_slug(topic))
        storage_slug = instance_storage_slug("card-mrvusdyxxjln", topic)
        self.assertRegex(storage_slug, r"^ext-camera-rgb--[0-9a-f]{8}$")
        self.assertNotIn("mrvusdyxxjln", storage_slug)


if __name__ == "__main__":
    unittest.main()
