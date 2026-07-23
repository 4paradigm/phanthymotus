from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path


INSPECTION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INSPECTION_ROOT))

from storage.layout import (  # noqa: E402
    HardwareIdentity,
    detect_device_id,
    detect_robot_identity,
    detect_source_identity,
    instance_storage_slug,
    segment_basename,
    segment_start_ns_from_name,
    source_storage_slug,
    storage_relative_directory,
)


class StorageLayoutTest(unittest.TestCase):
    def test_provisioned_robot_sn_has_priority_and_is_labeled_as_manufacturer_serial(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            serial = Path(tempdir) / "robot-sn"
            serial.write_text("G1-SN-20260723\n")

            identity = detect_robot_identity(serial_paths=(
                (serial, "provisioned-robot-sn", "unitree", True),
            ))

            self.assertEqual("unitree-g1-sn-20260723", identity.value)
            self.assertEqual("provisioned-robot-sn", identity.source)
            self.assertTrue(identity.manufacturer_serial)

    def test_jetson_serial_becomes_automatic_device_id(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            serial = Path(tempdir) / "device-tree" / "serial-number"
            serial.parent.mkdir()
            serial.write_bytes(b"1424525045894\x00")
            self.assertEqual(
                "jetson-1424525045894",
                detect_device_id(serial_paths=(serial,)),
            )

            identity = detect_robot_identity(serial_paths=(
                (serial, "jetson-module-serial", "jetson", False),
            ))
            self.assertEqual("jetson-1424525045894", identity.value)
            self.assertEqual("jetson-module-serial", identity.source)
            self.assertFalse(identity.manufacturer_serial)

    def test_source_device_uses_real_usb_serial_or_explicit_topology_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            usb_root = Path(tempdir)
            dji = usb_root / "1-2.3"
            dji.mkdir()
            (dji / "idVendor").write_text("2ca3\n")
            (dji / "idProduct").write_text("4011\n")
            (dji / "manufacturer").write_text("DJI Technology Co., Ltd.\n")
            (dji / "product").write_text("Wireless Mic Rx\n")
            (dji / "serial").write_text("XSP12345678B\n")
            robot = HardwareIdentity("jetson-1424525045894", "jetson-module-serial", False)

            microphone = detect_source_identity(
                "/phanthymotus_g1_driver/ext_mic/card_abc/audio",
                "audio",
                robot_identity=robot,
                usb_root=usb_root,
            )
            self.assertEqual("dji-xsp12345678b", microphone.value)
            self.assertEqual("usb-manufacturer-serial", microphone.source)
            self.assertTrue(microphone.manufacturer_serial)

            for child in usb_root.iterdir():
                for item in child.iterdir():
                    item.unlink()
                child.rmdir()
            insta360 = usb_root / "1-3"
            insta360.mkdir()
            (insta360 / "idVendor").write_text("2e1a\n")
            (insta360 / "idProduct").write_text("4c06\n")
            (insta360 / "manufacturer").write_text("Insta360\n")
            (insta360 / "product").write_text("Insta360 Link 2 Pro\n")

            camera = detect_source_identity(
                "/phanthymotus_g1_driver/ext_camera/card_abc/rgb",
                "video",
                robot_identity=robot,
                usb_root=usb_root,
            )
            self.assertEqual("insta360-2e1a4c06-port-1-3", camera.value)
            self.assertEqual("usb-topology-composite", camera.source)
            self.assertFalse(camera.manufacturer_serial)

    def test_ambiguous_same_category_usb_devices_use_explicit_topic_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            usb_root = Path(tempdir)
            for port, product in (("1-3", "Insta360 Link 2 Pro"), ("1-4", "USB Camera")):
                device = usb_root / port
                device.mkdir()
                (device / "idVendor").write_text("2e1a\n")
                (device / "idProduct").write_text("4c06\n")
                (device / "manufacturer").write_text("Insta360\n")
                (device / "product").write_text(product + "\n")
            robot = HardwareIdentity("jetson-1424525045894", "jetson-module-serial", False)

            camera = detect_source_identity(
                "/phanthymotus_g1_driver/ext_camera/card_mrvxi910w7ye/rgb",
                "video",
                robot_identity=robot,
                usb_root=usb_root,
            )

            self.assertTrue(camera.value.startswith("jetson-1424525045894-ext-camera-rgb-"))
            self.assertEqual("topic-composite-fallback", camera.source)
            self.assertFalse(camera.manufacturer_serial)

    def test_builtin_microphone_uses_robot_scoped_composite_identity(self) -> None:
        robot = HardwareIdentity("jetson-1424525045894", "jetson-module-serial", False)

        source = detect_source_identity(
            "/phanthymotus_g1_driver/mic/audio",
            "audio",
            robot_identity=robot,
            usb_root=Path("/path/that/does/not/exist"),
        )

        self.assertEqual("jetson-1424525045894-builtin-mic-array", source.value)
        self.assertEqual("robot-builtin-composite", source.source)

    def test_v2_layout_uses_robot_modality_device_and_local_date(self) -> None:
        wall_ns = int(datetime(2026, 7, 22, 10, 23, 45, tzinfo=timezone.utc).timestamp()) * 1_000_000_000 + 123
        robot = HardwareIdentity("jetson-1424525045894", "jetson-module-serial", False)
        source = HardwareIdentity("insta360-2e1a4c06-port-1-3", "usb-topology-composite", False)

        directory = storage_relative_directory(robot, source, "video", wall_ns)
        basename = segment_basename(wall_ns, 7, "mp4")

        self.assertEqual(
            Path(
                "robot=jetson-1424525045894",
                "video",
                "device=insta360-2e1a4c06-port-1-3",
                "date=2026-07-22",
            ),
            directory,
        )
        self.assertEqual("20260722T182345.000000123+0800--000007.mp4", basename)
        self.assertEqual(wall_ns, segment_start_ns_from_name(Path(basename)))

    def test_canvas_random_id_is_replaced_by_readable_topic_slug(self) -> None:
        topic = "/phanthymotus_g1_driver/ext_camera/card-mrvusdyxxjln/rgb"

        self.assertEqual("ext-camera-rgb", source_storage_slug(topic))
        storage_slug = instance_storage_slug("card-mrvusdyxxjln", topic)
        self.assertRegex(storage_slug, r"^ext-camera-rgb--[0-9a-f]{8}$")
        self.assertNotIn("mrvusdyxxjln", storage_slug)


if __name__ == "__main__":
    unittest.main()
