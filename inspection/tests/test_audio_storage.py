from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace


INSPECTION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INSPECTION_ROOT))

from plugins.audioinspector import AudioInspectorPlugin  # noqa: E402
from plugins.audioinspector.runtime import AudioWritePump, source_stamp_ns  # noqa: E402
from storage.atomic_writer import AudioSegmentWriter  # noqa: E402
from storage.layout import HardwareIdentity  # noqa: E402
from storage.ledger import SegmentLedger  # noqa: E402
from storage.models import SegmentState  # noqa: E402
from storage.recovery import reconcile_audio_store  # noqa: E402


class AudioStorageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.ledger = SegmentLedger(self.root / "state" / "segments.db")

    def tearDown(self) -> None:
        self.ledger.close()
        self.tempdir.cleanup()

    def make_writer(self) -> AudioSegmentWriter:
        return AudioSegmentWriter(
            data_root=self.root / "data",
            ledger=self.ledger,
            card_id="audioinspector",
            instance_id="mic-1",
            input_topic="/robot/mic/audio",
            session_id="session-1",
            robot_identity=HardwareIdentity("g1-sh-sn123", "unitree-robot-sn", True),
            source_identity=HardwareIdentity(
                "g1-sh-sn123-builtin-mic-array",
                "robot-builtin-composite",
                False,
            ),
            segment_seconds=1,
            sample_rate=8,
            channels=1,
            sample_width=2,
        )

    def test_finalize_wav_metadata_and_ledger_are_consistent(self) -> None:
        writer = self.make_writer()
        metadata = writer.write_chunk(b"\x01\x00" * 8, source_stamp_ns=123_000_000)

        self.assertIsNotNone(metadata)
        assert metadata is not None
        metadata_files = list((self.root / "data").rglob("*.json"))
        wav_files = list((self.root / "data").rglob("*.wav"))
        self.assertEqual(1, len(metadata_files))
        self.assertEqual(1, len(wav_files))
        self.assertFalse(list((self.root / "data").rglob("*.part")))
        self.assertFalse(list((self.root / "data").rglob("*.open.json")))

        with wave.open(str(wav_files[0]), "rb") as wav_file:
            self.assertEqual(1, wav_file.getnchannels())
            self.assertEqual(2, wav_file.getsampwidth())
            self.assertEqual(8, wav_file.getframerate())
            self.assertEqual(8, wav_file.getnframes())

        digest = hashlib.sha256(wav_files[0].read_bytes()).hexdigest()
        self.assertEqual(digest, metadata["sha256"])
        self.assertEqual(wav_files[0].stat().st_size, metadata["bytes"])
        record = self.ledger.get(metadata["segment_id"])
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(SegmentState.FINALIZED.value, record["state"])
        self.assertEqual(str(wav_files[0]), record["local_path"])
        relative = wav_files[0].relative_to(self.root / "data")
        self.assertEqual("g1-sh-sn123", relative.parts[0])
        self.assertEqual("audio", relative.parts[1])
        self.assertEqual("g1-sh-sn123-builtin-mic-array", relative.parts[2])
        self.assertRegex(relative.parts[3], r"^\d{4}-\d{2}-\d{2}$")
        self.assertRegex(relative.name, r"^\d{8}T\d{6}\.\d{9}\+0800--\d{6}\.wav$")
        self.assertEqual("audioinspector", metadata["card_id"])
        self.assertEqual("mic-1", metadata["instance_id"])
        self.assertEqual("3.0", metadata["schema_version"])
        self.assertEqual("g1-sh-sn123", metadata["robot_id"])
        self.assertEqual("unitree-robot-sn", metadata["robot_identity_source"])
        self.assertTrue(metadata["robot_identity_is_manufacturer_serial"])
        self.assertEqual("g1-sh-sn123-builtin-mic-array", metadata["source_device_id"])
        self.assertEqual("robot-builtin-composite", metadata["source_identity_source"])
        self.assertFalse(metadata["source_identity_is_manufacturer_serial"])
        self.assertEqual("audio-inspector", metadata["storage_card_slug"])
        self.assertEqual("audio", metadata["storage_modality"])
        self.assertEqual("/".join(relative.parts[:-1]), metadata["storage_relative_directory"])
        self.assertEqual("wal", self.ledger.journal_mode)
        self.assertGreaterEqual(self.ledger.synchronous, 2)

    def test_audio_segment_finalizes_when_byte_limit_arrives_first(self) -> None:
        writer = AudioSegmentWriter(
            data_root=self.root / "data",
            ledger=self.ledger,
            card_id="audioinspector",
            instance_id="mic-size-limit",
            input_topic="/robot/mic/audio",
            session_id="session-size-limit",
            robot_identity=HardwareIdentity("g1-sh-sn123", "unitree-robot-sn", True),
            source_identity=HardwareIdentity(
                "g1-sh-sn123-builtin-mic-array",
                "robot-builtin-composite",
                False,
            ),
            segment_seconds=600,
            max_segment_bytes=8,
            sample_rate=8,
            channels=1,
            sample_width=2,
        )

        metadata = writer.write_chunk(b"\x01\x00" * 4)

        self.assertIsNotNone(metadata)
        assert metadata is not None
        self.assertEqual(4, metadata["samples_or_frames"])
        self.assertFalse(list((self.root / "data").rglob("*.part")))

    def test_unclean_part_is_recovered_to_playable_wav(self) -> None:
        writer = self.make_writer()
        self.assertIsNone(writer.write_chunk(b"\x02\x00" * 4, source_stamp_ns=456_000_000))
        assert writer._raw_handle is not None
        writer._raw_handle.flush()
        writer._raw_handle.close()
        writer._raw_handle = None

        stats = reconcile_audio_store(self.root / "data", self.ledger)

        self.assertEqual(1, stats["parts_recovered"])
        metadata_path = next((self.root / "data").rglob("*.json"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertTrue(metadata["recovered_after_unclean_shutdown"])
        self.assertFalse(metadata["source_stamp_valid"])
        with wave.open(str(metadata_path.with_suffix(".wav")), "rb") as wav_file:
            self.assertEqual(4, wav_file.getnframes())

    def test_v3_part_without_open_state_rebuilds_relative_identity_path(self) -> None:
        writer = self.make_writer()
        self.assertIsNone(writer.write_chunk(b"\x02\x00" * 4))
        assert writer._raw_handle is not None
        writer._raw_handle.flush()
        writer._raw_handle.close()
        writer._raw_handle = None
        assert writer._open_state_path is not None
        writer._open_state_path.unlink()

        stats = reconcile_audio_store(self.root / "data", self.ledger)

        self.assertEqual(1, stats["parts_recovered"])
        metadata_path = next((self.root / "data").rglob("*.json"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        relative_directory = metadata_path.parent.relative_to(self.root / "data").as_posix()
        self.assertEqual("3.0", metadata["schema_version"])
        self.assertEqual(relative_directory, metadata["storage_relative_directory"])
        self.assertEqual("recovered-from-path", metadata["robot_identity_source"])
        self.assertEqual("recovered-from-path", metadata["source_identity_source"])

    def test_reconcile_rebuilds_missing_ledger_record(self) -> None:
        metadata = self.make_writer().write_chunk(b"\x03\x00" * 8)
        assert metadata is not None
        self.ledger.close()
        (self.root / "state" / "segments.db").unlink()
        for suffix in ("-wal", "-shm"):
            Path(str(self.root / "state" / "segments.db") + suffix).unlink(missing_ok=True)
        self.ledger = SegmentLedger(self.root / "state" / "segments.db")

        stats = reconcile_audio_store(self.root / "data", self.ledger)

        self.assertEqual(1, stats["finalized_rebuilt"])
        self.assertIsNotNone(self.ledger.get(metadata["segment_id"]))

    def test_empty_part_is_preserved_as_corrupt_diagnostic(self) -> None:
        part = self.root / "data" / "audioinspector" / "mic-1" / "2026-01-01" / "00" / "123_000000.wav.part"
        part.parent.mkdir(parents=True)
        part.touch()

        stats = reconcile_audio_store(self.root / "data", self.ledger)

        self.assertEqual(1, stats["corrupt_parts"])
        self.assertTrue(Path(str(part) + ".corrupt").exists())
        self.assertTrue(Path(str(part) + ".corrupt.json").exists())

    def test_audio_write_pump_accepts_audiochunk_and_drains_on_stop(self) -> None:
        message = SimpleNamespace(
            format="audio/pcm-16k",
            data=list(b"\x04\x00" * 8),
            header=SimpleNamespace(stamp=SimpleNamespace(sec=12, nanosec=345)),
        )
        pump = AudioWritePump(self.make_writer(), queue_chunks=2)
        pump.start()

        self.assertTrue(pump.submit_message(message))
        pump.stop(timeout=2)

        self.assertEqual(12_000_000_345, source_stamp_ns(message))
        self.assertEqual(1, pump.stats()["received"])
        self.assertEqual(0, pump.stats()["dropped"])
        self.assertEqual(1, self.ledger.summary(instance_id="mic-1")["finalized_segments"])

    def test_audio_write_pump_rejects_wrong_format(self) -> None:
        message = SimpleNamespace(format="audio/opus", data=[1, 2], header=None)
        pump = AudioWritePump(self.make_writer(), queue_chunks=1)
        pump.start()

        self.assertFalse(pump.submit_message(message))
        pump.stop(timeout=2)

        self.assertEqual(1, pump.stats()["invalid_format"])
        self.assertFalse(list((self.root / "data").rglob("*.wav")))

    def test_desired_recording_state_round_trips_without_credentials(self) -> None:
        self.ledger.set_instance_state(
            card_id="audioinspector",
            instance_id="mic-1",
            input_topic="/robot/mic/audio",
            desired_state="recording",
            auto_resume=False,
            session_id="session-1",
            config={"segment_seconds": 60, "credential_profile": "default"},
        )

        desired = self.ledger.list_desired_recording(card_id="audioinspector")

        self.assertEqual(1, len(desired))
        self.assertFalse(desired[0]["auto_resume"])
        self.assertEqual(60, desired[0]["config"]["segment_seconds"])

    def test_reboot_defaults_to_idle_and_requires_explicit_resume(self) -> None:
        config = {
            "runtime_mode": "ros2",
            "data_root": str(self.root / "plugin-data"),
            "state_root": str(self.root / "plugin-state"),
        }
        plugin = AudioInspectorPlugin(config, executor=object())
        assert plugin._ledger is not None
        plugin._ledger.set_instance_state(
            card_id="audioinspector",
            instance_id="canvas-mic",
            input_topic="/robot/mic/audio",
            desired_state="recording",
            auto_resume=False,
            session_id="session-before-reboot",
            config={"auto_resume_after_reboot": False},
        )
        plugin.shutdown()

        restarted = AudioInspectorPlugin(config, executor=object())
        info = restarted.dispatch("audioinspector", {"action": "info", "instance_id": "canvas-mic"})

        assert info is not None
        self.assertEqual("idle", info["state"])
        self.assertFalse(info["recording"])
        self.assertTrue(info["resume_required"])
        restarted.shutdown()


if __name__ == "__main__":
    unittest.main()
