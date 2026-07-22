from __future__ import annotations

import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock


INSPECTION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INSPECTION_ROOT))

from storage.atomic_writer import AudioSegmentWriter  # noqa: E402
from storage.cos_backend import COSCredentials, COSUploadCoordinator, TencentCOSBackend, load_cos_credentials  # noqa: E402
from storage.ledger import SegmentLedger  # noqa: E402
from storage.models import SegmentRecord, SegmentState  # noqa: E402
from storage.retention import RetentionSweeper  # noqa: E402


class FakeCOSBackend:
    def __init__(self) -> None:
        self.objects: dict[str, dict] = {}
        self.upload_calls: list[str] = []

    def head(self, *, bucket: str, key: str):
        return self.objects.get(f"{bucket}/{key}")

    def upload_file(self, *, bucket: str, key: str, path: Path, sha256: str) -> None:
        body = path.read_bytes()
        self.upload_calls.append(key)
        self.objects[f"{bucket}/{key}"] = {"size": len(body), "sha256": sha256, "body": body}

    def put_bytes(self, *, bucket: str, key: str, body: bytes, sha256: str) -> None:
        self.upload_calls.append(key)
        self.objects[f"{bucket}/{key}"] = {"size": len(body), "sha256": sha256, "body": body}


class ForbiddenError(RuntimeError):
    def get_status_code(self):
        return 403


class FailingCOSBackend(FakeCOSBackend):
    def upload_file(self, *, bucket: str, key: str, path: Path, sha256: str) -> None:
        raise ForbiddenError("403 forbidden")


class COSAndRetentionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.data_root = self.root / "data"
        self.ledger = SegmentLedger(self.root / "state" / "ledger.sqlite3")

    def tearDown(self) -> None:
        self.ledger.close()
        self.tempdir.cleanup()

    def make_segment(self, *, instance_id: str = "mic-1") -> dict:
        writer = AudioSegmentWriter(
            data_root=self.data_root,
            ledger=self.ledger,
            card_id="audioinspector",
            instance_id=instance_id,
            input_topic="/robot/mic/audio",
            session_id="session-1",
            device_id="g1-sh",
            segment_seconds=1,
            sample_rate=8,
            channels=1,
            sample_width=2,
        )
        metadata = writer.write_chunk(b"\x01\x00" * 8)
        assert metadata is not None
        return metadata

    def make_uploader(self, backend: FakeCOSBackend) -> COSUploadCoordinator:
        return COSUploadCoordinator(
            ledger=self.ledger,
            data_root=self.data_root,
            card_id="audioinspector",
            config={
                "cos_bucket": "inspection-1250000000",
                "cos_prefix": "inspection-data",
                "device_id": "sh-g1",
                "upload_concurrency": 1,
            },
            backend=backend,
        )

    def test_segment_and_metadata_upload_then_head_verify(self) -> None:
        metadata = self.make_segment()
        backend = FakeCOSBackend()
        uploader = self.make_uploader(backend)

        self.assertTrue(uploader.run_once())

        record = self.ledger.get(metadata["segment_id"])
        assert record is not None
        self.assertEqual(SegmentState.UPLOADED_VERIFIED.value, record["state"])
        self.assertEqual(2, len(backend.upload_calls))
        self.assertTrue(record["object_key"].endswith(".wav"))
        self.assertIn("/sh-g1/audioinspector/mic-1/", "/" + record["object_key"])
        media_path = Path(record["local_path"])
        compact_date = media_path.parent.parent.name.replace("-", "")
        expected_key = "/".join((
            "inspection-data", "sh-g1", "audioinspector", "mic-1",
            compact_date, media_path.parent.name, media_path.name,
        ))
        self.assertEqual(expected_key, record["object_key"])

        self.ledger.transition(metadata["segment_id"], SegmentState.FINALIZED)
        self.assertTrue(uploader.run_once())
        self.assertEqual(2, len(backend.upload_calls), "verified immutable objects must not be uploaded twice")

    def test_existing_mismatched_object_becomes_conflict(self) -> None:
        metadata = self.make_segment()
        backend = FakeCOSBackend()
        uploader = self.make_uploader(backend)
        record = self.ledger.get(metadata["segment_id"])
        assert record is not None
        media_path = Path(record["local_path"])
        key = uploader._object_key(media_path)
        backend.objects[f"{uploader.bucket}/{key}"] = {"size": 1, "sha256": "wrong"}

        uploader.run_once()

        conflicted = self.ledger.get(metadata["segment_id"])
        assert conflicted is not None
        self.assertEqual(SegmentState.CONFLICT.value, conflicted["state"])
        self.assertEqual([], backend.upload_calls)

    def test_failed_upload_uses_persistent_auth_backoff(self) -> None:
        metadata = self.make_segment()
        uploader = self.make_uploader(FailingCOSBackend())
        uploader.config["retry_max_seconds"] = 30

        self.assertTrue(uploader.run_once())

        record = self.ledger.get(metadata["segment_id"])
        assert record is not None
        self.assertEqual(SegmentState.FINALIZED.value, record["state"])
        self.assertEqual(1, record["attempts"])
        self.assertGreater(record["next_retry_at_ns"], time.time_ns() + 20_000_000_000)
        self.assertEqual(30.0, uploader.last_retry_delay_seconds)
        self.assertFalse(uploader.run_once(), "a second worker must not bypass the persisted retry window")

    def test_testupload_writes_and_verifies_health_object(self) -> None:
        backend = FakeCOSBackend()
        result = self.make_uploader(backend).test_upload()

        self.assertTrue(result["verified"])
        self.assertIn("/_health/", "/" + result["object_key"])
        self.assertEqual(1, len(backend.upload_calls))

    def test_tencent_sdk_receives_full_custom_metadata_header(self) -> None:
        calls = []
        fake_module = types.ModuleType("qcloud_cos")

        class FakeConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeClient:
            def __init__(self, _config):
                pass

            def put_object(self, **kwargs):
                calls.append(kwargs)

        fake_module.CosConfig = FakeConfig
        fake_module.CosS3Client = FakeClient
        path = self.root / "small.bin"
        path.write_bytes(b"payload")
        with mock.patch.dict(sys.modules, {"qcloud_cos": fake_module}):
            backend = TencentCOSBackend(
                region="ap-beijing",
                credentials=COSCredentials("id", "key"),
                multipart_threshold_mb=64,
                upload_concurrency=1,
            )
            backend.upload_file(bucket="bucket-1", key="small.bin", path=path, sha256="abc123")

        self.assertEqual({"x-cos-meta-sha256": "abc123"}, calls[0]["Metadata"])

    def test_credential_profile_cannot_escape_secret_root(self) -> None:
        with self.assertRaisesRegex(ValueError, "path characters"):
            load_cos_credentials("../../etc/passwd", secret_root=self.root)

    def test_upload_recovery_is_scoped_to_one_card(self) -> None:
        audio = self.make_segment()
        audio_id = audio["segment_id"]
        self.ledger.transition(audio_id, SegmentState.UPLOADING)
        video_record = self.ledger.get(audio_id)
        assert video_record is not None
        video_id = "seg-video"
        self.ledger.upsert_finalized(SegmentRecord(
            segment_id=video_id,
            kind="video",
            card_id="videoinspector",
            instance_id="camera-1",
            local_path=video_record["local_path"],
            metadata_path=video_record["metadata_path"],
            size=video_record["size"],
            sha256=video_record["sha256"],
            created_at_ns=video_record["created_at_ns"],
        ))
        self.ledger.transition(video_id, SegmentState.UPLOADING)

        reset = self.ledger.reset_uploading_for_recovery(card_id="audioinspector")

        self.assertEqual(1, reset)
        self.assertEqual(SegmentState.FINALIZED.value, self.ledger.get(audio_id)["state"])
        self.assertEqual(SegmentState.UPLOADING.value, self.ledger.get(video_id)["state"])

    def test_retention_deletes_only_uploaded_verified_expired_pair(self) -> None:
        metadata = self.make_segment()
        record = self.ledger.get(metadata["segment_id"])
        assert record is not None
        self.ledger.mark_upload_verified(metadata["segment_id"], object_key="verified/key.wav")
        self.ledger.set_instance_state(
            card_id="audioinspector",
            instance_id="mic-1",
            input_topic="/robot/mic/audio",
            desired_state="idle",
            auto_resume=False,
            session_id="session-1",
            config={"local_retention_hours": 1, "local_max_gb": 4},
        )
        sweeper = RetentionSweeper(
            ledger=self.ledger,
            card_id="audioinspector",
            data_root=self.data_root,
        )

        stats = sweeper.sweep_once(now_ns=int(record["created_at_ns"]) + 2 * 3600 * 1_000_000_000)

        self.assertEqual(1, stats["purged"])
        purged = self.ledger.get(metadata["segment_id"])
        assert purged is not None
        self.assertEqual(SegmentState.PURGED_LOCAL.value, purged["state"])
        self.assertFalse(Path(record["local_path"]).exists())
        self.assertFalse(Path(record["metadata_path"]).exists())
        self.assertFalse(Path(record["local_path"]).parent.exists())
        self.assertGreaterEqual(stats["empty_dirs_pruned"], 1)

    def test_retention_expires_corrupt_group_and_keeps_recent_diagnostic(self) -> None:
        instance_root = self.data_root / "videoinspector" / "camera-1" / "2026-07-21"
        expired_dir = instance_root / "08"
        recent_dir = instance_root / "09"
        expired_dir.mkdir(parents=True)
        recent_dir.mkdir(parents=True)
        expired = expired_dir / "100_000000.mp4.part.corrupt"
        expired_reason = Path(str(expired) + ".json")
        expired_open = expired_dir / "100_000000.mp4.open.json"
        recent = recent_dir / "200_000000.mp4.part.corrupt"
        for path in (expired, expired_reason, expired_open, recent):
            path.write_text("diagnostic", encoding="utf-8")
        now_ns = time.time_ns()
        expired_ns = now_ns - 2 * 3600 * 1_000_000_000
        for path in (expired, expired_reason, expired_open):
            os.utime(path, ns=(expired_ns, expired_ns))
        self.ledger.set_instance_state(
            card_id="videoinspector",
            instance_id="camera-1",
            input_topic="/robot/camera/image",
            desired_state="idle",
            auto_resume=False,
            session_id="session-1",
            config={"corrupt_retention_hours": 1, "local_retention_hours": 6, "local_max_gb": 20},
        )
        sweeper = RetentionSweeper(
            ledger=self.ledger,
            card_id="videoinspector",
            data_root=self.data_root,
        )

        stats = sweeper.sweep_once(now_ns=now_ns)

        self.assertEqual(3, stats["corrupt_files_purged"])
        self.assertFalse(expired.exists())
        self.assertFalse(expired_reason.exists())
        self.assertFalse(expired_open.exists())
        self.assertFalse(expired_dir.exists())
        self.assertTrue(recent.exists())
        self.assertEqual(3, sweeper.stats()["corrupt_files_purged"])

    def test_retention_never_deletes_unuploaded_at_disk_pressure(self) -> None:
        metadata = self.make_segment()
        record = self.ledger.get(metadata["segment_id"])
        assert record is not None
        self.ledger.set_instance_state(
            card_id="audioinspector",
            instance_id="mic-1",
            input_topic="/robot/mic/audio",
            desired_state="recording",
            auto_resume=False,
            session_id="session-1",
            config={"local_retention_hours": 1, "local_max_gb": 0.000000001},
        )
        critical = []
        sweeper = RetentionSweeper(
            ledger=self.ledger,
            card_id="audioinspector",
            on_critical=lambda instance, used, limit: critical.append((instance, used, limit)),
        )

        stats = sweeper.sweep_once(now_ns=time.time_ns() + 10 * 3600 * 1_000_000_000)

        self.assertEqual(0, stats["purged"])
        self.assertEqual(1, stats["critical"])
        self.assertTrue(critical)
        self.assertTrue(Path(record["local_path"]).exists())
        self.assertEqual(SegmentState.FINALIZED.value, self.ledger.get(metadata["segment_id"])["state"])

    def test_local_ring_expiry_can_delete_unuploaded_segment(self) -> None:
        metadata = self.make_segment(instance_id="local-ring")
        record = self.ledger.get(metadata["segment_id"])
        assert record is not None
        self.ledger.set_instance_state(
            card_id="audioinspector",
            instance_id="local-ring",
            input_topic="/robot/mic/audio",
            desired_state="idle",
            auto_resume=False,
            session_id="session-local-ring",
            config={
                "storage_mode": "local_ring",
                "local_retention_hours": 1,
                "local_max_gb": 4,
            },
        )
        sweeper = RetentionSweeper(
            ledger=self.ledger,
            card_id="audioinspector",
            data_root=self.data_root,
        )

        stats = sweeper.sweep_once(
            now_ns=int(record["created_at_ns"]) + 2 * 3600 * 1_000_000_000,
        )

        self.assertEqual(1, stats["purged"])
        self.assertEqual(SegmentState.PURGED_LOCAL.value, self.ledger.get(metadata["segment_id"])["state"])
        self.assertFalse(Path(record["local_path"]).exists())


if __name__ == "__main__":
    unittest.main()
