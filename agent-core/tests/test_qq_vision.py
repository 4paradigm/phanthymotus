"""QQ image handoff into the existing multimodal decision-core request."""
import asyncio
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from channel.qq_vision import (
    _approved_image_paths, add_qq_images, redact_inline_images,
)


def _trigger(path, *, qq=True, kind="image"):
    return {
        "_qq_channel_turn": qq,
        "_qq_channel_id": "qq_test",
        "_qq_image_files": [{"kind": kind, "path": str(path)}],
    }


class QQVisionTests(unittest.IsolatedAsyncioTestCase):
    async def test_qq_image_is_multimodal_for_existing_llm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "photo.png"
            image.write_bytes(b"\x89PNG\r\n\x1a\nvalid-test-bytes")
            messages = [{"role": "system", "content": "system"},
                        {"role": "user", "content": "图片里面是什么"}]
            reader = mock.AsyncMock(return_value=[
                {"type": "text", "text": "file info"},
                {"type": "image_url", "image_url": "data:image/png;base64,dGVzdA=="},
            ])
            attached = await add_qq_images(
                messages, _trigger(image), media_root=root, reader=reader,
                vision_enabled=True)
            self.assertTrue(attached)
            self.assertEqual(messages[-1]["content"][0]["text"], "图片里面是什么")
            self.assertEqual(messages[-1]["content"][-1]["type"], "image_url")
            self.assertTrue(messages[-1]["content"][-1]["image_url"]["url"].startswith("data:"))
            reader.assert_awaited_once_with(image)

    async def test_channel_event_collector_to_multimodal_prompt(self):
        from collector import _build_trigger
        import time
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "sent_from_qq.png"
            image.write_bytes(b"\\x89PNG\\r\\n\\x1a\\nqq")
            payload = {
                "platform": "qq",
                "channel_id": "qq_test",
                "chat_id": "GroupOpenID12345",
                "message_id": "msg-01",
                "user_id": "member-01",
                "sender_type": "user",
                "user_role": "viewer",
                "text": "图中有什么",
                "files": [{"kind": "image", "path": str(image), "name": "qq.png"}],
            }
            event = {
                "source": "/channel/request/qq_test",
                "text": json.dumps(payload, ensure_ascii=False),
                "ts": time.time(),
            }
            with mock.patch("channel.manager.manager.get_chat_history",
                            return_value=[]):
                trigger = _build_trigger([event], urgent=True)
            self.assertTrue(trigger["_qq_channel_turn"])
            self.assertEqual(trigger["_qq_image_files"][0]["path"], str(image))
            messages = [{"role": "user", "content": trigger["text"]}]
            reader = mock.AsyncMock(return_value=[
                {"type": "image_url", "image_url": "data:image/png;base64,dGVzdA=="}
            ])
            sent = await add_qq_images(messages, trigger, media_root=root,
                                       reader=reader, vision_enabled=True)
            self.assertTrue(sent)
            self.assertEqual(messages[-1]["content"][-1]["type"], "image_url")


    async def test_real_collector_trigger_uses_authenticated_channel_for_production_storage(self):
        """No hand-crafted payload.channel_ids: exercise the actual Collector output."""
        from collector import _build_trigger
        import channel.qq_vision as vision
        import time
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            qqdir = root / "qq_test" / "20261114"
            fsdir = root / "feishu_test" / "20261114"
            qqdir.mkdir(parents=True)
            fsdir.mkdir(parents=True)
            image = qqdir / "from_qq.png"
            other = fsdir / "private_feishu.png"
            image.write_bytes(b"simulated-png")
            other.write_bytes(b"private-other-channel")
            payload = {
                "platform": "qq", "channel_id": "qq_test",
                "chat_id": "OpenIDChat123", "chat_type": "p2p",
                "user_id": "official:1900000000:OpenIDUser1234567",
                "message_id": "message-1", "sender_type": "user",
                "user_role": "viewer", "text": "What is in this photo?",
                "files": [
                    {"kind": "image", "path": str(image), "name": "from_qq.png"},
                    {"kind": "image", "path": str(other), "name": "private_feishu.png"},
                ],
            }
            event = {
                "source": "/channel/request/qq_test",
                "text": json.dumps(payload), "ts": time.time(),
            }
            with mock.patch("channel.manager.manager.get_chat_history", return_value=[]):
                trigger = _build_trigger([event], urgent=True)
            self.assertTrue(trigger["_qq_channel_turn"])
            self.assertEqual(trigger["_qq_channel_id"], "qq_test")
            # Force the production shared-storage branch rather than the
            # custom-root test shortcut used by older tests.
            reader = mock.AsyncMock(return_value=[
                {"type": "image_url", "image_url": "data:image/png;base64,dGVzdA=="}
            ])
            messages = [{"role": "user", "content": trigger["text"]}]
            with mock.patch.object(vision, "_QQ_MEDIA_ROOT", root):
                self.assertEqual(vision._approved_image_paths(trigger, root), [image])
                self.assertTrue(await vision.add_qq_images(
                    messages, trigger, media_root=root, reader=reader,
                    vision_enabled=True))
                reader.assert_awaited_once_with(image)
                self.assertEqual(messages[0]["content"][-1]["type"], "image_url")
                # This field is optional and must not govern file identity.
                trigger.get("payload", {}).pop("channel_ids", None)
                self.assertEqual(vision._approved_image_paths(trigger, root), [image])
                trigger["_qq_channel_id"] = "not_this_channel"
                self.assertEqual(vision._approved_image_paths(trigger, root), [])

    async def test_qq_image_not_inline_when_vision_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "photo.jpg"
            image.write_bytes(b"x" * 50)
            messages = [{"role": "user", "content": "看图"}]
            reader = mock.AsyncMock()
            ok = await add_qq_images(
                messages, _trigger(image), media_root=root, reader=reader,
                vision_enabled=False)
            self.assertFalse(ok)
            self.assertIsInstance(messages[0]["content"], str)
            self.assertIn("vision input is disabled", messages[0]["content"])
            reader.assert_not_called()

    async def test_image_path_escape_and_symlink_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "qq"
            root.mkdir()
            outside = Path(tmp) / "private.png"
            outside.write_bytes(b"secret")
            link = root / "alias.png"
            link.symlink_to(outside)
            self.assertEqual(_approved_image_paths(_trigger(outside), root), [])
            self.assertEqual(_approved_image_paths(_trigger(link), root), [])
            self.assertEqual(_approved_image_paths(_trigger(outside, qq=False), root), [])

    async def test_shared_feishu_storage_is_scoped_to_one_qq_channel(self):
        import channel.qq_vision as vision
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            qq = root / "qq_test" / "20261112"
            fs = root / "feishu_test" / "20261112"
            qq.mkdir(parents=True)
            fs.mkdir(parents=True)
            good = qq / "photo.png"
            other = fs / "photo.png"
            good.write_bytes(b"legit")
            other.write_bytes(b"not-this-channel")
            trigger = {
                "_qq_channel_turn": True,
                "_qq_channel_id": "qq_test",
                "_qq_image_files": [
                    {"kind": "image", "path": str(good)},
                    {"kind": "image", "path": str(other)},
                ],
            }
            with mock.patch.object(vision, "_QQ_MEDIA_ROOT", root):
                self.assertEqual(vision._approved_image_paths(trigger, root), [good])
                # A forged/optional payload value cannot change the authenticated ID.
                trigger["payload"] = {"channel_ids": ["feishu_test"]}
                self.assertEqual(vision._approved_image_paths(trigger, root), [good])
                trigger["_qq_channel_id"] = "other_qq"
                self.assertEqual(vision._approved_image_paths(trigger, root), [])
                trigger["_qq_channel_id"] = ""
                self.assertEqual(vision._approved_image_paths(trigger, root), [])

    async def test_oversize_or_absent_image_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / "missing.png"
            messages = [{"role": "user", "content": "这是什么"}]
            result = await add_qq_images(
                messages, _trigger(p), media_root=root, vision_enabled=True,
                reader=mock.AsyncMock())
            self.assertFalse(result)
            self.assertIn("unavailable or unreadable", messages[0]["content"])
            p.write_bytes(b"x")
            with p.open("r+b") as f:
                f.truncate(11 * 1024 * 1024)
            self.assertEqual(_approved_image_paths(_trigger(p), root), [])

    async def test_multi_image_limit_and_non_image_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = []
            for n in range(3):
                p = root / f"{n}.png"
                p.write_bytes(b"abc")
                files.append({"kind": "image", "path": str(p)})
            files.append({"kind": "file", "path": str(root / "document.txt")})
            trigger = {"_qq_channel_turn": True, "_qq_channel_id": "qq_test",
                       "_qq_image_files": files}
            self.assertEqual(len(_approved_image_paths(trigger, root)), 2)

    async def test_gif_is_transcoded_to_jpeg_frames_before_llm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gif = root / "animation.gif"
            gif.write_bytes(b"GIF89a" + b"x" * 64)
            proc = mock.Mock(returncode=0)
            proc.communicate = mock.AsyncMock(return_value=(
                b"\xff\xd8one\xff\xd9\xff\xd8two\xff\xd9", b""))
            with mock.patch("asyncio.create_subprocess_exec",
                            new_callable=mock.AsyncMock, return_value=proc):
                messages = [{"role":"user", "content":"这张 GIF 里是什么？"}]
                self.assertTrue(await add_qq_images(
                    messages, _trigger(gif), media_root=root, vision_enabled=True))
            parts = messages[0]["content"]
            images = [x for x in parts if x["type"] == "image_url"]
            self.assertEqual(len(images), 2)
            self.assertTrue(all(x["image_url"]["url"].startswith(
                "data:image/jpeg;base64,") for x in images))
            self.assertIn("抽取 2 帧", str(parts))

    async def test_followup_refers_only_to_same_qq_senders_recent_image(self):
        from channel.manager import ChannelManager
        from channel.adapter import Attachment
        from types import SimpleNamespace
        mgr = ChannelManager()
        attachment = Attachment(kind="image", path="/work/resource/channel_files/qq_test/a.png",
                                name="a.png", mime="image/png", size=42)
        msg = SimpleNamespace(platform="qq", sender_type="user", channel_id="qq_test",
                              chat_type="p2p", chat_id="chat_A", user_id="user_A",
                              text="[QQ attachment]", attachments=[attachment])
        self.assertEqual(mgr._qq_followup_attachments(msg), [])
        msg.attachments = []
        msg.text = "这个能看到吗"
        prior = mgr._qq_followup_attachments(msg)
        self.assertEqual(len(prior), 1)
        self.assertEqual(prior[0]["path"], attachment.path)
        msg.user_id = "user_B"
        self.assertEqual(mgr._qq_followup_attachments(msg), [])
        msg.user_id = "user_A"
        msg.chat_id = "chat_B"
        self.assertEqual(mgr._qq_followup_attachments(msg), [])
        msg.chat_id = "chat_A"
        msg.text = "打开机器人"
        self.assertEqual(mgr._qq_followup_attachments(msg), [])

    def test_request_log_redacts_image_bytes_but_not_live_prompt(self):
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "identify"},
            {"type": "image_url", "image_url": "data:image/png;base64,private"},
        ]}]
        safe = redact_inline_images(messages)
        self.assertNotIn("private", json.dumps(safe))
        self.assertIn("private", json.dumps(messages))
        self.assertEqual(safe[0]["content"][0]["text"], "identify")
