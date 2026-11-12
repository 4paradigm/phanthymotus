"""QQ SDK 1.2.2 integration contracts using its real DTOs and an in-memory HTTP transport.

These tests never access QQ, expose secrets, or depend on a running robot.
"""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

HAS_SDK = importlib.util.find_spec("qqbot_agent_sdk") is not None


@unittest.skipUnless(HAS_SDK, "QQ official SDK missing")
class QQSDKRuntimeContracts(unittest.IsolatedAsyncioTestCase):

    @staticmethod
    def _image_event(scope="group", *, content="请描述图片", msg_id="m" * 180):
        from qqbot_agent_sdk import EventParser
        raw = {
            "id": msg_id, "content": content,
            "author": {"member_openid": "MemberOpenID12345"},
            "attachments": [{
                "url": "https://gchat.qpic.cn/test.png",
                "content_type": "image/png", "filename": "photo.png",
            }],
            "msg_elements": [{"content": "", "attachments": []}],
        }
        if scope == "group":
            raw["group_openid"] = "GroupOpenID12345"
            kind = "GROUP_AT_MESSAGE_CREATE"
        else:
            raw["author"]["user_openid"] = "MemberOpenID12345"
            kind = "C2C_MESSAGE_CREATE"
        return EventParser.parse(kind, raw), raw

    def _adapter(self, enabled=True):
        from channel.adapters.qq_official import QQOfficialAdapter
        return QQOfficialAdapter("test_channel", "qq", {
            "app_id": "1900000000",
            "enabled_scopes": ["c2c", "group"],
            "private_only": False,
            "enable_inbound_media": enabled,
        }, lambda _: None)

    async def test_group_image_real_sdk_parse_to_disk(self):
        from channel.adapters.qq_official_media import receive_media
        event, raw = self._image_event()
        self.assertIsNotNone(event)
        self.assertEqual(event.chat_scope, "group")
        adapter = self._adapter()
        inbound = adapter._parse_message(event)
        self.assertIsNotNone(inbound)
        self.assertEqual(inbound.chat_type, "group")
        self.assertEqual(inbound.chat_id, raw["group_openid"])
        image = b"\x89PNG\r\n\x1a\n" + b"test-image"

        def handler(request):
            self.assertEqual(str(request.url), "https://gchat.qpic.cn/test.png")
            return httpx.Response(
                200, content=image,
                headers={"content-type": "image/png", "content-length": str(len(image))})

        with tempfile.TemporaryDirectory() as folder:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                files = await receive_media(client, event, folder)
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0].kind, "image")
            self.assertEqual(Path(files[0].path).read_bytes(), image)
            self.assertTrue(Path(files[0].path).is_relative_to(Path(folder)))
            if os.name != "nt":
                self.assertEqual(Path(files[0].path).stat().st_mode & 0o777, 0o600)

    async def test_disabled_media_keeps_message_but_does_not_fetch_image(self):
        event, _ = self._image_event()
        adapter = self._adapter(enabled=False)
        result = adapter._parse_message(event)
        self.assertIsNotNone(result)
        self.assertIn("attachment not enabled", result.text)
        self.assertEqual(result.attachments, [])

    async def test_media_download_rejects_oversize_and_redirect(self):
        from channel.adapters.qq_official_media import receive_media
        event, _ = self._image_event()
        with tempfile.TemporaryDirectory() as folder:
            for response in (
                httpx.Response(302, headers={"location": "https://gchat.qpic.cn/other"}),
                httpx.Response(200, content=b"x", headers={
                    "content-length": "10485761",
                }),
            ):
                async with httpx.AsyncClient(
                        transport=httpx.MockTransport(lambda _: response)) as client:
                    self.assertEqual(await receive_media(client, event, folder), [])
                self.assertFalse(list(Path(folder).rglob("*")))

    async def test_real_sdk_group_media_reply_dto(self):
        from qqbot_agent_sdk import MediaUploader
        from channel.adapter import Attachment
        from channel.adapters.qq_official_media import send_media
        sent = mock.AsyncMock()
        api = types.SimpleNamespace(next_msg_seq=lambda: 4, post_group_message=sent)
        file = Attachment(kind="image", path="/work/resource/qq_outbox/picture.png",
                          name="picture.png")
        with mock.patch(
                "channel.adapters.qq_official_media._validate_outbound_file"):
            with mock.patch.object(
                    MediaUploader, "upload", new_callable=mock.AsyncMock,
                    return_value="media-file-token") as upload:
                await send_media(api, None, "group", "GroupOpenID12345",
                                 "source-group-01", file)
        self.assertEqual(upload.await_args.kwargs["chat_type"], "group")
        dto = sent.await_args.args[1]
        self.assertEqual(dto.msg_id, "source-group-01")
        self.assertEqual(dto.media.file_info, "media-file-token")
        sent.assert_awaited_once()

    def test_at_event_and_unrecognized_full_message_are_isolated(self):
        from channel.adapters.qq_official import _full_group_mentions_bot
        self.assertFalse(_full_group_mentions_bot({"mentions": [{"is_you": False}]}))
        self.assertFalse(_full_group_mentions_bot({}))
        self.assertTrue(_full_group_mentions_bot({"mentions": [{"is_you": True}]}))


if __name__ == "__main__":
    unittest.main()
