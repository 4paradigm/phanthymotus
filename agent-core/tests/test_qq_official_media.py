"""QQ official media contract tests; no external network or QQ credentials."""
import asyncio
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from channel.adapter import Attachment, KIND_IMAGE
from channel.adapters.qq_official_media import (
    _allowed_qq_url, receive_media, send_media,
)


class QQMediaTests(unittest.IsolatedAsyncioTestCase):
    def test_cdn_allowlist_rejects_redirect_hosts_and_private_networks(self):
        self.assertTrue(_allowed_qq_url("https://multimedia.nt.qq.com.cn/sample"))
        self.assertTrue(_allowed_qq_url("https://gchat.qpic.cn/sample"))
        for url in (
            "http://qq.com/file",
            "https://qq.com.evil.example/file",
            "https://127.0.0.1/file",
            "https://user@qq.com/file",
            "https://localhost/file",
            "file:///etc/passwd",
            "https://qq.com:1234/file",
        ):
            self.assertFalse(_allowed_qq_url(url), url)

    async def test_media_diagnostics_never_print_signed_urls(self):
        import contextlib
        import io
        from channel.adapters.qq_official_media import _MEDIA_FAILURES
        _MEDIA_FAILURES.clear()
        signature = "unique_signed_query_value"
        att = types.SimpleNamespace(
            url="https://unsafe.example.com/image.png?sig=" + signature,
            content_type="image/png", filename="private.png")
        event = types.SimpleNamespace(attachments=[att], msg_elements=[])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = await receive_media(None, event, "/tmp/qq-invalid-url-test")
        self.assertEqual(result, [])
        self.assertIn("untrusted_or_missing_cdn_url", output.getvalue())
        self.assertNotIn(signature, output.getvalue())

    async def test_untrusted_attachment_is_not_downloaded(self):
        att = types.SimpleNamespace(
            url="http://127.0.0.1/secret", content_type="image/png",
            filename="secret.png")
        event = types.SimpleNamespace(attachments=[att], msg_elements=[])
        client = mock.MagicMock()
        files = await receive_media(client, event, "/tmp/qq-test-never-created")
        self.assertEqual(files, [])
        client.stream.assert_not_called()

    async def test_media_uploads_use_correct_sdk_route(self):
        calls = []
        class FakeUploader:
            def __init__(self, api_client, http_client):
                self.client = api_client
            async def upload(self, **kw):
                calls.append(kw)
                return "file-token"
        class FakeMessage:
            def __init__(self, **kw):
                self.fields = kw
        class FakeApi:
            def next_msg_seq(self):
                return 42
            async def post_c2c_message(self, chat, dto):
                calls.append(("c2c", chat, dto.fields))
            async def post_group_message(self, chat, dto):
                calls.append(("group", chat, dto.fields))
        sdk = types.SimpleNamespace(
            MediaUploader=FakeUploader,
            MediaInfo=lambda **kw: types.SimpleNamespace(**kw),
            MessageToCreate=FakeMessage,
            QQMessageType=types.SimpleNamespace(RICH_MEDIA=7),
            constants=types.SimpleNamespace(MEDIA_TYPE_IMAGE=1),
        )
        a = Attachment(kind=KIND_IMAGE, path="/work/resource/qq.png", name="qq.png")
        with mock.patch.dict(sys.modules, {"qqbot_agent_sdk": sdk}):
            with mock.patch("channel.adapters.qq_official_media._validate_outbound_file"):
                for scope in ("c2c", "group"):
                    await send_media(FakeApi(), None, scope, "Target12345", "source-01", a)
        self.assertEqual([c["chat_type"] for c in calls if isinstance(c, dict)],
                         ["c2c", "group"])
        self.assertEqual([c[0] for c in calls if isinstance(c, tuple)], ["c2c", "group"])
        self.assertEqual([c[2]["msg_id"] for c in calls if isinstance(c, tuple)],
                         ["source-01", "source-01"])


    async def test_normal_core_generated_file_is_uploadable_without_qq_outbox(self):
        """Use the same real file resolver as channel_reply, then send via QQ SDK."""
        import tempfile
        import event.desktop as desktop
        from channel.media import resolve_outbound
        from channel.adapters.qq_official_media import _validate_outbound_file
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generated = root / "generated_result.png"
            generated.write_bytes(b"generated-image-bytes")
            calls = []
            class FakeUploader:
                def __init__(self, api_client, http_client):
                    pass
                async def upload(self, **kwargs):
                    calls.append(kwargs)
                    return "upload-token"
            class FakeMessage:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs
            class FakeApi:
                def next_msg_seq(self):
                    return 1
                async def post_c2c_message(self, chat, dto):
                    calls.append((chat, dto.kwargs))
            sdk = types.SimpleNamespace(
                MediaUploader=FakeUploader, MessageToCreate=FakeMessage,
                MediaInfo=lambda **kw: types.SimpleNamespace(**kw),
                QQMessageType=types.SimpleNamespace(RICH_MEDIA=7),
                constants=types.SimpleNamespace(MEDIA_TYPE_IMAGE=1),
            )
            # Production uses /work and /tmp; patch the shared allowlist to
            # this temporary output directory for a platform-neutral test.
            # The production whitelist is POSIX (/work, /tmp). Its current
            # helper compares "/" separators, so Windows CI uses the exact
            # temporary file as its allowlist entry (Linux tests the folder).
            import os
            allowed_paths = ([str(root.resolve())] if os.name != "nt"
                             else [str(generated.resolve())])
            with (mock.patch.object(desktop, "_ALLOWED_DIRS", allowed_paths),
                  mock.patch.dict(sys.modules, {"qqbot_agent_sdk": sdk})):
                files, errors = resolve_outbound([str(generated)], "qq")
                self.assertEqual(errors, [])
                self.assertEqual(len(files), 1)
                _validate_outbound_file(files[0])
                await send_media(FakeApi(), None, "c2c", "UserOpenID123",
                                 "original-message", files[0])
                self.assertEqual(calls[0]["source"], str(generated.resolve()))
                self.assertEqual(calls[1][0], "UserOpenID123")

    def test_qq_outbound_denies_unapproved_paths_and_managed_credentials(self):
        import tempfile
        import event.desktop as desktop
        from channel.adapters.qq_official_media import _validate_outbound_file
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            allowed = root / "allowed"
            forbidden = root / "forbidden"
            allowed.mkdir()
            forbidden.mkdir()
            ok = allowed / "generated.pdf"
            secret = allowed / "private.secret"
            outside = forbidden / "private.pdf"
            for f in (ok, secret, outside):
                f.write_bytes(b"private" * 5)
            import os
            allowed_paths = ([str(allowed.resolve())] if os.name != "nt"
                             else [str(ok.resolve()), str(secret.resolve())])
            with (mock.patch.object(desktop, "_ALLOWED_DIRS", allowed_paths),
                  mock.patch("channel.qq_credential_store.directory", return_value=allowed)):
                with self.assertRaisesRegex(ValueError, "credential"):
                    _validate_outbound_file(Attachment(kind="file", path=str(secret)))
                with self.assertRaisesRegex(ValueError, "allowed Core output"):
                    _validate_outbound_file(Attachment(kind="file", path=str(outside)))
            with mock.patch.object(desktop, "_ALLOWED_DIRS", allowed_paths):
                _validate_outbound_file(Attachment(kind="file", path=str(ok)))

    async def test_guild_media_fails_closed(self):
        a = Attachment(kind=KIND_IMAGE, path="/work/resource/qq.png", name="qq.png")
        with self.assertRaisesRegex(ValueError, "C2C/group"):
            await send_media(object(), None, "guild", "Channel12345", "message", a)


if __name__ == "__main__":
    unittest.main()
