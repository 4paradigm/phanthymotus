"""Read-only QQ upload preview: cross-channel and privilege regression."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from channel import store
from channel.adapter import Attachment, InboundMessage
from channel.attachment_preview import preview_message_files


def sample_sheet():
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    return (f'<worksheet xmlns="{ns}"><sheetData>'
            '<row r="1"><c r="A1" t="s"><v>0</v></c>'
            '<c r="B1" t="inlineStr"><is><t>20250000001</t></is></c></row>'
            '<row r="2"><c r="A2" t="s"><v>1</v></c>'
            '<c r="B2"><v>91</v></c></row>'
            '</sheetData></worksheet>').encode()


class QQAttachmentPreviewTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "channel_files"
        self.store_patch = mock.patch.object(store, "_ROOT", self.root)
        self.store_patch.start()
        self.addCleanup(self.store_patch.stop)
        self.own = self.root / "qq_test" / "20261113"
        self.own.mkdir(parents=True, exist_ok=True)

    def _att(self, path):
        return Attachment(kind="file", path=str(path),
                          name=path.name, mime="application/octet-stream", size=path.stat().st_size)

    def _xlsx(self):
        path = self.own / "grade.xlsx"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("xl/sharedStrings.xml",
                       '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                       '<si><t>姓名</t></si><si><t>示例学生</t></si></sst>')
            z.writestr("xl/worksheets/sheet1.xml", sample_sheet())
        return path

    async def test_xlsx_table_is_readable_without_desktop_access(self):
        path = self._xlsx()
        result = preview_message_files("qq_test", [self._att(path)])
        self.assertIn("A1=姓名", result)
        self.assertIn("B1=20250000001", result)
        self.assertIn("A2=示例学生", result)
        self.assertIn("B2=91", result)
        self.assertIn("只读解析结果", result)

    async def test_utf8_csv_and_gb18030_text(self):
        csv_path = self.own / "students.csv"
        csv_path.write_bytes("姓名,学号\n小王,1001\n".encode("gb18030"))
        result = preview_message_files("qq_test", [self._att(csv_path)])
        self.assertIn("小王", result)
        self.assertIn("1001", result)

    async def test_other_channel_and_symlink_are_blocked(self):
        foreign = self.root / "qq_other" / "20261113"
        foreign.mkdir(parents=True)
        path = foreign / "grades.csv"
        path.write_text("PRIVATE_GRADE", encoding="utf-8")
        self.assertEqual(preview_message_files("qq_test", [self._att(path)]), "")
        link = self.own / "foreign.csv"
        link.symlink_to(path)
        self.assertEqual(preview_message_files("qq_test", [self._att(link)]), "")

    async def test_zip_bomb_and_unsupported_file_do_not_expose_content(self):
        bomb = self.own / "large.xlsx"
        with zipfile.ZipFile(bomb, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("xl/worksheets/sheet1.xml", b"x" * (17 * 1024 * 1024))
        answer = preview_message_files("qq_test", [self._att(bomb)])
        self.assertIn("无法解析", answer)
        exe = self.own / "script.py"
        exe.write_text("secret", encoding="utf-8")
        self.assertEqual(preview_message_files("qq_test", [self._att(exe)]), "")

    async def test_document_word_payload(self):
        path = self.own / "notes.docx"
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("word/document.xml",
                       '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                       '<w:body><w:p><w:r><w:t>会议记录</w:t></w:r></w:p></w:body></w:document>')
        self.assertIn("会议记录", preview_message_files("qq_test", [self._att(path)]))

    async def test_manager_adds_only_current_upload_to_viewer_turn(self):
        from channel.manager import ChannelManager
        msg_file = self._xlsx()
        manager = ChannelManager()
        cfg = {
            "canvas_layout": {"cards": [
                {"id": "in", "mcpId": "channel", "toolName": "channel_request"}]},
            "tool_config:channel:channel_request:in": {"channel_id": "qq_test"},
            "channel_configs": [{"id": "qq_test", "platform": "qq",
                                 "enabled": True, "config": {"protocol": "official"}}],
        }
        message = InboundMessage(
            platform="qq", channel_id="qq_test",
            user_id="official:1900000000:owner-open-id", chat_id="owner-open-id",
            display_name="test-user", text="这个表格有哪些学生？",
            message_id="msg1", chat_type="p2p",
            attachments=[self._att(msg_file)],
        )
        with (mock.patch("channel.manager.config.main", cfg),
              mock.patch("channel.acl.get_user", return_value={
                  "role": "viewer", "tool_filter": "*"}),
              mock.patch("channel.acl.check_permission", return_value=(True, "ok")),
              mock.patch("api.inspection.publish_to_topic",
                         new_callable=mock.AsyncMock) as published,
              mock.patch("api.motus_stream.push_event",
                         new_callable=mock.AsyncMock)):
            await manager._on_inbound_message(message)
        published.assert_awaited_once()
        payload = json.loads(published.await_args.args[1])
        self.assertIn("B1=20250000001", payload["text"])
        self.assertEqual(payload["user_role"], "viewer")
        self.assertEqual(payload["files"][0]["path"], str(msg_file))


if __name__ == "__main__":
    unittest.main()
