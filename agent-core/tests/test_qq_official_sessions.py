"""QQ Gateway session persistence and file permissions."""
from pathlib import Path
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from channel.adapters.qq_official_sessions import QQSessionState


class FakeStore:
    def __init__(self, base_dir, filename):
        self.path = Path(base_dir) / filename
    def get(self, app_id):
        if self.path.exists():
            data = json.loads(self.path.read_text())
            return types.SimpleNamespace(
                is_resumable=True, is_fresh=lambda: True,
                session_id=data["session"], seq=data["seq"])
        return types.SimpleNamespace(
            is_resumable=False, is_fresh=lambda: False, session_id="", seq=None)
    def save(self, app_id, session, seq):
        self.path.write_text(json.dumps({"session": session, "seq": seq}))
    def clear(self, app_id):
        self.path.unlink(missing_ok=True)


class QQSessionTests(unittest.TestCase):
    def test_persist_restore_and_clear(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(sys.modules, {
                "qqbot_agent_sdk": types.SimpleNamespace(WSSessionStore=FakeStore)
            }):
                store = QQSessionState("1900000000", directory)
                self.assertEqual(store.read(), (None, None))
                store.save("opaque-session", 32)
                recovered = QQSessionState("1900000000", directory)
                self.assertEqual(recovered.read(), ("opaque-session", 32))
                if os.name != "nt":
                    mode = (Path(directory) / store.filename).stat().st_mode & 0o777
                    self.assertEqual(mode, 0o600)
                recovered.clear()
                self.assertEqual(recovered.read(), (None, None))


if __name__ == "__main__":
    unittest.main()
