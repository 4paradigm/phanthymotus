"""Real SQLite persistence across independent Agent Core Python processes."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CHANNEL = "qq_official_1906322115"


def run_isolated(db: Path, script: str):
    env = dict(os.environ)
    env["DB_PATH"] = str(db)
    env["PYTHONPATH"] = str(SRC)
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                            env=env, capture_output=True, text=True,
                            timeout=35)
    assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-1500:]
    return result.stdout


def test_persistence_and_private_group_separation_across_processes(tmp_path):
    db = tmp_path / "isolated_memory.db"
    creator = """
from channel.manager import ChannelManager
m = ChannelManager()
assert m.remember_chat_memory('qq_official_1906322115', 'p2p:SHAREDID', 'PRIVATE_A_MEMORY')
assert m.remember_chat_memory('qq_official_1906322115', 'group:SHAREDID', 'GROUP_ONLY_MEMORY')
assert m.remember_chat_memory('qq_official_1906322115', 'p2p:OTHERID', 'PRIVATE_B_MEMORY')
"""
    run_isolated(db, creator)
    assert db.exists()
    reader = """
from channel.manager import ChannelManager
m = ChannelManager()
assert m.get_chat_memory('qq_official_1906322115','p2p:SHAREDID') == ['PRIVATE_A_MEMORY']
assert m.get_chat_memory('qq_official_1906322115','group:SHAREDID') == ['GROUP_ONLY_MEMORY']
assert m.get_chat_memory('qq_official_1906322115','p2p:OTHERID') == ['PRIVATE_B_MEMORY']
assert m.get_chat_memory('qq_official_another','p2p:SHAREDID') == []
assert m.forget_chat_memory('qq_official_1906322115','group:SHAREDID') == 1
"""
    run_isolated(db, reader)
    third = """
from channel.manager import ChannelManager
m = ChannelManager()
assert m.get_chat_memory('qq_official_1906322115','group:SHAREDID') == []
assert m.get_chat_memory('qq_official_1906322115','p2p:SHAREDID') == ['PRIVATE_A_MEMORY']
"""
    run_isolated(db, third)
