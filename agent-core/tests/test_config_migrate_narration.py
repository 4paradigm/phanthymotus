"""
test_config_migrate_narration.py — event.llm 里主动播报相关键的迁移。

两件事，合在同一次 read-modify-write 里做：

1. **补种新键**。_seed_defaults 用的是整行粒度的 `INSERT OR IGNORE`，已部署机器上的
   'event' 行早就存在，新加的默认值永远进不去 —— 结果是阈值读成 0、功能静默不生效。
   这个坑 subagent 那几个键上真踩过（Orin5 实测读出来还是旧值）。
2. **删掉已废弃的键**。轮数维度取消后 narration_silence_rounds 没有任何读者，留在库里
   只会让人对着设置页猜"它还管不管用"。

迁移跑在模块导入期（config.py 末尾的 _migrate()），所以这里不 import config，而是直接
对一个临时 DB 跑同一段逻辑 —— 否则会碰到已经被别的测试模块导入过的那个单例。

Run: cd agent-core && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_config_migrate_narration.py
"""
import json
import os
import pathlib
import sqlite3
import subprocess
import sys
import tempfile
import unittest

_SRC = pathlib.Path(__file__).resolve().parents[1] / 'src'


def _run_migrate_on(event_row: dict) -> dict:
    """把 event 行塞进一个全新的 DB，跑一次真实的 config 导入，读回结果。

    用子进程而不是直接 import：_migrate() 只在模块导入期执行一次，而 config 这个单例
    在本次 pytest 里早被别的模块导入过了。
    """
    tmp = tempfile.mkdtemp()
    db = os.path.join(tmp, 'test.db')
    conn = sqlite3.connect(db)
    conn.execute('CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT)')
    conn.execute("INSERT INTO config (key, value) VALUES ('event', ?)",
                 (json.dumps(event_row),))
    conn.commit()
    conn.close()

    env = {**os.environ, 'DB_PATH': db, 'PYTHONPATH': str(_SRC)}
    subprocess.run([sys.executable, '-c', 'import config'], env=env, check=True,
                   capture_output=True)

    conn = sqlite3.connect(db)
    row = conn.execute("SELECT value FROM config WHERE key='event'").fetchone()
    conn.close()
    return json.loads(row[0])['llm']


class TestNarrationKeyMigration(unittest.TestCase):
    def test_drops_the_retired_rounds_key(self):
        """轮数维度已取消 —— 这个键必须从库里清掉，不是留着当孤儿。"""
        llm = _run_migrate_on({'llm': {
            'narration_silence_rounds': 4,
            'narration_silence_seconds': 15,
            'prompt_system': './resource/memory/prompt_system.md',
        }})
        self.assertNotIn('narration_silence_rounds', llm)
        self.assertEqual(llm['narration_silence_seconds'], 15)

    def test_seeds_missing_keys(self):
        """已部署机器上的 'event' 行早就存在，靠 INSERT OR IGNORE 补不进新默认值。"""
        llm = _run_migrate_on({'llm': {'prompt_system': './resource/memory/prompt_system.md'}})
        self.assertEqual(llm['narration_silence_seconds'], 15)
        self.assertEqual(llm['narration_context_chars'], 6000)
        self.assertEqual(llm['narration_timeout_s'], 20)

    def test_does_not_clobber_a_hand_tuned_value(self):
        """手工调过的值不动 —— 只补缺失的键、只改还停在旧默认上的。"""
        llm = _run_migrate_on({'llm': {
            'narration_silence_seconds': 90,
            'prompt_system': './resource/memory/prompt_system.md',
        }})
        self.assertEqual(llm['narration_silence_seconds'], 90)

    def test_retunes_a_value_still_sitting_on_the_old_default(self):
        """沉默阈值 25 → 15。

        25 是几小时前由这段迁移自己种进去的，不是谁选的 —— 停在 25 的机器要跟着改，
        否则它们会永远停在一个没人选过的旧默认上（Tianyi 就是这么来的）。
        """
        llm = _run_migrate_on({'llm': {
            'narration_silence_seconds': 25,
            'prompt_system': './resource/memory/prompt_system.md',
        }})
        self.assertEqual(llm['narration_silence_seconds'], 15)

    def test_auto_notify_false_does_not_carry_over(self):
        """旧键关着，新功能**不该**跟着被关死。

        auto_notify 原本管的是"把模型写的 content 自动念出来"。Tianyi 上有人因为那功能念
        的是内部推理而关掉它 —— 完全合理。现在 content 自动播报已废除，同一个键若被重新
        定义成"进度播报总开关"，那个旧决定就会静默地把一个它从没评价过的新功能也关死，
        而设置页上看不出任何异常。所以换键：新键按默认 True 生效，旧键删掉。
        """
        llm = _run_migrate_on({'llm': {
            'auto_notify': False,
            'prompt_system': './resource/memory/prompt_system.md',
        }})
        self.assertNotIn('auto_notify', llm, '作废的旧键要从库里删掉')
        self.assertIs(llm['auto_narration'], True, '新功能按自己的默认值生效')

    def test_an_explicit_new_key_is_respected(self):
        """换键只影响旧值继承；有人明确关了新键就得听他的。"""
        llm = _run_migrate_on({'llm': {
            'auto_narration': False,
            'prompt_system': './resource/memory/prompt_system.md',
        }})
        self.assertIs(llm['auto_narration'], False)

    def test_is_idempotent(self):
        """跑第二遍不该再改动任何东西（每次启动都会跑一次）。"""
        base = {'llm': {'narration_silence_rounds': 4,
                        'prompt_system': './resource/memory/prompt_system.md'}}
        once = _run_migrate_on(base)
        twice = _run_migrate_on({'llm': dict(once)})
        self.assertEqual(once, twice)
        self.assertNotIn('narration_silence_rounds', twice)


if __name__ == '__main__':
    unittest.main()
