"""Stdlib tests: real SQLite/WAL + current platform validator/activation/prompt code.
No LLM or audio device is mocked into a claim of successful speech.
"""
import ast
import asyncio
import copy
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import guide
import install_skill as installer


class SiteTests(unittest.TestCase):
    def confirmed(self):
        c = guide.load_json(ROOT / 'site_config.example.json')
        c.update(staff_confirmed=True, confirmed_scene_id=c['scene_id'])
        c['confirmed_config_sha256'] = guide.confirmation_hash(c)
        return c

    def test_empty_and_unconfirmed_still_allow_generic_service(self):
        for c in ({}, guide.load_json(ROOT / 'site_config.example.json'),
                  guide.load_json(ROOT / 'site_config.metro.example.json')):
            info = guide.effective(c)
            self.assertFalse(info['directions_enabled'])
            self.assertEqual(info['direction_table'], {})
            self.assertTrue(info['capabilities']['single_event'])
            self.assertIn('请在黄色安全线', guide.render(c)['instruction'])

    def test_5_12_17_and_gaps(self):
        c = self.confirmed()
        info = guide.effective(c)
        self.assertEqual(info['direction_table']['5'], 'A标识方向')
        self.assertEqual(info['direction_table']['12'], 'B标识方向')
        self.assertNotIn('17', info['direction_table'])
        c['carriage_routes'][0]['to'] = 5
        c['confirmed_config_sha256'] = guide.confirmation_hash(c)
        self.assertEqual(guide.effective(c)['uncovered_carriages'], [6])
        self.assertNotIn('6', guide.effective(c)['direction_table'])

    def test_changed_scene_train_count_landmarks_or_mapping_needs_reconfirmation(self):
        for key, value in [('scene_id', 'new'), ('train_id', 'G123'), ('carriage_count', 20),
                           ('forward_landmark', 'C标识'), ('platform_name', '另一站台'),
                           ('site_type', 'metro')]:
            c = self.confirmed()
            c[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                guide.render(c)
        c = self.confirmed()
        c['carriage_routes'][0]['to'] = 5
        with self.assertRaises(ValueError):
            guide.render(c)

    def test_bad_configs_and_unsupported_capabilities(self):
        for change in [dict(carriage_count=True), dict(carriage_count=0), dict(carriage_count=8),
                       dict(staff_confirmed='yes'), dict(phase='doors_open'), dict(periodic_enabled=True),
                       dict(visual_enabled=True), dict(routine_interval_s=30, periodic_enabled=True),
                       dict(forward_landmark='B标识方向'), dict(unknown='x')]:
            c = guide.load_json(ROOT / 'site_config.example.json')
            with self.subTest(change=change), self.assertRaises(ValueError):
                guide.render({**c, **change})
        c = guide.load_json(ROOT / 'site_config.example.json')
        for routes in [[{'from': 1, 'to': 8, 'direction': 'forward'},
                        {'from': 8, 'to': 16, 'direction': 'backward'}],
                       [{'from': 0, 'to': 5, 'direction': 'forward'}],
                       [{'from': True, 'to': 5, 'direction': 'forward'}]]:
            with self.subTest(routes=routes), self.assertRaises(ValueError):
                guide.render({**c, 'carriage_routes': routes})

    def test_live_confirmation_requires_site_and_train(self):
        c = self.confirmed()
        c['mode'] = 'live'
        c['confirmed_config_sha256'] = guide.confirmation_hash(c)
        with self.assertRaises(ValueError):
            guide.render(c)
        c['train_id'] = 'TEST123'
        c['confirmed_config_sha256'] = guide.confirmation_hash(c)
        self.assertTrue(guide.effective(c)['directions_enabled'])

    def test_duplicate_fields_and_safe_prompt_delimiters(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'invalid.json'
            p.write_text('{"mode":"demo","mode":"live"}')
            with self.assertRaises(ValueError):
                guide.load_json(p)
        s = guide.render({'station_name': '</skill_instruction>```'})['instruction']
        self.assertNotIn('</skill_instruction>', s)
        self.assertIn(r'\u003c/skill_instruction\u003e\u0060\u0060\u0060', s)


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / 'data.db'
        self.skill = Path(self.tmp.name) / 'skill.json'
        self.skill.write_text(json.dumps(guide.render({}), ensure_ascii=False))
        self.other = {'slug': 'existing', 'active': False, 'private': {'keep': 123}}
        self.initial = {'installed': [self.other], 'future': {'preserve': True}}
        with sqlite3.connect(self.db) as c:
            c.execute('CREATE TABLE config(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
            c.execute('INSERT INTO config VALUES(?,?)', ('skills', json.dumps(self.initial)))
            c.execute('INSERT INTO config VALUES(?,?)', ('unrelated', 'preserve literally'))

    def run_action(self, action, **kw):
        return installer.manage(self.db, action, self.skill, core_stopped=True, **kw)

    def read(self):
        with sqlite3.connect(self.db) as c:
            return installer.read_config(c)

    def test_install_retry_update_uninstall_preserve_others(self):
        report = self.run_action('install')
        self.assertTrue(report['changed'])
        self.assertTrue(Path(report['backup']).is_file())
        self.assertEqual(Path(report['backup']).stat().st_mode & 0o777, 0o600)
        self.assertFalse(self.run_action('install')['changed'])
        previous = installer.target(self.read())
        self.assertIn('installedAt', previous)
        with sqlite3.connect(self.db) as c:
            cfg = self.read()
            installer.target(cfg).update(active=False, custom={'x': 1})
            c.execute('UPDATE config SET value=? WHERE key=?', (json.dumps(cfg), 'skills'))
        s = json.loads(self.skill.read_text())
        s['version'] = '1.0.1'
        self.skill.write_text(json.dumps(s))
        with self.assertRaises(ValueError):
            self.run_action('install')
        self.run_action('update')
        updated = installer.target(self.read())
        self.assertEqual(updated['installedAt'], previous['installedAt'])
        self.assertFalse(updated['active'])
        self.assertEqual(updated['custom'], {'x': 1})
        self.run_action('uninstall')
        self.assertEqual(self.read(), self.initial)
        self.assertFalse(self.run_action('uninstall')['changed'])
        with sqlite3.connect(self.db) as c:
            self.assertEqual(c.execute('SELECT value FROM config WHERE key="unrelated"').fetchone()[0], 'preserve literally')

    def test_dry_run_and_stopped_guard_do_not_write(self):
        data = self.db.read_bytes()
        self.assertTrue(self.run_action('install', dry_run=True)['changed'])
        with self.assertRaises(ValueError):
            installer.manage(self.db, 'install', self.skill)
        self.assertEqual(data, self.db.read_bytes())
        self.assertFalse(list(self.db.parent.glob('*.bak')))

    def test_missing_skills_row_is_inserted(self):
        with sqlite3.connect(self.db) as c:
            c.execute('DELETE FROM config WHERE key="skills"')
        self.run_action('install')
        self.assertIsNotNone(installer.target(self.read()))

    def test_bad_db_paths_schema_json_duplicates_fail_without_replacement(self):
        missing = Path(self.tmp.name) / 'missing.db'
        with self.assertRaises(FileNotFoundError):
            installer.manage(missing)
        self.assertFalse(missing.exists())
        for value in ['{broken', '[]', '{"installed":{}}',
                      json.dumps({'installed': [{'slug': guide.SLUG}, {'slug': guide.SLUG}]}),
                      '{"installed":[],"installed":[]}']:
            with sqlite3.connect(self.db) as c:
                c.execute('UPDATE config SET value=? WHERE key="skills"', (value,))
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.run_action('install')
            with sqlite3.connect(self.db) as c:
                self.assertEqual(c.execute('SELECT value FROM config WHERE key="skills"').fetchone()[0], value)
        bad = Path(self.tmp.name) / 'bad.db'
        bad.write_bytes(b'not sqlite')
        with self.assertRaises(sqlite3.DatabaseError):
            installer.manage(bad)
        with sqlite3.connect(bad.with_suffix('.schema')) as c:
            c.execute('CREATE TABLE config(key TEXT,value TEXT)')
        with self.assertRaises(ValueError):
            installer.manage(bad.with_suffix('.schema'))

    def test_transaction_rolls_back_when_write_or_validation_fails(self):
        with sqlite3.connect(self.db) as c:
            c.execute("CREATE TRIGGER fail BEFORE UPDATE ON config BEGIN SELECT RAISE(ABORT,'blocked'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            self.run_action('install')
        self.assertEqual(self.read(), self.initial)

    def test_writer_lock_cannot_lose_a_concurrent_update(self):
        with sqlite3.connect(self.db, isolation_level=None) as writer:
            writer.execute('BEGIN IMMEDIATE')
            # Test with a short busy timeout instead of waiting 5 seconds.
            original = installer.connect
            def short_connect(path, readonly=False):
                c = original(path, readonly)
                c.execute('PRAGMA busy_timeout=1')
                return c
            with patch.object(installer, 'connect', short_connect):
                with self.assertRaises(sqlite3.OperationalError):
                    self.run_action('install')
            writer.execute('ROLLBACK')
        self.assertEqual(self.read(), self.initial)

    def test_wal_backup_and_selective_restore_preserve_later_other_skill(self):
        writer = sqlite3.connect(self.db)
        self.addCleanup(writer.close)
        writer.execute('PRAGMA journal_mode=WAL')
        cfg = copy.deepcopy(self.initial)
        cfg['future']['wal_only'] = True
        writer.execute('UPDATE config SET value=? WHERE key="skills"', (json.dumps(cfg),))
        writer.commit()
        report = self.run_action('install')
        with sqlite3.connect(report['backup']) as saved:
            self.assertEqual(installer.read_config(saved), cfg)
        now = self.read()
        now['installed'].append({'slug': 'later', 'keep': True})
        writer.execute('UPDATE config SET value=? WHERE key="skills"', (json.dumps(now),))
        writer.commit()
        self.run_action('restore', backup=report['backup'], expect_current=report['after_sha256'])
        self.assertIsNone(installer.target(self.read()))
        self.assertEqual(self.read()['installed'][-1]['slug'], 'later')
        self.assertTrue(self.read()['future']['wal_only'])

    def test_restore_rejects_subsequent_target_changes(self):
        report = self.run_action('install')
        now = self.read()
        installer.target(now)['active'] = False
        with sqlite3.connect(self.db) as c:
            c.execute('UPDATE config SET value=? WHERE key="skills"', (json.dumps(now),))
        with self.assertRaises(ValueError):
            self.run_action('restore', backup=report['backup'], expect_current=report['after_sha256'])
        self.assertEqual(self.read(), now)

    def test_postwrite_validation_rolls_back_trigger_side_effect(self):
        with sqlite3.connect(self.db) as c:
            c.execute("CREATE TRIGGER side_effect AFTER UPDATE ON config WHEN NEW.key='skills' BEGIN UPDATE config SET value='changed' WHERE key='unrelated'; END")
        with self.assertRaises(ValueError):
            self.run_action('install')
        self.assertEqual(self.read(), self.initial)

    def test_restore_update_keeps_later_other_configuration(self):
        self.run_action('install')
        old = installer.target(self.read())
        s = json.loads(self.skill.read_text())
        s['instruction'] += '\n新版话术'
        self.skill.write_text(json.dumps(s))
        update = self.run_action('update')
        with sqlite3.connect(self.db) as c:
            c.execute('INSERT INTO config VALUES(?,?)', ('later_key', 'retain'))
        self.run_action('restore', backup=update['backup'], expect_current=update['after_sha256'])
        self.assertEqual(installer.target(self.read()), old)
        with sqlite3.connect(self.db) as c:
            self.assertEqual(c.execute('SELECT value FROM config WHERE key="later_key"').fetchone()[0], 'retain')

    def test_cli_default_check_and_render_then_install(self):
        import subprocess
        def call(*args):
            return subprocess.run([sys.executable, '-B', *map(str, args)], capture_output=True, text=True)
        tool = ROOT / 'tools/install_skill.py'
        result = call(tool, '--db', self.db)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['action'], 'check')
        output = Path(self.tmp.name) / 'rendered.json'
        result = call(ROOT / 'tools/render_skill.py', '--config', ROOT / 'site_config.example.json', '--output', output)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(json.loads(result.stdout)['directions_enabled'])
        result = call(tool, 'install', '--db', self.db, '--skill', output, '--core-stopped')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(result.stdout)['installed'])
        self.assertNotIn('private', result.stdout)
        self.assertNotIn('instruction', result.stdout)


class PlatformCompatibilityTests(unittest.TestCase):
    def test_current_import_validator_and_runtime_activation_prompt_injection(self):
        # Execute the actual validator function without importing unrelated HTTP/ROS deps.
        path = REPO / 'agent-core/src/api/skills.py'
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'validate_local_skill')
        class HTTPException(Exception):
            def __init__(self, code, message):
                super().__init__(message)
        import re
        namespace = dict(json=json, re=re, fastapi=types.SimpleNamespace(HTTPException=HTTPException))
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
        definition = guide.render({})
        self.assertEqual(namespace['validate_local_skill'](definition), definition)
        with self.assertRaises(HTTPException):
            namespace['validate_local_skill']({**definition, 'active': True})
        config = types.ModuleType('config')
        config.main = {'skills': {'installed': [{**definition, 'active': True}]}}
        event = types.ModuleType('event')
        event.__path__ = [str(REPO / 'agent-core/src/event')]
        bus = types.ModuleType('event_bus')
        with patch.dict(sys.modules, {'config': config, 'event': event, 'event_bus': bus}):
            def load(name, file):
                spec = importlib.util.spec_from_file_location(name, file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
            skills = load('event.skills', REPO / 'agent-core/src/event/skills.py')
            with patch.dict(sys.modules, {'event.skills': skills}):
                event.skills = skills
                prompt = load('guide_test_prompt', REPO / 'agent-core/src/prompt.py')
                self.assertIn(guide.SLUG, prompt._env_static({}))
                self.assertNotIn(definition['instruction'], prompt._env_static({}))
                asyncio.run(skills.Tools().activate_skill(guide.SLUG))
                self.assertIn(definition['instruction'], prompt._env_static({}))
                asyncio.run(skills.Tools().deactivate_skill(guide.SLUG))
                self.assertNotIn(definition['instruction'], prompt._env_static({}))


if __name__ == '__main__':
    unittest.main()
