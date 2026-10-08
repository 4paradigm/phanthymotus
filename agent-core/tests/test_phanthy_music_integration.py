"""Music card registration, action permissions and config persistence boundaries."""
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('DB_PATH', str(Path(tempfile.mkdtemp()) / 'music.db'))
import config
import mcp_client
from api import canvas, solutions
from event.llm import _restricted_channel_tool_allowed
from subagent.agent import Subagent
from subagent.protocol import SubagentSpec


def card():
    # Execute the shipped declarative schema only, without importing ROS.
    source = ROOT.parent / 'perception/plugins/phanthy_music.py'
    names = {'FORMAT', 'OUTPUT_TOPIC', 'SEARCH_PARAMS', 'CONFIG_PARAMS', 'ACTIONS', 'ACTION_DESCRIPTIONS', 'TOOLS'}
    nodes = [n for n in ast.parse(source.read_text()).body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)]
    ns = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)
    return deepcopy(ns['TOOLS'][0])


def registry_entry():
    schemas = mcp_client._to_openai_schema('p', card())
    return {'online': True, 'transport': 'http', 'url': 'http://localhost:15720',
            'schemas': {s['name']: s for s in schemas},
            'tool_meta': {s['name']: {'type': 'processor', 'resource': frozenset({'music'})} for s in schemas},
            'split_map': {s['name']: {'tool': 'phanthy-music', 'action': s['name'].split('__')[-1]} for s in schemas},
            'tool_groups': {'phanthy-music': [s['name'] for s in schemas]}}


class ConfigTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.old_services = config.main.get('services', {})
        config.main['services'] = {'mcp': [{'id': 'p', 'tools': [card()]}]}

    def tearDown(self):
        config.main['services'] = self.old_services

    async def test_save_and_solution_reject_credentials_before_any_mutation(self):
        canary = 'fictional-secret-that-must-not-persist'
        invalid = {'catalogue_type': 'motus_music', 'api_key': canary}
        before = config.main.get('canvas_layout', {})
        from fastapi import HTTPException
        with patch.object(canvas, 'apply_tool_config') as apply:
            for call in (canvas.save_tool_config('p', 'phanthy-music', invalid),
                         canvas.save_instance_config('p', 'phanthy-music', 'card1', invalid)):
                with self.assertRaises(HTTPException) as ctx:
                    await call
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertNotIn(canary, str(ctx.exception))
            apply.assert_not_called()
        with self.assertRaises(HTTPException):
            await solutions._apply_canvas({'cards': [], 'toolConfigs': {'d0:phanthy-music': invalid}}, {'d0': 'p'})
        self.assertEqual(config.main.get('canvas_layout', {}), before)
        with config._get_conn() as conn:
            values = conn.execute('SELECT value FROM config').fetchall()
        self.assertNotIn(canary, json.dumps(values))

    async def test_valid_config_roundtrip_pack_and_absent_schema_fail_closed(self):
        values = {'catalogue_type': 'mock', 'volume': 25, 'timeout_ms': 3000}
        with patch.object(canvas, 'apply_tool_config'):
            await canvas.save_tool_config('p', 'phanthy-music', values)
        self.assertEqual((await canvas.get_tool_config('p', 'phanthy-music'))['data'], values)
        with patch.object(solutions, '_mcp_list', return_value=[{'id': 'p', 'tools': [card()]}]), \
             patch.object(solutions, '_layout', return_value={}), \
             patch.object(canvas, 'all_tool_configs', return_value={'p:phanthy-music': values}):
            packed, _ = solutions._pack_canvas({'p': 'd0'}, set())
        self.assertEqual(packed['toolConfigs']['d0:phanthy-music'], values)
        self.assertNotIn('api_key', json.dumps(packed))
        from fastapi import HTTPException
        with self.assertRaises(HTTPException):
            await canvas.save_tool_config('missing', 'phanthy-music', {'api_key': 'fictional'})


class ActionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.patch = patch.dict(mcp_client.registry, {'p': registry_entry()}, clear=True)
        self.patch.start()
        self.binding = patch('canvas_binding.is_bound', return_value=True)
        self.binding.start()

    def tearDown(self):
        self.binding.stop(); self.patch.stop()

    async def test_viewer_only_search_bot_no_new_permission(self):
        for action in card()['inputSchema']['properties']['action']['enum']:
            name = 'mcp__p__phanthy-music__' + action
            self.assertEqual(_restricted_channel_tool_allowed(name, bot_restricted=False), action == 'search')
            self.assertFalse(_restricted_channel_tool_allowed(name, bot_restricted=True))
        self.assertFalse(_restricted_channel_tool_allowed('mcp__p__phanthy-music', bot_restricted=False))
        self.assertFalse(_restricted_channel_tool_allowed('mcp__p__phanthy-music-fake__search', bot_restricted=False))
        with patch('canvas_binding.is_bound', return_value=False):
            self.assertFalse(_restricted_channel_tool_allowed('mcp__p__phanthy-music__search', bot_restricted=False))
        mcp_client.registry['p']['transport'] = 'peer'
        self.assertFalse(_restricted_channel_tool_allowed('mcp__p__phanthy-music__search', bot_restricted=False))

    async def test_main_and_subagent_hide_lifecycle_and_hooks(self):
        entry = mcp_client.registry['p']
        exposed = []
        for name, schema in entry['schemas'].items():
            out = mcp_client.present_to_llm(schema, entry['tool_meta'][name],
                                           split_action=entry['split_map'][name]['action'])
            if out: exposed.append(name)
        expected = {'search', 'play', 'pause', 'resume', 'interrupt', 'set_volume', 'status'}
        self.assertEqual({name.split('__')[-1] for name in exposed}, expected)
        agent = Subagent(SubagentSpec(goal='Find a song'))
        self.assertEqual({s['name'] for s in agent._get_all_mcp_schemas()}, set(exposed))
        with patch('canvas_binding.is_bound', return_value=False):
            self.assertEqual(agent._get_all_mcp_schemas(), [])
        for action in ('start', 'stop', 'config', 'duck'):
            result = await mcp_client.call_tool('mcp__p__phanthy-music__' + action, {})
            self.assertIn('managed by the canvas', result)

    async def test_split_search_cannot_be_overridden_into_play(self):
        from contextlib import asynccontextmanager
        @asynccontextmanager
        async def session(*a, **kw): yield object()
        result = {'content': [{'type': 'text', 'text': '{"tracks":[]}'}]}
        with patch.object(mcp_client.aiohttp, 'ClientSession', session), \
             patch.object(mcp_client, '_jrpc', new=AsyncMock(return_value=result)) as rpc:
            await mcp_client.call_tool('mcp__p__phanthy-music__search', {'action': 'play', 'query': 'sleep'})
        self.assertEqual(rpc.call_args.args[3]['arguments']['action'], 'search')
        self.assertEqual(rpc.call_args.args[3]['name'], 'phanthy-music')

    async def test_subagent_dispatch_honours_music_filter_not_only_presentation(self):
        agent = Subagent(SubagentSpec(goal='Find a song', tool_filter=['*__search']))
        with patch.object(agent, '_get_desktop_tool_schemas', return_value=[]), \
             patch.object(mcp_client, 'call_tool', new=AsyncMock()) as call:
            result = await agent._dispatch_tool('mcp__p__phanthy-music__play', {'track_id': 'fictional'})
        self.assertIn('not available', result)
        call.assert_not_called()
