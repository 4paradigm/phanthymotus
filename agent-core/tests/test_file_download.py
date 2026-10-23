"""Exercise the real registration/lifecycle/graph functions without ROS startup."""
import ast
import asyncio
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

SRC = Path(__file__).resolve().parents[1] / 'src'


def load_function(filename, name, namespace):
    tree = ast.parse((SRC / filename).read_text())
    node = next(n for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), filename, 'exec'), namespace)
    return namespace[name]


class FileDownloadTest(unittest.TestCase):
    def test_registration_adds_download_card_without_removing_existing_tools(self):
        import_values = types.ModuleType('api.mcp_manage')
        saved = []
        import_values._get_mcp_list = lambda: [{'id': 'driver'}, {'id': 'agentcore', 'tools': []}]
        import_values._save_mcp_list = saved.extend
        api = types.ModuleType('api')
        api.mcp_manage = import_values
        register = load_function('start.py', '_register_core_mcp', {})
        with patch.dict(sys.modules, {'api': api, 'api.mcp_manage': import_values}):
            register(silent=True)
        core = next(m for m in saved if m['id'] == 'agentcore')
        names = [t['name'] for t in core['tools']]
        self.assertIn('file_download', names)
        self.assertIn('decision_core', names)
        self.assertIn('remote_image', names)
        self.assertEqual(len([m for m in saved if m['id'] == 'agentcore']), 1)
        self.assertIn({'id': 'driver'}, saved)

    def test_discovery_preserves_provider_declaration_without_calling_listing(self):
        schema = {'properties': {'action': {'enum': ['list_assets']}},
                  'x-file-download': {'protocol': 'motus.files/1', 'action': 'list_assets'}}
        methods = []

        class Response:
            status = 200

            def __init__(self, method):
                self.method = method

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def json(self, **kwargs):
                return {'result': {'tools': [{'name': 'arbitrary_export', 'inputSchema': schema}]}}

        class Session(Response):
            def __init__(self, **kwargs):
                pass

            def post(self, url, json, **kwargs):
                methods.append(json['method'])
                return Response(json['method'])

        ping = load_function('api/mcp_manage.py', '_ping_mcp_http', {
            'aiohttp': types.SimpleNamespace(ClientTimeout=lambda **kwargs: None, ClientSession=Session),
            'asyncio': asyncio,
        })
        result = asyncio.run(ping('http://fixture/mcp'))
        self.assertEqual(result['tools'][0]['inputSchema'], schema)
        self.assertEqual(methods, ['initialize', 'tools/list', 'resources/list'])

    def test_download_lifecycle_does_not_start_camera_or_subscribe_to_dds(self):
        call = load_function('api/mcp_manage.py', 'mcp_call_tool', {
            'MCPCallRequest': object, 'typing': __import__('typing'),
            '_handle_agentcore_call': lambda req: asyncio.sleep(0, result={'code': 200, 'data': None}),
        })
        for action in ('start', 'info', 'stop'):
            result = asyncio.run(call('agentcore', types.SimpleNamespace(
                tool='file_download', arguments={'action': action, 'instance_id': 'download'})))
            self.assertEqual(result['code'], 200)
            self.assertIn(result['data']['state'], ('ready', 'idle'))
        result = asyncio.run(call('agentcore', types.SimpleNamespace(
            tool='file_download', arguments={'action': 'capture_photo'})))
        self.assertEqual(result['code'], 400)

    def test_file_link_is_not_an_unresolved_stream_during_project_start(self):
        file_link = {'fromCardId': 'capture', 'toCardId': 'download',
                     'format': 'file/reference', 'fromTopic': ''}
        normal_link = {'fromCardId': 'mic', 'toCardId': 'download',
                       'format': 'data/json', 'fromTopic': '/mic'}
        tree = ast.parse((SRC / 'api/config.py').read_text())
        start = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef)
                     and n.name == '_do_start_project_impl')
        assignment = next(n for n in start.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'connections' for t in n.targets))
        namespace = {'layout': {'connections': [file_link, normal_link]},
                     '_topic_of_connection': lambda c: c['fromTopic']}
        exec(compile(ast.Module(body=[assignment], type_ignores=[]), 'config.py', 'exec'), namespace)
        resolve = load_function('api/config.py', '_resolve_input_topics', namespace)
        self.assertEqual(resolve('download'), ('/mic', [], []))
        self.assertEqual(resolve('unconnected'), ('', [], []))


if __name__ == '__main__':
    unittest.main()
