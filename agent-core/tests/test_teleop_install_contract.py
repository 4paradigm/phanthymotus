"""Real source contract: Plugin -> MCP -> Core -> pinned TLS package/download.

ActuCore's production methods are compiled verbatim to avoid requiring ROS/RTC
in the Core test image. Only the already-initialized host and artifact directory
are supplied by the fixture; no MCP/package response shape is mocked.
"""
import ast
import asyncio
import copy
import hashlib
import importlib
import importlib.util
import json
import logging
from pathlib import Path
import ssl
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import urlsplit

from aiohttp import web
from cryptography import x509
from cryptography.hazmat.primitives import hashes

from test_teleop_install import APK, api, capture, client, installation_store


ROOT = Path(__file__).parents[2]


def source_nodes(path, names, class_name=None):
    nodes = ast.parse(path.read_text()).body
    if class_name:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    return [n for n in nodes if getattr(n, 'name', '') in names]


def test_actual_plugin_mcp_nested_summary_and_flat_tls_package_both_work(tmp_path, monkeypatch):
    # Compile the existing stdlib artifact verifier, not a fabricated metadata dict.
    path = ROOT/'actucore/plugins/teleop/onboarding.py'
    spec = importlib.util.spec_from_file_location('contract_onboarding', path)
    onboarding = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(onboarding)
    artifact = tmp_path/'artifact'
    artifact.mkdir()
    (artifact/'pico.apk').write_bytes(APK)
    manifest = {'filename':'pico.apk','application_id':'com.phanthymotus.picocapture',
        'build_type':'release','version':'test','version_code':1,'sha256':hashlib.sha256(APK).hexdigest(),
        'signing_certificate_sha256':'a'*64,'size_bytes':len(APK)}
    (artifact/'package.json').write_text(json.dumps(manifest))
    package_metadata = lambda: onboarding.package_metadata(artifact)

    async def run():
        # Reuse only generated localhost TLS files; production endpoints run on
        # their own server below, and this recorder must remain unused.
        async with capture(tmp_path) as (_, unused):
            path = ROOT/'actucore/plugins/teleop/capture_server.py'
            namespace = {'asyncio':asyncio, 'web':web, 'urlsplit':urlsplit,
                'package_metadata':package_metadata, 'APK_DIRECTORY':artifact,
                'APK_FILENAME':'pico.apk', 'MIME_TYPE':onboarding.MIME_TYPE}
            nodes = source_nodes(path, {'package_handler'})
            nodes += source_nodes(path, {'installation_info'}, 'CaptureWssServer')
            exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'), namespace)
            app = web.Application()
            app.router.add_get('/onboarding/package', namespace['package_handler'])
            app.router.add_get('/onboarding/apk', namespace['package_handler'])
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(tmp_path/'capture.pem', tmp_path/'capture.key')
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, '127.0.0.1', 0, ssl_context=context)
            await site.start()
            port = site._server.sockets[0].getsockname()[1]
            certificate = x509.load_pem_x509_certificate((tmp_path/'capture.pem').read_bytes())
            server = SimpleNamespace(_config={'public_wss_url':f'wss://127.0.0.1:{port}/ws/teleop-capture'},
                enrollment=SimpleNamespace(device_id=certificate.fingerprint(hashes.SHA256()).hex()))
            server.installation_info = lambda: namespace['installation_info'](server)

            path = ROOT/'actucore/plugins/teleop/plugin.py'
            tree = ast.parse(path.read_text())
            actions = next(n for n in tree.body if isinstance(n,ast.Assign) and any(
                isinstance(t,ast.Name) and t.id=='ACTIONS' for t in n.targets))
            methods = source_nodes(path, {'__init__','_actions','_ensure_host','dispatch'}, 'TeleopPlugin')
            cls = ast.ClassDef(name='TeleopPlugin', bases=[], keywords=[], body=methods, decorator_list=[])
            plugin_ns = {'copy':copy,'threading':threading,'hashlib':hashlib,'json':json,'Path':Path}
            exec(compile(ast.fix_missing_locations(ast.Module(body=[actions,cls],type_ignores=[])),str(path),'exec'),plugin_ns)
            plugin = plugin_ns['TeleopPlugin']({'robot_profile':'tianyi2'}, None)
            plugin.runtime = object()  # host already running; no ROS initialization
            plugin.server = server
            direct = plugin.dispatch('teleop', {'action':'installation_info'})
            assert direct['package']['available'] and 'available' not in direct

            path = ROOT/'actucore/main.py'
            calls = []
            def dispatch(name, arguments):
                calls.append((name,dict(arguments)))
                return plugin.dispatch(name,arguments)
            handler_ns = {'BaseHTTPRequestHandler':BaseHTTPRequestHandler,'json':json,
                'log':logging.getLogger('onboarding-contract'),'_brief':str,
                '_bundle':SimpleNamespace(dispatch=dispatch,server_name='actucore-bundle')}
            exec(compile(ast.Module(body=source_nodes(path,{'make_handler'}),type_ignores=[]),str(path),'exec'),handler_ns)
            mcp = ThreadingHTTPServer(('127.0.0.1',0),handler_ns['make_handler']())
            thread = threading.Thread(target=mcp.serve_forever,daemon=True)
            thread.start()
            url = f'http://127.0.0.1:{mcp.server_port}/mcp'
            key = tmp_path/'management-key'
            key.write_text('local-contract-fixture-'+'x'*40)
            monkeypatch.setenv('TELEOP_MANAGEMENT_URL',url)
            monkeypatch.setenv('TELEOP_MANAGEMENT_KEY_FILE',str(key))
            management = importlib.import_module('api.mcp_manage')
            monkeypatch.setattr(management,'_get_mcp_list',lambda:[
                {'id':'registered-ac','url':url,'tools':[{'name':'teleop'}]}])
            try:
                async with client(monkeypatch, None, stub_mcp=False) as c:
                    result = await c.post('/api/teleop-install/registered-ac')
                    assert result.status_code == 200, result.text
                    data = result.json()['data']
                    assert data['package'] == direct['package']
                    assert data['qr_svg'].startswith('<?xml')
                    downloaded = await c.get('/pico/'+data['ticket']+'/apk')
                    assert downloaded.status_code == 200 and downloaded.content == APK
                    assert calls == [('teleop',{'action':'installation_info'})]
                    assert not unused
            finally:
                mcp.shutdown();mcp.server_close();thread.join(timeout=2)
                await runner.cleanup()
    asyncio.run(run())
