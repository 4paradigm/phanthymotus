import importlib.util
import tempfile
import unittest
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
from subprocess import CompletedProcess

spec = importlib.util.spec_from_file_location('bridge_deploy', Path(__file__).parents[1] / 'plugins' / 'bridge_deploy.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DeployTests(unittest.TestCase):
    def offline_package(self, root):
        assets = root / 'offline'
        assets.mkdir()
        manifest = {}
        for name in ['python-3.12.10-amd64.exe', 'test.whl']:
            content = b'offline fixture'
            (assets / name).write_bytes(content)
            manifest[name] = hashlib.sha256(content).hexdigest()
        (assets / 'manifest.json').write_text(json.dumps(manifest))

    def test_transfer_uses_scp_and_strict_host_checking(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.offline_package(root)
            for name in ['bridge_core.py','http_server.py','edge_backend.py','run_bridge.py','login_edge.py','bridge_watchdog.ps1','remote_start_bridge.ps1','config.example.json','bootstrap.ps1']:
                (root / name).write_text('# package', encoding='utf-8')
            args = dict(bridge_url='http://192.168.1.2:8765', token='t'*32, remote_start_user='operator', remote_start_key_path='/secret/key')
            with patch.object(Path, 'is_file', return_value=True), patch.object(module.subprocess, 'run', return_value=CompletedProcess([], 0, b'UPLOAD_READY BRIDGE_BOOTSTRAP_OK', b'\xd5\xe2')) as run:
                self.assertEqual(module.deploy(args, root)['status'], 'installed')
                command = run.call_args.args[0]
                self.assertIn('StrictHostKeyChecking=yes', command)
                self.assertNotIn(args['token'], ' '.join(command))
                uploads = [c.args[0] for c in run.call_args_list if c.args[0][0] == 'scp']
                self.assertTrue(uploads)
                self.assertTrue(all('-P' in c and '-p' not in c for c in uploads))
                self.assertTrue(all(':D:/ParadigmVisitorBridge/upload/' in c[-1] for c in uploads))
                self.assertNotIn('input', run.call_args.kwargs)
                self.assertNotIn('text', run.call_args.kwargs)

    def test_restricted_start_key_cannot_report_install_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.offline_package(root)
            for name in ['bridge_core.py','http_server.py','edge_backend.py','run_bridge.py','login_edge.py','bridge_watchdog.ps1','remote_start_bridge.ps1','config.example.json','bootstrap.ps1']:
                (root / name).write_text('', encoding='utf-8')
            with patch.object(Path, 'is_file', return_value=True), patch.object(module.subprocess, 'run', return_value=CompletedProcess([], 0, b'STARTED', b'\xd5\xe2')):
                with self.assertRaises(RuntimeError):
                    module.deploy(dict(bridge_url='http://192.168.1.2:8765', token='t'*32, remote_start_user='operator', remote_start_key_path='/key'), root)

    def test_corrupt_asset_is_rejected_before_ssh(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.offline_package(root)
            for name in ['bridge_core.py','http_server.py','edge_backend.py','run_bridge.py','login_edge.py','bridge_watchdog.ps1','remote_start_bridge.ps1','config.example.json']:
                (root / name).write_text('')
            (root / 'offline' / 'test.whl').write_bytes(b'corrupt')
            with patch.object(Path, 'is_file', return_value=True), patch.object(module.subprocess, 'run') as run:
                with self.assertRaises(ValueError):
                    module.deploy(dict(bridge_url='http://192.168.1.2:8765', token='t'*32, remote_start_user='operator', remote_start_key_path='/key'), root)
                self.assertFalse(any('.chunk' in str(c) for c in run.call_args_list))
