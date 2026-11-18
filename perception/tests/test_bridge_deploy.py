import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from subprocess import CompletedProcess

spec = importlib.util.spec_from_file_location('bridge_deploy', Path(__file__).parents[1] / 'plugins' / 'bridge_deploy.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DeployTests(unittest.TestCase):
    def test_transfer_uses_stdin_and_strict_host_checking(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ['bridge_core.py','http_server.py','edge_backend.py','run_bridge.py','login_edge.py','bridge_watchdog.ps1','remote_start_bridge.ps1','config.example.json','bootstrap.ps1']:
                (root / name).write_text('# package', encoding='utf-8')
            args = dict(bridge_url='http://192.168.1.2:8765', token='t'*32, remote_start_user='operator', remote_start_key_path='/secret/key')
            with patch.object(Path, 'is_file', return_value=True), patch.object(module.subprocess, 'run', return_value=CompletedProcess([], 0, 'BRIDGE_BOOTSTRAP_OK', '')) as run:
                self.assertEqual(module.deploy(args, root)['status'], 'installed')
                command = run.call_args.args[0]
                self.assertIn('StrictHostKeyChecking=yes', command)
                self.assertNotIn(args['token'], ' '.join(command))
                self.assertIn(args['token'], run.call_args.kwargs['input'])

    def test_restricted_start_key_cannot_report_install_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ['bridge_core.py','http_server.py','edge_backend.py','run_bridge.py','login_edge.py','bridge_watchdog.ps1','remote_start_bridge.ps1','config.example.json','bootstrap.ps1']:
                (root / name).write_text('', encoding='utf-8')
            with patch.object(Path, 'is_file', return_value=True), patch.object(module.subprocess, 'run', return_value=CompletedProcess([], 0, 'STARTED', '')):
                with self.assertRaises(RuntimeError):
                    module.deploy(dict(bridge_url='http://192.168.1.2:8765', token='t'*32, remote_start_user='operator', remote_start_key_path='/key'), root)
