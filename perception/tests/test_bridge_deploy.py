import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

PERCEPTION = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location('bridge_deploy', PERCEPTION / 'plugins' / 'bridge_deploy.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DeployTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        assets = self.root / 'offline'
        assets.mkdir()
        manifest = {}
        for name in ['python-3.12.10-amd64.exe', 'test.whl']:
            content = b'offline fixture'
            (assets / name).write_bytes(content)
            manifest[name] = hashlib.sha256(content).hexdigest()
        (assets / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        for name in ['bridge_core.py', 'edge_backend.py', 'ssh_worker.py', 'login_edge.py',
                     'config.example.json', 'README.txt', 'bootstrap_ssh.ps1']:
            (self.root / name).write_text('# package', encoding='utf-8')
        self.args = dict(bridge_url='ssh://192.168.1.2', remote_start_user='operator',
                         remote_start_password='secret')
        self.uploaded_core = None
        self.patch_key = patch.object(Path, 'is_file', return_value=True)
        self.patch_key.start()
        self.addCleanup(self.patch_key.stop)

    def remote(self):
        def run(command, **kwargs):
            if command[0] == 'scp':
                if Path(command[-2]).name == 'core.json':
                    self.uploaded_core = json.loads(Path(command[-2]).read_text(encoding='utf-8'))
                return CompletedProcess(command, 0, b'', b'')
            if '-EncodedCommand ' in command[-1]:
                return CompletedProcess(command, 0, b'UPLOAD_READY\r\n', b'')
            return CompletedProcess(command, 0, b'SSH_AUTOMATION_OK\r\n', b'')
        return run

    def test_transfer_uses_only_ssh_and_scp(self):
        with patch.object(module.subprocess, 'run', side_effect=self.remote()) as run:
            result = module.deploy(self.args, self.root)
        self.assertEqual(result['status'], 'installed')
        self.assertNotIn('token', result)
        self.assertNotIn('token', self.uploaded_core)
        uploads = [call.args[0] for call in run.call_args_list if call.args[0][0] == 'scp']
        self.assertTrue(all(':D:/ParadigmVisitorAutomation/upload/' in command[-1] for command in uploads))
        self.assertTrue(any(command[-1].endswith('/bootstrap_ssh.ps1') for command in uploads))
        for call in run.call_args_list:
            command = call.args[0]
            self.assertIn(command[0], {'ssh', 'scp'})
            self.assertIn('StrictHostKeyChecking=accept-new', command)
            self.assertNotIn('secret', command)

    def test_corrupt_asset_is_rejected_before_ssh(self):
        (self.root / 'offline' / 'test.whl').write_bytes(b'corrupt')
        with patch.object(module.subprocess, 'run') as run:
            with self.assertRaises(ValueError):
                module.deploy(self.args, self.root)
        run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
