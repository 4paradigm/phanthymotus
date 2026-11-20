import base64
import hashlib
import importlib.util
import json
import secrets
import shutil
import subprocess
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
        for name in ['bridge_core.py', 'http_server.py', 'edge_backend.py', 'run_bridge.py',
                     'login_edge.py', 'bridge_watchdog.ps1', 'remote_start_bridge.ps1',
                     'config.example.json', 'README.txt', 'bootstrap.ps1']:
            (self.root / name).write_text('# package', encoding='utf-8')
        (self.root / 'bridge_config.ps1').write_bytes(
            (PERCEPTION / 'deploy' / 'windows_bridge' / 'bridge_config.ps1').read_bytes())
        self.args = dict(bridge_url='http://192.168.1.2:8765', token='t' * 32,
                         remote_start_user='operator', remote_start_key_path='/secret/key')
        self.uploaded_core = None
        self.uploaded_core_path = None
        self.patch_key = patch.object(Path, 'is_file', return_value=True)
        self.patch_key.start()
        self.addCleanup(self.patch_key.stop)

    @staticmethod
    def marker(info):
        return b'BRIDGE_CONFIG:' + base64.b64encode(json.dumps(info).encode('utf-8')) + b'\r\n'

    def remote(self, info=None, *, probe_output=None, probe_code=0):
        """Simulate each SSH/SCP step without touching a frontdesk machine."""
        if info is None:
            info = {'installed': False}

        def run(command, **kwargs):
            if command[0] == 'scp':
                if Path(command[-2]).name == 'core.json':
                    self.uploaded_core_path = Path(command[-2])
                    self.uploaded_core = json.loads(self.uploaded_core_path.read_text(encoding='utf-8'))
                return CompletedProcess(command, 0, b'', b'')
            if '-EncodedCommand ' in command[-1]:
                script = base64.b64decode(command[-1].rsplit(' ', 1)[-1]).decode('utf-16le')
                if 'Get-BridgeTokenInfo' in script:
                    output = self.marker(info) if probe_output is None else probe_output
                    return CompletedProcess(command, probe_code, output, output if probe_code else b'')
                return CompletedProcess(command, 0, b'UPLOAD_READY\r\n', b'\xd5\xe2')
            return CompletedProcess(command, 0, b'BRIDGE_BOOTSTRAP_OK\r\n', b'\xd5\xe2')
        return run

    def assert_no_uploads(self, run):
        self.assertFalse(any(c.args[0][0] == 'scp' for c in run.call_args_list))

    def test_transfer_uses_scp_and_strict_host_checking(self):
        with patch.object(module.subprocess, 'run', side_effect=self.remote()) as run:
            result = module.deploy(self.args, self.root)
        self.assertEqual(result['status'], 'installed')
        self.assertEqual(result['token'], self.args['token'])
        self.assertEqual(self.uploaded_core['token'], result['token'])
        self.assertFalse(self.uploaded_core_path.exists())
        uploads = [c.args[0] for c in run.call_args_list if c.args[0][0] == 'scp']
        self.assertTrue(uploads)
        self.assertTrue(all('-P' in c and '-p' not in c for c in uploads))
        self.assertTrue(all(':D:/ParadigmVisitorBridge/upload/' in c[-1] for c in uploads))
        self.assertTrue(any(c[-1].endswith('/bridge_config.ps1') for c in uploads))
        for call in run.call_args_list:
            command = call.args[0]
            self.assertIn('StrictHostKeyChecking=yes', command)
            self.assertIn('BatchMode=yes', command)
            self.assertNotIn(self.args['token'], ' '.join(command))
            self.assertNotIn('input', call.kwargs)
            self.assertNotIn('text', call.kwargs)
            if '-EncodedCommand ' in command[-1]:
                decoded = base64.b64decode(command[-1].rsplit(' ', 1)[-1]).decode('utf-16le')
                self.assertNotIn(self.args['token'], decoded)

    def test_existing_token_is_reused_even_with_stale_supplied_token(self):
        current_token = 'frontdesk-existing-secret-123456789'
        with patch.object(module.subprocess, 'run', side_effect=self.remote(
                {'installed': True, 'token': current_token})) as run, \
                patch.object(module.secrets, 'token_urlsafe') as generate:
            result = module.deploy(self.args, self.root)
        generate.assert_not_called()
        self.assertEqual(result['token'], current_token)
        self.assertEqual(self.uploaded_core['token'], current_token)
        self.assertEqual(result['bridge_url'], self.args['bridge_url'])
        self.assertTrue(all(current_token not in ' '.join(c.args[0]) for c in run.call_args_list))

    def test_new_computer_with_blank_token_generates_strong_secret(self):
        self.args['token'] = ''
        with patch.object(module.subprocess, 'run', side_effect=self.remote()), \
                patch.object(module.secrets, 'token_urlsafe', wraps=secrets.token_urlsafe) as generate:
            result = module.deploy(self.args, self.root)
        generate.assert_called_once_with(32)
        self.assertGreaterEqual(len(result['token']), 43)
        self.assertRegex(result['token'], r'^[A-Za-z0-9_-]+$')
        self.assertEqual(self.uploaded_core['token'], result['token'])

    def test_bad_probe_output_is_rejected_without_secret_leak_or_upload(self):
        secret = 'do-not-disclose-this-existing-secret'
        outputs = [b'', b'STARTED', b'BRIDGE_CONFIG:bad-base64!',
                   self.marker([]), self.marker({'installed': 'true', 'token': secret}),
                   self.marker({'installed': True, 'token': 'short'}),
                   self.marker({'installed': True, 'token': secret + ' '}),
                   self.marker({'installed': True, 'token': secret}) * 2]
        for index, output in enumerate(outputs):
            with self.subTest(case=index), \
                    patch.object(module.subprocess, 'run', side_effect=self.remote(probe_output=output)) as run:
                with self.assertRaises(RuntimeError) as raised:
                    module.deploy(self.args, self.root)
                self.assertNotIn(secret, str(raised.exception))
                if output:
                    self.assertNotIn(output.decode('ascii', errors='replace'), str(raised.exception))
                self.assertEqual(run.call_count, 1)
                self.assert_no_uploads(run)

    def test_failed_probe_does_not_expose_stdout_or_stderr(self):
        secret = b'existing-token-from-a-broken-config-12345'
        with patch.object(module.subprocess, 'run', side_effect=self.remote(
                probe_output=secret + self.args['token'].encode(), probe_code=1)) as run:
            with self.assertRaises(RuntimeError) as raised:
                module.deploy(self.args, self.root)
        self.assertNotIn(secret.decode(), str(raised.exception))
        self.assertNotIn(self.args['token'], str(raised.exception))
        self.assertEqual(run.call_count, 1)
        self.assert_no_uploads(run)

    def test_short_custom_token_does_not_start_upload(self):
        self.args['token'] = 'short-secret'
        with patch.object(module.subprocess, 'run', side_effect=self.remote()) as run:
            with self.assertRaises(ValueError) as raised:
                module.deploy(self.args, self.root)
        self.assertNotIn(self.args['token'], str(raised.exception))
        self.assert_no_uploads(run)

    def test_corrupt_asset_is_rejected_before_ssh(self):
        (self.root / 'offline' / 'test.whl').write_bytes(b'corrupt')
        with patch.object(module.subprocess, 'run') as run:
            with self.assertRaises(ValueError):
                module.deploy(self.args, self.root)
        run.assert_not_called()


@unittest.skipUnless(shutil.which('powershell.exe'), 'Windows PowerShell is not installed')
class InstallResolverTests(unittest.TestCase):
    def test_resolver_selects_active_task_or_single_existing_installation(self):
        # Exercise the actual function while mocking only Windows discovery.
        # No scheduled tasks, frontdesk files or security settings are changed.
        resolver = (PERCEPTION / 'deploy' / 'windows_bridge' / 'bridge_config.ps1').read_text(encoding='utf-8-sig')
        script = "$ErrorActionPreference = 'Stop'\n" + resolver + r'''
function Get-ScheduledTask { param($TaskName, $ErrorAction) return $script:bridgeTask }
function Join-Path { param($Path, $ChildPath) return $Path.TrimEnd('\') + '\' + $ChildPath }
function Test-Path { param($LiteralPath) return $script:existingPaths -contains $LiteralPath }

$script:bridgeTask = $null
$script:existingPaths = @()
if ((Get-BridgeInstallDirectory) -ne 'D:\ParadigmVisitorBridge\app') { throw 'Wrong clean install path' }

$script:existingPaths = @('D:\ParadigmVisitorBridge\config.json')
if ((Get-BridgeInstallDirectory) -ne 'D:\ParadigmVisitorBridge') { throw 'Legacy D installation was not reused' }

$script:existingPaths = @('C:\ProgramData\ParadigmVisitorBridge\config.json')
if ((Get-BridgeInstallDirectory) -ne 'C:\ProgramData\ParadigmVisitorBridge') { throw 'Legacy C installation was not reused' }

$script:existingPaths += 'D:\ParadigmVisitorBridge\app\config.json'
$rejected = $false
try { $null = Get-BridgeInstallDirectory } catch { $rejected = $true }
if (!$rejected) { throw 'Ambiguous installations must be rejected' }

$script:bridgeTask = [pscustomobject]@{ Actions = @([pscustomobject]@{ WorkingDirectory = 'D:\CustomBridge' }) }
$script:existingPaths += 'D:\CustomBridge\run_bridge.py'
if ((Get-BridgeInstallDirectory) -ne 'D:\CustomBridge') { throw 'Active task directory must take priority' }

$script:existingPaths = @('D:\ParadigmVisitorBridge\app\config.json')
$rejected = $false
try { $null = Get-BridgeInstallDirectory } catch { $rejected = $true }
if (!$rejected) { throw 'Broken task must not silently create another installation' }
Write-Output 'RESOLVER_OK'
'''
        encoded = base64.b64encode(script.encode('utf-16le')).decode('ascii')
        result = subprocess.run([shutil.which('powershell.exe'), '-NoProfile', '-NonInteractive',
                                 '-EncodedCommand', encoded], capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))
        self.assertIn(b'RESOLVER_OK', result.stdout)
