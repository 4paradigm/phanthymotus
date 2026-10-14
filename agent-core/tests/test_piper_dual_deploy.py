"""Deployment transaction tests; fake Docker/MCP, real config and backup files."""
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('piper_deploy', ROOT / 'deploy/piper_dual_deploy.py')
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.data = self.root / 'data'
        self.data.mkdir()
        self.config = self.data / 'config.dual.yaml'
        self.run_dir = self.root / 'run'
        self.run_dir.mkdir()
        self.env = self.root / '.env'
        self.env.write_text('ACCESS_TOKEN=do-not-print-this-token\n')
        self.fragment = self.run_dir / 'service.yml'
        self.fragment.write_text(yaml.safe_dump({'agilex-piper': {
            'image': '__IMAGE__', 'container_name': deploy.PRIMARY, 'network_mode': 'host',
            'volumes': [{'type': 'bind', 'source': str(self.data), 'target': str(self.data)}],
            'environment': {'CONFIG_PATH': 'old-config', 'PIPER_MOTION_ENABLED': '1', 'PIPER_CAMERA_ENABLED': '1'},
        }}))
        self.template = {'mcp_port': 15741, 'data_dir': str(self.data),
                         'arms': {'left': {'can_interface': 'can0'}, 'right': {'can_interface': 'can1'}},
                         'calibration_path': str(self.data / 'calibration.json')}
        self.old_compose = self.root / 'old-compose.yml'
        self.old_compose.write_text('original compose with private settings\n')
        self.items = {name: self.container(name) for name in deploy.TARGETS}
        self.items['embodied-pico'] = self.container('embodied-pico')
        self.items['agent-core'] = self.container('agent-core')
        self.calls = []
        self.project_running = False
        self.editor = None
        self.driver_busy = False
        self.motion_enabled = False
        self.output = io.StringIO()
        self.patches = [
            patch.object(deploy, 'DATA', self.data), patch.object(deploy, 'CONFIG', self.config),
            patch.object(deploy, 'CORE_ENV', self.env), patch.object(deploy, 'docker', side_effect=self.docker),
            patch.object(deploy, 'request', side_effect=self.request),
            patch.object(deploy, 'can_interfaces', return_value=['can0', 'can1']),
            patch.object(deploy.sys.stdin, 'isatty', return_value=True),
            patch.object(deploy.socket, 'socket'), patch('sys.stdout', self.output), patch('sys.stderr', self.output),
        ]
        for p in self.patches:
            mock = p.start()
            self.addCleanup(p.stop)
            if p.attribute == 'socket':
                mock.return_value.__enter__.return_value.connect_ex.return_value = 1

    def container(self, name, image_id='sha256:old'):
        return {'Id': name + '-' + image_id, 'Name': '/' + name, 'Image': image_id,
                'RestartCount': 0, 'State': {'Running': True, 'Paused': False, 'Restarting': False},
                'Config': {'Image': 'piper:old', 'Env': ['CONFIG_PATH=/work/config.yaml', 'SECRET=not-backed-up'],
                           'Labels': {'com.docker.compose.project': 'old-project',
                                      'com.docker.compose.project.config_files': str(self.old_compose)}},
                'HostConfig': {'RestartPolicy': {'Name': 'always', 'MaximumRetryCount': 0}}, 'Mounts': []}

    def docker(self, *args, data=None):
        self.calls.append(args)
        if args[0] == 'ps':
            return '\n'.join(self.items)
        if args[0] == 'inspect':
            return json.dumps([self.items[args[1]]])
        if args[:2] == ('image', 'inspect'):
            return 'sha256:new\n'
        if args[0] == 'run':
            self.assertIn('--read-only', args)
            self.assertIn('none', args)
            self.assertNotIn('--privileged', args)
            self.assertNotIn('-v', args)
            return json.dumps(yaml.safe_load(data) if data else self.template)
        if args[0] == 'compose':
            if 'config' in args:
                return json.dumps(yaml.safe_load(data))
            self.assertEqual(args[-4:], ('up', '-d', '--no-deps', deploy.SERVICE))
            self.items[deploy.PRIMARY] = self.container(deploy.PRIMARY, 'sha256:new')
            return ''
        if args[0] == 'update':
            self.assertEqual(args[1], '--restart=no')
            self.items[args[2]]['HostConfig']['RestartPolicy']['Name'] = 'no'
            return ''
        if args[0] == 'stop':
            self.items[args[-1]]['State']['Running'] = False
            return ''
        if args[0] == 'rm':
            self.assertFalse(self.items[args[1]]['State']['Running'])
            self.items.pop(args[1])
            return ''
        if args[0] == 'logs':
            return 'new container failed\n'
        raise AssertionError(args)

    def request(self, url, token='', payload=None):
        if url.startswith(deploy.CORE):
            self.assertEqual(token, 'do-not-print-this-token')
            if url.endswith('/project-running'):
                return {'running': self.project_running}
            if url.endswith('/edit-status'):
                return {'code': 200, 'editor': self.editor}
            if url.endswith('/api/mcp'):
                return {'data': [{'server_name': 'agilex-piper', 'url': 'http://localhost:15741/mcp',
                                  'tools': [{'name': n} for n in deploy.CARDS]}]}
        port = int(url.split(':')[2].split('/')[0])
        dual = self.items[deploy.PRIMARY]['Image'] == 'sha256:new' and port == 15741
        if payload['method'] == 'tools/list':
            names = ['piper_status', *deploy.CARDS] if dual else ['arm_control' if port == 15741 else 'piper_status']
            return {'result': {'tools': [{'name': n} for n in names]}}
        self.assertEqual(payload['params']['arguments'], {'action': 'info'})
        if payload['params']['name'] == 'piper_teleop':
            state = {'state': 'idle', 'output': False, 'recording': None}
        else:
            state = {'busy': self.driver_busy, 'motion_enabled': self.motion_enabled, 'active_card': None}
            if dual:
                state = {'arms': {'left': state, 'right': state}}
        return {'result': {'content': [{'type': 'text', 'text': json.dumps(state)}]}}

    def perform(self, answers=('can0', 'can1', 'yes')):
        with patch('builtins.input', side_effect=answers) as question:
            deploy.deploy('piper:new', self.fragment, 'pr-image-agilex-piper')
        return question

    def mutations(self):
        return [a for a in self.calls if a[0] in {'stop', 'rm', 'update'} or a[0] == 'compose' and 'up' in a]

    def test_first_run_persists_binding_replaces_only_piper_and_checks_registration(self):
        unrelated = {n: copy.deepcopy(self.items[n]) for n in ['embodied-pico', 'agent-core']}
        self.perform(('can1', 'can0', 'yes'))
        saved = yaml.safe_load(self.config.read_text())
        self.assertEqual(saved['arms']['left']['can_interface'], 'can1')
        self.assertEqual(saved['arms']['right']['can_interface'], 'can0')
        compose = json.loads((self.run_dir / 'docker-compose.yml').read_text())
        self.assertEqual(compose['services'][deploy.SERVICE]['environment'], {
            'CONFIG_PATH': str(self.config), 'PIPER_MOTION_ENABLED': '0', 'PIPER_CAMERA_ENABLED': '0'})
        self.assertFalse(self.items[deploy.SECONDARY]['State']['Running'])
        self.assertEqual(self.items[deploy.SECONDARY]['HostConfig']['RestartPolicy']['Name'], 'no')
        for name, item in unrelated.items():
            self.assertEqual(item, self.items[name])
        backup, = (self.data / 'deploy-backups').iterdir()
        previous = json.loads((backup / 'previous.json').read_text())
        self.assertEqual(set(previous['containers']), set(deploy.TARGETS))
        self.assertNotIn('SECRET', (backup / 'previous.json').read_text())
        self.assertIn(str(self.old_compose), previous['files'])
        self.assertEqual((backup / previous['files'][str(self.old_compose)]).read_text(), self.old_compose.read_text())
        self.assertEqual((backup / 'previous.json').stat().st_mode & 0o777, 0o600)
        self.assertTrue((backup / 'result.json').is_file())
        self.assertNotIn('do-not-print-this-token', self.output.getvalue())

    def test_repeat_run_reuses_config_without_prompt_or_changing_bytes(self):
        self.perform()
        saved = self.config.read_bytes()
        first_backups = list((self.data / 'deploy-backups').iterdir())
        with patch.object(deploy.sys.stdin, 'isatty', return_value=False):
            question = self.perform(())
        question.assert_not_called()
        self.assertEqual(self.config.read_bytes(), saved)
        self.assertEqual(len(list((self.data / 'deploy-backups').iterdir())), len(first_backups) + 1)

    def test_missing_or_duplicate_interface_never_stops_services(self):
        for answers in [('can0', 'can0'), ('can0', 'can9')]:
            with self.subTest(answers=answers), self.assertRaises(ValueError):
                self.perform(answers)
        self.assertEqual(self.mutations(), [])
        self.assertFalse(self.config.exists())

    def test_noninteractive_first_run_and_cancel_do_not_persist_or_stop(self):
        with patch.object(deploy.sys.stdin, 'isatty', return_value=False), self.assertRaisesRegex(RuntimeError, '终端'):
            self.perform(())
        with self.assertRaisesRegex(RuntimeError, '取消'):
            self.perform(('can0', 'can1', 'no'))
        self.assertEqual(self.mutations(), [])
        self.assertFalse(self.config.exists())

    def test_saved_missing_interface_is_not_silently_reassigned(self):
        self.template['arms']['left']['can_interface'] = 'can9'
        self.config.write_text(yaml.safe_dump(self.template))
        with self.assertRaisesRegex(ValueError, 'can9'):
            self.perform(())
        self.assertEqual(self.mutations(), [])

    def test_empty_saved_configuration_does_not_accept_default_mapping(self):
        self.config.write_text('')
        with self.assertRaisesRegex(ValueError, '为空'):
            self.perform(())
        self.assertEqual(self.mutations(), [])

    def test_project_editor_driver_and_permission_blocks_leave_all_services_running(self):
        for field, value in [('project_running', True), ('editor', 'other-user'),
                             ('driver_busy', True), ('motion_enabled', True)]:
            with self.subTest(field=field):
                old = getattr(self, field)
                setattr(self, field, value)
                with self.assertRaises(RuntimeError):
                    self.perform()
                setattr(self, field, old)
                self.assertEqual(self.mutations(), [])
                self.assertTrue(all(i['State']['Running'] for i in self.items.values()))

    def test_status_query_failure_never_stops_services(self):
        with patch.object(deploy, 'request', side_effect=OSError('unavailable')), self.assertRaises(OSError):
            self.perform()
        self.assertEqual(self.mutations(), [])

    def test_missing_idle_fields_fail_closed(self):
        with patch.object(deploy, 'info', return_value={'state': 'ready'}), self.assertRaises(RuntimeError):
            self.perform()
        self.assertEqual(self.mutations(), [])

    def test_failed_health_stops_new_container_and_preserves_recovery(self):
        with patch.object(deploy, 'health', side_effect=RuntimeError('missing cards')):
            with self.assertRaisesRegex(RuntimeError, 'missing cards'):
                self.perform()
        self.assertFalse(self.items[deploy.PRIMARY]['State']['Running'])
        self.assertFalse(self.items[deploy.SECONDARY]['State']['Running'])
        self.assertEqual(self.items[deploy.PRIMARY]['HostConfig']['RestartPolicy']['Name'], 'no')
        backup, = (self.data / 'deploy-backups').iterdir()
        self.assertIn('new container failed', (backup / 'failed.log').read_text())
        self.assertTrue((backup / 'previous.json').exists())
        self.assertFalse((backup / 'result.json').exists())

    def test_live_camera_in_use_blocks_switch(self):
        tools = {'tools': [{'name': 'piper_status'}, {'name': 'piper_camera_live'}]}
        states = [{'busy': False, 'motion_enabled': False}, {'state': 'running', 'publishing': True}]
        with patch.object(deploy, 'rpc', return_value=tools), patch.object(deploy, 'info', side_effect=states):
            with self.assertRaisesRegex(RuntimeError, '相机'):
                deploy.driver_idle(15741)
        self.assertEqual(self.mutations(), [])

    def test_old_image_without_template_cannot_stop_services(self):
        with patch.object(deploy, 'yaml_config', side_effect=RuntimeError('missing template')):
            with self.assertRaisesRegex(RuntimeError, 'missing template'):
                self.perform()
        self.assertEqual(self.mutations(), [])

    def test_replaced_container_during_confirmation_does_not_stop_anything(self):
        original = deploy.require_idle

        def changed(token, found):
            original(token, found)
            self.items[deploy.PRIMARY]['Id'] = 'another-deployment'

        # Inspect returns copies, like Docker; a concurrent replacement differs from the snapshot.
        with patch.object(deploy, 'require_idle', side_effect=changed), self.assertRaisesRegex(RuntimeError, '容器已变化'):
            self.perform()
        self.assertEqual(self.mutations(), [])

    def test_backup_failure_happens_before_container_mutations(self):
        with patch.object(deploy, 'backup', side_effect=PermissionError('read only')):
            with self.assertRaises(PermissionError):
                self.perform()
        self.assertEqual(self.mutations(), [])

    def test_missing_core_registration_does_not_report_success(self):
        self.items[deploy.PRIMARY] = self.container(deploy.PRIMARY, 'sha256:new')
        self.items[deploy.SECONDARY]['State']['Running'] = False
        original = self.request

        def without_cards(url, token='', payload=None):
            if url == deploy.CORE + '/api/mcp':
                return {'data': []}
            return original(url, token, payload)

        with patch.object(deploy, 'request', side_effect=without_cards), patch.object(deploy.time, 'monotonic', side_effect=[0, 46]):
            with self.assertRaisesRegex(RuntimeError, 'Core'):
                deploy.health('do-not-print-this-token', 'sha256:new')


class ShellDispatchTests(unittest.TestCase):
    def test_only_piper_runs_dual_setup_and_legacy_compose_is_preserved(self):
        for service in ('agilex-piper', 'pico-driver'):
            with self.subTest(service=service), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                binary = root / 'bin'
                binary.mkdir()
                calls = root / 'calls'
                helper = root / 'helper'
                runs = root / 'runs'
                runs.mkdir()
                (runs / 'docker-compose.yml').write_text('old compose')
                (runs / 'state').write_text('IMAGE=piper:old\nSERVICE=agilex-piper\nCONTAINER=embodied-agilex-piper\n')
                (binary / 'docker').write_text(f'#!{sys.executable}\n' + '''import os, sys, json
from pathlib import Path
with open(os.environ['CALLS'], 'a') as log:
    log.write(json.dumps(sys.argv[1:]) + '\\n')
if sys.argv[1] == 'create': print('temporary-extraction')
if sys.argv[1] == 'cp': Path(sys.argv[-1]).write_text(os.environ['FRAGMENT'])
''')
                (binary / 'python3').write_text(f'#!{sys.executable}\n' + '''import os, sys, json
from pathlib import Path
Path(os.environ['HELPER']).write_text(json.dumps(sys.argv[1:]))
''')
                for path in binary.iterdir():
                    path.chmod(0o755)
                env = {**os.environ, 'PATH': str(binary) + ':' + os.defpath, 'CALLS': str(calls),
                       'HELPER': str(helper), 'PR_IMAGE_DIR': str(runs),
                       'FRAGMENT': f'{service}:\n  image: __IMAGE__\n  container_name: example\n'}
                result = subprocess.run(['bash', str(ROOT / 'deploy/run-pr-image.sh'), 'test:image'],
                                        env=env, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                commands = [json.loads(line) for line in calls.read_text().splitlines()]
                self.assertFalse(any('--remove-orphans' in command for command in commands))
                if service == 'agilex-piper':
                    args = json.loads(helper.read_text())
                    self.assertTrue(args[0].endswith('/piper_dual_deploy.py'))
                    self.assertEqual(args[1:], ['test:image', str(runs / service / 'service.yml'), 'pr-image-agilex-piper'])
                    self.assertEqual((runs / service / 'docker-compose.yml').read_text(), 'old compose')
                    self.assertFalse(any('up' in command for command in commands))
                else:
                    self.assertFalse(helper.exists())
                    self.assertIn(['compose', '-p', 'pr-image-pico-driver', '-f', str(runs / service / 'docker-compose.yml'),
                                   'up', '-d'], commands)


if __name__ == '__main__':
    unittest.main()
