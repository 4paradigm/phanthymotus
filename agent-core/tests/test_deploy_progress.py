#!/usr/bin/env python3
"""
test_deploy_progress.py — preflight checks, the progress stream, pull-error text.

This was a hand-run demo script that printed its way through the three modules
and asserted nothing, so it could only ever fail by raising. It reported green
while calling `progress.update('com配置…')` with the required `message`
argument missing, and it decided its own result by opening a TCP connection to
docker.io and shelling out to the docker daemon — neither of which is present
on a build host, and neither of which says anything about this code.

Rewritten as real assertions against mocked boundaries. The three things worth
pinning, in the order a deployment meets them:

1. preflight classifies disk, network, registry, container and image size, and
   the overall verdict is the worst of them — `fail` is the only one that
   blocks.
2. `DeployProgress` writes every event into the run's replay buffer tagged with
   the run id, which is what lets a browser that connected late still see the
   deploy it asked for.

(`_explain_pull_error` is already covered by test_deploy_pull_errors.py and is
not repeated here.)

Also note it is a `IsolatedAsyncioTestCase`, not a bare `async def`: the suite
runs under `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` (a broken global `fugue_test`
plugin aborts collection otherwise), and that flag also suppresses
pytest-asyncio — so a module-level `async def test_` is a guaranteed red that
no change can fix. Every other async test in this suite uses the unittest base
class for the same reason.
"""

import asyncio
import json
import socket
import unittest
from unittest import mock

# `src` is on sys.path, and `docker` is the real SDK or conftest's stub.
import docker as docker_sdk

from api import deploy_stream
from api.deploy_stream import DeployProgress
from api.preflight import (check_disk_space, check_existing_container,
                           check_network, check_registry_auth,
                           estimate_image_size, run_preflight_checks)

PERCEPTION_IMAGE = 'ccr.ccs.tencentyun.com/phanthy-motus/perception:latest'


def _disk(free_gb, total_gb=100.0):
    """A shutil.disk_usage result with the free space the test cares about."""
    gib = 1 << 30
    return mock.Mock(free=int(free_gb * gib), total=int(total_gb * gib),
                     used=int((total_gb - free_gb) * gib))


class DiskSpaceTest(unittest.TestCase):
    """Only the absolute floor blocks; the comfort margin is a warning."""

    def test_plenty_of_space_passes(self):
        with mock.patch('api.preflight.shutil.disk_usage', return_value=_disk(50)):
            result = check_disk_space(required_gb=20.0)
        self.assertEqual(result['status'], 'pass')
        self.assertEqual(result['free_gb'], 50.0)

    def test_below_the_margin_warns_but_does_not_block(self):
        with mock.patch('api.preflight.shutil.disk_usage', return_value=_disk(5)):
            result = check_disk_space(required_gb=20.0, min_gb=1.0)
        self.assertEqual(result['status'], 'warning')
        self.assertIn('suggestion', result)

    def test_below_the_floor_fails(self):
        with mock.patch('api.preflight.shutil.disk_usage', return_value=_disk(0.5)):
            result = check_disk_space(required_gb=20.0, min_gb=1.0)
        self.assertEqual(result['status'], 'fail')
        self.assertIn('prune', result['suggestion'])

    def test_an_unreadable_filesystem_warns_rather_than_crashing(self):
        with mock.patch('api.preflight.shutil.disk_usage', side_effect=OSError('nope')):
            result = check_disk_space()
        self.assertEqual(result['status'], 'warning')


class NetworkTest(unittest.TestCase):
    """Latency buckets, and the two failures worth telling apart."""

    def _check(self, **kwargs):
        with mock.patch('socket.create_connection', **kwargs) as conn:
            conn.return_value = mock.Mock()
            return check_network('registry.example.com')

    def test_a_fast_connection_passes_and_reports_latency(self):
        result = self._check()
        self.assertEqual(result['status'], 'pass')
        self.assertIsNotNone(result['latency_ms'])

    def test_a_timeout_and_a_dns_failure_are_distinguished(self):
        timed_out = self._check(side_effect=socket.timeout())
        unresolved = self._check(side_effect=socket.gaierror())
        self.assertEqual(timed_out['status'], 'fail')
        self.assertEqual(unresolved['status'], 'fail')
        # The suggestions differ because the operator's next move does.
        self.assertNotEqual(timed_out['suggestion'], unresolved['suggestion'])
        self.assertIn('DNS', unresolved['suggestion'])

    def test_the_registry_argument_is_the_host_dialled(self):
        with mock.patch('socket.create_connection') as conn:
            check_network('registry.example.com')
        conn.assert_called_once()
        self.assertEqual(conn.call_args[0][0], ('registry.example.com', 443))


class DockerBackedChecksTest(unittest.TestCase):
    """container / registry / image-size, with the daemon mocked out."""

    def setUp(self):
        self.client = mock.Mock()
        patcher = mock.patch('api.preflight.docker_sdk.from_env', return_value=self.client)
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_an_existing_container_warns_and_reports_its_status(self):
        self.client.containers.get.return_value = mock.Mock(status='running')
        result = check_existing_container('embodied-perception')
        self.assertEqual(result['status'], 'warning')
        self.assertTrue(result['exists'])
        self.assertEqual(result['container_status'], 'running')

    def test_a_free_container_name_passes(self):
        self.client.containers.get.side_effect = docker_sdk.errors.NotFound('missing')
        result = check_existing_container('embodied-perception')
        self.assertEqual(result['status'], 'pass')
        self.assertFalse(result['exists'])

    def test_an_unreachable_daemon_warns_rather_than_blocking_the_deploy(self):
        self.client.containers.get.side_effect = RuntimeError('daemon is down')
        self.assertEqual(check_existing_container('x')['status'], 'warning')

    def test_a_local_image_reports_its_real_size(self):
        self.client.images.get.return_value = mock.Mock(attrs={'Size': 18 * (1 << 30)})
        self.assertEqual(estimate_image_size(PERCEPTION_IMAGE)['size_gb'], 18.0)

    def test_an_absent_perception_image_falls_back_to_the_known_ballpark(self):
        self.client.images.get.side_effect = docker_sdk.errors.ImageNotFound('missing')
        self.assertEqual(estimate_image_size(PERCEPTION_IMAGE)['size_gb'], 18.0)

    def test_an_absent_unrecognised_image_admits_it_does_not_know(self):
        """A guess here would be multiplied by 2.5 into a disk requirement."""
        self.client.images.get.side_effect = docker_sdk.errors.ImageNotFound('missing')
        self.assertIsNone(estimate_image_size('example.com/ns/mystery:latest')['size_gb'])

    def test_registry_auth_reports_the_registry_it_parsed(self):
        self.assertEqual(check_registry_auth(PERCEPTION_IMAGE)['registry'],
                         'ccr.ccs.tencentyun.com')
        self.assertEqual(check_registry_auth('library/nginx:latest')['registry'],
                         'docker.io')


class PreflightAggregateTest(unittest.TestCase):
    """The overall verdict, which is the only part a caller branches on."""

    def _run(self, disk_status):
        free = {'pass': 200.0, 'warning': 5.0, 'fail': 0.5}[disk_status]
        client = mock.Mock()
        client.images.get.return_value = mock.Mock(attrs={'Size': 18 * (1 << 30)})
        client.containers.get.side_effect = docker_sdk.errors.NotFound('missing')
        with mock.patch('api.preflight.docker_sdk.from_env', return_value=client), \
                mock.patch('api.preflight.shutil.disk_usage', return_value=_disk(free, 400.0)), \
                mock.patch('socket.create_connection'):
            return run_preflight_checks(PERCEPTION_IMAGE, 'embodied-perception')

    def test_every_check_is_reported_even_when_all_pass(self):
        report = self._run('pass')
        self.assertEqual(set(report['checks']),
                         {'image_size', 'disk', 'network', 'registry', 'container'})
        self.assertEqual(report['overall_status'], 'pass')
        self.assertTrue(report['can_proceed'])

    def test_a_warning_still_proceeds_and_surfaces_its_suggestion(self):
        report = self._run('warning')
        self.assertEqual(report['overall_status'], 'warning')
        self.assertTrue(report['can_proceed'])
        self.assertTrue(report['recommendations'])

    def test_a_single_failure_blocks_the_deploy(self):
        report = self._run('fail')
        self.assertEqual(report['overall_status'], 'fail')
        self.assertFalse(report['can_proceed'])

    def test_the_disk_requirement_is_derived_from_the_image_size(self):
        """2.5x the image, so an 18 GB perception image asks for 45 GB."""
        client = mock.Mock()
        client.images.get.return_value = mock.Mock(attrs={'Size': 18 * (1 << 30)})
        client.containers.get.side_effect = docker_sdk.errors.NotFound('missing')
        with mock.patch('api.preflight.docker_sdk.from_env', return_value=client), \
                mock.patch('api.preflight.check_disk_space') as disk, \
                mock.patch('socket.create_connection'):
            disk.return_value = {'status': 'pass'}
            run_preflight_checks(PERCEPTION_IMAGE, 'embodied-perception')
        self.assertAlmostEqual(disk.call_args[0][0], 45.0)


class ProgressStreamTest(unittest.IsolatedAsyncioTestCase):
    """Events reach the replay buffer, tagged with the run that produced them."""

    DRIVER = 'test-driver'

    def setUp(self):
        deploy_stream._runs.pop(self.DRIVER, None)
        deploy_stream._streams.pop(self.DRIVER, None)
        self.addCleanup(deploy_stream._runs.pop, self.DRIVER, None)
        self.addCleanup(deploy_stream._streams.pop, self.DRIVER, None)

    def events(self):
        return [json.loads(e) for e in deploy_stream._runs[self.DRIVER]['events']]

    async def test_a_deployment_writes_start_checks_progress_and_done(self):
        async with DeployProgress(self.DRIVER, PERCEPTION_IMAGE) as progress:
            await progress.check('disk', '检查磁盘空间…', status='pass')
            await progress.update('pull', '拉取镜像层…', percent=45.06, speed='2.3MB/s')
            await progress.done('部署完成')

        events = self.events()
        self.assertEqual([e['type'] for e in events],
                         ['start', 'check', 'progress', 'done'])
        self.assertEqual(events[1]['check_id'], 'disk')
        self.assertEqual(events[1]['status'], 'pass')
        # percent is rounded for the bar; speed rides along as free-form extra.
        self.assertEqual(events[2]['percent'], 45.1)
        self.assertEqual(events[2]['speed'], '2.3MB/s')

    async def test_every_event_carries_the_run_id_and_an_elapsed_time(self):
        """Replay is filtered on run_id — an untagged event is unshowable."""
        async with DeployProgress(self.DRIVER) as progress:
            await progress.done()
        run_id = deploy_stream._runs[self.DRIVER]['run_id']
        for event in self.events():
            self.assertEqual(event['run_id'], run_id)
            self.assertIn('elapsed', event)
            self.assertIn('ts', event)

    async def test_an_error_carries_its_suggestion(self):
        async with DeployProgress(self.DRIVER) as progress:
            await progress.error('disk', '磁盘空间不足', suggestion='docker image prune -a')
        error = self.events()[1]
        self.assertEqual(error['type'], 'error')
        self.assertEqual(error['error_type'], 'disk')
        self.assertEqual(error['suggestion'], 'docker image prune -a')

    async def test_an_exception_inside_the_block_is_reported_and_re_raised(self):
        """Swallowing it would leave the browser on a bar that never moves."""
        with self.assertRaises(ValueError):
            async with DeployProgress(self.DRIVER):
                raise ValueError('boom')
        last = self.events()[-1]
        self.assertEqual(last['type'], 'error')
        self.assertIn('boom', last['message'])

    async def test_the_run_is_closed_on_the_way_out(self):
        async with DeployProgress(self.DRIVER) as progress:
            self.assertTrue(deploy_stream._runs[self.DRIVER]['active'])
            await progress.done()
        self.assertFalse(deploy_stream._runs[self.DRIVER]['active'])

    async def test_a_live_listener_receives_the_same_events(self):
        queue: asyncio.Queue = asyncio.Queue()
        deploy_stream._streams.setdefault(self.DRIVER, set()).add(queue)
        async with DeployProgress(self.DRIVER) as progress:
            await progress.done()
        delivered = [json.loads(queue.get_nowait()[1])['type']
                     for _ in range(queue.qsize())]
        self.assertEqual(delivered, ['start', 'done'])


if __name__ == '__main__':
    unittest.main()
