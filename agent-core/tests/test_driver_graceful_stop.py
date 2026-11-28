"""Service controls must respect the container's configured stop deadline."""
import asyncio
import io
import os
import sys
import tarfile
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import auth  # noqa: E402
from api import drivers, drivers_async  # noqa: E402


def _archive():
    payload = b'perception:\n  container_name: embodied-perception\n'
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w') as archive:
        info = tarfile.TarInfo('service.yml')
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def test_stop_and_remove_use_container_deadline(monkeypatch):
    container = Mock()
    client = Mock()
    client.containers.get.return_value = container
    monkeypatch.setattr(drivers, '_docker', lambda: client)

    assert drivers._stop_sync('perception')['status'] == 'stopped'
    container.stop.assert_called_once_with()
    container.reset_mock()
    assert drivers._remove_sync('perception')['status'] == 'removed'
    assert container.mock_calls == [call.stop(), call.remove(force=True)]


def test_remove_does_not_force_delete_if_stop_fails(monkeypatch):
    container = Mock()
    container.stop.side_effect = RuntimeError('stop failed')
    client = Mock()
    client.containers.get.return_value = container
    monkeypatch.setattr(drivers, '_docker', lambda: client)
    with pytest.raises(RuntimeError):
        drivers._remove_sync('perception')
    container.remove.assert_not_called()


def test_sync_and_async_upgrade_stop_before_removing(monkeypatch, tmp_path):
    old = Mock(status='running', attrs={'Config': {'Image': 'old'}})
    temporary = Mock()
    temporary.get_archive.return_value = ([_archive()], None)
    client = Mock()
    client.containers.get.return_value = old
    client.containers.create.return_value = temporary
    client.api.pull.return_value = []
    monkeypatch.setenv('COMPOSE_DIR', str(tmp_path))
    monkeypatch.setattr(drivers, '_docker', lambda: client)
    monkeypatch.setattr(drivers_async, '_docker', lambda: client)
    monkeypatch.setattr(drivers, '_merge_service_into_compose', lambda *_: (True, ''))
    monkeypatch.setattr(drivers_async, '_merge_service_into_compose', lambda *_: (True, ''))
    monkeypatch.setattr(drivers, '_log_deploy', lambda *_: None)
    monkeypatch.setattr(drivers_async, '_log_deploy', lambda *_: None)
    calls = []
    def run(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(stdout='', stderr='', returncode=0)
    monkeypatch.setattr('subprocess.run', run)
    monkeypatch.setattr(auth, 'get_token', lambda: '"quoted$token"')
    driver = {'id': 'perception', 'image': 'new', 'container_name': 'embodied-perception'}

    assert drivers._deploy_sync_inner(driver)['status'] == 'starting'
    assert old.mock_calls == [call.stop(), call.remove(force=True)]
    old.reset_mock()
    result = asyncio.run(drivers_async._prepare_compose(driver, 'new', 'embodied-perception', None))
    assert result['success']
    assert old.mock_calls == [call.stop(), call.remove(force=True)]
    assert asyncio.run(drivers_async._start_container('perception', result, None))['success']
    assert [call['env']['ACCESS_TOKEN'] for call in calls] == ['"quoted$token"'] * 2
