"""Regression tests for pull-failure reporting in api/drivers.py.

The trap, observed on Tianyi 2026-09-07: `/` was 100% full, the image pull
failed with

    mkdir /var/lib/containerd/.../ingest/...: no space left on device

and the console showed

    404 Client Error for http+docker://localhost/v1.55/containers/create:
    Not Found ("No such image: .../perception:release.260907.8e31643-jetson-jp6.1")

which reads like the registry is missing the tag — it was not; the tag was in
TCR the whole time. The cause is that **a streamed docker pull reports failure
as a JSON line carrying `error`, and then ends normally.** It does not raise. So
the `for line in client.api.pull(..., stream=True)` loop completed without an
exception, the deploy fell through to `containers.create()`, and the operator
was shown the symptom instead of the cause.

Run: python3 -m pytest agent-core/tests/test_deploy_pull_errors.py -q
"""
import os
import shutil
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# The docker SDK is present in the image but not on a dev host, and
# _deploy_sync_inner imports it at call time. A stub with the one symbol it
# touches (errors.NotFound) is enough and keeps this test hostable anywhere.
if 'docker' not in sys.modules:
    import types
    _docker = types.ModuleType('docker')
    _errors = types.ModuleType('docker.errors')

    class _NotFound(Exception):
        pass

    _errors.NotFound = _NotFound
    _errors.ImageNotFound = _NotFound
    _docker.errors = _errors
    _docker.from_env = lambda: None
    sys.modules['docker'] = _docker
    sys.modules['docker.errors'] = _errors

from api.drivers import _deploy_sync_inner, _explain_pull_error  # noqa: E402

ENOSPC = (
    'mkdir /var/lib/containerd/io.containerd.content.v1.content/ingest/'
    '5e790ec45d4ce3ea37e7f408cf11eb983cdd4f883a7881865104b722e06a6a62: '
    'no space left on device'
)


# ── the explainer ────────────────────────────────────────────────────────────

def test_enospc_is_named_and_actionable():
    detail = _explain_pull_error(ENOSPC)
    assert '磁盘空间不足' in detail
    assert 'prune' in detail                     # tells them what to do
    assert 'MiB' in detail and 'GiB' in detail   # and how bad it is


def test_disk_quota_is_treated_as_disk_exhaustion():
    assert '磁盘空间不足' in _explain_pull_error('write /x: disk quota exceeded')


def test_no_such_image_points_back_at_the_pull():
    detail = _explain_pull_error(
        '404 Client Error for http+docker://localhost/v1.55/containers/create: '
        'Not Found ("No such image: registry/perception:tag")'
    )
    assert '上一步拉取实际失败' in detail


def test_other_errors_pass_through_unchanged():
    for message in ('unauthorized: authentication required',
                    'manifest unknown', 'connection refused'):
        assert _explain_pull_error(message) == message


def test_explainer_survives_an_unreadable_filesystem(monkeypatch):
    """The free-space figure is a nicety; losing it must not lose the error."""
    def boom(_path):
        raise OSError('stat failed')
    monkeypatch.setattr(shutil, 'disk_usage', boom)
    assert '磁盘空间不足' in _explain_pull_error(ENOSPC)


# ── the pull loop ────────────────────────────────────────────────────────────

class _FakeApi:
    def __init__(self, lines):
        self._lines = lines

    def pull(self, image, stream=True, decode=True):
        return iter(self._lines)


class _FakeContainers:
    def __init__(self):
        self.created = []

    def get(self, name):
        import docker as docker_sdk
        raise docker_sdk.errors.NotFound(name)

    def create(self, image):
        self.created.append(image)
        raise AssertionError(
            'containers.create must not be reached after a failed pull'
        )


class _FakeClient:
    def __init__(self, lines):
        self.api = _FakeApi(lines)
        self.containers = _FakeContainers()


@pytest.fixture
def _quiet_deploy(monkeypatch, tmp_path):
    import api.drivers as drivers
    monkeypatch.setattr(drivers, '_log_deploy', lambda *a, **k: None)
    monkeypatch.setattr(drivers, '_clear_deploy_log', lambda *a, **k: None)
    monkeypatch.setenv('COMPOSE_DIR', str(tmp_path))
    return drivers


def _driver():
    return {'id': 'perception', 'image': 'registry/perception:tag'}


def test_a_streamed_error_line_fails_the_deploy(_quiet_deploy, monkeypatch):
    """The whole point: the stream ends normally, so only the line says so."""
    client = _FakeClient([
        {'status': 'Pulling from phanthy-motus/perception', 'id': 'tag'},
        {'status': 'Downloading', 'id': 'abc123', 'progress': '[==>  ] 1MB/8GB'},
        {'errorDetail': {'message': ENOSPC}, 'error': ENOSPC},
    ])
    monkeypatch.setattr(_quiet_deploy, '_docker', lambda: client)

    result = _deploy_sync_inner(_driver())

    assert result['status'] == 'error'
    assert '磁盘空间不足' in result['error']
    assert 'prune' in result['error']
    assert client.containers.created == [], 'must stop before containers.create'


def test_error_without_errordetail_is_still_caught(_quiet_deploy, monkeypatch):
    client = _FakeClient([
        {'status': 'Pulling'},
        {'error': 'manifest for registry/perception:tag not found'},
    ])
    monkeypatch.setattr(_quiet_deploy, '_docker', lambda: client)

    result = _deploy_sync_inner(_driver())
    assert result['status'] == 'error'
    assert 'manifest' in result['error']


def test_an_error_line_mid_stream_is_not_masked_by_later_progress(
    _quiet_deploy, monkeypatch
):
    """Docker keeps emitting status lines for other layers after one fails."""
    client = _FakeClient([
        {'status': 'Downloading', 'id': 'a'},
        {'errorDetail': {'message': ENOSPC}, 'error': ENOSPC},
        {'status': 'Download complete', 'id': 'b'},
        {'status': 'Extracting', 'id': 'b'},
    ])
    monkeypatch.setattr(_quiet_deploy, '_docker', lambda: client)

    result = _deploy_sync_inner(_driver())
    assert result['status'] == 'error'
    assert '磁盘空间不足' in result['error']


def test_a_clean_pull_proceeds_past_the_check(_quiet_deploy, monkeypatch):
    """Guard against the fix rejecting healthy pulls."""
    class _Containers(_FakeContainers):
        def create(self, image):
            self.created.append(image)
            raise RuntimeError('reached create')

    client = _FakeClient([
        {'status': 'Pulling from phanthy-motus/perception'},
        {'status': 'Already exists', 'id': 'abc'},
        {'status': 'Status: Downloaded newer image for registry/perception:tag'},
    ])
    client.containers = _Containers()
    monkeypatch.setattr(_quiet_deploy, '_docker', lambda: client)

    result = _deploy_sync_inner(_driver())
    # create() was reached and raised; the deploy reports that, not a pull error.
    assert result['status'] == 'error'
    assert '镜像拉取失败' not in result['error']
    assert client.containers.created == ['registry/perception:tag']


def test_create_failure_is_explained_not_raised_raw(_quiet_deploy, monkeypatch):
    """Even with the pull fixed, a missing image must not surface as raw 404."""
    class _Containers(_FakeContainers):
        def create(self, image):
            raise Exception(
                '404 Client Error for http+docker://localhost/v1.55/'
                'containers/create: Not Found ("No such image: '
                'registry/perception:tag")'
            )

    client = _FakeClient([{'status': 'Status: Downloaded newer image'}])
    client.containers = _Containers()
    monkeypatch.setattr(_quiet_deploy, '_docker', lambda: client)

    result = _deploy_sync_inner(_driver())
    assert result['status'] == 'error'
    assert '上一步拉取实际失败' in result['error']
