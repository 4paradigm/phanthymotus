"""
test_restart_helper.py — agent-core 升级时 restart helper 失败要看得见。

2026-11-01 thor_pack：helper 的 docker 客户端太旧（API 1.43），新 Docker 直接拒绝，
helper 一秒内退出。它是 remove=True 起的，错误随容器一起消失；本进程没被换掉，
进度窗口停在 90% 不动。
"""

import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

import api.system as system  # noqa: E402

THOR_PACK_LOG = (
    '[restart] service=agent-core image=core:release.261101.6fbc404\n'
    '[restart] starting service via docker compose...\n'
    'Error response from daemon: client version 1.43 is too old. Minimum supported '
    'API version is 1.44, please upgrade your client to a newer version\n')


class _Helper:
    def __init__(self, states, code=1, logs=THOR_PACK_LOG):
        self._states, self.status = list(states), 'running'
        self.attrs = {'State': {'ExitCode': code}}
        self._logs, self.removed = logs, False

    def reload(self):
        if self._states:
            self.status = self._states.pop(0)

    def logs(self, tail=30):
        return self._logs.encode()

    def remove(self, force=False):
        self.removed = True


def _capture(monkeypatch):
    seen = {}
    monkeypatch.setattr(system, '_set_error',
                        lambda msg, suggestion='': seen.update(msg=msg, suggestion=suggestion))
    return seen


def test_a_failed_helper_is_reported_with_its_reason(monkeypatch):
    seen = _capture(monkeypatch)
    helper = _Helper(['running', 'exited'], code=1)
    system._watch_helper(helper, timeout_s=5, poll_s=0)
    assert '升级没有完成' in seen['msg'] and 'code 1' in seen['msg']
    assert 'docker 客户端太旧' in seen['msg']           # 认得的原因说人话
    assert '仍在运行' in seen['suggestion'] and helper.removed


def test_unknown_failure_shows_the_error_line():
    reason = system._helper_reason('[restart] extracting\n[restart] ERROR: /deploy/docker-compose.yml not found in image\n')
    assert reason == '[restart] ERROR: /deploy/docker-compose.yml not found in image'


def test_success_that_left_us_running_is_still_an_error(monkeypatch):
    seen = _capture(monkeypatch)
    system._watch_helper(_Helper(['exited'], code=0), timeout_s=5, poll_s=0)
    assert '没有被替换' in seen['msg']


def test_a_helper_that_never_finishes_times_out(monkeypatch):
    seen = _capture(monkeypatch)
    system._watch_helper(_Helper(['running'] * 1000), timeout_s=0.05, poll_s=0.01)
    assert '没有完成' in seen['msg']


def test_helper_gone_means_the_upgrade_finished(monkeypatch):
    seen = _capture(monkeypatch)

    class _Gone(_Helper):
        def reload(self):
            raise RuntimeError('No such container')
    system._watch_helper(_Gone([]), timeout_s=5, poll_s=0)
    assert seen == {}


def test_startup_cleanup_removes_the_finished_helper():
    helper = _Helper([], code=0)
    helper.status = 'exited'

    class _Client:
        class containers:
            @staticmethod
            def get(name):
                assert name == system.HELPER_NAME
                return helper
    system.cleanup_restart_helper(_Client, wait_s=0)
    assert helper.removed


def test_helper_is_kept_after_exit_so_its_error_can_be_read():
    src = (pathlib.Path(__file__).resolve().parents[1] / 'src' / 'api' / 'system.py').read_text()
    run = src[src.index('helper = client.containers.run('):]
    run = run[:run.index(')\n')]
    assert 'remove=False' in run and 'name=HELPER_NAME' in run
