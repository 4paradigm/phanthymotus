"""ConfigDB must not pay for a connection on every access.

`config.main` is read from 41 `async def` request handlers, so whatever it costs
lands on the event loop next to the ROS2 bridge and in-flight MCP calls. It used
to `sqlite3.connect()` and then re-run all 17 `CREATE TABLE/INDEX IF NOT EXISTS`
statements **per access** -- measured on an SSD, 0.286 ms of which 0.264 ms was
setup and 0.022 ms was the query. One `/execution/read` touches `config.main`
ten times.

Run: cd agent-core && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
         python3 -m pytest tests/test_configdb_connection_reuse.py -q
"""
import os
import pathlib
import sqlite3
import sys
import tempfile
import threading
import time

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'configdb-reuse.db'))

import config  # noqa: E402


@pytest.fixture
def fresh(monkeypatch, tmp_path):
    """A private DB and a cleared thread-local, so each case starts cold."""
    monkeypatch.setattr(config, 'DB_PATH', str(tmp_path / 'data.db'))
    monkeypatch.setattr(config, '_conn_local', threading.local())
    return tmp_path / 'data.db'


def test_the_same_thread_gets_the_same_connection(fresh):
    first = config._get_conn()
    for _ in range(20):
        config.main['k'] = 1
        config.main.get('k')
    assert config._get_conn() is first, '每次访问又开了新连接'


def test_the_schema_is_declared_once_per_connection(fresh, monkeypatch):
    """The 17 DDL statements were 93% of an access. They belong on connect."""
    calls = []
    real = config._ensure_schema
    monkeypatch.setattr(config, '_ensure_schema',
                        lambda conn: (calls.append(1), real(conn))[1])
    for _ in range(25):
        config.main.get('anything')
    assert len(calls) == 1, f'建表跑了 {len(calls)} 次'


def test_each_thread_gets_its_own(fresh):
    """sqlite3 refuses cross-thread use, and the ROS2 bridge, the MCP heartbeat
    and the request handlers are different threads."""
    seen, errors, lock = {}, [], threading.Lock()

    def work(index):
        try:
            conn = config._get_conn()
            for _ in range(40):
                config.main[f'k{index}'] = index
                assert config.main[f'k{index}'] == index
            with lock:
                seen[index] = id(conn)
        except Exception as error:      # noqa: BLE001
            with lock:
                errors.append(repr(error))

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
    assert not errors, errors
    assert len(set(seen.values())) == len(seen), '多个线程共用了一个连接'


def test_repointing_the_db_path_gets_a_new_connection(fresh, tmp_path):
    """A test that moves DB_PATH must not keep reading the previous database."""
    config.main['only-in-first'] = 1
    first = config._get_conn()
    config.DB_PATH = str(tmp_path / 'second.db')
    assert config._get_conn() is not first
    assert config.main.get('only-in-first') is None


def test_a_connection_closed_by_someone_else_is_reopened(fresh):
    """Three modules used to close this connection.

    `perf_log`, `api/inspection` and `api/performance` took `config._get_conn()`
    and called `.close()` on it -- correct while every call returned a fresh
    one, fatal once it is shared: the next read anywhere raised
    "Cannot operate on a closed database" during *import* of `client/`. They no
    longer close it, but no caller should be able to wedge the process, so the
    getter reopens rather than propagating.
    """
    config.main['k'] = 'before'
    config._get_conn().close()
    assert config.main.get('k') == 'before'
    config.main['k'] = 'after'
    assert config.main.get('k') == 'after'


def test_no_module_closes_the_shared_connection():
    """A regression guard with a real history, not a style rule."""
    root = pathlib.Path(__file__).resolve().parents[1] / 'src'
    offenders = []
    for path in root.rglob('*.py'):
        text = path.read_text(encoding='utf-8')
        if 'config._get_conn' not in text and 'from config import' not in text:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if 'conn.close()' in line and not line.strip().startswith('#'):
                offenders.append(f'{path.relative_to(root)}:{number}')
    assert not offenders, f'这些地方关了共享连接：{offenders}'


def test_two_hundred_reads_open_one_connection(fresh, monkeypatch):
    """The property, counted rather than timed.

    This replaced an absolute wall-clock bound (`per_read_ms < 0.12`) that was
    green on a laptop and red in the shipped image: the same code measures
    0.015 ms here and 0.217 ms in the container, a 14x environment difference
    that no fixed threshold survives. It was the same mistake as the
    `mock.patch` target in `test_interrupt_all_fallback` -- a test that only
    holds on the machine it was written on.
    """
    opened = []
    real_connect = sqlite3.connect
    monkeypatch.setattr(sqlite3, 'connect',
                        lambda *a, **k: (opened.append(a[0]), real_connect(*a, **k))[1])
    config.main['canvas_layout'] = {'cards': [{'id': f'c{i}'} for i in range(40)]}
    for _ in range(200):
        config.main.get('canvas_layout')
    assert len(opened) == 1, f'200 次读开了 {len(opened)} 个连接'


def test_reuse_is_measurably_cheaper_than_reopening(fresh):
    """A *ratio* on one machine, so the hardware cancels out.

    Measured 18.7x on a laptop (0.015 ms vs 0.274 ms). The bound is 2x because
    this also runs in a container on shared CPU -- the point is to catch "reuse
    stopped happening", not to pin a number.
    """
    config.main['canvas_layout'] = {'cards': [{'id': f'c{i}'} for i in range(40)]}

    def timed(read, rounds=120):
        for _ in range(5):
            read()                                  # warm
        started = time.perf_counter()
        for _ in range(rounds):
            read()
        return (time.perf_counter() - started) / rounds

    def reopen_every_time():
        config._conn_local = threading.local()      # the old behaviour, forced
        return config.main.get('canvas_layout')

    reused = timed(lambda: config.main.get('canvas_layout'))
    reopened = timed(reopen_every_time)
    assert reopened / reused > 2, (
        f'复用 {reused * 1000:.3f} ms vs 重开 {reopened * 1000:.3f} ms —— '
        f'只快 {reopened / reused:.1f}x，复用没生效？')


def test_connection_level_state_does_not_leak_between_consumers(fresh):
    """The half of reuse that is easy to miss, and the PR bot caught.

    A fresh connection per access meant connection-level state was discarded,
    so callers set it and never restored it: `perf_log` sets
    `row_factory = sqlite3.Row` at four sites; `chat_history` sets it back to
    `None` at two (dead code back then -- which is why it exists at all). Shared,
    whichever ran last on this thread would decide what the *next* reader's rows
    look like.

    The failure would also not have surfaced near the cause:
    `ConfigDB.__getitem__` reads `row[0]`, which `sqlite3.Row` supports, so
    config keeps working and something further away breaks instead.
    """
    config.main['k'] = {'v': 1}

    # A perf_log-style consumer, on the real shared connection.
    borrowed = config._get_conn()
    borrowed.row_factory = sqlite3.Row
    row = borrowed.execute('SELECT key, value FROM config WHERE key = ?', ('k',)).fetchone()
    assert row['key'] == 'k', '这个消费者自己要的 Row 行为得生效'

    # Every later handout is back to the default a fresh connection would have.
    assert config._get_conn().row_factory is None
    assert config.main.get('k') == {'v': 1}

    # And a plain tuple-positional reader -- the shape most of this tree uses --
    # still gets a tuple, not a Row.
    fetched = config._get_conn().execute(
        'SELECT COUNT(*) FROM config').fetchone()
    assert type(fetched) is tuple, f'拿到 {type(fetched).__name__}，row_factory 泄漏了'


def test_the_real_perf_log_consumer_does_not_poison_configdb(fresh):
    """Same property, driven through the actual module rather than a stand-in."""
    import perf_log

    config.main['canvas_layout'] = {'cards': []}
    perf_log.query_latest(1)          # sets row_factory = sqlite3.Row inside
    assert config.main.get('canvas_layout') == {'cards': []}
    assert config._get_conn().row_factory is None
