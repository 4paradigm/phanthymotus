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


def test_an_access_is_cheap_enough_for_the_event_loop(fresh):
    """Not a microbenchmark for its own sake -- the number is why this exists.

    Generous bound: this runs on CI and on laptops. Before the change a read was
    ~0.29 ms here; the point is that it is now dominated by the query rather
    than by opening a file and re-declaring 17 tables.
    """
    config.main['canvas_layout'] = {'cards': [{'id': f'c{i}'} for i in range(40)]}
    for _ in range(5):
        config.main.get('canvas_layout')          # warm
    started = time.perf_counter()
    for _ in range(200):
        config.main.get('canvas_layout')
    per_read_ms = (time.perf_counter() - started) / 200 * 1000
    assert per_read_ms < 0.12, f'一次读 {per_read_ms:.3f} ms，连接复用没生效？'
