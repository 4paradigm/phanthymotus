"""Regression tests for the host compose file merge in api/drivers.py.

The trap these guard, observed on orin6 2026-09-01: two deploys of one driver
landed in the same second (a retried console click). Both did

    compose = yaml.safe_load(open(f)) or {}      # read
    ...
    with open(f, 'w') as fh: yaml.dump(...)      # truncate at open, flush at close

open(...,'w') truncates when it opens and Python flushes when it closes, with no
truncate in between — so the writer that closed second laid a shorter document
over offset 0 of a longer one and left the longer one's tail behind. The result
parsed nowhere, and `or {}` had already let the loser read the truncated file as
"no other services exist" and drop agent-core from the compose entirely. Every
later retry then died in safe_load before it could write, so the host could not
self-heal.

Run: python3 -m pytest agent-core/tests/test_compose_merge_race.py -q
  or: python3 agent-core/tests/test_compose_merge_race.py
"""
import os
import sys
import tempfile
import threading

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from api.drivers import (  # noqa: E402
    _deploy_slot,
    _merge_service_into_compose,
)

# The exact shape found on orin6: a complete perception service, then the tail of
# another document starting mid-`environment`, then a duplicate service.
CORRUPT_COMPOSE = """\
services:
  perception:
    image: registry/perception:a
    container_name: embodied-perception
    environment:
    - ROS_DOMAIN_ID=42
    restart: unless-stopped
   - ROS_DOMAIN_ID=42
    - RMW_IMPLEMENTATION=rmw_fastrtps_cpp
    restart: unless-stopped
  perception:
    image: registry/perception:a
    restart: unless-stopped
"""

GOOD_COMPOSE = {
    'services': {
        'agent-core': {
            'image': 'registry/core:v1',
            'container_name': 'phanthy-motus-agent-core-1',
            'restart': 'unless-stopped',
        },
        'perception': {
            'image': 'registry/perception:v1',
            'container_name': 'embodied-perception',
            'restart': 'unless-stopped',
        },
    }
}


def _fragment(name, image='registry/x:new'):
    return {name: {'image': image, 'container_name': f'embodied-{name}',
                   'restart': 'unless-stopped'}}


@pytest.fixture
def compose_file():
    with tempfile.TemporaryDirectory() as d:
        yield os.path.join(d, 'docker-compose.yml')


def _write(path, obj):
    with open(path, 'w') as f:
        if isinstance(obj, str):
            f.write(obj)
        else:
            yaml.safe_dump(obj, f, sort_keys=False)


# ── The race itself ────────────────────────────────────────────────────────

def test_concurrent_merges_preserve_every_service(compose_file):
    """N threads merging distinct services at once: all of them survive.

    This is the orin6 scenario with the timing widened. Pre-lock, this left a
    file that yaml.safe_load rejects.
    """
    _write(compose_file, GOOD_COMPOSE)
    names = [f'driver-{i}' for i in range(12)]

    start = threading.Barrier(len(names))
    errors = []

    def worker(name):
        start.wait()
        ok, err = _merge_service_into_compose(compose_file, _fragment(name))
        if not ok:
            errors.append((name, err))

    threads = [threading.Thread(target=worker, args=(n,)) for n in names]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not any(t.is_alive() for t in threads), 'merge deadlocked'
    assert errors == [], f'merges failed: {errors}'

    with open(compose_file) as f:
        result = yaml.safe_load(f)          # would raise pre-fix
    assert set(result['services']) == {'agent-core', 'perception', *names}
    # The pre-existing services kept their definitions, not just their keys.
    assert result['services']['agent-core']['image'] == 'registry/core:v1'


def test_concurrent_reader_never_sees_a_partial_document(compose_file):
    """A reader polling during a write storm only ever observes valid YAML.

    os.replace() is what buys this: readers see the old inode or the new one.
    """
    _write(compose_file, GOOD_COMPOSE)
    stop = threading.Event()
    bad = []

    def reader():
        while not stop.is_set():
            try:
                with open(compose_file) as f:
                    doc = yaml.safe_load(f)
            except FileNotFoundError:
                bad.append('file vanished')
                continue
            except yaml.YAMLError as e:
                bad.append(f'unparseable: {e}')
                continue
            if not doc or 'agent-core' not in (doc.get('services') or {}):
                bad.append(f'agent-core missing from {sorted((doc or {}).get("services") or {})}')

    r = threading.Thread(target=reader, daemon=True)
    r.start()
    try:
        for i in range(40):
            ok, err = _merge_service_into_compose(compose_file, _fragment(f'd{i}'))
            assert ok, err
    finally:
        stop.set()
        r.join(timeout=10)
    assert bad == [], f'reader saw {len(bad)} bad state(s); first: {bad[0]}'


# ── Refusing to guess ──────────────────────────────────────────────────────

def test_refuses_empty_existing_file(compose_file):
    """An existing-but-empty compose is a damaged read, not a fresh install.

    `safe_load(...) or {}` used to turn this into an empty service map, so the
    merge wrote a compose holding only the driver being deployed.
    """
    _write(compose_file, '')
    ok, err = _merge_service_into_compose(compose_file, _fragment('perception'))
    assert not ok
    assert 'empty' in err
    with open(compose_file) as f:
        assert f.read() == '', 'refused merge must not touch the file'


def test_refuses_corrupt_existing_file(compose_file):
    """The orin6 file itself: abort, and leave it for a human to inspect."""
    _write(compose_file, CORRUPT_COMPOSE)
    ok, err = _merge_service_into_compose(compose_file, _fragment('perception'))
    assert not ok
    assert 'not valid YAML' in err
    with open(compose_file) as f:
        assert f.read() == CORRUPT_COMPOSE


def test_refuses_when_services_is_not_a_mapping(compose_file):
    _write(compose_file, {'services': ['perception']})
    ok, err = _merge_service_into_compose(compose_file, _fragment('perception'))
    assert not ok
    assert 'not a mapping' in err


# ── Ordinary behaviour still works ─────────────────────────────────────────

def test_missing_file_is_a_fresh_install(compose_file):
    assert not os.path.exists(compose_file)
    ok, err = _merge_service_into_compose(compose_file, _fragment('perception'))
    assert ok, err
    with open(compose_file) as f:
        assert set(yaml.safe_load(f)['services']) == {'perception'}


def test_merge_replaces_target_and_keeps_the_rest(compose_file):
    _write(compose_file, GOOD_COMPOSE)
    ok, err = _merge_service_into_compose(
        compose_file, _fragment('perception', image='registry/perception:v2'))
    assert ok, err
    with open(compose_file) as f:
        result = yaml.safe_load(f)
    assert result['services']['perception']['image'] == 'registry/perception:v2'
    assert result['services']['agent-core'] == GOOD_COMPOSE['services']['agent-core']


def test_merge_leaves_a_backup(compose_file):
    _write(compose_file, GOOD_COMPOSE)
    ok, err = _merge_service_into_compose(compose_file, _fragment('perception'))
    assert ok, err
    with open(compose_file + '.bak') as f:
        assert yaml.safe_load(f) == GOOD_COMPOSE


def test_no_tmp_file_left_behind(compose_file):
    _write(compose_file, GOOD_COMPOSE)
    ok, err = _merge_service_into_compose(compose_file, _fragment('perception'))
    assert ok, err
    assert not os.path.exists(compose_file + '.tmp')


# ── Per-driver in-flight guard ─────────────────────────────────────────────

def test_deploy_slot_rejects_a_second_concurrent_deploy():
    """Two console clicks on one driver: the second is told to wait.

    The already-running-same-image short circuit does not cover this — a driver
    that is not up yet lets both callers straight through.
    """
    with _deploy_slot('perception') as first:
        assert first is True
        with _deploy_slot('perception') as second:
            assert second is False, 'second concurrent deploy was allowed in'
        # A different driver is unaffected.
        with _deploy_slot('unitree-g1') as other:
            assert other is True


def test_deploy_slot_is_released_after_an_exception():
    class Boom(Exception):
        pass

    try:
        with _deploy_slot('perception'):
            raise Boom
    except Boom:
        pass
    with _deploy_slot('perception') as owned:
        assert owned is True, 'slot leaked after an exception'


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
