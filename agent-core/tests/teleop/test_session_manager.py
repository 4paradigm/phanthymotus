from __future__ import annotations

import asyncio
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields

import pytest

from teleop.models import _public_value
from teleop.session_manager import (
    MAX_DRIVER_EPOCH,
    EpochExhausted,
    SessionNotFound,
    SessionStateConflict,
    ShadowSessionManager,
)

CLIENT_ID = '7dbabfca-15c1-43ca-b600-75e7682c21d0'
OTHER_CLIENT_ID = '68991413-d37a-4603-9f07-3e25219d6d96'


class MutableClock:
    def __init__(self, *, monotonic: float = 100.0, wall: float = 1_000.0):
        self.monotonic = monotonic
        self.wall = wall

    def monotonic_now(self) -> float:
        return self.monotonic

    def wall_now(self) -> float:
        return self.wall


def make_manager(clock: MutableClock | None = None) -> ShadowSessionManager:
    clock = clock or MutableClock()
    return ShadowSessionManager(
        monotonic=clock.monotonic_now,
        wall_clock=clock.wall_now,
    )


async def reserve(
    manager: ShadowSessionManager,
    *,
    robot_id: str = 'robot-1',
    driver_id: str = 'driver-1',
    lease_seconds: float = 15.0,
    minimum_epoch: int = 1,
):
    return await manager.reserve(
        robot_id,
        'alice',
        driver_id=driver_id,
        boot_id='driver-boot-1',
        capability_digest='a' * 64,
        client_id=CLIENT_ID,
        lease_seconds=lease_seconds,
        minimum_epoch=minimum_epoch,
    )


def test_shadow_session_repr_and_public_snapshot_never_expose_fence():
    async def scenario():
        manager = make_manager()
        session = await reserve(manager)

        assert next(item for item in fields(session) if item.name == 'fence').repr is False
        assert session.fence not in repr(session)
        assert 'fence' not in repr(session).lower()

        public = manager.public_dict(session)
        assert session.fence not in repr(public)
        assert 'fence' not in repr(public).lower()
        assert 'token' not in repr(public).lower()
        assert public['boot_id'] == 'driver-boot-1'
        assert public['epoch'] == 1
        assert public['capability_digest'] == 'a' * 64
        assert public['configured_dry_run_profile'] == 'recording'
        assert public['operation'] == {'generation': 1, 'state': 'pending'}

    asyncio.run(scenario())


def test_session_pins_adapter_neutral_profile_and_limits_legacy_alias():
    async def scenario():
        manager = make_manager()
        session = await manager.reserve(
            'arm-robot',
            'alice',
            driver_id='arm-teleop',
            boot_id='arm-boot',
            capability_digest='b' * 64,
            client_id=CLIENT_ID,
            profile_id='dual_arm_profile_v1',
        )
        assert session.profile_id == 'dual_arm_profile_v1'
        assert manager.public_dict(session)['profile_id'] == 'dual_arm_profile_v1'

        with pytest.raises(ValueError, match='dry_run_profile'):
            await manager.reserve(
                'live-robot',
                'alice',
                driver_id='live-driver',
                boot_id='live-boot',
                capability_digest='c' * 64,
                client_id=CLIENT_ID,
                dry_run_profile='vendor_live_profile',
            )

        with pytest.raises(ValueError, match='profile_id'):
            await manager.reserve(
                'malformed-robot',
                'alice',
                driver_id='malformed-driver',
                boot_id='malformed-boot',
                capability_digest='d' * 64,
                client_id=CLIENT_ID,
                profile_id=['malformed_profile'],
            )

    asyncio.run(scenario())


def test_public_snapshot_sanitizer_recursively_removes_secret_shaped_keys():
    sanitized = _public_value({
        'safe': {
            'items': [
                {
                    'fence': 'hidden',
                    'accessToken': 'hidden',
                    'private_key': 'hidden',
                    'state': 'active',
                },
            ],
        },
    })
    assert sanitized == {'safe': {'items': [{'state': 'active'}]}}


def test_monotonic_deadline_survives_wall_clock_rollback():
    async def scenario():
        clock = MutableClock()
        manager = make_manager(clock)
        session = await reserve(manager, lease_seconds=15)
        await manager.activate(session.id, session.operation_generation)

        clock.monotonic += 10
        clock.wall = 10  # wall clock rolled back by much more than the lease
        public = manager.public_dict(session)
        assert public['remaining_seconds'] == 5
        assert public['expires_at'] == 15
        assert await manager.active_for_robot('robot-1') is session

        clock.monotonic += 5.001
        clock.wall = -50_000
        assert await manager.active_for_robot('robot-1') is None
        assert session.state == 'expired'
        assert session.operation_state == 'expired'

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ('requested', 'expected'),
    [
        (0, 15.0),
        (-100, 15.0),
        (5, 15.0),
        (15, 15.0),
        (120, 120.0),
        (999, 120.0),
        (math.nan, 15.0),
        (math.inf, 15.0),
    ],
)
def test_ownership_lease_is_bounded(requested, expected):
    async def scenario():
        session = await reserve(make_manager(), lease_seconds=requested)
        assert session.lease_seconds == expected

    asyncio.run(scenario())


def test_driver_epoch_persists_across_manager_instances_and_is_per_driver():
    async def scenario():
        first = await reserve(make_manager(), robot_id='r1', driver_id='driver-a')
        after_restart = await reserve(
            make_manager(),
            robot_id='r2',
            driver_id='driver-a',
        )
        independent = await reserve(
            make_manager(),
            robot_id='r3',
            driver_id='driver-b',
        )
        assert (first.epoch, after_restart.epoch, independent.epoch) == (1, 2, 1)

    asyncio.run(scenario())


def test_driver_reported_epoch_sets_a_persisted_allocation_floor():
    async def scenario():
        recovered = await reserve(
            make_manager(),
            robot_id='r1',
            driver_id='driver-a',
            minimum_epoch=42,
        )
        next_session = await reserve(
            make_manager(),
            robot_id='r2',
            driver_id='driver-a',
            minimum_epoch=3,
        )
        raised_again = await reserve(
            make_manager(),
            robot_id='r3',
            driver_id='driver-a',
            minimum_epoch=100,
        )
        assert (recovered.epoch, next_session.epoch, raised_again.epoch) == (42, 43, 100)

    asyncio.run(scenario())


def test_driver_epoch_exhaustion_is_detected_before_sqlite_overflow():
    async def scenario():
        final = await reserve(
            make_manager(),
            robot_id='last-safe-robot',
            driver_id='exhausted-driver',
            minimum_epoch=MAX_DRIVER_EPOCH,
        )
        assert final.epoch == MAX_DRIVER_EPOCH

        with pytest.raises(EpochExhausted):
            await reserve(
                make_manager(),
                robot_id='must-not-wrap',
                driver_id='exhausted-driver',
            )

        with pytest.raises(EpochExhausted):
            await reserve(
                make_manager(),
                robot_id='out-of-range-floor',
                driver_id='fresh-driver',
                minimum_epoch=MAX_DRIVER_EPOCH + 1,
            )

    asyncio.run(scenario())


def test_driver_epoch_allocation_is_atomic_across_concurrent_managers():
    def reserve_in_thread(index: int) -> int:
        async def scenario():
            session = await reserve(
                make_manager(),
                robot_id=f'robot-{index}',
                driver_id='shared-driver',
            )
            return session.epoch

        return asyncio.run(scenario())

    with ThreadPoolExecutor(max_workers=8) as pool:
        epochs = list(pool.map(reserve_in_thread, range(12)))

    assert sorted(epochs) == list(range(1, 13))


def test_terminal_session_history_is_bounded():
    async def scenario():
        manager = make_manager()
        sessions = []
        for index in range(300):
            session = await reserve(
                manager,
                robot_id=f'robot-{index}',
                driver_id='history-driver',
            )
            await manager.release(
                session.id,
                'alice',
                CLIENT_ID,
            )
            sessions.append(session)

        retained = await manager.retained_session_ids()
        assert len(retained) == 256
        assert sessions[0].id not in retained
        assert await manager.get(sessions[0].id) is None
        assert sessions[-1].id in retained
        assert await manager.get(sessions[-1].id) is sessions[-1]

    asyncio.run(scenario())


def test_delayed_prepare_failure_cannot_fault_an_activated_session():
    async def scenario():
        manager = make_manager()
        session = await reserve(manager)
        generation = session.operation_generation
        await manager.activate(session.id, generation)

        assert await manager.fail_reservation(session.id, generation) is None
        assert session.state == 'active'
        assert session.operation_state == 'succeeded'
        assert await manager.active_for_robot(session.robot_id) is session

    asyncio.run(scenario())


def test_heartbeat_worker_fault_is_generation_guarded_and_conditionally_unbinds():
    async def scenario():
        manager = make_manager()
        session = await reserve(manager)
        await manager.activate(session.id, session.operation_generation)

        assert await manager.fault(session.id, session.operation_generation + 1) is None
        assert await manager.get_current(session.id) is session
        assert await manager.authorize(session.id, 'alice') is session

        faulted = await manager.fault(session.id, session.operation_generation)
        assert faulted is session
        assert session.state == 'faulted'
        assert session.operation_state == 'failed'
        assert await manager.get_current(session.id) is None
        assert await manager.active_for_robot(session.robot_id) is None
        assert await manager.get_authorized(
            session.id,
            'alice',
            include_terminal=True,
        ) is session

    asyncio.run(scenario())


def test_old_async_failure_and_completion_cannot_clear_or_revive_replacement():
    async def scenario():
        clock = MutableClock()
        manager = make_manager(clock)
        old = await reserve(manager, lease_seconds=15)
        old_generation = old.operation_generation
        await manager.release(old.id, 'alice', CLIENT_ID)

        replacement = await reserve(manager, lease_seconds=120)
        assert replacement.operation_generation > old_generation

        assert await manager.fail_reservation(old.id, old_generation) is None
        with pytest.raises(SessionStateConflict):
            await manager.activate(old.id, old_generation)
        with pytest.raises(SessionNotFound):
            await manager.release(old.id, 'alice', CLIENT_ID)
        assert await manager.active_for_robot('robot-1') is replacement

        # Simulate stale mutation from an obsolete callback. Authority expiry
        # scans only the live robot map, so obsolete history is ignored and can
        # neither consume watchdog time nor unbind the replacement.
        old.state = 'active'
        old.operation_state = 'succeeded'
        clock.monotonic += 16
        expired = await manager.expire_due()
        assert expired == []
        assert await manager.active_for_robot('robot-1') is replacement

    asyncio.run(scenario())


def test_activate_rejects_expired_released_and_nonpreparing_sessions():
    async def scenario():
        clock = MutableClock()
        manager = make_manager(clock)

        active = await reserve(manager, robot_id='active')
        await manager.activate(active.id, active.operation_generation)
        with pytest.raises(SessionStateConflict):
            await manager.activate(active.id, active.operation_generation)

        released = await reserve(manager, robot_id='released')
        await manager.release(released.id, 'alice', CLIENT_ID)
        with pytest.raises(SessionStateConflict):
            await manager.activate(released.id, released.operation_generation)

        expiring = await reserve(manager, robot_id='expired', lease_seconds=15)
        clock.monotonic += 16
        with pytest.raises(SessionStateConflict):
            await manager.activate(expiring.id, expiring.operation_generation)

    asyncio.run(scenario())


def test_heartbeat_renews_only_live_owned_session_using_monotonic_time():
    async def scenario():
        clock = MutableClock()
        manager = make_manager(clock)
        session = await reserve(manager, lease_seconds=15)
        with pytest.raises(SessionStateConflict):
            await manager.heartbeat(session.id, 'alice', CLIENT_ID)

        await manager.activate(session.id, session.operation_generation)
        clock.monotonic += 14
        clock.wall -= 100_000
        await manager.heartbeat(session.id, 'alice', CLIENT_ID)
        assert session.deadline_monotonic == 129

        clock.monotonic = 128.9
        assert await manager.active_for_robot(session.robot_id) is session
        clock.monotonic = 129.1
        assert await manager.active_for_robot(session.robot_id) is None

    asyncio.run(scenario())


def test_soft_stop_exposes_idempotent_hold_that_can_heartbeat_and_pause():
    async def scenario():
        clock = MutableClock()
        manager = make_manager(clock)
        session = await reserve(manager, lease_seconds=15)
        await manager.activate(session.id, session.operation_generation)

        assert await manager.soft_stop(session.id, 'alice', CLIENT_ID) is session
        assert session.state == 'hold'
        assert manager.public_dict(session)['state'] == 'hold'
        assert await manager.soft_stop(session.id, 'alice', CLIENT_ID) is session

        clock.monotonic += 4
        await manager.heartbeat(session.id, 'alice', CLIENT_ID)
        assert session.deadline_monotonic == 119
        assert await manager.pause(session.id, 'alice', CLIENT_ID) is session
        assert session.state == 'paused'

    asyncio.run(scenario())


def test_expiry_detected_by_read_is_drained_exactly_once_for_notifications():
    async def scenario():
        clock = MutableClock()
        manager = make_manager(clock)
        session = await reserve(manager, lease_seconds=5)
        await manager.activate(session.id, session.operation_generation)

        clock.monotonic += 16
        # A status/read path notices expiry before the periodic worker.
        assert await manager.get_current(session.id) is None
        assert session.state == 'expired'

        # The worker still receives it once for Driver cleanup and audit.
        assert await manager.expire_due() == [session]
        assert await manager.expire_due() == []
        assert await manager.get(session.id) is session
        assert await manager.expire_due() == []

    asyncio.run(scenario())
