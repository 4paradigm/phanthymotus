"""Admission control between teleoperation authority and ordinary MCP commands."""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass

import auth
from teleop import audit

_LOCK_STRIPE_COUNT = 256
_READ_ONLY_TOOL_TYPES = frozenset({'resource', 'sensor'})
_READ_ONLY_ACTIONS = frozenset({'info', 'status'})
_SAFE_AUDIT_IDENTIFIER_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$')


@dataclass(frozen=True, slots=True)
class ToolAccess:
    """A fail-closed interpretation of one MCP tool invocation."""

    read_only: bool
    basis: str


class InvalidAuthorityBinding(RuntimeError):
    """The persisted Core-owned authority mapping is not safe to use."""


def authority_domain_for_target(
    mcp_id: str,
    target: object,
    *,
    targets: object = None,
) -> str:
    """Resolve the trusted physical-robot authority domain for an MCP target.

    A standalone teleop adapter and the robot's actuator MCP may have different
    MCP ids.  Trusted registration may bind both to the same ``robot_id``.  A
    legacy or untrusted record cannot choose another target's authority domain
    and therefore remains isolated under its own MCP id.
    """

    descriptor = target if isinstance(target, Mapping) else {}
    if descriptor.get('authority_binding_error') or (
        descriptor.get('authority_binding_required')
        and not descriptor.get('authority_domain')
    ):
        raise InvalidAuthorityBinding(mcp_id)
    trusted = descriptor.get('trust_state') == 'trusted' or descriptor.get('trusted') is True
    if trusted and not auth.driver_record_credential_available(
        mcp_id,
        descriptor.get('credential_binding'),
    ):
        raise InvalidAuthorityBinding(mcp_id)
    candidate = descriptor.get('authority_domain')
    if candidate is None or candidate == mcp_id:
        return mcp_id
    if (
        not trusted
        or not isinstance(candidate, str)
        or not _SAFE_AUDIT_IDENTIFIER_RE.fullmatch(candidate)
    ):
        raise InvalidAuthorityBinding(mcp_id)
    if targets is not None:
        records = targets if isinstance(targets, list) else []
        roots = [
            item for item in records
            if isinstance(item, Mapping) and item.get('id') == candidate
        ]
        if len(roots) != 1:
            raise InvalidAuthorityBinding(mcp_id)
        root = roots[0]
        root_domain = root.get('authority_domain')
        root_tools = root.get('tools')
        ordinary_actuator = any(
            isinstance(tool, Mapping)
            and tool.get('type') == 'actuator'
            and tool.get('name') != 'teleop_session'
            and 'x-teleop' not in tool
            for tool in (root_tools if isinstance(root_tools, list) else [])
        )
        if (
            root.get('trust_state') != 'trusted'
            or not auth.driver_record_credential_available(
                candidate,
                root.get('credential_binding'),
            )
            or root.get('transport') != 'http'
            or root.get('category') != 'driver'
            or root_domain not in (None, '', candidate)
            or not ordinary_actuator
        ):
            raise InvalidAuthorityBinding(mcp_id)
    return candidate


def authority_domain_for_mcp(mcp_id: str) -> str:
    """Resolve against persisted Core configuration, the authorization source."""

    import config

    services = config.main.get('services', {})
    records = services.get('mcp', []) if isinstance(services, Mapping) else []
    matches = [
        item for item in records
        if isinstance(item, Mapping) and item.get('id') == mcp_id
    ]
    if len(matches) != 1:
        raise InvalidAuthorityBinding(mcp_id)
    return authority_domain_for_target(mcp_id, matches[0], targets=records)


def _audit_identifier(value: object, *, verified: bool) -> str:
    if (
        verified
        and isinstance(value, str)
        and _SAFE_AUDIT_IDENTIFIER_RE.fullmatch(value)
    ):
        return value
    return '<unverified>'


def classify_tool_access(
    *,
    tool_type: object,
    annotations: object,
    action: object,
    action_declared: bool = False,
) -> ToolAccess:
    """Return whether a call is provably observational.

    Tool names are deliberately ignored: several robot Drivers have historical
    ``get_*`` actions that move hardware.  A sensor/resource declaration or the
    MCP readOnlyHint is explicit evidence only for tools without an action
    multiplexer.  Once an action is present, only an exactly declared ``info``
    or ``status`` action is admitted; every other action fails closed.
    """

    if action is not None:
        if action_declared and action in _READ_ONLY_ACTIONS:
            return ToolAccess(True, 'declared_diagnostic_action')
        return ToolAccess(False, 'action_not_proven_read_only')
    if isinstance(annotations, Mapping) and annotations.get('readOnlyHint') is True:
        return ToolAccess(True, 'annotation')
    if isinstance(tool_type, str) and tool_type.lower() in _READ_ONLY_TOOL_TYPES:
        return ToolAccess(True, 'tool_type')
    return ToolAccess(False, 'fail_closed')


@dataclass(frozen=True, slots=True)
class AuthorityClaim:
    robot_id: str
    token: str
    session_id: str
    principal_id: str
    state: str


@dataclass(slots=True)
class _MutableAuthorityClaim:
    robot_id: str
    token: str
    session_id: str = ''
    principal_id: str = ''
    state: str = 'acquiring'
    ready: bool = False

    def public_view(self) -> AuthorityClaim:
        return AuthorityClaim(
            robot_id=self.robot_id,
            token=self.token,
            session_id=self.session_id,
            principal_id=self.principal_id,
            state=self.state,
        )


class AuthorityAlreadyClaimed(RuntimeError):
    def __init__(self, claim: AuthorityClaim):
        self.claim = claim
        super().__init__(f'teleop authority already claimed for {claim.robot_id}')


class CommandDrainTimeout(RuntimeError):
    def __init__(self, robot_id: str):
        self.robot_id = robot_id
        super().__init__(f'ordinary command did not drain for {robot_id}')


class TeleopCommandBlocked(RuntimeError):
    code = 'teleop_command_blocked'
    reason = 'teleop_session_active'

    def __init__(self, claim: AuthorityClaim, *, tool: str, action: str):
        self.claim = claim
        self.tool = tool
        self.action = action
        super().__init__(f'ordinary command blocked by teleop authority for {claim.robot_id}')

    def public_detail(self) -> dict[str, str]:
        return {
            'code': self.code,
            'reason': self.reason,
            'robot_id': self.claim.robot_id,
            'session_id': self.claim.session_id,
            'state': self.claim.state,
        }


class CommandBroker:
    """Serialize authority admission with ordinary mutating MCP calls.

    State is keyed by canonical physical robot id while locks are fixed-size
    stripes, so untrusted/random identifiers cannot grow lock storage.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._conditions: tuple[asyncio.Condition, ...] = ()
        self._claims: dict[str, _MutableAuthorityClaim] = {}
        self._ordinary_inflight: dict[str, int] = {}

    def _ensure_loop(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is loop:
            return
        if self._claims or self._ordinary_inflight:
            raise RuntimeError('command broker loop changed with live admissions')
        self._conditions = tuple(
            asyncio.Condition() for _ in range(_LOCK_STRIPE_COUNT)
        )
        self._loop = loop

    def _condition(self, robot_id: str) -> asyncio.Condition:
        self._ensure_loop()
        return self._conditions[hash(robot_id) % _LOCK_STRIPE_COUNT]

    async def begin_authority(
        self,
        robot_id: str,
        token: str,
        *,
        drain_timeout_seconds: float = 5.0,
    ) -> AuthorityClaim:
        """Block new writes, then wait a bounded time for admitted writes."""

        condition = self._condition(robot_id)
        async with condition:
            existing = self._claims.get(robot_id)
            if existing is not None:
                if existing.token != token:
                    raise AuthorityAlreadyClaimed(existing.public_view())
                try:
                    await asyncio.wait_for(
                        condition.wait_for(
                            lambda: (
                                existing.ready
                                or self._claims.get(robot_id) is not existing
                            ),
                        ),
                        timeout=max(0.0, drain_timeout_seconds),
                    )
                except TimeoutError as error:
                    raise CommandDrainTimeout(robot_id) from error
                if self._claims.get(robot_id) is not existing:
                    raise AuthorityAlreadyClaimed(existing.public_view())
                return existing.public_view()

            claim = _MutableAuthorityClaim(robot_id=robot_id, token=token)
            self._claims[robot_id] = claim
            try:
                await asyncio.wait_for(
                    condition.wait_for(
                        lambda: self._ordinary_inflight.get(robot_id, 0) == 0,
                    ),
                    timeout=max(0.0, drain_timeout_seconds),
                )
            except TimeoutError as error:
                if self._claims.get(robot_id) is claim:
                    self._claims.pop(robot_id, None)
                    condition.notify_all()
                raise CommandDrainTimeout(robot_id) from error
            except BaseException:
                if self._claims.get(robot_id) is claim:
                    self._claims.pop(robot_id, None)
                    condition.notify_all()
                raise
            claim.ready = True
            condition.notify_all()
            return claim.public_view()

    async def update_authority(
        self,
        robot_id: str,
        token: str,
        *,
        session_id: str | None = None,
        principal_id: str | None = None,
        state: str | None = None,
    ) -> AuthorityClaim | None:
        condition = self._condition(robot_id)
        async with condition:
            claim = self._claims.get(robot_id)
            if claim is None or claim.token != token:
                return None
            if session_id is not None:
                claim.session_id = session_id
            if principal_id is not None:
                claim.principal_id = principal_id
            if state is not None:
                claim.state = state
            return claim.public_view()

    async def release_authority(self, robot_id: str, token: str) -> bool:
        condition = self._condition(robot_id)
        async with condition:
            claim = self._claims.get(robot_id)
            if claim is None or claim.token != token:
                return False
            self._claims.pop(robot_id, None)
            condition.notify_all()
            return True

    async def authority_for(self, robot_id: str) -> AuthorityClaim | None:
        condition = self._condition(robot_id)
        async with condition:
            claim = self._claims.get(robot_id)
            return claim.public_view() if claim is not None else None

    async def _finish_ordinary(self, robot_id: str) -> None:
        condition = self._condition(robot_id)
        async with condition:
            remaining = self._ordinary_inflight.get(robot_id, 0) - 1
            if remaining > 0:
                self._ordinary_inflight[robot_id] = remaining
            else:
                self._ordinary_inflight.pop(robot_id, None)
            condition.notify_all()

    @staticmethod
    async def _await_cleanup(task: asyncio.Task) -> None:
        cancelled = False
        while True:
            try:
                await asyncio.shield(task)
                break
            except asyncio.CancelledError:
                cancelled = True
                if task.done():
                    task.result()
                    break
        if cancelled:
            raise asyncio.CancelledError

    @asynccontextmanager
    async def ordinary_command(
        self,
        robot_id: str,
        *,
        read_only: bool,
        source: str,
        tool: str,
        action: str,
        tool_verified: bool = False,
        action_verified: bool = False,
    ) -> AsyncIterator[None]:
        """Admit one ordinary call or reject it before any network write."""

        if read_only:
            yield
            return

        condition = self._condition(robot_id)
        blocked: AuthorityClaim | None = None
        async with condition:
            claim = self._claims.get(robot_id)
            if claim is not None:
                blocked = claim.public_view()
            else:
                self._ordinary_inflight[robot_id] = (
                    self._ordinary_inflight.get(robot_id, 0) + 1
                )

        if blocked is not None:
            try:
                await audit.emit(
                    'teleop.command.blocked',
                    session_id=blocked.session_id,
                    robot_id=blocked.robot_id,
                    principal_id=blocked.principal_id,
                    source=source,
                    decision='blocked',
                    reason=TeleopCommandBlocked.reason,
                    tool=_audit_identifier(tool, verified=tool_verified),
                    action=_audit_identifier(action, verified=action_verified),
                    details={'state': blocked.state},
                )
            except Exception:  # noqa: BLE001 -- observability cannot change admission
                pass
            raise TeleopCommandBlocked(
                blocked,
                tool=_audit_identifier(tool, verified=tool_verified),
                action=_audit_identifier(action, verified=action_verified),
            )

        try:
            yield
        finally:
            cleanup = asyncio.create_task(
                self._finish_ordinary(robot_id),
                name=f'teleop-command-finish-{robot_id}',
            )
            await self._await_cleanup(cleanup)


broker = CommandBroker()


__all__ = [
    'AuthorityAlreadyClaimed',
    'AuthorityClaim',
    'InvalidAuthorityBinding',
    'authority_domain_for_mcp',
    'authority_domain_for_target',
    'CommandBroker',
    'CommandDrainTimeout',
    'TeleopCommandBlocked',
    'ToolAccess',
    'broker',
    'classify_tool_access',
]
