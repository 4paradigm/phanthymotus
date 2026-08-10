from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

_SECRET_KEY_PARTS = (
    'credential',
    'fence',
    'password',
    'privatekey',
    'secret',
    'token',
)


def _is_secret_key(key: object) -> bool:
    normalized = ''.join(character for character in str(key).lower() if character.isalnum())
    return any(part in normalized for part in _SECRET_KEY_PARTS)


def _public_value(value: Any, *, depth: int = 0) -> Any:
    """Build a JSON-safe value while recursively dropping secret-shaped keys.

    ``ShadowSession.public_dict`` currently constructs a fixed schema, but keeping
    the final redaction recursive makes future nested public fields safe by
    default instead of relying on every caller to remember the fence rule.
    """

    if depth > 6:
        return None
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {
            str(key): _public_value(item, depth=depth + 1)
            for key, item in value.items()
            if not _is_secret_key(key)
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_public_value(item, depth=depth + 1) for item in value]
    return str(value)


@dataclass
class ShadowSession:
    id: str
    robot_id: str
    driver_id: str
    principal_id: str
    boot_id: str
    epoch: int
    capability_digest: str
    client_id: str = field(repr=False)
    fence: str = field(repr=False)
    state: str
    operation_generation: int
    operation_state: str
    created_at: float
    lease_seconds: float
    deadline_monotonic: float = field(repr=False)
    mode: str = 'shadow'
    profile_id: str = 'recording'
    capabilities: dict[str, Any] = field(default_factory=dict)
    effectors: list[str] = field(default_factory=list)
    signaling_audience: str = 'teleop-shadow-rtc'
    live_confirmed: bool = True

    @property
    def fence_token(self) -> str:
        """Compatibility alias for internal driver calls; never serialize it."""

        return self.fence

    @property
    def dry_run_profile(self) -> str:
        """Compatibility alias for older Shadow-only internal callers."""

        return self.profile_id

    def remaining_seconds(self, *, monotonic_now: float | None = None) -> float:
        now = time.monotonic() if monotonic_now is None else monotonic_now
        return max(0.0, self.deadline_monotonic - now)

    def public_dict(
        self,
        *,
        monotonic_now: float | None = None,
        wall_now: float | None = None,
    ) -> dict[str, Any]:
        """Return a browser-safe snapshot with a wall-clock expiry estimate.

        Ownership decisions use only ``deadline_monotonic``.  ``expires_at`` is
        deliberately derived at snapshot time for display and is never read back
        by the session manager.
        """

        remaining = self.remaining_seconds(monotonic_now=monotonic_now)
        wall = time.time() if wall_now is None else wall_now
        snapshot = {
            'id': self.id,
            'robot_id': self.robot_id,
            'driver_id': self.driver_id,
            'principal_id': self.principal_id,
            'boot_id': self.boot_id,
            'epoch': self.epoch,
            'capability_digest': self.capability_digest,
            'state': self.state,
            'operation': {
                'generation': self.operation_generation,
                'state': self.operation_state,
            },
            'mode': self.mode,
            'profile_id': self.profile_id,
            'capabilities': self.capabilities,
            'effectors': self.effectors,
            'live_confirmed': self.live_confirmed,
            'created_at': self.created_at,
            'expires_at': wall + remaining,
            'lease_seconds': self.lease_seconds,
            'remaining_seconds': round(remaining, 3),
        }
        if self.mode == 'shadow':
            snapshot['configured_dry_run_profile'] = self.profile_id
        return _public_value(snapshot)
