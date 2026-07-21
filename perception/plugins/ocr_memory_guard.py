from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


_MIB = 1024 * 1024
_UNLIMITED_CGROUP_THRESHOLD = 1 << 60


@dataclass(frozen=True)
class MemoryGuardConfig:
    enabled: bool = True
    expected_workers: int = 10
    min_decode_mb: int = 8
    headroom_ratio: float = 0.2

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, object] | None
    ) -> MemoryGuardConfig:
        values = values or {}
        config = cls(
            **{
                name: values[name]
                for name in cls.__dataclass_fields__
                if name in values
            }
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.expected_workers <= 0:
            raise ValueError("memory_guard.expected_workers must be positive")
        if self.min_decode_mb <= 0:
            raise ValueError("memory_guard.min_decode_mb must be positive")
        if not 0 < self.headroom_ratio <= 1:
            raise ValueError(
                "memory_guard.headroom_ratio must be greater than 0 and at most 1"
            )


class OCRMemoryGuard:
    def __init__(self, config: MemoryGuardConfig):
        self.config = config

    @staticmethod
    def _read_int(path: str) -> int | None:
        try:
            value = Path(path).read_text(encoding="ascii").strip()
        except OSError:
            return None
        if not value or value == "max":
            return None
        try:
            return int(value)
        except ValueError:
            return None

    def _cgroup_headroom_bytes(self) -> int | None:
        candidates = (
            ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
            (
                "/sys/fs/cgroup/memory/memory.limit_in_bytes",
                "/sys/fs/cgroup/memory/memory.usage_in_bytes",
            ),
        )
        for limit_path, usage_path in candidates:
            limit = self._read_int(limit_path)
            usage = self._read_int(usage_path)
            if limit is None or usage is None:
                continue
            if limit >= _UNLIMITED_CGROUP_THRESHOLD:
                continue
            return max(0, limit - usage)
        return None

    @staticmethod
    def _host_available_bytes() -> int | None:
        try:
            lines = Path("/proc/meminfo").read_text(encoding="ascii").splitlines()
        except OSError:
            return None
        for line in lines:
            if line.startswith("MemAvailable:"):
                try:
                    return int(line.split()[1]) * 1024
                except (IndexError, ValueError):
                    return None
        return None

    def decode_limit_bytes(self, hard_limit_bytes: int) -> int:
        if hard_limit_bytes <= 0:
            raise ValueError("hard decode limit must be positive")
        if not self.config.enabled:
            return hard_limit_bytes

        candidates = [hard_limit_bytes]
        cgroup_headroom = self._cgroup_headroom_bytes()
        if cgroup_headroom is not None:
            candidates.append(
                int(cgroup_headroom * self.config.headroom_ratio)
            )

        host_available = self._host_available_bytes()
        if host_available is not None:
            per_worker = host_available / self.config.expected_workers
            candidates.append(int(per_worker * self.config.headroom_ratio))

        limit = max(0, min(candidates))
        if limit < self.config.min_decode_mb * _MIB:
            return 0
        return limit
