"""Public, dependency-free data contracts for the Phanthymotus memory core."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar, cast

from .errors import InvalidMemoryError

MAX_EPOCH_MS = (1 << 63) - 1

_EnumT = TypeVar("_EnumT", bound=Enum)


class ShareMode(str, Enum):
    PRIVATE = "private"
    SHARED = "shared"


class EntryState(str, Enum):
    ACTIVE = "active"
    DELETED = "deleted"


class ChangeKind(str, Enum):
    CREATE = "create"
    UPDATE = "update"
    CORRECT = "correct"
    DELETE = "delete"


def _invalid(field_name: str, requirement: str) -> InvalidMemoryError:
    return InvalidMemoryError(f"{field_name} {requirement}")


def _validate_text_encoding(value: str, field_name: str) -> None:
    if "\0" in value:
        raise _invalid(field_name, "must not contain U+0000")
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise _invalid(field_name, "must not contain a lone surrogate")


def normalize_text(value: object, field_name: str, *, allow_empty: bool = False) -> str:
    """Validate and trim user text while normalizing newlines."""

    if not isinstance(value, str):
        raise _invalid(field_name, "must be a string")
    _validate_text_encoding(value, field_name)
    normalized = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not allow_empty and not normalized:
        raise _invalid(field_name, "must be a non-empty string")
    return normalized


def normalize_identifier(value: object, field_name: str) -> str:
    """Normalize a required opaque identifier."""

    return normalize_text(value, field_name)


def normalize_optional_text(value: object, field_name: str) -> str:
    """Normalize text for a field whose empty string has meaning."""

    return normalize_text(value, field_name, allow_empty=True)


_AUDIT_LABEL_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.:-]{0,63}")


def normalize_audit_label(value: object, *, required: bool) -> str:
    """Validate a short, non-sensitive label suitable for append-only audit data."""

    normalized = normalize_optional_text(value, "reason")
    if not normalized:
        if required:
            raise _invalid("reason", "must be a non-empty audit label")
        return ""
    if _AUDIT_LABEL_PATTERN.fullmatch(normalized) is None:
        raise _invalid(
            "reason",
            "must use at most 64 lowercase a-z, 0-9, _, ., :, or - characters",
        )
    return normalized


def _normalize_text_values(values: object, field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise _invalid(field_name, "must be an iterable of strings")
    normalized_values: list[str] = []
    seen: set[str] = set()
    singular_name = field_name[:-1] if field_name.endswith("s") else "value"
    for value in values:
        normalized = normalize_text(value, singular_name)
        if normalized not in seen:
            seen.add(normalized)
            normalized_values.append(normalized)
    return tuple(normalized_values)


def normalize_tags(values: object) -> tuple[str, ...]:
    """Return normalized, order-preserving, duplicate-free tags."""

    return _normalize_text_values(values, "tags")


def normalize_kinds(values: object) -> tuple[str, ...]:
    """Return normalized, order-preserving, duplicate-free entry kinds."""

    return _normalize_text_values(values, "kinds")


def _normalize_key_set(values: object, field_name: str) -> frozenset[str]:
    return frozenset(_normalize_text_values(values, field_name))


def _validate_json_value(value: object, path: str) -> None:
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, str):
        _validate_text_encoding(value, path)
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _invalid(path, "must not contain NaN or Infinity")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise _invalid(path, "must use string object keys")
            _validate_text_encoding(key, f"{path} key")
            _validate_json_value(item, f"{path}.{key}")
        return
    raise _invalid(path, "must contain only JSON values")


def canonical_json(value: object) -> str:
    """Encode a JSON value deterministically for storage and fingerprints."""

    _validate_json_value(value, "value")
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise InvalidMemoryError("value must be canonically JSON encodable") from exc


def normalize_metadata(value: object) -> dict[str, Any]:
    """Validate metadata as a JSON object and return an independent plain dict."""

    if not isinstance(value, dict):
        raise _invalid("metadata", "must be a JSON object")
    encoded = canonical_json(value)
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):  # Defensive: the root check above guarantees this.
        raise _invalid("metadata", "must be a JSON object")
    return cast(dict[str, Any], decoded)


def validate_epoch_ms(
    value: object,
    field_name: str,
    *,
    allow_none: bool = False,
) -> int | None:
    """Validate a SQLite-safe non-negative epoch-millisecond value."""

    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_EPOCH_MS:
        suffix = " or None" if allow_none else ""
        raise _invalid(
            field_name,
            f"must be an epoch millisecond between 0 and {MAX_EPOCH_MS}{suffix}",
        )
    return value


def validate_positive_int(value: object, field_name: str) -> int:
    """Validate a positive integer while rejecting bool-as-int values."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise _invalid(field_name, "must be a positive integer")
    return value


def validate_limit(value: object) -> int:
    """Validate the common query limit contract."""

    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 100:
        raise _invalid("limit", "must be between 1 and 100")
    return value


def _coerce_enum(value: object, enum_type: type[_EnumT], field_name: str) -> _EnumT:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(repr(member.value) for member in enum_type)
        raise _invalid(field_name, f"must be one of {allowed}") from exc


@dataclass(frozen=True, slots=True)
class AccessContext:
    """Private ownership plus explicit shared-place capabilities."""

    owner_key: str
    actor_key: str
    shared_read_keys: frozenset[str] = field(default_factory=frozenset)
    shared_write_keys: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        read_keys = _normalize_key_set(self.shared_read_keys, "shared_read_keys")
        write_keys = _normalize_key_set(self.shared_write_keys, "shared_write_keys")
        if not write_keys.issubset(read_keys):
            raise _invalid("shared_write_keys", "must be a subset of shared_read_keys")
        object.__setattr__(self, "owner_key", normalize_identifier(self.owner_key, "owner_key"))
        object.__setattr__(self, "actor_key", normalize_identifier(self.actor_key, "actor_key"))
        object.__setattr__(self, "shared_read_keys", read_keys)
        object.__setattr__(self, "shared_write_keys", write_keys)


@dataclass(frozen=True, slots=True)
class MemoryPlace:
    share_mode: ShareMode
    place_key: str = ""

    def __post_init__(self) -> None:
        share_mode = _coerce_enum(self.share_mode, ShareMode, "share_mode")
        place_key = normalize_optional_text(self.place_key, "place_key")
        if share_mode is ShareMode.PRIVATE and place_key:
            raise _invalid("place_key", "must be empty for private memory")
        if share_mode is ShareMode.SHARED and not place_key:
            raise _invalid("place_key", "is required for shared memory")
        object.__setattr__(self, "share_mode", share_mode)
        object.__setattr__(self, "place_key", place_key)

    @classmethod
    def private(cls) -> MemoryPlace:
        return cls(ShareMode.PRIVATE)

    @classmethod
    def shared(cls, place_key: str) -> MemoryPlace:
        return cls(ShareMode.SHARED, place_key)


@dataclass(frozen=True, slots=True)
class MemoryDraft:
    title: str
    body: str
    kind: str = "note"
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "title", normalize_optional_text(self.title, "title"))
        object.__setattr__(self, "body", normalize_text(self.body, "body"))
        object.__setattr__(self, "kind", normalize_text(self.kind, "kind"))
        object.__setattr__(self, "tags", normalize_tags(self.tags))
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class MemoryPatch:
    title: str | None = None
    body: str | None = None
    kind: str | None = None
    tags: tuple[str, ...] | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if all(
            value is None for value in (self.title, self.body, self.kind, self.tags, self.metadata)
        ):
            raise InvalidMemoryError("memory patch must set at least one field")
        if self.title is not None:
            object.__setattr__(self, "title", normalize_optional_text(self.title, "title"))
        if self.body is not None:
            object.__setattr__(self, "body", normalize_text(self.body, "body"))
        if self.kind is not None:
            object.__setattr__(self, "kind", normalize_text(self.kind, "kind"))
        if self.tags is not None:
            object.__setattr__(self, "tags", normalize_tags(self.tags))
        if self.metadata is not None:
            object.__setattr__(self, "metadata", normalize_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class ListQuery:
    limit: int = 100
    kinds: tuple[str, ...] = ()
    include_deleted: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.include_deleted, bool):
            raise _invalid("include_deleted", "must be a bool")
        object.__setattr__(self, "limit", validate_limit(self.limit))
        object.__setattr__(self, "kinds", normalize_kinds(self.kinds))


@dataclass(frozen=True, slots=True)
class SearchQuery:
    text: str
    limit: int = 10
    kinds: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", normalize_text(self.text, "text"))
        object.__setattr__(self, "limit", validate_limit(self.limit))
        object.__setattr__(self, "kinds", normalize_kinds(self.kinds))


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    memory_id: str
    share_mode: ShareMode
    place_key: str
    owner_key: str
    title: str
    body: str
    kind: str
    tags: tuple[str, ...]
    metadata: dict[str, Any]
    revision: int
    state: EntryState
    created_ms: int
    updated_ms: int
    deleted_ms: int | None

    def __post_init__(self) -> None:
        share_mode = _coerce_enum(self.share_mode, ShareMode, "share_mode")
        state = _coerce_enum(self.state, EntryState, "state")
        place_key = normalize_optional_text(self.place_key, "place_key")
        owner_key = normalize_identifier(self.owner_key, "owner_key")
        if not place_key:
            raise _invalid("place_key", "is required for stored memory")
        if share_mode is ShareMode.PRIVATE and place_key != owner_key:
            raise _invalid("place_key", "must equal owner_key for private memory")
        revision = validate_positive_int(self.revision, "revision")
        created_ms = cast(int, validate_epoch_ms(self.created_ms, "created_ms"))
        updated_ms = cast(int, validate_epoch_ms(self.updated_ms, "updated_ms"))
        deleted_ms = validate_epoch_ms(self.deleted_ms, "deleted_ms", allow_none=True)
        if updated_ms < created_ms:
            raise _invalid("updated_ms", "must be greater than or equal to created_ms")
        if state is EntryState.ACTIVE and deleted_ms is not None:
            raise _invalid("deleted_ms", "must be None for active memory")
        if state is EntryState.DELETED and deleted_ms is None:
            raise _invalid("deleted_ms", "is required for deleted memory")
        if deleted_ms is not None and deleted_ms < updated_ms:
            raise _invalid("deleted_ms", "must be greater than or equal to updated_ms")
        object.__setattr__(self, "memory_id", normalize_identifier(self.memory_id, "memory_id"))
        object.__setattr__(self, "share_mode", share_mode)
        object.__setattr__(self, "place_key", place_key)
        object.__setattr__(self, "owner_key", owner_key)
        object.__setattr__(self, "title", normalize_optional_text(self.title, "title"))
        object.__setattr__(self, "body", normalize_text(self.body, "body"))
        object.__setattr__(self, "kind", normalize_text(self.kind, "kind"))
        object.__setattr__(self, "tags", normalize_tags(self.tags))
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata))
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "created_ms", created_ms)
        object.__setattr__(self, "updated_ms", updated_ms)
        object.__setattr__(self, "deleted_ms", deleted_ms)


@dataclass(frozen=True, slots=True)
class MemoryChange:
    change_seq: int
    memory_id: str
    actor_key: str
    change_kind: ChangeKind
    from_revision: int | None
    to_revision: int
    reason: str
    operation_ref: str
    changed_ms: int

    def __post_init__(self) -> None:
        change_seq = validate_positive_int(self.change_seq, "change_seq")
        change_kind = _coerce_enum(self.change_kind, ChangeKind, "change_kind")
        from_revision = self.from_revision
        if from_revision is not None:
            from_revision = validate_positive_int(from_revision, "from_revision")
        to_revision = validate_positive_int(self.to_revision, "to_revision")
        if change_kind is ChangeKind.CREATE:
            if from_revision is not None or to_revision != 1:
                raise InvalidMemoryError(
                    "create changes require from_revision=None and to_revision=1"
                )
        elif from_revision is None or to_revision != from_revision + 1:
            raise InvalidMemoryError("non-create changes require to_revision=from_revision+1")
        object.__setattr__(self, "change_seq", change_seq)
        object.__setattr__(self, "memory_id", normalize_identifier(self.memory_id, "memory_id"))
        object.__setattr__(self, "actor_key", normalize_identifier(self.actor_key, "actor_key"))
        object.__setattr__(self, "change_kind", change_kind)
        object.__setattr__(self, "from_revision", from_revision)
        object.__setattr__(self, "to_revision", to_revision)
        object.__setattr__(
            self,
            "reason",
            normalize_audit_label(
                self.reason,
                required=change_kind in {ChangeKind.CORRECT, ChangeKind.DELETE},
            ),
        )
        object.__setattr__(
            self,
            "operation_ref",
            normalize_identifier(self.operation_ref, "operation_ref"),
        )
        object.__setattr__(
            self,
            "changed_ms",
            cast(int, validate_epoch_ms(self.changed_ms, "changed_ms")),
        )


@dataclass(frozen=True, slots=True)
class WriteResult:
    record: MemoryRecord
    change_seq: int
    replayed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.record, MemoryRecord):
            raise _invalid("record", "must be a MemoryRecord")
        if not isinstance(self.replayed, bool):
            raise _invalid("replayed", "must be a bool")
        object.__setattr__(self, "change_seq", validate_positive_int(self.change_seq, "change_seq"))


@dataclass(frozen=True, slots=True)
class SearchHit:
    record: MemoryRecord
    score: float

    def __post_init__(self) -> None:
        if not isinstance(self.record, MemoryRecord):
            raise _invalid("record", "must be a MemoryRecord")
        if isinstance(self.score, bool):
            raise _invalid("score", "must be a finite number")
        try:
            score = float(self.score)
        except (TypeError, ValueError, OverflowError) as exc:
            raise _invalid("score", "must be a finite number") from exc
        if not math.isfinite(score):
            raise _invalid("score", "must be a finite number")
        object.__setattr__(self, "score", score)
