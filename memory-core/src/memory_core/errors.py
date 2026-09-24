"""Memory Core domain and persistence errors."""


class MemoryCoreError(Exception):
    """Base class for all expected Memory Core failures."""


class InvalidMemoryError(MemoryCoreError, ValueError):
    """A value violates the public Memory Core contract."""


class MemoryNotFoundError(MemoryCoreError, LookupError):
    """A memory does not exist or is not visible to the caller."""


class SharedPlaceDeniedError(MemoryCoreError, PermissionError):
    """The caller lacks the requested capability for a shared place."""


class RevisionConflictError(MemoryCoreError):
    """A write targeted a revision that is no longer current."""


class IdempotencyConflictError(MemoryCoreError):
    """An operation key was reused for a different request."""


class StorageBusyError(MemoryCoreError):
    """Storage could not accept an operation before its busy deadline."""


class StorageDamagedError(MemoryCoreError):
    """Storage is corrupt, malformed, or otherwise unsafe to use."""


class MigrationError(MemoryCoreError):
    """The on-disk schema cannot be safely migrated."""


class SearchUnavailableError(MemoryCoreError):
    """Full-text search is unavailable in the active storage engine."""


class UnsupportedSchemaVersionError(MigrationError):
    """The database schema is newer than this Memory Core build."""
