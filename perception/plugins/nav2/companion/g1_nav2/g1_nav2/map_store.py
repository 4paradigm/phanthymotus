"""Atomic, path-safe persistence for G1 Nav2 maps and semantic tags."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil
import stat
import tempfile
import unicodedata
import uuid


SCHEMA_VERSION = 1
MAP_FILES = ("map.yaml", "map.pgm", "map.posegraph", "map.data")
_JSON_LIMIT = 1024 * 1024


class MapStoreError(RuntimeError):
    """A stable persistence error suitable for the JSON command bridge."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class MappingSession:
    map_name: str
    path: Path
    started_at: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _plain_name(value: object, *, field: str, limit: int = 128) -> str:
    if not isinstance(value, str):
        raise MapStoreError("invalid_name", f"{field} must be a string")
    normalized = unicodedata.normalize("NFC", value.strip())
    if not normalized:
        raise MapStoreError("invalid_name", f"{field} is required")
    if len(normalized) > limit:
        raise MapStoreError(
            "invalid_name", f"{field} must not exceed {limit} characters"
        )
    if normalized in {".", ".."} or normalized.startswith("."):
        raise MapStoreError("invalid_name", f"{field} must be a plain visible name")
    if any(character in "/\\\x00" for character in normalized):
        raise MapStoreError("invalid_name", f"{field} must not contain path separators")
    if any(unicodedata.category(character).startswith("C") for character in normalized):
        raise MapStoreError("invalid_name", f"{field} must not contain control characters")
    return normalized


class MapStore:
    """Persist one directory per map below a non-symlink store root."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink() or not self.root.is_dir():
            raise MapStoreError("unsafe_map_store", f"map root is unsafe: {self.root}")
        self._root_resolved = self.root.resolve(strict=True)

    @staticmethod
    def validate_map_name(value: object) -> str:
        return _plain_name(value, field="map_name")

    @staticmethod
    def validate_tag_name(value: object) -> str:
        return _plain_name(value, field="tag name")

    def begin_mapping(self, map_name: object) -> MappingSession:
        name = self.validate_map_name(map_name)
        target = self.root / name
        if target.exists() or target.is_symlink():
            raise MapStoreError("map_exists", f"map already exists: {name}")

        stage = Path(tempfile.mkdtemp(prefix=".mapping-", dir=self.root))
        started_at = _utc_now()
        self._write_json(
            stage / "manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "g1_nav2_map",
                "state": "mapping",
                "map_name": name,
                "created_at": started_at,
                "updated_at": started_at,
            },
        )
        self._write_tags(stage, name, {}, timestamp=started_at)
        return MappingSession(name, stage, started_at)

    def finalize_mapping(self, session: MappingSession) -> dict:
        stage = self._managed_directory(session.path)
        manifest = self._read_json(stage / "manifest.json")
        if (
            manifest.get("state") != "mapping"
            or manifest.get("map_name") != session.map_name
        ):
            raise MapStoreError("invalid_mapping_session", "mapping session metadata changed")

        target = self.root / self.validate_map_name(session.map_name)
        if target.exists() or target.is_symlink():
            raise MapStoreError("map_exists", f"map already exists: {session.map_name}")

        file_info = {}
        for filename in MAP_FILES:
            path = self._regular_file(stage / filename, nonempty=True)
            file_info[filename] = {
                "size": path.stat().st_size,
                "sha256": self._sha256(path),
            }
        self._validate_map_yaml(stage / "map.yaml")

        tags = self._read_tags(stage, session.map_name)
        updated_at = _utc_now()
        self._write_json(
            stage / "manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "g1_nav2_map",
                "state": "ready",
                "map_name": session.map_name,
                "created_at": manifest.get("created_at", session.started_at),
                "updated_at": updated_at,
                "tag_count": len(tags),
                "files": file_info,
            },
        )
        os.replace(stage, target)
        self._fsync_directory(self.root)
        return self.map_summary(session.map_name)

    def list_maps(self) -> list[dict]:
        maps = []
        for child in sorted(self.root.iterdir(), key=lambda path: path.name):
            if child.name.startswith(".") or not child.is_dir() or child.is_symlink():
                continue
            try:
                maps.append(self.map_summary(child.name))
            except MapStoreError as exc:
                maps.append(
                    {
                        "map_name": child.name,
                        "status": "corrupt",
                        "error_code": exc.code,
                        "error": str(exc),
                    }
                )
        return maps

    def map_summary(self, map_name: object) -> dict:
        name = self.validate_map_name(map_name)
        directory = self._ready_map_directory(name)
        manifest = self._read_json(directory / "manifest.json")
        tags = self._read_tags(directory, name)
        return {
            "map_name": name,
            "status": "ready",
            "created_at": manifest.get("created_at"),
            "updated_at": manifest.get("updated_at"),
            "tag_count": len(tags),
            "map_yaml": str(directory / "map.yaml"),
        }

    def map_yaml(self, map_name: object) -> Path:
        name = self.validate_map_name(map_name)
        return self._ready_map_directory(name) / "map.yaml"

    def delete_map(self, map_name: object) -> dict:
        name = self.validate_map_name(map_name)
        directory = self._ready_map_directory(name)
        was_active = self.active_map() == name
        tombstone = self.root / f".deleting-{uuid.uuid4().hex}"
        os.replace(directory, tombstone)
        self._fsync_directory(self.root)
        shutil.rmtree(tombstone)
        if was_active:
            self.clear_active_map()
        return {"map_name": name, "status": "deleted"}

    def put_tag(
        self,
        directory: str | Path,
        map_name: object,
        tag_name: object,
        description: object,
        pose: dict,
    ) -> dict:
        managed = self._managed_directory(Path(directory))
        name = self.validate_map_name(map_name)
        tag = self.validate_tag_name(tag_name)
        if not isinstance(description, str):
            raise MapStoreError("invalid_description", "description must be a string")
        description = description.strip()
        if len(description) > 512:
            raise MapStoreError(
                "invalid_description", "description must not exceed 512 characters"
            )
        normalized_pose = self._pose(pose)
        tags = self._read_tags(managed, name)
        now = _utc_now()
        created_at = tags.get(tag, {}).get("created_at", now)
        tags[tag] = {
            "name": tag,
            "description": description,
            **normalized_pose,
            "created_at": created_at,
            "updated_at": now,
        }
        self._write_tags(managed, name, tags, timestamp=now)
        self._refresh_ready_manifest(managed, len(tags), now)
        return dict(tags[tag])

    def remove_tag(
        self, directory: str | Path, map_name: object, tag_name: object
    ) -> dict:
        managed = self._managed_directory(Path(directory))
        name = self.validate_map_name(map_name)
        tag = self.validate_tag_name(tag_name)
        tags = self._read_tags(managed, name)
        if tag not in tags:
            raise MapStoreError("tag_not_found", f"tag does not exist: {tag}")
        removed = tags.pop(tag)
        now = _utc_now()
        self._write_tags(managed, name, tags, timestamp=now)
        self._refresh_ready_manifest(managed, len(tags), now)
        return {"name": tag, "status": "removed", "pose": self._tag_pose(removed)}

    def list_tags(self, directory: str | Path, map_name: object) -> list[dict]:
        managed = self._managed_directory(Path(directory))
        name = self.validate_map_name(map_name)
        tags = self._read_tags(managed, name)
        return [dict(tags[tag]) for tag in sorted(tags)]

    def get_tag(
        self, directory: str | Path, map_name: object, tag_name: object
    ) -> dict:
        managed = self._managed_directory(Path(directory))
        name = self.validate_map_name(map_name)
        tag = self.validate_tag_name(tag_name)
        tags = self._read_tags(managed, name)
        if tag not in tags:
            raise MapStoreError("tag_not_found", f"tag does not exist: {tag}")
        return dict(tags[tag])

    def set_active_map(self, map_name: object) -> None:
        name = self.validate_map_name(map_name)
        self._ready_map_directory(name)
        self._write_json(
            self.root / ".active-map.json",
            {
                "schema_version": SCHEMA_VERSION,
                "map_name": name,
                "updated_at": _utc_now(),
            },
        )

    def active_map(self) -> str | None:
        path = self.root / ".active-map.json"
        if not path.exists():
            return None
        payload = self._read_json(path)
        name = self.validate_map_name(payload.get("map_name"))
        self._ready_map_directory(name)
        return name

    def clear_active_map(self) -> None:
        path = self.root / ".active-map.json"
        if path.exists():
            if path.is_symlink() or not path.is_file():
                raise MapStoreError("unsafe_map_store", "active-map metadata is unsafe")
            path.unlink()
            self._fsync_directory(self.root)

    def directory_for_map(self, map_name: object) -> Path:
        name = self.validate_map_name(map_name)
        return self._ready_map_directory(name)

    def _ready_map_directory(self, name: str) -> Path:
        directory = self._managed_directory(self.root / name)
        manifest = self._read_json(directory / "manifest.json")
        if manifest.get("state") != "ready" or manifest.get("map_name") != name:
            raise MapStoreError("map_not_ready", f"map is not ready: {name}")
        for filename in MAP_FILES:
            self._regular_file(directory / filename, nonempty=True)
        self._validate_map_yaml(directory / "map.yaml")
        return directory

    def _managed_directory(self, directory: Path) -> Path:
        if directory.is_symlink() or not directory.is_dir():
            raise MapStoreError("map_not_found", f"map directory does not exist: {directory.name}")
        resolved = directory.resolve(strict=True)
        if resolved.parent != self._root_resolved:
            raise MapStoreError("unsafe_map_path", "map directory escaped the map root")
        return directory

    @staticmethod
    def _regular_file(path: Path, *, nonempty: bool) -> Path:
        try:
            info = path.lstat()
        except FileNotFoundError as exc:
            raise MapStoreError("map_incomplete", f"missing map file: {path.name}") from exc
        if not stat.S_ISREG(info.st_mode):
            raise MapStoreError("unsafe_map_file", f"map file is not regular: {path.name}")
        if nonempty and info.st_size <= 0:
            raise MapStoreError("map_incomplete", f"map file is empty: {path.name}")
        return path

    def _read_json(self, path: Path) -> dict:
        regular = self._regular_file(path, nonempty=True)
        if regular.stat().st_size > _JSON_LIMIT:
            raise MapStoreError("invalid_metadata", f"metadata is too large: {path.name}")
        try:
            payload = json.loads(regular.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise MapStoreError("invalid_metadata", f"invalid JSON: {path.name}") from exc
        if not isinstance(payload, dict):
            raise MapStoreError("invalid_metadata", f"metadata must be an object: {path.name}")
        return payload

    def _write_json(self, path: Path, payload: dict) -> None:
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ) + "\n"
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            temporary = None
            self._fsync_directory(path.parent)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _read_tags(self, directory: Path, map_name: str) -> dict[str, dict]:
        path = directory / "tags.json"
        if not path.exists():
            return {}
        payload = self._read_json(path)
        if payload.get("map_name") != map_name or not isinstance(payload.get("tags"), dict):
            raise MapStoreError("invalid_metadata", "tags metadata does not match map")
        tags = payload["tags"]
        for key, value in tags.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                raise MapStoreError("invalid_metadata", "tags metadata is malformed")
            self._pose(value)
        return tags

    def _write_tags(
        self,
        directory: Path,
        map_name: str,
        tags: dict[str, dict],
        *,
        timestamp: str,
    ) -> None:
        self._write_json(
            directory / "tags.json",
            {
                "schema_version": SCHEMA_VERSION,
                "map_name": map_name,
                "updated_at": timestamp,
                "tags": tags,
            },
        )

    def _refresh_ready_manifest(
        self, directory: Path, tag_count: int, updated_at: str
    ) -> None:
        manifest_path = directory / "manifest.json"
        manifest = self._read_json(manifest_path)
        if manifest.get("state") != "ready":
            return
        manifest["updated_at"] = updated_at
        manifest["tag_count"] = tag_count
        self._write_json(manifest_path, manifest)

    @staticmethod
    def _pose(pose: object) -> dict:
        if not isinstance(pose, dict):
            raise MapStoreError("invalid_pose", "pose must be an object")
        result = {}
        for field in ("x", "y", "yaw"):
            value = pose.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise MapStoreError("invalid_pose", f"pose {field} must be a number")
            value = float(value)
            if not (-float("inf") < value < float("inf")):
                raise MapStoreError("invalid_pose", f"pose {field} must be finite")
            result[field] = value
        return result

    @staticmethod
    def _tag_pose(tag: dict) -> dict:
        return {field: float(tag[field]) for field in ("x", "y", "yaw")}

    @staticmethod
    def _validate_map_yaml(path: Path) -> None:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as exc:
            raise MapStoreError("invalid_map_yaml", "map.yaml is unreadable") from exc
        image_values = [
            line.split(":", 1)[1].strip().strip("'\"")
            for line in lines
            if line.lstrip().startswith("image:") and ":" in line
        ]
        if image_values != ["map.pgm"]:
            raise MapStoreError(
                "invalid_map_yaml", "map.yaml must reference the local map.pgm"
            )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(path, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
