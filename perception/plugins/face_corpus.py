"""Bounded corpus inputs; extracted image paths live only inside the context."""

import gzip
import json
import ntpath
import os
from pathlib import Path
import stat
import tarfile
import tempfile
from contextlib import contextmanager
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, build_opener
import zipfile

import cv2
import numpy as np

MAX_PACKAGE_BYTES = 512 * 1024 * 1024
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_ARCHIVE_ENTRIES = 10000
_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".jpe", ".png", ".bmp", ".webp",
                   ".tif", ".tiff", ".ppm", ".pgm", ".pbm", ".gif",
                   ".ico", ".jfif")
_CHUNK = 64 * 1024


def _relative(name):
    if not isinstance(name, str) or not name or "\x00" in name:
        raise ValueError("invalid package member name")
    name = name.replace("\\", "/")
    if name.startswith("/") or ntpath.splitdrive(name)[0] or ".." in name.split("/"):
        raise ValueError("unsafe package path: %r" % name)
    parts = [part for part in name.split("/") if part not in ("", ".")]
    if not parts:
        raise ValueError("empty package member path")
    return "/".join(parts)


def _local(source):
    try:
        path = Path(os.fsdecode(os.fspath(source))).absolute()
    except (TypeError, ValueError) as error:
        raise ValueError("expected a local path") from error
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError("symlinks are not allowed: %s" % part)
    return path


def _copy(src, dst, limit):
    total = 0
    while True:
        chunk = src.read(min(_CHUNK, limit - total + 1))
        if not chunk:
            return total
        total += len(chunk)
        if total > limit:
            raise ValueError("input exceeds byte limit")
        dst.write(chunk)


def _url(url):
    parsed = urlsplit(url)
    if parsed.scheme.lower() not in ("http", "https") or not parsed.hostname:
        raise ValueError("expected an HTTP(S) URL")
    return url


class _HTTPRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _download(url, destination, limit):
    with build_opener(_HTTPRedirects()).open(_url(url), timeout=30) as response:
        length = response.headers.get("Content-Length")
        if length is not None and (int(length) < 0 or int(length) > limit):
            raise ValueError("download exceeds byte limit")
        with open(destination, "wb") as out:
            _copy(response, out, limit)


def _file_limit(name):
    return MAX_IMAGE_BYTES if name.lower().endswith(_IMAGE_SUFFIXES) else MAX_PACKAGE_BYTES


class _LimitedReader:
    """Count the entire tar stream, including padding and extended headers."""

    def __init__(self, stream):
        self.stream = stream
        self.remaining = MAX_PACKAGE_BYTES

    def read(self, size=-1):
        size = self.remaining + 1 if size < 0 else min(size, self.remaining + 1)
        data = self.stream.read(size)
        self.remaining -= len(data)
        if self.remaining < 0:
            raise ValueError("decompressed package exceeds byte limit")
        return data


def _extract(source, root):
    count, total = 0, 0
    seen = set()

    def member(name, size, directory):
        nonlocal count, total
        count += 1
        if count > MAX_ARCHIVE_ENTRIES:
            raise ValueError("too many archive members")
        # Conventional tar archives may contain an explicit root directory.
        if directory and name in (".", "./"):
            return root
        name = _relative(name)
        if name in seen:
            raise ValueError("duplicate archive path: %s" % name)
        seen.add(name)
        if size < 0 or size > _file_limit(name):
            raise ValueError("archive member exceeds byte limit")
        total += size
        if total > MAX_PACKAGE_BYTES:
            raise ValueError("decompressed package exceeds byte limit")
        target = root / name
        if directory:
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
        return target

    if zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            for info in archive.infolist():
                mode = info.external_attr >> 16
                kind = stat.S_IFMT(mode)
                if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
                    raise ValueError("archive links and special files are forbidden")
                directory = info.is_dir()
                if kind == stat.S_IFDIR and not directory:
                    raise ValueError("invalid zip directory")
                target = member(info.filename, info.file_size, directory)
                if not directory:
                    with archive.open(info) as src, open(target, "xb") as dst:
                        actual = _copy(src, dst, min(info.file_size, _file_limit(info.filename)))
                    if actual != info.file_size:
                        raise ValueError("truncated zip member")
        return
    with open(source, "rb") as raw:
        compressed = raw.read(2) == b"\x1f\x8b"
        raw.seek(0)
        stream = gzip.GzipFile(fileobj=raw) if compressed else raw
        try:
            bounded = _LimitedReader(stream)
            with tarfile.open(fileobj=bounded, mode="r|") as archive:
                for info in archive:
                    if not (info.isfile() or info.isdir()) or info.issparse():
                        raise ValueError("archive links and special files are forbidden")
                    target = member(info.name, info.size, info.isdir())
                    if info.isfile():
                        with archive.extractfile(info) as src, open(target, "xb") as dst:
                            actual = _copy(src, dst, info.size)
                        if actual != info.size:
                            raise ValueError("truncated tar member")
            while bounded.read(_CHUNK):
                pass
        finally:
            if compressed:
                stream.close()


def _images(root, max_batch):
    images = []
    total = count = 0
    def walk_error(error):
        raise error
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=walk_error):
        for name in dirs + files:
            count += 1
            if count > MAX_ARCHIVE_ENTRIES:
                raise ValueError("too many package entries")
            path = Path(directory) / name
            _relative(path.relative_to(root).as_posix())
            info = path.lstat()
            if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
                raise ValueError("package links and special files are forbidden")
            if stat.S_ISREG(info.st_mode):
                total += info.st_size
                if info.st_size > _file_limit(name) or total > MAX_PACKAGE_BYTES:
                    raise ValueError("package exceeds byte limit")
                if name.lower().endswith(_IMAGE_SUFFIXES):
                    images.append(path)
                    if len(images) > max_batch:
                        raise ValueError("package exceeds max_batch")
    if not images:
        raise ValueError("package contains no images")
    return sorted(images, key=lambda path: path.relative_to(root).as_posix())


def _manifest(root):
    path = root / "manifest.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as src:
        raw = json.load(src)
    entries = {}
    if isinstance(raw, dict):
        for name, value in raw.items():
            entries[_relative(name)] = dict(value) if isinstance(value, dict) else {"profile": str(value)}
    elif isinstance(raw, list):
        for record in raw:
            if not isinstance(record, dict):
                continue
            name = record.get("file") or record.get("filename")
            if name:
                entries[_relative(str(name))] = {k: v for k, v in record.items() if k not in ("file", "filename")}
    else:
        raise ValueError("manifest.json must be an object or an array")
    return entries


def _identity(data):
    name, profile = data.get("name"), data.get("profile")
    if name is None and isinstance(profile, str):
        name, profile = profile, None
    return str(name or ""), profile if isinstance(profile, dict) else None


def _sidecar(path):
    sidecar = path.with_suffix(".json")
    if sidecar.is_file():
        try:
            with sidecar.open(encoding="utf-8") as src:
                data = json.load(src)
            if isinstance(data, dict):
                name, profile = _identity(data)
                return name, profile or None
            return str(data), None
        except (OSError, ValueError):
            pass
    sidecar = path.with_suffix(".txt")
    if sidecar.is_file():
        try:
            return sidecar.read_text(encoding="utf-8").strip(), None
        except (OSError, ValueError):
            pass
    return path.stem, None


@contextmanager
def corpus_entries(package, max_batch=1000):
    """Yield sorted (relative_name, image_path, name, profile, group) tuples.

    Accept a local directory, ZIP, tar/tar.gz, or HTTP(S) archive URL.
    profile is a dict or None; group is a stripped person/id string or "".
    Paths are strings and are valid during the context. Invalid/empty packages
    and exceeded limits raise ValueError; images are decoded by load_image.
    """
    if isinstance(max_batch, bool) or not isinstance(max_batch, int) or max_batch < 1:
        raise ValueError("max_batch must be a positive integer")
    with tempfile.TemporaryDirectory(prefix="face-corpus-") as staging:
        try:
            if not isinstance(package, (str, os.PathLike)) or not os.fspath(package):
                raise ValueError("package is required")
            if isinstance(package, str) and urlsplit(package).scheme.lower() in ("http", "https"):
                source = Path(staging) / "download"
                _download(package, source, MAX_PACKAGE_BYTES)
            else:
                source = _local(package)
            if source.is_dir():
                root = source
            else:
                if not stat.S_ISREG(source.stat().st_mode) or source.stat().st_size > MAX_PACKAGE_BYTES:
                    raise ValueError("invalid or oversized package")
                root = Path(staging) / "payload"
                root.mkdir()
                _extract(source, root)
            images = _images(root, max_batch)
            manifest = _manifest(root)
            entries = []
            for path in images:
                relative = path.relative_to(root).as_posix()
                entry = manifest.get(relative) or manifest.get(path.name) or {}
                name, profile = _identity(entry) if entry else _sidecar(path)
                group = str(entry.get("person") or entry.get("id") or "").strip()
                entries.append((relative, str(path), name, profile, group))
        except (OSError, ValueError, TypeError, EOFError, RuntimeError,
                tarfile.TarError, zipfile.BadZipFile, NotImplementedError) as error:
            raise ValueError("invalid corpus package: %s" % error) from error
        # Do not translate exceptions raised by the caller inside the context.
        yield entries


def load_image(source, is_url=False):
    """Decode a local photo path or explicit HTTP(S) URL to uint8 HxWx3 RGB.

    Encoded and decoded images are limited to 20 MiB. Invalid inputs raise
    ValueError. URL loading is explicit and permits localhost file servers.
    """
    try:
        if is_url:
            with tempfile.TemporaryDirectory(prefix="face-photo-") as staging:
                path = Path(staging) / "photo"
                _download(source, path, MAX_IMAGE_BYTES)
                data = path.read_bytes()
        else:
            path = _local(source)
            if not stat.S_ISREG(path.stat().st_mode) or path.stat().st_size > MAX_IMAGE_BYTES:
                raise ValueError("invalid or oversized image")
            with path.open("rb") as src:
                data = src.read(MAX_IMAGE_BYTES + 1)
        if not data or len(data) > MAX_IMAGE_BYTES:
            raise ValueError("empty or oversized image")
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.nbytes > MAX_IMAGE_BYTES:
            raise ValueError("invalid or oversized decoded image")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except (OSError, ValueError, TypeError, cv2.error) as error:
        raise ValueError("invalid image: %s" % error) from error
