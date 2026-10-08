"""Standalone unittest coverage; does not import the face plugin or ROS."""
import functools
import http.server
import importlib.util
import io
import json
from pathlib import Path
import stat
import tarfile
import tempfile
import threading
import unittest
from unittest.mock import patch
import zipfile

import cv2

# These contracts feed real pixel content through cv2 encode/decode and, in the
# detector case, cv2.dnn — the "WxH" marker fake cv2 (vision_stubs.__motus_fake__)
# cannot stand in for that. They run where cv2 is real: the perception image,
# whose container-side suite run is the documented acceptance pass (see
# perception/README.md § Running the tests inside a perception image). Elsewhere
# they skip rather than assert against stub pixels.
_NEEDS_REAL_CV2 = not getattr(cv2, "__motus_fake__", False)
import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "face_corpus", Path(__file__).resolve().parents[1] / "plugins" / "face_corpus.py")
corpus = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(corpus)


@unittest.skipUnless(_NEEDS_REAL_CV2, "needs real cv2")
class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.photos = self.root / "photos"
        self.photos.mkdir()
        self.png = cv2.imencode(".png", np.full((3, 4, 3), (10, 20, 200), np.uint8))[1].tobytes()

    def photo(self, name):
        path = self.photos / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.png)
        return path

    def manifest(self, data):
        (self.photos / "manifest.json").write_text(json.dumps(data))

    def test_manifest_list_sorting_group_and_precedence(self):
        self.photo("z.png")
        self.photo("sub/a.png")
        self.photo("b.png")
        (self.photos / "b.txt").write_text("sidecar ignored")
        self.manifest([
            {"file": "z.png", "profile": "Zed", "person": " person "},
            {"filename": "a.png", "name": "Alice", "profile": {"age": 7}, "id": "person"},
            {"file": "b.png", "name": "Bob", "profile": {}}, None, {}])
        with corpus.corpus_entries(self.photos) as entries:
            self.assertEqual([e[0] for e in entries], ["b.png", "sub/a.png", "z.png"])
            self.assertEqual([e[2:] for e in entries], [
                ("Bob", {}, ""), ("Alice", {"age": 7}, "person"), ("Zed", None, "person")])

    def test_mapping_and_sidecars(self):
        for name in "abcdefg":
            self.photo(name + ".png")
        self.manifest({"a.png": "Alice", "b.png": {"name": "B", "profile": {"x": 1}}})
        (self.photos / "c.json").write_text('{"profile":"C"}')
        (self.photos / "c.txt").write_text("ignored")
        (self.photos / "d.json").write_text('{"name":"D","profile":{"x":2}}')
        (self.photos / "e.json").write_text("broken JSON")
        (self.photos / "e.txt").write_text(" E \n")
        (self.photos / "f.json").write_text('"F"')
        with corpus.corpus_entries(self.photos) as entries:
            self.assertEqual([e[2] for e in entries], ["Alice", "B", "C", "D", "E", "F", "g"])
            self.assertEqual(entries[3][3], {"x": 2})

    def test_archives_and_cleanup(self):
        photo = self.photo("nested/a.png")
        for suffix in ("zip", "tar.gz", "tar"):
            archive = self.root / ("package." + suffix)
            if suffix == "zip":
                with zipfile.ZipFile(archive, "w") as out:
                    out.write(photo, "nested/a.png")
            else:
                with tarfile.open(archive, "w:gz" if suffix.endswith("gz") else "w") as out:
                    out.add(photo, arcname="nested/a.png")
            with corpus.corpus_entries(archive) as entries:
                extracted = Path(entries[0][1])
                self.assertEqual(extracted.read_bytes(), self.png)
            self.assertFalse(extracted.exists())

    def test_unsafe_zip_paths_and_links(self):
        for name in ("../escape.png", "/escape.png", "C:/escape.png", "a/../escape.png", "..\\escape.png"):
            archive = self.root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as out:
                out.writestr(name, self.png)
            with self.assertRaises(ValueError), corpus.corpus_entries(archive):
                pass
        info = zipfile.ZipInfo("link.png")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        with zipfile.ZipFile(archive, "w") as out:
            out.writestr(info, "target")
        with self.assertRaises(ValueError), corpus.corpus_entries(archive):
            pass

    def test_tar_special_files(self):
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE, tarfile.FIFOTYPE):
            archive = self.root / "bad.tar.gz"
            with tarfile.open(archive, "w:gz") as out:
                info = tarfile.TarInfo("bad.png")
                info.type = kind
                info.linkname = "../outside"
                out.addfile(info)
            with self.assertRaises(ValueError), corpus.corpus_entries(archive):
                pass

    def test_directory_symlink(self):
        photo = self.photo("a.png")
        (self.photos / "link.png").symlink_to(photo)
        with self.assertRaises(ValueError), corpus.corpus_entries(self.photos):
            pass
        with self.assertRaises(ValueError):
            corpus.load_image(self.photos / "link.png")

    def test_invalid_packages_and_limits(self):
        with self.assertRaises(ValueError), corpus.corpus_entries(self.photos):
            pass
        self.photo("a.png")
        self.photo("b.png")
        for limit in (0, True, 1.5, 1):
            with self.assertRaises(ValueError), corpus.corpus_entries(self.photos, limit):
                pass
        for value in (None, "", self.root / "missing"):
            with self.assertRaises(ValueError), corpus.corpus_entries(value):
                pass
        self.manifest(1)
        with self.assertRaises(ValueError), corpus.corpus_entries(self.photos):
            pass
        (self.photos / "manifest.json").unlink()
        for key, limit in (("MAX_IMAGE_BYTES", 1), ("MAX_PACKAGE_BYTES", len(self.png)), ("MAX_ARCHIVE_ENTRIES", 1)):
            with patch.object(corpus, key, limit):
                with self.assertRaises(ValueError), corpus.corpus_entries(self.photos):
                    pass
        archive = self.root / "bomb.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as out:
            out.writestr("a.png", self.png)
            out.writestr("large.txt", b"x" * 10000)
        with patch.object(corpus, "MAX_PACKAGE_BYTES", 1000):
            with self.assertRaises(ValueError), corpus.corpus_entries(archive):
                pass
        with patch.object(corpus, "MAX_ARCHIVE_ENTRIES", 1):
            with self.assertRaises(ValueError), corpus.corpus_entries(archive):
                pass

    def test_tar_decompressed_padding_limit(self):
        archive = self.root / "padding.tar.gz"
        with tarfile.open(archive, "w:gz") as out:
            info = tarfile.TarInfo("a.png")
            info.size = len(self.png)
            out.addfile(info, io.BytesIO(self.png))
        with patch.object(corpus, "MAX_PACKAGE_BYTES", 1000):
            with self.assertRaises(ValueError), corpus.corpus_entries(archive):
                pass

    def test_load_image_rgb_and_errors(self):
        photo = self.photo("a.png")
        image = corpus.load_image(photo)
        self.assertEqual(image.shape, (3, 4, 3))
        self.assertEqual(image.dtype, np.uint8)
        self.assertEqual(image[0, 0].tolist(), [200, 20, 10])
        photo.write_bytes(b"not an image")
        with self.assertRaises(ValueError):
            corpus.load_image(photo)
        with self.assertRaises(ValueError):
            corpus.load_image("file:///etc/passwd", is_url=True)

    def test_http_localhost(self):
        self.photo("a.png")
        archive = self.root / "package.zip"
        with zipfile.ZipFile(archive, "w") as out:
            out.writestr("a.png", self.png)
        class Handler(http.server.SimpleHTTPRequestHandler):
            def log_message(self, *args):
                pass
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(self.root)))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = "http://127.0.0.1:%d" % server.server_port
            with corpus.corpus_entries(url + "/package.zip") as entries:
                self.assertEqual(entries[0][2], "a")
            self.assertEqual(corpus.load_image(url + "/photos/a.png", True)[0, 0].tolist(), [200, 20, 10])
            with patch.object(corpus, "MAX_IMAGE_BYTES", 1), self.assertRaises(ValueError):
                corpus.load_image(url + "/photos/a.png", True)
            with self.assertRaises(ValueError), corpus.corpus_entries(url + "/missing.zip"):
                pass
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_caller_exception_and_cleanup(self):
        self.photo("a.png")
        with self.assertRaisesRegex(RuntimeError, "caller"):
            with corpus.corpus_entries(self.photos):
                raise RuntimeError("caller")


if __name__ == "__main__":
    unittest.main()
