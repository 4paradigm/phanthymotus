"""Offline resource preparation, invalidation, and real process-lock tests."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


PERCEPTION = Path(__file__).resolve().parents[1]
RESOURCE = PERCEPTION / "plugins/vits2_tts_trt/lexicon_resources.py"
COMPILER = PERCEPTION / "tools/vits2_prepare_external_lexicon.py"


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ResourceTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / "release"
        self.data = self.root / "frontend_data"
        self.data.mkdir(parents=True)
        self.cmu_path = self.data / "cmudict.rep"
        self.cmu_path.write_text("CAR  K AA1 R\n")
        self.compiler = module_from_file("_lexicon_resource_builder", COMPILER)
        raw = self.base / "gold.json"
        raw.write_text(json.dumps({"CAR": ["B", "AE1", "D"], "ACME": ["AE1", "K", "M", "IY0"]}))
        payload, _ = self.compiler.compile_lexicon(
            raw, self.compiler.parse_cmu_rep(self.cmu_path),
            expected_source_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
            cmu_source_sha256=hashlib.sha256(self.cmu_path.read_bytes()).hexdigest())
        self.source = self.base / "external_lexicon.json"
        self.source.write_bytes(self.compiler.encode_payload(payload))
        package = types.ModuleType("_lexicon_resources_test_package")
        package.__path__ = [str(RESOURCE.parent)]
        modules = patch.dict(sys.modules, {package.__name__: package})
        modules.start()
        self.addCleanup(modules.stop)
        self.module = module_from_file(package.__name__ + ".lexicon_resources", RESOURCE)
        self.module._SOURCE_BASE = self.base.as_uri()
        self.module._SOURCE_FILES = {self.source.name: {
            "size": self.source.stat().st_size,
            "sha256": hashlib.sha256(self.source.read_bytes()).hexdigest()}}
        # Other isolated tests restore sys.modules but can leave a stale
        # package attribute. Patch the module the runtime actually imports.
        model_downloader = importlib.import_module("utils.model_downloader")
        fetch = patch.object(model_downloader, "_fetch_pinned_file", wraps=model_downloader._fetch_pinned_file)
        self.fetch = fetch.start()
        self.addCleanup(fetch.stop)

    def explicit(self):
        path = self.base / "explicit.json"
        path.write_bytes(self.source.read_bytes())
        os.environ["VITS2_LEXICON_PATH"] = str(path)
        return path

    def test_cold_prepare_and_warm_cache_without_download_or_compile(self):
        result = self.module.ensure_external_lexicon(self.root)
        self.assertEqual(result, self.data / "lexicon/20261027-v3/external_lexicon.json")
        self.assertEqual(result.read_bytes(), self.source.read_bytes())
        self.source.unlink()
        self.assertEqual(self.module.ensure_external_lexicon(self.root), result)
        self.assertEqual(self.fetch.call_count, 1)
        self.assertFalse(hasattr(self.module, "_compiler"))
        self.assertFalse((self.data / "en_gold_pronunciations.json").exists())

    def test_corrupt_cache_is_replaced_with_the_same_pinned_resource(self):
        result = self.module.ensure_external_lexicon(self.root)
        result.write_text("broken")
        self.module.ensure_external_lexicon(self.root)
        self.assertEqual(result.read_bytes(), self.source.read_bytes())
        self.assertEqual(self.fetch.call_count, 2)

    def test_bad_download_never_replaces_valid_resource(self):
        result = self.module.ensure_external_lexicon(self.root)
        original = result.read_bytes()
        self.module._SOURCE_FILES[self.source.name]["sha256"] = "0" * 64
        self.fetch.side_effect = ValueError("controlled SHA mismatch")
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            self.module.ensure_external_lexicon(self.root)
        self.assertEqual(result.read_bytes(), original)

    def test_adapter_forwards_bytes_and_preparation_stages(self):
        stages, progress = [], []
        adapter = module_from_file(self.module.__package__ + ".adapter", RESOURCE.with_name("adapter.py"))
        fake_engine = object()
        with patch.dict(sys.modules, {self.module.__name__: self.module}), \
             patch.object(adapter, "Vits2TensorRTAdapter", return_value=fake_engine):
            result = adapter.build_adapter(
                {"model_dir": str(self.root), "engine_dir": "/unused"},
                progress_cb=lambda *values: progress.append(values), stage_cb=stages.append)
        self.assertIs(result, fake_engine)
        self.assertEqual(stages, ["verify", "download", "verify", "initialize"])
        self.assertTrue(progress)
        self.assertFalse(any("compile" in value for value in stages))

    def test_warm_cache_has_no_byte_progress_or_network_access(self):
        self.module.ensure_external_lexicon(self.root)
        stages, progress = [], []
        self.fetch.side_effect = AssertionError("warm cache downloaded")
        self.module.ensure_external_lexicon(
            self.root, progress_cb=lambda *values: progress.append(values), stage_cb=stages.append)
        self.assertEqual(progress, [])
        self.assertEqual(stages[-1], "verify")

    def test_stage_callback_failure_does_not_abort_preparation(self):
        def broken_status(stage):
            raise RuntimeError("UI status sink unavailable")
        result = self.module.ensure_external_lexicon(self.root, stage_cb=broken_status)
        self.assertEqual(len(json.loads(result.read_bytes())["entries"]), 2)

    def test_cmu_mismatch_is_rejected_instead_of_recompiled(self):
        self.module.ensure_external_lexicon(self.root)
        self.cmu_path.write_text("CAR  B AE1 D\n")
        with self.assertRaisesRegex(ValueError, "another CMU"):
            self.module.ensure_external_lexicon(self.root)
        self.assertEqual(self.fetch.call_count, 1)

    def test_explicit_file_is_preserved_while_shared_rules_are_prepared(self):
        path = self.explicit()
        original = path.read_bytes()
        os.environ["VITS2_LEXICON_SHA256"] = hashlib.sha256(original).hexdigest()
        result = self.module.ensure_external_lexicon(self.root)
        self.assertEqual(result, path)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(self.fetch.call_count, 1)
        self.assertTrue((self.data / "lexicon/20261027-v3/external_lexicon.json").exists())

    def test_explicit_missing_corrupt_or_bad_hash_never_falls_back(self):
        path = self.explicit()
        original = path.read_bytes()
        for state in ("missing", "broken", "hash"):
            with self.subTest(state=state):
                path.write_bytes(original)
                os.environ.pop("VITS2_LEXICON_SHA256", None)
                if state == "missing":
                    path.unlink()
                elif state == "broken":
                    path.write_text("broken")
                else:
                    os.environ["VITS2_LEXICON_SHA256"] = "0" * 64
                with self.assertRaises((OSError, ValueError)):
                    self.module.ensure_external_lexicon(self.root)
        self.fetch.assert_not_called()

    def test_explicit_cache_path_collision_is_rejected_without_overwriting(self):
        path = self.module.ensure_external_lexicon(self.root)
        data = json.loads(path.read_bytes())
        data["entries"]["ACME"]["arpa"] = ["EY1"]
        path.write_text(json.dumps(data))
        original = path.read_bytes()
        os.environ["VITS2_LEXICON_PATH"] = str(path)
        with self.assertRaisesRegex(ValueError, "conflicts with the shared resource cache"):
            self.module.ensure_external_lexicon(self.root)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(self.fetch.call_count, 1)

    def test_invalid_global_hash_is_rejected_before_download(self):
        os.environ["VITS2_LEXICON_SHA256"] = "invalid"
        with self.assertRaises(ValueError):
            self.module.ensure_external_lexicon(self.root)
        self.fetch.assert_not_called()

    def test_missing_cmu_is_rejected_before_download(self):
        self.cmu_path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.module.ensure_external_lexicon(self.root)
        self.fetch.assert_not_called()

    def test_failed_preparation_keeps_previous_release_paths(self):
        import importlib
        paths = importlib.import_module(self.module.__package__ + ".frontend.release_paths")
        previous = self.base / "previous-release"
        paths.configure_release_paths(previous)
        adapter = module_from_file(self.module.__package__ + ".adapter", RESOURCE.with_name("adapter.py"))
        with patch.dict(sys.modules, {self.module.__name__: self.module}), \
             patch.object(self.module, "ensure_external_lexicon", side_effect=ValueError("bad resource")):
            with self.assertRaises(ValueError):
                adapter._configure_frontend_paths(self.root)
        self.assertEqual(paths.frontend_data_dir(), previous / "frontend_data")

    def test_two_processes_share_one_verified_download(self):
        count = self.base / "downloads.txt"
        code = """
import hashlib,os,sys
from pathlib import Path
from plugins.vits2_tts_trt import lexicon_resources as r
from utils import model_downloader as md
root,source,count=map(Path,sys.argv[1:])
r._SOURCE_BASE=source.parent.as_uri()
r._SOURCE_FILES={source.name:{'size':source.stat().st_size,'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}}
original=md._fetch_pinned_file
def fetch(*args,**kwargs):
 with count.open('a') as f:f.write(str(os.getpid())+'\\n')
 return original(*args,**kwargs)
md._fetch_pinned_file=fetch
r.ensure_external_lexicon(root)
assert not {'torch','tensorrt','g2p_en'} & set(sys.modules)
"""
        env = dict(os.environ, PYTHONPATH=str(PERCEPTION), PYTHONDONTWRITEBYTECODE="1")
        command = [sys.executable, "-B", "-c", code, str(self.root), str(self.source), str(count)]
        children = [subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    for _ in range(2)]
        for child in children:
            stdout, stderr = child.communicate(timeout=30)
            self.assertEqual(child.returncode, 0, stdout + stderr)
        self.assertEqual(len(count.read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
