"""Offline resource preparation, invalidation, and real process-lock tests."""
import hashlib
import importlib.util
import json
import os
import shutil
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
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / "release"
        self.data = self.root / "frontend_data"
        self.data.mkdir(parents=True)
        self.cmu_path = self.data / "cmudict.rep"
        self.cmu_path.write_text("CAR  K AA1 R\n")
        self.policy = self.base / "case-policies.json"
        self.policy.write_text("{}")
        self.gold = {"CAR": ["S", "IY1", "EY1", "AA1", "R"],
                     "ACME": ["AE1", "K", "M", "IY0"]}
        self.source = self.base / "source.json"
        self.source.write_text(json.dumps(self.gold))
        self.compiler = module_from_file("_lexicon_resource_builder", COMPILER)
        package = types.ModuleType("_lexicon_resources_test_package")
        package.__path__ = [str(RESOURCE.parent)]
        self.package_patch = patch.dict(sys.modules, {package.__name__: package})
        self.package_patch.start()
        self.addCleanup(self.package_patch.stop)
        self.module = module_from_file(package.__name__ + ".lexicon_resources", RESOURCE)
        self.module._POLICY_PATH = self.policy
        self.module._SOURCE_CANONICAL_SHA = self.compiler.canonical_sha256(self.gold)
        self.calls = []

        def download(root, progress):
            self.calls.append((root, progress))
            return self.source

        self.module._download = download

    def explicit(self):
        payload, _ = self.compiler.compile_lexicon(self.source, self.compiler.parse_cmu_rep(self.cmu_path),
            expected_source_canonical_sha256=self.compiler.canonical_sha256(self.gold),
            cmu_source_sha256=hashlib.sha256(self.cmu_path.read_bytes()).hexdigest())
        path = self.base / "explicit.json"
        path.write_bytes(self.compiler.encode_payload(payload))
        os.environ["VITS2_LEXICON_PATH"] = str(path)
        return path

    def test_cold_prepare_and_warm_cache_without_download_or_compile(self):
        with patch.object(self.module, "_compiler", wraps=self.module._compiler) as compile_loader:
            result = self.module.ensure_external_lexicon(self.root)
            self.assertEqual(result, self.data / "external_lexicon.json")
            metadata = json.loads(result.with_suffix(".meta.json").read_text())
            self.assertEqual(metadata["entries"], 2)
            self.assertEqual(metadata["compiled_sha256"], hashlib.sha256(result.read_bytes()).hexdigest())
            self.source.unlink()  # Offline warm reuse must not need raw Gold.
            self.assertEqual(self.module.ensure_external_lexicon(self.root), result)
            self.assertEqual(len(self.calls), 1)
            self.assertEqual(compile_loader.call_count, 1)

    def test_adapter_forwards_external_lexicon_bytes_and_preparation_stages(self):
        events = []
        progress = lambda *values: events.append(("bytes", values))

        def download(root, callback):
            self.assertIs(callback, progress)
            callback(50, 0.32, 0.64)
            return self.source

        self.module._download = download
        adapter = module_from_file(self.module.__package__ + ".adapter", RESOURCE.with_name("adapter.py"))
        fake_engine = object()
        with patch.dict(sys.modules, {self.module.__name__: self.module}), \
             patch.object(adapter, "Vits2TensorRTAdapter", return_value=fake_engine):
            result = adapter.build_adapter(
                {"model_dir": str(self.root), "engine_dir": "/unused-in-test"},
                progress_cb=progress, stage_cb=lambda value: events.append(("stage", value)),
            )
        self.assertIs(result, fake_engine)
        self.assertEqual(events, [
            ("stage", "verify"), ("stage", "waiting"), ("stage", "download"),
            ("bytes", (50, 0.32, 0.64)), ("stage", "compile"),
            ("stage", "verify"), ("stage", "initialize"),
        ])

    def test_warm_cache_reports_verification_without_false_download_progress(self):
        self.module.ensure_external_lexicon(self.root)
        stages, progress = [], []
        self.source.unlink()
        self.module.ensure_external_lexicon(
            self.root, progress_cb=lambda *values: progress.append(values), stage_cb=stages.append)
        self.assertEqual(progress, [])
        self.assertIn("verify", stages)
        self.assertNotIn("download", stages)
        self.assertNotIn("compile", stages)

    def test_stage_status_failure_does_not_abort_valid_preparation(self):
        def broken_status(stage):
            raise RuntimeError("UI status sink unavailable")

        result = self.module.ensure_external_lexicon(self.root, stage_cb=broken_status)
        self.assertEqual(len(json.loads(result.read_bytes())["entries"]), 2)

    def test_corrupt_json_or_sidecar_rebuilds_default(self):
        result = self.module.ensure_external_lexicon(self.root)
        result.write_text("broken")
        self.module.ensure_external_lexicon(self.root)
        self.assertEqual(len(json.loads(result.read_text())["entries"]), 2)
        result.with_suffix(".meta.json").write_text("[]")
        self.module.ensure_external_lexicon(self.root)
        self.assertEqual(len(self.calls), 3)

    def test_policy_and_cmu_file_changes_invalidate_cache(self):
        result = self.module.ensure_external_lexicon(self.root)
        self.policy.write_text('{"ACME":"exact"}')
        self.module.ensure_external_lexicon(self.root)
        self.assertEqual(json.loads(result.read_text())["entries"]["ACME"]["mode"], "exact")
        self.cmu_path.write_text("CAR  K AA1 R\nACME  AE1 K M IY0\n")
        self.module.ensure_external_lexicon(self.root)
        self.assertEqual(json.loads(result.read_text())["entries"]["ACME"]["mode"], "cmu")
        self.assertEqual(len(self.calls), 3)

    def test_compiler_fingerprint_changes_invalidate_cache(self):
        self.module.ensure_external_lexicon(self.root)
        original_sha = self.module._sha
        with patch.object(self.module, "_sha", side_effect=lambda p: "e" * 64
                          if Path(p) == self.module._COMPILER_PATH else original_sha(p)):
            result = self.module.ensure_external_lexicon(self.root)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(json.loads(result.with_suffix(".meta.json").read_text())["compiler_sha256"], "e" * 64)

    def test_explicit_file_and_hash_bypass_download_and_default_writes(self):
        path = self.explicit()
        os.environ["VITS2_LEXICON_SHA256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.cmu_path.unlink()
        self.assertEqual(self.module.ensure_external_lexicon(self.root), path)
        self.assertFalse(self.calls)
        self.assertFalse((self.data / "external_lexicon.json").exists())

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
        self.assertFalse(self.calls)

    def test_invalid_global_hash_rejected_before_source_download(self):
        os.environ["VITS2_LEXICON_SHA256"] = "invalid"
        with self.assertRaises(ValueError):
            self.module.ensure_external_lexicon(self.root)
        self.assertFalse(self.calls)

    def test_missing_cmu_does_not_fetch_a_source_it_cannot_compile(self):
        self.cmu_path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.module.ensure_external_lexicon(self.root)
        self.assertFalse(self.calls)

    def test_failed_compile_does_not_publish_json_or_sidecar(self):
        self.source.write_text('{"ACME":["BAD"]}')
        self.module._SOURCE_CANONICAL_SHA = self.compiler.canonical_sha256(json.loads(self.source.read_text()))
        with self.assertRaises(ValueError):
            self.module.ensure_external_lexicon(self.root)
        self.assertFalse((self.data / "external_lexicon.json").exists())
        self.assertFalse((self.data / "external_lexicon.meta.json").exists())
        self.assertFalse(list(self.data.glob(".external-lexicon-*")))

    def test_default_checksum_mismatch_does_not_publish_generated_output(self):
        os.environ["VITS2_LEXICON_SHA256"] = "0" * 64
        with self.assertRaises(ValueError):
            self.module.ensure_external_lexicon(self.root)
        self.assertFalse((self.data / "external_lexicon.json").exists())

    def test_actual_verified_downloader_reuses_local_pinned_source(self):
        from utils import model_downloader as md
        self.module._download = lambda root, progress: Path(md.ensure_verified_bundle(
            "test-source", str(root / "frontend"), self.base.as_uri(),
            {self.source.name: {"size": self.source.stat().st_size,
                                "sha256": hashlib.sha256(self.source.read_bytes()).hexdigest()}},
            progress_cb=progress)[self.source.name])
        result = self.module.ensure_external_lexicon(self.root)
        self.assertEqual(len(json.loads(result.read_bytes())["entries"]), 2)
        self.assertTrue((self.root / "frontend/source.json").is_file())

    def test_failed_preparation_keeps_the_previous_release_paths(self):
        import importlib
        package_name = self.module.__package__
        paths = importlib.import_module(package_name + ".frontend.release_paths")
        previous = self.base / "previous-release"
        paths.configure_release_paths(previous)
        adapter = module_from_file(package_name + ".adapter", RESOURCE.with_name("adapter.py"))
        with patch.dict(sys.modules, {package_name + ".lexicon_resources": self.module}):
            with patch.object(self.module, "ensure_external_lexicon", side_effect=ValueError("bad resource")):
                with self.assertRaises(ValueError):
                    adapter._configure_frontend_paths(self.root)
        self.assertEqual(paths.frontend_data_dir(), previous / "frontend_data")

    def test_plugin_only_image_prepares_without_tools_directory(self):
        image = self.base / "image/work"
        package = image / "plugins/vits2_tts_trt"
        frontend = package / "frontend"
        frontend.mkdir(parents=True)
        (image / "plugins/__init__.py").write_text("")
        for filename in ("__init__.py", "lexicon_resources.py", "lexicon_compiler.py"):
            shutil.copy2(RESOURCE.parent / filename, package / filename)
        for filename in ("__init__.py", "symbols.py", "release_paths.py", "external_lexicon.py"):
            shutil.copy2(RESOURCE.parent / "frontend" / filename, frontend / filename)
        shutil.copy2(self.policy, frontend / "lexicon_case_policies.json")
        code = """
import hashlib,json,sys
from pathlib import Path
from plugins.vits2_tts_trt import lexicon_resources as r
image,root,source=map(Path,sys.argv[1:])
assert not (image/'tools').exists()
assert r._COMPILER_PATH == image/'plugins/vits2_tts_trt/lexicon_compiler.py'
r._SOURCE_CANONICAL_SHA=hashlib.sha256(json.dumps(json.loads(source.read_text()),sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
r._download=lambda root,progress:source
path=r.ensure_external_lexicon(root)
assert len(json.loads(path.read_text())['entries']) == 2
assert 'torch' not in sys.modules and 'tensorrt' not in sys.modules
"""
        env = dict(os.environ, PYTHONPATH=str(image), PYTHONDONTWRITEBYTECODE="1")
        done = subprocess.run([sys.executable, "-B", "-c", code, str(image), str(self.root),
                               str(self.source)], cwd=image, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)

    def test_two_processes_share_one_compilation(self):
        count = self.base / "compilations.txt"
        code = '''
import hashlib,json,os,sys
from pathlib import Path
from plugins.vits2_tts_trt import lexicon_resources as r
root,source,policy,count=map(Path,sys.argv[1:])
r._POLICY_PATH=policy
r._SOURCE_CANONICAL_SHA=hashlib.sha256(json.dumps(json.loads(source.read_text()),sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
r._SOURCE_BASE=source.parent.as_uri()
r._SOURCE_FILES={source.name:{'size':source.stat().st_size,'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}}
original_download=r._download
def download(root, progress):
 from utils.model_downloader import ensure_verified_bundle
 return Path(ensure_verified_bundle('test-source',str(root/'frontend'),r._SOURCE_BASE,r._SOURCE_FILES,progress_cb=progress)[source.name])
r._download=download
original_compiler=r._compiler
def compiler():
 with count.open('a') as f:f.write(str(os.getpid())+'\\n')
 return original_compiler()
r._compiler=compiler
r.ensure_external_lexicon(root)
'''
        env = dict(os.environ, PYTHONPATH=str(PERCEPTION), PYTHONDONTWRITEBYTECODE="1")
        command = [sys.executable, "-B", "-c", code, str(self.root), str(self.source), str(self.policy), str(count)]
        children = [subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    for _ in range(2)]
        for child in children:
            stdout, stderr = child.communicate(timeout=30)
            self.assertEqual(child.returncode, 0, stdout + stderr)
        self.assertEqual(len(count.read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
