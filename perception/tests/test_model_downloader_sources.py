"""Multi-source bundles: choosing a host, and falling through when one fails.

Weights are getting large — a SmolVLA deployment is ~3 GB across two repos — and
no single host is fastest from everywhere. The same wheel measured 12 KB/s from
one index and 5.7 MB/s from COS on the same machine; ModelScope, where it mirrors
a model, is comparable to COS and skips the staging step entirely.

The property that makes several sources *safe* rather than merely convenient is
that every file is pinned by size and SHA256. The integrity check does not care
which host answered, so falling through costs nothing in guarantees — and the
test below that a corrupt source cannot be accepted is the one holding that up.

No network: urlopen and the fetch are stubbed.

Run: cd phanthymotus && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest perception/tests/test_model_downloader_sources.py -q
"""

from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import model_downloader as md  # noqa: E402


FILES = {
    "config.json": {"size": 100, "sha256": "a" * 64},
    "model.safetensors": {"size": 900_000_000, "sha256": "b" * 64},
}


# ── building a URL ───────────────────────────────────────────────────────────

def test_a_plain_base_url_is_joined():
    assert md._source_url("https://cos/base", "model.safetensors") == \
        "https://cos/base/model.safetensors"


def test_a_nested_path_is_quoted_per_segment():
    assert md._source_url("https://cos/base", "engines/jp61/flow.plan") == \
        "https://cos/base/engines/jp61/flow.plan"


def test_a_template_source_substitutes_the_file():
    """ModelScope's repo API takes the path as a query parameter, not a path."""
    template = "https://modelscope/api/models/x/repo?Revision=master&FilePath={file}"
    assert md._source_url(template, "model.safetensors").endswith(
        "FilePath=model.safetensors")


# ── choosing ─────────────────────────────────────────────────────────────────

def test_one_source_is_not_probed(monkeypatch):
    """Measuring a decision with one outcome only adds latency to the start."""
    probes = []
    monkeypatch.setattr(md, "_probe_source",
                        lambda s, f: probes.append(s) or 1.0)

    assert md._order_sources("m", ["only"], FILES) == ["only"]
    assert probes == []


def test_the_fastest_source_goes_first(monkeypatch):
    rates = {"slow": 20_000_000.0, "fast": 90_000_000.0}
    monkeypatch.setattr(md, "_probe_source", lambda s, f: rates[s])

    assert md._order_sources("m", ["slow", "fast"], FILES) == ["fast", "slow"]


def test_the_probe_uses_the_largest_file(monkeypatch):
    """Small files are often served from a different tier than large ones."""
    seen = []
    monkeypatch.setattr(md, "_probe_source",
                        lambda s, f: seen.append(f) or 1_000_000.0)

    md._order_sources("m", ["a", "b"], FILES)

    assert set(seen) == {"model.safetensors"}


def test_an_unusable_source_is_dropped(monkeypatch):
    monkeypatch.setattr(md, "_probe_source",
                        lambda s, f: 0.0 if s == "dead" else 5_000_000.0)

    assert md._order_sources("m", ["dead", "live"], FILES) == ["live"]


def test_all_probes_failing_keeps_the_declared_order(monkeypatch):
    """A probe is a heuristic; a transient failure must not mask a good host."""
    monkeypatch.setattr(md, "_probe_source", lambda s, f: 0.0)

    assert md._order_sources("m", ["a", "b"], FILES) == ["a", "b"]


def test_a_source_answering_instantly_with_nothing_does_not_win(monkeypatch):
    """An error page returns fast; speed alone would rank it first."""
    class _Response:
        def __init__(self, payload):
            self._payload = payload

        def read(self, _n):
            return self._payload

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    monkeypatch.setattr(md, "urlopen", lambda req, timeout=None: _Response(b"nope"))
    assert md._probe_source("https://broken", "model.safetensors") == 0.0


# ── falling through ──────────────────────────────────────────────────────────

@pytest.fixture
def fetches(monkeypatch):
    """Records every (source-url) a fetch was attempted from."""
    seen = []

    def _fetch(name, url, destination, metadata, **kwargs):
        seen.append(url)
        if "broken" in url:
            raise OSError("connection reset")
        pathlib.Path(destination).write_bytes(b"x")

    monkeypatch.setattr(md, "_fetch_pinned_file", _fetch)
    monkeypatch.setattr(md, "_order_sources", lambda name, sources, files: sources)
    return seen


def test_a_failing_source_falls_through_to_the_next(tmp_path, fetches):
    md._download_verified_bundle(
        "m", ["https://broken", "https://good"], str(tmp_path),
        {"a.bin": {"size": 1, "sha256": "x"}})

    assert [u.split("/")[2] for u in fetches] == ["broken", "good"]
    assert (tmp_path / "a.bin").exists()


def test_every_source_failing_raises_the_last_error(tmp_path, fetches):
    with pytest.raises(OSError):
        md._download_verified_bundle(
            "m", ["https://broken1", "https://broken2"], str(tmp_path),
            {"a.bin": {"size": 1, "sha256": "x"}})


def test_a_single_string_still_works(tmp_path, fetches):
    """Every existing caller passes one base_url; none of them change."""
    md._download_verified_bundle(
        "m", "https://good", str(tmp_path), {"a.bin": {"size": 1, "sha256": "x"}})

    assert len(fetches) == 1


def test_verification_is_what_makes_several_sources_safe(tmp_path, monkeypatch):
    """A second host cannot smuggle in a different file, only serve it faster.

    The fall-through would be reckless without this: it is the pinned size and
    SHA256, checked inside _fetch_pinned_file, that make "try somewhere else" a
    performance decision rather than a trust decision.
    """
    attempted = []

    def _fetch(name, url, destination, metadata, **kwargs):
        attempted.append(url)
        raise ValueError("sha256 mismatch")     # what a corrupt source produces

    monkeypatch.setattr(md, "_fetch_pinned_file", _fetch)
    monkeypatch.setattr(md, "_order_sources", lambda name, sources, files: sources)

    with pytest.raises(ValueError):
        md._download_verified_bundle(
            "m", ["https://a", "https://b"], str(tmp_path),
            {"a.bin": {"size": 1, "sha256": "x"}})

    # Both were tried, and neither was accepted.
    assert len(attempted) == 2
    assert not (tmp_path / "a.bin").exists()
