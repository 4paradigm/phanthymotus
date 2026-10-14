"""
tests/test_face_models_registry.py — the FACE_MODELS registry and its bundles.

The registry exists so that adding a detector+recogniser pair is a table row
(`plugins/face_runtime.FACE_MODELS`) rather than a second plugin. These tests
pin the invariants that make that safe, because every one of them fails *late*
otherwise — a mistyped filename downloads a verified bundle and then dies
creating the session, and a missing `match_threshold` silently matches people
at another model's scale.

PR #284 originally added this model as a 2 500-line parallel plugin with its own
identity database; it is a row instead, and these tests are what keep the next
one a row too.
"""

from __future__ import annotations

import re

import pytest

from vision_stubs import PERCEPTION_ROOT  # noqa: F401  (installs stubs, sets sys.path)

import plugins.face as face_plugin  # noqa: E402
from plugins.face_runtime import (  # noqa: E402
    DEFAULT_DB_DIR,
    DEFAULT_FACE_MODEL,
    DEFAULT_FACE_MODEL_DIR,
    DEFAULT_MATCH_THRESHOLD,
    FACE_MODELS,
    default_match_threshold,
    face_bundle_for,
    face_db_dir_for,
)
from utils.model_downloader import FACE_MODEL_BUNDLES  # noqa: E402

MODEL_NAMES = sorted(FACE_MODELS)


def test_the_registry_has_more_than_one_row():
    """Guards the premise: a one-row registry is indistinguishable from hardcoding.

    If this ever drops back to a single entry, the per-model threshold and
    bundle plumbing below stops being exercised by anything real.
    """
    assert len(FACE_MODELS) >= 2, MODEL_NAMES
    assert DEFAULT_FACE_MODEL in FACE_MODELS


@pytest.mark.parametrize("model", MODEL_NAMES)
def test_every_row_declares_the_full_contract(model):
    spec = FACE_MODELS[model]
    for key in ("dir", "det", "rec", "bundle", "description", "match_threshold"):
        assert key in spec, f"{model} is missing {key!r}"
    assert spec["dir"].startswith("/models/"), spec["dir"]
    assert spec["det"].endswith(".onnx") and spec["rec"].endswith(".onnx")
    # The description is what the card's dropdown shows; an empty one leaves the
    # operator picking between two opaque identifiers.
    assert spec["description"].strip()


@pytest.mark.parametrize("model", MODEL_NAMES)
def test_each_rows_weights_are_in_the_bundle_it_names(model):
    """The failure this prevents: bundle verifies, then the session load 404s locally.

    `ensure_face_model` only guarantees the files *it* knows about. If a row's
    `det`/`rec` names are not among them, the download reports success and
    FaceAnalyzer dies on a missing path — far from the typo that caused it.
    """
    spec = FACE_MODELS[model]
    assert spec["bundle"] in FACE_MODEL_BUNDLES, spec["bundle"]
    _base, files = FACE_MODEL_BUNDLES[spec["bundle"]]
    assert spec["det"] in files, (model, spec["det"], sorted(files))
    assert spec["rec"] in files, (model, spec["rec"], sorted(files))


@pytest.mark.parametrize("model", MODEL_NAMES)
def test_every_pinned_file_has_an_exact_size_and_sha256(model):
    """No bundle may fall back to trusting the transport."""
    _base, files = FACE_MODEL_BUNDLES[FACE_MODELS[model]["bundle"]]
    for name, pin in files.items():
        assert isinstance(pin.get("size"), int) and pin["size"] > 0, (name, pin)
        assert re.fullmatch(r"[0-9a-f]{64}", pin.get("sha256", "")), (name, pin)


def test_no_bundle_is_orphaned():
    """A bundle no row names is weights nothing can ever fetch."""
    named = {spec["bundle"] for spec in FACE_MODELS.values()}
    assert set(FACE_MODEL_BUNDLES) == named, set(FACE_MODEL_BUNDLES) ^ named


@pytest.mark.parametrize("model", MODEL_NAMES)
def test_the_default_model_dir_resolves_to_the_rows_own_dir(model):
    """Selecting a model must move the weights path with it.

    `model_dir` keeps the buffalo_sc default in config.yaml, so a card that sets
    only `model` would otherwise look for the new weights in the old directory.
    """
    resolved, bundle = face_bundle_for(model, DEFAULT_FACE_MODEL_DIR)
    assert resolved == FACE_MODELS[model]["dir"]
    assert bundle == FACE_MODELS[model]["bundle"]


@pytest.mark.parametrize("model", MODEL_NAMES)
def test_an_explicit_model_dir_still_wins(model):
    """The documented escape hatch: point at a local copy of the weights."""
    resolved, bundle = face_bundle_for(model, "/models/face/my-own-copy")
    assert resolved == "/models/face/my-own-copy"
    assert bundle == FACE_MODELS[model]["bundle"]


def test_an_unknown_model_names_the_ones_that_exist():
    with pytest.raises(ValueError) as excinfo:
        face_bundle_for("no-such-model", DEFAULT_FACE_MODEL_DIR)
    message = str(excinfo.value)
    assert "no-such-model" in message
    for model in MODEL_NAMES:
        assert model in message


# ── per-model match_threshold ────────────────────────────────────────────────
#
# A cosine threshold belongs to the embedding space, not to the plugin. Measured
# over a 100-image LFW sample (4849 impostor pairs): buffalo_sc's impostors top
# out at 0.323 and scrfd25g_edgeface_s's at 0.450, so one number cannot serve
# both — 0.35 lets four of the latter's impostor pairs through.


def test_the_shipped_defaults_are_the_edgeface_pair_at_0_55():
    """The configured product default, pinned so a drift is visible in review."""
    assert DEFAULT_FACE_MODEL == "scrfd25g_edgeface_s"
    assert DEFAULT_MATCH_THRESHOLD == 0.55
    import yaml
    with open(PERCEPTION_ROOT / "config.yaml") as handle:
        face = yaml.safe_load(handle)["plugins"]["face_recognition"]
    # config.yaml and the registry must agree, or the card's default and the
    # file's default disagree depending on which one a robot was built from
    assert face["model"] == DEFAULT_FACE_MODEL
    assert face["match_threshold"] == DEFAULT_MATCH_THRESHOLD


def test_the_two_shipped_models_disagree_about_the_threshold():
    """Pins the measurement, not just the mechanism.

    If these ever become equal, the per-model plumbing is dead code and the
    comments in FACE_MODELS are describing something that no longer happens.
    """
    assert FACE_MODELS["buffalo_sc"]["match_threshold"] == 0.35
    assert FACE_MODELS["scrfd25g_edgeface_s"]["match_threshold"] == 0.55


@pytest.mark.parametrize("model", MODEL_NAMES)
def test_default_match_threshold_follows_the_row(model):
    assert default_match_threshold(model) == FACE_MODELS[model]["match_threshold"]
    assert 0.0 < default_match_threshold(model) < 1.0


def test_default_match_threshold_falls_back_rather_than_raising():
    """Only a default-picker — FaceAnalyzer gives the real error for a bad name."""
    assert default_match_threshold("no-such-model") == DEFAULT_MATCH_THRESHOLD
    assert DEFAULT_MATCH_THRESHOLD == FACE_MODELS[DEFAULT_FACE_MODEL]["match_threshold"]


@pytest.mark.parametrize("model", MODEL_NAMES)
def test_gates_take_the_configured_models_threshold(model):
    """The actual wiring: switching `model` must move the threshold with it."""
    gates = face_plugin._gates({"model": model})
    assert gates["match_threshold"] == FACE_MODELS[model]["match_threshold"]


def test_an_explicit_threshold_survives_a_model_switch():
    """A robot tuned by hand keeps its number; the default is only a default."""
    gates = face_plugin._gates({"model": "scrfd25g_edgeface_s",
                                "match_threshold": 0.42})
    assert gates["match_threshold"] == 0.42


def test_an_absent_model_key_uses_the_default_models_threshold():
    """config.yaml may predate the `model` key entirely."""
    assert face_plugin._gates({})["match_threshold"] == DEFAULT_MATCH_THRESHOLD


def test_the_card_offers_every_model_and_states_each_threshold():
    """The dropdown and its help text are generated, so they cannot drift."""
    schema = (face_plugin.TOOLS[0]["configSchema"]["properties"])
    assert "model" in schema, "the card no longer offers a model dropdown"
    assert schema["model"]["enum"] == MODEL_NAMES
    assert schema["model"]["default"] == DEFAULT_FACE_MODEL
    for model in MODEL_NAMES:
        assert model in schema["model"]["description"]
        # the threshold help has to name each model's recommended value, or an
        # operator switching models has no way to know the old one is wrong
        assert str(FACE_MODELS[model]["match_threshold"]) in \
            schema["match_threshold"]["description"]


# ── per-model identity store ─────────────────────────────────────────────────
#
# Embeddings from different networks are not comparable, so IdentityDB refuses a
# store written by another model. With one shared default that refusal fired on the
# act of *selecting* a second model: the card went to `state: error` and the only
# way out was for the operator to invent a db_dir by hand. The default now moves
# with the model, and the refusal is a backstop again.


def test_buffalo_sc_owns_the_deployed_roster_path_whatever_the_default_is():
    """Non-negotiable, and NOT keyed on being the default — it no longer is.

    Every robot deployed before the default moved has its roster at exactly
    `/models/face_db`. The first cut of face_db_dir_for keyed the pin on
    DEFAULT_FACE_MODEL, so flipping the default would have handed the new default a
    store written by buffalo_sc — the `state: error` this function exists to
    prevent — and orphaned the real roster behind `face_db-buffalo_sc`.
    """
    assert DEFAULT_DB_DIR == "/models/face_db"
    assert face_db_dir_for("buffalo_sc", DEFAULT_DB_DIR) == DEFAULT_DB_DIR
    assert FACE_MODELS["buffalo_sc"]["db_dir"] == DEFAULT_DB_DIR
    # specifically: the pin must survive buffalo_sc not being the default
    assert DEFAULT_FACE_MODEL != "buffalo_sc"


@pytest.mark.parametrize("model", [m for m in MODEL_NAMES if m != "buffalo_sc"])
def test_every_model_without_a_pin_gets_its_own_roster(model):
    resolved = face_db_dir_for(model, DEFAULT_DB_DIR)
    assert resolved != DEFAULT_DB_DIR
    assert resolved.startswith("/models/")
    assert model in resolved


def test_the_shipped_default_does_not_claim_the_old_roster():
    """The upgrade must land on a fresh store, not read buffalo_sc's embeddings."""
    assert face_db_dir_for(DEFAULT_FACE_MODEL, DEFAULT_DB_DIR) != DEFAULT_DB_DIR


def test_no_two_models_share_a_roster():
    """A collision would reintroduce the refusal this is meant to remove."""
    dirs = [face_db_dir_for(m, DEFAULT_DB_DIR) for m in MODEL_NAMES]
    assert len(set(dirs)) == len(dirs), dirs


@pytest.mark.parametrize("model", MODEL_NAMES)
def test_an_explicit_db_dir_pins_it_for_every_model(model):
    """The operator's own path must not get a per-model suffix appended."""
    assert face_db_dir_for(model, "/models/my_own_db") == "/models/my_own_db"


def test_an_unknown_model_does_not_invent_a_directory():
    """A bad name must not silently create a roster; FaceAnalyzer raises instead."""
    assert face_db_dir_for("no-such-model", DEFAULT_DB_DIR) == DEFAULT_DB_DIR


@pytest.mark.parametrize("model", MODEL_NAMES)
def test_db_options_resolve_the_roster_per_model(model):
    """The wiring, through the function the plugin actually calls."""
    options = face_plugin._db_options({"model": model})
    assert options["db_dir"] == face_db_dir_for(model, DEFAULT_DB_DIR)
    # the model is recorded in the store too, which is what makes the backstop work
    assert options["model"] == model


def test_db_options_honour_the_config_yaml_default_as_unset():
    """config.yaml ships `db_dir` set, and agent-core persists it into every card.

    So "the caller did not choose" cannot be observed as a missing key — only as
    the default value. If this regresses, selecting a second model silently reuses
    the first one's roster and the card errors, which is the bug being fixed.
    """
    options = face_plugin._db_options({"model": "scrfd25g_edgeface_s",
                                       "db_dir": DEFAULT_DB_DIR})
    assert options["db_dir"] == "/models/face_db-scrfd25g_edgeface_s"


def test_switching_back_and_forth_is_stable():
    """Each model must land on the same roster every time, so nothing is orphaned."""
    for _ in range(3):
        for model in MODEL_NAMES:
            assert face_plugin._db_options({"model": model})["db_dir"] == \
                face_db_dir_for(model, DEFAULT_DB_DIR)
