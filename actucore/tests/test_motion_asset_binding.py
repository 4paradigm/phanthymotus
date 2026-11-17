"""Complete MuJoCo asset binding, including XML includes and external meshes."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

mujoco = pytest.importorskip("mujoco")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motion_service.contracts import load_model_fresh, model_fingerprint, validate_profile
from motion_service.prepare_profile import export_runtime_scene


def profile(path):
    return {"id": "binding-test", "label": "Asset binding", "robot_type": "unitree_g1",
            "model_version": "v1", "model_path": str(path), "joint_names": ["joint"],
            "control_hz": 50, "physics_dt": .002, "default_joint_pos": [0.],
            "action_scale": [.1], "kp": [10.], "kd": [1.],
            "joint_limits": {"lower": [-1.], "upper": [1.]},
            "quaternion_order": "wxyz", "history_length": 10}


def write_include(path, mass):
    path.write_text('<mujoco><worldbody><body name="pelvis" pos="0 0 1">'
                    '<joint name="joint" type="hinge" range="-90 90"/>'
                    f'<geom type="sphere" size=".1" mass="{mass}"/>'
                    '</body></worldbody></mujoco>')


@pytest.fixture
def included_model(tmp_path):
    source = tmp_path / "scene.xml"
    source.write_text('<mujoco model="binding"><option timestep=".002"/>'
                      '<include file="body.xml"/>'
                      '<actuator><motor joint="joint"/></actuator></mujoco>')
    include = tmp_path / "body.xml"
    write_include(include, 1.)
    return source, include


def test_fingerprint_is_stable_across_recompiles_and_process_restart(included_model):
    source, _ = included_model
    first = validate_profile(profile(source))
    again = validate_profile(json.loads(json.dumps(first)))
    assert again["profile_hash"] == first["profile_hash"]
    assert first["mujoco_version"] == mujoco.__version__
    script = ("import sys; sys.path.insert(0,sys.argv[1]); import mujoco; "
              "from motion_service.contracts import model_fingerprint; "
              "print(model_fingerprint(mujoco.MjModel.from_xml_path(sys.argv[2])))")
    other = subprocess.check_output([sys.executable, "-c", script,
                                     str(Path(__file__).resolve().parents[1]), str(source)], text=True)
    assert other.strip() == first["compiled_model_sha256"]


def test_changed_include_invalidates_profile_without_top_level_xml_change(included_model):
    source, include = included_model
    first = validate_profile(profile(source))
    write_include(include, 2.)
    second = validate_profile(first)
    assert second["model_sha256"] == first["model_sha256"]
    assert second["compiled_model_sha256"] != first["compiled_model_sha256"]
    assert second["profile_hash"] != first["profile_hash"]


def write_mesh(path, height):
    path.write_text(f"v 0 0 0\nv .1 0 0\nv 0 .1 0\nv 0 0 {height}\n"
                    "f 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n")


def test_external_mesh_content_is_bound_but_unrelated_files_are_not(tmp_path):
    mesh = tmp_path / "shape.obj"
    write_mesh(mesh, .1)
    source = tmp_path / "scene.xml"
    source.write_text('<mujoco model="mesh_binding"><option timestep=".002"/>'
                      '<asset><mesh name="shape" file="shape.obj"/></asset>'
                      '<worldbody><body name="pelvis" pos="0 0 1">'
                      '<joint name="joint" type="hinge" range="-90 90"/>'
                      '<geom type="mesh" mesh="shape" mass="1"/></body></worldbody>'
                      '<actuator><motor joint="joint"/></actuator></mujoco>')
    first = validate_profile(profile(source))
    (tmp_path / "unrelated.txt").write_text("not an asset")
    assert validate_profile(first)["profile_hash"] == first["profile_hash"]
    original_stat = mesh.stat()
    write_mesh(mesh, .2)
    # File size and timestamp need not change when an asset is replaced.
    # A valid content fingerprint must not rely on MuJoCo's timestamp cache.
    assert mesh.stat().st_size == original_stat.st_size
    os.utime(mesh, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    changed = validate_profile(first)
    assert changed["model_sha256"] == first["model_sha256"]
    assert changed["compiled_model_sha256"] != first["compiled_model_sha256"]


def test_binary_model_roundtrip_has_same_compiled_fingerprint(included_model, tmp_path):
    source, _ = included_model
    model = mujoco.MjModel.from_xml_path(str(source))
    expected = model_fingerprint(model)
    binary = tmp_path / "scene.mjb"
    mujoco.mj_saveModel(model, str(binary), None)
    loaded = mujoco.MjModel.from_binary_path(str(binary))
    assert model_fingerprint(loaded) == expected
    assert validate_profile(profile(binary))["compiled_model_sha256"] == expected


def test_profile_export_preserves_runtime_physics_not_original_scene(included_model, tmp_path):
    source, _ = included_model
    model = mujoco.MjModel.from_xml_path(str(source))
    # mjlab applies these after MjSpec compilation; exporting the source scene
    # would silently restore its old .002 timestep and default solver settings.
    model.opt.timestep = .005
    model.opt.iterations = 17
    model.opt.gravity[2] = -8.5
    model.actuator_gainprm[0, 0] = 2.

    def write_source(directory):
        directory.mkdir(parents=True)
        (directory / "scene.xml").write_text(source.read_text())

    base = SimpleNamespace(scene=SimpleNamespace(write=write_source),
                           sim=SimpleNamespace(mj_model=model))
    path = export_runtime_scene(base, tmp_path / "exported" / "scene")
    assert path.suffix == ".mjb"
    loaded = load_model_fresh(path)
    assert loaded.opt.timestep == .005
    assert loaded.opt.iterations == 17
    assert loaded.opt.gravity[2] == -8.5
    assert loaded.actuator_gainprm[0, 0] == 2.
    assert model_fingerprint(loaded) == model_fingerprint(model)


def test_model_fingerprint_detects_changed_physics(included_model):
    source, _ = included_model
    model = mujoco.MjModel.from_xml_path(str(source))
    before = model_fingerprint(model)
    model.body_mass[1] *= 2
    assert model_fingerprint(model) != before


def test_fresh_load_restores_cache_capacity_after_compile_failure(tmp_path):
    broken = tmp_path / "broken.xml"
    broken.write_text("<mujoco><invalid/></mujoco>")
    cache = mujoco.mj_getCache()
    capacity = mujoco.mj_getCacheCapacity(cache)
    with pytest.raises(ValueError):
        load_model_fresh(broken)
    assert mujoco.mj_getCacheCapacity(cache) == capacity
