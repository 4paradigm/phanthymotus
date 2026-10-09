"""
tests/test_identity_db_dim.py — IdentityDB 在非 512 维下的行为，以及换模型/换维度的拒绝路径。

`test_identity_db.py` covers everything face already relied on, at 512 dims. This
file covers the part that generalisation added and nothing exercised before: a
second dimension (192, which is what the CAM++ voiceprint model emits), and the
two refusals that stop a database written by one model from being read by another.

Why the refusals matter more than they look: matching across two networks does not
raise. `matrix @ embedding` succeeds whenever the widths agree, every similarity is
meaningless, and the row count still matches the metadata — so the database looks
healthy while quietly confusing identities. The only safe behaviour is to refuse,
and to say which two models disagree.

Pure host-side: disk + numpy.
Run: python -m pytest perception/tests -q
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from vision_stubs import PERCEPTION_ROOT  # noqa: F401  (puts perception on sys.path)

import plugins.identity_db as identity_db_module  # noqa: E402
from plugins.identity_db import IdentityDB, IdentityDBError  # noqa: E402

VOICE_DIM = 192


@pytest.fixture(autouse=True)
def _allow_tmp_db(monkeypatch):
    monkeypatch.setattr(
        identity_db_module, "require_models_subpath",
        lambda path, root="/models": str(path),
    )


def vec(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim).astype(np.float32)


# ── 维度无关性 ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dim", [64, VOICE_DIM, 256, 512])
def test_roundtrip_at_any_dim(tmp_path, dim):
    db = IdentityDB(db_dir=str(tmp_path), dim=dim)
    record = db.add("小王", [vec(dim, 1)], profile={"team": "运营"})
    assert record["name"] == "小王"

    reopened = IdentityDB(db_dir=str(tmp_path), dim=dim)
    again = reopened.get_person(record["id"])
    assert again["name"] == "小王"
    assert again["profile"] == {"team": "运营"}
    # 同一条 embedding 必须认回同一个人，且是余弦 1.0
    pid, score = reopened.match(vec(dim, 1), threshold=0.5)
    assert pid == record["id"]
    assert score == pytest.approx(1.0, abs=1e-5)


def test_wrong_width_embedding_is_rejected(tmp_path):
    db = IdentityDB(db_dir=str(tmp_path), dim=VOICE_DIM)
    with pytest.raises(ValueError, match=f"must have {VOICE_DIM} dims"):
        db.add("错宽度", [vec(512, 2)])


@pytest.mark.parametrize("bad", [0, -1])
def test_nonpositive_dim_refused(tmp_path, bad):
    with pytest.raises(ValueError, match="dim must be positive"):
        IdentityDB(db_dir=str(tmp_path), dim=bad)


def test_two_dims_coexist_in_separate_dirs(tmp_path):
    """face 和声纹各自一个目录，互不影响 —— 这是生产里的实际布局。"""
    faces = IdentityDB(db_dir=str(tmp_path / "face_db"), dim=512, label="face_db")
    voices = IdentityDB(db_dir=str(tmp_path / "voice_db"), dim=VOICE_DIM,
                        label="voice_db")
    f = faces.add("脸", [vec(512, 3)])
    v = voices.add("声", [vec(VOICE_DIM, 3)])
    # 两边都从 p-1 开始：id 命名空间是 per-database 的，不是全局的
    assert f["id"] == v["id"] == "p-1"
    assert faces.get_person("p-1")["name"] == "脸"
    assert voices.get_person("p-1")["name"] == "声"


# ── 换维度 / 换模型的拒绝 ─────────────────────────────────────────────────────

def test_dim_mismatch_on_load_is_refused(tmp_path):
    IdentityDB(db_dir=str(tmp_path), dim=VOICE_DIM).add("甲", [vec(VOICE_DIM, 4)])
    with pytest.raises(IdentityDBError) as excinfo:
        IdentityDB(db_dir=str(tmp_path), dim=512)
    message = str(excinfo.value)
    assert "192-dim" in message and "512" in message
    # 必须说出补救办法，否则读到这条错的人只能猜
    assert "re-enrol" in message


def test_model_mismatch_on_load_is_refused(tmp_path):
    IdentityDB(db_dir=str(tmp_path), dim=VOICE_DIM,
               model="campplus_zh_en").add("乙", [vec(VOICE_DIM, 5)])
    with pytest.raises(IdentityDBError) as excinfo:
        IdentityDB(db_dir=str(tmp_path), dim=VOICE_DIM, model="eres2netv2_zh")
    message = str(excinfo.value)
    assert "campplus_zh_en" in message and "eres2netv2_zh" in message


def test_same_model_reopens_fine(tmp_path):
    IdentityDB(db_dir=str(tmp_path), dim=VOICE_DIM,
               model="campplus_zh_en").add("丙", [vec(VOICE_DIM, 6)])
    again = IdentityDB(db_dir=str(tmp_path), dim=VOICE_DIM, model="campplus_zh_en")
    assert again.get_person("p-1")["name"] == "丙"


def test_legacy_db_without_model_or_dim_still_opens(tmp_path):
    """每台已部署机器上的 face_db 都没有这两个字段，不能因此打不开。

    这是真实的迁移路径，不是假想的：persons.json 的 version 2 没有 dim/model，
    而拒绝加载会让升级后的 face 卡片直接进 error 状态。
    """
    db = IdentityDB(db_dir=str(tmp_path), dim=512, model="buffalo_sc")
    db.add("老库", [vec(512, 7)])

    persons_path = os.path.join(str(tmp_path), "persons.json")
    with open(persons_path, encoding="utf-8") as handle:
        state = json.load(handle)
    state["version"] = 2
    del state["dim"]
    del state["model"]
    with open(persons_path, "w", encoding="utf-8") as handle:
        json.dump(state, handle)

    # 两种 model 都应打开：缺字段意味着「不知道」，而不是「不匹配」
    assert IdentityDB(db_dir=str(tmp_path), dim=512,
                      model="buffalo_sc").get_person("p-1")["name"] == "老库"
    assert IdentityDB(db_dir=str(tmp_path), dim=512,
                      model="something_else").get_person("p-1")["name"] == "老库"


def test_unlabelled_instance_opens_a_labelled_db(tmp_path):
    """调用方不声明 model 时不该被拦 —— 它没能力被保护，拦它只是碍事。"""
    IdentityDB(db_dir=str(tmp_path), dim=VOICE_DIM,
               model="campplus_zh_en").add("丁", [vec(VOICE_DIM, 8)])
    assert IdentityDB(db_dir=str(tmp_path),
                      dim=VOICE_DIM).get_person("p-1")["name"] == "丁"


def test_dim_is_checked_before_the_generic_consistency_error(tmp_path):
    """维度不符要报维度，不能报成「数据库不一致」—— 后者把人送去查文件损坏。"""
    IdentityDB(db_dir=str(tmp_path), dim=VOICE_DIM).add("戊", [vec(VOICE_DIM, 9)])
    with pytest.raises(IdentityDBError) as excinfo:
        IdentityDB(db_dir=str(tmp_path), dim=256)
    assert "inconsistent" not in str(excinfo.value)
