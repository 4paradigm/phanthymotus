"""
tests/test_action_vocab.py — 126-action vocabulary contract tests.

A wrong byte in one class name silently breaks exact-match consumers. What is
covered here is everything that fails silently if it is wrong — the table's
size, id coverage, name uniqueness, the byte-level spellings in the specification,
exact-match (never folded) lookups, and the overlap with the NTU-60 names
plugins/pose_stgcn.py carries.

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest perception/tests -q
"""

from __future__ import annotations

import hashlib
import json

import pytest

from plugins.action_vocab import (  # noqa: E402
    DIRECTION_SENSITIVE_PAIRS,
    NONE_CLASS_ID,
    OFFICIAL_126_CLASSES,
    canonical_form,
    id_for_name,
    id_for_zh,
    name_for_id,
    table_sha256,
    zh_for_id,
)
from plugins.pose_stgcn import NTU60, USABLE_CLASSES  # noqa: E402


def test_table_shape() -> None:
    assert len(OFFICIAL_126_CLASSES) == 126
    assert sorted(OFFICIAL_126_CLASSES) == list(range(1, 127))


def test_names_unique() -> None:
    names = [en for _, en in OFFICIAL_126_CLASSES.values()]
    zh = [zh for zh, _ in OFFICIAL_126_CLASSES.values()]
    assert len(set(names)) == 126
    assert len(set(zh)) == 126


def test_table_pinned_by_sha256() -> None:
    """The whole table, pinned. Any edit — name, id, spelling, slash — moves
    this hash, and the change must be a deliberate one."""
    assert table_sha256() == (
        "12c66e555d071ed97642b214705fdd2bae6fd8459b12e6586eb64d47063db0c5")


def test_canonical_form_is_deterministic_json() -> None:
    parsed = json.loads(canonical_form())
    assert len(parsed) == 126
    assert parsed["87"] == ["比OK手势", "make OK sign"]
    # Re-serialising the parsed form reproduces the same bytes.
    assert json.dumps(parsed, ensure_ascii=False, sort_keys=True) == canonical_form()


# The spellings a normalisation pass would destroy. Each case is the official
# one, copied from the introduction document — not "close enough".
BYTE_TRAPS = {
    33: ("闻气味", "sniff/smell"),            # ASCII slash, no spaces
    68: ("抛帽子", "throw up cap/hat"),        # ASCII slash, no spaces
    87: ("比OK手势", "make OK sign"),          # capital OK
    96: ("点头/鞠躬", "nod head / bow"),       # spaced slash in English only
    114: ("引起注意", "Attention"),            # capitalised (CTPGesture)
    115: ("召集／集合", "Rally"),              # full-width U+FF0F in Chinese
    116: ("后退", "MoveInReverse"),            # camel case (CTPGesture)
    117: ("打喷嚏/咳嗽", "sneeze / cough"),    # spaced slash in English only
    124: ("恶心/呕吐", "nausea / vomiting"),   # spaced slash in English only
}


@pytest.mark.parametrize("class_id", sorted(BYTE_TRAPS))
def test_byte_trap_spellings(class_id: int) -> None:
    zh, en = BYTE_TRAPS[class_id]
    assert OFFICIAL_126_CLASSES[class_id] == (zh, en)


def test_full_width_slash_is_not_the_half_width_one() -> None:
    assert id_for_zh("召集／集合") == 115
    assert id_for_zh("召集/集合") is None


def test_lookups_are_exact_not_folded() -> None:
    assert id_for_name("Attention") == 114
    assert id_for_name("attention") is None
    assert id_for_name("make OK sign") == 87
    assert id_for_name("make ok sign") is None
    assert id_for_name("MoveInReverse") == 116
    assert id_for_name("moveinreverse") is None


def test_none_class_is_a_real_row() -> None:
    assert OFFICIAL_126_CLASSES[NONE_CLASS_ID] == ("无动作", "none")


def test_round_trip() -> None:
    for i, (zh, en) in OFFICIAL_126_CLASSES.items():
        assert id_for_name(en) == i
        assert id_for_zh(zh) == i
        assert name_for_id(i) == en
        assert zh_for_id(i) == zh
    assert name_for_id(0) is None
    assert name_for_id(127) is None


def test_direction_pairs_are_valid_and_distinct() -> None:
    ids = set(OFFICIAL_126_CLASSES)
    seen = set()
    for a, b in DIRECTION_SENSITIVE_PAIRS:
        assert a in ids and b in ids
        assert a != b
        assert a not in seen and b not in seen
        seen.update((a, b))
    assert len(DIRECTION_SENSITIVE_PAIRS) == 12


def test_ntu60_usable_names_all_present() -> None:
    """Every single-person NTU-60 class the stgcn backend can emit is a 126
    class under the same English name, so skeleton-model vocabularies transfer
    by exact name. The 11 two-person classes must NOT be here."""
    for idx in USABLE_CLASSES:
        assert id_for_name(NTU60[idx][0]) is not None, NTU60[idx][0]
    from plugins.pose_stgcn import MUTUAL_CLASSES
    for idx in MUTUAL_CLASSES:
        assert id_for_name(NTU60[idx][0]) is None, NTU60[idx][0]


def test_hash_is_sha256_of_canonical_form() -> None:
    digest = hashlib.sha256(canonical_form().encode("utf-8")).hexdigest()
    assert digest == table_sha256()
