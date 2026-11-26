#!/usr/bin/env python3
"""
plugins/action_vocab.py — fixed 126-action vocabulary, read-only.

Single-person skeleton-action models emit ids from a fixed 126-class table:
`activity` carries these exact English strings and `activity_zh` carries these
exact Chinese strings.

**Byte-level traps, all of them real.** The list mixes spellings that a single
`str.lower()` would silently destroy, so nothing in this module case-folds
anything and the lookups below are exact-match only:

* 87  "make OK sign"        — capital OK.
* 114 "Attention"           — capitalised, as CTPGesture spells it.
* 115 "召集／集合" / "Rally"  — the Chinese uses a **full-width** U+FF0F slash,
                              not the ASCII one; the English is capitalised.
* 116 "MoveInReverse"       — camel case, again CTPGesture's own spelling.
* 33 / 68 "sniff/smell", "throw up cap/hat" — ASCII slash, **no** spaces.
* 96 / 117 / 124 "nod head / bow", "sneeze / cough", "nausea / vomiting"
                              — ASCII slash **with** spaces on both sides.

NTU-120's own label files must not be substituted for this table either: this
list re-orders, drops the 11 two-person NTU classes and adds gesture, kitchen
and traffic classes NTU never had. Source-dataset class mappings live outside
this module; this file is the destination table only.

Class 38 ("无动作", "none") is part of the table: "no action" may be expressed
via this class row, via an empty `persons` array, or by staying under the
rejection threshold depending on the caller's configuration.
"""

from __future__ import annotations

import hashlib
import json

#: Class table: id -> (name_zh, name). Verbatim spellings — see the module
#: docstring for the entries that must never be "normalised".
OFFICIAL_126_CLASSES = {
    1: ("喝水", "drink water"), 2: ("吃东西", "eat meal"), 3: ("刷牙", "brush teeth"),
    4: ("梳头", "brush hair"), 5: ("穿外套", "wear jacket"), 6: ("脱外套", "take off jacket"),
    7: ("穿鞋", "wear a shoe"), 8: ("脱鞋", "take off a shoe"), 9: ("戴眼镜", "wear on glasses"),
    10: ("摘眼镜", "take off glasses"), 11: ("戴帽子", "put on a hat"), 12: ("摘帽子", "take off a hat"),
    13: ("切东西", "cutting something"), 14: ("倒液体", "pouring something"), 15: ("烹饪", "cooking something"),
    16: ("打开冰箱", "opening refrigerator"), 17: ("洗碗", "washing dishes"), 18: ("清洁马桶", "cleaning toilet"),
    19: ("擦洗窗户", "cleaning windows"), 20: ("洗衣服", "doing laundry"), 21: ("叠衣服", "folding clothes"),
    22: ("手洗衣服", "hand washing clothes"), 23: ("熨衣服", "ironing"), 24: ("整理床铺", "making the bed"),
    25: ("拖地", "mopping floor"), 26: ("搬动家具", "moving furniture"), 27: ("扫地", "sweeping floor"),
    28: ("给植物浇水", "watering plants"), 29: ("搓手", "rub two hands"), 30: ("拨头发", "flick hair"),
    31: ("擦脸", "wipe face"), 32: ("剪指甲", "cutting nails"), 33: ("闻气味", "sniff/smell"),
    34: ("脸上涂面霜", "apply cream on face"), 35: ("手上涂护手霜", "apply cream on hand"), 36: ("打哈欠", "yawn"),
    37: ("伸懒腰", "stretch oneself"), 38: ("无动作", "none"), 39: ("看书", "reading"),
    40: ("写字", "writing"), 41: ("撕纸", "tear up paper"), 42: ("东西掉了", "drop"),
    43: ("捡东西", "pickup"), 44: ("扔东西", "throw"), 45: ("踢东西", "kicking something"),
    46: ("掏口袋", "reach into pocket"), 47: ("打电话", "phone call"), 48: ("玩手机", "play with phone"),
    49: ("打字", "type on keyboard"), 50: ("自拍", "taking a selfie"), 51: ("看表", "check time"),
    52: ("戴耳机", "put on headphone"), 53: ("摘耳机", "take off headphone"), 54: ("订书", "staple book"),
    55: ("数钱", "counting money"), 56: ("剪纸", "cutting paper"), 57: ("开瓶子", "open bottle"),
    58: ("抛硬币", "toss a coin"), 59: ("折纸", "fold paper"), 60: ("揉纸成团", "ball up paper"),
    61: ("玩魔方", "play magic cube"), 62: ("背上包", "put on bag"), 63: ("取下包", "take off bag"),
    64: ("把物品放进包里", "put object into bag"), 65: ("从包里取出物品", "take object out of bag"), 66: ("打开盒子", "open a box"),
    67: ("搬重物", "move heavy objects"), 68: ("抛帽子", "throw up cap/hat"), 69: ("蹲下", "squat down"),
    70: ("坐下", "sit down"), 71: ("起身", "stand up"), 72: ("单脚跳", "hopping"),
    73: ("跳起", "jump up"), 74: ("投篮", "shoot at basket"), 75: ("拍球", "bounce ball"),
    76: ("挥网球拍", "tennis bat swing"), 77: ("颠乒乓球", "juggle table tennis ball"), 78: ("手臂绕环", "arm circles"),
    79: ("手臂摆动", "arm swings"), 80: ("原地跑步", "run on the spot"), 81: ("后踢腿", "butt kicks"),
    82: ("交叉摸脚尖", "cross toe touch"), 83: ("侧踢腿", "side kick"), 84: ("做嘘声手势", "hush"),
    85: ("竖大拇指", "thumb up"), 86: ("竖大拇指向下", "thumb down"), 87: ("比OK手势", "make OK sign"),
    88: ("比胜利手势", "make victory sign"), 89: ("打响指", "snap fingers"), 90: ("挥拳", "shake fist"),
    91: ("鼓掌", "clapping"), 92: ("敬礼", "salute"), 93: ("欢呼", "cheer up"),
    94: ("挥手", "hand waving"), 95: ("指向某处", "point to something"), 96: ("点头/鞠躬", "nod head / bow"),
    97: ("摇头", "shake head"), 98: ("合掌", "put palms together"), 99: ("双手交叉(制止)", "cross hands in front"),
    100: ("举手投降", "capitulate"), 101: ("双臂交叉抱胸", "cross arms"), 102: ("跟随", "follow"),
    103: ("行走", "walking"), 104: ("停止", "stop"), 105: ("转向", "turn"),
    106: ("过来", "come here"), 107: ("直行", "forward"), 108: ("左转", "left turn"),
    109: ("左转待转", "left turn waiting"), 110: ("右转", "right turn"), 111: ("变道", "lane changing"),
    112: ("减速", "slow down"), 113: ("靠边停车", "pull over"), 114: ("引起注意", "Attention"),
    115: ("召集／集合", "Rally"), 116: ("后退", "MoveInReverse"), 117: ("打喷嚏/咳嗽", "sneeze / cough"),
    118: ("踉跄", "staggering"), 119: ("跌倒", "falling down"), 120: ("捂头(头痛)", "touch head"),
    121: ("捂胸(胸痛)", "touch chest"), 122: ("捂背(背痛)", "touch back"), 123: ("捂脖子", "touch neck"),
    124: ("恶心/呕吐", "nausea / vomiting"), 125: ("扇风(热)", "fan self"), 126: ("擤鼻涕", "blow nose"),
}

#: The "no action" class id in OFFICIAL_126_CLASSES.
NONE_CLASS_ID = 38

#: Twelve pairs the docstring of the training pipeline must never flip: if the
#: augmentation swaps one of these into its partner the label is not "close",
#: it is the opposite action. Used by the data builders to forbid blind
#: horizontal flips / temporal reversal on these ids.
DIRECTION_SENSITIVE_PAIRS = (
    (5, 6),     # wear / take off jacket
    (7, 8),     # wear / take off a shoe
    (9, 10),    # wear on / take off glasses
    (11, 12),   # put on / take off a hat
    (52, 53),   # put on / take off headphone
    (62, 63),   # put on / take off bag
    (64, 65),   # put object into / out of bag
    (70, 71),   # sit down / stand up
    (42, 43),   # drop / pickup
    (85, 86),   # thumb up / thumb down
    (108, 110), # left / right turn
    (107, 116), # forward / MoveInReverse
)


def canonical_form() -> str:
    """The table as a stable string, for pinning and diffing.

    `json.dumps` with sorted ids and `ensure_ascii=False` keeps the Chinese
    readable; any edit to a name, id or order changes this string and breaks
    the SHA-256 the test pins.
    """
    return json.dumps(
        {str(i): list(v) for i, v in sorted(OFFICIAL_126_CLASSES.items())},
        ensure_ascii=False, sort_keys=True)


def table_sha256() -> str:
    return hashlib.sha256(canonical_form().encode("utf-8")).hexdigest()


def _build_reverse() -> tuple[dict, dict]:
    by_name, by_zh = {}, {}
    for i, (zh, name) in OFFICIAL_126_CLASSES.items():
        if name in by_name:
            raise ValueError(f"duplicate English name {name!r}")
        if zh in by_zh:
            raise ValueError(f"duplicate Chinese name {zh!r}")
        by_name[name] = i
        by_zh[zh] = i
    return by_name, by_zh


_BY_NAME, _BY_ZH = _build_reverse()


def id_for_name(name: str) -> int | None:
    """Exact-match English lookup. Deliberately not case-folded: 114 must be
    found as "Attention", not "attention" (which returns None)."""
    return _BY_NAME.get(name)


def id_for_zh(zh: str) -> int | None:
    """Exact-match Chinese lookup, full-width and half-width slashes distinct."""
    return _BY_ZH.get(zh)


def name_for_id(i: int) -> str | None:
    entry = OFFICIAL_126_CLASSES.get(i)
    return entry[1] if entry else None


def zh_for_id(i: int) -> str | None:
    entry = OFFICIAL_126_CLASSES.get(i)
    return entry[0] if entry else None
