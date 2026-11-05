"""Describe OCR text layout with fixed templates and no image/model calls.

Regions are text groups inferred from whitespace, not detected screens or
boards. Single fragments are valid groups, so a missing physical outline
never prevents a summary. The raw OCR payload remains the evidence record.
"""
from __future__ import annotations

from statistics import median
import re
import unicodedata


_NUMBER_TEXT = re.compile(
    r"[¥$€£]?[+-]?[0-9OoIlLSsBZbz][0-9OoIlLSsBZbz,.]*(?:[x×]\d+)?"
    r"(?:万|千|亿)?(?:%|\+|元|人|家|项|次|台|个|年|月|天|米|秒|kg|km|cm|mm|m|g)?[↑↓]?"
)
_DATE_TEXT = re.compile(
    r"((?:19|20)\d{2})(?:[./-](\d{1,2})(?:[./-](\d{1,2}))?|H([12]))?"
)
_CLOCK_TEXT = re.compile(r"(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?")


def _reading_order(item):
    x1, y1, x2, y2 = item["bbox"]
    return y1, x1, y2, x2, item["text"]


def _gaps(items, axis):
    intervals = sorted((item["bbox"][axis], item["bbox"][axis + 2]) for item in items)
    end = intervals[0][1]
    gaps = []
    for start, next_end in intervals[1:]:
        if start > end:
            gaps.append((start - end, (start + end) / 2))
        end = max(end, next_end)
    return gaps


def _layout_groups(items):
    if len(items) < 2:
        return [items]
    height = median(max(1, item["bbox"][3] - item["bbox"][1]) for item in items)
    glyph_height = median(_font_size(item) for item in items)
    horizontal_factor = 2.5
    # A tilted long line has a tall axis-aligned box. That height must not
    # bridge the whitespace to an unrelated sign or a distant lower label.
    if height > 1.5 * glyph_height and _gaps(items, 0):
        height = min(_font_size(item) for item in items) if len(items) == 2 else glyph_height
        horizontal_factor = 2.
    splits = []
    # A larger vertical allowance keeps ordinary line spacing together.
    for axis, factor in ((0, horizontal_factor), (1, 4.0)):
        for gap, split in _gaps(items, axis):
            if gap >= factor * height:
                splits.append((gap / (factor * height), -axis, split))
    if not splits:
        return [items]
    _, neg_axis, split = max(splits)
    axis = -neg_axis
    first = [item for item in items if item["bbox"][axis + 2] <= split]
    second = [item for item in items if item["bbox"][axis] >= split]
    return _layout_groups(first) + _layout_groups(second)


def _font_size(item):
    """Conservative glyph-size proxy; a tilted long line has a tall bbox."""
    x1, y1, x2, y2 = item["bbox"]
    units = sum(1.0 if unicodedata.east_asian_width(char) in {"W", "F"} else .5
                for char in item["text"] if not char.isspace())
    return min(max(1, y2 - y1), max(1, x2 - x1) / max(1, units))


def _unique_items(items):
    # Repeated numeric observations may be different table cells or prices.
    # Only literal non-numeric repeats can share a presentation entry.
    unique = {}
    numeric = []
    for item in sorted(items, key=_reading_order):
        if any(char.isdigit() for char in item["text"]):
            numeric.append(item)
            continue
        previous = unique.get(item["text"])
        if previous is None or _font_size(item) > _font_size(previous):
            unique[item["text"]] = item
    return sorted([*unique.values(), *numeric], key=_reading_order)


def _quotes(items):
    return "、".join(
        f"“{item['text']}”" + ("（OCR 分数较低，待核实）" if item.get("score", 1.) < .95 else "")
        for item in _unique_items(items)
    )


def _title_candidates(members):
    if len(members) < 2:
        return []
    top = min(item["bbox"][1] for item in members)
    bottom = max(item["bbox"][3] for item in members)
    upper = []
    for item in members:
        text = unicodedata.normalize("NFKC", item["text"]).replace(" ", "")
        numeric = any(char.isdigit() for char in text) and (_NUMBER_TEXT.fullmatch(text) or _CLOCK_TEXT.fullmatch(text))
        if not numeric and item["bbox"][1] <= top + .4 * (bottom - top):
            upper.append(item)
    if not upper:
        return []
    largest = max(_font_size(item) for item in upper)
    typical = median(_font_size(item) for item in members)
    if largest < 1.35 * typical:
        return []
    return [item for item in upper if _font_size(item) >= .8 * largest]


def _bounds(members):
    return [min(item["bbox"][0] for item in members),
            min(item["bbox"][1] for item in members),
            max(item["bbox"][2] for item in members),
            max(item["bbox"][3] for item in members)]


def _row_order(members):
    """Recognize a gently tilted row, without inferring a shared source."""
    if len(members) < 3 or any(_numeric_observation(item) or _date_key(item) is not None for item in members):
        return None
    ordered = sorted(members, key=lambda item: (item["bbox"][0], _reading_order(item)))
    if any(left["bbox"][2] >= right["bbox"][0] for left, right in zip(ordered, ordered[1:])):
        return None
    sizes = [_font_size(item) for item in ordered]
    if max(sizes) > 2 * min(sizes):
        return None
    centers = [((item["bbox"][0] + item["bbox"][2]) / 2,
                (item["bbox"][1] + item["bbox"][3]) / 2) for item in ordered]
    slopes = [(y2 - y1) / (x2 - x1) for (x1, y1), (x2, y2) in zip(centers, centers[1:])]
    slope = median(slopes)
    intercept = median(y - slope * x for x, y in centers)
    if abs(slope) > .2 or any(abs(y - slope * x - intercept) > .6 * size
                               for (x, y), size in zip(centers, sizes)):
        return None
    return ordered


def _merge_text_rows(groups):
    """Keep aligned singleton labels together across whitespace splits."""
    groups = list(groups)
    while True:
        singletons = [(index, group[0]) for index, group in enumerate(groups)
                      if len(group) == 1 and not _numeric_observation(group[0])
                      and _date_key(group[0]) is None]
        candidates = []
        for offset, (first_index, first) in enumerate(singletons):
            fx, fy = [(first["bbox"][axis] + first["bbox"][axis + 2]) / 2 for axis in (0, 1)]
            for second_index, second in singletons[offset + 1:]:
                sx, sy = [(second["bbox"][axis] + second["bbox"][axis + 2]) / 2 for axis in (0, 1)]
                if sx == fx or abs((sy - fy) / (sx - fx)) > .2:
                    continue
                slope = (sy - fy) / (sx - fx)
                aligned = [(index, item) for index, item in singletons
                           if abs((item["bbox"][1] + item["bbox"][3]) / 2 - fy
                                  - slope * ((item["bbox"][0] + item["bbox"][2]) / 2 - fx))
                           <= .6 * _font_size(item)]
                row = _row_order([item for index, item in aligned])
                if row:
                    candidates.append((len(row), tuple(sorted(index for index, item in aligned)), row))
        if not candidates:
            return groups
        _, indices, row = min(candidates, key=lambda candidate: (-candidate[0], candidate[1]))
        groups = [group for index, group in enumerate(groups) if index not in indices] + [row]


def _date_key(item):
    text = unicodedata.normalize("NFKC", item["text"]).strip()
    match = _DATE_TEXT.fullmatch(text)
    if not match:
        return None
    year, month, day, half = match.groups()
    if (month and not 1 <= int(month) <= 12) or (day and not 1 <= int(day) <= 31):
        return None
    return int(year), int(month or (int(half) * 6 if half else 0)), int(day or 0)


def _vertical_blocks(members):
    """Keep nearby, aligned lines together without bridging adjacent columns.

    Each observation can have one predecessor and one successor. A wide line
    cannot join several columns through overlapping text boxes.
    """
    ordered = sorted(members, key=_reading_order)
    edges = []
    for lower_index, lower in enumerate(ordered):
        lx1, ly1, lx2, ly2 = lower["bbox"]
        for upper_index, upper in enumerate(ordered[:lower_index]):
            ux1, uy1, ux2, uy2 = upper["bbox"]
            scale = max(1, min(ly2 - ly1, uy2 - uy1))
            if (ly1 + ly2 - uy1 - uy2) / 2 < .4 * scale:
                continue
            gap = ly1 - uy2
            if gap > scale or min(ly2, uy2) - max(ly1, uy1) > .65 * scale:
                continue
            left_offset = abs(lx1 - ux1)
            center_offset = abs(lx1 + lx2 - ux1 - ux2) / 2
            width_ratio = max(lx2 - lx1, ux2 - ux1) / max(1, min(lx2 - lx1, ux2 - ux1))
            if left_offset > .75 * scale and not (width_ratio < 1.5 and center_offset < .5 * scale):
                continue
            edges.append((max(0, gap) + 2 * min(left_offset, center_offset), upper_index, lower_index))
    successor, predecessor = {}, {}
    for _distance, upper, lower in sorted(edges):
        if upper not in successor and lower not in predecessor:
            successor[upper] = lower
            predecessor[lower] = upper
    blocks = []
    for index in range(len(ordered)):
        if index in predecessor:
            continue
        chain = [ordered[index]]
        while index in successor:
            index = successor[index]
            chain.append(ordered[index])
        blocks.append(chain)
    return blocks


def _dated_blocks(members):
    """Organize date markers and unambiguous aligned text, without filling gaps.

    Association describes geometry only. Text above a marker, between equally
    plausible columns, or beyond the next marker in that column stays separate.
    """
    anchors = [item for item in members if _date_key(item) is not None]
    if len({_date_key(item) for item in anchors}) < 4:
        return None
    x1, _y1, x2, _y2 = _bounds(members)
    if max(item["bbox"][0] for item in anchors) - min(item["bbox"][0] for item in anchors) < .4 * (x2 - x1):
        return None
    groups = [[anchor] for anchor in anchors]
    remaining = []
    for item in sorted(members, key=_reading_order):
        if any(item is anchor for anchor in anchors):
            continue
        bx1, by1, bx2, by2 = item["bbox"]
        candidates = []
        for index, anchor in enumerate(anchors):
            ax1, ay1, ax2, ay2 = anchor["bbox"]
            scale = max(1, ay2 - ay1)
            offset = abs(bx1 - ax1)
            if offset > max(1.4 * scale, .5 * (ax2 - ax1)):
                continue
            previous_bottom = max(member["bbox"][3] for member in groups[index])
            if (by1 + by2) / 2 <= (ay1 + ay2) / 2 or by1 - previous_bottom > 6 * scale:
                continue
            following = [other["bbox"][1] for other in anchors
                         if other["bbox"][1] > ay1 and abs(other["bbox"][0] - ax1) <= max(1.4 * scale, .5 * (ax2 - ax1))]
            if following and by1 >= min(following):
                continue
            candidates.append((offset + .15 * max(0, by1 - ay2), index))
        candidates.sort()
        if not candidates or (len(candidates) > 1 and candidates[1][0] - candidates[0][0] < .5 * max(1, by2 - by1)):
            remaining.append(item)
        else:
            groups[candidates[0][1]].append(item)
    groups.sort(key=lambda group: (_date_key(group[0]), _reading_order(group[0])))
    return groups, remaining


def _local_position(box, bounds):
    x1, y1, x2, y2 = bounds
    x = ((box[0] + box[2]) / 2 - x1) / max(1, x2 - x1)
    y = ((box[1] + box[3]) / 2 - y1) / max(1, y2 - y1)
    return (0 if x < 1 / 3 else 2 if x > 2 / 3 else 1,
            0 if y < 1 / 3 else 2 if y > 2 / 3 else 1)


def _numeric_observation(item):
    text = unicodedata.normalize("NFKC", item["text"]).replace(" ", "").strip("()")
    return bool(_NUMBER_TEXT.fullmatch(text))


def _local_layout(members):
    bounds = _bounds(members)
    typical = median(_font_size(item) for item in members)
    prominent = [item for item in members
                 if _date_key(item) is None
                 and not _numeric_observation(item)
                 and _font_size(item) >= 1.35 * typical
                 and item["bbox"][2] - item["bbox"][0] >= .35 * (bounds[2] - bounds[0])]
    body = [item for item in members if not any(item is title for title in prominent)]
    # A dense numeric table needs row/column matching. Merely listing its
    # numeric columns separately loses useful row context; keep the fallback.
    numeric = sum(_numeric_observation(item) for item in body)
    if body and numeric >= len(body) / 2:
        return None
    blocks = _vertical_blocks(body)
    # Refine only genuinely two-dimensional groups. Ordinary paragraphs and
    # small title/body groups retain their existing concise presentation.
    columns = {_local_position(_bounds(block), bounds)[0] for block in blocks}
    if len(blocks) < 3 or len(columns) < 2:
        return None
    parallel = False
    for index, block in enumerate(blocks):
        first = _bounds(block)
        for other in blocks[index + 1:]:
            second = _bounds(other)
            if _local_position(first, bounds)[0] == _local_position(second, bounds)[0]:
                continue
            horizontal_overlap = min(first[2], second[2]) - max(first[0], second[0])
            if horizontal_overlap > .3 * min(first[2] - first[0], second[2] - second[0]):
                continue
            overlap = min(first[3], second[3]) - max(first[1], second[1])
            if overlap > .3 * min(first[3] - first[1], second[3] - second[1]):
                parallel = True
                break
        if parallel:
            break
    if not parallel:
        return None
    return bounds, prominent, blocks


def _describe_local_layout(layout):
    bounds, prominent, blocks = layout
    clauses = []
    for title in sorted(prominent, key=_reading_order):
        vertical = _local_position(title["bbox"], bounds)[1]
        where = ["上方", "中部", "下方"][vertical]
        clauses.append(f"组内{where}醒目文字为{_quotes([title])}")
    for column, label in enumerate(["偏左", "中间", "偏右"]):
        located = [block for block in blocks if _local_position(_bounds(block), bounds)[0] == column]
        if not located:
            continue
        located.sort(key=lambda block: _reading_order({"bbox": _bounds(block), "text": ""}))
        entries = []
        for block in located:
            vertical = _local_position(_bounds(block), bounds)[1]
            entries.append(f"{['上部', '中部', '下部'][vertical]}文字块为{_quotes(block)}")
        clauses.append(f"组内{label}，从上到下：" + "；".join(entries))
    return "；".join(clauses)


def _describe_group(members):
    row = _row_order(members)
    if row is not None:
        return "同一横排文字从左到右为" + "、".join(_quotes([item]) for item in row)
    dated = _dated_blocks(members)
    if dated is not None:
        groups, remaining = dated
        clauses = []
        for group in groups:
            clause = _quotes(group[:1])
            if len(group) > 1:
                clause += f"，同列下方原文为{_quotes(group[1:])}"
            clauses.append(clause)
        result = "日期文字按年月顺序整理（附近原文仅按位置归组）：" + "；".join(clauses)
        if remaining:
            layout = _local_layout(remaining)
            result += "；其余文字另列：" + (_describe_local_layout(layout) if layout else _quotes(remaining))
        return result
    layout = _local_layout(members)
    if layout is not None:
        return _describe_local_layout(layout)
    members = _unique_items(members)
    titles = _title_candidates(members)
    if not titles:
        return f"原文为{_quotes(members)}"
    first = min(sum(item["bbox"][1::2]) / 2 for item in titles)
    last = max(sum(item["bbox"][1::2]) / 2 for item in titles)
    other = [item for item in members if not any(item is title for title in titles)]
    before = [item for item in other if sum(item["bbox"][1::2]) / 2 < first]
    after = [item for item in other if sum(item["bbox"][1::2]) / 2 > last]
    nearby = [item for item in other if item not in before and item not in after]
    clauses = [f"候选标题为{_quotes(titles)}"]
    for label, entries in [("上方另有", before), ("下方原文为", after), ("附近另有", nearby)]:
        if entries:
            clauses.append(f"{label}{_quotes(entries)}")
    return "，".join(clauses)


def _describe_zone(regions, image_height):
    # Identical text in distinct locations is reported as repetition, not as
    # independent exhibits or an inferred shared source.
    repeated = {}
    for _position, _y, _x, members in regions:
        key = tuple(item["text"] for item in _unique_items(members))
        repeated.setdefault(key, []).append(members)
    clauses = []
    for occurrences in repeated.values():
        contains_numbers = any(char.isdigit() for item in occurrences[0] for char in item["text"])
        if len(occurrences) > 1 and not contains_numbers:
            representative = [item for members in occurrences for item in members]
            clauses.append(f"{len(occurrences)}处重复出现{_quotes(representative)}")
        else:
            for members in occurrences:
                clause = _describe_group(members)
                if _bounds(members)[3] - _bounds(members)[1] < image_height / 3:
                    y = next(region[1] for region in regions if region[3] is members)
                    if y < 1 / 3 or y > 2 / 3:
                        clause += f"（位于画面{'上部' if y < 1 / 3 else '下部'}）"
                clauses.append(clause)
    prefix = "按从上到下的顺序，" if len(clauses) > 1 else ""
    return prefix + "；另一个文字组中，".join(clauses)


def build_ocr_summary(items, image_size=None):
    """Return one paragraph, including every non-empty OCR fragment.

    Without frame dimensions, text is still returned but no left/right
    position is invented. Whitespace grouping always keeps singleton text.
    Numbers and labels are independent quoted observations. Coordinates are
    supplied by the OCR adapter; input items are never modified.
    """
    readable = sorted((item for item in items if item.get("text")), key=_reading_order)
    if not readable:
        return "当前画面没有识别到可用文字。"
    if image_size is None:
        return f"识别到的文字内容是{_quotes(readable)}。"

    width, height = image_size
    regions = []
    for members in _merge_text_rows(_layout_groups(readable)):
        members = sorted(members, key=_reading_order)
        x1 = min(item["bbox"][0] for item in members)
        y1 = min(item["bbox"][1] for item in members)
        x2 = max(item["bbox"][2] for item in members)
        y2 = max(item["bbox"][3] for item in members)
        x = (x1 + x2) / (2 * width)
        y = (y1 + y2) / (2 * height)
        position = "偏左" if x < 1 / 3 else "偏右" if x > 2 / 3 else "中间"
        regions.append((position, y, x, members))
    # Center is described first as a spatial convention, not a user-intent vote.
    regions.sort(key=lambda region: (
        {"中间": 0, "偏左": 1, "偏右": 2}[region[0]], region[1], region[2],
        tuple(_reading_order(item) for item in region[3]),
    ))
    descriptions = []
    for position in ["中间", "偏左", "偏右"]:
        located = [region for region in regions if region[0] == position]
        if located:
            descriptions.append(f"画面{position}：{_describe_zone(located, height)}")
    return "。".join(descriptions) + "。"
