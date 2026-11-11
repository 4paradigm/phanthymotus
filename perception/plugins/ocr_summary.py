"""Describe OCR layout from existing text boxes without model calls.

Original observations remain untouched. Ambiguous associations stay separate."""
from __future__ import annotations
from statistics import median
import math
import re
import unicodedata
_NUMBER_TEXT = re.compile('[¥$€£]?[+-]?[0-9OoIlLSsBZbz][0-9OoIlLSsBZbz,.]*(?:[x×]\\d+)?(?:万|千|亿)?(?:%|\\+|元|人|家|项|次|台|个|年|月|天|米|秒|kg|km|cm|mm|m|g)?[↑↓]?')
_DATE_TEXT = re.compile('((?:19|20)\\d{2})(?:[./-](\\d{1,2})(?:[./-](\\d{1,2}))?|H([12]))?')
_CLOCK_TEXT = re.compile('(?:[01]?\\d|2[0-3]):[0-5]\\d(?::[0-5]\\d)?')

def _reading_order(item):
    (x1, y1, x2, y2) = item['bbox']
    return (y1, x1, y2, x2, item['text'])

def _gaps(items, axis):
    intervals = sorted(((item['bbox'][axis], item['bbox'][axis + 2]) for item in items))
    end = intervals[0][1]
    gaps = []
    for (start, next_end) in intervals[1:]:
        if start > end:
            gaps.append((start - end, (start + end) / 2))
        end = max(end, next_end)
    return gaps

def _layout_groups(items):
    if len(items) < 2:
        return [items]
    height = median((max(1, item['bbox'][3] - item['bbox'][1]) for item in items))
    glyph_height = median((_font_size(item) for item in items))
    horizontal_factor = 2.5
    if height > 1.5 * glyph_height and _gaps(items, 0):
        height = min((_font_size(item) for item in items)) if len(items) == 2 else glyph_height
        horizontal_factor = 2.0
    splits = []
    for (axis, factor) in ((0, horizontal_factor), (1, 4.0)):
        for (gap, split) in _gaps(items, axis):
            if gap >= factor * height:
                splits.append((gap / (factor * height), -axis, split))
    if not splits:
        return [items]
    (_, neg_axis, split) = max(splits)
    axis = -neg_axis
    first = [item for item in items if item['bbox'][axis + 2] <= split]
    second = [item for item in items if item['bbox'][axis] >= split]
    return _layout_groups(first) + _layout_groups(second)

def _font_size(item):
    """Conservative glyph-size proxy; a tilted long line has a tall bbox."""
    (x1, y1, x2, y2) = item['bbox']
    units = sum((1.0 if unicodedata.east_asian_width(char) in {'W', 'F'} else 0.5 for char in item['text'] if not char.isspace()))
    return min(max(1, y2 - y1), max(1, x2 - x1) / max(1, units))

def _unique_items(items):
    unique = {}
    numeric = []
    for item in sorted(items, key=_reading_order):
        if any((char.isdigit() for char in item['text'])):
            numeric.append(item)
            continue
        previous = unique.get(item['text'])
        if previous is None or _font_size(item) > _font_size(previous):
            unique[item['text']] = item
    return sorted([*unique.values(), *numeric], key=_reading_order)

def _quotes(items):
    return '、'.join((f"“{item['text']}”" + ('（OCR 分数较低，待核实）' if item.get('score', 1.0) < 0.95 else '') for item in _unique_items(items)))

def _title_candidates(members):
    if len(members) < 2:
        return []
    top = min((item['bbox'][1] for item in members))
    bottom = max((item['bbox'][3] for item in members))
    upper = []
    for item in members:
        text = unicodedata.normalize('NFKC', item['text']).replace(' ', '')
        numeric = any((char.isdigit() for char in text)) and (_NUMBER_TEXT.fullmatch(text) or _CLOCK_TEXT.fullmatch(text))
        if not numeric and item['bbox'][1] <= top + 0.4 * (bottom - top):
            upper.append(item)
    if not upper:
        return []
    largest = max((_font_size(item) for item in upper))
    typical = median((_font_size(item) for item in members))
    if largest < 1.35 * typical:
        return []
    return [item for item in upper if _font_size(item) >= 0.8 * largest]

def _bounds(members):
    return [min((item['bbox'][0] for item in members)), min((item['bbox'][1] for item in members)), max((item['bbox'][2] for item in members)), max((item['bbox'][3] for item in members))]

def _row_order(members):
    """Recognize a gently tilted row, without inferring a shared source."""
    if len(members) < 3 or any((_numeric_observation(item) or _date_key(item) is not None for item in members)):
        return None
    ordered = sorted(members, key=lambda item: (item['bbox'][0], _reading_order(item)))
    if any((left['bbox'][2] >= right['bbox'][0] for (left, right) in zip(ordered, ordered[1:]))):
        return None
    sizes = [_font_size(item) for item in ordered]
    if max(sizes) > 2 * min(sizes):
        return None
    centers = [((item['bbox'][0] + item['bbox'][2]) / 2, (item['bbox'][1] + item['bbox'][3]) / 2) for item in ordered]
    slopes = [(y2 - y1) / (x2 - x1) for ((x1, y1), (x2, y2)) in zip(centers, centers[1:])]
    slope = median(slopes)
    intercept = median((y - slope * x for (x, y) in centers))
    if abs(slope) > 0.2 or any((abs(y - slope * x - intercept) > 0.6 * size for ((x, y), size) in zip(centers, sizes))):
        return None
    return ordered

def _merge_text_rows(groups):
    """Keep aligned singleton labels together across whitespace splits."""
    groups = list(groups)
    while True:
        singletons = [(index, group[0]) for (index, group) in enumerate(groups) if len(group) == 1 and (not _numeric_observation(group[0])) and (_date_key(group[0]) is None)]
        candidates = []
        for (offset, (first_index, first)) in enumerate(singletons):
            (fx, fy) = [(first['bbox'][axis] + first['bbox'][axis + 2]) / 2 for axis in (0, 1)]
            for (second_index, second) in singletons[offset + 1:]:
                (sx, sy) = [(second['bbox'][axis] + second['bbox'][axis + 2]) / 2 for axis in (0, 1)]
                if sx == fx or abs((sy - fy) / (sx - fx)) > 0.2:
                    continue
                slope = (sy - fy) / (sx - fx)
                aligned = [(index, item) for (index, item) in singletons if abs((item['bbox'][1] + item['bbox'][3]) / 2 - fy - slope * ((item['bbox'][0] + item['bbox'][2]) / 2 - fx)) <= 0.6 * _font_size(item)]
                row = _row_order([item for (index, item) in aligned])
                if row:
                    candidates.append((len(row), tuple(sorted((index for (index, item) in aligned))), row))
        if not candidates:
            return groups
        (_, indices, row) = min(candidates, key=lambda candidate: (-candidate[0], candidate[1]))
        groups = [group for (index, group) in enumerate(groups) if index not in indices] + [row]

def _date_key(item):
    text = unicodedata.normalize('NFKC', item['text']).strip()
    match = _DATE_TEXT.fullmatch(text)
    if not match:
        return None
    (year, month, day, half) = match.groups()
    if month and (not 1 <= int(month) <= 12) or (day and (not 1 <= int(day) <= 31)):
        return None
    return (int(year), int(month or (int(half) * 6 if half else 0)), int(day or 0))

def _vertical_blocks(members):
    """Keep nearby, aligned lines together without bridging adjacent columns.

    Each observation can have one predecessor and one successor. A wide line
    cannot join several columns through overlapping text boxes.
    """
    ordered = sorted(members, key=_reading_order)
    edges = []
    for (lower_index, lower) in enumerate(ordered):
        (lx1, ly1, lx2, ly2) = lower['bbox']
        for (upper_index, upper) in enumerate(ordered[:lower_index]):
            (ux1, uy1, ux2, uy2) = upper['bbox']
            scale = max(1, min(ly2 - ly1, uy2 - uy1))
            if (ly1 + ly2 - uy1 - uy2) / 2 < 0.4 * scale:
                continue
            gap = ly1 - uy2
            if gap > scale or min(ly2, uy2) - max(ly1, uy1) > 0.65 * scale:
                continue
            left_offset = abs(lx1 - ux1)
            center_offset = abs(lx1 + lx2 - ux1 - ux2) / 2
            width_ratio = max(lx2 - lx1, ux2 - ux1) / max(1, min(lx2 - lx1, ux2 - ux1))
            if left_offset > 0.75 * scale and (not (width_ratio < 1.5 and center_offset < 0.5 * scale)):
                continue
            edges.append((max(0, gap) + 2 * min(left_offset, center_offset), upper_index, lower_index))
    (successor, predecessor) = ({}, {})
    for (_distance, upper, lower) in sorted(edges):
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
    (x1, _y1, x2, _y2) = _bounds(members)
    if max((item['bbox'][0] for item in anchors)) - min((item['bbox'][0] for item in anchors)) < 0.4 * (x2 - x1):
        return None
    groups = [[anchor] for anchor in anchors]
    remaining = []
    for item in sorted(members, key=_reading_order):
        if any((item is anchor for anchor in anchors)):
            continue
        (bx1, by1, bx2, by2) = item['bbox']
        candidates = []
        for (index, anchor) in enumerate(anchors):
            (ax1, ay1, ax2, ay2) = anchor['bbox']
            scale = max(1, ay2 - ay1)
            offset = abs(bx1 - ax1)
            if offset > max(1.4 * scale, 0.5 * (ax2 - ax1)):
                continue
            previous_bottom = max((member['bbox'][3] for member in groups[index]))
            if (by1 + by2) / 2 <= (ay1 + ay2) / 2 or by1 - previous_bottom > 6 * scale:
                continue
            following = [other['bbox'][1] for other in anchors if other['bbox'][1] > ay1 and abs(other['bbox'][0] - ax1) <= max(1.4 * scale, 0.5 * (ax2 - ax1))]
            if following and by1 >= min(following):
                continue
            candidates.append((offset + 0.15 * max(0, by1 - ay2), index))
        candidates.sort()
        if not candidates or (len(candidates) > 1 and candidates[1][0] - candidates[0][0] < 0.5 * max(1, by2 - by1)):
            remaining.append(item)
        else:
            groups[candidates[0][1]].append(item)
    groups.sort(key=lambda group: (_date_key(group[0]), _reading_order(group[0])))
    return (groups, remaining)

def _local_position(box, bounds):
    (x1, y1, x2, y2) = bounds
    x = ((box[0] + box[2]) / 2 - x1) / max(1, x2 - x1)
    y = ((box[1] + box[3]) / 2 - y1) / max(1, y2 - y1)
    return (0 if x < 1 / 3 else 2 if x > 2 / 3 else 1, 0 if y < 1 / 3 else 2 if y > 2 / 3 else 1)

def _numeric_observation(item):
    text = unicodedata.normalize('NFKC', item['text']).replace(' ', '').strip('()')
    return bool(_NUMBER_TEXT.fullmatch(text))

def _local_layout(members):
    bounds = _bounds(members)
    typical = median((_font_size(item) for item in members))
    prominent = [item for item in members if _date_key(item) is None and (not _numeric_observation(item)) and (_font_size(item) >= 1.35 * typical) and (item['bbox'][2] - item['bbox'][0] >= 0.35 * (bounds[2] - bounds[0]))]
    body = [item for item in members if not any((item is title for title in prominent))]
    numeric = sum((_numeric_observation(item) for item in body))
    if body and numeric >= len(body) / 2:
        return None
    blocks = _vertical_blocks(body)
    columns = {_local_position(_bounds(block), bounds)[0] for block in blocks}
    if len(blocks) < 3 or len(columns) < 2:
        return None
    parallel = False
    for (index, block) in enumerate(blocks):
        first = _bounds(block)
        for other in blocks[index + 1:]:
            second = _bounds(other)
            if _local_position(first, bounds)[0] == _local_position(second, bounds)[0]:
                continue
            horizontal_overlap = min(first[2], second[2]) - max(first[0], second[0])
            if horizontal_overlap > 0.3 * min(first[2] - first[0], second[2] - second[0]):
                continue
            overlap = min(first[3], second[3]) - max(first[1], second[1])
            if overlap > 0.3 * min(first[3] - first[1], second[3] - second[1]):
                parallel = True
                break
        if parallel:
            break
    if not parallel:
        return None
    return (bounds, prominent, blocks)

def _describe_local_layout(layout):
    (bounds, prominent, blocks) = layout
    clauses = []
    for title in sorted(prominent, key=_reading_order):
        vertical = _local_position(title['bbox'], bounds)[1]
        where = ['上方', '中部', '下方'][vertical]
        clauses.append(f'组内{where}醒目文字为{_quotes([title])}')
    for (column, label) in enumerate(['偏左', '中间', '偏右']):
        located = [block for block in blocks if _local_position(_bounds(block), bounds)[0] == column]
        if not located:
            continue
        located.sort(key=lambda block: _reading_order({'bbox': _bounds(block), 'text': ''}))
        entries = []
        for block in located:
            vertical = _local_position(_bounds(block), bounds)[1]
            entries.append(f"{['上部', '中部', '下部'][vertical]}文字块为{_quotes(block)}")
        clauses.append(f'组内{label}，从上到下：' + '；'.join(entries))
    return '；'.join(clauses)

def _describe_group(members):
    row = _row_order(members)
    if row is not None:
        return '同一横排文字从左到右为' + '、'.join((_quotes([item]) for item in row))
    dated = _dated_blocks(members)
    if dated is not None:
        (groups, remaining) = dated
        clauses = []
        for group in groups:
            clause = _quotes(group[:1])
            if len(group) > 1:
                clause += f'，同列下方原文为{_quotes(group[1:])}'
            clauses.append(clause)
        result = '日期文字按年月顺序整理（附近原文仅按位置归组）：' + '；'.join(clauses)
        if remaining:
            layout = _local_layout(remaining)
            result += '；其余文字另列：' + (_describe_local_layout(layout) if layout else _quotes(remaining))
        return result
    layout = _local_layout(members)
    if layout is not None:
        return _describe_local_layout(layout)
    members = _unique_items(members)
    titles = _title_candidates(members)
    if not titles:
        return f'原文为{_quotes(members)}'
    first = min((sum(item['bbox'][1::2]) / 2 for item in titles))
    last = max((sum(item['bbox'][1::2]) / 2 for item in titles))
    other = [item for item in members if not any((item is title for title in titles))]
    before = [item for item in other if sum(item['bbox'][1::2]) / 2 < first]
    after = [item for item in other if sum(item['bbox'][1::2]) / 2 > last]
    nearby = [item for item in other if item not in before and item not in after]
    clauses = [f'候选标题为{_quotes(titles)}']
    for (label, entries) in [('上方另有', before), ('下方原文为', after), ('附近另有', nearby)]:
        if entries:
            clauses.append(f'{label}{_quotes(entries)}')
    return '，'.join(clauses)

def _describe_zone(regions, image_height):
    repeated = {}
    for (_position, _y, _x, members) in regions:
        key = tuple((item['text'] for item in _unique_items(members)))
        repeated.setdefault(key, []).append(members)
    clauses = []
    emitted = set()
    for (_position, y, _x, members) in regions:
        key = tuple((item['text'] for item in _unique_items(members)))
        occurrences = repeated[key]
        contains_numbers = any((char.isdigit() for item in members for char in item['text']))
        if len(occurrences) > 1 and (not contains_numbers):
            if key in emitted:
                continue
            emitted.add(key)
            representative = [item for group in occurrences for item in group]
            clauses.append(f'{len(occurrences)}处重复出现' + _quotes(representative))
        else:
            clause = _describe_group(members)
            if _bounds(members)[3] - _bounds(members)[1] < image_height / 3:
                if y < 1 / 3 or y > 2 / 3:
                    clause += '（位于画面' + ('上部' if y < 1 / 3 else '下部') + '）'
            clauses.append(clause)
    prefix = '按从上到下的顺序，' if len(clauses) > 1 else ''
    return prefix + '；另一个文字组中，'.join(clauses)

def _positioned_summary(items, image_size=None):
    """Return one paragraph, including every non-empty OCR fragment.

    Without frame dimensions, text is still returned but no left/right
    position is invented. Whitespace grouping always keeps singleton text.
    Numbers and labels are independent quoted observations. Coordinates are
    supplied by the OCR adapter; input items are never modified.
    """
    readable = sorted((item for item in items if item.get('text')), key=_reading_order)
    if not readable:
        return '当前画面没有识别到可用文字。'
    if image_size is None:
        return f'识别到的文字内容是{_quotes(readable)}。'
    (width, height) = image_size
    regions = []
    for members in _merge_text_rows(_layout_groups(readable)):
        members = sorted(members, key=_reading_order)
        x1 = min((item['bbox'][0] for item in members))
        y1 = min((item['bbox'][1] for item in members))
        x2 = max((item['bbox'][2] for item in members))
        y2 = max((item['bbox'][3] for item in members))
        x = (x1 + x2) / (2 * width)
        y = (y1 + y2) / (2 * height)
        position = '偏左' if x < 1 / 3 else '偏右' if x > 2 / 3 else '中间'
        regions.append((position, y, x, members))
    regions.sort(key=lambda region: ({'中间': 0, '偏左': 1, '偏右': 2}[region[0]], region[1], region[2], tuple((_reading_order(item) for item in region[3]))))
    descriptions = []
    for position in ['中间', '偏左', '偏右']:
        located = [region for region in regions if region[0] == position]
        if located:
            descriptions.append(f'画面{position}：{_describe_zone(located, height)}')
    return '。'.join(descriptions) + '。'

def _plain_quotes(items):
    return '、'.join(('“' + item['text'] + '”' for item in _unique_items(items)))

def _annotated_date_key(item):
    text = unicodedata.normalize('NFKC', item['text']).strip()
    text = re.sub('\\s*\\([^()]*\\)\\s*$', '', text)
    proxy = dict(item, text=text)
    return _date_key(proxy)

def _date_columns(items):
    anchors = [item for item in items if _annotated_date_key(item) is not None]
    if len({_annotated_date_key(item) for item in anchors}) < 4:
        return None
    box = _bounds(items)
    scale = median((_font_size(item) for item in items))
    centers = [((item['bbox'][0] + item['bbox'][2]) / 2, (item['bbox'][1] + item['bbox'][3]) / 2) for item in anchors]
    if max((x for (x, y) in centers)) - min((x for (x, y) in centers)) < 0.55 * (box[2] - box[0]) or max((y for (x, y) in centers)) - min((y for (x, y) in centers)) > 2.5 * scale or max((y for (x, y) in centers)) - box[1] > 16 * scale:
        return None
    anchors.sort(key=lambda item: (item['bbox'][0], _reading_order(item)))
    numeric_left = [item for item in items if _numeric_observation(item) and item['bbox'][2] <= min((a['bbox'][0] for a in anchors))]
    if len(numeric_left) >= 3:
        xs = [(i['bbox'][0] + i['bbox'][2]) / 2 for i in numeric_left]
        if max(xs) - min(xs) > scale:
            numeric_left = []
    else:
        numeric_left = []
    paired = [[] for _ in anchors]
    remaining = []
    for item in items:
        if any((item is a for a in anchors)) or any((item is a for a in numeric_left)):
            continue
        ix = (item['bbox'][0] + item['bbox'][2]) / 2
        iy = (item['bbox'][1] + item['bbox'][3]) / 2
        distances = []
        for (index, anchor) in enumerate(anchors):
            ax = (anchor['bbox'][0] + anchor['bbox'][2]) / 2
            ay = (anchor['bbox'][1] + anchor['bbox'][3]) / 2
            neighbor_gap = min((abs(ax - (other['bbox'][0] + other['bbox'][2]) / 2) for other in anchors if other is not anchor))
            limit = max(2.5 * scale, 0.32 * neighbor_gap)
            if abs(ix - ax) <= limit and abs(iy - ay) <= 12 * scale:
                distances.append((abs(ix - ax), index))
        distances.sort()
        if not distances or (len(distances) > 1 and distances[1][0] - distances[0][0] < 0.5 * scale):
            remaining.append(item)
        else:
            paired[distances[0][1]].append(item)
    if sum((bool(group) for group in paired)) < 3:
        return None
    return {'kind': 'dates', 'items': items, 'anchors': anchors, 'paired': paired, 'left_numbers': numeric_left, 'remaining': remaining}

def _partition(items, depth=0):
    items = sorted(items, key=_reading_order)
    if len(items) <= 3 or depth >= 5:
        return {'kind': 'leaf', 'items': items}
    timeline = _date_columns(items)
    if timeline:
        return timeline
    box = _bounds(items)
    (width, height) = (max(1, box[2] - box[0]), max(1, box[3] - box[1]))
    scale = median((_font_size(item) for item in items))
    candidates = []
    for (axis, span) in ((1, height), (0, width)):
        for (gap, split) in _gaps(items, axis):
            if gap < max((1.2 if axis == 1 else 1.6) * scale, 0.035 * span):
                continue
            first = [i for i in items if i['bbox'][axis + 2] <= split]
            second = [i for i in items if i['bbox'][axis] >= split]
            if not first or not second or len(first) + len(second) != len(items):
                continue
            (a, b) = (_bounds(first), _bounds(second))
            if axis == 1:
                wide_bands = min(a[2] - a[0], b[2] - b[0]) >= 0.55 * width
                strong = gap >= 4 * scale
                if not strong and (not (wide_bands and min(len(first), len(second)) >= 2)):
                    continue
                priority = 2 if strong or (wide_bands and gap >= 2 * scale) else 0
            else:
                overlap = min(a[3], b[3]) - max(a[1], b[1])
                if overlap < 0.15 * min(a[3] - a[1], b[3] - b[1]):
                    continue
                priority = 1
            candidates.append((priority, gap / scale, -axis, split, first, second, gap))
    if not candidates:
        return {'kind': 'leaf', 'items': items}
    (priority, strength, neg_axis, split, first, second, gap) = max(candidates, key=lambda c: c[:4])
    axis = -neg_axis
    children = [_partition(first, depth + 1), _partition(second, depth + 1)]
    if axis == 0:
        expanded = []
        for child in children:
            if child['kind'] == 'split' and child['axis'] == 0 and (child['gap'] >= 0.7 * gap):
                expanded.extend(child['children'])
            else:
                expanded.append(child)
        children = expanded
    return {'kind': 'split', 'items': items, 'axis': axis, 'gap': gap, 'children': children}

def _describe_dates(node):
    parts = []
    remaining = node['remaining']
    paired_items = [item for group in node['paired'] for item in group]
    if remaining:
        bounds = _bounds(node['items'])
        upper = [i for i in remaining if i['bbox'][3] <= min((x['bbox'][1] for x in paired_items)) and i['bbox'][2] - i['bbox'][0] >= 0.25 * (bounds[2] - bounds[0])]
        other = [i for i in remaining if not any((i is u for u in upper))]
        if upper:
            parts.append('上方文字为' + _plain_quotes(upper))
        for (column, label) in enumerate(('左侧', '中间', '右侧')):
            located = [i for i in other if _local_position(i['bbox'], bounds)[0] == column]
            if located:
                parts.append('另有' + label + '文字' + _plain_quotes(located))
    if node['left_numbers']:
        parts.append('左侧数值列从上到下为' + _plain_quotes(node['left_numbers']))
    columns = []
    for (anchor, group) in zip(node['anchors'], node['paired']):
        above = [i for i in group if i['bbox'][3] <= anchor['bbox'][1]]
        below = [i for i in group if i['bbox'][1] >= anchor['bbox'][3]]
        nearby = [i for i in group if not any((i is x for x in above + below))]
        text = _plain_quotes([anchor])
        if not group:
            text += '，同列说明尚未明确配对'
        for (label, members) in (('上方为', above), ('下方为', below), ('附近为', nearby)):
            if members:
                text += '，' + label + _plain_quotes(members)
        columns.append(text)
    parts.append('日期及同列附近文字从左到右为：' + '；'.join(columns))
    return '；'.join(parts)

def _describe_partition(node):
    if node['kind'] == 'leaf':
        return _plain_quotes(node['items'])
    if node['kind'] == 'dates':
        return _describe_dates(node)
    children = node['children']
    if node['axis'] == 0:
        if len(children) == 2:
            return '左侧为' + _describe_partition(children[0]) + '；右侧为' + _describe_partition(children[1])
        return f'{len(children)}组并列文字，从左到右分别为：' + '；'.join(('（' + str(i + 1) + '）' + _describe_partition(child) for (i, child) in enumerate(children)))
    return '上半部分：' + _describe_partition(children[0]) + '。\n下半部分：' + _describe_partition(children[1])

def _hierarchy_eligible(tree):
    """Use a date-and-metric hierarchy only with clear geometric anchors."""
    if tree['kind'] != 'split' or tree['axis'] != 1:
        return False
    (header, body) = tree['children']
    scale = median((_font_size(item) for item in tree['items']))
    if len(header['items']) > 3 or tree['gap'] < 4 * scale or body['kind'] != 'split' or (body['axis'] != 1):
        return False
    (intro, dates) = body['children']
    if dates['kind'] != 'dates' or intro['kind'] != 'split' or intro['axis'] != 0 or (len(intro['children']) != 2):
        return False
    (left, right) = intro['children']
    if len(left['items']) > 8 or any((_numeric_observation(i) or _annotated_date_key(i) is not None for i in left['items'])) or right['kind'] != 'split' or (right['axis'] != 0):
        return False
    return len(right['children']) >= 2 and all((column['kind'] == 'leaf' and any((_numeric_observation(item) for item in column['items'])) and any((not _numeric_observation(item) and _annotated_date_key(item) is None for item in column['items'])) for column in right['children']))

def _hierarchical_summary(items, image_size=None):
    readable = sorted((i for i in items if i.get('text')), key=_reading_order)
    if not readable:
        return '当前画面没有识别到可用文字。'
    if image_size is None:
        result = '识别到的文字内容是' + _plain_quotes(readable) + '。'
    else:
        tree = _partition(readable)
        if _hierarchy_eligible(tree):
            lower = tree['children'][1]
            (intro, dates) = lower['children']
            (left, right) = intro['children']
            result = '文字内容分为上下两个区域。\n上方：' + _describe_partition(tree['children'][0]) + '。\n下方区域分为上下两部分：\n上半部分：左侧为' + _plain_quotes(left['items']) + '；右侧为' + _describe_partition(right) + '。\n下半部分：' + _describe_dates(dates) + '。'
            values = [unicodedata.normalize('NFKC', i['text']).strip() for i in dates['left_numbers']]
            matches = [re.fullmatch('([+-]?\\d+(?:\\.\\d+)?)(万|千|亿)?', value) for value in values]
            if len(matches) >= 3 and all(matches) and (len({m[2] for m in matches}) == 1):
                numbers = [float(m[1]) for m in matches]
                differences = [b - a for (a, b) in zip(numbers, numbers[1:])]
                if any((d > 0 for d in differences)) and any((d < 0 for d in differences)):
                    result += '\n数值备注：左侧数值列的上下顺序不连续，需对照原图核实；原始数值未改写。'
        else:
            result = _positioned_summary(readable, image_size).replace('”（OCR 分数较低，待核实）', '”')
    uncertain = []
    for item in readable:
        if item.get('score', 1.0) < 0.95 and item['text'] not in uncertain:
            uncertain.append(item['text'])
    if uncertain:
        result += '\n识别备注：' + '、'.join(uncertain) + '的 OCR 置信度较低，待核实。'
    return result

def _xy(item):
    b = item['bbox']
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)

def _scale(items):
    return median((_font_size(i) for i in items))

def _groups_by(values, coordinate, tolerance):
    groups = []
    for value in sorted(values, key=coordinate):
        if not groups or coordinate(value) - median((coordinate(v) for v in groups[-1])) > tolerance:
            groups.append([value])
        else:
            groups[-1].append(value)
    return groups

def _blocks(items):
    ordered = sorted(items, key=_reading_order)
    edges = []
    for (j, lower) in enumerate(ordered):
        (lx, ly) = _xy(lower)
        for (i, upper) in enumerate(ordered[:j]):
            (ux, uy) = _xy(upper)
            size = min(_font_size(lower), _font_size(upper))
            if ly - uy < 0.45 * size or lower['bbox'][1] - upper['bbox'][3] > 2.2 * size:
                continue
            if min(lower['bbox'][3], upper['bbox'][3]) - max(lower['bbox'][1], upper['bbox'][1]) > 0.65 * min(lower['bbox'][3] - lower['bbox'][1], upper['bbox'][3] - upper['bbox'][1]):
                continue
            left = abs(lower['bbox'][0] - upper['bbox'][0])
            if left > 1.4 * size and abs(lx - ux) > 0.9 * size:
                continue
            edges.append((max(0, lower['bbox'][1] - upper['bbox'][3]) + 2 * min(left, abs(lx - ux)), i, j))
    (successor, predecessor) = ({}, {})
    for (distance, i, j) in sorted(edges):
        if i not in successor and j not in predecessor:
            (successor[i], predecessor[j]) = (j, i)
    result = []
    for i in range(len(ordered)):
        if i in predecessor:
            continue
        group = [ordered[i]]
        while i in successor:
            i = successor[i]
            group.append(ordered[i])
        result.append(group)
    return sorted(result, key=lambda g: _reading_order({'bbox': _bounds(g), 'text': ''}))

def _heading(items):
    """Peel a broad leading heading, with up to two aligned subtitle lines."""
    if len(items) < 4:
        return ([], items)
    box = _bounds(items)
    scale = _scale(items)
    options = [i for i in items if not _numeric_observation(i) and _annotated_date_key(i) is None and (i['bbox'][1] <= box[1] + 0.23 * (box[3] - box[1])) and (i['bbox'][2] - i['bbox'][0] >= 0.4 * (box[2] - box[0])) and (_font_size(i) >= 1.35 * scale)]
    if not options:
        return ([], items)
    first = min(options, key=_reading_order)
    rivals = [i for i in items if i is not first and i['bbox'][1] > first['bbox'][3] + 2 * scale and (_font_size(i) >= 0.8 * _font_size(first)) and (i['bbox'][2] - i['bbox'][0] >= 0.5 * (first['bbox'][2] - first['bbox'][0]))]
    if len(rivals) >= 2:
        return ([], items)
    preceding = [i for i in items if i is not first and _xy(i)[1] < _xy(first)[1] and (first['bbox'][1] - i['bbox'][3] <= 3 * scale) and (abs(i['bbox'][0] - first['bbox'][0]) <= 2 * scale) and (i['bbox'][2] - i['bbox'][0] >= 0.3 * (first['bbox'][2] - first['bbox'][0])) and (not _numeric_observation(i)) and (_annotated_date_key(i) is None)]
    heading = preceding + [first]
    for item in sorted(items, key=_reading_order):
        if item is first or len(heading) >= 3:
            continue
        last = heading[-1]
        if _xy(item)[1] > _xy(last)[1] and item['bbox'][1] <= last['bbox'][3] + 2.5 * scale and (abs(item['bbox'][0] - first['bbox'][0]) <= 1.5 * scale) and (item['bbox'][2] - item['bbox'][0] >= 0.5 * (first['bbox'][2] - first['bbox'][0])) and (not _numeric_observation(item)) and (_annotated_date_key(item) is None):
            heading.append(item)
    return (sorted(heading, key=_reading_order), [i for i in items if not any((i is h for h in heading))])

def _split(items):
    if len(items) < 4:
        return None
    scale = _scale(items)
    bounds = _bounds(items)
    options = []
    protected = []
    for method in (_numbered_bindings, _comparison_bindings):
        bound = method(items)
        if bound:
            protected.append(_bounds([item for row in bound['rows'] for item in row]))
    for axis in (0, 1):
        span = bounds[axis + 2] - bounds[axis]
        for (gap, at) in _gaps(items, axis):
            if any((box[axis] < at < box[axis + 2] for box in protected)):
                continue
            if gap < max((1.5 if axis == 0 else 1.3) * scale, 0.035 * span):
                continue
            first = [i for i in items if i['bbox'][axis + 2] <= at]
            second = [i for i in items if i['bbox'][axis] >= at]
            if not first or not second:
                continue
            (a, b) = (_bounds(first), _bounds(second))
            if axis == 0:
                overlap = min(a[3], b[3]) - max(a[1], b[1])
                if overlap < 0.3 * min(a[3] - a[1], b[3] - b[1]):
                    continue
                priority = 1
            else:
                wide = min(a[2] - a[0], b[2] - b[0]) >= 0.55 * (bounds[2] - bounds[0])
                broad_band = max(a[2] - a[0], b[2] - b[0]) >= 0.8 * (bounds[2] - bounds[0]) and min(len(first), len(second)) >= 3 and (gap >= 3 * scale)
                if not wide and (not broad_band) and (gap < 4 * scale):
                    continue
                priority = 2 if broad_band else 1.5 if wide else 1
            options.append((priority * gap / scale, gap / scale, -axis, at, first, second))
    if not options:
        return None
    (_, strength, neg_axis, at, first, second) = max(options, key=lambda x: x[:4])
    axis = -neg_axis
    if axis == 0:
        cuts = sorted((option[3] for option in options if option[2] == 0 and option[1] >= 0.5 * strength))
        groups = []
        remaining = list(items)
        for cut in cuts:
            group = [i for i in remaining if i['bbox'][2] <= cut]
            if group:
                groups.append(group)
                remaining = [i for i in remaining if not any((i is x for x in group))]
        if remaining:
            groups.append(remaining)
        if len(groups) > 2 and any((len(group) < 2 for group in groups)):
            groups = [first, second]
        return (axis, groups)
    return (axis, [first, second])

def _region_records(items, path=(), depth=0):
    row = _row_order(items)
    if row:
        return [{'path': path, 'kind': 'row', 'rows': [[[i] for i in row]]}]
    parallel = _parallel_blocks(items)
    if parallel:
        return [{'path': path, 'kind': 'parallel', 'rows': [[g for g in parallel]]}]
    (heading, body) = _heading(items)
    result = []
    if heading:
        result.append({'path': path, 'kind': 'heading', 'rows': [[heading]]})
    if not body:
        return result
    for method in (_numbered_rows, _comparison_rows):
        bound = method(body)
        if bound:
            result.append({'path': path, 'kind': bound['kind'], 'rows': [[row] for row in bound['rows']]})
            return result
    blocks = _blocks(body)
    boxes = [_bounds(g) for g in blocks]
    shared_x = max((b[0] for b in boxes)) < min((b[2] for b in boxes))
    single_column = len(blocks) >= 2 and (shared_x or max((b[0] for b in boxes)) - min((b[0] for b in boxes)) <= 2 * _scale(body))
    split = _split(body) if depth < 3 and (not single_column) else None
    if split:
        (axis, groups) = split
        for (index, group) in enumerate(groups):
            result += _region_records(group, path + ((axis, index, len(groups)),), depth + 1)
        return result
    if len(blocks) == 1:
        result.append({'path': path, 'kind': 'body', 'rows': [[blocks[0]]]})
    else:
        rows = []
        for block in blocks:
            box = _bounds(block)
            if rows and all(((box[2] <= _bounds(other)[0] or _bounds(other)[2] <= box[0]) and min(box[3], _bounds(other)[3]) - max(box[1], _bounds(other)[1]) >= 0.25 * min(box[3] - box[1], _bounds(other)[3] - _bounds(other)[1]) for other in rows[-1])):
                rows[-1].append(block)
            else:
                rows.append([block])
        for row in rows:
            row.sort(key=lambda g: _bounds(g)[0])
        result.append({'path': path, 'kind': 'body', 'rows': rows})
    return result

def _short_fragment(item):
    text = item['text'].strip()
    return item.get('score', 1.0) < 0.95 and bool(re.fullmatch('[A-Za-z]{1,2}', text)) and (not text.isupper())

def _visible_quotes(items):
    return _plain_quotes([i for i in items if not _short_fragment(i)])

def _step_label(step):
    (axis, index, count) = step
    if axis == 1:
        return '上方' if index == 0 else '下方'
    if count == 2:
        return '左侧' if index == 0 else '右侧'
    if count == 3:
        return ('左列', '中列', '右列')[index]
    return '第' + str(index + 1) + '列'

def _short_path(path, local=False):
    if local and len(path) == 1:
        (axis, index, count) = path[0]
        if axis == 1:
            return '上层' if index == 0 else '下层'
        if count == 2:
            return '左列' if index == 0 else '右列'
    if len(path) == 2 and {path[0][0], path[1][0]} == {0, 1} and (path[0][2] == path[1][2] == 2):
        horizontal = next((s[1] for s in path if s[0] == 0))
        vertical = next((s[1] for s in path if s[0] == 1))
        return ('左' if horizontal == 0 else '右') + ('上' if vertical == 0 else '下')
    return '·'.join((_step_label(step) for step in path))

def _record_units(records):
    units = []
    for record in records:
        if record['kind'] == 'paired':
            units.extend((_paired_row_text(row[0]) for row in record['rows']))
            continue
        if record['kind'] == 'numbered':
            units.extend((_numbered_row_text(row[0]) for row in record['rows']))
            continue
        for row in record['rows']:
            blocks = [_visible_quotes(group) for group in row]
            blocks = [text for text in blocks if text]
            if not blocks:
                continue
            text = blocks[0] if len(blocks) == 1 else '从左到右为' + '；'.join(blocks)
            units.append(text)
    return units

def _inline_units(units):
    if len(units) == 1:
        return units[0]
    return '；'.join(('（' + str(i + 1) + '）' + text for (i, text) in enumerate(units)))

def _render_region_records(records):
    root = {'records': [], 'children': {}, 'path': ()}
    for record in records:
        node = root
        for step in record['path']:
            node = node['children'].setdefault(step, {'records': [], 'children': {}, 'path': node['path'] + (step,)})
        node['records'].append(record)
    children = list(root['children'].items())
    if not children:
        units = _record_units(root['records'])
        if not units:
            return []
        if len(units) == 1:
            return ['内容：' + units[0] + '。']
        return ['内容从上到下：' + _inline_units(units) + '。']
    first = children[0][0]
    (axis, count) = (first[0], first[2])
    if axis == 0:
        overview = '主体分为左右两部分' if count == 2 else '主体有' + str(count) + '列，从左到右阅读'
    else:
        overview = '主体分为上下两部分'
    major = children
    branch_nodes = [(step, node) for (step, node) in children if node['children']]
    if len(children) == 2 and len(branch_nodes) == 1:
        (step, branch) = branch_nodes[0]
        if len(children) - 1 + len(branch['children']) <= 4:
            other_axis = next(iter(branch['children']))[0]
            count_inside = len(branch['children'])
            division = ('左、右两块' if other_axis == 0 else '上、下两块') if count_inside == 2 else str(count_inside) + ('列' if other_axis == 0 else '部分')
            overview += '，' + _step_label(step) + '再分为' + division
            major = []
            for (child_step, node) in children:
                major.extend(node['children'].items() if node is branch else [(child_step, node)])
    lines = [overview + '。']
    initial = _record_units(root['records'])
    if initial:
        lines.append('上方文字：' + _inline_units(initial) + '。')

    def render_node(node, label):
        result = []
        own = _record_units(node['records'])
        sub = list(node['children'].items())
        if not sub:
            if not own:
                return result
            if len(own) >= 3:
                result.append('【' + label + '：' + str(len(own)) + '组内容，从上到下】')
                result.extend(('（' + str(i + 1) + '）' + text + '。' for (i, text) in enumerate(own)))
            else:
                result.append('【' + label + '】' + _inline_units(own) + '。')
            return result
        direction = '从左到右' if sub[0][0][0] == 0 else '从上到下'
        unit = '列' if sub[0][0][0] == 0 else '层'
        result.append('【' + label + '：' + str(len(sub)) + unit + '，' + direction + '】')
        if own:
            result.append(_inline_units(own) + '。')

        def leaves(current, local=()):
            entries = []
            current_units = _record_units(current['records'])
            if current_units:
                entries.append((local, current_units))
            for (child_step, child) in current['children'].items():
                entries += leaves(child, local + (child_step,))
            return entries
        for (step, child) in sub:
            for (local, units) in leaves(child, (step,)):
                result.append('  ' + _short_path(local, True) + '：' + _inline_units(units) + '。')
        return result
    for (step, node) in major:
        label = _short_path(node['path'])
        lines += render_node(node, label)
    return lines

def _region_lines(items, path=(), depth=0):
    return _render_region_records(_region_records(items, path, depth))

def _parallel_blocks(items):
    blocks = _blocks(items)
    if not 2 <= len(blocks) <= 6:
        return None
    blocks.sort(key=lambda g: _bounds(g)[0])
    boxes = [_bounds(g) for g in blocks]
    if any((a[2] >= b[0] for (a, b) in zip(boxes, boxes[1:]))):
        return None
    overlap = min((b[3] for b in boxes)) - max((b[1] for b in boxes))
    if overlap < 0.25 * min((b[3] - b[1] for b in boxes)):
        return None
    return blocks

def _reading_column(items):
    """Keep a single-column introduction/list together across large indents."""
    box = _bounds(items)
    width = box[2] - box[0]
    if max((i['bbox'][0] for i in items)) - min((i['bbox'][0] for i in items)) > 0.28 * width:
        return None
    cuts = sorted((at for (gap, at) in _gaps(items, 1) if gap >= 2 * _scale(items)))
    groups = []
    remaining = list(items)
    for cut in cuts:
        group = [i for i in remaining if i['bbox'][3] <= cut]
        if group:
            groups.append(group)
            remaining = [i for i in remaining if not any((i is g for g in group))]
    if remaining:
        groups.append(remaining)
    if len(groups) == 1:
        text = '文字按从上到下排列：' + _plain_quotes(items) + '。'
    else:
        text = '单栏文字从上到下分为' + str(len(groups)) + '段：\n' + '\n'.join(('（' + str(j + 1) + '）' + _plain_quotes(g) + '。' for (j, g) in enumerate(groups)))
    return {'mode': 'reading-column', 'text': text, 'paragraphs': len(groups)}

def _timeline(items):
    anchors = [i for i in items if _annotated_date_key(i) is not None]
    if len({_annotated_date_key(i) for i in anchors}) < 4:
        return None
    bounds = _bounds(items)
    if max((_xy(i)[0] for i in anchors)) - min((_xy(i)[0] for i in anchors)) < 0.4 * (bounds[2] - bounds[0]):
        return None
    paired = [[] for a in anchors]
    remaining = []
    for item in items:
        if any((item is a for a in anchors)):
            continue
        choices = []
        (x, y) = _xy(item)
        for (j, anchor) in enumerate(anchors):
            (ax, ay) = _xy(anchor)
            scale = min(_font_size(anchor), _font_size(item))
            offset = abs(item['bbox'][0] - anchor['bbox'][0])
            if offset > max(1.5 * scale, 0.5 * (anchor['bbox'][2] - anchor['bbox'][0])):
                continue
            if abs(y - ay) > 14 * max(scale, _font_size(anchor)):
                continue
            between = [a for a in anchors if a is not anchor and min(ay, y) < _xy(a)[1] < max(ay, y) and (abs(a['bbox'][0] - anchor['bbox'][0]) <= max(1.5 * scale, 0.5 * (anchor['bbox'][2] - anchor['bbox'][0])))]
            if between:
                continue
            choices.append((offset + 0.12 * abs(y - ay), j))
        choices.sort()
        if not choices or (len(choices) > 1 and choices[1][0] - choices[0][0] < 0.4 * _font_size(item)):
            remaining.append(item)
        else:
            paired[choices[0][1]].append(item)
    clear = [(i, anchors[j]) for (j, g) in enumerate(paired) for i in g]
    below_count = sum((_xy(i)[1] >= _xy(a)[1] for (i, a) in clear))
    if len(clear) >= 4 and below_count >= 0.8 * len(clear):
        still_unpaired = []
        for item in remaining:
            candidates = []
            for (j, anchor) in enumerate(anchors):
                size = min(_font_size(item), _font_size(anchor))
                offset = abs(item['bbox'][0] - anchor['bbox'][0])
                dy = _xy(item)[1] - _xy(anchor)[1]
                if 0 <= dy <= 14 * max(size, _font_size(anchor)) and offset <= max(1.5 * size, 0.5 * (anchor['bbox'][2] - anchor['bbox'][0])):
                    between = [a for a in anchors if a is not anchor and _xy(anchor)[1] < _xy(a)[1] < _xy(item)[1] and (abs(a['bbox'][0] - anchor['bbox'][0]) <= max(1.5 * size, 0.5 * (anchor['bbox'][2] - anchor['bbox'][0])))]
                    if not between:
                        candidates.append((offset + 0.12 * dy, j))
            candidates.sort()
            if candidates and (len(candidates) == 1 or candidates[1][0] - candidates[0][0] >= 0.4 * _font_size(item)):
                paired[candidates[0][1]].append(item)
            else:
                still_unpaired.append(item)
        remaining = still_unpaired
    if sum((bool(g) for g in paired)) < 3:
        return None
    lines = []
    if remaining:
        lines += ['日期记录以外的文字：'] + _region_lines(remaining)
    lines.append('日期记录按年月顺序整理；附近文字仅按同列位置归组：')
    for index in sorted(range(len(anchors)), key=lambda j: (_annotated_date_key(anchors[j]), _reading_order(anchors[j]))):
        (anchor, group) = (anchors[index], paired[index])
        text = _plain_quotes([anchor])
        for (relation, members) in [('上方', [i for i in group if _xy(i)[1] < _xy(anchor)[1]]), ('下方', [i for i in group if _xy(i)[1] >= _xy(anchor)[1]])]:
            if members:
                text += '；同列' + relation + '：' + _plain_quotes(members)
        if not group:
            text += '；未找到可明确配对的说明'
        lines.append(text + '。')
    return {'mode': 'date-records', 'text': '\n'.join(lines), 'date_pairs': [(a['text'], [i['text'] for i in group]) for (a, group) in zip(anchors, paired)]}

def _radial(items):
    if len(items) < 10:
        return None
    bounds = _bounds(items)
    (width, height) = (bounds[2] - bounds[0], bounds[3] - bounds[1])
    footers = [i for i in items if _xy(i)[1] > bounds[1] + 0.85 * height and i['bbox'][2] - i['bbox'][0] >= 0.3 * width]
    body = [i for i in items if not any((i is f for f in footers))]
    box = _bounds(body)
    (cx, cy) = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    options = [i for i in body if abs(_xy(i)[0] - cx) < 0.14 * width and abs(_xy(i)[1] - cy) < 0.14 * height and (not _numeric_observation(i)) and (_font_size(i) >= 1.1 * _scale(body))]
    if not options:
        return None
    center = min(options, key=lambda i: (((_xy(i)[0] - cx) / width) ** 2 + ((_xy(i)[1] - cy) / height) ** 2, _reading_order(i)))
    (cx, cy) = _xy(center)
    left = [i for i in body if _xy(i)[0] < cx - 0.32 * width]
    right = [i for i in body if _xy(i)[0] > cx + 0.32 * width]
    around = [i for i in body if i is not center and (not any((i is x for x in left + right)))]
    if min(len(left), len(right)) < 2 or len(around) < 4:
        return None
    radii = [math.hypot(_xy(i)[0] - cx, _xy(i)[1] - cy) for i in around]
    if min(radii) < 2 * _scale(body) or max(radii) > 1.75 * min(radii):
        return None
    directions = ['上方', '右上', '右侧', '右下', '下方', '左下', '左侧', '左上']
    located = {}
    for item in around:
        (x, y) = _xy(item)
        direction = int(math.atan2(x - cx, cy - y) % (2 * math.pi) / (math.pi / 4) + 0.5) % 8
        located.setdefault(direction, []).append(item)
    if len(located) < 5:
        return None
    lines = ['中央文字：' + _plain_quotes([center]) + '。', '中心周围按位置分布：' + '；'.join((directions[d] + '为' + _plain_quotes(located[d]) for d in sorted(located))) + '。', '左侧外围从上到下：' + _plain_quotes(left) + '。', '右侧外围从上到下：' + _plain_quotes(right) + '。']
    if footers:
        lines.append('底部文字：' + _plain_quotes(footers) + '。')
    return {'mode': 'center-surround', 'text': '\n'.join(lines), 'center': center['text'], 'surround_count': len(around)}

def _table(items):
    numeric = [i for i in items if re.fullmatch('\\d{1,2}(?:\\.\\d{1,3})?', unicodedata.normalize('NFKC', i['text']).strip())]
    if len(numeric) < 12:
        return None
    scale = _scale(numeric)
    cols = _groups_by(numeric, lambda i: _xy(i)[0], 2 * scale)
    if len(cols) < 3 or any((len(c) < 3 for c in cols)):
        return None
    column = {id(i): j for (j, group) in enumerate(cols) for i in group}
    slopes = []
    for (j, a) in enumerate(numeric):
        for b in numeric[j + 1:]:
            if column[id(a)] == column[id(b)]:
                continue
            (ax, ay) = _xy(a)
            (bx, by) = _xy(b)
            slope = (by - ay) / (bx - ax)
            if abs(slope) <= 0.25:
                slopes.append(slope)
    if not slopes:
        return None
    bins = {}
    for slope in slopes:
        bins.setdefault(round(slope / 0.01), []).append(slope)
    dominant = max(bins, key=lambda b: (len(bins[b]), -abs(b)))
    skew = median(bins[dominant])
    ycoord = lambda i: _xy(i)[1] - skew * _xy(i)[0]
    rows = _groups_by(numeric, ycoord, 0.9 * scale)
    if len(rows) < 3 or any((len({column[id(i)] for i in row}) < 3 for row in rows)):
        return None
    if any((len({column[id(i)] for i in row}) != len(row) for row in rows)):
        return None
    centers = [median((_xy(i)[0] for i in c)) for c in cols]
    row_ys = [median((ycoord(i) for i in row)) for row in rows]
    labels = [[] for row in rows]
    headings = [[] for col in cols]
    (row_heading, remaining) = ([], [])
    used = {id(i) for i in numeric}
    for item in items:
        if id(item) in used:
            continue
        (x, y) = _xy(item)
        corrected = ycoord(item)
        if x < centers[0] - 2 * scale:
            nearest = min(range(len(rows)), key=lambda k: abs(corrected - row_ys[k]))
            if abs(corrected - row_ys[nearest]) <= 1.6 * scale:
                labels[nearest].append(item)
            else:
                remaining.append(item)
        else:
            nearest = min(range(len(cols)), key=lambda k: abs(x - centers[k]))
            spacing = min((abs(centers[nearest] - v) for (j, v) in enumerate(centers) if j != nearest))
            if corrected < row_ys[0] - 0.7 * scale and abs(x - centers[nearest]) <= 0.55 * spacing and (item['bbox'][2] - item['bbox'][0] <= 1.4 * spacing):
                headings[nearest].append(item)
            else:
                remaining.append(item)
    if any((not label for label in labels)) or any((not h for h in headings)):
        return None
    label_x = median((_xy(i)[0] for group in labels for i in group))
    recovered = [i for i in remaining if abs(_xy(i)[0] - label_x) < 2 * scale and ycoord(i) < row_ys[0] and (ycoord(i) > row_ys[0] - 8 * scale)]
    row_heading += recovered
    remaining = [i for i in remaining if not any((i is v for v in recovered))]
    lines = _region_lines(remaining) if remaining else []
    lines.append(f'表格式文字共{len(cols)}列数据，下面按行保留对应关系。')
    if row_heading:
        lines.append('行名称栏：' + _plain_quotes(row_heading) + '。')
    lines.append('数据列从左到右：' + '；'.join(('第' + str(j + 1) + '列 ' + _plain_quotes(group) for (j, group) in enumerate(headings))) + '。')
    for (label, row) in zip(labels, rows):
        cells = {column[id(i)]: i for i in row}
        values = ['第' + str(j + 1) + '列 ' + _plain_quotes([cells[j]]) if j in cells else '第' + str(j + 1) + '列未识别到值' for j in range(len(cols))]
        lines.append(_plain_quotes(label) + '：' + '；'.join(values) + '。')
    return {'mode': 'table-rows', 'text': '\n'.join(lines), 'rows': len(rows), 'columns': len(cols), 'skew': skew}

def organize(items, image_size=None):
    readable = sorted((i for i in items if i.get('text')), key=_reading_order)
    if not readable:
        return {'mode': 'empty', 'text': '当前画面没有识别到可用文字。'}
    if image_size is None:
        return {'mode': 'unpositioned', 'text': '识别到的文字内容是' + _plain_quotes(readable) + '。'}
    if len(readable) <= 3:
        return {'mode': 'simple', 'text': _positioned_summary(readable, image_size).replace('”（OCR 分数较低，待核实）', '”')}
    for method in (_numbered_rows, _comparison_rows):
        bound = method(readable)
        if bound:
            return bound
    (heading, body) = _heading(readable)
    if heading:
        for method in (_numbered_rows, _comparison_rows):
            bound = method(body)
            if bound:
                bound['text'] = '上方文字：' + _plain_quotes(heading) + '。\n' + bound['text']
                return bound
    tree = _partition(readable)
    if _hierarchy_eligible(tree):
        return {'mode': 'hierarchy', 'text': _hierarchical_summary(readable, image_size).split('\n识别备注：', 1)[0]}
    for method in (_table, _timeline, _radial):
        result = method(readable)
        if result:
            return result
    row = _row_order(readable)
    if row:
        return {'mode': 'parallel-blocks', 'text': '同一横排从左到右：' + '、'.join((_plain_quotes([i]) for i in row)) + '。'}
    parallel = _parallel_blocks(readable)
    if parallel:
        return {'mode': 'parallel-blocks', 'text': '从左到右有' + str(len(parallel)) + '组并列文字：' + '；'.join(('（' + str(j + 1) + '）' + _plain_quotes(g) for (j, g) in enumerate(parallel))) + '。'}
    if len(readable) <= 8:
        return {'mode': 'simple', 'text': _positioned_summary(readable, image_size).replace('”（OCR 分数较低，待核实）', '”')}
    column = _reading_column(readable)
    if column:
        return column
    scale = _scale(readable)
    headers = []
    for (gap, cut) in sorted(_gaps(readable, 1), reverse=True):
        upper = [i for i in readable if i['bbox'][3] <= cut]
        lower = [i for i in readable if i['bbox'][1] >= cut]
        if gap >= 4 * scale and 1 <= len(upper) <= 3 and (len(lower) >= 3):
            (headers, readable) = (upper, lower)
            break
    if headers:
        column = _reading_column(readable)
        if column:
            column['text'] = '上方独立文字：' + _plain_quotes(headers) + '。\n下方内容：\n' + column['text']
            return column
        return {'mode': 'regions', 'text': '上方独立文字：' + _plain_quotes(headers) + '。\n下方内容：\n' + '\n'.join(_region_lines(readable))}
    return {'mode': 'regions', 'text': '\n'.join(_region_lines(readable))}


def build_ocr_summary(items, image_size=None):
    """Describe existing text boxes without changing or completing observations.

    Positions use the adapter's coordinate frame. Unknown frame dimensions
    retain the text without guessing a direction; ambiguous row associations
    retain the general geometric description.
    """
    result = organize(items, image_size)['text']
    uncertain = []
    moved = []
    for item in sorted(items, key=_reading_order):
        if item.get('text') and _short_fragment(item) and ('“' + item['text'] + '”' not in result) and (item['text'] not in moved):
            moved.append(item['text'])
        if item.get('text') and item.get('score', 1.0) < 0.95 and (item['text'] not in uncertain):
            uncertain.append(item['text'])
    if moved:
        result += '\n识别残片：' + '、'.join(('“' + text + '”' for text in moved)) + '置信度较低，单独保留供核实。'
        uncertain = [text for text in uncertain if text not in moved]
    if uncertain:
        result += '\n识别备注：' + '、'.join(uncertain) + '的 OCR 置信度较低，待核实。'
    return result

def _paired_row_text(row):
    (label, value) = row
    return _plain_quotes([label]) + '；同排右侧数值为' + _plain_quotes([value])

def _numbered_row_text(row):
    return '原编号' + _plain_quotes(row[:1]) + '，右侧邻近文字为' + _plain_quotes(row[1:])

def _comparison_bindings(items):
    """Find one uniquely aligned comparison block, leaving other items alone."""
    if not 4 <= len(items) <= 96:
        return None
    scale = _scale(items)
    usable = [i for i in items if _annotated_date_key(i) is None and (not _CLOCK_TEXT.fullmatch(i['text'].strip()))]
    values = [i for i in usable if _numeric_observation(i)]
    labels = [i for i in usable if not _numeric_observation(i) and _font_size(i) < 1.5 * scale]
    if len(values) < 2 or len(labels) < 2:
        return None
    accepted = []
    for column in _groups_by(labels, lambda i: i['bbox'][0], 2 * scale):
        column.sort(key=lambda i: (_xy(i)[1], _reading_order(i)))
        if not 2 <= len(column) <= 6:
            continue
        if any((_xy(b)[1] - _xy(a)[1] < 2.5 * scale for (a, b) in zip(column, column[1:]))):
            continue
        slopes = {0.0}
        for label in column:
            (lx, ly) = _xy(label)
            for value in values:
                (vx, vy) = _xy(value)
                if value['bbox'][0] >= label['bbox'][2] + 0.5 * scale:
                    slope = (vy - ly) / (vx - lx)
                    if abs(slope) <= 0.15:
                        slopes.add(slope)
        matches = {}
        for skew in sorted(slopes):
            corrected = lambda i: _xy(i)[1] - skew * _xy(i)[0]
            (rows, used, score) = ([], set(), 0.0)
            for label in column:
                options = [(abs(corrected(value) - corrected(label)), _reading_order(value), value) for value in values if value['bbox'][0] >= label['bbox'][2] + 0.5 * scale]
                options.sort(key=lambda option: option[:2])
                if not options or options[0][0] > 0.75 * scale:
                    break
                if len(options) > 1 and options[1][0] - options[0][0] < 0.5 * scale:
                    break
                (distance, _order, value) = options[0]
                if id(value) in used:
                    break
                used.add(id(value))
                rows.append([label, value])
                score += distance
            if len(rows) != len(column):
                continue
            key = tuple((_reading_order(row[1]) for row in rows))
            if key not in matches or score < matches[key][0]:
                matches[key] = (score, rows)
        ranked = sorted(matches.values(), key=lambda value: (value[0], tuple((_reading_order(row[1]) for row in value[1]))))
        if not ranked or (len(ranked) > 1 and ranked[1][0] - ranked[0][0] < 0.5 * scale):
            continue
        accepted.append(ranked[0][1])
    if len(accepted) != 1:
        return None
    rows = accepted[0]
    used = {id(i) for row in rows for i in row}
    return {'rows': rows, 'remaining': [i for i in items if id(i) not in used]}

def _comparison_rows(items):
    """Render a complete comparison section without inventing a chart type."""
    bound = _comparison_bindings(items)
    if not bound or bound['remaining']:
        return None
    rows = bound['rows']
    lines = ['文字分为' + str(len(rows)) + '行，按从上到下保留标签与同排数值：']
    lines += ['第' + str(index + 1) + '行：' + _paired_row_text(row) + '。' for (index, row) in enumerate(rows)]
    return {'mode': 'paired-rows', 'kind': 'paired', 'rows': rows, 'text': '\n'.join(lines)}

def _numbered_bindings(items):
    """Find a local index column and its nearest text column."""
    anchors = [i for i in items if re.fullmatch('\\d{1,2}', unicodedata.normalize('NFKC', i['text']).strip())]
    if len(anchors) < 3:
        return None
    scale = _scale(items)
    candidates = []
    for column in _groups_by(anchors, lambda i: _xy(i)[0], 1.5 * scale):
        if not 3 <= len(column) <= 12:
            continue
        column.sort(key=lambda i: (_xy(i)[1], _reading_order(i)))
        values = [int(unicodedata.normalize('NFKC', i['text']).strip()) for i in column]
        differences = [b - a for (a, b) in zip(values, values[1:])]
        if not (all((0 < d <= 2 for d in differences)) or all((-2 <= d < 0 for d in differences))):
            continue
        if sum((abs(d) == 1 for d in differences)) < len(differences) / 2:
            continue
        if any((_xy(b)[1] - _xy(a)[1] < 2 * scale for (a, b) in zip(column, column[1:]))):
            continue
        candidates.append(column)
    if len(candidates) != 1:
        return None
    anchors = candidates[0]
    centers = [_xy(i)[1] for i in anchors]
    edges = [centers[0] - (centers[1] - centers[0]) / 2]
    edges += [(a + b) / 2 for (a, b) in zip(centers, centers[1:])]
    edges.append(centers[-1] + (centers[-1] - centers[-2]) / 2)
    groups = [[] for _ in anchors]
    for item in items:
        if any((item is anchor for anchor in anchors)):
            continue
        y = _xy(item)[1]
        possible = [index for (index, anchor) in enumerate(anchors) if edges[index] < y < edges[index + 1] and item['bbox'][0] >= anchor['bbox'][2] - 0.25 * scale]
        if len(possible) == 1:
            groups[possible[0]].append(item)
    rows = []
    for (anchor, group) in zip(anchors, groups):
        if not group:
            return None
        text = [i for i in group if not _numeric_observation(i) and _annotated_date_key(i) is None]
        if not text:
            return None
        left = min((i['bbox'][0] for i in text))
        if left - anchor['bbox'][2] > 6 * scale:
            return None
        neighbors = [i for i in group if abs(i['bbox'][0] - left) <= 1.5 * scale]
        rows.append([anchor] + sorted(neighbors, key=_reading_order))
    used = {id(i) for row in rows for i in row}
    return {'rows': rows, 'remaining': [i for i in items if id(i) not in used]}

def _numbered_rows(items):
    """Render a complete numbered section in its detected spatial order."""
    bound = _numbered_bindings(items)
    if not bound or bound['remaining']:
        return None
    rows = bound['rows']
    lines = ['编号文字按画面从上到下排列，保留原编号：']
    lines += [_numbered_row_text(row) + '。' for row in rows]
    return {'mode': 'numbered-rows', 'kind': 'numbered', 'rows': rows, 'text': '\n'.join(lines)}
