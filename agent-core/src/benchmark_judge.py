"""benchmark_judge.py — 用 LLM 判那几样算不出来的。

## 它判的东西很少，这是有意的

绝大多数判定在 `benchmark_case.check_targets` 里就做完了，而且**完全不经 LLM**：一个数
超没超一条线，是算出来的。留给这里的只有三样：

1. **回答效果** —— 讲得对不对、全不全。真的没有可算指标。
2. **参考流程的偏离是否合理** —— 偏离不等于失败，好的 agent 可能找到更优的顺序。
3. **用例自己写的人话要求** —— 而且判的时候手里拿着全部指标。

第三样是关键：裁判不是在估「用户体验好不好」，而是在核「懵逼时长 23.4 秒超没超 15 秒」。
前者它做不了，后者它做得可靠。**模糊判定要少**，少到每一处都有数撑着。

## 裁判不给工具

`tool_list` 是空的，而且必须一直是空的。裁判读材料、回判决，不动手 —— 一个能调工具的
裁判可以在评判一次运行的过程中改变这次运行。

## 解析不出来就记 error，不记 0 分

模型回了一坨不是 JSON 的东西，最省事的处理是当成「没有通过任何一条」记 0 分。那是基准
测试在撒谎：0 分的意思是「测了，没过」，而实际发生的是「没测成」。所以解析失败重试一次，
再失败就把这次运行记成 `error` 并留下原始回复。
"""

from __future__ import annotations

import json

import benchmark_case

# 材料的总字符上限。超了要截断，而且**截断这件事必须写进喂给裁判的材料里** ——
# 不写的话它会以为自己看到了全部，然后为一段它根本没见过的过程下判决。
MAX_CHARS = 60000

VERDICTS = ('pass', 'fail', 'unmeasurable')

SYSTEM = """你是具身智能体的评测裁判。你读一次运行的完整记录，对给定的每一条要求给出判决。

规则：
1. 只依据材料下判决。材料里没有的，判 "unmeasurable" 并说明缺什么 —— 不要推测，也不要
   因为「看起来应该没问题」就判 pass。
2. "pass" / "fail" / "unmeasurable" 三选一。判不了和判了没过是两件事。
3. 每条判决都要给 reason（一句话，说明依据），并尽量引用材料里的具体时刻或数字。
4. 指标已经替你算好了，直接用，不要自己从事件流里重新估算时间。
5. 涉及参考流程的那条要求，额外给 steps：逐步比对，每步 match=true/false，偏离的写明
   差异以及这个偏离是否合理（deviation_ok）。偏离不一定是错 —— 更优的顺序也是偏离。

只输出 JSON，不要任何其它文字：
{"verdicts": [{"id": "<要求 id>", "verdict": "pass|fail|unmeasurable",
               "reason": "...",
               "steps": [{"step": "1. 打开地图", "match": true,
                          "note": "", "deviation_ok": true}]}]}"""


def _clip(text: str, budget: int) -> tuple[str, int]:
    """截到预算以内，返回截掉了多少字符。"""
    if len(text) <= budget:
        return text, 0
    return text[:budget], len(text) - budget


def build_material(case: dict, observations: dict, facts: dict,
                   agent_track: list, budget: int = MAX_CHARS) -> str:
    """裁判看的那份材料 —— 和人复盘看的是同一份。

    顺序是有讲究的：**指标放最前面**。它们是算出来的、最可靠，而且后面那两条轨道很长，
    放在后面被截断的风险最大 —— 真被截了，至少那几个数还在。
    """
    metrics = json.dumps(observations, ensure_ascii=False, indent=1, default=str)
    events = json.dumps(facts.get('events') or [], ensure_ascii=False, default=str)
    posts = json.dumps(facts.get('acp_posts') or [], ensure_ascii=False, default=str)
    track = json.dumps(agent_track or [], ensure_ascii=False, default=str)

    head = f'## 指标（已算好，直接用）\n{metrics}\n'
    spare = max(0, budget - len(head) - 400)
    per = spare // 3
    events, cut_e = _clip(events, per)
    posts, cut_p = _clip(posts, per)
    track, cut_t = _clip(track, spare - 2 * per)

    parts = [head,
             f'\n## 世界事件流{_cut_note(cut_e)}\n{events}\n',
             f'\n## ACP 上报{_cut_note(cut_p)}\n{posts}\n',
             f'\n## agent 这一侧（被什么唤醒、想了什么、调了什么、播报了什么）'
             f'{_cut_note(cut_t)}\n{track}\n']
    return ''.join(parts)


def _cut_note(cut: int) -> str:
    # 截断说明必须在材料**里面**，不是在日志里。裁判看不见日志。
    return '' if not cut else f'（已截断 {cut} 个字符，下面不是全部）'


def build_request(case: dict, observations: dict, facts: dict,
                  agent_track: list) -> list[dict]:
    items = benchmark_case.requirements(case)
    asked = []
    for item in items:
        entry = {'id': item['id'], 'text': item['text'],
                 'dimension': item['dimension']}
        if item.get('procedure'):
            entry['procedure'] = item['procedure']
        asked.append(entry)

    material = build_material(case, observations, facts, agent_track)
    return [
        {'role': 'system', 'content': SYSTEM},
        {'role': 'user', 'content':
            f'## 这次运行要做的事\n{(case.get("test") or {}).get("run", {}).get("prompt", "")}\n\n'
            f'{material}\n'
            f'## 要判的要求\n{json.dumps(asked, ensure_ascii=False, indent=1)}'},
    ]


def parse(text: str) -> dict:
    """把模型的回复解析成判决表。解析不出来抛 ValueError。

    宽容一点：模型爱在 JSON 外面包一层 ```json。但**只宽容包装，不宽容内容** ——
    猜一个缺失的 verdict 等于替它判了一次。
    """
    raw = (text or '').strip()
    if raw.startswith('```'):
        raw = raw.split('\n', 1)[-1]
        raw = raw.rsplit('```', 1)[0]
    start, end = raw.find('{'), raw.rfind('}')
    if start < 0 or end <= start:
        raise ValueError('回复里没有 JSON 对象')
    data = json.loads(raw[start:end + 1])

    out: dict[str, dict] = {}
    for entry in data.get('verdicts') or []:
        key = str(entry.get('id') or '')
        verdict = str(entry.get('verdict') or '')
        if not key or verdict not in VERDICTS:
            continue
        out[key] = {'verdict': verdict, 'reason': str(entry.get('reason') or ''),
                    'steps': entry.get('steps') or []}
    if not out:
        raise ValueError('回复里没有一条可用的判决')
    return out


def to_items(case: dict, verdicts: dict) -> list[dict]:
    """判决 → 计分项。形状和 `check_targets` 产出的一样，两者在计分上没有区别。

    **模型漏掉的那一条记「判不了」，不记通过。** 漏判当成通过，等于让一个不肯回答的
    裁判给出满分。
    """
    items = []
    for item in benchmark_case.requirements(case):
        got = verdicts.get(item['id'])
        if got is None:
            items.append({**_base(item), 'ok': False, 'measurable': False,
                          'detail': '裁判没有给出这一条的判决'})
            continue
        items.append({**_base(item),
                      'ok': got['verdict'] == 'pass',
                      'measurable': got['verdict'] != 'unmeasurable',
                      'detail': got['reason'],
                      'steps': got.get('steps') or []})
    return items


def _base(item: dict) -> dict:
    return {'id': item['id'], 'kind': 'requirement', 'text': item['text'],
            'dimension': item['dimension'], 'weight': item['weight']}


async def judge(case: dict, observations: dict, facts: dict, agent_track: list,
                model: str | None = None) -> dict:
    """跑一次裁判。返回 `{'items': [...], 'model': ..., 'error': ...}`。

    走 `client.call` 这个统一入口，token 用量才会被记上 —— 裁判也是要花钱的，而一次
    跑分里裁判花了多少，和被测系统花了多少一样该被看见。
    """
    import client

    messages = build_request(case, observations, facts, agent_track)
    chosen = model or ((case.get('test') or {}).get('judge') or {}).get('model') or None

    last_error = ''
    # 先给一个空的：`client.call` 自己抛 ValueError 的话，下面那条重试分支会去读
    # `response` —— 没有这一行就是 UnboundLocalError，而它会把一个「调用失败」伪装成
    # 一个莫名其妙的内部错误，真正的原因反而丢了。
    response: dict = {}
    for attempt in range(2):
        try:
            response = await client.call(
                messages, [],                      # 裁判不给工具
                model_override=chosen,
                caller_info={'agent_type': 'benchmark_judge'})
            return {'items': to_items(case, parse(response.get('content') or '')),
                    'model': chosen or _default_model(), 'error': '',
                    'raw': ''}
        except ValueError as exc:                  # 解析失败 —— 再给一次机会
            last_error = str(exc)
            messages = messages + [
                {'role': 'assistant', 'content': (response.get('content') or '')[:2000]},
                {'role': 'user', 'content': '上面不是合法 JSON。只输出 JSON 对象，'
                                            '不要任何其它文字。'}]
        except Exception as exc:                   # noqa: BLE001 — 调用本身失败
            last_error = f'{type(exc).__name__}: {exc}'
            break

    # 记 error，**不记 0 分**。0 分的意思是「测了，没过」，而这里发生的是「没测成」。
    return {'items': [], 'model': chosen or _default_model(), 'error': last_error,
            'raw': str(response.get('content') or '')[:2000]}


def _default_model() -> str:
    import config
    configs = config.main.get('client', {}).get('llm', [])
    return configs[0]['model'] if configs else ''
