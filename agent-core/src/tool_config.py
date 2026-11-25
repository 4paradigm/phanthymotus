"""
tool_config.py — the saved-tool-config scope contract, in one place.

A tool's `configSchema` marks each property with `scope: "instance"` or leaves
it shared (the default). That split is not cosmetic — it decides how an
`action: config` call must be addressed:

  - shared settings go WITHOUT instance_id
  - per-instance settings go WITH it

A plugin that keeps one inference engine per process (perception's `ocr`, for
one) rejects shared keys addressed to a single instance, so a merged payload
fails outright rather than partially applying.

Lives here rather than in api/mcp_manage.py because mcp_client.py needs it too,
and api/mcp_manage.py already imports mcp_client — the reverse would cycle.
"""

import json

import config


def find_tool(mcp_id: str, tool_name: str) -> dict:
    """The tool object for mcp_id:tool_name, or {} (tools may be bare strings)."""
    # `or {}` on both hops, not just the inner one: a config row can hold a
    # literal null, and `.get(key, {})` does not apply its default to a stored
    # None. Reached from the stop path now, which must never raise.
    mcps = (config.main.get('services') or {}).get('mcp') or []
    target = next((m for m in mcps if m.get('id') == mcp_id), None)
    for tool in (target or {}).get('tools', []) or []:
        if isinstance(tool, dict) and tool.get('name') == tool_name:
            return tool
    return {}


def split_config_by_scope(tool_obj: dict, cfg: dict) -> tuple[dict, dict]:
    """Partition a saved config body into (shared, instance) by declared scope.

    Keys absent from the *current* configSchema are dropped. A schema that no
    longer advertises a field cannot accept it, and a stale row would otherwise
    be replayed on every start forever — including malformed values such as the
    literal '[object Object]' written by older UI builds that rendered
    object-typed fields as text inputs. Those then blow up in the plugin's own
    option parsing (`dict("[object Object]")`), turning a dead config row into a
    hard start failure.
    """
    props = ((tool_obj or {}).get('configSchema') or {}).get('properties') or {}
    shared: dict = {}
    instance: dict = {}
    for key, value in (cfg or {}).items():
        prop = props.get(key)
        if prop is None:
            continue
        (instance if prop.get('scope') == 'instance' else shared)[key] = value
    return shared, instance


def missing_required_config(tool_obj: dict, merged_cfg: dict) -> list[str]:
    """Required configSchema keys that merged_cfg does not satisfy.

    Only `required` counts. Having a configSchema at all is not evidence that a
    tool needs configuring — perception's `ocr` and `vop` declare only optional
    knobs with defaults, so blocking their start until *something* had been
    saved rejected a perfectly startable card.
    """
    schema = (tool_obj or {}).get('configSchema') or {}
    props = schema.get('properties') or {}
    return [
        key for key in (schema.get('required') or [])
        if key in props and merged_cfg.get(key) in (None, '')
        and required_applies(props, key, merged_cfg)
    ]


def required_applies(props: dict, key: str, cfg: dict) -> bool:
    """`x-required-when: {field: value | [values]}` 收窄一个 required 字段的适用条件。

    decision_core 的云端 URL / Key 只在「大模型」选了云端时必填 —— 选本机模型时它们
    是后备，留空也该能启动。控制字段没存值时按它的 default 算。
    """
    cond = (props.get(key) or {}).get('x-required-when')
    if not cond:
        return True
    for ctrl, want in cond.items():
        actual = cfg.get(ctrl)
        if actual in (None, ''):
            actual = (props.get(ctrl) or {}).get('default', '')
        if actual not in (want if isinstance(want, list) else [want]):
            return False
    return True


def plan_config_calls(tool_obj: dict, saved_shared: dict, saved_instance: dict,
                      instance_id: str) -> list[tuple[dict, dict]]:
    """Build the ordered (body, extra_args) config calls for a start.

    Either saved row can hold either kind of key — older UI builds saved without
    filtering by scope — so both are partitioned and then recombined by scope
    rather than by which row they came from.
    """
    shared_a, instance_a = split_config_by_scope(tool_obj, saved_shared)
    shared_b, instance_b = split_config_by_scope(tool_obj, saved_instance)
    shared_cfg = {**shared_a, **shared_b}
    instance_cfg = {**instance_a, **instance_b}

    if instance_id:
        calls = [(shared_cfg, {}), (instance_cfg, {'instance_id': instance_id})]
    else:
        # No instance to address: instance-scope values act as shared defaults.
        calls = [({**shared_cfg, **instance_cfg}, {})]
    return [(body, extra) for body, extra in calls if body]


def saved_config_rows(mcp_id: str, tool_name: str,
                      instance_id: str = '') -> tuple[dict, dict]:
    """The persisted (shared, instance) rows for a tool, as the UI saved them."""
    shared = config.main.get(f'tool_config:{mcp_id}:{tool_name}', None) or {}
    instance = {}
    if instance_id:
        instance = config.main.get(
            f'tool_config:{mcp_id}:{tool_name}:{instance_id}', None) or {}
    return shared, instance


def plan_saved_config_calls(mcp_id: str, tool_name: str,
                            instance_id: str = '') -> tuple[list, list[str], dict]:
    """(calls, missing, tool_obj) for replaying a tool's saved config.

    One reader for every path that pushes saved config at a device. Each used
    to re-implement row loading, stale-key dropping and the required check, and
    they had already drifted: the reconnect restore sent instance-scope keys
    without addressing (its own docstring said it did not), and the LLM bridge
    skipped the required check whenever any row existed — so a row holding an
    emptied required field started a card the dashboard path would refuse.

    `missing` is computed on the same merged view `plan_config_calls` will push,
    so a refusal here and a push there can never disagree about what is set.
    Callers decide what a non-empty `missing` means: the start paths refuse,
    the reconnect restore deliberately ignores it.
    """
    tool_obj = find_tool(mcp_id, tool_name)
    saved_shared, saved_instance = saved_config_rows(mcp_id, tool_name, instance_id)
    calls = plan_config_calls(tool_obj, saved_shared, saved_instance, instance_id)
    merged: dict = {}
    for body, _extra in calls:
        merged.update(body)
    missing = missing_required_config(tool_obj, merged)
    return calls, missing, tool_obj


def config_result_adapter_ok(result: dict) -> bool:
    """False only when a config call explicitly reported unusable credentials.

    `adapter_ok: false` is a driver's verdict on the url/key it was handed.
    Everything else — a non-JSON ack, an unexpected shape, a missing field —
    defaults to True: a parsing hiccup in the *checker* must not invent a new
    way for a card to fail to start.
    """
    try:
        text = ((result or {}).get('content') or [{}])[0].get('text', '{}')
        parsed = json.loads(text)
        return bool(parsed.get('adapter_ok', True))
    except (json.JSONDecodeError, IndexError, KeyError, TypeError, AttributeError):
        return True
