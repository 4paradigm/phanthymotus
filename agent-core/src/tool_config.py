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

import config


def find_tool(mcp_id: str, tool_name: str) -> dict:
    """The tool object for mcp_id:tool_name, or {} (tools may be bare strings)."""
    mcps = config.main.get('services', {}).get('mcp', []) or []
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
    ]


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
