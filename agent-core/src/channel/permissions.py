"""Human-friendly permission presets; server-side authorization remains final."""

OWNER_FILTER = '*'  # Only for the verified QQ C2C Owner with separate local owner grant.
CHAT_FILTER = 'mcp__channel__channel_reply,finish'
DAILY_FILTER = (CHAT_FILTER
                + ',mcp__*__tts'
                + ',mcp__*__speaker__get_volume'
                + ',mcp__*__speaker__set_volume')

PRESETS = {
    'chat': ('viewer', CHAT_FILTER),
    'daily': ('operator', DAILY_FILTER),
}


def assign_preset(platform: str, user_id: str, preset: str) -> dict:
    from channel import acl
    if preset not in PRESETS:
        raise ValueError('Unsupported permission preset')
    current = acl.get_user(platform, user_id)
    if current is None:
        raise ValueError('User must send a message before permissions can be assigned')
    role, filt = PRESETS[preset]
    if current['role'] == 'owner':
        import config
        grant = config.main.get('qq_c2c_owner_grant', {})
        if platform == 'qq' and isinstance(grant, dict) and grant.get('principal') == user_id and grant.get('enabled'):
            raise ValueError('Revoke the bound QQ owner before changing its preset')
    return acl.upsert_user(platform, user_id, current.get('display_name', ''),
                           role=role, tool_filter=filt)
