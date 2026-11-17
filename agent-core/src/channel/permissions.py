"""Human-friendly permission presets; server-side authorization remains final.

QQ is chat-only; Feishu and other providers retain their existing presets.
"""

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
    if platform == 'qq' and preset != 'chat':
        raise ValueError('QQ supports chat-only permissions')
    current = acl.get_user(platform, user_id)
    if current is None:
        raise ValueError('User must send a message before permissions can be assigned')
    role, filt = PRESETS[preset]
    return acl.upsert_user(platform, user_id, current.get('display_name', ''),
                           role=role, tool_filter=filt)
