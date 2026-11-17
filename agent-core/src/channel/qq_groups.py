"""QQ SDK group auto-discovery and persistent group-level permission ceilings."""
import time
from collections import Counter

import config

_REJECTION_COUNTS = Counter()

_POLICIES = 'qq_group_policies'


def _groups() -> dict:
    raw = config.main.get(_POLICIES, {})
    return raw if isinstance(raw, dict) else {}


def list_groups(channel_id: str) -> list[dict]:
    return [{'group_id': gid, **record, 'default_permission': 'chat'}
            for gid, record in _groups().get(channel_id, {}).items()]


def get_group(channel_id: str, group_id: str) -> dict:
    return _groups().get(channel_id, {}).get(group_id, {})


def approved(channel_id: str, group_id: str) -> bool:
    return get_group(channel_id, group_id).get('status') == 'approved'


def _reject_observation(reason: str) -> bool:
    # Do not log user text, group OpenIDs or app credentials.
    _REJECTION_COUNTS[reason] += 1
    count = _REJECTION_COUNTS[reason]
    if count == 1 or count % 100 == 0:
        print(f'[qq-group] auto_enable_rejected reason={reason} count={count}', flush=True)
    return False


def observe(msg) -> bool:
    """Auto-enable a QQ group on its first authenticated SDK @-message.

    Tencent's Gateway event proves the bot can receive messages from this
    group, not who invited it. Never infer Owner privileges from group
    membership. Explicitly disabled groups remain blocked across restarts.
    """
    if (msg.platform != 'qq' or msg.chat_type != 'group'
            or msg.sender_type != 'user' or not isinstance(msg.chat_id, str)):
        return _reject_observation('invalid_source')
    parts = msg.user_id.split(':') if isinstance(msg.user_id, str) else []
    if (len(parts) != 5 or parts[0] != 'official'
            or parts[2] != 'group' or parts[3] != msg.chat_id
            or not parts[1].isdigit() or not parts[4]):
        return _reject_observation('invalid_principal')
    from channel.manager import get_channel_config
    channel = get_channel_config(msg.channel_id) or {}
    cfg = channel.get('config') or {}
    if (channel.get('platform') != 'qq'
            or not channel.get('enabled', False)
            or cfg.get('app_id') != parts[1]
            or 'group' not in cfg.get('enabled_scopes', ['c2c'])):
        return _reject_observation('channel_config')
    current = get_group(msg.channel_id, msg.chat_id)
    if current:
        if current.get('status') == 'disabled':
            return _reject_observation('group_explicitly_disabled')
        if current.get('status') == 'approved' and current.get('default_permission') == 'chat':
            return True
        # Upgrade old unverified/pending records on the next real SDK @event.
        # Group membership is sufficient; only an explicit disable blocks.
    all_groups = dict(_groups())
    groups = dict(all_groups.get(msg.channel_id, {}))
    groups[msg.chat_id] = {
        'status': 'approved',
        'default_permission': 'chat',
        'mention_only': True,
        'discovered_at': time.time(),
    }
    all_groups[msg.channel_id] = groups
    config.main[_POLICIES] = all_groups
    print('[qq-group] first authorized SDK group message: auto-enabled'
          ' with chat-only permission', flush=True)
    return True


def ensure_default_group_scope(channel: dict) -> dict:
    """Enable group @ by default on upgraded QQ installations, once.

    An explicit administrator opt-out remains authoritative; no group
    whitelist, new pairing, or per-group permission approval is required.
    This modifies only the named QQ channel, never other platform channels.
    """
    if channel.get('platform') != 'qq':
        return channel
    cfg = channel.get('config') or {}
    if (cfg.get('protocol') != 'official'
            or cfg.get('qq_group_scope_opt_out') is True
            or 'group' in cfg.get('enabled_scopes', ['c2c'])):
        return channel

    scopes = list(cfg.get('enabled_scopes', ['c2c']))
    if not scopes:
        return channel
    upgraded = dict(cfg, enabled_scopes=[*scopes, 'group'],
                    private_only=False)
    from channel.adapters.qq_official import validate_qq_official_config
    from channel.manager import update_channel_config
    validate_qq_official_config(upgraded)
    saved = update_channel_config(channel['id'], config=upgraded)
    print('[qq-group] default group scope enabled for existing QQ channel',
          flush=True)
    return saved or {**channel, 'config': upgraded}


def update(channel_id: str, group_id: str, *,
           default_permission: str, mention_only: bool, enabled: bool) -> dict:
    if default_permission != 'chat':
        raise ValueError('QQ group members are fixed to chat-only permissions')
    if not isinstance(mention_only, bool) or not isinstance(enabled, bool):
        raise ValueError('Invalid group setting')
    all_groups = dict(_groups())
    groups = dict(all_groups.get(channel_id, {}))
    current = groups.get(group_id)
    if not current:
        raise ValueError('QQ group has not been discovered from an authenticated event')
    current = dict(current)
    current.update(status='approved' if enabled else 'disabled',
                   default_permission=default_permission, mention_only=mention_only)
    groups[group_id] = current
    all_groups[channel_id] = groups
    config.main[_POLICIES] = all_groups
    return {'group_id': group_id, **current}
