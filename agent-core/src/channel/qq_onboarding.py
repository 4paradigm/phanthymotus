"""Secure short-lived QQ C2C owner pairing for Dashboard onboarding.

The Dashboard issues a code. Only an actual QQ SDK private message can redeem
it; the authenticated Dashboard operator must confirm the matched principal.
"""
import hashlib
import hmac
import secrets
import time
import threading
from functools import wraps

import config
from channel import acl
from channel.tool_policy import _qq_owner_dm_identity

_LOCK = threading.RLock()


def _synchronized(fn):
    @wraps(fn)
    def inner(*args, **kwargs):
        with _LOCK:
            return fn(*args, **kwargs)
    return inner


_KEY = 'qq_onboarding_pairing'
_TTL = 300


def _digest(code: str) -> str:
    return hashlib.sha256(code.encode('ascii')).hexdigest()


def _state(channel_id: str) -> dict:
    state = config.main.get(_KEY, {})
    if (not isinstance(state, dict) or state.get('channel_id') != channel_id
            or time.time() - state.get('created_at', 0) > _TTL):
        return {}
    return state


@_synchronized
def start(channel_id: str) -> str:
    code = secrets.token_hex(4).upper()
    config.main[_KEY] = {
        'channel_id': channel_id, 'code_hash': _digest(code),
        'created_at': time.time(), 'principal': '',
    }
    return code


@_synchronized
def receive(msg) -> bool:
    """Capture a verified QQ private message; never trust user-supplied OpenIDs."""
    if (msg.platform != 'qq' or msg.chat_type != 'p2p'
            or msg.sender_type != 'user'
            or not _qq_owner_dm_identity(msg.user_id)):
        return False
    state = _state(msg.channel_id)
    if not state or state.get('principal'):
        return False
    text = msg.text.strip()
    if not text.startswith('绑定 ') or len(text) != 11:
        return False
    code = text[3:].upper()
    if not all(c in '0123456789ABCDEF' for c in code):
        return False
    if not hmac.compare_digest(_digest(code), state['code_hash']):
        return False
    state['principal'] = msg.user_id
    state.pop('code_hash', None)
    config.main[_KEY] = state
    return True


def status(channel_id: str) -> dict:
    state = _state(channel_id)
    if not state:
        return {'status': 'expired'}
    principal = state.get('principal', '')
    return {'status': 'verified' if principal else 'waiting',
            'user_suffix': principal[-6:] if principal else ''}


@_synchronized
def confirm(channel_id: str) -> dict:
    state = _state(channel_id)
    principal = state.get('principal', '')
    if not principal or not _qq_owner_dm_identity(principal):
        raise ValueError('No verified QQ private message; start pairing again')
    from channel.manager import get_channel_config
    channel = get_channel_config(channel_id)
    if not channel or channel['platform'] != 'qq':
        raise ValueError('QQ channel not found')
    if principal.split(':')[1] != channel['config'].get('app_id'):
        raise ValueError('QQ account does not match this bot')
    old = config.main.get('qq_c2c_owner_grant', {})
    if isinstance(old, dict) and old.get('enabled') and old.get('principal') != principal:
        raise ValueError('Another QQ owner is already bound; revoke explicitly first')
    # Owner identity and privileged tool grants are distinct.
    from channel.permissions import OWNER_FILTER
    previous = acl.get_user('qq', principal)
    acl.upsert_user('qq', principal, (previous or {}).get('display_name', 'QQ owner'),
                    role='owner', tool_filter=OWNER_FILTER)
    config.main['qq_c2c_owner_grant'] = {'enabled': True, 'principal': principal}
    config.main[_KEY] = {}
    return {'status': 'bound', 'user_suffix': principal[-6:]}


@_synchronized
def restore_owner_full_access(channel_id: str) -> dict:
    """Upgrade only the Dashboard-confirmed QQ C2C owner from older presets.

    No other user or group can gain privileges through this migration.
    """
    from channel.manager import get_channel_config
    from channel.permissions import OWNER_FILTER

    channel = get_channel_config(channel_id)
    if channel is None or channel.get('platform') != 'qq':
        raise ValueError('QQ channel not found')
    app_id = str((channel.get('config') or {}).get('app_id', ''))
    grant = config.main.get('qq_c2c_owner_grant', {})
    principal = grant.get('principal', '') if isinstance(grant, dict) else ''
    if not (isinstance(grant, dict)
            and grant.get('enabled') is True
            and _qq_owner_dm_identity(principal)
            and principal.split(':')[1] == app_id):
        raise ValueError('No verified owner grant for this QQ bot')
    user = acl.get_user('qq', principal)
    if not user or user.get('role') != 'owner':
        raise ValueError('QQ principal is not an Owner')
    if user.get('tool_filter') != OWNER_FILTER:
        acl.upsert_user('qq', principal, user.get('display_name', ''),
                        role='owner', tool_filter=OWNER_FILTER)
    return {'status': 'full_access', 'user_suffix': principal[-6:]}


@_synchronized
def revoke(channel_id: str) -> dict:
    """Explicit Dashboard administrator recovery; never implicit from a role edit."""
    from channel.manager import get_channel_config
    from channel.permissions import CHAT_FILTER
    channel = get_channel_config(channel_id)
    if channel is None or channel.get('platform') != 'qq':
        raise ValueError('QQ channel not found')
    app_id = (channel.get('config') or {}).get('app_id', '')
    grant = config.main.get('qq_c2c_owner_grant', {})
    principal = grant.get('principal', '') if isinstance(grant, dict) else ''
    if not (app_id and principal.startswith('official:' + app_id + ':')):
        raise ValueError('This channel has no bound QQ owner')
    previous = acl.get_user('qq', principal)
    if previous and previous.get('role') == 'owner':
        acl.upsert_user('qq', principal, previous.get('display_name', ''),
                        role='viewer', tool_filter=CHAT_FILTER)
    config.main['qq_c2c_owner_grant'] = {}
    config.main[_KEY] = {}
    return {'status': 'revoked', 'user_suffix': principal[-6:]}
