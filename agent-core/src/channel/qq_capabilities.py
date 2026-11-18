"""QQ transport defaults and honest runtime capability descriptions.

A feature enabled in Core is not proof that Tencent granted it or that a
client received a message. Only subscribe to baseline C2C/group intents by
default: additional Guild/interaction intents can make the whole Gateway fail.
Never change explicitly configured False values during an upgrade.
"""
from __future__ import annotations
import time

# No extra Gateway intent required for these options. Voice ASR is a no-op
# without a configured STT provider; this must be reflected in diagnostics.
SAFE_DEFAULTS = {
    'enable_inbound_media': True,
    'enable_outbound_media': True,
    'enable_typing': True,
    'enable_session_resume': True,
    'enable_voice_transcription': True,
}

# These are not silently subscribed: the Tencent bot may lack entitlement.
PLATFORM_GATED = (
    'enable_markdown', 'enable_interactions', 'enable_approvals',
    'enable_guild_plain_messages',
)


def apply_safe_defaults(cfg: dict, *, standard_mode: bool = False) -> dict:
    """Apply safe QQ defaults, respecting advanced-mode explicit opt-outs.

    Older 'standard' channels had disabled defaults; upgrade those to open
    communication. Advanced settings are deliberately preserved.
    """
    result = dict(cfg)
    for key, enabled in SAFE_DEFAULTS.items():
        if standard_mode:
            result[key] = enabled
        else:
            result.setdefault(key, enabled)
    return result


def describe(cfg: dict, *, gateway: bool, flow_ready: bool,
             recent: dict | None = None) -> dict:
    """Report configured / observed / unavailable separately.

    Observations are process-local and expire on restart. Never call a
    configuration toggle or mock test 'verified'.
    """
    recent = recent or {}
    counts = recent.get('counts') or {}
    last_seen = recent.get('recent') or {}
    now = time.time()
    def feature(enabled: bool, success: str, *, requirement: str = '') -> dict:
        if not enabled:
            return {'state': 'not_enabled', 'reason': requirement or 'disabled_in_configuration'}
        if not gateway:
            return {'state': 'not_connected', 'reason': 'gateway_not_ready'}
        if not flow_ready:
            return {'state': 'not_ready', 'reason': 'canvas_not_ready'}
        last = last_seen.get(success)
        if (counts.get(success, 0) and isinstance(last, (int, float))
                and 0 <= now - last <= 1800):
            return {'state': 'observed', 'reason': 'recent_platform_event_or_ack'}
        return {'state': 'awaiting_real_test',
                'reason': 'observation_expired' if counts.get(success, 0)
                          else 'no_live_evidence'}

    group = 'group' in cfg.get('enabled_scopes', ['c2c'])
    result = {
        'private_text': feature('c2c' in cfg.get('enabled_scopes', ['c2c']), 'private_inbound'),
        'group_at_text': feature(group, 'group_inbound'),
        'inbound_media': feature(bool(cfg.get('enable_inbound_media')), 'media_received'),
        'outbound_media': feature(bool(cfg.get('enable_outbound_media')), 'media_sent'),
        'voice_transcription': feature(bool(cfg.get('enable_voice_transcription')), 'voice_transcribed'),
        'typing': feature(bool(cfg.get('enable_typing')), 'typing_sent'),
        'session_resume': {
            'state': 'configured' if cfg.get('enable_session_resume') else 'not_enabled',
            'reason': 'restart_recovery_requires_real_test',
        },
    }
    for name in PLATFORM_GATED:
        result[name] = {
            'state': 'configured_unverified' if cfg.get(name) else 'requires_platform_or_workflow',
            'reason': 'platform_entitlement_or_business_flow_not_verified',
        }
    return result
