"""Compatibility tombstone for retired QQ administrator pairing.

Ordinary QQ users are admitted by the regular authenticated SDK/ACL path.
Legacy pairing state and stored owner grants must never grant permissions.
This module deliberately cannot create, recover or upgrade a QQ owner.
"""
from __future__ import annotations


def start(channel_id: str) -> str:
    raise ValueError('QQ administrator pairing is retired')


def receive(msg) -> bool:
    # Do not intercept ordinary QQ messages resembling old pairing commands.
    return False


def status(channel_id: str) -> dict:
    return {'status': 'retired', 'admin_pairing_required': False}


def confirm(channel_id: str) -> dict:
    raise ValueError('QQ administrator pairing is retired')


def restore_owner_full_access(channel_id: str) -> dict:
    raise ValueError('QQ administrator privilege upgrades are retired')


def revoke(channel_id: str) -> dict:
    # Neither creates nor changes ACL records. Migration of old QQ role labels
    # must happen in explicitly reviewed data maintenance, not chat onboarding.
    raise ValueError('QQ administrator pairing is retired')
