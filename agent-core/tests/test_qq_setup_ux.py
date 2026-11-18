"""Onboarding must not require QQ administrator identity or leak chats."""
import time
from unittest import mock

from api.channel_setup import qq_test_status

CHANNEL = 'qq_official_1900000000'
OTHER_CHANNEL = 'qq_official_1900000001'
P2P = 'p2p:ABCDEFGHIJKLMN'
GROUP = 'group:ABCDEFGHIJKLMN'


def row(role, text, ts):
    return {'role': role, 'text': text, 'ts': ts}


def check(chats, *, since=None, channel=CHANNEL):
    state = {'channel_chat_history': {
        CHANNEL: chats,
        OTHER_CHANNEL: {P2P: [row('user', '你好', time.time()),
                               row('assistant', 'not your bot', time.time())]},
    }}
    with (mock.patch('api.channel_setup.auth.is_enabled', return_value=True),
          mock.patch('api.channel_setup._qq',
                     return_value={'platform': 'qq', 'config': {'app_id': '1900000000'}}),
          mock.patch('config.main', state)):
        return qq_test_status(channel, since if since is not None else time.time() - 5)


def test_no_admin_pairing_required_to_verify_private_message():
    now = time.time()
    assert check({}) == {'message_received': False, 'reply_sent': False}
    assert check({P2P: [row('user', '你好', now)]}) == {
        'message_received': True, 'reply_sent': False}
    assert check({P2P: [row('user', '你好', now),
                        row('assistant', 'hello', now + 0.1)]}) == {
        'message_received': True, 'reply_sent': True}


def test_stale_other_text_group_and_other_bot_do_not_count():
    now = time.time()
    assert check({P2P: [row('user', '你好', now - 100),
                        row('assistant', 'old', now - 90)]}, since=now - 5) == {
        'message_received': False, 'reply_sent': False}
    assert check({P2P: [row('user', 'not greeting', now),
                        row('assistant', 'hello', now + .1)]}) == {
        'message_received': False, 'reply_sent': False}
    assert check({GROUP: [row('user', '你好', now),
                          row('assistant', 'in group', now + .1)]}) == {
        'message_received': False, 'reply_sent': False}
    assert check({}) == {'message_received': False, 'reply_sent': False}


def test_reply_in_other_private_chat_does_not_mark_success():
    now = time.time()
    assert check({
        P2P: [row('user', '你好', now)],
        'p2p:SECOND_OPEN_ID': [row('assistant', 'other chat', now + .3)],
    }) == {'message_received': True, 'reply_sent': False}


def test_old_owner_grant_does_not_affect_public_chat_verification():
    now = time.time()
    chats = {P2P: [row('user', '你好', now),
                    row('assistant', 'reply accepted', now + .1)]}
    assert check(chats)['reply_sent'] is True
