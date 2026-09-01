"""Channel list actions must not compile user-controlled IDs as JavaScript."""

import pathlib
import unittest


CHANNELS_JS = pathlib.Path(__file__).resolve().parents[1] / 'web/js/channels.js'


class ChannelWebSecurityTest(unittest.TestCase):
    def test_channel_actions_use_event_listeners_not_inline_javascript(self):
        source = CHANNELS_JS.read_text(encoding='utf-8')

        self.assertNotIn('onclick=', source)
        self.assertNotIn('window._channel', source)
        self.assertIn("_channelList.addEventListener('click', _handleChannelAction)", source)
        for action in ('bot-to-bot', 'trusted-bots', 'stop', 'restart', 'delete'):
            self.assertIn(f'data-channel-action="{action}"', source)

    def test_channel_ids_are_encoded_only_at_the_request_boundary(self):
        source = CHANNELS_JS.read_text(encoding='utf-8')

        self.assertNotIn('data-id="${_esc(ch.id)}"', source)
        self.assertGreaterEqual(source.count('encodeURIComponent(id)'), 4)


if __name__ == '__main__':
    unittest.main()
