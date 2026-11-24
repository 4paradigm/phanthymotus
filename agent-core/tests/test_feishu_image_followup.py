"""Optional Feishu images must supplement only their sender's recent @ request."""

import asyncio
import json
import os
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

import collector  # noqa: E402
import config  # noqa: E402
from channel.adapter import Attachment, KIND_IMAGE  # noqa: E402
from channel.adapters.feishu import FeishuAdapter  # noqa: E402
from channel.manager import ChannelManager  # noqa: E402


def _message(message_id, *, msg_type='text', mention=True, user='ou_alice',
             chat='oc_group', chat_type='group', sender_type='user', content=None):
    if content is None:
        content = ({'image_key': 'img_test'} if msg_type == 'image'
                   else {'text': '@_user_1 请识别图片中的文字'})
    return {
        'message_id': message_id, 'message_type': msg_type,
        'content': json.dumps(content), 'chat_id': chat, 'chat_type': chat_type,
        'sender_id': user, 'sender_type': sender_type,
        'mentions': [{'key': '@_user_1', 'open_id': 'ou_self'}] if mention else [],
    }


def _image(message_id='om_image', **kwargs):
    return _message(message_id, msg_type='image', mention=False, **kwargs)


class FeishuImageFollowupTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.received = []

        async def receive(msg):
            self.received.append(msg)

        self.adapter = self.make_adapter(receive)

    def make_adapter(self, callback):
        adapter = FeishuAdapter('feishu_test', 'feishu', {
            'bot_to_bot_enabled': True,
        }, callback)
        adapter._bot_open_id = 'ou_self'
        adapter._download = mock.AsyncMock(return_value=Attachment(
            kind=KIND_IMAGE, path='/work/resource/channel_files/photo.jpg', name='photo.jpg',
        ))
        return adapter

    async def test_text_is_delivered_immediately_without_any_image(self):
        await self.adapter._process_event(_message('om_text', content={'text': '@_user_1 你好'}))

        self.assertEqual(len(self.received), 1)
        self.assertEqual(self.received[0].text, '@_user_1 你好')
        self.assertEqual(self.received[0].attachments, [])
        self.assertEqual(self.received[0].related_request, {})
        self.adapter._download.assert_not_awaited()

    async def test_image_keeps_its_own_id_and_carries_the_original_request(self):
        await self.adapter._process_event(_message('om_text'))
        await self.adapter._process_event(_image())

        self.assertEqual(len(self.received), 2)
        supplement = self.received[1]
        self.assertEqual(supplement.message_id, 'om_image')
        self.assertEqual(supplement.related_request, {
            'message_id': 'om_text', 'text': self.received[0].text,
        })
        self.assertEqual(supplement.attachments[0].kind, KIND_IMAGE)
        self.adapter._download.assert_awaited_once_with(
            'om_image', 'img_test', 'image', KIND_IMAGE,
            name='image.jpg', fallback_ext='.jpg',
        )

    async def test_other_senders_chats_and_channels_cannot_reuse_a_request(self):
        await self.adapter._process_event(_message('om_text'))
        await self.adapter._process_event(_image('om_bob', user='ou_bob'))
        await self.adapter._process_event(_image('om_other_chat', chat='oc_other'))
        other_channel = self.make_adapter(mock.AsyncMock())
        other_channel.channel_id = 'feishu_other'
        await other_channel._process_event(_image('om_other_channel'))

        self.assertEqual(len(self.received), 1)
        self.adapter._download.assert_not_awaited()
        other_channel._download.assert_not_awaited()
        await self.adapter._process_event(_image())
        self.assertEqual(len(self.received), 2)

    async def test_unrelated_group_messages_are_filtered_before_downloading(self):
        await self.adapter._process_event(_message('om_text'))
        for kind in ('text', 'audio', 'file', 'media', 'sticker'):
            await self.adapter._process_event(_message('om_' + kind, msg_type=kind, mention=False))
        other_mention = _message('om_other_bot')
        other_mention['mentions'][0]['open_id'] = 'ou_other_bot'
        await self.adapter._process_event(other_mention)

        self.assertEqual(len(self.received), 1)
        self.adapter._download.assert_not_awaited()

    async def test_legacy_disabled_setting_does_not_block_followups(self):
        self.adapter.config['image_followup_enabled'] = False
        await self.adapter._process_event(_message('om_text'))
        await self.adapter._process_event(_image())
        await self.adapter._process_event(_message('om_unmentioned', mention=False))
        self.assertEqual(len(self.received), 2)
        self.assertEqual(self.received[-1].related_request['message_id'], 'om_text')
        self.adapter._download.assert_awaited_once()

    async def test_private_images_remain_standalone_inputs(self):
        await self.adapter._process_event(_image(chat_type='p2p'))

        self.assertEqual(len(self.received), 1)
        self.assertEqual(self.received[0].related_request, {})
        self.assertTrue(self.received[0].attachments)

    async def test_multiple_images_share_context_without_extending_the_time_window(self):
        with mock.patch('channel.adapters.feishu.time.monotonic', return_value=100):
            await self.adapter._process_event(_message('om_text'))
        with mock.patch('channel.adapters.feishu.time.monotonic', return_value=130):
            await self.adapter._process_event(_image('om_image1'))
            await self.adapter._process_event(_image('om_image2'))
        with mock.patch('channel.adapters.feishu.time.monotonic', return_value=160):
            await self.adapter._process_event(_image('om_expired'))

        self.assertEqual([msg.message_id for msg in self.received], [
            'om_text', 'om_image1', 'om_image2',
        ])
        self.assertEqual(self.adapter._download.await_count, 2)

    async def test_latest_mention_replaces_context_and_duplicate_images_are_dropped(self):
        await self.adapter._process_event(_message('om_old'))
        await self.adapter._process_event(_message('om_new', content={'text': '新的要求'}))
        await self.adapter._process_event(_image())
        await self.adapter._process_event(_image())

        self.assertEqual(len(self.received), 3)
        self.assertEqual(self.received[-1].related_request, {
            'message_id': 'om_new', 'text': '新的要求',
        })
        self.adapter._download.assert_awaited_once()

    async def test_image_can_arrive_while_the_text_callback_is_still_busy(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def receive(msg):
            self.received.append(msg)
            if msg.message_id == 'om_text':
                entered.set()
                await release.wait()

        self.adapter._on_message = receive
        text_task = asyncio.create_task(self.adapter._process_event(_message('om_text')))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            await self.adapter._process_event(_image())
        finally:
            release.set()
            await text_task

        self.assertEqual(self.received[-1].related_request['message_id'], 'om_text')

    async def test_context_is_snapshotted_before_a_slow_image_download(self):
        entered, release = asyncio.Event(), asyncio.Event()
        attachment = self.adapter._download.return_value

        async def download(*_args, **_kwargs):
            entered.set()
            await release.wait()
            return attachment

        await self.adapter._process_event(_message('om_old'))
        self.adapter._download.side_effect = download
        image_task = asyncio.create_task(self.adapter._process_event(_image()))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            await self.adapter._process_event(_message('om_new'))
        finally:
            release.set()
            await image_task

        self.assertEqual(self.received[-1].related_request['message_id'], 'om_old')
        await self.adapter._process_event(_image('om_next_image'))
        self.assertEqual(self.received[-1].related_request['message_id'], 'om_new')

    async def test_rich_text_image_followup_is_supported(self):
        await self.adapter._process_event(_message('om_text'))
        await self.adapter._process_event(_message('om_post', msg_type='post', mention=False,
            content={'content': [[{'tag': 'img', 'image_key': 'img_post'}]]}))

        self.assertEqual(len(self.received), 2)
        self.assertEqual(self.received[-1].related_request['message_id'], 'om_text')
        self.assertTrue(self.received[-1].attachments)

    async def test_slow_earlier_post_cannot_replace_a_newer_request(self):
        entered, release = asyncio.Event(), asyncio.Event()
        attachment = self.adapter._download.return_value

        async def download(*_args, **_kwargs):
            entered.set()
            await release.wait()
            return attachment

        self.adapter._download.side_effect = download
        post = _message('om_old_post', msg_type='post', content={'content': [[
            {'tag': 'text', 'text': '旧的图片要求'},
            {'tag': 'img', 'image_key': 'img_post'},
        ]]})
        post_task = asyncio.create_task(self.adapter._process_event(post))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            await self.adapter._process_event(_message('om_new', content={'text': '新的要求'}))
        finally:
            release.set()
            await post_task
        await self.adapter._process_event(_image())

        self.assertEqual(self.received[-1].related_request, {
            'message_id': 'om_new', 'text': '新的要求',
        })

    async def test_download_failure_is_explicit_and_allows_a_fresh_image(self):
        await self.adapter._process_event(_message('om_text'))
        self.adapter._download.return_value = None
        await self.adapter._process_event(_image('om_failed'))
        self.assertIn('下载失败', self.received[-1].text)
        self.assertEqual(self.received[-1].attachments, [])

        self.adapter._download.return_value = Attachment(kind=KIND_IMAGE, path='/tmp/retry.jpg')
        await self.adapter._process_event(_image('om_retry'))
        self.assertEqual(self.received[-1].related_request['message_id'], 'om_text')
        self.assertTrue(self.received[-1].attachments)

    async def test_failed_image_only_post_is_not_silently_dropped(self):
        await self.adapter._process_event(_message('om_text'))
        self.adapter._download.return_value = None
        await self.adapter._process_event(_message('om_post', msg_type='post', mention=False,
            content={'content': [[{'tag': 'img', 'image_key': 'img_post'}]]}))

        self.assertEqual(len(self.received), 2)
        self.assertIn('下载失败', self.received[-1].text)

    async def test_missing_bot_identity_fails_closed_for_group_followups(self):
        self.adapter._bot_open_id = ''
        await self.adapter._process_event(_message('om_text'))
        await self.adapter._process_event(_image())
        self.assertEqual(self.received, [])
        self.adapter._download.assert_not_awaited()

    async def test_bot_messages_do_not_seed_or_use_human_image_context(self):
        await self.adapter._process_event(_message('om_bot', sender_type='bot'))
        await self.adapter._process_event(_image('om_human_image'))
        await self.adapter._process_event(_message('om_human'))
        await self.adapter._process_event(_image('om_bot_image', sender_type='bot'))

        self.assertEqual([msg.message_id for msg in self.received], ['om_bot', 'om_human'])
        self.adapter._download.assert_not_awaited()

    async def test_stop_clears_the_temporary_context(self):
        await self.adapter._process_event(_message('om_text'))
        await self.adapter.stop()
        await self.adapter._process_event(_image())
        self.assertEqual(len(self.received), 1)
        self.adapter._download.assert_not_awaited()


class FeishuImagePipelineTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.manager = ChannelManager()
        settings = {
            'canvas_layout': {'cards': [{
                'id': 'input', 'mcpId': 'channel', 'toolName': 'channel_request',
            }]},
            'tool_config:channel:channel_request:input': {'channel_id': 'feishu_test'},
        }
        self.published = mock.AsyncMock()
        self.pushed = mock.AsyncMock()
        inspection = types.ModuleType('api.inspection')
        inspection.publish_to_topic = self.published
        motus_stream = types.ModuleType('api.motus_stream')
        motus_stream.push_event = self.pushed
        patches = [
            mock.patch.object(config, 'main', settings),
            mock.patch.dict(sys.modules, {'api.inspection': inspection, 'api.motus_stream': motus_stream}),
            mock.patch('channel.manager.acl.get_user', return_value={'role': 'operator'}),
            mock.patch('channel.manager.acl.check_permission', return_value=(True, 'ok')),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.adapter = FeishuAdapter('feishu_test', 'feishu', {}, self.manager._on_inbound_message)
        self.adapter._bot_open_id = 'ou_self'
        self.adapter._download = mock.AsyncMock(return_value=Attachment(
            kind=KIND_IMAGE, path='/work/resource/channel_files/photo.jpg', name='photo.jpg',
        ))

    async def test_context_and_files_reach_collector_with_exact_reply_routing(self):
        await self.adapter._process_event(_message('om_text'))
        text_payload = json.loads(self.published.await_args.args[1])
        self.assertNotIn('files', text_payload)
        self.assertNotIn('related_request', text_payload)

        await self.adapter._process_event(_image())
        topic, body = self.published.await_args.args
        payload = json.loads(body)
        self.assertEqual(payload['message_id'], 'om_image')
        self.assertEqual(payload['related_request']['message_id'], 'om_text')
        self.assertEqual(payload['files'][0]['kind'], KIND_IMAGE)
        self.assertEqual(self.pushed.await_args.args[0]['payload']['related_request'],
                         payload['related_request'])
        saved = config.main['channel_message_contexts']['feishu_test']
        self.assertEqual(saved['om_image']['chat_id'], 'oc_group')
        self.assertEqual(saved['om_image']['user_id'], 'ou_alice')

        with mock.patch('channel.manager.manager', self.manager):
            trigger = collector._build_trigger([{'source': 'dds:' + topic, 'text': body, 'ts': 1}], True)
        self.assertIn('related_request', trigger['text'])
        self.assertIn(payload['related_request']['text'], trigger['text'])
        self.assertIn(payload['files'][0]['path'], trigger['text'])
        self.assertEqual(trigger['_channel_message_ids'], ['om_image'])

    async def test_blocked_users_still_cannot_publish_requests_or_images(self):
        with mock.patch('channel.manager.acl.check_permission', return_value=(False, 'blocked')):
            await self.adapter._process_event(_message('om_text'))
            await self.adapter._process_event(_image())
        self.published.assert_not_awaited()
        self.pushed.assert_not_awaited()

    async def test_supplement_rechecks_the_current_role(self):
        await self.adapter._process_event(_message('om_text'))
        with mock.patch('channel.manager.acl.get_user', return_value={'role': 'viewer'}):
            await self.adapter._process_event(_image())
        payload = json.loads(self.published.await_args.args[1])
        self.assertEqual(payload['user_role'], 'viewer')
        self.assertEqual(payload['trusted_bot_id'], '')


class FeishuImageDefaultTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        patch = mock.patch.object(config, 'main', {})
        patch.start()
        self.addCleanup(patch.stop)

    async def test_new_channel_does_not_require_an_image_setting(self):
        from api import channel as api

        result = await api.add_channel(api.AddChannelReq(
            id='feishu_test', platform='feishu',
            config={'app_id': 'test-app', 'app_secret': 'test-secret'},
        ))
        self.assertNotIn('image_followup_enabled', result['channel'])
        status = ChannelManager().get_status()[0]
        self.assertNotIn('image_followup_enabled', status)
        self.assertNotIn('test-secret', json.dumps(status))

    async def test_existing_disabled_channel_accepts_followups_after_restart(self):
        from channel import manager as channels

        channel = channels.add_channel_config(
            'feishu_test', 'feishu', {'app_id': 'test-app'},
            enabled=True,
        )
        channel['image_followup_enabled'] = False
        manager = ChannelManager()
        manager._on_inbound_message = mock.AsyncMock()
        with mock.patch.dict(channels._ADAPTER_CLASSES, {'feishu': FeishuAdapter}), \
                mock.patch.object(FeishuAdapter, 'start', new=mock.AsyncMock()):
            await manager._start_adapter(channel, retries=1)

        adapter = manager._adapters['feishu_test']
        adapter._bot_open_id = 'ou_self'
        adapter._download = mock.AsyncMock(return_value=Attachment(
            kind=KIND_IMAGE, path='/work/resource/channel_files/photo.jpg', name='photo.jpg',
        ))
        await adapter._process_event(_message('om_text'))
        await adapter._process_event(_image())
        self.assertEqual(manager._on_inbound_message.await_count, 2)
        self.assertEqual(manager._on_inbound_message.await_args.args[0].related_request['message_id'], 'om_text')


if __name__ == '__main__':
    unittest.main()
