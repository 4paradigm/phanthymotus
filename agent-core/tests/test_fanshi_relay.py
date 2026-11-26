"""Gateway protocol regression tests using an injected transport only."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fanshi_relay import RelayClient, RelayError

ACTION = 'fanshi_' + 'a' * 32
PAYLOAD = {'message': {'header': {'toid': [123456]}, 'body': [
    {'type': 'TEXT', 'content': 'test'}, {'type': 'IMAGE', 'content': 'YWJj'},
    {'type': 'AT', 'atuserids': ['manager'], 'atall': False}]}}


class RelayTests(unittest.TestCase):
    def setUp(self):
        self.client = RelayClient.__new__(RelayClient)
        self.client._request = Mock()

    def run_send(self, cancelled=lambda: False, **kw):
        return self.client.run(ACTION, PAYLOAD, cancelled, lambda value: None,
                               poll_interval=0, **kw)

    def test_send_omits_destination_and_returns_receipt(self):
        self.client._request.return_value = {'action_id': ACTION, 'status': 'accepted', 'ok': True}
        result = self.run_send()
        self.assertEqual(result['status'], 'accepted')
        method, route, data = self.client._request.call_args.args
        self.assertEqual((method, route), ('POST', '/v1/send'))
        self.assertEqual(data, {'action_id': ACTION, 'text': 'test',
                                'image_base64': 'YWJj', 'at_userid': 'manager'})

    def test_lost_post_response_recovers_with_get_only(self):
        self.client._request.side_effect = [RelayError('lost response'),
            {'action_id': ACTION, 'status': 'accepted', 'ok': True}]
        self.assertEqual(self.run_send()['status'], 'accepted')
        self.assertEqual([call.args[:2] for call in self.client._request.call_args_list],
                         [('POST', '/v1/send'), ('GET', '/v1/result/' + ACTION)])

    def test_stop_before_submission_never_posts(self):
        self.assertEqual(self.run_send(cancelled=lambda: True)['status'], 'cancelled')
        self.client._request.assert_not_called()

    def test_stop_after_submission_requests_gateway_cancellation(self):
        self.client._request.side_effect = [
            {'action_id': ACTION, 'status': 'queued'},
            {'action_id': ACTION, 'status': 'cancelled'}]
        cancelled = Mock(side_effect=[False, True])
        self.assertEqual(self.run_send(cancelled=cancelled)['status'], 'cancelled')
        self.assertEqual(self.client._request.call_args.args[:2], ('POST', '/v1/cancel'))
        self.assertEqual(self.client._request.call_count, 2)

    def test_timeout_is_unknown_and_does_not_resubmit(self):
        self.client._request.return_value = {'action_id': ACTION, 'status': 'sending'}
        self.assertEqual(self.run_send(timeout=0)['status'], 'unknown')
        self.assertEqual(self.client._request.call_count, 1)

    def test_wrong_action_and_unrecognized_status_are_rejected(self):
        for value in ({'action_id': 'wrong', 'status': 'accepted'},
                      {'action_id': ACTION, 'status': 'sent'}):
            with self.assertRaises(RelayError):
                self.client._validate_result(value, ACTION)
        self.assertNotIn('token', self.client._validate_result(
            {'action_id': ACTION, 'status': 'accepted', 'token': 'TEST_SECRET'}, ACTION))

    def test_missing_file_is_optional_but_malformed_file_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'client.json'
            with patch.dict(os.environ, {'FANSHI_RELAY_CONFIG': str(path)}):
                self.assertIsNone(RelayClient.load())
                for data in ('invalid json', 'x' * 32769, json.dumps({'relay_url': 'http://example.invalid'})):
                    path.write_text(data)
                    with self.assertRaises(RelayError):
                        RelayClient.load()

    def test_provisioned_tls_and_robot_auth_are_used(self):
        cfg = {'relay_url': 'https://example.invalid:18767', 'device_id': 'a' * 32,
               'token': 'b' * 48, 'ca_pem': 'TEST_CERTIFICATE'}
        with patch('fanshi_relay.ssl.create_default_context') as tls, patch('fanshi_relay.build_opener') as opener:
            client = RelayClient(cfg)
            tls.assert_called_once_with(cadata='TEST_CERTIFICATE')
            response = opener.return_value.open.return_value
            response.status = 200
            response.read.return_value = json.dumps({'action_id': ACTION, 'status': 'accepted'}).encode()
            client.result(ACTION)
            req = opener.return_value.open.call_args.args[0]
            self.assertEqual(req.get_header('Authorization'), 'Bearer ' + cfg['token'])
            self.assertEqual(req.get_header('X-robot-id'), cfg['device_id'])
            self.assertNotIn(cfg['token'], client.identity)


if __name__ == '__main__':
    unittest.main()
