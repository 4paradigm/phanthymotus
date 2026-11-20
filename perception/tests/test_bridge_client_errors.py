import io
import unittest
import urllib.error
from unittest.mock import patch

from plugins.ql_printer import BridgeClient, BridgeError


class BridgeClientErrorsTest(unittest.TestCase):
    def test_timeouts_are_not_reported_as_unreachable_or_retried(self):
        for failure in (TimeoutError('timed out'), urllib.error.URLError(TimeoutError('timed out'))):
            with self.subTest(failure=failure), patch('urllib.request.urlopen', side_effect=failure) as request:
                with self.assertRaises(BridgeError) as caught:
                    BridgeClient('http://frontdesk').request('POST', '/list_visits', body={})
                self.assertEqual(caught.exception.code, 'request_timeout')
                self.assertIn('是否已登录', str(caught.exception))
                self.assertEqual(request.call_count, 1)

    def test_print_timeout_warns_of_unknown_result(self):
        with patch('urllib.request.urlopen', side_effect=TimeoutError()):
            with self.assertRaises(BridgeError) as caught:
                BridgeClient('http://frontdesk').request('POST', '/print_tag', body={})
            self.assertIn('不要立即重复打印', str(caught.exception))

    def test_explicit_login_required_is_preserved(self):
        failure = urllib.error.HTTPError('http://frontdesk', 400, 'error', {},
                                         io.BytesIO(b'{"code":"login_required","error":"Login required"}'))
        with patch('urllib.request.urlopen', side_effect=failure):
            with self.assertRaises(BridgeError) as caught:
                BridgeClient('http://frontdesk').request('POST', '/list_visits', body={})
            self.assertEqual(caught.exception.code, 'login_required')

    def test_refused_connection_remains_unreachable(self):
        with patch('urllib.request.urlopen', side_effect=urllib.error.URLError(ConnectionRefusedError(111, 'refused'))):
            with self.assertRaises(BridgeError) as caught:
                BridgeClient('http://frontdesk').request('GET', '/health', retry_refused=False)
            self.assertEqual(caught.exception.code, 'unreachable')
