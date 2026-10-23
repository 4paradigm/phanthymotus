"""Exercise the real HTTP endpoint using the Core MCP URL convention."""
import http.client
import json
import threading
import time
import unittest

from tests.test_actucore_executor import _load_main_module


class CompletionSSETest(unittest.TestCase):
    def setUp(self):
        self.module = _load_main_module()
        self.server = self.module.ThreadingHTTPServer(
            ("127.0.0.1", 0), self.module.make_handler())
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.connection = http.client.HTTPConnection(*self.server.server_address, timeout=2)

    def tearDown(self):
        self.connection.close()
        # Wake handlers to detect a closed client, rather than wait for the ping.
        deadline = time.monotonic() + 2
        while self.module._sse_clients and time.monotonic() < deadline:
            self.module.sse_push({"type": "test_cleanup"})
            time.sleep(.01)
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)
        self.assertFalse(self.module._sse_clients)

    def check_completion(self, suffix):
        self.connection.request("GET", "/mcp" + suffix)
        response = self.connection.getresponse()
        try:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.getheader("Content-Type"), "text/event-stream")
            deadline = time.monotonic() + 2
            while not self.module._sse_clients and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertTrue(self.module._sse_clients)
            event = {"type": "action_complete", "action_id": "navigation-test",
                     "status": "arrived", "payload": {"nav_id": "navigation-test",
                                                       "status": "arrived"}}
            self.module.sse_push(event)
            line = response.readline().decode()
            self.assertTrue(line.startswith("data: "))
            self.assertEqual(json.loads(line[len("data: "):]), event)
            self.assertEqual(response.readline(), b"\n")
        finally:
            response.close()

    def test_core_subscription_receives_navigation_completion(self):
        self.check_completion("/sse")

    def test_query_string_is_supported(self):
        self.check_completion("/sse?client=core")

    def test_old_path_is_not_an_alias(self):
        self.connection.request("GET", "/sse")
        response = self.connection.getresponse()
        self.assertEqual(response.status, 404)
        response.close()
        self.assertFalse(self.module._sse_clients)
