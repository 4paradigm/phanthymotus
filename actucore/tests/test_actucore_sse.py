"""Exercise the real HTTP endpoint using the Core MCP URL convention."""
import http.client
import json
import threading
import time
import unittest

from unittest import mock

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

    def test_mcp_logs_never_include_request_or_response_payload(self):
        arguments = {"action": "secret-action-value", "nested": {"plain": "request-secret"}}
        result = {"plain": "response-secret"}
        bundle = mock.Mock()
        bundle.dispatch.return_value = result
        self.module._bundle = bundle
        with self.assertLogs(self.module.log, level="INFO") as logs:
            self.connection.request("POST", "/mcp", json.dumps({
                "jsonrpc": "2.0", "id": 42, "method": "tools/call",
                "params": {"name": "custom-tool-secret", "arguments": arguments},
            }))
            response = self.connection.getresponse()
            payload = json.loads(response.read())
        bundle.dispatch.assert_called_once_with("custom-tool-secret", arguments)
        self.assertEqual(json.loads(payload["result"]["content"][0]["text"]), result)
        self.assertIn("tools/call completed", " ".join(logs.output))
        for secret in ("secret-action-value", "request-secret", "response-secret", "custom-tool-secret"):
            self.assertNotIn(secret, " ".join(logs.output))

    def test_unknown_tool_and_info_heartbeat_do_not_log_payload(self):
        bundle = mock.Mock()
        bundle.dispatch.return_value = None
        self.module._bundle = bundle
        with self.assertLogs(self.module.log, level="INFO") as logs:
            self.connection.request("POST", "/mcp", json.dumps({
                "jsonrpc": "2.0", "id": 44, "method": "tools/call",
                "params": {"name": "unknown-name-secret", "arguments": {"action": "unknown"}},
            }))
            response = self.connection.getresponse()
            payload = json.loads(response.read())
        self.assertEqual(payload["error"]["message"], "Unknown tool: unknown-name-secret")
        self.assertIn("unknown tool", " ".join(logs.output))
        self.assertNotIn("unknown-name-secret", " ".join(logs.output))

        bundle.dispatch.return_value = {"value": "heartbeat-secret"}
        with mock.patch.object(self.module.log, "info") as info_log:
            self.connection.request("POST", "/mcp", json.dumps({
                "jsonrpc": "2.0", "id": 45, "method": "tools/call",
                "params": {"name": "tool", "arguments": {"action": "info"}},
            }))
            response = self.connection.getresponse()
            payload = json.loads(response.read())
        self.assertEqual(json.loads(payload["result"]["content"][0]["text"]),
                         {"value": "heartbeat-secret"})
        info_log.assert_not_called()

    def test_rpc_exception_logs_do_not_include_exception_payload(self):
        bundle = mock.Mock()
        bundle.dispatch.side_effect = RuntimeError("exception-secret")
        self.module._bundle = bundle
        with self.assertLogs(self.module.log, level="INFO") as logs:
            self.connection.request("POST", "/mcp", json.dumps({
                "jsonrpc": "2.0", "id": 43, "method": "tools/call",
                "params": {"name": "tool", "arguments": {"action": "call"}},
            }))
            response = self.connection.getresponse()
            payload = json.loads(response.read())
        self.assertEqual(payload["error"]["message"], "exception-secret")
        self.assertIn("RPC handler failed", " ".join(logs.output))
        self.assertNotIn("exception-secret", " ".join(logs.output))
        self.assertNotIn("Traceback", " ".join(logs.output))

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
