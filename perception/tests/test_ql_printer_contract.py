"""QL_printer 卡片与安全桥接协议的契约测试。"""

from __future__ import annotations

import json
import sys
import threading
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from plugins.ql_printer import QLPrinterPlugin  # noqa: E402
from ql_printer_mock_bridge import MockBridge  # noqa: E402

TOKEN = "test-token"


class VisitorBadgeTest(unittest.TestCase):
    def setUp(self):
        self.bridge = MockBridge(TOKEN)
        self.server = self.bridge.serve("127.0.0.1", 0)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.card = QLPrinterPlugin({})
        cfg = self.card.dispatch("config", {"bridge_url": self.url, "token": TOKEN})
        self.assertTrue(cfg["adapter_ok"])

    def tearDown(self):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()

    def call(self, action, **args):
        result = self.card.dispatch("QL_printer", dict(args, action=action))
        json.dumps(result, ensure_ascii=False)
        return result

    def find_ticket(self, name="王磊", host=""):
        result = self.call("find", name=name, host=host)
        self.assertEqual(result["status"], "found")
        return result["visits"][0]["ticket"]

    def test_tool_definition_uses_ticket_protocol(self):
        definition = self.card.get_tool()
        self.assertEqual(definition["name"], "QL_printer")
        self.assertEqual(definition["type"], "actuator")
        actions = definition["inputSchema"]["properties"]["action"]["enum"]
        self.assertEqual(set(actions), {"start", "stop", "info", "config", "find", "print"})
        schema_text = json.dumps(definition["inputSchema"], ensure_ascii=False)
        self.assertIn('"print"', schema_text)
        props = definition["configSchema"]["properties"]
        self.assertEqual(props["token"]["format"], "password")
        self.assertTrue(props["bridge_url"]["x-sensitive"])
        self.assertEqual(props["timeout_s"]["default"], 8)
        self.assertEqual(props["print_timeout_s"]["default"], 60)
        self.assertFalse(props["remote_start_enabled"]["default"])
        self.assertEqual(props["remote_start_port"]["default"], 22)
        self.assertEqual(definition["configSchema"]["required"], ["bridge_url", "token"])

    def test_print_timeout_defaults_to_60_seconds_and_is_clamped(self):
        self.assertEqual(self.card._cfg["print_timeout_s"], 60)
        self.card.dispatch("config", {"print_timeout_s": 999})
        self.assertEqual(self.card._cfg["print_timeout_s"], 120)
        self.card.dispatch("config", {"print_timeout_s": 3})
        self.assertEqual(self.card._cfg["print_timeout_s"], 10)

    def test_lifecycle_and_configuration(self):
        self.assertEqual(self.call("start")["state"], "ready")
        self.assertTrue(self.call("info")["configured"])
        self.assertEqual(self.call("stop")["state"], "idle")
        card = QLPrinterPlugin({})
        self.assertEqual(card.dispatch("find", {"name": "王磊"})["code"], "not_configured")

    def test_config_rejects_bad_url_without_losing_old_value(self):
        result = self.card.dispatch("config", {"bridge_url": "ftp://x"})
        self.assertFalse(result["adapter_ok"])
        self.assertEqual(self.call("find", name="王磊")["status"], "found")

    def test_find_single_multiple_host_and_none(self):
        one = self.call("find", name="王磊")
        self.assertEqual(one["status"], "found")
        self.assertEqual(one["visits"][0]["host"], "张三")
        self.assertNotIn("visitor_id", one["visits"][0])
        self.assertNotIn("phone", one["visits"][0])
        self.assertEqual(self.call("find", name="李娜")["status"], "multiple")
        narrowed = self.call("find", name="李娜", host="李臻")
        self.assertEqual(narrowed["status"], "found")
        self.assertEqual(narrowed["visits"][0]["tail"], "5678")
        self.assertEqual(self.call("find", name="赵六")["status"], "not_found")
        self.assertEqual(self.call("find")["code"], "bad_args")

    def test_print_uses_ticket_and_request_id(self):
        ticket = self.find_ticket()
        result = self.call("print", ticket=ticket, request_id="req-001")
        self.assertEqual(result["status"], "printed")
        self.assertEqual(result["request_id"], "req-001")
        self.assertEqual(self.bridge.print_log, ["V1001"])

    def test_find_and_print_hides_ticket_and_generates_request_id(self):
        result = self.call("print", name="王磊", host="张三")
        self.assertEqual(result["status"], "printed")
        self.assertTrue(result["request_id"].startswith("direct-"))
        self.assertEqual(self.bridge.print_log, ["V1001"])

    def test_find_and_print_rejects_ambiguous_or_completed_visit(self):
        ambiguous = self.call("print", name="李娜", host="")
        self.assertEqual(ambiguous["code"], "visitor_ambiguous")
        first = self.call("print", name="王磊", host="张三")
        self.assertEqual(first["status"], "printed")
        second = self.call("print", name="王磊", host="张三")
        self.assertEqual(second["code"], "status_not_printable")
        self.assertEqual(self.bridge.print_log, ["V1001"])

    def test_same_request_id_is_idempotent(self):
        ticket = self.find_ticket()
        first = self.call("print", ticket=ticket, request_id="same-request")
        second = self.call("print", ticket=ticket, request_id="same-request")
        self.assertEqual(first["status"], "printed")
        self.assertEqual(second["status"], "printed")
        self.assertEqual(self.bridge.print_log, ["V1001"])

    def test_second_print_with_new_request_is_rejected(self):
        ticket = self.find_ticket()
        self.call("print", ticket=ticket, request_id="req-a")
        result = self.call("print", ticket=ticket, request_id="req-b")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["code"], "already_printed")
        self.assertEqual(self.bridge.print_log, ["V1001"])

    def test_ticket_is_required_and_expires(self):
        self.assertEqual(self.call("print")["code"], "bad_args")
        ticket = self.find_ticket()
        self.assertEqual(self.call("print", ticket=ticket)["code"], "bad_args")
        self.bridge.tickets[ticket]["expires_at"] = 0
        result = self.call("print", ticket=ticket, request_id="expired")
        self.assertEqual(result["code"], "ticket_expired")

    def test_printer_failure_is_reported(self):
        ticket = self.find_ticket()
        self.bridge.printer_ready = False
        result = self.call("print", ticket=ticket, request_id="printer-down")
        self.assertEqual(result["code"], "printer_not_ready")
        self.assertIn("缺纸", result["error"])

    def test_wrong_token_and_bridge_down(self):
        self.card.dispatch("config", {"token": "wrong"})
        self.assertEqual(self.call("find", name="王磊")["code"], "unauthorized")
        self.card.dispatch("config", {"token": TOKEN, "timeout_s": 2})
        self.server.shutdown()
        self.server.server_close()
        self.server = None
        result = self.call("find", name="王磊")
        self.assertEqual(result["code"], "unreachable")

    def test_remote_start_runs_ssh_before_query_when_bridge_is_down(self):
        self.card._cfg.update({
            "remote_start_enabled": True,
            "remote_start_user": "4paradigm",
            "remote_start_port": 22,
            "remote_start_key_path": "/run/test-key",
        })
        with patch.object(self.card, "_bridge_is_healthy", side_effect=[False, True]), patch(
            "plugins.ql_printer.os.path.isfile", return_value=True
        ), patch(
            "plugins.ql_printer.subprocess.run",
            return_value=CompletedProcess([], 0, stdout="STARTED", stderr=""),
        ) as run, patch("plugins.ql_printer.time.sleep"):
            result = self.call("find", name="王磊")
        self.assertEqual(result["status"], "found")
        command = run.call_args.args[0]
        self.assertEqual(command[0], "ssh")
        self.assertIn("4paradigm@127.0.0.1", command)

    def test_unknown_action(self):
        self.assertEqual(self.call("explode")["code"], "bad_action")


if __name__ == "__main__":
    unittest.main()



