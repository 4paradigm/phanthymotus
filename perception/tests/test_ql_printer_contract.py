"""QL_printer 卡片与安全桥接协议的契约测试。"""

from __future__ import annotations

import json
import base64
import sys
import threading
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from plugins.ql_printer import BridgeClient, QLPrinterPlugin, SSHBridgeClient  # noqa: E402
from ql_printer_mock_bridge import MockBridge  # noqa: E402

TOKEN = "test-token"


class VisitorBadgeTest(unittest.TestCase):
    def setUp(self):
        self.bridge = MockBridge(TOKEN)
        self.server = self.bridge.serve("127.0.0.1", 0)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.card = QLPrinterPlugin({})
        cfg = self.card.dispatch("config", {"bridge_url": self.url, "remote_start_user": "operator",
                                              "remote_start_password": "secret",
                                              "visitor_page_url": "https://visitor.test/#/list",
                                              "printer_name": "Brother"})
        self.assertTrue(cfg["adapter_ok"])
        # Exercise the card contract against the existing mock business API;
        # SSH framing itself has dedicated tests below.
        self.card._client = lambda cfg=None: BridgeClient(self.url, TOKEN, 8)

    def tearDown(self):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()

    def call(self, action, **args):
        result = self.card.dispatch("QL_printer", dict(args, action=action))
        json.dumps(result, ensure_ascii=False)
        return result

    def find_ticket(self, name="王磊", host=""):
        _, visits = self.card._daily_visits()
        matches = [v for v in visits if v["name"] == name and (not host or host in v["host"])]
        self.assertEqual(len(matches), 1)
        return matches[0]["ticket"]

    def test_tool_definition_uses_ticket_protocol(self):
        definition = self.card.get_tool()
        self.assertEqual(definition["name"], "Visitor_Badge_Printer")
        self.assertEqual(definition["type"], "actuator")
        actions = definition["inputSchema"]["properties"]["action"]["enum"]
        self.assertEqual(set(actions), {"start", "stop", "info", "config", "list", "print", "printers", "initialize", "initialize_status"})
        schema_text = json.dumps(definition["inputSchema"], ensure_ascii=False)
        self.assertIn('"print"', schema_text)
        props = definition["configSchema"]["properties"]
        self.assertEqual(props["printer_name"]["scope"], "instance")
        self.assertEqual(props["printer_name"]["format"], "bridge-printer")
        self.assertNotIn("token", props)
        self.assertIn("remote_start_password", props)
        self.assertIn("visitor_page_url", props)
        self.assertEqual(definition["configSchema"]["required"],
                         ["bridge_url", "remote_start_user", "remote_start_password"])

    def test_operator_can_enter_only_frontdesk_ip(self):
        card = QLPrinterPlugin({"bridge_url": "10.100.10.14"})
        self.assertEqual(card._cfg["bridge_url"], "ssh://10.100.10.14")
        result = card.dispatch("config", {"bridge_url": "192.168.1.20"})
        self.assertEqual(result["status"], "ok")
        self.assertEqual(card._cfg["bridge_url"], "ssh://192.168.1.20")

    def test_address_validation_is_consistent_for_setup_and_discovery(self):
        for address in ("user:password@192.168.1.2", "http://host:bad", "host:99999", "host/path", "a b", "http://[broken"):
            for action in ("config", "initialize", "printers"):
                with self.subTest(address=address, action=action):
                    self.assertEqual(self.call(action, bridge_url=address)["status"], "error")
        for address, expected in (("192.168.1.2", "ssh://192.168.1.2"),
                                  ("::1", "ssh://[::1]"),
                                  ("https://example.com/", "ssh://example.com")):
            self.assertEqual(self.call("config", bridge_url=address)["status"], "ok")
            self.assertEqual(self.card._cfg["bridge_url"], expected)

    def test_discovery_requires_ssh_identity(self):
        self.assertEqual(self.call("printers", bridge_url="192.168.1.20", remote_start_user="")["code"], "not_configured")

    def test_initialization_result_is_bound_to_its_job(self):
        from plugins import bridge_deploy
        finished = threading.Event()
        release = threading.Event()
        def deploy(config):
            self.assertEqual(config["bridge_url"], "ssh://192.168.1.20")
            release.wait(3)
            finished.set()
            return {"status": "installed", "bridge_url": config["bridge_url"]}
        with patch.object(bridge_deploy, "deploy", side_effect=deploy):
            started = self.call("initialize", bridge_url="192.168.1.20")
            try:
                self.assertEqual(started["status"], "installing")
                self.assertEqual(self.call("initialize", bridge_url="192.168.1.21")["code"], "installing")
                self.assertEqual(self.call("initialize_status", deployment_id="other")["code"], "unknown_deployment")
            finally:
                release.set()
            self.assertTrue(finished.wait(3))
            for _ in range(100):
                state = self.call("initialize_status", deployment_id=started["deployment_id"])
                if state["status"] != "installing":
                    break
                threading.Event().wait(0.01)
            self.assertEqual(state["status"], "installed")
            self.assertNotIn("token", self.call("initialize_status"))
            self.assertNotIn("deployment_id", self.call("initialize_status"))
            self.assertEqual(self.card._cfg["bridge_url"], "ssh://127.0.0.1")  # Save remains explicit.

    def test_print_timeout_covers_on_demand_browser_job_and_is_clamped(self):
        self.assertEqual(self.card._cfg["print_timeout_s"], 120)
        self.card.dispatch("config", {"print_timeout_s": 999})
        self.assertEqual(self.card._cfg["print_timeout_s"], 120)
        self.card.dispatch("config", {"print_timeout_s": 3})
        self.assertEqual(self.card._cfg["print_timeout_s"], 10)

    def test_lifecycle_and_configuration(self):
        self.assertEqual(self.call("start")["state"], "ready")
        self.assertTrue(self.call("info")["configured"])
        self.assertEqual(self.call("stop")["state"], "idle")
        card = QLPrinterPlugin({})
        self.assertEqual(card.dispatch("list", {})["code"], "not_configured")

    def test_config_rejects_bad_url_without_losing_old_value(self):
        result = self.card.dispatch("config", {"bridge_url": "ftp://x"})
        self.assertFalse(result["adapter_ok"])
        self.assertEqual(self.call("list")["status"], "listed")

    def test_list_returns_all_unadmitted_visitors_without_tickets(self):
        result = self.call("list")
        self.assertEqual(result["status"], "listed")
        self.assertEqual(result["date"], "2026-11-17")
        self.assertEqual(result["count"], 3)
        self.assertEqual({v["name"] for v in result["visits"]}, {"王磊", "李娜"})
        self.assertTrue(all(v["status"] == "未入场" for v in result["visits"]))
        self.assertTrue(all("ticket" not in v for v in result["visits"]))
        self.assertTrue(all("visitor_id" not in v and "phone" not in v for v in result["visits"]))

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
        self.assertEqual(second["status"], "not_found")
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

    def test_business_endpoint_down_is_reported(self):
        self.server.shutdown()
        self.server.server_close()
        self.server = None
        result = self.call("list")
        # Closed ports can be rejected immediately or time out depending on
        # the host network stack. Neither outcome proves a login failure.
        self.assertIn(result["code"], {"unreachable", "request_timeout"})

    def test_ssh_is_the_only_remote_transport(self):
        client = SSHBridgeClient("10.100.10.14", "4paradigm", "/run/test-key")
        payload = base64.b64encode(json.dumps({"state": "running"}).encode()).decode()
        completed = CompletedProcess([], 0, stdout=("VISITOR_RESULT:" + payload + "\n").encode(), stderr=b"")
        with patch("plugins.ql_printer.os.path.isfile", return_value=True), patch(
            "plugins.ql_printer.subprocess.run", return_value=completed
        ) as run:
            self.assertEqual(client.request("GET", "/health")["state"], "running")
        command = run.call_args.args[0]
        self.assertEqual(command[0], "ssh")
        self.assertIn("4paradigm@10.100.10.14", command)
        self.assertNotIn("curl", command)

    def test_browser_job_waits_for_login_edge_to_close_cleanly(self):
        client = SSHBridgeClient("10.100.10.14", "4paradigm", "secret")
        payload = base64.b64encode(json.dumps({"state": "listed"}).encode()).decode()
        completed = CompletedProcess([], 0, stdout=("VISITOR_RESULT:" + payload + "\n").encode(), stderr=b"")
        with patch("plugins.ql_printer.subprocess.run", return_value=completed) as run:
            client.request("POST", "/list_visits", body={"visitor_page_url": "https://visitor.test/list"})
        command = run.call_args.args[0]
        script = base64.b64decode(command[-1].rsplit(" ", 1)[-1]).decode("utf-16le")
        self.assertIn("Paradigm Visitor Login", script)
        self.assertIn("login.stop", script)
        self.assertIn("Set-Content", script)
        self.assertIn("Start-Sleep -Milliseconds 250", script)
        self.assertIn("($i % 20) -eq 0", script)
        self.assertIn("$currentTask.State -ne 'Running'", script)
        self.assertGreaterEqual(script.count("Start-ScheduledTask -TaskName 'Paradigm Visitor Automation'"), 2)
        self.assertNotIn("Stop-ScheduledTask -TaskName 'Paradigm Visitor Login'", script)

    def test_unknown_action(self):
        self.assertEqual(self.call("explode")["code"], "bad_action")


if __name__ == "__main__":
    unittest.main()



