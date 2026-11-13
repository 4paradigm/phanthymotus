"""Chromium acceptance against real loopback MCP and upstream Workbench host.

All RGB data are explicitly synthetic. Does not import rclpy or access hardware.
Run with a Python containing playwright, numpy, cv2 and jsonschema.
"""

import argparse
import json
import re
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "perception"))
sys.path.insert(0, str(ROOT / "perception/tests"))
from plugins.tabletop_roi_goal_verifier.contracts import Refusal
from plugins.tabletop_roi_goal_verifier.plugin import TabletopROIGoalVerifierPlugin
from plugins.tabletop_roi_goal_verifier.server import handler
from test_tabletop_roi_goal_verifier import configuration, picture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chromium", type=str)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    card = TabletopROIGoalVerifierPlugin(configuration())
    current = [picture()]
    halt = threading.Event()

    def feed():
        while not halt.wait(0.03):
            try:
                card.sources.feed_image(
                    "camera", current[0], time.time(), provenance="synthetic"
                )
            except Refusal as exc:
                if exc.code != "SOURCE_CLOSED":
                    raise

    class Host(handler(card)):
        def do_GET(self):
            if self.path == "/core-test":
                data = b"""<script type="module">
                import {openWorkbench} from '/js/workbench.js';
                const rpc=async(action)=>{const r=await fetch('/mcp',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({jsonrpc:'2.0',id:1,method:'tools/call',params:{name:'tabletop_roi_goal_verifier',arguments:{action}}})});return JSON.parse((await r.json()).result.content[0].text)};
                const r=await fetch('/mcp',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({jsonrpc:'2.0',id:1,method:'tools/list'})});
                const tool=(await r.json()).result.tools[0];await openWorkbench({mcp:'test',tool:tool.name},tool.inputSchema['x-workbench'],await rpc('ui'));
                </script>"""
            elif self.path == "/js/workbench.js":
                data = (ROOT / "agent-core/web/js/workbench.js").read_bytes()
            else:
                return super().do_GET()
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "text/javascript" if self.path.endswith(".js") else "text/html",
            )
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if self.path.startswith("/api/mcp/test/call"):
                raw = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self.send(
                    {
                        "code": 200,
                        "data": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    card.dispatch(raw["tool"], raw["arguments"])
                                ),
                            }
                        ],
                    }
                )
            else:
                super().do_POST()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Host)
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    feeding = threading.Thread(target=feed, daemon=True)
    feeding.start()
    report = []
    try:
        from playwright.sync_api import expect, sync_playwright

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, executable_path=args.chromium)
            page = browser.new_page(viewport={"width": 1250, "height": 1100})
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            for host in ("standalone", "upstream-workbench"):
                page.goto(
                    f"http://127.0.0.1:{server.server_port}/"
                    + ("" if host == "standalone" else "core-test")
                )
                frame = page if host == "standalone" else page.frame_locator("iframe")
                frame.locator("#source option").wait_for(state="attached")
                frame.locator("#start").click()
                frame.locator("#live").click()
                frame.locator("#measurement").filter(has_text="synthetic").wait_for()
                for expected in ("FAIL", "PASS", "INCONCLUSIVE"):
                    current[0] = picture()
                    time.sleep(0.1)
                    # Real pointer drag; Core embeds the unchanged upstream host.
                    canvas = frame.locator("#view")
                    canvas.scroll_into_view_if_needed()
                    box = canvas.bounding_box()
                    page.mouse.move(
                        box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.25
                    )
                    page.mouse.down()
                    page.mouse.move(
                        box["x"] + box["width"] * 0.9,
                        box["y"] + box["height"] * 0.8,
                        steps=10,
                    )
                    page.mouse.up()
                    frame.locator("#roi-preset").select_option(
                        "strict" if expected == "PASS" else "standard"
                    )
                    frame.locator("#session").click()
                    frame.locator("#state").filter(has_text="session_token").wait_for()
                    frame.locator("#before").click()
                    frame.locator("#state").filter(
                        has_text="before_captured"
                    ).wait_for()
                    if expected == "PASS":
                        current[0] = picture(True)
                    if expected == "INCONCLUSIVE":
                        image = picture()
                        image[60:192, 160:288] = 40
                        current[0] = image
                    time.sleep(0.1)
                    frame.locator("#mark").click()
                    frame.locator("#state").filter(has_text="marker_id").wait_for()
                    frame.locator("#after").click()
                    try:
                        frame.locator(f"#result.{expected}").wait_for(timeout=10000)
                    except Exception:
                        print(
                            host,
                            expected,
                            frame.locator("#state").inner_text(),
                            frame.locator("#logs").inner_text(),
                            flush=True,
                        )
                        page.screenshot(
                            path=str(args.output / "failure.png"), full_page=True
                        )
                        raise
                    result = json.loads(frame.locator("#state").inner_text())
                    assert result["result"] == expected, result
                    assert frame.locator("#roi-before").get_attribute("src")
                    assert frame.locator("#roi-after").get_attribute("src")
                    frame.locator("#export").click()
                    expect(frame.locator("#export-data")).to_have_value(
                        re.compile('"result": "' + expected + '"')
                    )
                    evidence = json.loads(frame.locator("#export-data").input_value())
                    assert evidence["result"]["result"] == expected
                    assert "session_token" not in evidence
                    (args.output / f"{host}-{expected}.json").write_text(
                        json.dumps(evidence, indent=2)
                    )
                    page.screenshot(
                        path=str(args.output / f"{host}-{expected}.png"), full_page=True
                    )
                    report.append(
                        {
                            "host": host,
                            "result": expected,
                            "code": result["code"],
                            "provenance": result["provenance"],
                        }
                    )
                frame.locator("#stop").click()
                frame.locator("#state").filter(has_text="stopped").wait_for()
            assert not errors, errors
            browser.close()
        (args.output / "browser.json").write_text(
            json.dumps(
                {"scenarios": report, "page_errors": errors, "hardware_tested": False},
                indent=2,
            )
        )
        print(json.dumps(report))
    finally:
        halt.set()
        feeding.join(3)
        server.shutdown()
        serving.join(3)
        server.server_close()
        card.stop()


if __name__ == "__main__":
    main()
