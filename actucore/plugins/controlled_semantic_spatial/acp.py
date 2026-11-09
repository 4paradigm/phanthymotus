"""Report navigation completion using the same HTTP ACP contract as navi."""

import json
import logging
import os
import ssl
import threading
import time
import urllib.request

log = logging.getLogger(__name__)


def complete_navigation(event: dict) -> None:
    action_id = event.get("action_id")
    if not isinstance(action_id, str) or not action_id:
        return
    navigation_status = event.get("status")
    if navigation_status in {"arrived", "succeeded"}:
        status = "completed"
    elif navigation_status in {"stopped", "cancelled"}:
        status = "cancelled"
    elif navigation_status in {"timeout", "error", "aborted", "rejected"}:
        status = "failed"
    else:
        return
    body = {
        "action_id": action_id,
        "status": status,
        "result": dict(event.get("payload") or {}),
        "tool": "controlled_semantic_spatial",
        "ts": time.time(),
    }
    # Never block the ROS status callback on Core/network I/O.
    threading.Thread(target=_post_acp, args=(body,), daemon=True,
                     name="navigation_acp").start()


def _post_acp(body: dict) -> None:
    try:
        url = os.environ.get("AGENT_CORE_URL", "https://localhost:15678")
        # Match navi's existing local Core/self-signed TLS deployment contract.
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        request = urllib.request.Request(
            f"{url.rstrip('/')}/api/acp/complete",
            data=json.dumps(body, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=10, context=context) as response:
            acknowledgement = json.loads(response.read())
        if acknowledgement.get("ok") is not True:
            raise ValueError("Core did not acknowledge ACP completion")
        log.info("navigation ACP %s: %s", body["action_id"], body["status"])
    except Exception as error:
        log.warning("navigation ACP callback failed for %s: %s", body["action_id"], error)
