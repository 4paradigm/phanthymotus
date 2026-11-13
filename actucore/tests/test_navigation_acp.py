import json
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plugins.controlled_semantic_spatial import acp


@pytest.mark.parametrize("nav_status,expected", [
    ("arrived", "completed"), ("succeeded", "completed"),
    ("stopped", "cancelled"), ("cancelled", "cancelled"),
    ("timeout", "failed"), ("error", "failed"),
    ("aborted", "failed"), ("rejected", "failed"),
])
def test_completion_is_posted_off_callback_thread(nav_status, expected):
    result = {"nav_id": "nav-1", "status": nav_status, "reason": "detail"}
    with mock.patch.object(acp.threading, "Thread") as thread, \
         mock.patch.object(acp.urllib.request, "urlopen") as post:
        acp.complete_navigation({"action_id": "nav-1", "status": nav_status,
                                 "payload": result})
    post.assert_not_called()
    thread.return_value.start.assert_called_once_with()
    assert thread.call_args.kwargs["daemon"] is True
    assert thread.call_args.kwargs["target"] is acp._post_acp
    body = thread.call_args.kwargs["args"][0]
    assert body["action_id"] == "nav-1"
    assert body["status"] == expected
    assert body["result"] == result
    assert body["tool"] == "controlled_semantic_spatial"


def test_no_post_for_missing_identity_or_nonterminal():
    with mock.patch.object(acp.threading, "Thread") as thread:
        for event in ({}, {"action_id": "", "status": "arrived"},
                      {"action_id": "nav-1", "status": "paused"}):
            acp.complete_navigation(event)
    thread.assert_not_called()


@pytest.mark.parametrize("url", ["https://localhost:15678", "https://core.example/"])
def test_http_contract_and_timeout(url):
    body = {"action_id": "nav-1", "status": "completed", "result": {"status": "arrived"}}
    with mock.patch.dict(acp.os.environ, {"AGENT_CORE_URL": url}), \
         mock.patch.object(acp.urllib.request, "urlopen") as post:
        post.return_value.__enter__.return_value.read.return_value = b'{"ok":true}'
        acp._post_acp(body)
    request = post.call_args.args[0]
    assert request.full_url == url.rstrip("/") + "/api/acp/complete"
    assert request.get_method() == "POST"
    assert json.loads(request.data) == body
    assert post.call_args.kwargs["timeout"] == 10


@pytest.mark.parametrize("failure", [OSError("offline"), b'{"ok":false}', b'not json'])
def test_failed_callback_is_logged_not_raised(failure):
    with mock.patch.object(acp.urllib.request, "urlopen") as post, \
         mock.patch.object(acp.log, "warning") as warning:
        if isinstance(failure, Exception):
            post.side_effect = failure
        else:
            post.return_value.__enter__.return_value.read.return_value = failure
        acp._post_acp({"action_id": "nav-1", "status": "failed"})
    warning.assert_called_once()
