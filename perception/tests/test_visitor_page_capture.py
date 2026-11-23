"""Fast browser capture and evidence-based failure reporting without a live portal."""
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "windows_bridge"))
from bridge_core import BridgeFailure
from edge_backend import EdgeVisitorBackend


class BrowserTimeout(Exception):
    pass


@pytest.fixture
def browser(monkeypatch):
    monkeypatch.setitem(sys.modules, "playwright", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "playwright.sync_api", SimpleNamespace(TimeoutError=BrowserTimeout))
    backend = EdgeVisitorBackend.__new__(EdgeVisitorBackend)
    backend.list_url = "https://visitor.test/list"
    backend.query_marker = "/visitInfo/page/query"
    page = MagicMock()
    page.url = backend.list_url
    page.is_closed.return_value = False
    page.reload.return_value = SimpleNamespace(status=200)
    page.goto.return_value = SimpleNamespace(status=200)
    button = page.get_by_role.return_value.first
    button.count.return_value = 1
    button.is_visible.return_value = True
    request = SimpleNamespace(all_headers=lambda: {"token": "secret", "cookie": "private"})
    outcomes = [request]

    @contextmanager
    def expect(*args, **kwargs):
        outcome = outcomes.pop(0)
        yield SimpleNamespace(value=outcome)
        if isinstance(outcome, Exception):
            raise outcome

    page.expect_request.side_effect = expect
    return backend, page, button, request, outcomes


def test_first_navigation_captures_query_without_reload_or_sleep(browser):
    backend, page, _, _, _ = browser
    page.url = "about:blank"
    assert backend._ensure_login(page) == {"token": "secret"}
    page.goto.assert_called_once_with(backend.list_url, wait_until="domcontentloaded", timeout=5000)
    page.reload.assert_not_called()
    page.wait_for_timeout.assert_not_called()
    page.remove_listener.assert_called_once()


def test_search_fallback_has_short_explicit_deadlines(browser):
    backend, page, button, request, outcomes = browser
    outcomes[:] = [BrowserTimeout(), request]
    assert backend._ensure_login(page) == {"token": "secret"}
    assert [call.kwargs["timeout"] for call in page.expect_request.call_args_list] == [3000, 2000]
    button.click.assert_called_once_with(timeout=1000)


@pytest.mark.parametrize("failure", ["load", "http", "login", "missing", "method", "click", "request", "closed", "operation"])
def test_failure_codes_describe_observed_stage(browser, failure):
    backend, page, button, _, outcomes = browser
    expected = {
        "load": "page_load_timeout", "http": "page_http_error", "login": "login_required",
        "missing": "query_button_missing", "method": "query_method_mismatch",
        "click": "query_click_timeout", "request": "query_request_timeout",
        "closed": "browser_closed", "operation": "browser_operation_failed",
    }[failure]
    if failure == "load":
        page.reload.side_effect = BrowserTimeout()
    elif failure == "http":
        page.reload.return_value = SimpleNamespace(status=503)
    elif failure == "login":
        def redirect(**kwargs):
            page.url = "https://visitor.test/login"
        page.reload.side_effect = redirect
    elif failure in {"closed", "operation"}:
        page.is_closed.return_value = failure == "closed"
        page.reload.side_effect = RuntimeError("sensitive browser details")
    else:
        outcomes[:] = [BrowserTimeout(), BrowserTimeout()]
        if failure in {"missing", "method"}:
            button.count.return_value = 0
        if failure == "method":
            def observe(**kwargs):
                listener = page.on.call_args.args[1]
                listener(SimpleNamespace(url="https://visitor.test/visitInfo/page/query", method="GET"))
                return SimpleNamespace(status=200)
            page.reload.side_effect = observe
        if failure == "click":
            button.click.side_effect = BrowserTimeout()
    with pytest.raises(BridgeFailure) as caught:
        backend._ensure_login(page)
    assert caught.value.code == expected
    if failure == "missing":
        assert str(caught.value) == "未检测到访客查询功能，请打开正确的访客列表网址"
    assert "sensitive" not in str(caught.value)
    if failure == "http":
        assert "503" in str(caught.value)
    page.remove_listener.assert_called_once()


def test_unrelated_post_is_not_proof_of_endpoint_mismatch(browser):
    backend, page, _, _, outcomes = browser
    outcomes[:] = [BrowserTimeout(), BrowserTimeout()]
    def other_request(**kwargs):
        page.on.call_args.args[1](SimpleNamespace(url="https://visitor.test/analytics", method="POST"))
        return SimpleNamespace(status=200)
    page.reload.side_effect = other_request
    with pytest.raises(BridgeFailure) as caught:
        backend._ensure_login(page)
    assert caught.value.code == "query_request_timeout"
    assert str(caught.value) == "未检测到访客查询请求，请确认已打开正确的访客列表网址；若网址正确，请稍后重试。"
