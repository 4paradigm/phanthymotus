"""Fast browser capture and evidence-based failure reporting without a live portal."""
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "windows_bridge"))
from bridge_core import BridgeFailure, Visit
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
    buttons = page.get_by_role.return_value
    buttons.count.return_value = 1
    button = buttons.nth.return_value
    button.is_visible.return_value = True
    request = SimpleNamespace(url="https://visitor.test/visitInfo/page/query", method="POST",
                              all_headers=lambda: {"token": "secret", "cookie": "private"})
    outcomes = [request]
    def auto_post(*args, **kwargs):
        page.on.call_args.args[1](request)
        return SimpleNamespace(status=200)
    page.goto.side_effect = auto_post
    page.reload.side_effect = auto_post

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
    page.expect_request.assert_not_called()
    page.remove_listener.assert_called_once()


def test_search_fallback_has_short_explicit_deadlines(browser):
    backend, page, button, request, outcomes = browser
    page.reload.side_effect = None
    outcomes[:] = [request]
    assert backend._ensure_login(page) == {"token": "secret"}
    assert [call.kwargs["timeout"] for call in page.expect_request.call_args_list] == [2000]
    button.click.assert_called_once_with(timeout=1000)


@pytest.mark.parametrize("failure", ["load", "render", "http", "login", "missing", "hidden", "method", "click", "request", "closed", "operation"])
def test_failure_codes_describe_observed_stage(browser, failure):
    backend, page, button, _, outcomes = browser
    expected = {
        "load": "page_load_timeout", "render": "page_render_timeout", "http": "page_http_error", "login": "login_required",
        "missing": "query_button_missing", "method": "query_method_mismatch",
        "hidden": "query_button_unavailable",
        "click": "query_click_timeout", "request": "query_request_timeout",
        "closed": "browser_closed", "operation": "page_load_failed",
    }[failure]
    page.reload.side_effect = None
    if failure == "load":
        page.reload.side_effect = BrowserTimeout()
    elif failure == "render":
        page.wait_for_function.side_effect = BrowserTimeout()
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
        outcomes[:] = [BrowserTimeout()]
        if failure in {"missing", "method"}:
            page.get_by_role.return_value.count.return_value = 0
        if failure == "hidden":
            button.is_visible.return_value = False
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
        assert str(caught.value) == "当前页面未找到访客查询按钮，请确认已打开正确的访客列表网址。"
        page.expect_request.assert_not_called()
        button.click.assert_not_called()
        page.wait_for_timeout.assert_not_called()
    if failure in {"load", "render", "http", "login", "hidden", "method", "closed", "operation"}:
        page.expect_request.assert_not_called()
    assert "sensitive" not in str(caught.value)
    if failure == "http":
        assert "503" in str(caught.value)
    page.remove_listener.assert_called_once()


def test_unrelated_post_is_not_proof_of_endpoint_mismatch(browser):
    backend, page, _, _, outcomes = browser
    outcomes[:] = [BrowserTimeout()]
    def other_request(**kwargs):
        page.on.call_args.args[1](SimpleNamespace(url="https://visitor.test/analytics", method="POST"))
        return SimpleNamespace(status=200)
    page.reload.side_effect = other_request
    with pytest.raises(BridgeFailure) as caught:
        backend._ensure_login(page)
    assert caught.value.code == "query_request_timeout"
    assert str(caught.value) == "查询操作超时（2 秒内未检测到访客查询请求），请稍后重试。"


def test_hidden_first_button_does_not_hide_visible_search_button(browser):
    backend, page, button, _, _ = browser
    page.reload.side_effect = None
    hidden = MagicMock()
    hidden.is_visible.return_value = False
    buttons = page.get_by_role.return_value
    buttons.count.return_value = 2
    buttons.nth.side_effect = lambda index: [hidden, button][index]
    assert backend._ensure_login(page) == {"token": "secret"}
    hidden.click.assert_not_called()
    button.click.assert_called_once_with(timeout=1000)


def test_query_arriving_while_form_is_loading_does_not_need_a_click(browser):
    backend, page, button, request, _ = browser
    page.reload.side_effect = None
    page.wait_for_function.side_effect = lambda *args, **kwargs: page.on.call_args.args[1](request)
    assert backend._ensure_login(page) == {"token": "secret"}
    button.click.assert_not_called()
    page.expect_request.assert_not_called()


def test_cleanup_failure_does_not_replace_browser_closed_error(browser):
    backend, page, _, _, _ = browser
    page.reload.side_effect = RuntimeError("Target page, context or browser has been closed")
    page.is_closed.return_value = True
    page.remove_listener.side_effect = RuntimeError("emitter closed")
    with pytest.raises(BridgeFailure) as caught:
        backend._ensure_login(page)
    assert caught.value.code == "browser_closed"


@pytest.mark.parametrize("count, code, message", [
    (0, "booking_row_missing", "当前网页列表中未找到对应记录"),
    (2, "booking_row_ambiguous", "多条相同预约编号"),
])
def test_booking_row_failures_distinguish_missing_and_multiple_rows(browser, count, code, message):
    backend, page, _, _, _ = browser
    rows = page.locator.return_value
    rows.count.return_value = count
    rows.nth.return_value.inner_text.return_value = "BOOK-123"
    rows.nth.return_value.get_by_text.return_value.count.return_value = 1
    visit = Visit(visit_id="V1", name="王磊", booking_code="BOOK-123")
    with pytest.raises(BridgeFailure) as caught:
        backend._open_detail_if_needed(page, visit)
    assert caught.value.code == code
    assert message in str(caught.value)
    rows.nth.return_value.get_by_text.return_value.click.assert_not_called()
