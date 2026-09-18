"""Tests for Deploy Approval command parsing."""
from __future__ import annotations

import pytest
from ..commands import parse_command, command_starts_line_any


class TestRequestDeploy:
    def test_parameterless(self):
        cmd = parse_command("/request_deploy")
        assert cmd.kind == "request_deploy"
        assert cmd.is_command

    def test_rejects_build_parameter(self):
        cmd = parse_command("/request_deploy build=0")
        assert cmd.kind == "unknown"
        assert not cmd.is_command

    def test_rejects_machine_parameter(self):
        cmd = parse_command("/request_deploy machine=x")
        assert cmd.kind == "unknown"

    def test_rejects_test_mode_parameter(self):
        cmd = parse_command("/request_deploy test-mode=plan")
        assert cmd.kind == "unknown"

    def test_rejects_test_plan_parameter(self):
        cmd = parse_command("/request_deploy test-plan=...")
        assert cmd.kind == "unknown"

    def test_rejects_test_case_parameter(self):
        cmd = parse_command("/request_deploy test-case=...")
        assert cmd.kind == "unknown"

    def test_rejects_positional(self):
        cmd = parse_command("/request_deploy 1")
        assert cmd.kind == "unknown"


class TestApproveDeploy:
    def test_requires_machine(self):
        cmd = parse_command("/approve_deploy machine=g1-bj-wifi")
        assert cmd.kind == "approve_deploy"
        assert cmd.is_command
        assert cmd.machine_alias == "g1-bj-wifi"

    def test_rejects_without_machine(self):
        cmd = parse_command("/approve_deploy")
        assert cmd.kind == "unknown"

    def test_rejects_positional(self):
        cmd = parse_command("/approve_deploy dpl_x machine=x")
        assert cmd.kind == "unknown"


class TestRecordTest:
    def test_result_pass(self):
        cmd = parse_command("/record_test result=pass")
        assert cmd.kind == "record_test"
        assert cmd.is_command
        assert cmd.result == "pass"

    def test_result_fail(self):
        cmd = parse_command("/record_test result=fail")
        assert cmd.kind == "record_test"
        assert cmd.result == "fail"

    def test_result_fail_with_summary(self):
        cmd = parse_command('/record_test result=fail summary="test failed"')
        assert cmd.kind == "record_test"
        assert cmd.result == "fail"
        assert cmd.summary == "test failed"

    def test_rejects_machine_parameter(self):
        cmd = parse_command("/record_test machine=x result=pass")
        assert cmd.kind == "unknown"

    def test_rejects_positional(self):
        cmd = parse_command("/record_test dpl_x result=pass")
        assert cmd.kind == "unknown"

    def test_rejects_evidence_parameter(self):
        cmd = parse_command("/record_test result=pass evidence=...")
        assert cmd.kind == "unknown"

    def test_invalid_result(self):
        cmd = parse_command("/record_test result=maybe")
        assert cmd.kind == "unknown"


class TestDeployStatus:
    def test_parameterless(self):
        cmd = parse_command("/deploy_status")
        assert cmd.kind == "deploy_status"
        assert cmd.is_command

    def test_rejects_positional(self):
        cmd = parse_command("/deploy_status dpl_01abc")
        assert cmd.kind == "unknown"


class TestDeployHelp:
    def test_parameterless(self):
        cmd = parse_command("/deploy_help")
        assert cmd.kind == "deploy_help"
        assert cmd.is_command

    def test_with_topic(self):
        cmd = parse_command("/deploy_help request_deploy")
        assert cmd.kind == "deploy_help"
        assert cmd.help_topic == "request_deploy"


class TestUnknownCommands:
    def test_legacy_reject(self):
        assert parse_command("/reject_deploy").kind == "unknown"

    def test_legacy_rollback(self):
        assert parse_command("/rollback_deploy").kind == "unknown"

    def test_legacy_cancel(self):
        assert parse_command("/cancel_deploy").kind == "unknown"

    def test_legacy_resume(self):
        assert parse_command("/resume_deploy").kind == "unknown"

    def test_empty(self):
        cmd = parse_command("")
        assert cmd.kind == "unknown"
        assert not cmd.is_command

    def test_not_a_command(self):
        assert not command_starts_line_any("not a command")

    def test_unbalanced_quote(self):
        cmd = parse_command('/approve_deploy machine="unclosed')
        assert cmd.kind == "unknown"


class TestCommandStartsLine:
    def test_command_starts_line(self):
        assert command_starts_line_any("/request_deploy")
        assert command_starts_line_any("/approve_deploy machine=g1")
        assert command_starts_line_any("/record_test result=pass")
        assert command_starts_line_any("/deploy_status")
        assert command_starts_line_any("/deploy_help")
        assert not command_starts_line_any("")
        assert not command_starts_line_any("not a command")


class TestIterTopLevelCommandLines:
    def test_normal_command(self):
        from ..commands import _iter_top_level_command_lines
        lines = _iter_top_level_command_lines("/request_deploy")
        assert lines == ["/request_deploy"]

    def test_backtick_fence(self):
        from ..commands import _iter_top_level_command_lines
        text = '```python\n/request_deploy\n```\n/request_deploy'
        lines = _iter_top_level_command_lines(text)
        assert lines == ["/request_deploy"]

    def test_tilde_fence(self):
        from ..commands import _iter_top_level_command_lines
        text = '~~~\n/request_deploy\n~~~\n/request_deploy'
        lines = _iter_top_level_command_lines(text)
        assert lines == ["/request_deploy"]

    def test_mixed_fence_regression(self):
        from ..commands import _iter_top_level_command_lines
        # Backtick fence opened, then tilde fence appears — tilde must NOT close backtick
        text = '```\n/request_deploy\n~~~\n/request_deploy\n```\n/request_deploy'
        lines = _iter_top_level_command_lines(text)
        assert lines == ["/request_deploy"]

    def test_multiline_html_comment(self):
        from ..commands import _iter_top_level_command_lines
        text = '<!--\n/request_deploy\n-->\n/request_deploy'
        lines = _iter_top_level_command_lines(text)
        assert lines == ["/request_deploy"]

    def test_single_line_html_comment_with_command(self):
        from ..commands import _iter_top_level_command_lines
        text = '<!-- /request_deploy -->\n/request_deploy'
        lines = _iter_top_level_command_lines(text)
        assert lines == ["/request_deploy"]

    def test_indented_fence_cannot_execute_approve(self):
        from ..commands import _iter_top_level_command_lines
        # Indented fence (3 spaces) — fence detected but command inside is NOT column-0
        text = '   ```text\n/approve_deploy machine=test\n```\n/request_deploy'
        lines = _iter_top_level_command_lines(text)
        assert lines == ["/request_deploy"]

    def test_indented_html_comment_cannot_execute_approve(self):
        from ..commands import _iter_top_level_command_lines
        # Indented HTML comment (3 spaces) — command inside must not be yielded
        text = '   <!--\n/approve_deploy machine=test\n-->\n/request_deploy'
        lines = _iter_top_level_command_lines(text)
        assert lines == ["/request_deploy"]

    def test_fence_with_trailing_text_does_not_close_and_execute(self):
        from ..commands import _iter_top_level_command_lines
        # Closing fence with trailing text (not space/tab/EOL) should NOT close fence
        text = '```\n/approve_deploy machine=test\n```not-a-close\n/request_deploy\n```\n/request_deploy'
        lines = _iter_top_level_command_lines(text)
        assert lines == ["/request_deploy"]

    def test_html_comment_nested_opener_does_not_toggle(self):
        from ..commands import _iter_top_level_command_lines
        # Nested <!-- inside an open HTML comment must NOT toggle state back to False
        text = '<!--\n/request_deploy\n<!-- nested comment -->\n/request_deploy'
        lines = _iter_top_level_command_lines(text)
        assert lines == []  # Both commands hidden — '<!-- ... -->' has close>open, doesn't close multiline

    def test_html_comment_nested_opener_stays_inside_comment(self):
        from ..commands import _iter_top_level_command_lines
        # Multiple nested <!-- should all be treated as content
        text = '<!-- start\n<!-- middle\n<!-- deep\n-->\n/request_deploy'
        lines = _iter_top_level_command_lines(text)
        assert lines == ["/request_deploy"]

    def test_html_comment_nested_opener_keeps_command_hidden(self):
        from ..commands import _iter_top_level_command_lines
        # Command after nested <!-- that doesn't close should remain hidden
        text = '<!--\n<!-- no close yet\n/request_deploy'
        lines = _iter_top_level_command_lines(text)
        assert lines == []
