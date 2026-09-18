"""Tests for `/request_bot_review` argument parsing.

The bug these exist for: the help message advertises `jp511` / `jp61`, and its
own combining example is `/request_bot_review force perception jp61`. The token
pattern required a dot, so that exact line parsed to no JetPack variant, fell
back to the 5.11 default, and built the wrong image — successfully, with nothing
on the PR or in the log to say so. So the spellings the help promises are
tested here against the help text itself.

Run from the repo root:

    python -m pytest agents/pr_review/tests -q
"""

import re

import pytest

from agents.pr_review.comments import format_help_message
from agents.pr_review.models import (
    SUPPORTED_JP_VERSIONS,
    parse_trigger_command,
)


def parse(line: str) -> dict:
    result = parse_trigger_command(line)
    assert result is not None, f"trigger not recognised in {line!r}"
    return result


# ── JetPack version spellings ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "token,expected",
    [
        ("jp61", "6.1"),
        ("jp6.1", "6.1"),
        ("jp511", "5.11"),
        ("jp5.11", "5.11"),
        ("6.1", "6.1"),
        ("5.11", "5.11"),
        ("jetson-jp6.1", "6.1"),
        ("jetson-5.11", "5.11"),
        ("JP61", "6.1"),
    ],
)
def test_every_accepted_spelling_resolves_to_one_version(token, expected):
    assert parse(f"/request_bot_review {token}")["perception_variants"] == [expected]


def test_a_version_on_its_own_asks_for_perception():
    assert parse("/request_bot_review jp61")["force_targets"] == ["perception"]


def test_the_helps_combining_example_builds_jp61():
    result = parse("/request_bot_review force perception jp61")
    assert result["force"] is True
    assert result["force_targets"] == ["perception"]
    assert result["perception_variants"] == ["6.1"]


@pytest.mark.parametrize(
    "line",
    [
        "/request_bot_review actucore jp61",
        "/request_bot_review jp61 actucore",  # order must not decide this
    ],
)
def test_a_version_next_to_actucore_does_not_also_build_perception(line):
    # It used to: the implication fired inside the arg loop, so asking for
    # actucore at jp6.1 built a perception jp6.1 image nobody asked for.
    result = parse(line)
    assert result["force_targets"] == ["actucore"]
    assert result["perception_variants"] == ["6.1"]


def test_a_version_alongside_core_still_means_perception():
    # core has no JetPack variant, so the version has to be for something else.
    result = parse("/request_bot_review core jp61")
    assert set(result["force_targets"]) == {"core", "perception"}


def test_naming_perception_and_actucore_builds_both_at_the_asked_version():
    result = parse("/request_bot_review perception actucore jp61")
    assert result["force_targets"] == ["perception", "actucore"]
    assert result["perception_variants"] == ["6.1"]


def test_both_versions_can_be_asked_for_at_once():
    result = parse("/request_bot_review jp511 jp61")
    assert result["perception_variants"] == ["5.11", "6.1"]
    assert result["force_targets"] == ["perception"]


def test_a_repeated_version_is_requested_once():
    assert parse("/request_bot_review jp61 jp6.1")["perception_variants"] == ["6.1"]


@pytest.mark.parametrize("token", ["jp7.0", "jp70", "jp4", "jetson-9.9"])
def test_an_unsupported_version_is_dropped_rather_than_passed_to_the_build(token):
    # The build scripts exit 1 on a version they do not know, which would cost
    # the PR its whole job.
    assert parse(f"/request_bot_review {token}")["perception_variants"] == []


def test_an_unsupported_version_is_logged(caplog):
    with caplog.at_level("WARNING"):
        parse("/request_bot_review jp7.0")
    assert "jp7.0" in caplog.text


def test_an_unrecognised_argument_is_logged(caplog):
    with caplog.at_level("WARNING"):
        parse("/request_bot_review jpp61")
    assert "jpp61" in caplog.text


# ── The help message and the parser must agree ───────────────────────────────


def test_every_version_token_the_help_advertises_actually_parses():
    """The regression that started this: the help taught a spelling the parser
    rejected, so following the documentation built the wrong image."""
    help_text = format_help_message()
    tokens = set(re.findall(r"/request_bot_review\s+([^\n#`]+)", help_text))
    advertised = {
        tok
        for line in tokens
        for tok in line.split()
        if re.fullmatch(r"(?:jetson-)?(?:jp)?\d+(?:\.\d+)?", tok, re.IGNORECASE)
    }
    assert advertised, "help message no longer shows any JetPack token"
    for token in sorted(advertised):
        variants = parse(f"/request_bot_review {token}")["perception_variants"]
        assert variants, f"help advertises {token!r} but the parser drops it"
        assert variants[0] in SUPPORTED_JP_VERSIONS


# ── The other flags, so a regex change cannot quietly eat them ───────────────


def test_flags_and_targets_parse_together():
    result = parse("/request_bot_review force skip-tests core unitree/g1")
    assert result["force"] is True
    assert result["skip_tests"] is True
    assert result["force_targets"] == ["core", "unitree/g1"]
    assert result["perception_variants"] == []


@pytest.mark.parametrize(
    "arg,key",
    [
        ("skip-build", "skip_build"),
        ("build-only", "build_only"),
        ("skip-tests", "skip_tests"),
        ("force", "force"),
        ("help", "help"),
    ],
)
def test_each_flag_sets_its_own_field(arg, key):
    assert parse(f"/request_bot_review {arg}")[key] is True


def test_a_bare_trigger_requests_nothing_in_particular():
    result = parse("/request_bot_review")
    assert result["force_targets"] == []
    assert result["perception_variants"] == []
    assert not any(
        result[k] for k in ("force", "help", "skip_build", "build_only", "skip_tests")
    )


def test_prose_that_merely_mentions_the_trigger_is_not_a_trigger():
    assert parse_trigger_command("you can run /request_bot_review to rebuild") is None


def test_a_quoted_trigger_line_still_counts():
    body = "> /request_bot_review force jp61\n\n感谢审查。"
    assert parse(body)["perception_variants"] == ["6.1"]
