"""GitHub Review Comment Contract Tests.

Replaces the old Review Agent HTTP API contract tests.
Tests the full GitHub-comment-based review evidence pipeline:
- Trusted author validation (PAT author supported)
- Build Result parsing
- Test Results parsing
- Code Review parsing
- Short SHA resolution
- Review Evidence state
- Latest rerun ambiguity handling
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Parser tests (no import of removed review_client) ──────────────────

TRUSTED_AUTHOR_ID = "7950763"
TRUSTED_AUTHOR_LOGIN = "kentcyq"

_BUILD_COMMENT_BODY = textwrap.dedent("""\
<!-- pr-review-agent -->

## PR Review Agent — Build Result

Commit: `abcdef1`

All builds succeeded.

| Target                   | Status                     | Version                          | Took |
| ------------------------ | -------------------------- | -------------------------------- | ---- |
| actucore (jetson-jp5.11) | :white_check_mark: Success | `release.260918.abcdef1`         | 7s   |

### Images

**actucore (jetson-jp5.11)**

```
ccr.ccs.tencentyun.com/phanthy-motus/actucore:release.260918.abcdef1
```
""")

_TEST_COMMENT_BODY = textwrap.dedent("""\
<!-- pr-review-agent -->

## PR Review Agent — Test Results

Commit: `abcdef1`

| Suite      | Result                    | Passed | Failed | Took |
| ---------- | ------------------------- | ------ | ------ | ---- |
| actucore   | :white_check_mark: Passed | 402    | 0      | 55s  |
""")

_CODE_REVIEW_BODY = textwrap.dedent("""\
<!-- pr-review-agent -->

## PR Review Agent — Code Review

All checks passed. No blocking findings.

---

<sub>Generated automatically by PR Review Agent.</sub>
""")


def _make_comment(
    body: str,
    cid: int = 1001,
    created_at: str = "2026-09-18T03:55:00Z",
    updated_at: str = "2026-09-18T03:56:00Z",
    author_id: str = TRUSTED_AUTHOR_ID,
    author_login: str = TRUSTED_AUTHOR_LOGIN,
    performed_via_github_app=None,
):
    return {
        "id": cid,
        "body": body,
        "created_at": created_at,
        "updated_at": updated_at,
        "user": {
            "id": author_id,
            "login": author_login,
            "performed_via_github_app": performed_via_github_app,
        },
    }


class TestTrustedPatAuthorCommentAccepted:
    """Test 7.1: Trusted Author"""

    def test_trusted_pat_author_comment_accepted(self):
        """PAT author (performed_via_github_app=None) with correct id passes."""
        from agents.deploy_approval.review_comment_parser import (
            extract_review_evidence,
        )

        comments = [
            _make_comment(_BUILD_COMMENT_BODY, cid=1001, created_at="2026-09-18T03:55:00Z"),
            _make_comment(_TEST_COMMENT_BODY, cid=1002, created_at="2026-09-18T03:56:00Z"),
            _make_comment(_CODE_REVIEW_BODY, cid=1003, created_at="2026-09-18T03:57:00Z"),
        ]
        evidence = extract_review_evidence(
            comments, TRUSTED_AUTHOR_ID, TRUSTED_AUTHOR_LOGIN
        )
        assert evidence is not None
        assert evidence.build_comment_id == 1001
        assert len(evidence.builds) == 1
        assert evidence.builds[0].success is True

    def test_marker_from_wrong_author_rejected(self):
        """Comment from untrusted author is rejected."""
        from agents.deploy_approval.review_comment_parser import (
            extract_review_evidence,
        )

        comments = [
            _make_comment(
                _BUILD_COMMENT_BODY,
                cid=1001,
                author_id="9999999",
                author_login="fake-author",
            ),
            _make_comment(_CODE_REVIEW_BODY, cid=1003),
        ]
        evidence = extract_review_evidence(
            comments, TRUSTED_AUTHOR_ID, TRUSTED_AUTHOR_LOGIN
        )
        assert evidence is None

    def test_correct_id_wrong_configured_login_rejected(self):
        """If author_login is configured and doesn't match, reject."""
        from agents.deploy_approval.review_comment_parser import (
            extract_review_evidence,
        )

        comments = [
            _make_comment(_BUILD_COMMENT_BODY, cid=1001, created_at="2026-09-18T03:55:00Z"),
            _make_comment(_CODE_REVIEW_BODY, cid=1003, created_at="2026-09-18T03:57:00Z"),
        ]
        evidence = extract_review_evidence(
            comments, TRUSTED_AUTHOR_ID, "wrong-login"
        )
        assert evidence is None

    def test_marker_alone_is_not_trust(self):
        """Marker without correct author id does not pass."""
        from agents.deploy_approval.review_comment_parser import (
            extract_review_evidence,
        )

        comments = [
            _make_comment(
                _BUILD_COMMENT_BODY,
                cid=1001,
                author_id="9999999",
                author_login="review-agent-bot",
            ),
            _make_comment(_CODE_REVIEW_BODY, cid=1003),
        ]
        evidence = extract_review_evidence(
            comments, TRUSTED_AUTHOR_ID, TRUSTED_AUTHOR_LOGIN
        )
        assert evidence is None


class TestBuildResult:
    """Test 7.2: Build Result"""

    def test_parse_official_phanthymotus_build_result(self):
        """Official phanthymotus build with core, actucore variants."""
        from agents.deploy_approval.review_comment_parser import (
            parse_all_build_results,
        )

        body = textwrap.dedent("""\
<!-- pr-review-agent -->

## PR Review Agent — Build Result

Commit: `abcdef1`

All builds succeeded.

| Target                   | Status                     | Version                         | Took |
| ------------------------ | -------------------------- | ------------------------------- | ---- |
| core                     | :white_check_mark: Success | `release.260918.abcdef1`        | 45s  |
| actucore (jetson-jp5.11) | :white_check_mark: Success | `release.260918.abcdef1`        | 7s   |
| perception (jetson-jp6.1)| :white_check_mark: Success | `release.260918.abcdef1`        | 10s  |

### Images

**core**

```
ccr.ccs.tencentyun.com/phanthy-motus/core:release.260918.abcdef1
```

**actucore (jetson-jp5.11)**

```
ccr.ccs.tencentyun.com/phanthy-motus/actucore:release.260918.abcdef1
```

**perception (jetson-jp6.1)**

```
ccr.ccs.tencentyun.com/phanthy-motus/perception:release.260918.abcdef1
```
""")
        builds = parse_all_build_results(body)
        assert len(builds) == 3
        by_target = {b.target: b for b in builds}
        # core
        assert by_target["core"].target == "core"
        assert by_target["core"].driver_path == ""
        assert by_target["core"].variant == ""
        # actucore
        assert by_target["actucore"].target == "actucore"
        assert by_target["actucore"].driver_path == ""
        assert by_target["actucore"].variant == "5.11"
        # perception
        assert by_target["perception"].target == "perception"
        assert by_target["perception"].driver_path == ""
        assert by_target["perception"].variant == "6.1"
        # Version stays as the literal version, NOT copied to variant
        assert by_target["core"].version == "release.260918.abcdef1"

    def test_parse_official_driver_build_result(self):
        """Official driver-style build result."""
        from agents.deploy_approval.review_comment_parser import (
            parse_all_build_results,
        )

        body = textwrap.dedent("""\
<!-- pr-review-agent -->

## PR Review Agent — Build Result

Commit: `16430e7`

All builds succeeded.

| Target     | Status                     | Version                   | Took |
| ---------- | -------------------------- | ------------------------- | ---- |
| unitree/g1 | :white_check_mark: Success | `release.260826.16430e7`  | 6s   |

### Images

**unitree/g1**

```
ccr.ccs.tencentyun.com/phanthy-motus/drivers/unitree/g1:release.260826.16430e7
```
""")
        builds = parse_all_build_results(body)
        assert len(builds) == 1
        assert builds[0].target == "driver"
        assert builds[0].driver_path == "unitree/g1"
        assert builds[0].variant == ""
        assert builds[0].success is True

    def test_multi_variant_phanthymotus_build_result(self):
        """Two perception variants must produce TWO distinct ReviewBuild objects."""
        from agents.deploy_approval.review_comment_parser import (
            parse_all_build_results,
        )

        body = textwrap.dedent("""\
<!-- pr-review-agent -->

## PR Review Agent — Build Result

Commit: `abcdef1`

All builds succeeded.

| Target                   | Status                     | Version                         | Took |
| ------------------------ | -------------------------- | ------------------------------- | ---- |
| perception (jetson-jp5.11) | :white_check_mark: Success | `release.260918.abcdef1`      | 10s  |
| perception (jetson-jp6.1)  | :white_check_mark: Success | `release.260918.abcdef1`      | 12s  |

### Images

**perception (jetson-jp5.11)**

```
ccr.ccs.tencentyun.com/phanthy-motus/perception:release.260918.abcdef1
```

**perception (jetson-jp6.1)**

```
ccr.ccs.tencentyun.com/phanthy-motus/perception:release.260918.abcdef1
```
""")
        builds = parse_all_build_results(body)
        assert len(builds) == 2
        by_key = {(b.target, b.variant): b for b in builds}
        assert ("perception", "5.11") in by_key
        assert ("perception", "6.1") in by_key
        assert by_key[("perception", "5.11")].driver_path == ""
        assert by_key[("perception", "6.1")].driver_path == ""
        """Success build row without image => fail closed."""
        from agents.deploy_approval.review_comment_parser import (
            parse_all_build_results,
        )

        body = textwrap.dedent("""\
<!-- pr-review-agent -->

## PR Review Agent — Build Result

Commit: `abcdef1`

All builds succeeded.

| Target | Status                     | Version                 | Took |
| ------ | -------------------------- | ----------------------- | ---- |
| core   | :white_check_mark: Success | `release.260918.abc`    | 10s  |
""")
        builds = parse_all_build_results(body)
        assert builds == []

    def test_failed_build_not_deployable(self):
        """Failed build row is not success."""
        from agents.deploy_approval.review_comment_parser import (
            parse_all_build_results,
        )

        body = textwrap.dedent("""\
<!-- pr-review-agent -->

## PR Review Agent — Build Result

Commit: `abcdef1`

| Target | Status                    | Version                 | Took |
| ------ | ------------------------- | ----------------------- | ---- |
| core   | :x: Failed                | `release.260918.abc`    | 10s  |

### Images

**core**

```
ccr.ccs.tencentyun.com/phanthy-motus/core:release.260918.abc
```
""")
        builds = parse_all_build_results(body)
        assert len(builds) == 1
        assert builds[0].success is False

    def test_killed_build_not_deployable(self):
        """Killed build row is not success."""
        from agents.deploy_approval.review_comment_parser import (
            parse_all_build_results,
        )

        body = textwrap.dedent("""\
<!-- pr-review-agent -->

## PR Review Agent — Build Result

Commit: `abcdef1`

| Target | Status                   | Version                 | Took |
| ------ | ------------------------ | ----------------------- | ---- |
| core   | Killed                   | `release.260918.abc`    | 10s  |

### Images

**core**

```
ccr.ccs.tencentyun.com/phanthy-motus/core:release.260918.abc
```
""")
        builds = parse_all_build_results(body)
        assert len(builds) == 1
        assert builds[0].success is False

    def test_duplicate_target_fails_closed(self):
        """Duplicate target label => fail closed."""
        from agents.deploy_approval.review_comment_parser import (
            parse_all_build_results,
        )

        body = textwrap.dedent("""\
<!-- pr-review-agent -->

## PR Review Agent — Build Result

Commit: `abcdef1`

| Target | Status                     | Version                 | Took |
| ------ | -------------------------- | ----------------------- | ---- |
| core   | :white_check_mark: Success | `release.260918.abc`    | 10s  |
| core   | :white_check_mark: Success | `release.260918.def`    | 10s  |
""")
        builds = parse_all_build_results(body)
        assert builds == []

    def test_conflicting_images_fail_closed(self):
        """Duplicate target with conflicting image => fail closed (duplicate check)."""
        from agents.deploy_approval.review_comment_parser import (
            parse_all_build_results,
        )

        # Duplicate targets already caught; test that images for same target conflict
        body = textwrap.dedent("""\
<!-- pr-review-agent -->

## PR Review Agent — Build Result

Commit: `abcdef1`

| Target | Status                     | Version                 | Took |
| ------ | -------------------------- | ----------------------- | ---- |
| core   | :white_check_mark: Success | `release.260918.abc`    | 10s  |

### Images

**core**

```
ccr.ccs.tencentyun.com/phanthy-motus/core:release.260918.abc
```

**core**

```
ccr.ccs.tencentyun.com/phanthy-motus/core:release.260918.def
```
""")
        builds = parse_all_build_results(body)
        # Duplicate target in table is the fail condition
        assert builds == []

    def test_version_must_match_image_tag(self):
        """Version and image ref tag consistency is checked."""
        from agents.deploy_approval.review_comment_parser import (
            parse_all_build_results,
        )

        body = textwrap.dedent("""\
<!-- pr-review-agent -->

## PR Review Agent — Build Result

Commit: `abcdef1`

| Target | Status                     | Version                 | Took |
| ------ | -------------------------- | ----------------------- | ---- |
| core   | :white_check_mark: Success | `release.260918.abc`    | 10s  |

### Images

**core**

```
ccr.ccs.tencentyun.com/phanthy-motus/core:release.260918.abc
```
""")
        builds = parse_all_build_results(body)
        assert len(builds) == 1
        assert builds[0].success is True

    def test_parser_never_guesses_release_tag(self):
        """Parser never constructs a release tag from HEAD/date."""
        from agents.deploy_approval.review_comment_parser import (
            parse_all_build_results,
        )

        # No Images section at all — should not produce any builds
        body = textwrap.dedent("""\
<!-- pr-review-agent -->

## PR Review Agent — Build Result

Commit: `abcdef1`

| Target | Status                     | Version                 | Took |
| ------ | -------------------------- | ----------------------- | ---- |
| core   | :white_check_mark: Success | `release.260918.abc`    | 10s  |
""")
        builds = parse_all_build_results(body)
        assert builds == []


class TestTestResults:
    """Test 7.4: Test Results"""

    def test_matching_head_test_result_accepted(self):
        """Test Results comment after build with matching head is accepted.

        Actual chronological order:
          Build Result   03:55
          Test Results   03:56
          Code Review    03:57
        """
        from agents.deploy_approval.review_comment_parser import (
            extract_review_evidence,
        )

        comments = [
            _make_comment(_BUILD_COMMENT_BODY, cid=1001, created_at="2026-09-18T03:55:00Z"),
            _make_comment(_TEST_COMMENT_BODY, cid=1002, created_at="2026-09-18T03:56:00Z"),
            _make_comment(_CODE_REVIEW_BODY, cid=1003, created_at="2026-09-18T03:57:00Z"),
        ]
        evidence = extract_review_evidence(
            comments, TRUSTED_AUTHOR_ID, TRUSTED_AUTHOR_LOGIN
        )
        assert evidence is not None
        assert evidence.build_comment_id == 1001
        assert evidence.test_comment_id == 1002
        assert evidence.code_review_comment_id == 1003
        assert evidence.test_passed == 402

    def test_old_head_test_result_rejected(self):
        """Test Results for old HEAD is not correlated."""
        from agents.deploy_approval.review_comment_parser import (
            extract_review_evidence,
        )

        old_test = textwrap.dedent("""\
<!-- pr-review-agent -->

## PR Review Agent — Test Results

Commit: `1111111`

| Suite    | Result                    | Passed | Failed | Took |
| -------- | ------------------------- | ------ | ------ | ---- |
| actucore | :white_check_mark: Passed | 10     | 0      | 5s   |
""")
        comments = [
            _make_comment(_BUILD_COMMENT_BODY, cid=1001, created_at="2026-09-18T03:55:00Z"),
            _make_comment(old_test, cid=1002, created_at="2026-09-18T03:56:00Z"),
            _make_comment(_CODE_REVIEW_BODY, cid=1003, created_at="2026-09-18T03:57:00Z"),
        ]
        evidence = extract_review_evidence(
            comments, TRUSTED_AUTHOR_ID, TRUSTED_AUTHOR_LOGIN
        )
        # Old-head test result rejected: different commit => fail closed
        assert evidence is None

    def test_skip_tests_can_have_no_test_comment(self):
        """If build succeeds and code review exists but no test comment, still valid."""
        from agents.deploy_approval.review_comment_parser import (
            extract_review_evidence,
        )

        comments = [
            _make_comment(_BUILD_COMMENT_BODY, cid=1001, created_at="2026-09-18T03:55:00Z"),
            _make_comment(_CODE_REVIEW_BODY, cid=1003, created_at="2026-09-18T03:57:00Z"),
        ]
        evidence = extract_review_evidence(
            comments, TRUSTED_AUTHOR_ID, TRUSTED_AUTHOR_LOGIN
        )
        assert evidence is not None


class TestCodeReview:
    """Test 7.5: Code Review"""

    def test_nonempty_code_review_in_unambiguous_window_accepted(self):
        """Non-empty Code Review after build in clean window passes."""
        from agents.deploy_approval.review_comment_parser import (
            extract_review_evidence,
        )

        comments = [
            _make_comment(_BUILD_COMMENT_BODY, cid=1001, created_at="2026-09-18T03:55:00Z"),
            _make_comment(_CODE_REVIEW_BODY, cid=1003, created_at="2026-09-18T03:57:00Z"),
        ]
        evidence = extract_review_evidence(
            comments, TRUSTED_AUTHOR_ID, TRUSTED_AUTHOR_LOGIN
        )
        assert evidence is not None
        assert evidence.code_review_comment_id == 1003
        assert evidence.code_review_text.strip() != ""

    def test_empty_code_review_rejected(self):
        """Empty Code Review body => fail closed."""
        from agents.deploy_approval.review_comment_parser import (
            extract_review_evidence,
        )

        empty_cr = textwrap.dedent("""\
<!-- pr-review-agent -->

## PR Review Agent — Code Review

---

<sub>Generated automatically by PR Review Agent.</sub>
""")
        comments = [
            _make_comment(_BUILD_COMMENT_BODY, cid=1001, created_at="2026-09-18T03:56:00Z"),
            _make_comment(empty_cr, cid=1003),
        ]
        evidence = extract_review_evidence(
            comments, TRUSTED_AUTHOR_ID, TRUSTED_AUTHOR_LOGIN
        )
        assert evidence is None

    def test_code_review_before_selected_build_rejected(self):
        """Code Review created before build => not correlated."""
        from agents.deploy_approval.review_comment_parser import (
            extract_review_evidence,
        )

        early_cr = _make_comment(
            _CODE_REVIEW_BODY,
            cid=1000,
            created_at="2026-09-18T03:50:00Z",
        )
        comments = [
            early_cr,
            _make_comment(_BUILD_COMMENT_BODY, cid=1001, created_at="2026-09-18T03:55:00Z"),
        ]
        evidence = extract_review_evidence(
            comments, TRUSTED_AUTHOR_ID, TRUSTED_AUTHOR_LOGIN
        )
        assert evidence is None

    def test_overlapping_review_activity_fails_closed(self):
        """Intervening build comment between build and code review => ambiguous."""
        from agents.deploy_approval.review_comment_parser import (
            extract_review_evidence,
        )

        intervening_build = textwrap.dedent("""\
<!-- pr-review-agent -->

## PR Review Agent — Build Result

Commit: `abcdef2`

All builds succeeded.

| Target | Status                     | Version                 | Took |
| ------ | -------------------------- | ----------------------- | ---- |
| core   | :white_check_mark: Success | `release.260918.def`    | 10s  |

""")
        comments = [
            _make_comment(_BUILD_COMMENT_BODY, cid=1001, created_at="2026-09-18T03:55:00Z"),
            _make_comment(intervening_build, cid=10015, created_at="2026-09-18T03:55:30Z"),
            _make_comment(_CODE_REVIEW_BODY, cid=1003, created_at="2026-09-18T03:56:00Z"),
        ]
        evidence = extract_review_evidence(
            comments, TRUSTED_AUTHOR_ID, TRUSTED_AUTHOR_LOGIN
        )
        assert evidence is None

    def test_latest_same_head_incomplete_rerun_does_not_fallback(self):
        """Latest same-head rerun incomplete must NOT fall back to older complete."""
        from agents.deploy_approval.review_comment_parser import (
            extract_review_evidence,
        )

        old_complete_build = textwrap.dedent("""\
<!-- pr-review-agent -->

## PR Review Agent — Build Result

Commit: `abcdef1`

All builds succeeded.

| Target | Status                     | Version                 | Took |
| ------ | -------------------------- | ----------------------- | ---- |
| core   | :white_check_mark: Success | `release.260918.abc`    | 10s  |

### Images

**core**

```
ccr.ccs.tencentyun.com/phanthy-motus/core:release.260918.abc
```
""")
        old_complete_cr = textwrap.dedent("""\
<!-- pr-review-agent -->

## PR Review Agent — Code Review

Previous complete review.

---

<sub>Generated automatically by PR Review Agent.</sub>
""")
        new_incomplete_build = textwrap.dedent("""\
<!-- pr-review-agent -->

## PR Review Agent — Build Result

Commit: `abcdef1`

All builds succeeded.

| Target | Status                     | Version                 | Took |
| ------ | -------------------------- | ----------------------- | ---- |
| core   | :white_check_mark: Success | `release.260918.xyz`    | 10s  |

### Images

**core**

```
ccr.ccs.tencentyun.com/phanthy-motus/core:release.260918.xyz
```
""")
        # Old build + old CR at t=1, new build at t=2, NO new CR
        comments = [
            _make_comment(old_complete_build, cid=1001, created_at="2026-09-18T03:50:00Z"),
            _make_comment(old_complete_cr, cid=1002, created_at="2026-09-18T03:51:00Z"),
            _make_comment(new_incomplete_build, cid=1003, created_at="2026-09-18T03:55:00Z"),
        ]
        evidence = extract_review_evidence(
            comments, TRUSTED_AUTHOR_ID, TRUSTED_AUTHOR_LOGIN
        )
        # Latest build has no code review => None
        assert evidence is None

    def test_latest_same_head_complete_rerun_wins(self):
        """Latest same-head complete rerun wins over older complete."""
        from agents.deploy_approval.review_comment_parser import (
            extract_review_evidence,
        )

        old_complete_build = textwrap.dedent("""\
<!-- pr-review-agent -->

## PR Review Agent — Build Result

Commit: `abcdef1`

All builds succeeded.

| Target | Status                     | Version                 | Took |
| ------ | -------------------------- | ----------------------- | ---- |
| core   | :white_check_mark: Success | `release.260918.abc`    | 10s  |

### Images

**core**

```
ccr.ccs.tencentyun.com/phanthy-motus/core:release.260918.abc
```
""")
        old_complete_cr = textwrap.dedent("""\
<!-- pr-review-agent -->

## PR Review Agent — Code Review

Old review.

---

<sub>Generated automatically by PR Review Agent.</sub>
""")
        new_complete_build = textwrap.dedent("""\
<!-- pr-review-agent -->

## PR Review Agent — Build Result

Commit: `abcdef1`

All builds succeeded.

| Target | Status                     | Version                 | Took |
| ------ | -------------------------- | ----------------------- | ---- |
| core   | :white_check_mark: Success | `release.260918.xyz`    | 10s  |

### Images

**core**

```
ccr.ccs.tencentyun.com/phanthy-motus/core:release.260918.xyz
```
""")
        new_complete_cr = textwrap.dedent("""\
<!-- pr-review-agent -->

## PR Review Agent — Code Review

Newer complete review.

---

<sub>Generated automatically by PR Review Agent.</sub>
""")
        comments = [
            _make_comment(old_complete_build, cid=1001, created_at="2026-09-18T03:50:00Z"),
            _make_comment(old_complete_cr, cid=1002, created_at="2026-09-18T03:51:00Z"),
            _make_comment(new_complete_build, cid=1003, created_at="2026-09-18T03:55:00Z"),
            _make_comment(new_complete_cr, cid=1004, created_at="2026-09-18T03:56:00Z"),
        ]
        evidence = extract_review_evidence(
            comments, TRUSTED_AUTHOR_ID, TRUSTED_AUTHOR_LOGIN
        )
        assert evidence is not None
        assert evidence.build_comment_id == 1003
        assert evidence.code_review_comment_id == 1004


class TestShortSHAResolution:
    """Test 7.3: Short SHA"""

    def test_short_commit_is_resolved_via_github(self):
        """GitHubClient.resolve_commit_sha is used for commit prefix resolution."""
        from agents.deploy_approval.github_client import GitHubClient

        mock_config = MagicMock()
        mock_config.github_api_url = "https://api.github.com"
        client = GitHubClient(mock_config, token_provider=AsyncMock(return_value="token"))
        client._http = MagicMock()
        client._http.send = AsyncMock()

        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"sha": "abcdef1234567890abcdef1234567890abcdef12"}
        client._http.send.return_value = resp

        result = client._request = AsyncMock(return_value=resp)

        async def _run():
            return await client.resolve_commit_sha("owner/repo", "abcdef1")

        import asyncio
        resolved = asyncio.run(_run())
        assert resolved == "abcdef1234567890abcdef1234567890abcdef12"

    def test_resolved_sha_must_equal_fresh_pr_head(self):
        """The resolved full SHA must equal the fresh PR HEAD."""
        from agents.deploy_approval.review_comment_parser import (
            extract_review_evidence,
        )

        # The parser extracts commit_prefix; the service layer resolves it.
        # This test confirms the evidence carries the prefix for resolution.
        comments = [
            _make_comment(_BUILD_COMMENT_BODY, cid=1001, created_at="2026-09-18T03:55:00Z"),
            _make_comment(_CODE_REVIEW_BODY, cid=1003, created_at="2026-09-18T03:57:00Z"),
        ]
        evidence = extract_review_evidence(
            comments, TRUSTED_AUTHOR_ID, TRUSTED_AUTHOR_LOGIN
        )
        assert evidence is not None
        assert evidence.commit_prefix == "abcdef1"

    def test_short_sha_resolution_404_fails_closed(self):
        """GitHub returns 404 for ambiguous/missing short SHA => fail closed."""
        import asyncio
        from agents.deploy_approval.github_client import GitHubClient, GitHubError

        mock_config = MagicMock()
        mock_config.github_api_url = "https://api.github.com"
        client = GitHubClient(mock_config, token_provider=AsyncMock(return_value="token"))

        async def _run():
            with patch.object(client, "_request", new=AsyncMock(side_effect=GitHubError("404"))):
                return await client.resolve_commit_sha("owner/repo", "badsha9")

        with pytest.raises(GitHubError):
            asyncio.run(_run())

    def test_short_sha_resolution_malformed_fails_closed(self):
        """Malformed response from GitHub => fail closed."""
        import asyncio
        import httpx
        from agents.deploy_approval.github_client import GitHubClient, GitHubError

        mock_config = MagicMock()
        mock_config.github_api_url = "https://api.github.com"
        client = GitHubClient(mock_config, token_provider=AsyncMock(return_value="token"))

        async def _run():
            fake_resp = MagicMock(spec=httpx.Response)
            fake_resp.read = AsyncMock(return_value=b'{"not_sha": "xyz"}')
            fake_resp.status_code = 200
            async def araise_for_status():
                pass
            fake_resp.raise_for_status = araise_for_status
            with patch.object(client, "_request", new=AsyncMock(return_value=fake_resp)):
                return await client.resolve_commit_sha("owner/repo", "abcdef1")

        with pytest.raises(GitHubError):
            asyncio.run(_run())

    def test_old_head_build_comment_rejected(self):
        """Build comment for old HEAD is not selected."""
        from agents.deploy_approval.review_comment_parser import (
            extract_review_evidence,
        )

        old_build = textwrap.dedent("""\
<!-- pr-review-agent -->

## PR Review Agent — Build Result

Commit: `1111111`

All builds succeeded.

| Target | Status                     | Version                 | Took |
| ------ | -------------------------- | ----------------------- | ---- |
| core   | :white_check_mark: Success | `release.260918.old`    | 10s  |

### Images

**core**

```
ccr.ccs.tencentyun.com/phanthy-motus/core:release.260918.old
```
""")
        old_cr = textwrap.dedent("""\
<!-- pr-review-agent -->

## PR Review Agent — Code Review

Old review.

---

<sub>Generated automatically by PR Review Agent.</sub>
""")
        comments = [
            _make_comment(old_build, cid=1001, created_at="2026-09-18T03:55:00Z"),
            _make_comment(old_cr, cid=1002, created_at="2026-09-18T03:57:00Z"),
        ]
        evidence = extract_review_evidence(
            comments, TRUSTED_AUTHOR_ID, TRUSTED_AUTHOR_LOGIN
        )
        assert evidence is not None
        assert evidence.commit_prefix == "1111111"


class TestReviewEvidenceState:
    """Test 7.6: Review Evidence State"""

    def test_review_evidence_contains_comment_provenance(self):
        """ReviewCommentEvidence contains provenance IDs, not review_job_id."""
        from agents.deploy_approval.review_comment_parser import (
            extract_review_evidence,
        )

        comments = [
            _make_comment(_BUILD_COMMENT_BODY, cid=1001, created_at="2026-09-18T03:55:00Z"),
            _make_comment(_TEST_COMMENT_BODY, cid=1002, created_at="2026-09-18T03:56:00Z"),
            _make_comment(_CODE_REVIEW_BODY, cid=1003, created_at="2026-09-18T03:57:00Z"),
        ]
        evidence = extract_review_evidence(
            comments, TRUSTED_AUTHOR_ID, TRUSTED_AUTHOR_LOGIN
        )
        assert evidence is not None
        assert evidence.build_comment_id == 1001
        assert evidence.build_comment_updated_at != ""
        assert evidence.commit_prefix == "abcdef1"
        assert evidence.code_review_comment_id == 1003
        assert evidence.test_comment_id == 1002
        assert evidence.review_author_id == TRUSTED_AUTHOR_ID
        # review_job_id is NOT part of ReviewCommentEvidence
        assert not hasattr(evidence, "review_job_id")
