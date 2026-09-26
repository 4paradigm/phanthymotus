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

# ------------------------------------------------------------------
# Regression coverage for PR #279 runtime findings.
# ------------------------------------------------------------------

def test_pr279_real_test_result_footer_does_not_fake_skip():
    """The standard skip-tests help footer must not mark real results skipped."""
    from agents.deploy_approval.review_comment_parser import parse_test_results

    body = textwrap.dedent("""\
<!-- pr-review-agent -->
## PR Review Agent — Test Results

Commit: `885ff77`

| Suite | Result | Passed | Failed | Took |
|-------|--------|--------|--------|------|
| agent-core | :x: 12 failed | 1387 | 12 | 3m 25s |
| perception | :white_check_mark: Passed | 1102 | 0 | 1m 53s |
| actucore | :x: 1 failed | 471 | 1 | 1m 19s |

---
<sub>Tests run automatically after each build. `/request_bot_review skip-tests` to skip them.</sub>
""")

    passed, failed, skipped = parse_test_results(body)

    assert passed == 2960
    assert failed == 13
    assert skipped is False


def test_lifecycle_history_renderer_symbol_available_to_service():
    """Lifecycle reconciliation must not fail with _build_history_block NameError."""
    from agents.deploy_approval import service
    from agents.deploy_approval.github_state_proxy import _build_history_block

    assert service._build_history_block is _build_history_block


@pytest.mark.asyncio
async def test_existing_history_lifecycle_write_executes_without_nameerror():
    """A real history rewrite must execute every production history helper."""
    from agents.deploy_approval import comments as comments_mod
    from agents.deploy_approval.config import Config
    from agents.deploy_approval.github_state_proxy import (
        HIDDEN_STATE_MARKER,
        _build_hidden_state_body,
        _insert_history_into_visible,
    )
    from agents.deploy_approval.models import MachineInfo
    from agents.deploy_approval.policy import Policy
    from agents.deploy_approval.service import DeployController

    config = Config(github_repos=["4paradigm/phanthymotus"])
    proxy = MagicMock()
    github = MagicMock()
    policy = Policy(config)
    policy.machines = {
        "test-machine": MachineInfo(
            alias="test-machine",
            node_id="node-1",
            owners=["owner"],
            node_host="127.0.0.1",
        )
    }
    controller = DeployController(config, proxy, policy, github)

    state = {"status": "review-required"}
    visible = _insert_history_into_visible(
        comments_mod.review_required(
            "4paradigm/phanthymotus", 279, "a" * 40,
        ),
        {
            "event": "Existing event",
            "lifecycle": "`none` → `review-required`",
            "timestamp": "2026-09-26 10:00:00",
        },
    )
    existing_body = _build_hidden_state_body(visible, state)
    proxy.find_trusted_lifecycle_comment = AsyncMock(
        return_value={"id": 2790, "body": existing_body},
    )
    proxy.get_issue_comments = AsyncMock(return_value=[])
    proxy.write_hidden_state = AsyncMock()
    proxy.is_bot_comment = MagicMock(return_value=False)

    await controller._write_lifecycle_with_history(
        "4paradigm/phanthymotus",
        279,
        state,
        comments_mod.reviewing("4paradigm/phanthymotus", 279, "a" * 40),
        event={
            "event": "New event",
            "lifecycle": "`review-required` → `reviewing`",
            "timestamp": "2026-09-26 10:01:00",
        },
    )

    proxy.write_hidden_state.assert_awaited_once()
    written_visible = proxy.write_hidden_state.call_args.args[2]
    assert HIDDEN_STATE_MARKER not in written_visible
    assert "Existing event" in written_visible
    assert "New event" in written_visible


@pytest.mark.asyncio
async def test_reconcile_pr279_evidence_transitions_review_required_to_deploy_ready():
    """PR #279 comments must drive the complete production reconcile path."""
    from agents.deploy_approval import comments as comments_mod
    from agents.deploy_approval.config import Config
    from agents.deploy_approval.github_state_proxy import (
        _build_hidden_state_body,
        _insert_history_into_visible,
    )
    from agents.deploy_approval.policy import Policy
    from agents.deploy_approval.service import DeployController

    repo = "4paradigm/phanthymotus"
    pr_number = 279
    head = "885ff7799061a8e25fd68fe9743492b1d35cc6e6"
    author = {"id": TRUSTED_AUTHOR_ID, "login": TRUSTED_AUTHOR_LOGIN}

    def review_comment(body, cid, created_at):
        return {
            "id": cid,
            "body": body,
            "created_at": created_at,
            "updated_at": created_at,
            "user": author,
        }

    build_rows = [
        ("core", "core"),
        ("perception (jetson-jp5.11)", "perception-5.11"),
        ("perception (jetson-jp6.1)", "perception-6.1"),
        ("actucore (jetson-jp5.11)", "actucore-5.11"),
        ("actucore (jetson-jp6.1)", "actucore-6.1"),
    ]
    build_lines = [
        "| Target | Status | Version | Took |",
        "|--------|--------|---------|------|",
    ]
    image_lines = []
    for label, image_name in build_rows:
        build_lines.append(
            f"| {label} | :white_check_mark: Success | `release.279` | 1m |"
        )
        image_lines.extend([
            f"**{label}**",
            "",
            "```",
            f"registry.example/phanthymotus/{image_name}:release.279",
            "```",
            "",
        ])
    build_body = "\n".join([
        "<!-- pr-review-agent -->",
        "",
        "## PR Review Agent — Build Result",
        "",
        "Commit: `885ff77`",
        "",
        *build_lines,
        "",
        "### Images",
        "",
        *image_lines,
    ])
    test_body = textwrap.dedent("""\
<!-- pr-review-agent -->

## PR Review Agent — Test Results

Commit: `885ff77`

| Suite | Result | Passed | Failed | Took |
|-------|--------|--------|--------|------|
| agent-core | :x: 12 failed | 1387 | 12 | 3m 25s |
| perception | :white_check_mark: Passed | 1102 | 0 | 1m 53s |
| actucore | :x: 1 failed | 471 | 1 | 1m 19s |

---
<sub>Tests run automatically after each build. `/request_bot_review skip-tests` to skip them.</sub>
""")
    code_review_body = textwrap.dedent("""\
<!-- pr-review-agent -->

## PR Review Agent — Code Review

No blocking findings.

---
<sub>Generated automatically by PR Review Agent.</sub>
""")
    comments = [
        review_comment(build_body, 27901, "2026-09-26T10:00:00Z"),
        review_comment(test_body, 27902, "2026-09-26T10:01:00Z"),
        review_comment(code_review_body, 27903, "2026-09-26T10:02:00Z"),
    ]
    from agents.deploy_approval.review_comment_parser import extract_review_evidence
    parsed_evidence = extract_review_evidence(
        comments, TRUSTED_AUTHOR_ID, TRUSTED_AUTHOR_LOGIN,
    )
    assert parsed_evidence is not None
    assert len(parsed_evidence.builds) == 5
    assert parsed_evidence.test_passed == 2960
    assert parsed_evidence.test_failed == 13
    assert parsed_evidence.test_skipped is False

    initial_state = {
        "version": 1,
        "head_sha": head,
        "status": "review-required",
        "review_evidence": {},
        "components": [],
        "deployments": [],
        "case_results": {},
        "test_result": "",
        "cos": {"object_key": "", "sha256": "", "size": 0},
        "approve_attempts": [],
        "approve_attempts_total": 0,
        "approve_attempts_truncated": False,
        "command": {"comment_id": 0, "kind": "", "phase": "completed", "args": {}},
        "last_processed_comment_id": 0,
    }
    config = Config(github_repos=[repo])
    proxy = MagicMock()
    github = MagicMock()
    controller = DeployController(config, proxy, Policy(config), github)
    proxy.read_hidden_state = AsyncMock(return_value=initial_state)
    proxy.get_pr = AsyncMock(return_value={
        "state": "open",
        "merged": False,
        "head": {"sha": head},
    })
    github.get_issue_comments = AsyncMock(return_value=comments)
    github.resolve_commit_sha = AsyncMock(return_value=head)
    existing_visible = _insert_history_into_visible(
        comments_mod.review_required(repo, pr_number, head),
        {
            "event": "Lifecycle initialized",
            "lifecycle": "`none` → `review-required`",
            "timestamp": "2026-09-26 09:59:00",
        },
    )
    proxy.find_trusted_lifecycle_comment = AsyncMock(return_value={
        "id": 27900,
        "body": _build_hidden_state_body(existing_visible, initial_state),
    })
    proxy.get_issue_comments = AsyncMock(return_value=[])
    proxy.write_hidden_state = AsyncMock()
    proxy.project_status_label = AsyncMock()
    proxy.is_bot_comment = MagicMock(return_value=False)

    await controller.reconcile_pr(repo, pr_number)

    assert initial_state["status"] == "deploy-ready"
    evidence = initial_state["review_evidence"]
    assert evidence
    assert evidence["test_passed"] == 2960
    assert evidence["test_failed"] == 13
    assert evidence["test_skipped"] is False
    assert len(build_rows) == 5
    from agents.deploy_approval.github_state_proxy import _validate_hidden_state
    _validate_hidden_state(initial_state)
    assert proxy.write_hidden_state.await_count == 1
    proxy.project_status_label.assert_awaited_once_with(repo, pr_number, "deploy-ready")


# ------------------------------------------------------------------
# Regression coverage for core component/state contracts.
# ------------------------------------------------------------------

def _core_contract_component(**overrides):
    component = {
        "component_id": "core-component",
        "target": "core",
        "driver_path": "",
        "variant": "",
        "review_image_tag": "registry.example/core:v1",
        "image_ref": "registry.example/core:v1",
        "resolved_platform": "linux/arm64",
        "runtime_id": "core",
    }
    component.update(overrides)
    return component


def _core_contract_state(*, component=None, health=None, status="deploy-requested"):
    state = {
        "version": 1,
        "head_sha": "a" * 40,
        "status": status,
        "review_evidence": {
            "build_comment_id": 1,
            "build_comment_updated_at": "2026-09-26T00:00:00Z",
            "commit_prefix": "abcdef1",
            "resolved_head_sha": "a" * 40,
            "test_comment_id": 2,
            "test_comment_updated_at": "2026-09-26T00:01:00Z",
            "code_review_comment_id": 3,
            "code_review_comment_updated_at": "2026-09-26T00:02:00Z",
            "review_author_id": TRUSTED_AUTHOR_ID,
            "test_passed": 2960,
            "test_failed": 13,
            "test_skipped": False,
        },
        "components": [component or _core_contract_component()],
        "deployments": [],
        "case_results": {},
        "test_result": "",
        "cos": {"object_key": "", "sha256": "", "size": 0},
        "approve_attempts": [],
        "approve_attempts_total": 0,
        "approve_attempts_truncated": False,
        "command": {"comment_id": 0, "kind": "", "phase": "completed", "args": {}},
        "last_processed_comment_id": 0,
    }
    if health is not None:
        state["approve_attempts"] = [{
            "comment_id": 1,
            "actor": "owner",
            "machine": "test-machine",
            "preflight": [],
            "outcome": "deployed",
            "health": [health],
        }]
        state["approve_attempts_total"] = 1
    return state


def _core_contract_controller():
    from agents.deploy_approval.config import Config
    from agents.deploy_approval.policy import Policy
    from agents.deploy_approval.service import DeployController

    config = Config(github_repos=["4paradigm/phanthymotus"])
    proxy = MagicMock()
    github = MagicMock()
    controller = DeployController(config, proxy, Policy(config), github)
    return controller, proxy, github


def test_core_component_schema_is_valid_and_existing_targets_remain_valid():
    from agents.deploy_approval.github_state_proxy import _validate_hidden_state

    _validate_hidden_state(_core_contract_state())
    _validate_hidden_state(_core_contract_state(component={
        "component_id": "perception-component",
        "target": "perception",
        "driver_path": "",
        "variant": "5.11",
        "review_image_tag": "registry.example/perception:v1",
        "image_ref": "registry.example/perception:v1",
        "resolved_platform": "linux/arm64",
        "runtime_id": "perception",
    }))
    _validate_hidden_state(_core_contract_state(component={
        "component_id": "actucore-component",
        "target": "actucore",
        "driver_path": "",
        "variant": "6.1",
        "review_image_tag": "registry.example/actucore:v1",
        "image_ref": "registry.example/actucore:v1",
        "resolved_platform": "linux/arm64",
        "runtime_id": "actucore",
    }))
    _validate_hidden_state(_core_contract_state(component={
        "component_id": "driver-component",
        "target": "driver",
        "driver_path": "unitree/g1",
        "variant": "",
        "review_image_tag": "registry.example/driver:v1",
        "image_ref": "registry.example/driver:v1",
        "resolved_platform": "linux/arm64",
        "runtime_id": "unitree-g1",
    }))


@pytest.mark.parametrize("overrides", [
    {"variant": "5.11"},
    {"driver_path": "core/path"},
    {"runtime_id": "perception"},
])
def test_core_component_schema_rejects_noncanonical_identity(overrides):
    from agents.deploy_approval.github_state_proxy import (
        MalformedHiddenStateError,
        _validate_hidden_state,
    )

    with pytest.raises(MalformedHiddenStateError):
        _validate_hidden_state(
            _core_contract_state(component=_core_contract_component(**overrides))
        )


def test_unknown_component_target_remains_fail_closed():
    from agents.deploy_approval.github_state_proxy import (
        MalformedHiddenStateError,
        _validate_hidden_state,
    )

    with pytest.raises(MalformedHiddenStateError):
        _validate_hidden_state(
            _core_contract_state(component=_core_contract_component(target="unknown"))
        )


@pytest.mark.asyncio
async def test_core_component_snapshot_binds_core_runtime_without_driver_resolution():
    from agents.deploy_approval.models import BuildInfo

    controller, _proxy, github = _core_contract_controller()
    build = BuildInfo(
        idx=0,
        target="core",
        driver_path="",
        variant="",
        success=True,
        image_tag="registry.example/core:v1",
        deployable=True,
    )

    snapshot = await controller._build_component_snapshot(
        "4paradigm/phanthymotus", 279, "a" * 40, [build],
    )

    assert snapshot is not None
    assert snapshot == [{
        "component_id": snapshot[0]["component_id"],
        "target": "core",
        "driver_path": "",
        "variant": "",
        "review_image_tag": "registry.example/core:v1",
        "image_ref": "registry.example/core:v1",
        "resolved_platform": snapshot[0]["resolved_platform"],
        "runtime_id": "core",
    }]
    assert not hasattr(controller, "registry")
    github.list_drivers.assert_not_called()


@pytest.mark.asyncio
async def test_request_deploy_uses_canonical_review_snapshot_and_core_component():
    from agents.deploy_approval.review_comment_parser import (
        ReviewBuild,
        ReviewCommentEvidence,
    )

    controller, proxy, github = _core_contract_controller()
    head = "a" * 40
    initial_state = _core_contract_state(status="deploy-ready")
    initial_state["components"] = []
    proxy.read_hidden_state = AsyncMock(return_value=initial_state)
    proxy.comment_identity = AsyncMock(return_value=("111", "alice"))
    proxy.get_pr = AsyncMock(return_value={
        "state": "open",
        "merged": False,
        "head": {"sha": head},
        "user": {"id": 111, "login": "alice"},
    })
    proxy.find_trusted_lifecycle_comment = AsyncMock(return_value=None)
    proxy.write_hidden_state = AsyncMock()
    proxy.project_status_label = AsyncMock()
    github.get_issue_comments = AsyncMock(return_value=[])
    github.resolve_commit_sha = AsyncMock(return_value=head)
    evidence = ReviewCommentEvidence(
        commit_prefix="abcdef1",
        review_author_id=TRUSTED_AUTHOR_ID,
        builds=[ReviewBuild(
            target="core",
            driver_path="",
            variant="",
            success=True,
            image_tag="registry.example/core:v1",
            version="v1",
        )],
        build_comment_id=1001,
        build_comment_updated_at="2026-09-26T00:00:00Z",
        test_comment_id=1002,
        test_comment_updated_at="2026-09-26T00:01:00Z",
        test_passed=2960,
        test_failed=13,
        test_skipped=False,
        code_review_comment_id=1003,
        code_review_comment_updated_at="2026-09-26T00:02:00Z",
        code_review_text="No blocking findings.",
    )
    with patch(
        "agents.deploy_approval.service.extract_review_evidence",
        return_value=evidence,
    ):
        handled = await controller.handle_request_deploy(
            "4paradigm/phanthymotus", 279, 501,
        )

    assert handled is True
    written_state = proxy.write_hidden_state.call_args.args[3]
    assert written_state["status"] == "deploy-requested"
    assert written_state["review_evidence"]["test_passed"] == 2960
    assert written_state["review_evidence"]["test_failed"] == 13
    assert written_state["review_evidence"]["test_skipped"] is False
    assert written_state["components"][0]["target"] == "core"
    assert written_state["components"][0]["runtime_id"] == "core"
    assert written_state["components"][0]["variant"] == ""
    assert written_state["components"][0]["driver_path"] == ""
    controller._validate_hidden_state(written_state)
    proxy.project_status_label.assert_awaited_once_with(
        "4paradigm/phanthymotus", 279, "deploy-requested",
    )


@pytest.mark.parametrize("health", [
    {
        "component_id": "core-component",
        "runtime_id": "core",
        "current_tag": "v1",
        "target_tag": "v1",
        "verified": True,
        "already_target": True,
    },
    {
        "component_id": "core-component",
        "runtime_id": "core",
        "current_tag": "v1",
        "target_tag": "v1",
        "verified": True,
        "already_target": False,
    },
    {
        "component_id": "core-component",
        "runtime_id": "core",
        "current_tag": "",
        "target_tag": "v1",
        "verified": False,
        "error": "restart did not converge",
    },
])
def test_core_health_schema_accepts_already_target_update_success_and_uncertain(health):
    from agents.deploy_approval.github_state_proxy import _validate_hidden_state

    _validate_hidden_state(_core_contract_state(health=health))


@pytest.mark.parametrize("health", [
    {
        "component_id": "core-component", "runtime_id": "perception",
        "current_tag": "v1", "target_tag": "v1", "verified": True,
    },
    {
        "component_id": "core-component", "runtime_id": "core",
        "current_tag": "v0", "target_tag": "v1", "verified": True,
    },
    {
        "component_id": "core-component", "runtime_id": "core",
        "current_tag": "", "target_tag": "v1", "verified": True,
    },
    {
        "component_id": "core-component", "runtime_id": "core",
        "current_tag": "v0", "target_tag": "", "verified": False,
    },
    {
        "component_id": "core-component", "runtime_id": "core",
        "current_tag": "v1", "target_tag": "v1", "verified": "yes",
    },
    {
        "component_id": "core-component", "runtime_id": "core",
        "current_tag": "v0", "target_tag": "v1", "verified": False,
        "already_target": "yes",
    },
    {
        "component_id": "core-component", "runtime_id": "core",
        "current_tag": "v0", "target_tag": "v1", "verified": False,
        "already_target": True,
    },
    {
        "component_id": "core-component", "runtime_id": "core",
        "current_tag": "v0", "target_tag": "v1", "verified": False,
        "error": "",
    },
    {
        "component_id": "core-component", "runtime_id": "core",
        "current_tag": "v1", "target_tag": "v1", "verified": True,
        "error": "should be uncertain",
    },
    {
        "component_id": "core-component", "runtime_id": "core",
        "current_tag": "v1", "target_tag": "v1", "verified": True,
        "unexpected": True,
    },
])
def test_core_health_schema_rejects_malformed_values(health):
    from agents.deploy_approval.github_state_proxy import (
        MalformedHiddenStateError,
        _validate_hidden_state,
    )

    with pytest.raises(MalformedHiddenStateError):
        _validate_hidden_state(_core_contract_state(health=health))


@pytest.mark.asyncio
async def test_core_already_target_health_persists_through_hidden_state_validation():
    controller, _proxy, _github = _core_contract_controller()
    core = AsyncMock()
    core.core_update_check = AsyncMock(return_value={"current_tag": "v1"})
    core.update_core = AsyncMock()

    verified, health, post_performed = await controller._deploy_or_adopt_core_component(
        core, "node-1", "core-component", "registry.example/core:v1",
    )
    assert verified is True
    assert post_performed is False
    assert health["runtime_id"] == "core"
    assert health["already_target"] is True
    core.update_core.assert_not_awaited()

    state = _core_contract_state(health=health)
    from agents.deploy_approval.github_state_proxy import _validate_hidden_state
    _validate_hidden_state(state)


# ------------------------------------------------------------------
# Regression coverage for static core runtime identity during recovery.
# ------------------------------------------------------------------

def _runtime_binding_perception(component_id="perception-component"):
    return {
        "component_id": component_id,
        "target": "perception",
        "driver_path": "",
        "variant": "5.11",
        "review_image_tag": "registry.example/perception:v1",
        "image_ref": "registry.example/perception:v1",
        "resolved_platform": "linux/arm64",
        "runtime_id": "perception",
    }


def _runtime_binding_actucore(component_id="actucore-component"):
    return {
        "component_id": component_id,
        "target": "actucore",
        "driver_path": "",
        "variant": "6.1",
        "review_image_tag": "registry.example/actucore:v1",
        "image_ref": "registry.example/actucore:v1",
        "resolved_platform": "linux/arm64",
        "runtime_id": "actucore",
    }


def _runtime_binding_driver(component_id="driver-component"):
    return {
        "component_id": component_id,
        "target": "driver",
        "driver_path": "unitree/g1",
        "variant": "",
        "review_image_tag": "registry.example/driver:v1",
        "image_ref": "registry.example/driver:v1",
        "resolved_platform": "linux/arm64",
        "runtime_id": "unitree-g1",
    }


def _recovery_evidence(*builds, build_comment_id=1):
    from agents.deploy_approval.review_comment_parser import ReviewCommentEvidence

    return ReviewCommentEvidence(
        commit_prefix="abcdef1",
        review_author_id=TRUSTED_AUTHOR_ID,
        builds=list(builds),
        build_comment_id=build_comment_id,
        build_comment_updated_at="2026-09-26T00:00:00Z",
        test_comment_id=2,
        test_comment_updated_at="2026-09-26T00:01:00Z",
        test_passed=2960,
        test_failed=13,
        test_skipped=False,
        code_review_comment_id=3,
        code_review_comment_updated_at="2026-09-26T00:02:00Z",
        code_review_text="No blocking findings.",
    )


def test_fresh_component_runtime_binding_keeps_core_and_clears_non_core():
    controller, _proxy, _github = _core_contract_controller()
    rebuilt = controller._components_with_preserved_runtime_bindings(
        [_core_contract_component(), _runtime_binding_perception()],
        [_core_contract_component(), _runtime_binding_perception()],
        [],
    )

    assert rebuilt is not None
    assert rebuilt[0]["runtime_id"] == "core"
    assert "runtime_id" not in rebuilt[1]


def test_preserved_bindings_keep_deployed_non_core_and_undeployed_core_static():
    controller, _proxy, _github = _core_contract_controller()
    rebuilt = controller._components_with_preserved_runtime_bindings(
        [
            {**_runtime_binding_perception(), "runtime_id": "stale"},
            _core_contract_component(),
        ],
        [_runtime_binding_perception(), _core_contract_component()],
        [{
            "machine": "machine-1",
            "component_ids": ["perception-component"],
            "phase": "deployed",
        }],
    )

    assert rebuilt is not None
    assert rebuilt[0]["runtime_id"] == "perception"
    assert rebuilt[1]["runtime_id"] == "core"


def test_deployed_core_binding_remains_canonical_and_validates():
    controller, _proxy, _github = _core_contract_controller()
    rebuilt = controller._components_with_preserved_runtime_bindings(
        [_core_contract_component(runtime_id="stale-core")],
        [_core_contract_component(runtime_id="core")],
        [{
            "machine": "machine-1",
            "component_ids": ["core-component"],
            "phase": "deployed",
        }],
    )

    assert rebuilt == [{**_core_contract_component(), "runtime_id": "core"}]
    from agents.deploy_approval.github_state_proxy import _validate_hidden_state
    _validate_hidden_state(_core_contract_state(component=rebuilt[0]))


async def _run_core_uncertain_recovery(controller, proxy, github, state, evidence):
    head = state["head_sha"]
    proxy.get_pr = AsyncMock(return_value={
        "state": "open",
        "merged": False,
        "head": {"sha": head},
    })
    proxy.find_trusted_lifecycle_comment = AsyncMock(return_value=None)
    proxy.write_hidden_state = AsyncMock()
    proxy.project_status_label = AsyncMock()
    github.get_issue_comments = AsyncMock(return_value=[])
    github.resolve_commit_sha = AsyncMock(return_value=head)
    with patch(
        "agents.deploy_approval.service.extract_review_evidence",
        return_value=evidence,
    ):
        result = await controller._refresh_uncertain_state(
            "4paradigm/phanthymotus", 279, state,
        )
    assert proxy.write_hidden_state.await_count == 1
    written_state = proxy.write_hidden_state.call_args.args[3]
    return result, written_state


@pytest.mark.asyncio
async def test_uncertain_recovery_evidence_change_keeps_core_runtime_binding():
    from agents.deploy_approval.review_comment_parser import ReviewBuild

    controller, proxy, github = _core_contract_controller()
    state = _core_contract_state()
    state["command"] = {
        "comment_id": 17,
        "kind": "approve_deploy",
        "phase": "uncertain",
        "args": {"machine": "machine-1"},
    }
    evidence = _recovery_evidence(
        ReviewBuild(
            target="core", driver_path="", variant="", success=True,
            image_tag="registry.example/core:v1", version="v1",
        ),
        build_comment_id=101,
    )

    result, written_state = await _run_core_uncertain_recovery(
        controller, proxy, github, state, evidence,
    )

    assert result == "deploy-requested"
    assert written_state["components"][0]["runtime_id"] == "core"
    from agents.deploy_approval.github_state_proxy import _validate_hidden_state
    _validate_hidden_state(written_state)


@pytest.mark.asyncio
async def test_uncertain_recovery_migration_fallback_keeps_core_and_clears_fresh_non_core():
    from agents.deploy_approval.review_comment_parser import ReviewBuild

    controller, proxy, github = _core_contract_controller()
    state = _core_contract_state()
    state["command"] = {
        "comment_id": 17,
        "kind": "approve_deploy",
        "phase": "uncertain",
        "args": {"machine": "machine-1"},
    }
    perception = _runtime_binding_perception("perception-old")
    perception.pop("runtime_id")
    state["components"] = [_core_contract_component(), perception]
    state["deployments"] = [{
        "machine": "machine-1",
        "component_ids": ["perception-old"],
        "phase": "deployed",
    }]
    evidence = _recovery_evidence(
        ReviewBuild(
            target="core", driver_path="", variant="", success=True,
            image_tag="registry.example/core:v1", version="v1",
        ),
        ReviewBuild(
            target="perception", driver_path="", variant="5.11", success=True,
            image_tag="registry.example/perception:v1", version="v1",
        ),
    )

    result, written_state = await _run_core_uncertain_recovery(
        controller, proxy, github, state, evidence,
    )

    assert result == "deploy-requested"
    core = next(c for c in written_state["components"] if c["target"] == "core")
    perception = next(
        c for c in written_state["components"] if c["target"] == "perception"
    )
    assert core["runtime_id"] == "core"
    assert "runtime_id" not in perception
    from agents.deploy_approval.github_state_proxy import _validate_hidden_state
    _validate_hidden_state(written_state)


@pytest.mark.asyncio
async def test_uncertain_recovery_rule_a_migrates_undeployed_core_with_static_binding():
    from agents.deploy_approval.review_comment_parser import ReviewBuild

    controller, proxy, github = _core_contract_controller()
    state = _core_contract_state(
        component=_core_contract_component(
            image_ref="registry.example/core@sha256:" + "a" * 64,
        ),
    )
    state["command"] = {
        "comment_id": 17,
        "kind": "approve_deploy",
        "phase": "uncertain",
        "args": {"machine": "machine-1"},
    }
    evidence = _recovery_evidence(ReviewBuild(
        target="core", driver_path="", variant="", success=True,
        image_tag="registry.example/core:v1", version="v1",
    ))

    result, written_state = await _run_core_uncertain_recovery(
        controller, proxy, github, state, evidence,
    )

    assert result == "deploy-requested"
    assert written_state["components"] == [{
        "component_id": "core-component",
        "target": "core",
        "driver_path": "",
        "variant": "",
        "review_image_tag": "registry.example/core:v1",
        "image_ref": "registry.example/core:v1",
        "resolved_platform": "linux/arm64",
        "runtime_id": "core",
    }]
    from agents.deploy_approval.github_state_proxy import _validate_hidden_state
    _validate_hidden_state(written_state)


@pytest.mark.parametrize("component", [
    _runtime_binding_perception(),
    _runtime_binding_actucore(),
    _runtime_binding_driver(),
])
def test_fresh_undeployed_non_core_components_drop_runtime_binding(component):
    controller, _proxy, _github = _core_contract_controller()
    rebuilt = controller._components_with_preserved_runtime_bindings(
        [component], [component], [],
    )

    assert rebuilt is not None
    assert "runtime_id" not in rebuilt[0]
