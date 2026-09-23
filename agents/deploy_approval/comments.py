"""Status comment rendering for a deployment (stateless GitHub persistence).

One lifecycle comment per PR. Uses a hidden marker for idempotent PATCH.
Each action-required state shows: status, bound head, who acts next, exact commands.
No obsolete lifecycle states, no build_index, no dpl_x.
"""

from __future__ import annotations

import datetime as _dt
from zoneinfo import ZoneInfo

import urllib.parse
from .models import BuildInfo

BOT_MARKER = "<!-- deploy-approval-agent -->"
_BEIJING = ZoneInfo("Asia/Shanghai")


def beijing_now_str() -> str:
    now = _dt.datetime.now(_BEIJING)
    return now.strftime("%Y-%m-%d %H:%M:%S")


def last_checked_line(checked_at: float | None = None) -> str:
    if isinstance(checked_at, (int, float)) and checked_at > 0:
        try:
            moment = _dt.datetime.fromtimestamp(checked_at, tz=_BEIJING).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        except (OverflowError, OSError, ValueError):
            moment = beijing_now_str()
    else:
        moment = beijing_now_str()
    return f"Last checked: {moment} (UTC+08:00, Asia/Shanghai)"


def lifecycle_marker(repo: str, pr_number: int) -> str:
    return f"<!-- deploy-approval-lifecycle:{repo}:{pr_number} -->"


def _short(sha: str) -> str:
    return (sha or "")[:7]


def _short_digest(image_ref: str) -> str:
    """Return a compact display for an image ref: short digest or truncated tag."""
    if "@sha256:" in (image_ref or ""):
        digest = image_ref.split("@sha256:", 1)[1].strip()
        if len(digest) < 12:
            return ""
        return f"@sha256:{digest[:12]}"
    # Tag form — show the tag portion (after last :)
    if image_ref and ":" in image_ref:
        tag = image_ref.rsplit(":", 1)[-1]
        if len(tag) > 40:
            return tag[:40] + "…"
        return tag
    return ""


def _compact_image_ref(image_ref: str) -> str:
    """Display an image_ref: full tag for tag form, short digest for legacy digest form."""
    if "@sha256:" in (image_ref or ""):
        digest = image_ref.split("@sha256:", 1)[1].strip()
        if len(digest) < 12:
            return ""
        return f"@sha256:{digest[:12]}"
    # Tag form — show the FULL image_ref (safe, no credentials)
    return image_ref if image_ref else ""


def _compact_running_image(image_ref: str) -> str:
    """Compact display for running_image evidence."""
    result = _short_digest(image_ref)
    if result:
        return result
    return "occupied"


def _escape(text: str) -> str:
    return (text or "").replace("`", "").replace("\r", " ").replace("\n", " ")[:500]


def _safe_download_url(url: str) -> str:
    """Validate a COS presigned download URL without truncation.

    Returns the original URL if valid, or empty string otherwise.
    Never truncates or logs the URL.
    """
    if not isinstance(url, str):
        return ""
    if not url:
        return ""
    if len(url) > 8192:
        return ""
    if "\r" in url or "\n" in url:
        return ""
    for ch in url:
        if ord(ch) < 0x20 and ch not in ("\t",):
            return ""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https":
        return ""
    if not parsed.hostname:
        return ""
    if parsed.username is not None:
        return ""
    if parsed.password is not None:
        return ""
    return url



def _build_table(builds: list[BuildInfo]) -> str:
    """Render the build results table for deploy-ready comment."""
    lines = [
        "| build | target | variant/path | build | deploy approval |",
        "|------:|--------|--------------|-------|-----------------|",
    ]
    for b in builds:
        variant = _escape(b.variant or b.driver_path or chr(0x2014))
        build_status = "success" if b.success else "failed"
        eligibility = "deployable" if b.deployable else "unsupported"
        if not b.success:
            eligibility = "not deployable"
        lines.append(
            f"| {b.idx} | {_escape(b.target)} | {variant} | {build_status} | {eligibility} |"
        )
    return "\n".join(lines)


def _cos_evidence_block(
    object_key: str = "",
    sha256: str = "",
    size: int = 0,
    download_url: str = "",
) -> list[str]:
    """Render the COS evidence block. Returns empty list if no evidence."""
    if not object_key:
        return []
    lines = [""]
    if download_url:
        safe_url = _safe_download_url(download_url)
        if safe_url:
            lines.append(f"[Download COS evidence]({safe_url})")
    text = f"COS: `{_escape(object_key)}`"
    if sha256:
        text += f" · `@sha256:{_escape(sha256[:12])}`"
    if size:
        text += f" · `{_human_size(size)}`"
    lines.append(text)
    return lines


def _human_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    else:
        return f"{size / (1024 * 1024):.1f} MB"


# Lifecycle comment builders


def review_required(repo: str, pr_number: int, head_sha: str) -> str:
    """status: review-required"""
    return "\n".join([
        BOT_MARKER,
        lifecycle_marker(repo, pr_number),
        "### Deploy Approval \u2014 Lifecycle",
        "",
        "**Status:** `review-required`",
        f"**Bound HEAD:** `{_short(head_sha)}`",
        "",
        "**Next action \u2014 Developer**",
        "",
        "`/request_bot_review`",
        "",
        "After the Review Agent completes its review of this exact HEAD, "
        "Deploy Approval will update this comment with the build list.",
        "",
        last_checked_line(),
    ])


def reviewing(repo: str, pr_number: int, head_sha: str) -> str:
    return "\n".join([
        BOT_MARKER,
        lifecycle_marker(repo, pr_number),
        "### Deploy Approval \u2014 Lifecycle",
        "",
        "**Status:** `reviewing`",
        f"**Bound HEAD:** `{_short(head_sha)}`",
        "",
        "Review Agent is reviewing/building this exact HEAD.",
        "No action required yet.",
        "",
        last_checked_line(),
    ])


def deploy_ready(repo: str, pr_number: int, head_sha: str,
                 builds: list[BuildInfo]) -> str:
    lines = [
        BOT_MARKER,
        lifecycle_marker(repo, pr_number),
        "### Deploy Approval \u2014 Lifecycle",
        "",
        "**Status:** `deploy-ready`",
        f"**Bound HEAD:** `{_short(head_sha)}`",
        "",
        f"Builds for reviewed HEAD `{_short(head_sha)}`",
        "",
        _build_table(builds),
        "",
    ]
    deployable = [b for b in builds if b.success and b.deployable]
    if deployable:
        lines.append("**Next action \u2014 Developer**")
        lines.append("")
        lines.append("`/request_deploy`")
    else:
        lines.append("No deployable builds. Review Agent must fix build errors.")
    lines.append("")
    lines.append(last_checked_line())
    return "\n".join(lines)


def deploy_requested(
    repo: str, pr_number: int, head_sha: str,
    components: list[dict],
    machine_groups: list[dict],
    gate_note: list[str] | None = None,
) -> str:
    """status: deploy-requested — shows components and compatible machines."""
    comp_lines = []
    for c in components:
        target = c.get("target", "")
        variant = c.get("variant", "") or ""
        driver_path = c.get("driver_path", "") or ""
        image_ref = c.get("image_ref", "") or ""
        label = target
        if target == "driver" and driver_path:
            label = f"driver {driver_path}"
        elif variant:
            label = f"{target} {variant}"
        elif driver_path:
            label = f"{target} {driver_path}"
        display = _compact_image_ref(image_ref)
        extra = f" · `{display}`" if display else ""
        comp_lines.append(f"- {_escape(label)}{extra}")
    lines = [
        BOT_MARKER,
        lifecycle_marker(repo, pr_number),
        "### Deploy Approval \u2014 Lifecycle",
        "",
        "**Status:** `deploy-requested`",
        f"**Bound HEAD:** `{_short(head_sha)}`",
        "",
        "### Components to deploy",
        "",
    ] + comp_lines + [
        "",
        "### Full-coverage machines",
        "",
    ]
    for mg in machine_groups:
        alias = mg.get("alias", "?")
        cids = mg.get("component_ids", [])
        cid_str = ", ".join(cids) if cids else "none"
        lines.append(f"- `{_escape(alias)}`: {cid_str}")
    if gate_note:
        lines += [
            "",
            *gate_note,
        ]
    lines += [
        "",
        "**Next action \u2014 Machine Owner**",
        "",
        "`/approve_deploy machine=<alias>`",
        "",
        last_checked_line(),
    ]
    return "\n".join(lines)


def failed_comment(
    repo: str, pr_number: int, head_sha: str,
    error: str = "",
    cos_object_key: str = "",
    cos_bundle_sha256: str = "",
    cos_bundle_size: int = 0,
    cos_download_url: str = "",

) -> str:
    lines = [
        BOT_MARKER,
        lifecycle_marker(repo, pr_number),
        "### Deploy Approval \u2014 Lifecycle",
        "",
        "**Status:** `failed`",
        f"**Bound HEAD:** `{_short(head_sha)}`",
        "",
        "Deployment or validation failed. The deployment is not accepted.",
    ]
    if error:
        lines.append(f"**Error:** {_escape(error)}")
    lines.extend(_cos_evidence_block(cos_object_key, cos_bundle_sha256, cos_bundle_size, cos_download_url))
    lines.append(last_checked_line())
    return "\n".join(lines)


def testing(
    repo: str, pr_number: int, head_sha: str,
    deployment_id: str = "",

    case_result: str = "",
) -> str:
    lines = [
        BOT_MARKER,
        lifecycle_marker(repo, pr_number),
        "### Deploy Approval \u2014 Lifecycle",
        "",
        "**Status:** `testing`",
        f"**Bound HEAD:** `{_short(head_sha)}`",
        "",
        "All required components have been deployed.",
        "Fixed Case results are advisory only.",
    ]
    if case_result:
        lines.append(f"**Automated case results:** {case_result}")
        lines.append("Fixed Case results are advisory only.")
    lines.append("")
    lines.append("**Next action \u2014 Machine Owner**")
    lines.append("")
    lines.append("`/record_test result=pass|fail [summary=\"...\"]`")
    lines.append("")
    lines.append(last_checked_line())
    return "\n".join(lines)


def succeeded_comment(
    repo: str, pr_number: int, head_sha: str,
    cos_object_key: str = "",
    cos_bundle_sha256: str = "",
    cos_bundle_size: int = 0,
    cos_download_url: str = "",

) -> str:
    lines = [
        BOT_MARKER,
        lifecycle_marker(repo, pr_number),
        "### Deploy Approval \u2014 Lifecycle",
        "",
        "**Status:** `succeeded`",
        f"**Bound HEAD:** `{_short(head_sha)}`",
        "",
        "Testing passed. The deployment is accepted.",
    ]
    lines.extend(_cos_evidence_block(cos_object_key, cos_bundle_sha256, cos_bundle_size, cos_download_url))
    lines.append(last_checked_line())
    return "\n".join(lines)


def build_test_failed_comment(
    repo: str, pr_number: int, head_sha: str,
    cos_object_key: str = "",
    cos_bundle_sha256: str = "",
    cos_bundle_size: int = 0,

) -> str:
    return failed_comment(
        repo, pr_number, head_sha,
        error="Testing failed. The deployment is not accepted.",
        cos_object_key=cos_object_key,
        cos_bundle_sha256=cos_bundle_sha256,
        cos_bundle_size=cos_bundle_size,
    )


def deploy_failed(
    repo: str, pr_number: int, head_sha: str,
    error: str = "",
    cos_object_key: str = "",
    cos_bundle_sha256: str = "",
    cos_bundle_size: int = 0,

) -> str:
    return failed_comment(
        repo, pr_number, head_sha,
        error=error or "Deployment execution failed.",
        cos_object_key=cos_object_key,
        cos_bundle_sha256=cos_bundle_sha256,
        cos_bundle_size=cos_bundle_size,
    )


def approve_deploy_occupied_comment(
    repo: str,
    pr_number: int,
    head_sha: str,
    machine_alias: str,
    occupied_components: list[dict],
    running_image_by_component: dict[str, str],
) -> str:
    lines = [
        BOT_MARKER,
        lifecycle_marker(repo, pr_number),
        "### Deploy Approval \u2014 Lifecycle",
        "",
        "**Status:** `deploy-requested`",
        f"**Bound HEAD:** `{_short(head_sha)}`",
        "",
        "CLEAN GATE: occupied runtime image.",
        "ZERO deployment was performed.",
        "",
    ]
    for comp in occupied_components:
        target = _escape(str(comp.get("target", "")))
        runtime_id = _escape(str(comp.get("runtime_id", "")))
        image_ref = running_image_by_component.get(str(comp.get("component_id", "")), "")
        compact = _compact_running_image(image_ref)
        if compact:
            lines.append(f"- `{target}` runtime `{runtime_id}` -> `{compact}`")
        else:
            lines.append(f"- `{target}` runtime `{runtime_id}` -> `occupied`")
    lines.extend([
        "",
        f"Clear `{_escape(machine_alias)}` manually, then send:",
        f"`/approve_deploy machine={machine_alias}`",
        "",
        "Re-read `running_image` on the next NEW approve.",
        "Controller does not perform stop/remove/cleanup.",
        "",
        last_checked_line(),
    ])
    return "\n".join(lines)


def approve_deploy_revoked_comment(
    repo: str,
    pr_number: int,
    head_sha: str,
    machine_alias: str,
) -> str:
    lines = [
        BOT_MARKER,
        lifecycle_marker(repo, pr_number),
        "### Deploy Approval \u2014 Lifecycle",
        "",
        "**Status:** `deploy-requested`",
        f"**Bound HEAD:** `{_short(head_sha)}`",
        "",
        "The approval command changed, was removed, or could not be revalidated "
        "before deployment.",
        "ZERO deployment was performed.",
        "",
        "Machine Owner must send a NEW:",
        "",
        f"`/approve_deploy machine={machine_alias}`",
        "",
        last_checked_line(),
    ]
    return "\n".join(lines)


def superseded_comment(repo: str, pr_number: int, old_head: str,
                       new_head: str) -> str:
    return "\n".join([
        BOT_MARKER,
        lifecycle_marker(repo, pr_number),
        "### Deploy Approval \u2014 Lifecycle",
        "",
        "**Status:** `review-required`",
        "",
        f"PR HEAD has changed: `{_short(old_head)}` \u2192 `{_short(new_head)}`",
        "The old deployment is no longer valid.",
        "",
        "**Next action \u2014 Developer**",
        "",
        "`/request_bot_review`",
        "",
        last_checked_line(),
    ])


def uncertain_comment(
    repo: str, pr_number: int, head_sha: str,
) -> str:
    """deploy-requested lifecycle with command.phase=uncertain."""
    return "\n".join([
        BOT_MARKER,
        lifecycle_marker(repo, pr_number),
        "### Deploy Approval \u2014 Lifecycle",
        "",
        "**Status:** `deploy-requested`",
        "**Command phase:** `uncertain`",
        f"**Bound HEAD:** `{_short(head_sha)}`",
        "",
        "Background polling keeps this command `uncertain`.",
        "- ZERO automatic replay",
        "- ZERO deployment",
        "- The old `/approve_deploy` comment will not be replayed.",
        "",
        "**Next action \u2014 Machine Owner**",
        "",
        "`/approve_deploy machine=<alias>`",
        "",
        "Only a NEW `/approve_deploy` starts recovery validation:",
        "- re-check the current full HEAD",
        "- refresh the validation / frozen Review Agent image snapshot",
        "- then run the running_image-only CLEAN GATE",
        "",
        last_checked_line(),
    ])


def deploy_status_comment(
    status: str, head_sha: str, repo: str, pr_number: int,
    components: list | None = None,
    deployments: list | None = None,
    cos_object_key: str = "",
    cos_bundle_sha256: str = "",
    cos_bundle_size: int = 0,
    cos_download_url: str = "",

) -> str:
    lines = [
        BOT_MARKER,
        lifecycle_marker(repo, pr_number),
        "### Deploy Approval \u2014 Lifecycle",
        "",
        f"**Status:** `{status}`",
        f"**Bound HEAD:** `{_short(head_sha)}`",
        "",
    ]
    if components:
        lines.append("Components:")
        for c in components:
            target = c.get("target", "")
            lines.append(f"- {_escape(target)}")
        lines.append("")
    if deployments:
        lines.append("Deployments:")
        for d in deployments:
            machine = d.get("machine", "")
            comps = ", ".join(d.get("component_ids", []))
            lines.append(f"- {_escape(machine)}: {comps}")
        lines.append("")
        lines.append("")
    lines.extend(_cos_evidence_block(
        cos_object_key, cos_bundle_sha256, cos_bundle_size, cos_download_url,
    ))
    lines.append(last_checked_line())
    return "\n".join(lines)


def deploy_help_text(topic: str = "") -> str:
    if topic == "request_deploy":
        return "\n".join([
            BOT_MARKER,
            "### Deploy Approval \u2014 Help: /request_deploy",
            "",
            "Request a deployment for the current PR HEAD.",
            "",
            "**Syntax:**",
            "`/request_deploy`",
            "",
            "**Parameters:** None.",
            "Binds the current PR HEAD's latest complete current-HEAD trusted Review Agent GitHub comment evidence and all deployable components.",
            "",
            "**Who can run:** PR Author only.",
            "**When:** PR must be open and not merged.",
        ])
    if topic == "approve_deploy":
        return "\n".join([
            BOT_MARKER,
            "### Deploy Approval \u2014 Help: /approve_deploy",
            "",
            "Approve a deployment and bind it to a machine.",
            "",
            "**Syntax:**",
            "`/approve_deploy machine=<machine-alias>`",
            "",
            "**Parameters:**",
            "- `machine=<alias>` \u2014 Machine alias, not IP/URL.",
            "",
            "**Who can run:** Machine Owner (machine owners[] or write/maintain/admin collaborator).",
            "**Note:** Clean gate reads `running_image` for every selected component "
            "before any deploy POST. If a runtime is occupied, zero deployment is "
            "performed and the owner must clean it manually, then send a NEW "
            "`/approve_deploy machine=<alias>`.",
        ])
    if topic == "record_test":
        return "\n".join([
            BOT_MARKER,
            "### Deploy Approval \u2014 Help: /record_test",
            "",
            "Record the overall validation result for the current PR.",
            "",
            "**Syntax:**",
            "`/record_test result=pass|fail [summary=\"...\"]`",
            "",
            "**Parameters:**",
            "- `result=pass|fail` \u2014 Required. Overall verdict.",
            "",
            "**Who can run:** Machine Owner or write/maintain/admin collaborator.",
            "**When:** Deployment status must be `testing`.",
            "**Effect:** `result=fail` sets the PR to `failed`.",
        ])
    return "\n".join([
        BOT_MARKER,
        "### Deploy Approval \u2014 Help",
        "",
        "**Developer commands:**",
        "- `/request_bot_review` \u2014 Trigger Review Agent for current HEAD",
        "- `/request_deploy` \u2014 Request deployment for current HEAD",
        "",
        "**Machine Owner commands:**",
        "- `/approve_deploy machine=<alias>` \u2014 Approve and bind machine",
        "- `/record_test result=pass|fail [summary=\"...\"]` \u2014 Record overall test result",
        "",
        "**Read-only commands:**",
        "- `/deploy_status` \u2014 Show deployment status",
        "- `/deploy_help [topic]` \u2014 Show this help",
        "",
        "**Detailed help:**",
        "- `/deploy_help request_deploy`",
        "- `/deploy_help approve_deploy`",
        "- `/deploy_help record_test`",
    ])
