"""Status comment rendering for a deployment (stateless GitHub persistence).

One lifecycle comment per PR. Uses a hidden marker for idempotent PATCH.
Each action-required state shows: status, bound head, who acts next, exact commands.
No obsolete lifecycle states, no build_index, no dpl_x.
"""

from __future__ import annotations

import datetime as _dt
import re
import html
import json
import ipaddress
from zoneinfo import ZoneInfo

import urllib.parse
from .models import BuildInfo

BOT_MARKER = "<!-- deploy-approval-agent -->"
_BEIJING = ZoneInfo("Asia/Shanghai")

_MAX_VISIBLE_LIFECYCLE_BYTES = 48 * 1024  # 48 KiB soft budget


def beijing_now_str() -> str:
    now = _dt.datetime.now(_BEIJING)
    return now.strftime("%Y-%m-%d %H:%M:%S")


def last_lifecycle_event_line(timestamp: str = "") -> str:
    """Last lifecycle event line. Only updated on meaningful business events.

    Args:
        timestamp: Timestamp string in "YYYY-MM-DD HH:MM:SS" format (Asia/Shanghai)
    Returns:
        "Last lifecycle event: YYYY-MM-DD HH:MM:SS (UTC+08:00, Asia/Shanghai)"
        If timestamp is empty, uses beijing_now_str().
    """
    if not timestamp:
        timestamp = beijing_now_str()
    return f"Last lifecycle event: {timestamp} (UTC+08:00, Asia/Shanghai)"


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
            return tag[:40] + "\u2026"
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


def _escape(text: str) -> str:
    """Escape untrusted text for Markdown cells, inline code and HTML."""
    value = str(text if text is not None else "")
    value = html.escape(value, quote=True).replace("\r", " ").replace("\n", " ")
    for char in "`|[]*_~\\":
        value = value.replace(char, f"&#{ord(char)};")
    return value


def _lifecycle_body(lines: list[str]) -> str:
    # The service owns the visible event block; this sentinel does not create a
    # second visible-history block when that writer preserves existing events.
    return "\n".join([*lines, "<!-- deploy-approval-history -->"])


def _deployment_table(deployments=None, components=None, machine_groups=None) -> str:
    snapshots = {c.get("component_id"): c for c in components or [] if isinstance(c, dict)}
    machines = {m.get("alias"): m.get("ip", "") for m in machine_groups or [] if isinstance(m, dict)}
    lines = ["| Machine | IP | Component | Variant/Path | Image | Result |",
             "|---|---|---|---|---|---|"]
    count = 0
    for dep in deployments or []:
        if not isinstance(dep, dict):
            continue
        local = dict(snapshots)
        snapshot = dep.get("components", dep.get("component_snapshot", []))
        if isinstance(snapshot, dict):
            snapshot = ([snapshot] if "component_id" in snapshot else
                        [dict(c, component_id=k) for k, c in snapshot.items() if isinstance(c, dict)])
        snapshot_ids = []
        for c in snapshot or []:
            if isinstance(c, dict):
                local[c.get("component_id")] = c
                if c.get("component_id"):
                    snapshot_ids.append(c["component_id"])
        machine = dep.get("machine", dep.get("alias", "?"))
        ip = dep.get("ip") or machines.get(machine) or "—"
        phase = dep.get("phase") or dep.get("result") or dep.get("status") or "unknown"
        result = str(phase)
        if dep.get("error"):
            result += ": " + str(dep["error"])
        if dep.get("updated_at") or dep.get("timestamp"):
            result += " · " + str(dep.get("updated_at") or dep.get("timestamp"))
        ids = dep.get("component_ids") or ([dep["component_id"]] if dep.get("component_id") else snapshot_ids or ["unknown"])
        for cid in ids:
            c = local.get(cid, dep if dep.get("component_id") == cid else {})
            values = (machine, ip, c.get("target") or cid,
                      c.get("variant") or c.get("driver_path") or "—",
                      c.get("image_ref") or dep.get("image_ref") or "—", result)
            lines.append("| " + " | ".join(_escape(v) for v in values) + " |")
            count += 1
    if count:
        return "\n".join(lines)
    return ("Deployment records exist; details unavailable." if deployments else
            "No deployment has been performed yet.")


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
    if any(ch.isspace() or ch in "<>\\`" for ch in url):
        return ""
    try:
        parsed = urllib.parse.urlsplit(url)
    except ValueError:
        return ""
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
        "| target | variant/path | build | deploy approval |",
        "|--------|--------------|-------|-----------------|",
    ]
    for b in builds:
        variant = _escape(b.variant or b.driver_path or chr(0x2014))
        build_status = "success" if b.success else "failed"
        eligibility = "deployable" if b.success and b.deployable else "unsupported"
        if not b.success:
            eligibility = "not deployable"
        lines.append(
            f"| {_escape(b.target)} | {variant} | {build_status} | {eligibility} |"
        )
    return "\n".join(lines)


def _cos_evidence_block(
    object_key: str = "",
    sha256: str = "",
    size: int = 0,
    download_url: str = "",
) -> list[str]:
    """Render the COS evidence block. Returns empty list if no evidence."""
    # A validated signed URL is intentionally visible to readers of this
    # Summary. Callers provide it only after verifying the COS object.
    safe_url = _safe_download_url(download_url)
    if (isinstance(object_key, str) and object_key and safe_url and isinstance(sha256, str)
            and re.fullmatch(r"[0-9a-f]{64}", sha256)
            and isinstance(size, int) and not isinstance(size, bool) and size > 0):
        return ["", f"[下载已验证归档文件（{_human_size(size)}）](<{safe_url}>)"]
    return []


def _human_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    else:
        return f"{size / (1024 * 1024):.1f} MB"


# ── Workflow helper ──────────────────────────────────────────────


def _workflow_lines(
    status: str, evidence_complete: bool = False,
    deployments_complete: bool | None = None,
    record_test_complete: bool = False,
) -> str:
    """Return the fixed 8-step workflow as markdown.

    Always forward order. [x] for complete, [ ] for pending.
    First unchecked step marked with \u2190 **Next**.
    If all checked, last step is marked done.
    """
    step_defs = [
        ("1. Request Review", 0, "review"),
        ("2. Review Agent completed", 1, "reviewed"),
        ("3. Request deployment", 2, "request"),
        ("4. Machine Owner approval(s)", 3, "approval"),
        ("5. All required components deployed", 4, "deployment"),
        ("6. Machine Owner uploads evidence", 4, "evidence"),
        ("7. Record validation result", 5, "result"),
        ("8. Deployment accepted", 6, "accepted"),
    ]

    status_levels = {
        "review-required": 0,
        "reviewing": 1,
        "deploy-ready": 2,
        "deploy-requested": 3,
        "testing": 5,
        "succeeded": 7,
        "failed": 5,
    }
    current_level = status_levels.get(status, 0)

    lines = []
    next_marked = False
    for label, required_level, step in step_defs:
        checked = current_level > required_level
        if status == "testing" and step == "deployment":
            checked = bool(deployments_complete)
        if status == "testing" and step == "evidence":
            checked = bool(evidence_complete)
        if status == "testing" and step == "result":
            checked = bool(record_test_complete)
        mark = "[x]" if checked else "[ ]"
        line = f"- {mark} {label}"
        if not checked and not next_marked:
            line += " \u2190 **Next**"
            next_marked = True
        lines.append(line)

    # If all steps are checked, no Next marker needed
    if not next_marked:
        return "\n".join(lines)

    return "\n".join(lines)


def _testing_next_action(evidence_complete: bool) -> str:
    if evidence_complete:
        return ("**Machine Owner**\n\n"
                '`/record_test result=pass|fail [summary="..."]`')
    return ("**PR developer and deployed Machine Owner**\n\n"
            "Upload evidence for this HEAD and round first; after it is archived, "
            'record the result with `/record_test result=pass|fail [summary="..."]`.')


# ── History helpers ──────────────────────────────────────────────


def _archives_link_block(archives: list[dict]) -> str:
    """Render archived history links."""
    if not archives:
        return ""

    lines = ["### Archived history", ""]
    for arch in archives:
        page = arch.get("page", 1)
        url = arch.get("url", "")
        if url:
            lines.append(f"- [History Archive Page {page}]({url})")
        else:
            lines.append(f"- History Archive Page {page}")
    lines.append("")
    return "\n".join(lines)


# ── Lifecycle comment builders ───────────────────────────────────


def review_required(
    repo: str, pr_number: int, head_sha: str, last_lifecycle_event: str = "",
) -> str:
    """status: review-required \u2014 initial comment with full 8-step workflow."""
    return _lifecycle_body([
        BOT_MARKER,
        lifecycle_marker(repo, pr_number),
        "### Deploy Approval \u2014 Lifecycle",
        "",
        "**Status:** `review-required`",
        f"**Bound HEAD:** `{_short(head_sha)}`",
        f"**Repository:** `{repo}`",
        "",
        "### Workflow",
        "",
        _workflow_lines("review-required"),
        "",
        "### Next action",
        "",
        "**Developer**",
        "",
        "`/request_bot_review`",
        "",
        "### Current Deployment",
        "",
        "No deployment has been requested yet.",
        "",
        last_lifecycle_event_line(last_lifecycle_event),
    ])


def reviewing(
    repo: str, pr_number: int, head_sha: str, last_lifecycle_event: str = "",
) -> str:
    return _lifecycle_body([
        BOT_MARKER,
        lifecycle_marker(repo, pr_number),
        "### Deploy Approval \u2014 Lifecycle",
        "",
        "**Status:** `reviewing`",
        f"**Bound HEAD:** `{_short(head_sha)}`",
        f"**Repository:** `{repo}`",
        "",
        "### Workflow",
        "",
        _workflow_lines("reviewing"),
        "",
        "### Next action",
        "",
        "**Waiting for Review Agent**",
        "",
        "No action required.",
        "",
        last_lifecycle_event_line(last_lifecycle_event),
    ])


def deploy_ready(
    repo: str, pr_number: int, head_sha: str,
    builds: list[BuildInfo], last_lifecycle_event: str = "",
    review_evidence: dict | None = None,
) -> str:
    lines = [
        BOT_MARKER,
        lifecycle_marker(repo, pr_number),
        "### Deploy Approval \u2014 Lifecycle",
        "",
        "**Status:** `deploy-ready`",
        f"**Bound HEAD:** `{_short(head_sha)}`",
        f"**Repository:** `{repo}`",
        "",
        "### Workflow",
        "",
        _workflow_lines("deploy-ready"),
        "",
        "### Next action",
        "",
        "**Developer**",
        "",
        "`/request_deploy`",
        "",
        "### Review evidence",
        "",
        f"- **Build Result:** {'success' if builds and all(b.success for b in builds) else 'failed'}",
    ]
    evidence = review_evidence or {}
    if evidence.get("test_comment_id", 0) == 0:
        lines.append("- **Tests:** not run")
    elif evidence.get("test_skipped", False):
        lines.append("- **Tests:** skipped")
    else:
        lines.append(
            f"- **Tests:** {int(evidence.get('test_passed', 0))} passed / "
            f"{int(evidence.get('test_failed', 0))} failed"
        )
    lines.extend([
        "- **Code Review:** findings available — manual review required",
        "",
        "> Deploy Approval does not interpret Review Agent findings; "
        "the developer/owner must read the Review Agent review comment. "
        "The developer's `/request_deploy` confirms this manual review; "
        "machine owner/collaborator approval is still required before deployment.",
        "",
        "### Components",
        "",
        _build_table(builds),
        "",
        last_lifecycle_event_line(last_lifecycle_event),
    ])
    return _lifecycle_body(lines)


def deploy_requested(
    repo: str, pr_number: int, head_sha: str,
    components: list[dict],
    machine_groups: list[dict],
    gate_note: list[str] | None = None,
    deployments: list[dict] | None = None,
    last_lifecycle_event: str = "",
    all_components: list[dict] | None = None,
    component_snapshot: list[dict] | None = None,
    attempt: dict | None = None,
) -> str:
    """status: deploy-requested with IP table, workflow, deployment, history."""
    # Build component labels
    comp_lines = []
    for c in components:
        if isinstance(c, dict):
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
            extra = f" \u2014 `{display}`" if display else ""
            comp_lines.append(f"- {_escape(label)}{extra}")

    # Build machine table with IP
    table_lines = [
        "| Machine | IP | Components |",
        "|---|---|---|",
    ]
    for mg in machine_groups:
        if not isinstance(mg, dict):
            continue
        alias = mg.get("alias", "?")
        ip = mg.get("ip", "")
        remaining_ids = {c.get("component_id") for c in components if isinstance(c, dict)}
        cids = [cid for cid in mg.get("component_ids", []) if cid in remaining_ids]
        if not cids:
            continue
        comp_labels = []
        for cid in cids:
            for c in components if isinstance(components, list) else []:
                if isinstance(c, dict) and c.get("component_id") == cid:
                    target = c.get("target", "")
                    variant = c.get("variant", "") or ""
                    driver_path = c.get("driver_path", "") or ""
                    lbl = target
                    if target == "driver" and driver_path:
                        lbl = f"driver {driver_path}"
                    elif variant:
                        lbl = f"{target} {variant}"
                    comp_labels.append(lbl)
                    break
            if not any(
                isinstance(c, dict) and c.get("component_id") == cid
                for c in (components or [])
            ):
                comp_labels.append(cid)
        ip_str = f"`{_escape(ip)}`" if ip else "`\u2014`"
        comps_str = (
            ", ".join(f"`{_escape(l)}`" for l in comp_labels) if comp_labels else "`none`"
        )
        table_lines.append(f"| `{_escape(alias)}` | {ip_str} | {comps_str} |")

    lines = [
        BOT_MARKER,
        lifecycle_marker(repo, pr_number),
        "### Deploy Approval \u2014 Lifecycle",
        "",
        "**Status:** `deploy-requested`",
        f"**Bound HEAD:** `{_short(head_sha)}`",
        f"**Repository:** `{repo}`",
    ]
    if last_lifecycle_event:
        lines.append(f"**Last lifecycle event:** {last_lifecycle_event} (UTC+08:00, Asia/Shanghai)")
    lines.extend([
        "",
        "### Workflow",
        "",
        _workflow_lines("deploy-requested"),
        "",
        "### Next action",
        "",
        "**Machine Owner**",
        "",
        "`/approve_deploy machine=<alias-or-ip>`",
        "",
        "Machine Owner may use either the machine alias or the listed IP address.",
        "",
        "### Compatible machines for remaining components",
        "",
    ])

    if len(table_lines) > 2:
        lines.extend(table_lines)
    else:
        lines.extend(["No compatible machines found.", ""])

    if gate_note:
        lines.extend(["", *gate_note])

    # Current Deployment section
    lines.extend(["", "### Current Deployment", ""])
    lines.append(_deployment_table(
        deployments, all_components or component_snapshot or components, machine_groups,
    ))
    lines.extend(_attempt_lines(attempt))

    lines.extend([
        "",
        last_lifecycle_event_line(last_lifecycle_event),
    ])

    return _lifecycle_body(lines)


def testing(
    repo: str, pr_number: int, head_sha: str,
    deployment_id: str = "", case_result: str = "",
    last_lifecycle_event: str = "",
    deployments: list[dict] | None = None,
    components: list[dict] | None = None,
    machine_groups: list[dict] | None = None,
    attempt: dict | None = None,
    evidence_complete: bool = False,
    deployments_complete: bool | None = None,
    record_test_complete: bool = False,
) -> str:
    lines = [
        BOT_MARKER,
        lifecycle_marker(repo, pr_number),
        "### Deploy Approval \u2014 Lifecycle",
        "",
        "**Status:** `testing`",
        f"**Bound HEAD:** `{_short(head_sha)}`",
        f"**Repository:** `{repo}`",
        "",
        "### Workflow",
        "",
        _workflow_lines("testing", evidence_complete=evidence_complete,
                        deployments_complete=deployments_complete,
                        record_test_complete=record_test_complete),
        "",
        "### Next action",
        "",
        *_testing_next_action(evidence_complete).split("\n"),
        "",
        "### Current Deployment",
    ]
    if deployments:
        lines.extend(["", _deployment_table(deployments, components, machine_groups)])
    else:
        lines.extend(["", "No deployment record is available for this round yet."])
    if case_result:
        lines.extend([
            "",
            f"**Automated case results:** {case_result}",
            "Fixed Case results are advisory only.",
        ])
    lines.extend([
        "",
        last_lifecycle_event_line(last_lifecycle_event),
    ])
    lines.extend(_attempt_lines(attempt))
    return _lifecycle_body(lines)


def succeeded_comment(
    repo: str, pr_number: int, head_sha: str,
    cos_object_key: str = "", cos_bundle_sha256: str = "",
    cos_bundle_size: int = 0, cos_download_url: str = "",
    deployments: list[dict] | None = None,
    components: list[dict] | None = None,
    machine_groups: list[dict] | None = None,
    last_lifecycle_event: str = "",
) -> str:
    lines = [
        BOT_MARKER,
        lifecycle_marker(repo, pr_number),
        "### Deploy Approval \u2014 Lifecycle",
        "",
        "**Status:** \u2705 `succeeded`",
        f"**Bound HEAD:** `{_short(head_sha)}`",
        f"**Repository:** `{repo}`",
        "",
        "### Workflow",
        "",
        _workflow_lines("succeeded"),
        "",
        "### Next action",
        "",
        "**None. Deployment lifecycle is complete.**",
        "",
        "### Current Deployment",
        "",
    ]
    lines.append(_deployment_table(deployments, components, machine_groups)
                 if deployments else "Deployment completed; deployment records unavailable.")

    lines.extend(_cos_evidence_block(cos_object_key, cos_bundle_sha256, cos_bundle_size, cos_download_url))
    lines.extend([
        "",
        last_lifecycle_event_line(last_lifecycle_event),
    ])
    return _lifecycle_body(lines)


def failed_comment(
    repo: str, pr_number: int, head_sha: str,
    error: str = "", cos_object_key: str = "", cos_bundle_sha256: str = "",
    cos_bundle_size: int = 0, cos_download_url: str = "",
    last_lifecycle_event: str = "",
    deployments: list[dict] | None = None,
    components: list[dict] | None = None,
    machine_groups: list[dict] | None = None,
    attempt: dict | None = None,
) -> str:
    lines = [
        BOT_MARKER,
        lifecycle_marker(repo, pr_number),
        "### Deploy Approval \u2014 Lifecycle",
        "",
        "**Status:** \u274c `failed`",
        f"**Bound HEAD:** `{_short(head_sha)}`",
        f"**Repository:** `{repo}`",
        "",
        "### Workflow",
        "",
        _workflow_lines("failed"),
        "",
        "### Next action",
        "",
        "**None. Deployment lifecycle has failed.**",
        "",
        "The deployment is not accepted.",
    ]
    if error:
        lines.append(f"**Error:** {_escape(error)}")
    lines.extend(_cos_evidence_block(cos_object_key, cos_bundle_sha256, cos_bundle_size, cos_download_url))
    lines.extend([
        "",
        last_lifecycle_event_line(last_lifecycle_event),
    ])
    lines.extend(_attempt_lines(attempt))
    if deployments:
        lines.extend(["### Deployment records", _deployment_table(deployments, components, machine_groups)])
    return _lifecycle_body(lines)


def superseded_comment(
    repo: str, pr_number: int, old_head: str, new_head: str,
    last_lifecycle_event: str = "",
) -> str:
    return _lifecycle_body([
        BOT_MARKER,
        lifecycle_marker(repo, pr_number),
        "### Deploy Approval \u2014 Lifecycle",
        "",
        "**Status:** `review-required`",
        f"**Bound HEAD:** `{_short(new_head)}`",
        f"**Repository:** `{repo}`",
        "",
        "### Workflow",
        "",
        _workflow_lines("review-required"),
        "",
        f"PR HEAD has changed: `{_short(old_head)}` \u2192 `{_short(new_head)}`",
        "The old deployment is no longer valid.",
        "",
        "### Next action",
        "",
        "**Developer**",
        "",
        "`/request_bot_review`",
        "",
        last_lifecycle_event_line(last_lifecycle_event),
    ])


def uncertain_comment(
    repo: str, pr_number: int, head_sha: str,
    last_lifecycle_event: str = "",
    deployments: list[dict] | None = None,
    components: list[dict] | None = None,
    machine_groups: list[dict] | None = None,
    attempt: dict | None = None,
) -> str:
    lines = [
        BOT_MARKER,
        lifecycle_marker(repo, pr_number),
        "### Deploy Approval \u2014 Lifecycle",
        "",
        "**Status:** `deploy-requested`",
        "**Command phase:** `uncertain`",
        f"**Bound HEAD:** `{_short(head_sha)}`",
        f"**Repository:** `{repo}`",
        "",
        "### Workflow",
        "",
        _workflow_lines("deploy-requested"),
        "",
        "### Next action",
        "",
        "**Machine Owner**",
        "",
        "`/approve_deploy machine=<alias-or-ip>`",
        "",
        "Only a NEW `/approve_deploy` starts recovery validation.",
        "",
        last_lifecycle_event_line(last_lifecycle_event),
    ]
    lines.extend(_attempt_lines(attempt))
    if deployments:
        lines.extend(["### Current Deployment", _deployment_table(deployments, components, machine_groups)])
    return _lifecycle_body(lines)


def deploy_status_comment(
    status: str, head_sha: str, repo: str, pr_number: int,
    components: list | None = None,
    deployments: list | None = None,
    cos_object_key: str = "", cos_bundle_sha256: str = "",
    cos_bundle_size: int = 0, cos_download_url: str = "",
    last_lifecycle_event: str = "",
    machine_groups: list[dict] | None = None,
    attempt: dict | None = None,
) -> str:
    lines = [
        BOT_MARKER,
        lifecycle_marker(repo, pr_number),
        "### Deploy Approval \u2014 Lifecycle",
        "",
        f"**Status:** `{status}`",
        f"**Bound HEAD:** `{_short(head_sha)}`",
        f"**Repository:** `{repo}`",
        "",
        "### Workflow",
        "",
        _workflow_lines(status),
        "",
    ]
    if components:
        lines.append("Components:")
        for c in components:
            target = c.get("target", "")
            lines.append(f"- {_escape(target)}")
        lines.append("")
    lines.extend(["### Current Deployment", "", _deployment_table(deployments, components, machine_groups), ""])
    lines.extend(_attempt_lines(attempt))
    lines.extend(_cos_evidence_block(
        cos_object_key, cos_bundle_sha256, cos_bundle_size, cos_download_url,
    ))
    lines.extend([
        "",
        last_lifecycle_event_line(last_lifecycle_event),
    ])
    return _lifecycle_body(lines)


def deploying_comment(
    repo: str, pr_number: int, head_sha: str,
    deployments: list[dict] | None = None,
    components: list[dict] | None = None,
    machine_groups: list[dict] | None = None,
    attempt: dict | None = None,
    last_lifecycle_event: str = "",
) -> str:
    """Full lifecycle body for the service's transient executing phase."""
    details = dict(attempt or {})
    details.setdefault("phase", "executing")
    return deploy_status_comment(
        "deploy-requested", head_sha, repo, pr_number,
        components=components, deployments=deployments,
        machine_groups=machine_groups, attempt=details,
        last_lifecycle_event=last_lifecycle_event,
    )


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
            "Binds the current PR HEAD\'s latest complete current-HEAD trusted Review Agent GitHub comment evidence and all deployable components.",
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
            "`/approve_deploy machine=<alias-or-ip>`",
            "",
            "**Parameters:**",
            "- `machine=<alias-or-ip>` \u2014 Machine alias or unique IPv4 address.",
            "",
            "**Who can run:** Machine Owner (machine owners[] or write/maintain/admin collaborator).",
            "**Note:** Deploy Approval reads `running_image` as evidence only before "
            "each deploy POST. Agent Core handles container replacement. "
            "If a runtime is already running, Agent Core will perform an in-place "
            "upgrade or idempotent no-op as appropriate.",
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
            "- `result=pass` — Validation succeeded; all required checks passed.",
            "- `result=fail` — Validation failed; the PR lifecycle becomes `failed`.",
            "- `summary=\"...\"` — Optional short explanation; never include credentials or signed URLs.",
            "",
            "**Examples:**",
            "- `/record_test result=pass summary=\"Video preview and COS SHA256 verified\"` — success.",
            "- `/record_test result=fail summary=\"MP4 preview returned an error\"` — failure.",
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
        "- `/approve_deploy machine=<alias-or-ip>` \u2014 Approve and bind machine",
        "- `/record_test result=pass|fail [summary=\"...\"]` \u2014 Record overall test result",
        "",
        "**Read-only commands:**",
        "- Verified COS evidence is available from the download links in this PR Summary.",
        "- `/deploy_status` \u2014 Show deployment status",
        "- `/deploy_help [topic]` \u2014 Show this help",
        "",
        "**Detailed help:**",
        "- `/deploy_help request_deploy`",
        "- `/deploy_help approve_deploy`",
        "- `/deploy_help record_test`",
    ])


def _attempt_lines(attempt: dict | None) -> list[str]:
    if not attempt:
        return []
    values = {
        "Machine": attempt.get("machine") or attempt.get("alias") or "unknown",
        "IP": attempt.get("ip") or "unknown",
        "Components": attempt.get("component_ids") or attempt.get("components") or attempt.get("args", {}).get("pending_component_ids") or "unknown",
        "Phase": attempt.get("phase") or "unknown",
        "Time": attempt.get("timestamp") or attempt.get("updated_at") or attempt.get("started_at") or "unknown",
        "Error": attempt.get("error") or "—",
    }
    return ["", "### Deployment attempt", "", *[
        f"- **{label}:** {_escape(value)}" for label, value in values.items()
    ]]


def _public_base(value: str) -> str:
    """Accept HTTPS origins and HTTP origins on internal hosts only."""
    if not isinstance(value, str) or any(ord(c) < 33 for c in value):
        return ""
    try:
        parsed = urllib.parse.urlsplit(value)
        if (parsed.scheme not in {"https", "http"} or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in ("", "/") or parsed.netloc.endswith(":")
                or any(c.isspace() or ord(c) < 33 or ord(c) == 127 for c in parsed.netloc)
                or any(c in value for c in '<>"\\')):
            return ""
        if parsed.port is not None and not 1 <= parsed.port <= 65535:
            return ""
        if parsed.scheme == "http":
            host = parsed.hostname.lower()
            if host != "localhost" and not host.endswith(".localhost"):
                try:
                    address = ipaddress.ip_address(host)
                except ValueError:
                    return ""
                if (address.is_unspecified or address.is_multicast or not
                        (address.is_private or address.is_loopback or address.is_link_local)):
                    return ""
        return value.rstrip("/")
    except (TypeError, ValueError):
        return ""


def _evidence_path(repo: str, pr_number: int) -> str:
    parts = repo.split("/")
    if len(parts) != 2 or not all(re.fullmatch(r"[A-Za-z0-9_.-]+", p) and p not in {'.', '..'} for p in parts):
        return ""
    if type(pr_number) is not int or pr_number <= 0:
        return ""
    return f"/api/evidence/{parts[0]}/{parts[1]}/pulls/{pr_number}"


def _trusted_archive_url(url: str, repo: str, pr_number: int, base: str) -> str:
    """Archive input is from trusted discovery, never arbitrary record URLs."""
    try:
        parsed = urllib.parse.urlsplit(url)
        if parsed.username or parsed.password or any(c in url for c in '\r\n<>"\\'):
            return ""
        prefix = f"/{repo}/"
        if (parsed.scheme == 'https' and parsed.netloc == 'github.com'
                and parsed.path in {prefix + f'issues/{pr_number}', prefix + f'pull/{pr_number}'}
                and re.fullmatch(r'issuecomment-[0-9]+', parsed.fragment) and not parsed.query):
            return url
        if base and url.startswith(base + _evidence_path(repo, pr_number) + '/'):
            return url if not parsed.query and not parsed.fragment else ""
    except (ValueError, TypeError):
        pass
    return ""


def _trusted_github_attachment_url(item: dict) -> str:
    """Return only persisted GitHub-native attachment URLs with trusted provenance."""
    comment_id = item.get("source_comment_id")
    if (type(comment_id) is int and comment_id > 0):
        valid_comment = True
    elif isinstance(comment_id, str) and re.fullmatch(r"[1-9][0-9]{0,19}", comment_id):
        valid_comment = True
    else:
        valid_comment = False
    if not valid_comment or item.get("actor_verified") is not True or item.get("status") != "ready":
        return ""
    value = item.get("source_url")
    if not isinstance(value, str) or len(value) > 2048 or any(c in value for c in '\r\n<>"\\'):
        return ""
    try:
        parsed = urllib.parse.urlsplit(value)
    except (TypeError, ValueError):
        return ""
    if (parsed.scheme != "https" or parsed.netloc != "github.com"
            or parsed.query or parsed.fragment
            or not re.fullmatch(
                r"/user-attachments/(?:assets/[A-Za-z0-9_-]{1,128}|files/[A-Za-z0-9_-]{1,128}/[A-Za-z0-9._~-]+)",
                parsed.path)):
        return ""
    return value


def render_evidence_section(repo, pr_number, state, records, archive_links, *, public_base_url="") -> str:
    """Render current evidence and three archived rounds without mutating history.

    Records may be attachment rows, round bundles, or archive_snapshot entries.
    Archive links must come from trusted server-side discovery. COS download
    URLs are used only when supplied on a verified ready attachment row.
    The caller supplies evidence_enabled=True in the render state only when
    the configured feature is enabled; absent flags fail closed for API links.
    """
    from .evidence_models import round_id_for_state

    state = state or {}
    if state.get("status") not in {"testing", "succeeded", "failed"}:
        return "<!-- deploy-approval-evidence:start -->\n<!-- deploy-approval-evidence:end -->"
    head = state.get("head_sha", "")
    try:
        current_round = round_id_for_state(state)
    except ValueError:
        current_round = state.get("round_id", "")
    base = _public_base(public_base_url)
    if not base.startswith("https://"):
        base = ""
    enabled = (state.get("evidence_enabled", state.get("enabled", False)) is True
               and state.get("enabled", True) is not False)
    api = _evidence_path(repo, pr_number)
    if not enabled or not api:
        base = ""
    groups = {(head, current_round): dict(
        state, attachments=list(state.get("attachments") or []),
        attachment_details_complete=isinstance(state.get("attachments"), list),
    )}
    if isinstance(records, dict):
        records = records.get("rounds", records.get("records", records.get("attachments", [])))
    archive_links = list(archive_links or [])
    entries = list(records or []) + [a for a in archive_links if isinstance(a, dict) and "snapshot" in a]
    for row in entries:
        if not isinstance(row, dict):
            continue
        if isinstance(row.get("data"), dict):
            row = row["data"]
        snapshot = row.get("snapshot", row)
        h = row.get("head_sha") or snapshot.get("head_sha") or "unknown"
        r = row.get("round_id") or snapshot.get("round_id") or "unknown"
        key = (h, r)
        group = groups.setdefault(key, {"attachments": []})
        if "snapshot" in row or "attachments" in row or "deployments" in row:
            attachments = list(group.get("attachments") or [])
            if key != (head, current_round):
                capability = group.get("capability")
                group.update(snapshot)
                group.update({k: v for k, v in row.items() if k != "snapshot"})
                if not group.get("capability") and capability:
                    group["capability"] = capability
            group["attachments"] = attachments + list(snapshot.get("attachments") or [])
            if row is not snapshot:
                group["attachments"].extend(row.get("attachments") or [])
            if isinstance(snapshot.get("attachments"), list):
                group["attachment_details_complete"] = True
            if isinstance(row.get("attachments"), list):
                group["attachment_details_complete"] = True
            for record in row.get("attachment_records", []):
                if isinstance(record, dict) and isinstance(record.get("data"), dict):
                    group["attachments"].append(record["data"])
                    group["attachment_details_complete"] = True
        else:
            group.setdefault("attachments", []).append(row)
            group["attachment_details_complete"] = True

    page_path = api.replace("/api/evidence/", "/evidence/", 1)

    def page_url(capability):
        if not base or not capability or not page_path.startswith("/evidence/"):
            return ""
        return (base + page_path + "?cap="
                + urllib.parse.quote(str(capability), safe=""))

    def latest_attachments(group):
        latest_items = {}
        for offset, item in enumerate(group.get("attachments", [])):
            if not isinstance(item, dict):
                continue
            identity = item.get("attachment_id") or item.get("id") or f"record-{offset}"
            previous = latest_items.get(identity)
            # Immutable archives may contain earlier revisions after the live
            # tombstone. Deletion always wins over an old ready revision.
            rank = lambda row: 4 if row.get("deleted_at") or row.get("status") == "deleted" else 3 if row.get("status") == "ready" else 1
            if previous is None or rank(item) >= rank(previous):
                latest_items[identity] = item
        return list(latest_items.values())

    def is_internal_test_note(item):
        if not isinstance(item, dict) or item.get("status") != "text":
            return False
        actor = item.get("actor_login") or item.get("actor") or item.get("uploader")
        if not isinstance(actor, str) or actor.casefold() != "4pd-pr-assistant[bot]":
            return False
        note = item.get("note") or item.get("text") or item.get("filename")
        if not isinstance(note, str):
            return False
        normalized = " ".join(note.split()).casefold()
        return normalized in {
            "server-side cos multipart acceptance",
            "server-side 20mb h.264 cos transport acceptance",
        }

    def visible_attachments(group):
        return [item for item in latest_attachments(group) if not is_internal_test_note(item)]

    def parsed_timestamp(value):
        if not isinstance(value, str) or not value.strip():
            return None
        try:
            return _dt.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None

    def summary_time(value):
        parsed = parsed_timestamp(value)
        if parsed is None:
            return "时间未知"
        if parsed.tzinfo is not None and parsed.utcoffset() is not None:
            return parsed.astimezone(_BEIJING).strftime("%Y-%m-%d %H:%M") + "（北京时间）"
        # Preserve naive historical wall time without assigning it a timezone.
        return parsed.strftime("%Y-%m-%d %H:%M")

    def sortable_time(value):
        parsed = parsed_timestamp(value)
        if parsed is None:
            return (0, "")
        if parsed.tzinfo is not None and parsed.utcoffset() is not None:
            return (2, parsed.astimezone(_dt.timezone.utc).timestamp())
        return (1, parsed.isoformat())

    def summary_date(value):
        parsed = parsed_timestamp(value)
        if parsed is None:
            return "时间未知"
        if parsed.tzinfo is not None and parsed.utcoffset() is not None:
            parsed = parsed.astimezone(_BEIJING)
        return parsed.date().isoformat()

    def first_valid_time(*values):
        for value in values:
            if parsed_timestamp(value) is not None:
                return value
        return None

    def render_round(key, group, *, historical=False, round_number=1):
        h, r = key
        items = visible_attachments(group)
        items.sort(key=lambda item: sortable_time(first_valid_time(
            item.get("created_at"), item.get("uploaded_at"))), reverse=True)
        lines = []
        if historical:
            historical_time = first_valid_time(
                group.get("archived_at"), group.get("ended_at"),
                group.get("round_started_at"), group.get("started_at"),
                group.get("created_at"),
            )
            if historical_time:
                lines.append(f"**时间：** {_escape(summary_time(historical_time))}")
            lines.extend([
                f"**状态：** {_escape(group.get('status') if isinstance(group.get('status'), str) else '未知')}", "",
            ])
        if items:
            if historical:
                lines.extend([f"### 第 {round_number} 轮", ""])
            lines.extend(["| 文件 | 大小 | 上传者 | 上传时间 |", "|---|---:|---|---|"])
        note_blocks = []
        for item in items:
            if not isinstance(item, dict):
                continue
            deleted = bool(item.get("deleted_at") or item.get("status") == "deleted")
            status = "deleted" if deleted else item.get("status", "pending")
            filename = item.get("filename")
            if (not isinstance(filename, str) or not filename.strip()
                    or re.fullmatch(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", filename.strip())):
                filename = "附件说明" if status == "text" else "附件文件"
            else:
                filename = filename.strip()
            size = item.get("size", "—")
            if isinstance(size, int) and not isinstance(size, bool) and size >= 0:
                size = _human_size(size)
            if item.get("actor_verified") is True and item.get("upload_method") != "signed_invite":
                actor = item.get("actor_login") or item.get("actor") or "已验证用户"
            elif item.get("upload_method") == "signed_invite":
                actor = "分享链接（身份未验证）"
            else:
                actor = "身份未验证"
            uploaded_at = summary_time(first_valid_time(item.get("created_at"), item.get("uploaded_at")))
            if deleted:
                filename = filename + "（已删除）"
            lines.append("| " + " | ".join(_escape(value) for value in
                (filename, size, actor, uploaded_at)) + " |")
            note = item.get("note") or (item.get("text") if status == "text" else "")
            if isinstance(note, str) and note.strip():
                note_blocks.extend([
                    "<details>",
                    f"<summary>{_escape(filename)} · 备注</summary>",
                    "",
                    _escape(note.strip()),
                    "",
                    "</details>",
                    "",
                ])
        lines.extend(note_blocks)
        if not items:
            if historical:
                lines.extend([f"### 第 {round_number} 轮", "",
                              "附件明细请查看历史归档" if not group.get("attachment_details_complete")
                              else "该轮未保存附件"])
            else:
                lines.append("本轮暂无附件，上传后会显示")
        if group.get("deployments"):
            lines.extend(["", "### 部署记录", "",
                          _deployment_table(group["deployments"], group.get("components"), group.get("machine_groups"))])
        test_result = group.get("test_result")
        if isinstance(test_result, str) and test_result in {"pass", "fail"}:
            lines.extend(["", "**测试结果：** " + ("通过" if test_result == "pass" else "未通过")])
        case_results = group.get("case_results")
        if isinstance(case_results, dict) and case_results:
            labels = {"pass": "通过", "fail": "失败", "running": "运行中", "n/a": "不适用"}
            counts = {}
            for value in case_results.values():
                if not isinstance(value, str) or value not in labels:
                    continue
                label = labels[value]
                counts[label] = counts.get(label, 0) + 1
            if counts:
                lines.extend(["", "**自动用例：** " + " · ".join(
                    f"{_escape(label)} {count}" for label, count in counts.items())])
        summary = group.get("summary")
        if isinstance(summary, str) and summary.strip():
            summary_text = summary.strip()
            try:
                json.loads(summary_text)
                summary_text = ""
            except (TypeError, ValueError):
                pass
            if summary_text:
                lines.extend(["", "**摘要：** " + _escape(summary_text[:500])])
        history = group.get("history")
        if isinstance(history, (list, dict)) and history:
            history_count = len(history)
            lines.extend(["", f"历史事件：{history_count} 条（详情保留在归档记录中）"])
        log = _safe_download_url(group.get("cos_log_download_url", ""))
        cos_metadata = group.get("cos") if isinstance(group.get("cos"), dict) else {}
        if log and (cos_metadata.get("object_key") or group.get("log_available") or group.get("log_ready")):
            lines.extend(["", f"[下载本轮归档日志](<{log}>)"])
        return lines

    round_keys = list(groups.keys())
    def round_sort_time(key):
        group = groups[key]
        value = first_valid_time(group.get("round_started_at"), group.get("started_at"), group.get("created_at"), group.get("archived_at"), group.get("ended_at"))
        return sortable_time(value)
    ordered_rounds = sorted(round_keys, key=round_sort_time)
    round_numbers = {key: index + 1 for index, key in enumerate(ordered_rounds)}
    current_round_number = round_numbers.get((head, current_round), max(1, len(round_numbers)))
    lines = ["<!-- deploy-approval-evidence:start -->"]
    lines.extend(render_round((head, current_round), groups[(head, current_round)], round_number=current_round_number))
    round_pattern = re.escape(str(head)) + r"-[0-9]+"
    current_context_valid = (
        state.get("pr_state") == "open" and not state.get("merged")
        and re.fullmatch(r"[0-9a-f]{40}", str(head))
        and isinstance(current_round, str) and re.fullmatch(round_pattern, current_round)
        and state.get("evidence_enabled") is True and bool(api)
    )
    if current_context_valid:
        upload_url = (page_url(state.get("upload_capability"))
                      if state.get("status") == "testing" else "")
        read_url = page_url(state.get("capability"))
        if upload_url:
            lines.extend(["", f"[打开附件页](<{upload_url}>)"])
        elif read_url:
            lines.extend(["", f"[打开附件页](<{read_url}>)"])
    if current_context_valid and state.get("status") == "testing" and page_url(state.get("upload_capability")):
        pending = sum(1 for item in state.get("attachments", [])
                      if isinstance(item, dict)
                      and item.get("status") not in {"ready", "text", "deleted", "invalid"})
        archived = sum(1 for item in state.get("attachments", [])
                       if isinstance(item, dict) and item.get("status") == "ready"
                       and item.get("cos_projection_verified") is True
                       and item.get("object_key") and isinstance(item.get("sha256"), str)
                       and re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
                       and item.get("head_sha") == head and item.get("round_id") == current_round
                       and ((item.get("actor_verified") is True
                             and item.get("upload_method") != "signed_invite")
                            or item.get("upload_method") == "signed_invite"
                            and item.get("upload_link_authorized") is True
                            and item.get("actor_verified") is False
                            and item.get("github_user_authenticated") is False
                            and item.get("cos_content_verified") is True))
        lines.extend([
            "上传权限：仅限本轮测试邀请；分享链接上传会标示身份未验证。",
            "可上传视频、图片、文件或文字说明；单文件 512 MiB；本轮最多 20 个文件、总计 10 GiB。",
            f"当前状态：{pending} 个待处理 · {archived} 个已归档并完成 COS SHA256 校验。归档后再记录 `/record_test result=pass|fail`。",
        ])
    old = [(key, group) for key, group in groups.items() if key != (head, current_round)]
    old.sort(key=lambda pair: round_sort_time(pair[0]), reverse=True)
    if old:
        lines.extend(["", "### 历史轮次", ""])
        for key, group in old[:3]:
            timestamp = first_valid_time(
                group.get("archived_at"), group.get("ended_at"),
                group.get("round_started_at"), group.get("started_at"),
                group.get("created_at"),
            )
            attachment_count = sum(
                1 for item in visible_attachments(group)
                if not (item.get("deleted_at") or item.get("status") == "deleted")
            )
            count_label = (f"{attachment_count} 个附件" if attachment_count else
                           "0 个附件" if group.get("attachment_details_complete")
                           else "附件明细请查看历史归档")
            date_label = summary_date(timestamp)
            if date_label == "时间未知":
                date_label = "历史归档"
            lines.extend(["<details>",
                          f"<summary>第 {round_numbers.get(key, 1)} 轮 · {_escape(date_label)} · {_escape(count_label)}</summary>",
                          "", *render_round(key, group, historical=True, round_number=round_numbers.get(key, 1)), "", "</details>", ""])
    linked_index = False
    seen_archive_urls = set()
    for link in archive_links or []:
        if isinstance(link, dict):
            url = _trusted_archive_url(link.get("url", ""), repo, pr_number, base)
            if url and url not in seen_archive_urls:
                seen_archive_urls.add(url)
                lines.append(f"[查看更多历史记录](<{url}>)")
                linked_index = True
    if len(old) > 3:
        lines.append(f"其余 {len(old) - 3} 轮历史仍保留。" +
                     ("可通过可信历史索引查看更多。" if linked_index else
                      "可信历史索引暂不可用。"))
    lines.append("<!-- deploy-approval-evidence:end -->")
    return "\n".join(lines)
