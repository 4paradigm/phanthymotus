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
    if not object_key:
        return []
    lines = [""]
    text = f"COS: `{_escape(object_key)}`"
    if sha256:
        text += f" \u00b7 `@sha256:{_escape(sha256)}`"
    if size:
        text += f" \u00b7 `{_human_size(size)}`"
    lines.append(text)
    # Keep the argument for existing callers, but never persist expiring COS
    # URLs in the lifecycle summary. render_evidence_section owns stable links.
    if (isinstance(sha256, str) and re.fullmatch(r"[0-9a-f]{64}", sha256)
            and isinstance(size, int) and not isinstance(size, bool)
            and 1 <= size <= 10 * 1024 * 1024):
        lines.append("For a short-lived download link, post `/evidence_link` on this open PR.")
    return lines


def _human_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    else:
        return f"{size / (1024 * 1024):.1f} MB"


# ── Workflow helper ──────────────────────────────────────────────


def _workflow_lines(status: str) -> str:
    """Return the fixed 8-step workflow as markdown.

    Always forward order. [x] for complete, [ ] for pending.
    First unchecked step marked with \u2190 **Next**.
    If all checked, last step is marked done.
    """
    step_defs = [
        ("1. Request Review", 0),
        ("2. Review Agent completed", 1),
        ("3. Request deployment", 2),
        ("4. Machine Owner approval(s)", 3),
        ("5. All required components deployed", 4),
        ("6. Machine Owner uploads evidence", 4),
        ("7. Record validation result", 5),
        ("8. Deployment accepted", 6),
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
    for label, required_level in step_defs:
        checked = current_level > required_level
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
        _workflow_lines("testing"),
        "",
        "### Next action",
        "",
        "**Machine Owner**",
        "",
        '`/record_test result=pass|fail [summary="..."]`',
        "",
        "### Current Deployment",
        "",
        "Deployment validation is in progress; see recorded deployment results below.",
    ]
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
    if deployments:
        lines.extend(["### Deployment records", _deployment_table(deployments, components, machine_groups)])
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
        "- `/approve_deploy machine=<alias-or-ip>` \u2014 Approve and bind machine",
        "- `/record_test result=pass|fail [summary=\"...\"]` \u2014 Record overall test result",
        "",
        "**Read-only commands:**",
        "- `/evidence_link` \u2014 Request a short-lived evidence download link (open PRs only)",
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


def render_evidence_section(repo, pr_number, state, records, archive_links, *, public_base_url="") -> str:
    """Render current evidence and three archived rounds without mutating history.

    Records may be attachment rows, round bundles, or archive_snapshot entries.
    Archive links must come from trusted server-side discovery. Signed COS URLs
    and user-supplied URLs are deliberately not used as download destinations.
    The caller supplies evidence_enabled=True in the render state only when
    the configured feature is enabled; absent flags fail closed for API links.
    """
    from .evidence_models import round_id_for_state

    state = state or {}
    head = state.get("head_sha", "")
    try:
        current_round = round_id_for_state(state)
    except ValueError:
        current_round = state.get("round_id", "")
    base = _public_base(public_base_url)
    enabled = (state.get("evidence_enabled", state.get("enabled", False)) is True
               and state.get("enabled", True) is not False)
    api = _evidence_path(repo, pr_number)
    if not enabled or not api:
        base = ""
    groups = {(head, current_round): dict(state, attachments=list(state.get("attachments") or []))}
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
            for record in row.get("attachment_records", []):
                if isinstance(record, dict) and isinstance(record.get("data"), dict):
                    group["attachments"].append(record["data"])
        else:
            group.setdefault("attachments", []).append(row)

    def download(identifier, suffix, capability):
        identifier = str(identifier or "")
        if not base or not re.fullmatch(r"[A-Za-z0-9_-]+", identifier):
            return ""
        if not capability:
            return ""
        return base + api + suffix.format(id=identifier) + "?cap=" + urllib.parse.quote(capability, safe="")

    def render_round(key, group):
        h, r = key
        lines = [f"**HEAD:** `{_escape(h)}` · **round:** `{_escape(r)}`",
                 f"**Submitter:** {_escape(group.get('actor') or group.get('actor_login') or group.get('submitted_by') or group.get('requested_by') or group.get('command', {}).get('args', {}).get('actor') or 'unknown')}",
                 f"**Time:** {_escape(group.get('created_at') or group.get('started_at') or group.get('round_started_at') or 'unknown')}",
                 f"**Status:** {_escape(group.get('status') or 'unknown')}", "",
                 "| Attachment | Size | Submitter | Time | HEAD | Round | Note | Status | Download |",
                 "|---|---|---|---|---|---|---|---|---|"]
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
        for item in latest_items.values():
            if not isinstance(item, dict):
                continue
            deleted = bool(item.get("deleted_at") or item.get("status") == "deleted")
            status = "deleted" if deleted else item.get("status", "pending")
            link = download(item.get("attachment_id") or item.get("id"),
                            "/attachments/{id}/download", item.get("download_capability", ""))
            cells = [item.get("filename") or item.get("text") or "Note", item.get("size", "—"),
                     ((item.get("actor_login") or item.get("actor"))
                      if item.get("actor_verified") is True else "unverified"),
                     item.get("created_at") or "unknown", item.get("head_sha") or h,
                     item.get("round_id") or r, item.get("note", ""), status]
            if item.get("replaces_id"):
                cells[6] = str(cells[6]) + " · replaces " + str(item["replaces_id"])
            if deleted:
                cells[6] = str(cells[6]) + " · deleted " + str(item.get("deleted_at") or "") + " by " + str(item.get("deleted_by") or "unknown")
            lines.append("| " + " | ".join(_escape(v) for v in cells) + " | " +
                         (f"[Download]({link})" if link and not deleted and status == "ready" else "Unavailable") + " |")
        if group.get("deployments"):
            lines.extend(["", _deployment_table(group["deployments"], group.get("components"), group.get("machine_groups"))])
        for field in ("case_results", "test_result", "summary", "history", "cos_object_key", "cos_bundle_sha256", "cos"):
            if group.get(field):
                lines.extend(["", f"**{_escape(field)}:** {_escape(json.dumps(group[field], ensure_ascii=False) if isinstance(group[field], (list, dict)) else group[field])}"])
        log = download(r, "/rounds/{id}/log/download", group.get("log_capability", ""))
        if log and (group.get("cos_object_key") or group.get("cos", {}).get("object_key") or group.get("log_available") or group.get("log_ready")):
            lines.extend(["", f"[Download round COS log]({log})"])
        return lines

    lines = ["<!-- deploy-approval-evidence:start -->", "### Current round evidence", ""]
    lines.extend(render_round((head, current_round), groups[(head, current_round)]))
    if (base and state.get("evidence_enabled") is True and state.get("status") == "testing"
            and state.get("pr_state", "open") == "open" and not state.get("merged")):
        # The signed Summary capability is bound to this trusted HEAD/round.
        capability = state.get("capability", "")
        if capability:
            evidence_url = f"{base}/evidence/{repo}/pulls/{pr_number}?cap={urllib.parse.quote(capability, safe='')}"
            lines.extend(["", f"[View Evidence]({evidence_url})"])
    old = [(key, group) for key, group in groups.items() if key != (head, current_round)]
    old.sort(key=lambda pair: str(pair[1].get("archived_at") or pair[1].get("ended_at") or pair[1].get("created_at") or ""), reverse=True)
    lines.extend(["", "### Previous rounds", ""])
    for key, group in old[:3]:
        lines.extend(["<details>", f"<summary>HEAD {_escape(key[0])} · round {_escape(key[1])} · {_escape(group.get('status', 'unknown'))}</summary>", "", *render_round(key, group), "", "</details>", ""])
    linked_index = False
    for link in archive_links or []:
        if isinstance(link, dict):
            url = _trusted_archive_url(link.get("url", ""), repo, pr_number, base)
            if url:
                lines.append(f"[History index {_escape(link.get('page', ''))}]({url})")
                linked_index = True
    if len(old) > 3:
        lines.append(f"{len(old) - 3} additional rounds retained. " +
                     ("Use the trusted history index above." if linked_index else
                      "History index unavailable; history has not been deleted."))
    lines.append("<!-- deploy-approval-evidence:end -->")
    return "\n".join(lines)
