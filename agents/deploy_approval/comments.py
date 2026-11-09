"""Status comment rendering for a deployment (stateless GitHub persistence).

One lifecycle comment per PR. Uses a hidden marker for idempotent PATCH.
Each action-required state shows: status, bound head, who acts next, exact commands.
No obsolete lifecycle states, no build_index, no dpl_x.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import ipaddress
import re
import html
import json
from zoneinfo import ZoneInfo

import urllib.parse
from .models import BuildInfo

BOT_MARKER = "<!-- deploy-approval-agent -->"
_BEIJING = ZoneInfo("Asia/Shanghai")

_MAX_VISIBLE_LIFECYCLE_BYTES = 48 * 1024  # 48 KiB soft budget


def beijing_now_str() -> str:
    now = _dt.datetime.now(_BEIJING)
    return now.strftime("%Y-%m-%d %H:%M:%S")


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


_RUNTIME_CHECKS_START = "<!-- deploy-approval-runtime-checks:start -->"
_RUNTIME_CHECKS_END = "<!-- deploy-approval-runtime-checks:end -->"


def _runtime_component_label(component: dict) -> str:
    target = str(component.get("target", "") or "")
    if target == "driver":
        return "driver " + str(component.get("driver_path", "") or "—")
    variant = str(component.get("variant", "") or "")
    return f"{target} {variant}" if variant else target


def runtime_image_checks_block(
    runtime_checks, components, *, lifecycle_status: str = "",
) -> str:
    """Render bounded, verified runtime image observations for the Lifecycle Summary."""
    if not isinstance(runtime_checks, list) or not runtime_checks:
        return ""
    component_map = {
        item.get("component_id"): item for item in components or []
        if isinstance(item, dict) and isinstance(item.get("component_id"), str)
    }
    rows = []
    status_text = {
        "match": "运行镜像与审批镜像一致",
        "mismatch": "运行镜像不是当前审批版本，需要重新部署",
        "not_running": "机器组件当前未运行，需要重新确认",
        "unavailable": "无法确认机器状态",
        "identity_changed": "无法确认机器状态（机器绑定已变化）",
    }
    for check in runtime_checks:
        if not isinstance(check, dict):
            continue
        component = component_map.get(check.get("component_id"), {})
        machine = str(check.get("machine", "") or "")
        ip_value = str(check.get("ip", "") or "")
        target_image = str(check.get("target_image", "") or "")
        observed_image = str(check.get("observed_image", "") or "")
        if component.get("target") == "core" and observed_image:
            observed_image = f"current_tag: {observed_image}"
        checked_at = str(check.get("checked_at", "") or "")
        status = check.get("status", "unavailable")
        next_step = "等待下次状态检查"
        if lifecycle_status == "deploying":
            status_label = status_text.get(status, "无法确认机器状态")
            status_text_value = f"部署进行中；上次只读检查：{status_label}"
            next_step = "正在部署，请等待结果；不要重复审批"
        elif status in {"mismatch", "not_running"}:
            try:
                address = ipaddress.ip_address(ip_value)
            except ValueError:
                address = None
            if address is not None and address.version == 4:
                next_step = f"`/approve_deploy machine={address}`"
            else:
                next_step = "先检查当前仓库的机器配置"
        elif status == "match":
            next_step = "无需重新部署"
        elif status == "identity_changed":
            next_step = "检查当前仓库的机器白名单与绑定"
        rows.append("| " + " | ".join((
            f"`{_escape(machine)}`",
            f"`{_escape(ip_value)}`",
            _escape(_runtime_component_label(component)),
            f"`{_escape(target_image)}`",
            f"`{_escape(observed_image)}`" if observed_image else "未取得",
            _escape(checked_at),
            _escape(status_text_value if lifecycle_status == "deploying"
                    else status_text.get(status, "无法确认机器状态")),
            next_step,
        )) + " |")
    if not rows:
        return ""
    return "\n".join([
        _RUNTIME_CHECKS_START,
        "### 已部署组件镜像检查",
        "",
        "| 机器 | IP | 组件 | 审批目标镜像 | 当前镜像 | 最后检查 | 状态 | 下一步 |",
        "|---|---|---|---|---|---|---|---|",
        *rows,
        "",
        "`/request_deploy` 是可选操作，可直接按上方实际 IP 命令审批。",
        _RUNTIME_CHECKS_END,
    ])


def update_runtime_image_checks(
    markdown: str, runtime_checks, components, *, lifecycle_status: str = "",
) -> str:
    """Replace only the runtime-check section, keeping the rest of the summary intact."""
    if not isinstance(markdown, str):
        return markdown
    start_count = markdown.count(_RUNTIME_CHECKS_START)
    end_count = markdown.count(_RUNTIME_CHECKS_END)
    if start_count != end_count or start_count > 1:
        return markdown
    updated = markdown
    if start_count:
        start = updated.index(_RUNTIME_CHECKS_START)
        end = updated.index(_RUNTIME_CHECKS_END, start) + len(_RUNTIME_CHECKS_END)
        updated = updated[:start].rstrip() + updated[end:]
    block = runtime_image_checks_block(
        runtime_checks, components, lifecycle_status=lifecycle_status,
    )
    if not block:
        return updated.rstrip()
    insert_at = updated.find("<!-- deploy-approval-visible-history:start -->")
    if insert_at < 0:
        history_heading = re.search(r"(?m)^### History\s*$", updated)
        insert_at = history_heading.start() if history_heading else len(updated)
    prefix = updated[:insert_at].rstrip()
    suffix = updated[insert_at:].lstrip()
    return "\n\n".join(part for part in (prefix, block, suffix) if part)


def _build_table(builds: list[BuildInfo], components: list[dict] | None = None) -> str:
    """Render the build results table for deploy-ready comment."""
    image_refs = {
        (item.get("target", ""), item.get("driver_path", ""), item.get("variant", "")):
            item.get("image_ref", "")
        for item in components or [] if isinstance(item, dict)
    }
    lines = [
        "| target | variant/path | image | build | deploy approval |",
        "|--------|--------------|-------|-------|-----------------|",
    ]
    for b in builds:
        variant = _escape(b.variant or b.driver_path or chr(0x2014))
        image_ref = image_refs.get((b.target, b.driver_path, b.variant)) or b.image_tag
        build_status = "success" if b.success else "failed"
        eligibility = "deployable" if b.success and b.deployable else "unsupported"
        if not b.success:
            eligibility = "not deployable"
        lines.append(
            f"| {_escape(b.target)} | {variant} | `{_escape(image_ref)}` | "
            f"{build_status} | {eligibility} |"
        )
    return "\n".join(lines)


def _cos_evidence_block(
    object_key: str = "",
    sha256: str = "",
    size: int = 0,
    download_url: str = "",
) -> list[str]:
    """Keep signed COS bearer URLs out of GitHub Summary markdown."""
    return []


def _human_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    else:
        return f"{size / (1024 * 1024):.1f} MB"


def evidence_upload_comment(
    repo: str, pr_number: int, head_sha: str, round_id: str,
    attachment_id: str, filename: str, *, download_url: str,
    preview_url: str = "", note: str = "", public_base_url: str = "",
) -> str:
    """Build one safe, idempotently identifiable notification for a ready file."""
    if (not _evidence_path(repo, pr_number)
            or not isinstance(head_sha, str)
            or re.fullmatch(r"[0-9a-f]{40}", head_sha) is None
            or not isinstance(round_id, str)
            or re.fullmatch(re.escape(head_sha) + r"-[0-9]+", round_id) is None
            or not isinstance(attachment_id, str)
            or re.fullmatch(r"[A-Za-z0-9_-]{1,128}", attachment_id) is None):
        return ""

    stable_base = _public_base(public_base_url)
    if not stable_base:
        return ""
    api_path = (_evidence_path(repo, pr_number) + "/attachments/"
                + urllib.parse.quote(attachment_id, safe=""))

    def trusted_access_url(value: str, action: str) -> str:
        if not isinstance(value, str) or len(value) > 4096:
            return ""
        try:
            parsed = urllib.parse.urlsplit(value)
            query = urllib.parse.parse_qs(parsed.query, strict_parsing=True)
        except (TypeError, ValueError):
            return ""
        if (parsed.scheme != "https"
                or parsed.netloc != urllib.parse.urlsplit(stable_base).netloc
                or parsed.username or parsed.password or parsed.fragment
                or parsed.path != api_path + "/" + action
                or set(query) != {"cap"} or len(query["cap"]) != 1
                or not re.fullmatch(r"[A-Za-z0-9_.-]{1,2048}", query["cap"][0])):
            return ""
        return value

    download_url = trusted_access_url(download_url, "download")
    preview_url = trusted_access_url(preview_url, "preview")
    if not download_url:
        return ""
    clean_filename = filename if isinstance(filename, str) and filename.strip() else "测试证据"
    label = _escape(clean_filename.strip())
    suffix = clean_filename.rsplit(".", 1)[-1].lower() if "." in clean_filename else ""
    image_suffixes = {"png", "jpg", "jpeg", "gif", "webp"}
    video_suffixes = {"mp4", "webm"}
    lines = [
        "### 测试证据已上传",
        "",
        f"已保存：[{label}](<{download_url}>)",
    ]
    if suffix in image_suffixes and preview_url:
        lines.extend(["", f"![{label}](<{preview_url}>)", "", f"[下载图片](<{download_url}>)"])
    elif suffix in video_suffixes:
        lines[-1] = f"已保存：[{label}](<{preview_url or download_url}>)"
        if preview_url:
            lines.extend(["", f"[下载视频](<{download_url}>)"])
    if isinstance(note, str) and note.strip():
        from .evidence_builder import _redact_secrets

        safe_note = _redact_secrets(note.strip())
        safe_note = re.sub(
            r"(?i)((?:x-evidence-secret|x-evidence-client-key|evidence-capability|capability|cap|token)\s*[=:]\s*)[^\s&;]+",
            r"\1[已隐藏]", safe_note)
        safe_note = re.sub(r"https?://[^\s<>()]+", "[链接已隐藏]", safe_note)
        escaped_note = _escape(safe_note)
        lines.extend(["", "备注：", "", *("> " + line for line in escaped_note.splitlines())])
    notification_key = hashlib.sha256(
        f"{repo}:{pr_number}:{head_sha}:{round_id}:{attachment_id}".encode("utf-8")
    ).hexdigest()
    return "\n".join([
        BOT_MARKER,
        f"<!-- deploy-approval-evidence-upload:{notification_key} -->",
        *lines,
    ])


def stable_attachment_url(repo, pr_number, attachment_id, capability, *, action,
                          public_base_url="") -> str:
    """Build a stable Evidence route; the route signs private COS access on click."""
    base = _public_base(public_base_url)
    api = _evidence_path(repo, pr_number)
    if (not base or not api or action not in {"download", "preview"}
            or not isinstance(attachment_id, str)
            or re.fullmatch(r"[A-Za-z0-9_-]{1,128}", attachment_id) is None
            or not isinstance(capability, str)
            or not re.fullmatch(r"[A-Za-z0-9_.-]{1,2048}", capability)):
        return ""
    path = api + "/attachments/" + urllib.parse.quote(attachment_id, safe="") + "/" + action
    return base + path + "?" + urllib.parse.urlencode({"cap": capability})


# ── Workflow helper ──────────────────────────────────────────────


def _workflow_lines(
    status: str, evidence_complete: bool = False,
    deployments_complete: bool | None = None,
    record_test_complete: bool = False,
) -> str:
    """Keep the progress note short and separate from the next action."""
    if status == "review-required":
        return "当前 HEAD 尚无可信代码审查结果。"
    if status == "reviewing":
        return "Review Agent 正在检查当前 HEAD。"
    if status == "deploy-ready":
        return "当前 HEAD 的构建和审核已通过。"
    if status == "deploy-requested":
        return "本轮部署申请已收到。"
    if status == "deploying":
        return "部署命令已提交，结果正在确认。"
    if status == "testing":
        return "本轮部署已完成，可开始测试。"
    if status == "succeeded":
        return "本轮验收已通过。"
    if status == "failed":
        return "本轮流程失败，请查看具体部署或验收记录。"
    return "等待更新。"


def _trusted_upload_page_url(value: str, repo: str, pr_number: int) -> str:
    """Accept only the configured HTTPS upload page for this exact PR."""
    if not isinstance(value, str) or len(value) > 4096:
        return ""
    try:
        parsed = urllib.parse.urlsplit(value)
        query = urllib.parse.parse_qs(parsed.query, strict_parsing=True)
        from .config import is_evidence_ingress_url
        origin = f"{parsed.scheme}://{parsed.netloc}"
        expected_path = _evidence_path(repo, pr_number).replace(
            "/api/evidence/", "/evidence/", 1,
        )
    except (TypeError, UnicodeError, ValueError):
        return ""
    if (not expected_path or not is_evidence_ingress_url(origin)
            or parsed.path != expected_path or parsed.fragment
            or parsed.username or parsed.password
            or set(query) != {"cap"} or len(query["cap"]) != 1
            or re.fullmatch(r"[A-Za-z0-9_.-]{1,2048}", query["cap"][0]) is None):
        return ""
    return value


def _testing_next_action(
    evidence_complete: bool, evidence_url: str = "", *,
    repo: str = "", pr_number: int = 0,
) -> str:
    evidence_url = _trusted_upload_page_url(evidence_url, repo, pr_number)
    lines = []
    if evidence_complete:
        lines.append("测试证据已上传。如需补充视频或图片：")
        if evidence_url:
            lines.append(f"[继续上传测试视频/图片](<{evidence_url}>)")
        else:
            lines.extend([
                "[继续上传测试视频/图片][deploy-approval-upload]",
                "若入口暂不可用，请查看下方附件区的状态说明。",
            ])
    else:
        lines.extend(["请上传本轮测试视频或图片：", ""])
        if evidence_url:
            lines.append(f"[上传测试视频/图片](<{evidence_url}>)")
        else:
            lines.extend([
                "[上传测试视频/图片][deploy-approval-upload]",
                "若入口暂不可用，请查看下方附件区的状态说明。",
            ])
        lines.extend(["", "测试完成后，由本轮机器负责人在 PR 评论中输入：", "", "`/request_acceptance`"])
        return "\n".join(lines)
    lines.extend(["", "测试完成后，由本轮机器负责人在 PR 评论中输入：", "", "`/request_acceptance`"])
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
        "等待 Review Agent 返回当前 HEAD 的审查与构建结果。",
    ])


def deploy_ready(
    repo: str, pr_number: int, head_sha: str,
    builds: list[BuildInfo], last_lifecycle_event: str = "",
    review_evidence: dict | None = None,
    machine_groups: list[dict] | None = None,
    components: list[dict] | None = None,
) -> str:
    deployable_ids = {
        component.get("component_id") for component in components or []
        if isinstance(component, dict) and isinstance(component.get("component_id"), str)
        and component.get("component_id")
    }
    if not deployable_ids:
        deployable_ids = {
            build.component_id for build in builds
            if build.success and build.deployable and isinstance(build.component_id, str) and build.component_id
        }
    commands = []
    full_machine_ips = set()
    seen_ips = set()
    for group in machine_groups or []:
        if not isinstance(group, dict):
            continue
        compatible = group.get("component_ids")
        if not isinstance(compatible, list):
            continue
        compatible_ids = {component for component in compatible if isinstance(component, str)}
        if not deployable_ids or not deployable_ids.intersection(compatible_ids):
            continue
        try:
            ip = ipaddress.ip_address(group.get("ip"))
        except (TypeError, ValueError):
            continue
        if ip.version != 4:
            continue
        if deployable_ids.issubset(compatible_ids):
            full_machine_ips.add(str(ip))
        if str(ip) in seen_ips:
            continue
        seen_ips.add(str(ip))
        alias = group.get("alias")
        label = f"{_escape(alias)}（`{ip}`）" if isinstance(alias, str) and alias else f"`{ip}`"
        commands.append(f"- {label}：`/approve_deploy machine={ip}`")

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
        "机器负责人请按下方适配机器的真实 IP 命令审批：",
        "",
        *(commands or ["当前没有可用机器 IP，暂时无法生成直接部署命令。"]),
    ]
    if len(full_machine_ips) == 1:
        lines.extend(["", "当前只有一台完整适配机器，也可以直接使用简写：", "", "`/approve_deploy`"])
    lines.extend([
        "",
        "也可以选择用 `/request_deploy` 申请部署。",
        "",
        "### Review evidence",
        "",
        f"- **Build Result:** {'success' if builds and all(b.success for b in builds) else 'failed'}",
    ])
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
        "> 请先阅读 Review Agent 的审查意见，再选择上方部署命令。",
        "",
        "### Components",
        "",
        _build_table(builds, components),
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
    display_status: str = "deploy-requested",
) -> str:
    """status: deploy-requested with IP table, workflow, deployment, history."""
    if (not isinstance(display_status, str)
            or display_status not in {"deploy-requested", "deploying"}):
        raise ValueError("display_status must be deploy-requested or deploying")
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

    remaining_ids = {
        item.get("component_id") for item in components
        if isinstance(item, dict) and isinstance(item.get("component_id"), str)
    }
    machine_commands = []
    seen_commands = set()
    for group in machine_groups or []:
        if not isinstance(group, dict):
            continue
        component_ids = group.get("component_ids")
        if (not isinstance(component_ids, list)
                or not any(component_id in remaining_ids for component_id in component_ids)):
            continue
        raw_ip = group.get("ip")
        try:
            address = ipaddress.ip_address(raw_ip)
        except (TypeError, ValueError):
            continue
        if address.version != 4:
            continue
        command = f"/approve_deploy machine={address}"
        if command in seen_commands:
            continue
        seen_commands.add(command)
        machine_commands.append((str(group.get("alias") or ""), str(address), command))

    lines = [
        BOT_MARKER,
        lifecycle_marker(repo, pr_number),
        "### Deploy Approval \u2014 Lifecycle",
        "",
        f"**Status:** `{display_status}`",
        f"**Bound HEAD:** `{_short(head_sha)}`",
        f"**Repository:** `{repo}`",
    ]
    lines.extend([
        "",
        "### Workflow",
        "",
        _workflow_lines(display_status),
        "",
        "### Next action",
        "",
        ("正在部署，请等待结果。" if display_status == "deploying"
         else "等待机器负责人按下方对应机器的真实 IP 命令审批："),
        "",
        *([] if display_status == "deploying" else
          [f"- `{_escape(alias)}`（`{_escape(ip)}`）：`{command}`"
           for alias, ip, command in machine_commands]
          if machine_commands else
          ["当前没有可用的机器 IP，暂时无法生成审批命令，请检查机器配置。"]),
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
    acceptance_pending: bool = False,
    acceptance_reviewers_configured: bool = True,
    acceptance_reviewers: list[str] | None = None,
    evidence_url: str = "",
) -> str:
    if acceptance_pending:
        reviewers = [name.strip() for name in acceptance_reviewers or []
                     if isinstance(name, str)
                     and re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38}", name.strip())]
        next_action = []
        if reviewers:
            waiting_for = "、".join(f"@{_escape(name)}" for name in reviewers)
            next_action.extend([
                f"正在等待以下验收人确认：{waiting_for}。",
                "其中一位验收人请在 PR 评论中执行其中一条命令：\n"
                "- `/record_test result=pass`\n"
                "- `/record_test result=fail`",
            ])
        else:
            if not acceptance_reviewers_configured:
                reason = "当前仓库没有配置可用的验收账号，无法验收。请管理员检查 test_reviewers.yaml。"
            else:
                reason = "验收名单中没有可用账号，无法验收。请管理员检查 test_reviewers.yaml。"
            next_action.append(reason)
        next_action = "\n\n".join(next_action)
    else:
        next_action = _testing_next_action(
            evidence_complete, evidence_url, repo=repo, pr_number=pr_number,
        )
    lines = [
        BOT_MARKER,
        lifecycle_marker(repo, pr_number),
        "### Deploy Approval \u2014 Lifecycle",
        "",
        ("**Status:** `testing` · 测试完成 · 待验收"
         if acceptance_pending else "**Status:** `testing`"),
        f"**Repository:** `{repo}`",
        "",
        "### Workflow",
        "",
        ("已收到本轮验收申请。" if acceptance_pending else
         _workflow_lines("testing", evidence_complete=evidence_complete,
                         deployments_complete=deployments_complete,
                         record_test_complete=record_test_complete)),
        "",
        "### Next action",
        "",
        *next_action.split("\n"),
    ]
    if not (deployments_complete is True and deployments):
        lines.extend(["", "### Current Deployment", ""])
        if deployments:
            lines.append(_deployment_table(deployments, components, machine_groups))
        else:
            lines.append("No deployment record is available for this round yet.")
    if case_result:
        lines.extend([
            "",
            f"**Automated case results:** {case_result}",
            "Fixed Case results are advisory only.",
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
        f"**Repository:** `{repo}`",
        "",
        "### Workflow",
        "",
        _workflow_lines("succeeded"),
        "",
        "### Next action",
        "",
        "无需进一步操作。",
        "",
    ]
    lines.extend(_cos_evidence_block(cos_object_key, cos_bundle_sha256, cos_bundle_size, cos_download_url))
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
        f"**Repository:** `{repo}`",
        "",
        "### Workflow",
        "",
        _workflow_lines("failed"),
        "",
        "### Next action",
        "",
        "请查看下方失败原因和部署记录。",
    ]
    if error:
        lines.append(f"**Error:** {_escape(error)}")
    lines.extend(_cos_evidence_block(cos_object_key, cos_bundle_sha256, cos_bundle_size, cos_download_url))
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
        "**Status:** `deploying`",
        "**Command phase:** `uncertain`",
        f"**Bound HEAD:** `{_short(head_sha)}`",
        f"**Repository:** `{repo}`",
        "",
        "### Workflow",
        "",
        _workflow_lines("deploying"),
        "",
        "### Next action",
        "",
        "部署结果正在核实，请等待状态更新；请勿重复提交部署审批命令。",
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
        "deploying", head_sha, repo, pr_number,
        components=components, deployments=deployments,
        machine_groups=machine_groups, attempt=details,
        last_lifecycle_event=last_lifecycle_event,
    )


def deploy_help_text(topic: str = "") -> str:
    if topic == "request_bot_review":
        return "\n".join([
            BOT_MARKER,
            "### Deploy Approval — Help: /request_bot_review",
            "",
            "用途：触发当前 HEAD 的代码审查。",
            "执行人：PR 开发者（PR 作者）。",
            "阶段：PR 开放且处于 review-required；审查绑定当前 HEAD。",
            "",
            "命令：`/request_bot_review`",
        ])
    if topic == "request_deploy":
        return "\n".join([
            BOT_MARKER,
            "### Deploy Approval \u2014 Help: /request_deploy",
            "",
            "用途：开发者确认已阅读当前 HEAD 的审查意见并发起部署请求。",
            "",
            "执行人：PR 作者。",
            "阶段：PR 开放且状态为 deploy-ready。",
            "",
            "命令：",
            "`/request_deploy`",
        ])
    if topic == "approve_deploy":
        return "\n".join([
            BOT_MARKER,
            "### Deploy Approval \u2014 Help: /approve_deploy",
            "",
            "用途：审批部署并将其绑定到指定机器。",
            "",
            "执行人：对应 Machine Owner，或有 write/maintain/admin 权限的协作者。",
            "阶段：deploy-ready 时可直接审批；deploy-requested 时为常规机器审批阶段。",
            "",
            "请使用 PR Summary 中为兼容机器列出的真实 IP 命令。仅一台机器完全适配时，可使用 `/approve_deploy`；`/request_deploy` 是可选操作。",
        ])
    if topic == "record_test":
        return "\n".join([
            BOT_MARKER,
            "### Deploy Approval \u2014 Help: /record_test",
            "",
            "用途：记录本轮部署验证结果。",
            "",
            "两个目标仓库分别维护各自的 `test_reviewers.yaml`；执行 `/record_test` 只使用当前仓库的验收名单。",
            "执行人：当前仓库验收名单中的 GitHub 用户。部署审批权限与最终验收权限相互独立。",
            "请求验收后，Summary 会显示当前仓库有效的 `@账号`；只有这些验收白名单账号可以执行 `/record_test`。",
            "阶段：本轮已在 testing 阶段请求验收后。",
            "",
            "参数：",
            "- `/record_test result=pass [summary=\"...\"]`",
            "- `/record_test result=fail [summary=\"...\"]`",
            "- `result=pass` 表示本轮测试成功。",
            "- `result=fail` 表示本轮测试失败。",
            "- `summary` 是测试结果说明（可选）；不要写入凭据或签名链接。",
            "",
            "示例：",
            "- `/record_test result=pass summary=\"视频播放和附件校验通过\"`",
            "- `/record_test result=fail summary=\"视频播放失败，HTTP 403\"`",
            "",
        ])
    return "\n".join([
        BOT_MARKER,
        "### Deploy Approval \u2014 Help",
        "",
        "命令用途、执行人和阶段：",
        "- `/request_bot_review`：触发当前 HEAD 代码审查；PR 作者在开放 PR 的 review-required 阶段执行。",
        "- `/request_deploy`：PR 作者在开放 PR 的 deploy-ready 阶段发起部署请求。",
        "- 机器审批：对应 Machine Owner 或有 write/maintain/admin 权限的协作者；deploy-ready 时按 Summary 中兼容机器的真实 IP 命令直接审批，仅一台机器完全适配时可使用 `/approve_deploy`；deploy-requested 时按 Summary 的机器 IP 命令常规审批。`/request_deploy` 是可选操作。",
        "- `/request_acceptance`：本轮实际部署机器的 Machine Owner 在 testing 阶段请求验收；仅更新待验收标记，不结束 testing。",
        "- 两个目标仓库分别维护自己的 `test_reviewers.yaml`；部署审批权限不会授予验收权限。",
        "- `/record_test result=pass summary=\"视频播放和附件校验通过\"`：当前仓库验收白名单中的用户在待验收阶段记录成功。",
        "- `/record_test result=fail summary=\"视频播放失败，HTTP 403\"`：当前仓库验收白名单中的用户在待验收阶段记录失败。",
        "- 请求验收后，Summary 会标出当前仓库有效的 `@账号`；只有这些账号可以执行 `/record_test`。",
        "- `/deploy_status`：只读查询当前部署状态，可在任意阶段使用。",
        "- `/deploy_help record_test`：查看验收命令说明；可在任意阶段使用。",
        "",
        "结果含义：`pass`=成功；`fail`=失败；`summary`=测试结果说明（可选）。",
        "",
        "详细帮助：`/deploy_help request_bot_review`、`/deploy_help request_deploy`、`/deploy_help approve_deploy`、`/deploy_help record_test`。",
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
    """Accept only a configured, stable public HTTPS origin for Summary links."""
    from .config import is_stable_public_base_url

    return value.rstrip("/") if is_stable_public_base_url(value) else ""


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


def _machine_log_lines(group, key, api, origin):
    machine_logs = group.get("deployment_logs", {})
    deployed_machines = {
        deployment.get("machine") for deployment in group.get("deployments", [])
        if isinstance(deployment, dict) and deployment.get("phase") == "deployed"
        and isinstance(deployment.get("machine"), str)
    }
    valid_log_statuses = {"pending", "ready", "unavailable"}
    if isinstance(machine_logs, dict):
        deployed_machines.update(
            machine for machine, record in machine_logs.items()
            if isinstance(machine, str) and isinstance(record, dict)
            and record.get("head_sha") == key[0]
            and record.get("round_id") == key[1]
            and record.get("status") in valid_log_statuses
        )
    deployed_machines = sorted(deployed_machines)
    if not deployed_machines:
        return []
    lines = []
    for machine in deployed_machines:
        record = machine_logs.get(machine) if isinstance(machine_logs, dict) else None
        if (not isinstance(record, dict)
                or record.get("head_sha") != key[0]
                or record.get("round_id") != key[1]
                or record.get("status") not in valid_log_statuses):
            record = None
        status = record.get("status") if record is not None else "missing"
        capability = record.get("capability") if record is not None else None
        if status == "ready" and origin and isinstance(capability, str) and capability:
            path = api + "/rounds/" + urllib.parse.quote(key[1], safe="")
            path += "/machines/" + urllib.parse.quote(machine, safe="") + "/log/download"
            url = origin + path + "?" + urllib.parse.urlencode({"cap": capability})
            lines.append(f"- `{_escape(machine)}`：[下载部署日志](<{url}>)")
        elif status == "ready":
            lines.append(f"- `{_escape(machine)}`：已归档，下载链接暂不可用")
        elif status == "pending":
            lines.append(f"- `{_escape(machine)}`：正在保存部署日志")
        elif status == "unavailable":
            lines.append(f"- `{_escape(machine)}`：记录包不可用")
        else:
            lines.append(f"- `{_escape(machine)}`：未归档")
    return lines


def render_evidence_section(
    repo, pr_number, state, records, archive_links, *, public_base_url="",
    upload_public_base_url="", cos_download_url_expiry=None,
) -> str:
    """Render current evidence and archived rounds without mutating history.

    Records may be attachment rows, round bundles, or archive_snapshot entries.
    Archive links must come from trusted server-side discovery. COS download
    links are rendered only after the projection and URL signature checks pass.
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
    def valid_round_binding(bound_head, bound_round):
        return (isinstance(bound_head, str)
                and re.fullmatch(r"[0-9a-f]{40}", bound_head) is not None
                and isinstance(bound_round, str)
                and re.fullmatch(re.escape(bound_head) + r"-[0-9]+", bound_round) is not None)

    show_current_round = (
        state.get("status") in {"testing", "succeeded", "failed"}
        and valid_round_binding(head, current_round)
    )
    base = _public_base(public_base_url)
    if not base.startswith("https://"):
        base = ""
    from .config import is_evidence_ingress_url
    upload_base = (
        upload_public_base_url.rstrip("/")
        if is_evidence_ingress_url(upload_public_base_url) else ""
    )
    enabled = (state.get("evidence_enabled", state.get("enabled", False)) is True
               and state.get("enabled", True) is not False)
    api = _evidence_path(repo, pr_number)
    if not enabled or not api:
        base = ""
        upload_base = ""
    def attachment_list_matches(items, key):
        return (isinstance(items, list) and all(
            isinstance(item, dict)
            and item.get("head_sha", key[0]) == key[0]
            and item.get("round_id", key[1]) == key[1]
            for item in items
        ))

    current_attachments = state.get("attachments")
    groups = {(head, current_round): dict(
        state, attachments=list(current_attachments) if isinstance(current_attachments, list) else [],
        attachment_details_complete=attachment_list_matches(current_attachments, (head, current_round)),
        attachment_details_unreadable=(
            "attachments" in state
            and not attachment_list_matches(current_attachments, (head, current_round))
        ),
    )}
    if isinstance(records, dict):
        records = records.get("rounds", records.get("records", records.get("attachments", [])))
    archive_links = list(archive_links or [])
    trusted_archive_urls = list(dict.fromkeys(
        url for link in archive_links if isinstance(link, dict)
        for url in [_trusted_archive_url(link.get("url", ""), repo, pr_number, base)]
        if url
    ))
    entries = list(records or []) + [a for a in archive_links if isinstance(a, dict) and "snapshot" in a]
    for envelope in entries:
        row = envelope
        if not isinstance(row, dict):
            continue
        if isinstance(row.get("data"), dict):
            row = row["data"]
        snapshot = row.get("snapshot", row)
        if not isinstance(snapshot, dict):
            continue
        sources = (envelope, row, snapshot)
        declared_repos = [source.get("repo") for source in sources if source.get("repo") is not None]
        declared_prs = [source.get("pr_number") for source in sources if source.get("pr_number") is not None]
        if (not declared_repos or not declared_prs
                or any(value != repo for value in declared_repos)
                or any(type(value) is not int or value != pr_number for value in declared_prs)):
            continue
        row_head, snapshot_head = row.get("head_sha"), snapshot.get("head_sha")
        row_round, snapshot_round = row.get("round_id"), snapshot.get("round_id")
        if (row_head is not None and snapshot_head is not None and row_head != snapshot_head
                or row_round is not None and snapshot_round is not None and row_round != snapshot_round):
            continue
        h = row_head if row_head is not None else snapshot_head
        r = row_round if row_round is not None else snapshot_round
        if not valid_round_binding(h, r):
            continue
        key = (h, r)
        trusted_archive_round_number = (
            envelope.get("round_number")
            if isinstance(envelope, dict) and isinstance(envelope.get("snapshot"), dict)
            else None
        )
        group = groups.setdefault(key, {
            "attachments": [], "attachment_details_complete": False,
            "attachment_details_unreadable": False,
        })
        if "snapshot" in row or "attachments" in row or "deployments" in row:
            attachments = list(group.get("attachments") or [])
            details_complete = group.get("attachment_details_complete") is True
            if key != (head, current_round):
                capability = group.get("capability")
                group.update({k: v for k, v in snapshot.items() if k != "round_number"})
                group.update({k: v for k, v in row.items()
                              if k not in {"snapshot", "round_number"}})
                if (type(trusted_archive_round_number) is int
                        and trusted_archive_round_number > 0):
                    group["round_number"] = trusted_archive_round_number
                if not group.get("capability") and capability:
                    group["capability"] = capability
            snapshot_attachments = snapshot.get("attachments")
            row_attachments = row.get("attachments")
            if isinstance(snapshot_attachments, list):
                group["attachments"] = attachments + snapshot_attachments
                snapshot_complete = attachment_list_matches(snapshot_attachments, key)
                details_complete = details_complete or snapshot_complete
                group["attachment_details_unreadable"] |= not snapshot_complete
            else:
                group["attachments"] = attachments
                group["attachment_details_unreadable"] |= "attachments" in snapshot
            if row is not snapshot and isinstance(row_attachments, list):
                group["attachments"].extend(row_attachments)
                row_complete = attachment_list_matches(row_attachments, key)
                details_complete = details_complete or row_complete
                group["attachment_details_unreadable"] |= not row_complete
            elif row is not snapshot and "attachments" in row:
                group["attachment_details_unreadable"] = True
            attachment_records = row.get("attachment_records")
            if isinstance(attachment_records, list):
                records_complete = True
                for record in attachment_records:
                    if isinstance(record, dict) and record.get("kind") == "operation":
                        continue
                    if not isinstance(record, dict) or record.get("kind") != "attachment":
                        records_complete = False
                        group["attachment_details_unreadable"] = True
                        continue
                    data = record.get("data")
                    if (not isinstance(data, dict)
                            or data.get("repo") != repo
                            or type(data.get("pr_number")) is not int
                            or data.get("pr_number") != pr_number
                            or data.get("head_sha") != key[0]
                            or data.get("round_id") != key[1]):
                        records_complete = False
                        group["attachment_details_unreadable"] = True
                        continue
                    group["attachments"].append(data)
                details_complete = details_complete or records_complete
            elif "attachment_records" in row:
                group["attachment_details_unreadable"] = True
            group["attachment_details_complete"] = details_complete
        else:
            group.setdefault("attachments", []).append(row)

    page_path = api.replace("/api/evidence/", "/evidence/", 1)

    def page_url_at(capability, origin):
        if not origin or not capability or not page_path.startswith("/evidence/"):
            return ""
        return (origin + page_path + "?cap="
                + urllib.parse.quote(str(capability), safe=""))

    def stable_read_page_url(key, origin):
        if (not origin or not page_path.startswith("/evidence/")
                or not isinstance(key, tuple) or len(key) != 2
                or not valid_round_binding(key[0], key[1])):
            return ""
        query = urllib.parse.urlencode({
            "view": "read", "head_sha": key[0], "round_id": key[1],
        })
        return origin + page_path + "?" + query

    def stable_attachment_download_url(item, key, origin):
        if (not isinstance(item, dict) or not origin or not api
                or item.get("status") != "ready" or item.get("deleted_at")
                or item.get("cos_projection_verified") is not True
                or item.get("head_sha") != key[0] or item.get("round_id") != key[1]
                or not isinstance(item.get("sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None):
            return ""
        verified_content = (
            item.get("cos_content_verified") is True
            or item.get("actor_verified") is True
            and item.get("upload_method") != "signed_invite"
        )
        attachment_id = item.get("attachment_id")
        capability = item.get("download_capability")
        size = item.get("size")
        if (not verified_content
                or not isinstance(attachment_id, str)
                or re.fullmatch(r"[A-Za-z0-9_-]{1,128}", attachment_id) is None
                or not isinstance(capability, str) or not capability
                or len(capability) > 2048
                or isinstance(size, bool) or not isinstance(size, int)
                or not 1 <= size <= 512 * 1024 * 1024
                or any(char.isspace() or ord(char) < 32 for char in capability)):
            return ""
        path = api + "/attachments/" + urllib.parse.quote(attachment_id, safe="") + "/download"
        return origin + path + "?" + urllib.parse.urlencode({"cap": capability})

    def latest_attachments(group, key):
        latest_items = {}
        for offset, item in enumerate(group.get("attachments", [])):
            if not isinstance(item, dict):
                continue
            if (item.get("head_sha", key[0]) != key[0]
                    or item.get("round_id", key[1]) != key[1]):
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
        if not isinstance(item, dict) or item.get("actor_verified") is not True:
            return False
        actor = item.get("actor_login") or item.get("actor") or item.get("uploader")
        if not isinstance(actor, str) or actor.casefold() != "4pd-pr-assistant[bot]":
            return False
        candidates = [item.get("note")]
        if item.get("status") == "text":
            candidates.extend([item.get("text"), item.get("filename")])
        templates = {
            "server-side cos multipart acceptance",
            "server-side 20mb h.264 cos transport acceptance",
        }
        for note in candidates:
            if not isinstance(note, str):
                continue
            normalized = " ".join(note.split()).casefold()
            for template in templates:
                suffix = normalized[len(template):] if normalized.startswith(template) else None
                if suffix is not None and (not suffix or suffix[0] in " :;,.!?…-"):
                    return True
        return False

    def visible_attachments(group, key):
        return [item for item in latest_attachments(group, key)
                if not (item.get("status") == "text" and is_internal_test_note(item))]

    def only_hidden_internal_notes(group, key):
        items = latest_attachments(group, key)
        return bool(items) and all(
            item.get("status") == "text" and is_internal_test_note(item)
            for item in items
        )

    def parsed_timestamp(value):
        if not isinstance(value, str) or not value.strip():
            return None
        try:
            return _dt.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None

    def safe_note_text(value):
        from .evidence_builder import _redact_secrets

        value = _redact_secrets(value)
        value = re.sub(
            r'(?i)((?:x-evidence-secret|x-evidence-client-key|evidence-capability|capability|cap)\s*[=:]\s*)[^\s&;]+',
            r'\1[REDACTED]', value)

        def redact_url(match):
            url = match.group(0)
            try:
                return "[签名链接已省略]" if urllib.parse.urlsplit(url).query else url
            except ValueError:
                return "[链接已省略]"

        return re.sub(r"https?://[^\s<>()]+", redact_url, value)

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

    def first_valid_time(*values):
        for value in values:
            if parsed_timestamp(value) is not None:
                return value
        return None

    def round_label(group, *, historical=False):
        round_number = group.get("round_number")
        if type(round_number) is int and round_number > 0:
            return f"第 {round_number} 轮"
        return "历史轮次" if historical else "当前轮"

    def round_status(group, key):
        request = group.get("acceptance_request")
        if (group.get("status") == "testing" and isinstance(request, dict)
                and request.get("status") == "pending"
                and request.get("head_sha") == key[0]
                and request.get("round_id") == key[1]):
            return "测试完成 · 待验收"
        return {
            "testing": "测试中", "succeeded": "已通过", "failed": "未通过",
            "review-required": "待审查", "reviewing": "审查中",
            "deploy-ready": "待部署", "deploy-requested": "等待部署审批",
            "deploying": "正在部署", "deployed": "已部署", "cancelled": "已取消",
        }.get(group.get("status"), "状态未知")

    def render_round(key, group, *, historical=False):
        h, r = key
        remaining_note_chars = 900
        notes_truncated = False
        test_result = group.get("test_result")
        result_labels = {"pass": "通过", "fail": "失败"}
        component_by_id = {
            component.get("component_id"): component
            for component in group.get("components", [])
            if isinstance(component, dict)
            and isinstance(component.get("component_id"), str)
        }
        case_labels = {
            "running": "运行中", "pass": "通过", "fail": "失败", "n/a": "不适用",
        }
        case_lines = []
        case_results = group.get("case_results")
        if isinstance(case_results, dict):
            for component_id, result in case_results.items():
                if (not isinstance(component_id, str) or not isinstance(result, str)
                        or result not in case_labels):
                    continue
                component = component_by_id.get(component_id, {})
                target = component.get("target")
                if target == "driver" and component.get("driver_path"):
                    label = f"driver {component['driver_path']}"
                elif target and component.get("variant"):
                    label = f"{target} {component['variant']}"
                else:
                    label = target or component_id
                case_lines.append(f"- {_escape(label)}：{case_labels[result]}")

        test_summary = ""
        command = group.get("command")
        command_args = command.get("args") if isinstance(command, dict) else None
        request = group.get("acceptance_request")
        if (isinstance(test_result, str) and test_result in result_labels
                and isinstance(request, dict) and request.get("status") == "completed"
                and request.get("head_sha") == key[0]
                and request.get("round_id") == key[1]
                and request.get("result") == test_result
                and isinstance(request.get("summary"), str)):
            test_summary = request["summary"].strip()
        elif (isinstance(test_result, str) and test_result in result_labels
              and isinstance(command, dict) and command.get("kind") == "record_test"
              and command.get("phase") == "completed"
              and isinstance(command_args, dict)
              and command_args.get("result") == test_result
              and isinstance(command_args.get("summary"), str)):
            test_summary = command_args["summary"].strip()
        if test_summary:
            try:
                json.loads(test_summary)
                test_summary = ""
            except (TypeError, ValueError):
                test_summary = safe_note_text(test_summary[:500])
        test_details = []
        if case_lines:
            test_details.extend(case_lines)
        if test_summary:
            test_details.append(f"- **测试摘要：** {_escape(test_summary)}")
        items = visible_attachments(group, key)
        items.sort(key=lambda item: sortable_time(first_valid_time(
            item.get("created_at"), item.get("uploaded_at"))), reverse=True)
        lines = []
        lines.extend([
            "", f"**Commit：** `{_escape(h)}`",
            f"**状态：** {_escape(round_status(group, key))}",
            f"**测试时间：** {_escape(summary_time(first_valid_time(group.get('round_started_at'), group.get('started_at'))))}",
        ])
        if isinstance(test_result, str) and test_result in result_labels:
            lines.extend(["", f"**验收结果：** {result_labels[test_result]}"])
        if test_details:
            lines.extend(["", "<details>", "<summary>测试详情</summary>", "",
                          *test_details, "", "</details>"])
        lines.extend(["", "### 附件", ""])
        if items:
            lines.extend(["| 文件 | 大小 | 上传时间 |", "|---|---:|---|"])
        notes = []
        seen_notes = set()
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
            uploaded_at = summary_time(first_valid_time(item.get("created_at"), item.get("uploaded_at")))
            if deleted:
                filename = filename + "（已删除）"
            download_url = ("" if deleted else
                            stable_attachment_download_url(item, key, base or upload_base))
            filename_cell = _escape(filename)
            if download_url:
                filename_cell = f"[{filename_cell}](<{download_url}>)"
            lines.append("| " + " | ".join((
                filename_cell, _escape(size), _escape(uploaded_at),
            )) + " |")
            note = item.get("note") or (item.get("text") if status == "text" else "")
            if is_internal_test_note(item):
                note = ""
            if item.get("evidence_kind") == "note":
                complete_text_note = False
                if status == "text" and isinstance(note, str):
                    try:
                        note_bytes = note.encode("utf-8")
                        complete_text_note = (
                            len(note) <= 5000 and item.get("size") == len(note_bytes)
                            and isinstance(item.get("sha256"), str)
                            and hashlib.sha256(note_bytes).hexdigest() == item["sha256"]
                            and type(item.get("source_comment_id")) is int
                            and item["source_comment_id"] > 0
                            and item.get("actor_verified") is True
                        )
                    except UnicodeError:
                        complete_text_note = False
                if not complete_text_note:
                    note = ((note.rstrip() + "\n\n") if isinstance(note, str) and note.strip() else "") + \
                        "[摘要中仅保留备注摘要，完整内容请在附件页查看；内容不完整]"
            if isinstance(note, str) and note.strip():
                rendered_note = safe_note_text(note.strip())
                note_identity = item.get("attachment_id") or item.get("id")
                if not isinstance(note_identity, (str, int)):
                    note_identity = filename.casefold()
                note_key = (note_identity, rendered_note)
                if note_key not in seen_notes:
                    seen_notes.add(note_key)
                    if remaining_note_chars <= 0:
                        notes_truncated = True
                        continue
                    if len(rendered_note) > remaining_note_chars:
                        rendered_note = rendered_note[:remaining_note_chars] + "…（摘要已截短；完整说明保留在附件记录中）"
                        notes_truncated = True
                    remaining_note_chars -= min(len(rendered_note), remaining_note_chars)
                    notes.append(f"- **{_escape(filename)}：** {_escape(rendered_note)}")
        if notes:
            lines.extend(["", "<details>", "<summary>附件说明</summary>", "", *notes, "", "</details>"])
        if notes_truncated:
            lines.extend(["", "备注摘要已精简；可打开本轮附件页查看完整内容。"])
        if historical and items and group.get("attachment_details_unreadable"):
            lines.append("部分历史附件明细无法读取或校验；此处仅显示通过校验的记录。")
        if not items:
            if historical:
                if group.get("attachment_details_unreadable"):
                    detail_status = (
                        "请查看下方原始归档链接。"
                        if trusted_archive_urls else "可信原始归档链接暂不可用。"
                    )
                    lines.append(f"历史附件数据不可读取或校验失败；{detail_status}")
                elif not group.get("attachment_details_complete"):
                    detail_status = (
                        "请查看下方原始归档链接。"
                        if trusted_archive_urls else "可信原始归档链接暂不可用。"
                    )
                    lines.append(f"历史快照未包含附件明细；{detail_status}")
                elif only_hidden_internal_notes(group, key):
                    lines.append("内部验收备注已隐藏；该轮未保存可展示附件")
                else:
                    lines.append("该轮未保存附件")
            elif state.get("status") in {"succeeded", "failed"}:
                lines.append("本轮未保存附件")
        compact_projection = group.get("_compact_projection") is True
        if group.get("deployments") and not compact_projection:
            lines.extend(["", "### 部署记录", "",
                          _deployment_table(group["deployments"], group.get("components"),
                                            group.get("machine_groups"))])
        machine_log_lines = _machine_log_lines(group, key, api, base or upload_base)
        if machine_log_lines:
            lines.extend(["", "### 部署日志", ""])
            lines.extend(machine_log_lines)
        attachment_page = round_attachment_page_url(key, group)
        upload_entry_url = (
            current_upload_entry_url()
            if key == (head, current_round) else ""
        )
        if attachment_page and not upload_entry_url:
            lines.extend(["", f"[打开附件页](<{attachment_page}>)"])
        return lines

    def round_summary(key, group):
        timestamp = first_valid_time(
            group.get("round_started_at"), group.get("started_at"),
        )
        current = key == (head, current_round)
        label = round_label(group, historical=not current)
        if current:
            round_number = group.get("round_number")
            label = (f"第 {round_number} 轮（当前轮次）"
                     if type(round_number) is int and round_number > 0
                     else "当前轮（当前轮次）")
        return (f"<summary>{_escape(label)}"
                f" · {_escape(round_status(group, key))} · {_escape(summary_time(timestamp))}</summary>")

    def round_sort_time(key):
        group = groups[key]
        round_number = group.get("round_number")
        value = first_valid_time(group.get("round_started_at"), group.get("started_at"), group.get("created_at"), group.get("archived_at"), group.get("ended_at"))
        trusted_number = round_number if type(round_number) is int and round_number > 0 else 0
        return (trusted_number, sortable_time(value))
    round_pattern = re.escape(str(head)) + r"-[0-9]+"
    round_read_context_valid = (
        re.fullmatch(r"[0-9a-f]{40}", str(head))
        and isinstance(current_round, str) and re.fullmatch(round_pattern, current_round)
        and state.get("evidence_enabled") is True and bool(api)
    )

    def round_attachment_page_url(key, group):
        if not round_read_context_valid:
            return ""
        origin = base or upload_base
        if key == (head, current_round):
            if state.get("status") == "testing" and state.get("browser_upload_ready") is True:
                upload_url = page_url_at(state.get("upload_capability"), upload_base)
                if upload_url:
                    return upload_url
            return page_url_at(group.get("capability"), origin)
        return stable_read_page_url(key, origin)

    def current_upload_entry_url():
        if (state.get("status") != "testing"
                or state.get("browser_upload_ready") is not True
                or not round_read_context_valid):
            return ""
        candidate = page_url_at(state.get("upload_capability"), upload_base)
        return _trusted_upload_page_url(candidate, repo, pr_number)

    lines = ["<!-- deploy-approval-evidence:start -->"]
    if show_current_round:
        current_key = (head, current_round)
        current_group = groups[current_key]
        lines.extend([
            "<details open>",
            round_summary(current_key, current_group),
            "", *render_round(current_key, current_group), "", "</details>",
        ])
    test_statuses = {"testing", "succeeded", "failed"}
    old = [(key, group) for key, group in groups.items()
           if key != (head, current_round) and group.get("status") in test_statuses]
    old.sort(key=lambda pair: round_sort_time(pair[0]), reverse=True)
    if old:
        lines.append("")
        for key, group in old:
            lines.extend(["<details>",
                          round_summary(key, group),
                          "", *render_round(key, group, historical=True), "", "</details>", ""])
    for url in trusted_archive_urls:
        lines.append(f"[查看原始归档](<{url}>)")
    lines.append("<!-- deploy-approval-evidence:end -->")
    if state.get("status") == "testing":
        upload_entry_url = current_upload_entry_url()
        if upload_entry_url:
            lines.extend([
                "",
                f"[deploy-approval-upload]: <{upload_entry_url}>",
            ])
        else:
            lines.extend([
                "",
                '<a id="deploy-approval-upload-status"></a>',
                "**上传入口状态：** 当前轮安全上传链接暂不可用。请检查证据服务、访问权限和可信 HTTPS 上传入口配置，或联系管理员。",
                "",
                "[deploy-approval-upload]: #deploy-approval-upload-status",
            ])
    return "\n".join(lines)
