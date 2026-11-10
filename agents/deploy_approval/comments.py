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


def _safe_display_text(value) -> str:
    text = str(value if value is not None else "")
    try:
        from .evidence_builder import _redact_secrets

        text = _redact_secrets(text)
    except ImportError:
        pass
    text = re.sub(
        r"(?i)\b(?:[\w.-]*token|[\w.-]*password|[\w.-]*secret|capability|authorization)\b\s*[:=]\s*[^\s,;&]+",
        "[凭据已隐藏]", text,
    )
    text = re.sub(
        r"(?i)\b(?:token|password|secret|capability)\s+[A-Za-z0-9._~+/=-]{8,}",
        "[凭据已隐藏]", text,
    )
    text = re.sub(
        r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}", "Bearer [凭据已隐藏]", text,
    )
    text = re.sub(
        r"(?i)(https?://)[^/@\s]+:[^/@\s]+@", r"\1[凭据已隐藏]@", text,
    )
    return text


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
        result = _safe_display_text(phase)
        if dep.get("error"):
            result += ": " + _safe_display_text(dep["error"])
        if dep.get("updated_at") or dep.get("timestamp"):
            result += " · " + str(dep.get("updated_at") or dep.get("timestamp"))
        ids = dep.get("component_ids") or ([dep["component_id"]] if dep.get("component_id") else snapshot_ids or ["unknown"])
        for cid in ids:
            c = local.get(cid, dep if dep.get("component_id") == cid else {})
            values = tuple(_safe_display_text(value) for value in (
                machine, ip, c.get("target") or cid,
                c.get("variant") or c.get("driver_path") or "—",
                c.get("image_ref") or dep.get("image_ref") or "—", result,
            ))
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
    machine_groups=None, machine_inventory=None, machine_diagnostics=None,
    machine_context=None, test_only: bool | None = None,
) -> str:
    """Render bounded, verified runtime image observations for the Lifecycle Summary."""
    if not isinstance(runtime_checks, list) or not runtime_checks:
        return ""
    component_map = {
        item.get("component_id"): item for item in components or []
        if isinstance(item, dict) and isinstance(item.get("component_id"), str)
    }
    rows = []
    context = machine_context if isinstance(machine_context, dict) else {}
    groups = _record_rows(machine_groups)
    inventory = _machine_inventory_records(machine_inventory)
    diagnostics = _machine_diagnostic_rows(machine_diagnostics)
    group_by_alias = {_machine_alias(row).casefold(): row for row in groups
                      if _machine_alias(row)}
    inventory_by_alias = {_machine_alias(row).casefold(): row for row in inventory
                          if _machine_alias(row)}
    diagnostic_by_alias = {_machine_alias(row).casefold(): row for row in diagnostics
                           if _machine_alias(row)}
    physical_ip_counts = {}
    physical_alias_counts = {}
    for row in inventory:
        primary_ip = _canonical_ip(_machine_ip(row))
        alias = _machine_alias(row).casefold()
        if primary_ip:
            physical_ip_counts[primary_ip] = physical_ip_counts.get(primary_ip, 0) + 1
        if alias:
            physical_alias_counts[alias] = physical_alias_counts.get(alias, 0) + 1
    policy_aliases_by_ip = {}
    for row in groups:
        alias = _machine_alias(row)
        node_host = _canonical_ip(_record_value(row, "node_host"))
        if alias and node_host:
            policy_aliases_by_ip.setdefault(node_host, set()).add(alias.casefold())
    status_text = {
        "match": "MATCH：运行镜像与审批目标一致",
        "mismatch": "MISMATCH：运行镜像与当前审批目标不一致",
        "not_running": "UNAVAILABLE：组件当前未运行，无法比较镜像",
        "unavailable": "UNAVAILABLE：机器连接或状态不可用",
        "identity_changed": "IDENTITY_CHANGED：机器身份绑定已变化，等待重新确认",
    }
    for check in runtime_checks:
        if not isinstance(check, dict):
            continue
        component = component_map.get(check.get("component_id"), {})
        machine = str(check.get("machine", "") or "")
        key = machine.casefold()
        group = group_by_alias.get(key, {})
        physical = inventory_by_alias.get(key, {})
        diagnostic = diagnostic_by_alias.get(key, {})
        supplied_context = context.get(key, {})
        if not isinstance(supplied_context, dict):
            supplied_context = {}
        ip_value = (
            _canonical_ip(_record_value(check, "primary_ip"))
            or _canonical_ip(_record_value(check, "main_ip"))
            or _canonical_ip(_machine_ip(physical))
            or _canonical_ip(_record_value(supplied_context, "primary_ip"))
            or _canonical_ip(_record_value(supplied_context, "main_ip"))
            or _canonical_ip(_record_value(check, "ip"))
            or "未记录"
        )
        owners = _owner_names(check, diagnostic, physical, group)
        owner = _owner_label(
            owners, test_only=test_only is True, mention=test_only is False,
        )
        # An explicit empty value means the current controller has revoked the
        # selector. Only use a live machine group when the check omitted it;
        # never revive a command parsed from Lifecycle Markdown history.
        raw_selector = (
            check.get("command_selector") if "command_selector" in check
            else _record_value(group, "command_selector", default="")
        )
        command_selector = str(raw_selector or "").strip()
        selected_ip = _canonical_ip(command_selector)
        selected_as_alias = bool(
            command_selector == machine and machine
            and not _canonical_ip(command_selector)
        )
        identity_values = [value for value in (
            _field_flag(record, "identity_confirmed", "machine_identity_confirmed")
            for record in (check, diagnostic, group)
        ) if value is not None]
        identity_confirmed = (
            False if False in identity_values else
            True if True in identity_values else None
        )
        group_node_host = next((value for value in (
            _canonical_ip(_record_value(group, "node_host")),
            _canonical_ip(_record_value(group, "policy_node_host")),
            _canonical_ip(_record_value(diagnostic, "node_host")),
            _canonical_ip(_record_value(diagnostic, "policy_node_host")),
            _canonical_ip(_record_value(check, "node_host")),
            _canonical_ip(_record_value(check, "policy_node_host")),
            _canonical_ip(_record_value(group, "primary_ip")),
        ) if value), "")
        duplicate_ip = bool(
            physical_ip_counts.get(ip_value, 0) > 1
            or _field_flag(check, "duplicate_ip") is True
            or _field_flag(diagnostic, "duplicate_ip") is True
            or _field_flag(group, "duplicate_ip") is True
            or "PRIMARY_IP_AMBIGUOUS" in _display_values(
                _record_value(group, "blocking_reasons", default=[])
            )
            or len(policy_aliases_by_ip.get(group_node_host, set())) > 1
        )
        repo_authorization = next((value for value in (
            _field_flag(record, "repository_authorized", "repo_authorized")
            for record in (check, diagnostic, group)
        ) if value is not None), None)
        binding_matches = bool(ip_value and group_node_host and ip_value == group_node_host)
        selector_is_safe = bool(
            selected_ip == ip_value and not duplicate_ip
            and (not group_node_host or binding_matches)
        )
        if selected_as_alias:
            selector_is_safe = bool(
                duplicate_ip and identity_confirmed is True and binding_matches
                and physical_alias_counts.get(key, 0) == 1
                and _machine_alias(physical).casefold() == key
            )
        if (not selector_is_safe
                or _field_flag(group, "approval_available") is False
                or repo_authorization is False
                or (group and group_node_host and group_node_host != ip_value)
                or identity_confirmed is False):
            command_selector = ""
        target_image = _runtime_digest(check.get("target_image", ""))
        observed_image = _runtime_digest(check.get("observed_image", ""))
        checked_at = str(check.get("checked_at", "") or "未扫描")
        status = str(check.get("status", "unavailable") or "unavailable").casefold()
        if status not in status_text:
            status = "unavailable"
        next_step = "等待下次状态检查"
        if lifecycle_status == "deploying":
            status_label = status_text[status]
            status_text_value = f"部署进行中；上次只读检查：{status_label}"
            next_step = "正在部署，请等待结果；不要重复审批"
        elif status == "match":
            next_step = "MATCH：无需重新部署。"
        elif status == "mismatch":
            if command_selector:
                next_step = (
                    f"MISMATCH：直接重新审批：`/approve_deploy machine={_escape(command_selector)}`。"
                    "`/request_deploy` 可选。"
                )
            else:
                next_step = "MISMATCH：当前身份或仓库授权未确认，暂不生成审批命令。"
        elif status == "unavailable":
            next_step = "UNAVAILABLE：机器连接或状态不可用，不能确认是否需要重新部署。"
        elif status == "identity_changed":
            next_step = "身份绑定变化，等待重新确认；当前不生成审批命令。"
        elif status == "not_running":
            next_step = "组件未运行，无法比较镜像；先确认机器状态，不生成重新部署命令。"
        rows.append("| " + " | ".join((
            f"`{_escape(machine)}`",
            f"`{_escape(ip_value)}`",
            owner,
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
        "| 机器 | 主 IP | Owner | 组件 | 审批目标 digest | 当前运行 digest | 状态 | 最后检查 | 下一步 |",
        "|---|---|---|---|---|---|---|---|---|",
        *rows,
        "",
        "MISMATCH 行给出的直接审批命令可以单独使用；`/request_deploy` 可选。",
        _RUNTIME_CHECKS_END,
    ])


def update_runtime_image_checks(
    markdown: str, runtime_checks, components, *, lifecycle_status: str = "",
    machine_groups=None, machine_inventory=None, machine_diagnostics=None,
    test_only: bool | None = None,
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
    if test_only is None and "[test only]" in updated.casefold():
        test_only = True
    machine_context = _lifecycle_machine_display_context(updated)
    block = runtime_image_checks_block(
        runtime_checks, components, lifecycle_status=lifecycle_status,
        machine_groups=machine_groups, machine_inventory=machine_inventory,
        machine_diagnostics=machine_diagnostics, machine_context=machine_context,
        test_only=test_only,
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
    build_rows = builds if isinstance(builds, list) else []
    builds_passed = bool(build_rows) and all(
        _field_flag(build, "success") is True
        and _field_flag(build, "deployable") is True
        for build in build_rows
    )
    evidence = review_evidence if isinstance(review_evidence, dict) else {}
    review_passed = (
        type(evidence.get("build_comment_id")) is int
        and evidence.get("build_comment_id", 0) > 0
        and type(evidence.get("code_review_comment_id")) is int
        and evidence.get("code_review_comment_id", 0) > 0
        and evidence.get("resolved_head_sha") == head_sha
    )
    approval_evidence_ready = builds_passed and review_passed
    deployable_ids = {
        component.get("component_id") for component in components or []
        if isinstance(component, dict) and isinstance(component.get("component_id"), str)
        and component.get("component_id")
    }
    if not deployable_ids:
        deployable_ids = {
            _record_value(build, "component_id") for build in build_rows
            if _field_flag(build, "success") is True
            and _field_flag(build, "deployable") is True
            and isinstance(_record_value(build, "component_id"), str)
            and _record_value(build, "component_id")
        }
    commands = []
    full_machine_aliases = set()
    group_rows = [group for group in (machine_groups or [])
                  if isinstance(group, dict)]
    alias_counts = {}
    primary_ip_counts = {}
    aliases_by_node_host = {}

    def ipv4(value):
        if not isinstance(value, str) or not value.strip():
            return ""
        try:
            address = ipaddress.ip_address(value.strip())
        except ValueError:
            return ""
        return str(address) if address.version == 4 else ""

    for group in group_rows:
        alias = group.get("alias")
        if isinstance(alias, str) and alias.strip():
            key = alias.strip().casefold()
            alias_counts[key] = alias_counts.get(key, 0) + 1
        primary_ip = ipv4(group.get("primary_ip"))
        node_host = ipv4(group.get("node_host"))
        if primary_ip:
            primary_ip_counts[primary_ip] = primary_ip_counts.get(primary_ip, 0) + 1
        if isinstance(alias, str) and alias and node_host:
            aliases_by_node_host.setdefault(node_host, set()).add(alias.casefold())

    if approval_evidence_ready:
        seen_commands = set()
        for group in group_rows:
            compatible = group.get("component_ids")
            if not isinstance(compatible, list):
                continue
            compatible_ids = {
                component_id for component_id in compatible
                if isinstance(component_id, str) and component_id
            }
            if not deployable_ids or not deployable_ids.intersection(compatible_ids):
                continue
            alias = group.get("alias")
            if (not isinstance(alias, str)
                    or re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", alias) is None
                    or alias_counts.get(alias.casefold(), 0) != 1
                    or group.get("approval_available") is not True
                    or group.get("repo_authorized") is not True
                    or group.get("identity_confirmed") is not True):
                continue
            node_id = group.get("node_id")
            primary_ip = ipv4(group.get("primary_ip"))
            node_host = ipv4(group.get("node_host"))
            if (not isinstance(node_id, str) or not node_id.strip()
                    or not primary_ip or primary_ip != node_host):
                continue

            reasons = set(_display_values(group.get("blocking_reasons")))
            ambiguous_ip = bool(
                group.get("duplicate_ip") is True
                or "PRIMARY_IP_AMBIGUOUS" in reasons
                or primary_ip_counts.get(primary_ip, 0) > 1
                or len(aliases_by_node_host.get(node_host, set())) > 1
            )
            # A present selector is authoritative, including an empty value.
            # Only legacy groups with no selector field may use a verified IP.
            if "command_selector" in group:
                raw_selector = group.get("command_selector")
            else:
                raw_selector = (
                    group.get("ip") if group.get("ip_command_safe") is True
                    else ""
                )
            selector = raw_selector.strip() if isinstance(raw_selector, str) else ""
            if not selector:
                continue

            selector_ip = ipv4(selector)
            if selector_ip:
                if (selector_ip != primary_ip or selector_ip != node_host
                        or group.get("ip_command_safe") is not True
                        or ambiguous_ip):
                    continue
                command_selector = selector_ip
            else:
                if (selector != alias or not ambiguous_ip
                        or group.get("ip_command_safe") is not False
                        or (not aliases_by_node_host.get(node_host)
                            or alias.casefold() not in aliases_by_node_host[node_host])):
                    continue
                command_selector = alias

            command = f"/approve_deploy machine={command_selector}"
            if command in seen_commands:
                continue
            seen_commands.add(command)
            label = f"{_escape(alias)}（主 IP：`{primary_ip}`）"
            commands.append(f"- {label}：`{command}`")
            if deployable_ids.issubset(compatible_ids):
                full_machine_aliases.add(alias.casefold())

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
        "机器负责人请按下方已验证的机器命令审批：",
        "",
    ]
    if commands:
        lines.extend(commands)
    elif approval_evidence_ready:
        lines.append("当前没有通过仓库授权、机器身份和组件兼容校验的可执行审批命令。")
    else:
        lines.append("Review/Build 证据尚未确认通过，暂不显示可执行部署审批方式。")
    if approval_evidence_ready and len(full_machine_aliases) == 1:
        lines.extend(["", "当前只有一台完整适配机器，也可以直接使用简写：", "", "`/approve_deploy`"])
    if approval_evidence_ready:
        lines.extend(["", "也可以选择用 `/request_deploy` 申请部署。"])
    lines.extend(["", "### Review evidence", ""])
    lines.append(
        f"- **Build Result:** {'success' if builds_passed else 'failed'}"
    )
    if evidence.get("test_skipped", False):
        lines.append("- **Tests:** skipped")
    elif evidence.get("test_comment_id", 0) == 0:
        lines.append("- **Tests:** not run")
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


_V44_CANDIDATE_REVIEWERS = ("Haohao-end", "shiguangchuan", "kentcyq")
_OWNER_LOGIN_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38}\Z")


def _record_value(record, *names, default=None):
    for name in names:
        if isinstance(record, dict) and name in record:
            return record[name]
        if record is not None and hasattr(record, name):
            return getattr(record, name)
    return default


def _record_rows(value, container_names=()):
    if value is None:
        return []
    for name in container_names:
        nested = _record_value(value, name)
        if isinstance(nested, (list, tuple)):
            return list(nested)
    if isinstance(value, dict):
        if any(name in value for name in ("alias", "machine_alias", "machine_name", "machine", "name")):
            return [value]
        rows = []
        display_fields = (
            "index", "number", "brand", "manufacturer", "model", "model_name",
            "alias", "name", "machine_name", "main_ip", "primary_ip", "ip",
            "node_host", "auxiliary_ip", "auxiliary_ips", "aux_ips", "secondary_ips",
            "owner", "owners", "owner_login", "owner_logins", "platform", "platform_name",
            "platforms",
            "authorized", "repository_authorized", "identity_confirmed",
            "agent_core_credentials_available", "agent_core_status",
            "platform_confirmed", "component_compatible",
            "components_compatible", "component_ids", "safe_to_approve", "status",
            "reason", "reasons", "compatibility_status", "duplicate_ip",
            "ip_conflicts",
        )
        for key, row in value.items():
            key_text = str(key)
            try:
                ip_key = str(ipaddress.ip_address(key_text))
            except ValueError:
                ip_key = ""
            if isinstance(row, dict):
                normalized = dict(row)
                if ip_key:
                    normalized.setdefault("ip", ip_key)
                else:
                    normalized.setdefault("alias", key_text)
                rows.append(normalized)
            elif row is not None:
                normalized = {name: getattr(row, name) for name in display_fields
                             if hasattr(row, name)}
                if ip_key:
                    normalized.setdefault("ip", ip_key)
                else:
                    normalized.setdefault("alias", key_text)
                rows.append(normalized)
        return rows
    if isinstance(value, (list, tuple)):
        return list(value)
    try:
        return list(value)
    except TypeError:
        return []


def _machine_inventory_records(value):
    records = _record_rows(value, ("machines", "inventory", "items"))
    if records:
        return records
    try:
        from . import machine_inventory as inventory_module
    except ImportError:
        return []
    for name in ("MACHINE_INVENTORY", "PHYSICAL_MACHINES", "INVENTORY", "_INVENTORY"):
        value = getattr(inventory_module, name, None)
        if value is not None:
            records = _record_rows(value, ("machines", "inventory", "items"))
            if records:
                return records
    return []


def _display_values(value):
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        values = []
    return [str(item).strip() for item in values
            if isinstance(item, str) and item.strip()]


def _field_flag(record, *names):
    value = _record_value(record, *names)
    if type(value) is bool:
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold().replace("_", "-")
        if normalized in {"true", "yes", "authorized", "available", "confirmed",
                          "compatible", "safe", "ready", "reachable", "configured",
                          "present", "exists"}:
            return True
        if normalized in {"false", "no", "unauthorized", "unavailable", "unknown",
                          "unconfirmed", "incompatible", "unsafe", "offline", "missing",
                          "absent", "not-configured", "not-found"}:
            return False
    return None


def _machine_alias(record):
    value = _record_value(record, "alias", "machine_alias", "machine", "machine_name", "name")
    return value.strip() if isinstance(value, str) else ""


def _machine_ip(record):
    # ``ip`` is intentionally empty for some ambiguous repository bindings.
    # Keep looking for the physical primary address or the policy node_host.
    for name in ("main_ip", "primary_ip", "primary_ip_address", "ip", "node_host"):
        value = _record_value(record, name)
        if not isinstance(value, str) or not value.strip():
            continue
        try:
            return str(ipaddress.ip_address(value.strip()))
        except ValueError:
            return value.strip()
    return ""


def _canonical_ip(value):
    if not isinstance(value, str):
        return ""
    try:
        return str(ipaddress.ip_address(value.strip()))
    except ValueError:
        return ""


def _test_only_status(*records):
    for record in records:
        if isinstance(record, (list, tuple)):
            if _test_only_status(*record):
                return True
            continue
        value = _field_flag(record, "test_only", "is_test_only")
        if value is True:
            return True
        title = _record_value(record, "title", "pr_title")
        if isinstance(title, str) and "[test only]" in title.casefold():
            return True
    return False


def _owner_names(*records):
    def clean(values):
        normalized = [value[1:] if value.startswith("@") else value
                      for value in _display_values(values)]
        ignored = {"负责人待配置", "未知", "unknown", "pending", "none"}
        return list(dict.fromkeys(
            value for value in normalized if value.casefold() not in ignored
        ))

    # Repository policy is current for the PR's repository and always takes
    # priority over the shared physical-inventory owner snapshot.
    for record in records:
        owners = clean(_record_value(record, "policy_owners"))
        if owners:
            return owners
    # Some current machine-group records expose policy owners as ``owners``.
    for record in records:
        if _field_flag(record, "repo_authorized") is True:
            owners = clean(_record_value(
                record, "owners", "owner", "owner_login", "owner_logins",
            ))
            if owners:
                return owners
    # Prefer explicit physical owners, then older inventory-compatible fields.
    for record in records:
        owners = clean(_record_value(record, "physical_owners"))
        if owners:
            return owners
    for record in records:
        owners = clean(_record_value(record, "owners", "owner", "owner_login", "owner_logins"))
        if owners:
            return owners
    return []


def _owner_label(owners, *, test_only=False, mention=True):
    labels = []
    for owner in owners:
        if _OWNER_LOGIN_RE.fullmatch(owner):
            labels.append(f"@{owner}" if mention and not test_only else owner)
        else:
            labels.append(owner.replace("@", "＠"))
    return " ".join(_escape(owner) for owner in labels) if labels else "负责人待配置"


def _machine_diagnostic_rows(value):
    return _record_rows(
        value, ("machines", "machine_diagnostics", "items", "rows"),
    )


def _repo_policy_display_status(repo, status, diagnostics):
    if not isinstance(status, dict):
        status = {}
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    app_authorized = _field_flag(
        status, "github_app_authorized", "app_authorized", "authorized",
    )
    if app_authorized is None:
        app_state = _record_value(
            status, "github_app_status", "app_status", "installation_status",
            "status", default="",
        )
        app_authorized = isinstance(app_state, str) and app_state.strip().casefold() in {
            "authorized", "installed", "active", "available",
        }
    machines_configured = _field_flag(status, "machines_configured", "machines_yaml_exists")
    reviewers_configured = _field_flag(status, "reviewers_configured", "test_reviewers_yaml_exists")
    missing_files = set(_display_values(status.get("missing_files")))
    machines_yaml_state = status.get("machines_yaml_status")
    reviewers_yaml_state = status.get("test_reviewers_yaml_status")
    if machines_configured is None:
        machines_configured = (
            False if "machines.yaml" in missing_files
            or isinstance(machines_yaml_state, str)
            and machines_yaml_state.casefold() in {"missing", "absent", "not-found"} else
            True if isinstance(machines_yaml_state, str)
            and machines_yaml_state.casefold() in {"configured", "present", "valid", "exists"} else
            None
        )
    if reviewers_configured is None:
        reviewers_configured = (
            False if "test_reviewers.yaml" in missing_files
            or isinstance(reviewers_yaml_state, str)
            and reviewers_yaml_state.casefold() in {"missing", "absent", "not-found"} else
            True if isinstance(reviewers_yaml_state, str)
            and reviewers_yaml_state.casefold() in {"configured", "present", "valid", "exists"} else
            None
        )
    if not app_authorized:
        app_status = "未授权或无法确认"
    else:
        app_status = "已授权"
    machines_status = (
        "已配置" if machines_configured is True else
        "main 缺少" if machines_configured is False or "machines.yaml" in missing_files else
        "配置无效" if isinstance(machines_yaml_state, str)
        and machines_yaml_state.casefold() in {"invalid", "corrupt", "malformed"} else
        "未配置或无法确认"
    )
    reviewers_status = (
        "已配置" if reviewers_configured is True else
        "main 缺少" if reviewers_configured is False or "test_reviewers.yaml" in missing_files else
        "配置无效" if isinstance(reviewers_yaml_state, str)
        and reviewers_yaml_state.casefold() in {"invalid", "corrupt", "malformed"} else
        "未配置或无法确认"
    )
    authorized_machines = _record_value(
        status, "authorized_machines", "machines",
        default=_record_value(diagnostics, "authorized_machines", "authorized"),
    )
    if isinstance(authorized_machines, dict):
        authorized_aliases = {
            str(alias).strip().casefold() for alias, machine in authorized_machines.items()
            if machine is not False
        }
        authorized_count = len(authorized_aliases)
    elif isinstance(authorized_machines, (list, tuple, set)):
        authorized_aliases = {
            str(_record_value(item, "alias", "name", default=item)).strip().casefold()
            for item in authorized_machines
        }
        authorized_count = len(authorized_aliases)
    else:
        authorized_aliases = set()
        explicit_count = _record_value(
            status, "authorized_machine_count",
            default=_record_value(diagnostics, "authorized_machine_count"),
        )
        authorized_count = (explicit_count if type(explicit_count) is int and explicit_count >= 0
                            else 0 if machines_configured is False else None)
    if machines_configured is False:
        authorized_aliases = set()
        authorized_count = 0
    reviewers = _display_values(_record_value(
        status, "reviewers", "test_reviewers", "acceptance_reviewers",
        default=_record_value(diagnostics, "reviewers", "test_reviewers",
                              "acceptance_reviewers"),
    ))
    candidate_reviewers = _display_values(_record_value(
        status, "candidate_reviewers",
        default=_record_value(diagnostics, "candidate_reviewers"),
    ))
    candidate_reviewers = list(dict.fromkeys(
        [*candidate_reviewers, *_V44_CANDIDATE_REVIEWERS]
    ))
    reviewer_display = (
        "验收人：" + "、".join(_escape(name) for name in reviewers)
        if reviewers else
        "验收人：未配置或名单未提供"
    )
    count_display = str(authorized_count) if authorized_count is not None else "未提供"
    return {
        "repo": repo,
        "app_authorized": app_authorized,
        "app_status": app_status,
        "machines_configured": machines_configured is True,
        "machines_configured_known": machines_configured is not None,
        "machines_status": machines_status,
        "reviewers_configured": reviewers_configured is True,
        "reviewers_status": reviewers_status,
        "authorized_aliases": authorized_aliases,
        "authorized_count": authorized_count,
        "reviewers": reviewers,
        "reviewer_display": reviewer_display,
        "candidate_reviewers": candidate_reviewers,
        "count_display": count_display,
    }


def _runtime_digest(value):
    if not isinstance(value, str) or not value:
        return "未取得"
    match = re.search(r"(?:@)?sha256:([0-9a-fA-F]{64})", value)
    return f"sha256:{match.group(1).lower()}" if match else "未取得"


def _lifecycle_machine_display_context(markdown):
    """Read already-rendered safe machine identities for runtime-only refreshes."""
    result = {}
    if not isinstance(markdown, str):
        return result

    def cells(line):
        values = line.strip().strip("|").split("|")
        return [html.unescape(value.strip()).strip().strip("`").strip()
                for value in values]

    section = ""
    for line in markdown.splitlines():
        if line.startswith("### "):
            heading = line.casefold()
            section = (
                "inventory" if "machine inventory" in heading
                else "compatible" if "compatible machines" in heading
                else ""
            )
            continue
        if not line.startswith("|") or line.lstrip().startswith("|---"):
            continue
        row = cells(line)
        if section == "inventory" and len(row) >= 7:
            alias = row[3]
            if alias and alias.casefold() not in {"别名", "alias"}:
                entry = result.setdefault(alias.casefold(), {})
                entry["alias"] = alias
                entry["primary_ip"] = row[4] if row[4] not in {"—", "未提供"} else ""
                if row[6] and row[6] not in {"—", "负责人待配置"}:
                    entry["owner"] = row[6]
        elif section == "compatible" and len(row) >= 4:
            alias = row[0]
            if not alias or alias.casefold() in {"机器名称", "machine"}:
                continue
            entry = result.setdefault(alias.casefold(), {})
            entry["alias"] = alias
            if row[1] not in {"—", "未提供"}:
                entry["primary_ip"] = row[1]
            if row[2] and row[2] not in {"—", "负责人待配置"}:
                entry["owner"] = row[2]
    return result


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
    machine_inventory=None,
    machine_diagnostics=None,
    repository_policy_status=None,
    runtime_checks=None,
) -> str:
    """Render a repository-scoped lifecycle with physical and authorized machines."""
    if (not isinstance(display_status, str)
            or display_status not in {"deploy-requested", "deploying"}):
        raise ValueError("display_status must be deploy-requested or deploying")
    inventory_rows = _machine_inventory_records(machine_inventory)
    diagnostic_rows = _machine_diagnostic_rows(machine_diagnostics)
    group_rows = _record_rows(machine_groups)
    policy = _repo_policy_display_status(repo, repository_policy_status, machine_diagnostics)
    test_only = _test_only_status(
        repository_policy_status, machine_diagnostics, group_rows, components,
    ) or any(
        isinstance(note, str) and "[test only]" in note.casefold()
        for note in [last_lifecycle_event, *(gate_note or [])]
    )

    def alias_key(record):
        return _machine_alias(record).casefold()

    diagnostic_by_alias = {alias_key(row): row for row in diagnostic_rows if alias_key(row)}
    inventory_by_alias = {alias_key(row): row for row in inventory_rows if alias_key(row)}
    group_by_alias = {alias_key(row): row for row in group_rows if alias_key(row)}
    diagnostics_by_ip = {
        _canonical_ip(_machine_ip(row)): row for row in diagnostic_rows
        if _canonical_ip(_machine_ip(row))
    }
    inventory_by_ip = {
        _canonical_ip(_machine_ip(row)): row for row in inventory_rows
        if _canonical_ip(_machine_ip(row))
    }
    components = components if isinstance(components, list) else []
    component_ids = {
        item.get("component_id") for item in components
        if isinstance(item, dict) and isinstance(item.get("component_id"), str)
        and item.get("component_id")
    }
    known_aliases_by_ip = {}
    inventory_ip_counts = {}
    inventory_alias_counts = {}
    policy_aliases_by_ip = {}
    policy_alias_counts = {}
    for row in inventory_rows:
        ip_value = _canonical_ip(_machine_ip(row))
        alias = _machine_alias(row)
        if alias:
            alias_key_value = alias.casefold()
            inventory_alias_counts[alias_key_value] = (
                inventory_alias_counts.get(alias_key_value, 0) + 1
            )
        if ip_value:
            inventory_ip_counts[ip_value] = inventory_ip_counts.get(ip_value, 0) + 1
            known_aliases_by_ip.setdefault(ip_value, []).append(alias or "别名未提供")
    for row in group_rows:
        alias = _machine_alias(row)
        alias_key_value = alias.casefold()
        node_host = next((value for value in (
            _canonical_ip(_record_value(row, "node_host")),
            _canonical_ip(_record_value(row, "policy_node_host")),
            _canonical_ip(_record_value(row, "primary_ip")),
        ) if value), "")
        if alias:
            policy_alias_counts[alias_key_value] = policy_alias_counts.get(alias_key_value, 0) + 1
        if node_host:
            policy_aliases_by_ip.setdefault(node_host, []).append(alias or "别名未提供")
        ip_value = _canonical_ip(_machine_ip(row))
        if ip_value and alias and not inventory_ip_counts.get(ip_value):
            aliases = known_aliases_by_ip.setdefault(ip_value, [])
            if alias not in aliases:
                aliases.append(alias)

    def machine_parts(machine):
        alias = _machine_alias(machine)
        key = alias.casefold()
        ip_value = _canonical_ip(_machine_ip(machine))
        diagnostic = diagnostic_by_alias.get(key)
        physical = inventory_by_alias.get(key)
        if ip_value and inventory_ip_counts.get(ip_value, 0) == 1:
            diagnostic = diagnostic or diagnostics_by_ip.get(ip_value, {})
            physical = physical or inventory_by_ip.get(ip_value)
        physical = physical or machine
        group = group_by_alias.get(key) or {}
        if not group and diagnostic:
            group = diagnostic
        return alias, ip_value, diagnostic, physical, group

    def machine_flags(machine):
        alias, ip_value, diagnostic, physical, group = machine_parts(machine)
        records = (diagnostic, group, physical)
        duplicate_aliases = known_aliases_by_ip.get(ip_value, [])
        physical_ip = _canonical_ip(_record_value(
            physical, "primary_ip", "main_ip", "primary_ip_address", "ip",
        )) or ip_value
        policy_node_host = next((value for value in (
            _canonical_ip(_record_value(group, "node_host")),
            _canonical_ip(_record_value(group, "policy_node_host")),
            _canonical_ip(_record_value(diagnostic, "node_host")),
            _canonical_ip(_record_value(diagnostic, "policy_node_host")),
            _canonical_ip(_record_value(group, "primary_ip")),
            _canonical_ip(_record_value(diagnostic, "primary_ip")),
        ) if value), "")
        group_primary_ip = next((value for value in (
            _canonical_ip(_record_value(group, "primary_ip")),
            _canonical_ip(_record_value(group, "main_ip")),
            _canonical_ip(_record_value(group, "primary_ip_address")),
        ) if value), "")
        duplicate_ip = bool(
            inventory_ip_counts.get(physical_ip, 0) > 1
            or len({name.casefold() for name in duplicate_aliases}) > 1
            or _field_flag(diagnostic, "duplicate_ip") is True
            or "PRIMARY_IP_AMBIGUOUS" in _display_values(
                _record_value(group, "blocking_reasons", default=[])
            )
            or len({name.casefold() for name in policy_aliases_by_ip.get(
                policy_node_host, [],
            )}) > 1
        )
        authorized = _field_flag(
            diagnostic, "repository_authorized", "authorized", "machine_authorized",
        )
        if authorized is None:
            authorized = alias.casefold() in policy["authorized_aliases"]
        authorized = bool(
            policy["app_authorized"] and policy["machines_configured"] and authorized
        )
        identity_flags = [value for value in (
            _field_flag(record, "identity_confirmed", "machine_identity_confirmed")
            for record in records
        ) if value is not None]
        identity_flag = (
            False if False in identity_flags else
            True if True in identity_flags else None
        )
        identity_confirmed = identity_flag is True
        configured_flags = [value for value in (
            _field_flag(record, "agent_core_credentials_configured")
            for record in records
        ) if value is not None]
        if False in configured_flags:
            credentials_configured = False
        elif True in configured_flags:
            credentials_configured = True
        else:
            legacy_configured_flags = [value for value in (
                _field_flag(record, "agent_core_credentials_available")
                for record in records
            ) if value is not None]
            credentials_configured = (
                False if False in legacy_configured_flags else
                True if True in legacy_configured_flags else None
            )

        core_status_values = [
            str(value).strip().casefold().replace("_", "-")
            for value in (_record_value(record, "agent_core_status") for record in records)
            if isinstance(value, str) and value.strip()
        ]
        denied_core_statuses = {
            "credential-missing", "credentials-missing", "missing-credentials",
            "unavailable", "offline", "unreachable",
        }
        if any(value in denied_core_statuses for value in core_status_values):
            agent_core_status = next(
                value for value in core_status_values if value in denied_core_statuses
            )
        elif any(value in {"available", "healthy", "reachable", "ok"}
                 for value in core_status_values):
            agent_core_status = "available"
        elif core_status_values:
            agent_core_status = "unknown"
        else:
            legacy_health_flags = [value for value in (
                _field_flag(record, "agent_core_available", "agent_core_reachable")
                for record in records
            ) if value is not None]
            agent_core_status = (
                "unavailable" if False in legacy_health_flags else
                "available" if True in legacy_health_flags else "unknown"
            )

        explicit_core_failure = agent_core_status in denied_core_statuses
        credentials_available = bool(
            not explicit_core_failure
            and credentials_configured is not False
            and (
                agent_core_status == "available"
                or agent_core_status == "unknown" and credentials_configured is True
            )
        )
        credentials_flag = (
            False if explicit_core_failure or credentials_configured is False else
            True if credentials_available else None
        )
        platform_flag = next((value for value in (
            _field_flag(record, "platform_confirmed", "platform_known")
            for record in records
        ) if value is not None), None)
        platform_confirmed = platform_flag is True
        if platform_flag is None:
            platform_value = _record_value(
                diagnostic, "platform", "platform_name", "platforms",
                default=_record_value(physical, "platform", "platform_name", "platforms"),
            )
            platform_values = _display_values(platform_value)
            platform_confirmed = bool(platform_values) and all(
                value.casefold() not in {"unknown", "unconfirmed", "未确认", "未知"}
                for value in platform_values
            )
        group_component_ids = _record_value(group, "component_ids", default=[])
        if not isinstance(group_component_ids, (list, tuple, set)):
            group_component_ids = []
        diagnostic_component_ids = _record_value(diagnostic, "component_ids", default=[])
        if not isinstance(diagnostic_component_ids, (list, tuple, set)):
            diagnostic_component_ids = []
        machine_component_ids = {
            component_id for component_id in group_component_ids
            if isinstance(component_id, str) and component_id
        }
        diagnosed_component_ids = {
            component_id for component_id in diagnostic_component_ids
            if isinstance(component_id, str) and component_id
        }
        component_flags = [value for value in (
            _field_flag(record, "component_compatible", "components_compatible")
            for record in (diagnostic, group)
        ) if value is not None]
        component_flag = (
            False if False in component_flags else
            True if True in component_flags else None
        )
        component_coverage_confirmed = bool(
            machine_component_ids
            and diagnosed_component_ids
            and machine_component_ids.issubset(diagnosed_component_ids)
        )
        if component_flag is False:
            component_compatible = False
        else:
            component_compatible = component_coverage_confirmed
        ip_identity_matches = bool(
            physical_ip and policy_node_host and physical_ip == policy_node_host
            and (not group_primary_ip or group_primary_ip == physical_ip)
        )
        alias_identity_matches = bool(
            alias
            and _machine_alias(physical).casefold() == alias.casefold()
            and _machine_alias(group).casefold() == alias.casefold()
            and inventory_alias_counts.get(alias.casefold(), 0) == 1
            and policy_alias_counts.get(alias.casefold(), 0) == 1
        )
        identity_confirmed = bool(
            identity_flag is True and alias_identity_matches and ip_identity_matches
        )
        identity_mismatch = False
        try:
            identity_mismatch = bool(
                physical_ip and policy_node_host and physical_ip != policy_node_host
            )
        except ValueError:
            pass
        safe_flag = _field_flag(diagnostic, "safe_to_approve", "approval_safe")
        provided_selector = str(
            _record_value(group, "command_selector", default="")
            or _record_value(diagnostic, "command_selector", default="")
            or ""
        ).strip()
        group_approval_flag = _field_flag(group, "approval_available")
        physical_ip_unique = bool(
            physical_ip
            and inventory_ip_counts.get(physical_ip, 0) == 1
            and len({name.casefold() for name in policy_aliases_by_ip.get(
                policy_node_host, [],
            )}) == 1
            and not duplicate_ip
        )
        if identity_confirmed and authorized and group_approval_flag is not False:
            if physical_ip_unique and provided_selector in {"", physical_ip}:
                command_selector = physical_ip
            elif (duplicate_ip and alias_identity_matches
                  and provided_selector in {"", alias}):
                command_selector = alias
            else:
                command_selector = ""
        else:
            command_selector = ""
        group_approval_available = (
            group_approval_flag is True
        )
        safe_flag_allows = safe_flag is not False or bool(
            duplicate_ip and command_selector == alias and group_approval_available
            and component_coverage_confirmed
        )
        safe = bool(
            authorized and identity_confirmed and credentials_available
            and platform_confirmed and component_compatible and command_selector
            and safe_flag_allows
        )
        reasons = []
        supplied_reasons = _record_value(diagnostic, "reasons", "reason", "diagnostic")
        reasons.extend(_safe_display_text(reason) for reason in _display_values(supplied_reasons))
        if not policy["app_authorized"]:
            reasons.append("GitHub App 对该仓库的授权未确认。")
        elif not policy["machines_configured"]:
            reasons.append(
                "当前仓库 main 缺少有效的 machines.yaml。"
                if policy["machines_status"] == "main 缺少"
                else "当前仓库机器授权状态未确认。"
            )
        elif not authorized:
            reasons.append("当前仓库 main 缺少机器授权。")
        if duplicate_ip:
            conflicting_aliases = _display_values(_record_value(
                diagnostic, "ip_conflicts", "conflicting_aliases",
            ))
            if inventory_ip_counts.get(ip_value, 0) > 1:
                conflicting_aliases.extend(duplicate_aliases)
            conflict_labels = sorted(
                set(conflicting_aliases), key=str.casefold,
            )
            reasons.append(
                "主 IP 存在冲突，涉及别名："
                + ("、".join(conflict_labels) or "冲突别名待确认") + "。"
            )
        if identity_mismatch:
            reasons.append("物理主 IP 与当前仓库 node_host 不一致，禁止审批命令。")
        elif not identity_confirmed:
            reasons.append("机器别名、node_id 与主 IP 的身份绑定未确认。")
        if credentials_flag is False:
            reasons.append("Agent Core 凭据或连接不可用。")
        elif not credentials_available:
            reasons.append("Agent Core 凭据或连接状态未确认。")
        if platform_flag is False:
            reasons.append("机器平台状态未确认。")
        elif not platform_confirmed:
            reasons.append("机器平台尚未确认。")
        if not component_compatible:
            reasons.append(
                "当前 PR 所需组件与该机器不兼容。"
                if component_flag is False else "当前 PR 组件匹配结果尚未确认。"
            )
        if safe_flag is False and not (
                duplicate_ip and command_selector == alias and group_approval_available):
            reasons.append("机器诊断未允许安全审批。")
        return {
            "alias": alias,
            "ip": physical_ip or ip_value or _machine_ip(machine),
            "command_selector": command_selector,
            "diagnostic": diagnostic,
            "physical": physical,
            "group": group,
            "owners": _owner_names(diagnostic, physical, group),
            "authorized": authorized,
            "identity_flag": identity_flag,
            "identity_confirmed": identity_confirmed,
            "credentials_flag": credentials_flag,
            "credentials_available": credentials_available,
            "platform_flag": platform_flag,
            "platform_confirmed": platform_confirmed,
            "component_flag": component_flag,
            "component_compatible": component_compatible,
            "duplicate_ip": duplicate_ip,
            "safe": safe,
            "reasons": list(dict.fromkeys(reasons)),
        }

    machine_states = [machine_flags(row) for row in inventory_rows]
    safe_machines = {
        item["alias"].casefold(): item for item in machine_states
        if item["safe"] and item["alias"]
    }
    machine_state_by_alias = {
        item["alias"].casefold(): item for item in machine_states if item["alias"]
    }
    physical_machine_count = len(inventory_rows)
    inventory_lines = [
        "### Machine Inventory · 机器完整清单", "",
        "| 编号 | 品牌 | 型号 | 别名 | 主 IP | 辅助 IP | Owner | 仓库授权 | Agent Core | 平台 | 当前 PR 组件兼容性 | 原因 |",
        "|---:|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for ordinal, (machine, state) in enumerate(zip(inventory_rows, machine_states), 1):
        index = _record_value(machine, "index", "number", "id", default=ordinal)
        brand = _record_value(machine, "brand", "manufacturer", default="未提供")
        model = _record_value(machine, "model", "model_name", default="未提供")
        alias = state["alias"] or "未提供"
        main_ip = _machine_ip(machine) or "未提供"
        auxiliary_ips = _display_values(_record_value(
            machine, "auxiliary_ip", "auxiliary_ips", "aux_ips", "secondary_ips",
        ))
        owner_text = _owner_label(state["owners"], test_only=test_only, mention=not test_only)
        auth_text = "已授权" if state["authorized"] else "未授权或未确认"
        agent_text = (
            "凭据可用" if state["credentials_available"] else
            "不可用" if state["credentials_flag"] is False else "未确认"
        )
        platform_value = _record_value(
            state["diagnostic"], "platform", "platform_name", "platforms",
            default=_record_value(machine, "platform", "platform_name", "platforms"),
        )
        platform_text = "、".join(_display_values(platform_value)) or "未确认"
        compatibility = (
            "兼容" if state["component_compatible"] else
            "不兼容" if state["component_flag"] is False else
            "部分兼容" if component_ids.intersection(
                set(_record_value(state["group"], "component_ids", default=[]) or [])
                | set(_record_value(state["diagnostic"], "component_ids", default=[]) or [])
            ) else "未确认"
        )
        reasons = "；".join(state["reasons"]) or "所有审批条件已确认。"
        inventory_lines.append("| " + " | ".join(_escape(value) for value in (
            index, brand, model, alias, main_ip,
            "、".join(auxiliary_ips) if auxiliary_ips else "—",
            owner_text, auth_text, agent_text, platform_text, compatibility, reasons,
        )) + " |")
    if not inventory_rows:
        inventory_lines.extend([
            "| — | — | — | 未提供 | — | — | — | 未确认 | 未确认 | 未确认 | 未确认 | 未收到物理机器清单，不能据此推断机器不存在。 |",
        ])

    compatible_lines = [
        "### Compatible Machines · 本轮适配机器", "",
    ]
    compatible_states = [item for item in machine_states if item["safe"]]
    if display_status == "deploying":
        compatible_lines.append("当前部署正在执行，请等待结果；此阶段不重复审批。")
    elif compatible_states:
        compatible_lines.extend([
            "| 机器名称 | 主 IP | Owner | 审批命令 |",
            "|---|---|---|---|",
        ])
        for state in compatible_states:
            owner = _owner_label(state["owners"], test_only=test_only, mention=not test_only)
            command = f"/approve_deploy machine={state['command_selector']}"
            compatible_lines.append("| " + " | ".join(_escape(value) for value in (
                state["alias"], state["ip"], owner, command,
            )) + " |")
    else:
        compatible_lines.append("尚无通过仓库授权、身份、Agent Core、平台、组件及安全审批检查的机器。")

    pending_lines = [
        "### Pending / Incompatible Machines · 待确认机器", "",
    ]
    pending_states = [item for item in machine_states if not item["safe"]]
    known_no_authorization = (
        policy["machines_status"] == "main 缺少"
        or policy["authorized_count"] == 0
    )
    authorization_unknown = (
        not policy["app_authorized"] or not policy["machines_configured"]
        or policy["authorized_count"] is None
    )
    if (known_no_authorization or authorization_unknown) and inventory_rows:
        if known_no_authorization and physical_machine_count == 21:
            pending_lines.extend([
                "物理机器已登记，但当前仓库尚未授权。",
                "所有21台仍正常展示。没有授权的机器不生成可执行审批命令。",
                "",
            ])
        elif known_no_authorization:
            pending_lines.extend([
                f"物理机器已登记，但当前仓库尚未授权；清单中的 {physical_machine_count} 台设备仍正常展示。",
                "没有授权的机器不生成可执行审批命令。",
                "",
            ])
        else:
            pending_lines.extend([
                "物理机器已登记，但当前仓库授权状态未确认。",
                "所有物理机器仍正常展示；未确认授权的机器不生成可执行审批命令。",
                "",
            ])
    if pending_states:
        for state in pending_states:
            reasons = "；".join(state["reasons"]) or "兼容性或安全审批条件未确认。"
            pending_lines.extend([
                f"- 机器 `{_escape(state['alias'] or '未提供')}`",
                f"  - Owner: {_owner_label(state['owners'], test_only=test_only, mention=not test_only)}",
                f"  - 主 IP：`{_escape(state['ip'] or '未提供')}`",
                f"  - 状态：{_escape('重复 IP' if state['duplicate_ip'] else '待确认 / 不兼容')}",
                f"  - 原因：{_escape(reasons)}",
            ])
    elif inventory_rows:
        pending_lines.append("所有已登记机器均通过当前审批条件。")

    policy_lines = [
        "### Repository Authorization · 仓库授权", "",
        f"- 仓库：`{_escape(repo)}`",
        f"- GitHub App 授权：{policy['app_status']}",
        f"- machines.yaml：{policy['machines_status']}",
        f"- test_reviewers.yaml：{policy['reviewers_status']}",
        f"- 已授权机器数量：{policy['count_display']}",
        f"- {policy['reviewer_display']}",
        "- 验收人候选账号（实际授权以当前仓库 test_reviewers.yaml 为准）："
        + "、".join(_escape(name) for name in policy["candidate_reviewers"]),
    ]
    if inventory_rows and (
            policy["machines_status"] == "main 缺少"
            or policy["authorized_count"] == 0):
        policy_lines.extend([
            "",
            "物理机器已登记，但当前仓库尚未授权。所有物理机器仍展示；未授权机器不会出现可执行审批命令。",
        ])
    elif inventory_rows and (
            not policy["app_authorized"] or not policy["machines_configured"]
            or policy["authorized_count"] is None):
        policy_lines.extend([
            "",
            "当前仓库授权状态未确认。所有物理机器仍展示；未确认授权的机器不会出现可执行审批命令。",
        ])

    runtime_rows = _record_rows(runtime_checks, ("checks", "runtime_checks", "items"))
    runtime_lines = ["### Runtime Image Status · 运行镜像状态", ""]
    if runtime_rows:
        runtime_lines.extend([
            "| 机器 | 主 IP | Owner | 审批目标 digest | 当前运行 digest | 比较状态 | 最后扫描时间 | 下一步 |",
            "|---|---|---|---|---|---|---|---|",
        ])
        runtime_status_labels = {
            "match": "MATCH：运行镜像与审批目标一致",
            "mismatch": "MISMATCH：运行镜像与审批目标不一致",
            "not_running": "UNAVAILABLE：组件未运行，无法比较镜像",
            "unavailable": "UNAVAILABLE：机器连接或状态不可用",
            "identity_changed": "IDENTITY_CHANGED：机器身份绑定已变化，等待重新确认",
        }
        for check in runtime_rows:
            alias = _machine_alias(check)
            alias_key_value = alias.casefold()
            machine_state = machine_state_by_alias.get(alias_key_value, {})
            diagnostic = (
                diagnostic_by_alias.get(alias_key_value)
                or machine_state.get("diagnostic")
                or diagnostics_by_ip.get(_canonical_ip(_machine_ip(check)), {})
            )
            physical = (
                inventory_by_alias.get(alias_key_value)
                or machine_state.get("physical")
                or inventory_by_ip.get(_canonical_ip(_machine_ip(check)), {})
            )
            group = machine_state.get("group", {})
            owners = _owner_names(check, diagnostic, physical, group)
            ip_value = (
                _canonical_ip(_record_value(check, "primary_ip"))
                or _canonical_ip(_record_value(check, "main_ip"))
                or _canonical_ip(_machine_ip(physical))
                or _canonical_ip(machine_state.get("ip"))
                or _canonical_ip(_machine_ip(check))
            )
            status = str(_record_value(check, "status", default="unavailable")).casefold()
            if status not in runtime_status_labels:
                status = "unavailable"
            if display_status == "deploying":
                next_step = "正在部署，请等待结果；不要重复审批。"
            elif status == "match":
                next_step = "MATCH：无需重新部署。"
            elif status == "unavailable":
                next_step = "UNAVAILABLE：等待机器恢复连接；当前不能判断是否需要重新部署。"
            elif status == "identity_changed":
                next_step = "身份绑定变化，等待重新确认；当前不生成审批命令。"
            elif status == "not_running":
                next_step = "UNAVAILABLE：组件未运行，先确认机器状态；不生成重新部署命令。"
            else:
                safe_state = safe_machines.get(alias_key_value)
                selector = safe_state.get("command_selector", "") if safe_state else ""
                if selector:
                    next_step = (
                        f"MISMATCH：直接重新审批：`/approve_deploy machine={selector}`；"
                        "`/request_deploy` 可选。"
                    )
                else:
                    next_step = "MISMATCH：当前身份或仓库授权未确认，暂不生成审批命令。"
            target_digest = _runtime_digest(_record_value(check, "target_image", "target_digest"))
            observed_digest = _runtime_digest(_record_value(check, "observed_image", "observed_digest"))
            checked_at = (
                _record_value(check, "checked_at", "last_scanned_at", default="未扫描")
                or "未扫描"
            )
            owner = _owner_label(owners, test_only=test_only, mention=not test_only)
            status_label = runtime_status_labels[status]
            if display_status == "deploying":
                status_label = f"部署进行中；上次只读检查：{status_label}"
            runtime_lines.append("| " + " | ".join(_escape(value) for value in (
                alias or _record_value(check, "machine", default="未提供"),
                ip_value or "未提供", owner, target_digest, observed_digest,
                status_label, checked_at, next_step,
            )) + " |")
    else:
        runtime_lines.append("尚无可展示的机器镜像检查结果。")

    lines = [
        BOT_MARKER,
        lifecycle_marker(repo, pr_number),
        "### Deploy Approval — Lifecycle",
        "",
        f"**Status:** `{display_status}`",
        f"**Bound HEAD:** `{_short(head_sha)}`",
        f"**Repository:** `{_escape(repo)}`",
        "",
        *policy_lines,
        "",
        *inventory_lines,
        "",
        *compatible_lines,
        "",
        *pending_lines,
        "",
        *runtime_lines,
        "",
        "### Workflow",
        "",
        _workflow_lines(display_status),
        "",
        "### Next action",
        "",
    ]
    if display_status == "deploying":
        lines.append("正在部署，请等待结果；请勿重复审批。")
    elif compatible_states:
        lines.extend([
            "等待对应机器负责人使用 Compatible Machines 中列出的命令审批。",
            "`/request_deploy` 可选，不是直接审批的前提。",
        ])
    else:
        lines.append("暂时没有通过完整安全检查的可执行审批命令；先解决 Pending / Incompatible Machines 中列出的原因。")

    if gate_note:
        lines.extend(["", *[_safe_display_text(note) for note in gate_note
                             if isinstance(note, str)]])
    lines.extend(["", "### Current Deployment", ""])
    lines.append(_deployment_table(
        deployments, all_components or component_snapshot or components, group_rows,
    ))
    lines.extend(_attempt_lines(attempt))
    body = _lifecycle_body(lines)
    return body.replace("@", "＠") if test_only else body


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
        "Error": _safe_display_text(attempt.get("error") or "—"),
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
