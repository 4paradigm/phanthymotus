"""Dependency-free evidence page. Authentication and storage belong to the API.

Serve javascript_source() at /evidence/assets/app.js with a JavaScript MIME type.
No credentials, signed URLs or CSRF tokens are persisted by this client.
"""
from __future__ import annotations

import html

from .comments import _evidence_path


def render_upload_page(repo, pr_number, head_sha, round_id) -> str:
    api = _evidence_path(repo, pr_number)
    if not api:
        raise ValueError("invalid repository or PR")
    esc = lambda value: html.escape(str(value), quote=True)
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="referrer" content="no-referrer"><title>附件 · {esc(repo)} PR #{esc(pr_number)}</title>
<style>
*{{box-sizing:border-box}}
body{{margin:0;background:#f5f5f7;color:#1d1d1f;font:16px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif}}
main{{width:min(100% - 32px,780px);margin:48px auto 72px}}
.page-header{{display:flex;align-items:center;justify-content:space-between;gap:20px;margin:0 0 26px}}
.eyebrow{{margin:0 0 4px;color:#6e6e73;font-size:13px;font-weight:600;letter-spacing:.02em}}
h1{{margin:0;font-size:clamp(28px,5vw,36px);line-height:1.15;letter-spacing:-.035em}}
.subtitle{{margin:8px 0 0;color:#6e6e73}}
.header-actions{{display:flex;align-items:center;gap:12px;flex-shrink:0}}
.status-pill{{display:inline-flex;align-items:center;min-height:30px;padding:4px 11px;border-radius:999px;background:#e8e8ed;color:#424245;font-size:14px;font-weight:600;white-space:nowrap}}
.status-pill[data-state="testing"]{{background:#e8f2ff;color:#0758b7}}
.status-pill[data-state="succeeded"]{{background:#e7f6ec;color:#147333}}
.status-pill[data-state="failed"]{{background:#fff0ef;color:#b42318}}
section.panel{{margin:16px 0;padding:24px;background:#fff;border:1px solid #e5e5ea;border-radius:16px}}
.section-heading{{margin-bottom:18px}}
h2{{margin:0;font-size:20px;letter-spacing:-.02em}}
.muted{{margin:5px 0 0;color:#6e6e73;font-size:14px}}
button,input,textarea,select{{font:inherit}}
button{{min-height:44px;padding:9px 16px;border:1px solid #d2d2d7;border-radius:10px;background:#fff;color:#1d1d1f;cursor:pointer}}
button:hover{{background:#f5f5f7}}
button:disabled{{cursor:not-allowed;opacity:.55}}
button:focus-visible,a:focus-visible,input:focus-visible,textarea:focus-visible,select:focus-visible,summary:focus-visible{{outline:3px solid #77aaff;outline-offset:2px}}
.primary{{border-color:#0071e3;background:#0071e3;color:#fff;font-weight:600}}
.primary:hover{{border-color:#0077ed;background:#0077ed}}
.header-actions button{{min-width:76px}}
#message{{margin:0 0 16px;color:#b42318;white-space:pre-wrap}}
#message:empty{{display:none}}
fieldset{{min-width:0;margin:0;padding:0;border:0}}
#dropzone{{display:flex;min-height:132px;align-items:center;justify-content:center;flex-direction:column;gap:8px;padding:20px;border:1px dashed #a1a1a6;border-radius:13px;color:#1d1d1f;text-align:center;cursor:pointer}}
#dropzone.dragging{{border-color:#0071e3;background:#f0f7ff}}
#dropzone input{{max-width:100%;font-size:14px}}
.file-hint{{color:#6e6e73;font-size:13px}}
.secondary{{margin-top:14px;border-top:1px solid #e5e5ea;padding-top:12px}}
.secondary summary,.history-record summary,.attachment-details summary{{cursor:pointer;color:#424245;font-weight:600}}
.secondary-fields{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}}
label.control{{display:block;color:#424245;font-size:14px;font-weight:600}}
textarea,select{{display:block;width:100%;margin-top:6px;padding:10px 12px;border:1px solid #d2d2d7;border-radius:9px;background:#fff;color:#1d1d1f}}
textarea{{min-height:88px;resize:vertical}}
.secondary-actions{{display:flex;justify-content:flex-end;margin-top:12px}}
.primary-row{{display:flex;justify-content:flex-end;gap:10px;margin-top:14px}}
.list-heading{{margin:24px 0 8px;font-size:16px}}
.queue,.attachments{{list-style:none;margin:0;padding:0}}
.queue:empty,.attachments:empty{{display:none}}
.job,.attachment{{margin:10px 0;padding:15px;border:1px solid #e5e5ea;border-radius:12px;background:#fff}}
.job-heading,.attachment-heading{{display:flex;align-items:flex-start;justify-content:space-between;gap:12px}}
.job-title,.attachment-title{{min-width:0;overflow-wrap:anywhere;font-weight:600}}
.file-size{{flex-shrink:0;color:#6e6e73;font-size:14px;white-space:nowrap}}
progress{{display:block;width:100%;height:8px;margin:12px 0 7px;accent-color:#0071e3}}
.job-footer{{display:flex;align-items:center;justify-content:space-between;gap:12px}}
.job-state{{color:#424245;font-size:14px;font-weight:600}}
.job-detail{{color:#6e6e73;font-size:13px;overflow-wrap:anywhere}}
.job-footer button,.attachment-actions button,.attachment-actions a{{min-height:40px}}
.attachment-actions{{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}}
.attachment-actions a{{display:inline-flex;align-items:center;padding:7px 12px;border:1px solid #d2d2d7;border-radius:9px;color:#06c;text-decoration:none}}
.attachment-actions a:hover{{background:#f5f5f7}}
.attachment-details{{margin-top:9px;font-size:14px}}
.attachment-meta{{display:grid;grid-template-columns:auto 1fr;gap:4px 12px;margin:10px 0 0;color:#6e6e73}}
.attachment-meta dt{{color:#86868b}}
.attachment-meta dd{{margin:0;color:#424245;overflow-wrap:anywhere}}
.media-preview{{margin-top:12px}}
.media-preview video,.media-preview img{{display:block;width:100%;max-height:520px;border-radius:10px;background:#111}}
.attachment-meta-inline{{margin:5px 0 0;color:#6e6e73;font-size:13px}}
.attachment-note{{margin-top:10px;padding:10px 12px;border-radius:10px;background:#f5f5f7;overflow-wrap:anywhere}}
.attachment-note p{{margin:0 0 8px}}.attachment-note p:last-child{{margin-bottom:0}}
.attachment-note pre,.attachment-note code{{white-space:pre-wrap;overflow-wrap:anywhere}}
.media-hint,.empty-state{{color:#6e6e73;font-size:14px}}
.media-hint{{margin:7px 0 0}}
.media-hint:empty{{display:none}}
.history-record{{margin:10px 0;padding:14px;border:1px solid #e5e5ea;border-radius:12px}}
.history-record summary{{overflow-wrap:anywhere}}
.history-record ul{{list-style:none;margin:12px 0 0;padding:0}}
.history-links{{margin-top:12px}}
.history-links a{{color:#06c}}
[hidden]{{display:none!important}}
@media (max-width:560px){{
  main{{width:min(100% - 24px,780px);margin:28px auto 48px}}
  .page-header{{align-items:flex-start;flex-direction:column;gap:12px}}
  .header-actions{{width:100%;justify-content:space-between}}
  section.panel{{padding:18px;border-radius:14px}}
  .secondary-fields{{grid-template-columns:1fr}}
  .primary-row{{flex-direction:column}}
  .primary-row button{{width:100%}}
  .job-heading,.attachment-heading{{gap:6px;flex-direction:column}}
  .file-size{{white-space:normal}}
  .job-footer{{align-items:flex-start;flex-direction:column}}
  .job-footer button{{width:100%}}
}}
</style><script src="/evidence/assets/app.js" defer></script></head>
<body>
<main id="app" data-api="{esc(api)}" data-repo="{esc(repo)}" data-pr="{esc(pr_number)}" data-head="{esc(head_sha)}" data-round="{esc(round_id)}">
<header class="page-header"><div><p class="eyebrow">部署验证</p><h1>附件</h1>
<p class="subtitle">{esc(repo)} · PR #{esc(pr_number)}</p></div></header>
<p id="message" role="status" aria-live="polite"></p>
<section id="current-section" class="panel" hidden><div class="section-heading"><h2 id="round-title">附件</h2><p id="summary" class="muted">当前轮附件与说明。</p></div>
<fieldset id="writes" disabled hidden><legend>上传附件</legend>
<label id="dropzone" for="file"><strong>选择文件，或拖放到这里</strong><span class="file-hint">最多 20 个文件 · 单个文件最大 512 MiB · 总计最大 10 GiB</span><input id="file" type="file" multiple></label>
<div class="secondary"><div class="secondary-fields">
<label class="control" for="note">文件说明或独立备注<textarea id="note" aria-describedby="note-help"></textarea><span id="note-help" class="file-hint">最多 5000 字，支持 Markdown；短备注可随文件保存或单独保存。超出后，点击“单独保存备注”或“上传文件”都会将原文完整转为 UTF-8 TXT 附件。图片可作为独立附件上传。配额按当前可见附件及本浏览器未完成任务估算；其他并发上传以服务端最终校验为准。</span></label>
<label class="control" for="replace">附件操作<select id="replace"><option value="">新增附件（保留已有附件）</option><optgroup id="replace-targets" label="替换指定附件（旧记录保留历史）"></optgroup></select></label>
</div></div>
<div class="primary-row"><button id="save-note" type="button">单独保存备注</button><button id="upload" class="primary" type="button">上传所选文件</button></div>
</fieldset>
<h3 class="list-heading" id="queue-heading" hidden>上传进度</h3><ul id="queue" class="queue" aria-labelledby="queue-heading" aria-live="polite"></ul>
<h3 class="list-heading" id="attachments-heading">附件</h3><ul id="attachments" class="attachments" aria-labelledby="attachments-heading"></ul>
</section>
<section id="history-section" class="panel" hidden><h2>历史轮次</h2><div id="history"></div></section>
</main></body></html>'''


def javascript_source() -> str:
    return _JAVASCRIPT


_JAVASCRIPT = r'''"use strict";
(() => {
  const app = document.getElementById("app");
  if (!app) return;
  const $ = id => document.getElementById(id);
  const api = app.dataset.api;
  const bound = {head_sha: app.dataset.head, round_id: app.dataset.round};
  const storageKey = "evidence:" + api + ":" + bound.head_sha + ":" + bound.round_id;
  const proofKey = "evidence-proof:" + api + ":" + bound.head_sha + ":" + bound.round_id;
  const clientKeyStorage = "evidence-client-key:" + api + ":" + bound.head_sha + ":" + bound.round_id;
  const PART_BYTES = 8 * 1024 * 1024, MAX_BYTES = 512 * 1024 * 1024, MAX_NOTE_CHARS = 5000;
  const MAX_FILES = 20, MAX_TOTAL_BYTES = 10 * 1024 * 1024 * 1024;
  const active = new Set(), jobs = [];
  let context = null, stopped = false, busy = false, proofSecret = "", clientKey = "", savedUploads = {};
  let noteOperation = null;
  const fragment = new URLSearchParams(window.location.hash.slice(1));
  const incomingProof = fragment.get("proof");
  if (window.location.hash) history.replaceState(history.state, "", window.location.pathname + window.location.search);
  try {
    if (incomingProof !== null) {
      if (/^[0-9a-f]{64}$/.test(incomingProof)) {
        sessionStorage.setItem(proofKey, incomingProof);
        proofSecret = incomingProof;
      } else sessionStorage.removeItem(proofKey);
    } else {
      const stored = sessionStorage.getItem(proofKey) || "";
      if (/^[0-9a-f]{64}$/.test(stored)) proofSecret = stored;
      else sessionStorage.removeItem(proofKey);
    }
  } catch (_) { proofSecret = ""; }
  const message = text => { $("message").textContent = String(text); };
  const id = value => {
    if (typeof value !== "string" || !/^[A-Za-z0-9_-]+$/.test(value)) throw new Error("服务端返回了无效标识符");
    return encodeURIComponent(value);
  };
  const uid = () => crypto.randomUUID();
  const randomHex = (size = 32) => [...crypto.getRandomValues(new Uint8Array(size))]
    .map(value => value.toString(16).padStart(2, "0")).join("");
  const validClientKey = value => typeof value === "string" && /^[0-9a-f]{64}$/.test(value);
  let savedClientKey = "";
  try {
    const saved = localStorage.getItem(clientKeyStorage) || "";
    if (validClientKey(saved)) savedClientKey = saved;
  } catch (_) {}
  if (!savedClientKey) {
    try {
      const legacy = sessionStorage.getItem(clientKeyStorage) || "";
      if (validClientKey(legacy)) savedClientKey = legacy;
    } catch (_) {}
  }
  clientKey = savedClientKey || randomHex();
  let clientKeyStored = false;
  try { localStorage.setItem(clientKeyStorage, clientKey); clientKeyStored = true; } catch (_) {}
  if (!clientKeyStored) {
    try { sessionStorage.setItem(clientKeyStorage, clientKey); } catch (_) {}
  }
  const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
  const roundMatches = value => value && value.head_sha === bound.head_sha && value.round_id === bound.round_id;
  const lifecycleLabels = {
    testing: "测试中", succeeded: "已通过", failed: "未通过", "deploy-ready": "待部署",
    "deploy-requested": "等待部署审批", deploying: "正在部署", deployed: "已部署",
    reviewing: "代码审查中", pending: "等待更新", text: "备注已保存"
  };
  const lifecycleLabel = value => Object.prototype.hasOwnProperty.call(lifecycleLabels, value)
    ? lifecycleLabels[value] : "状态暂不可用";
  function formatBytes(value) {
    if (value === null || value === undefined || value === "") return "";
    const bytes = Number(value);
    if (!Number.isFinite(bytes) || bytes < 0) return "";
    if (bytes < 1024) return Math.round(bytes) + " B";
    const units = ["KB", "MB", "GB", "TB"];
    let size = bytes, unit = "B";
    for (const next of units) { size /= 1024; unit = next; if (size < 1024 || next === "TB") break; }
    return size.toFixed(1).replace(/\.0$/, "") + " " + unit;
  }
  function formatDate(value, includeTime = false) {
    if (typeof value !== "string" || !value) return "时间未知";
    const date = new Date(value);
    if (!Number.isFinite(date.getTime())) return "时间未知";
    const options = includeTime
      ? {timeZone: "Asia/Shanghai", year: "numeric", month: "long", day: "numeric", hour: "2-digit", minute: "2-digit", hourCycle: "h23"}
      : {timeZone: "Asia/Shanghai", year: "numeric", month: "long", day: "numeric"};
    const formatted = new Intl.DateTimeFormat("zh-CN", options).format(date);
    return includeTime ? formatted + "（北京时间）" : formatted;
  }
  const jobStatusLabel = status => ({
    排队中: "等待上传", 核对中: "校验中", running: "上传中", verifying: "校验中",
    waiting: "校验中", succeeded: "已完成", failed: "失败", paused: "已暂停"
  }[status] || "处理中");
  function proofTransportAllowed(target) {
    if (target.origin !== window.location.origin) return false;
    if (window.location.protocol === "https:") return target.protocol === "https:";
    return window.location.origin === "http://127.0.0.1:25001" && target.origin === window.location.origin;
  }
  function inviteTransportAllowed(target) {
    const invite = context && (context.auth_method === "signed_invite" || context.upload_invite === true);
    return !!invite && target.origin === window.location.origin &&
      window.location.protocol === "https:" && target.protocol === "https:";
  }
  function writable() {
    const secureInvite = context && context.auth_method === "signed_invite" && context.can_write === true &&
      context.actor_verified === false && context.github_user_authenticated === false &&
      window.location.protocol === "https:";
    const invite = context && context.upload_invite === true &&
      context.actor_verified === false && context.github_user_authenticated === false;
    const proofUser = context && context.actor_verified === true &&
      context.github_user_authenticated === true && !!proofSecret;
    return context && roundMatches(context) && context.enabled !== false && context.evidence_enabled !== false &&
      context.can_write === true && context.status === "testing" &&
      (secureInvite || (invite && proofTransportAllowed(new URL(api, window.location.href))) || proofUser) &&
      context.readiness && context.readiness.COS_READY === true &&
      context.readiness.EVIDENCE_BACKEND_READY === true &&
      context.readiness.BROWSER_UPLOAD_READY === true &&
      context.pr_state === "open" && !context.merged && !stopped;
  }
  function controls() {
    const canWrite = writable();
    $("writes").hidden = !canWrite; $("writes").disabled = !canWrite;
    $("upload").disabled = busy || !canWrite; $("file").disabled = busy || !canWrite;
    $("note").disabled = busy || !canWrite; $("save-note").disabled = busy || !canWrite;
    $("replace").disabled = busy || !canWrite;
    for (const button of document.querySelectorAll("button[data-job-action]"))
      button.disabled = !canWrite || (busy && button.dataset.jobAction !== "pause");
    for (const button of document.querySelectorAll("button[data-delete]")) button.disabled = busy || !canWrite;
  }
  function abortRequest(request) { try { request.abort(); } catch (_) {} }
  function halt(text) {
    stopped = true;
    for (const request of active) abortRequest(request);
    controls(); message(text);
  }
  class ApiError extends Error {
    constructor(status, text, detail = "") { super(text); this.status = status; this.detail = detail; }
  }
  function clearProof() {
    proofSecret = "";
    try { sessionStorage.removeItem(proofKey); } catch (_) {}
  }
  async function request(path, {method = "GET", body, key, job, timeoutMs = 30000, authMode = "auto"} = {}) {
    const write = method !== "GET";
    if (write && !writable()) throw new Error("当前轮不可写，请刷新并核对 HEAD/round。");
    const target = new URL(api + path, window.location.href);
    if (target.origin !== window.location.origin) throw new Error("拒绝跨站 Evidence 请求。");
    const headers = {Accept: "application/json"};
    headers["X-Evidence-Client-Key"] = clientKey;
    const sendProof = proofSecret && (authMode === "proof" ||
      authMode === "auto" && !(context && (context.auth_method === "signed_invite" || context.upload_invite === true))) &&
      proofTransportAllowed(target);
    const transportAllowed = sendProof ? proofTransportAllowed(target) : inviteTransportAllowed(target);
    if (sendProof) {
      headers["X-Evidence-Secret"] = proofSecret;
      headers["X-Evidence-Nonce"] = randomHex();
    } else if (write && !transportAllowed) {
      throw new Error("当前链接没有有效上传授权，或请求不是安全的同源 HTTPS 请求。");
    }
    if (write) headers["Content-Type"] = "application/json";
    const controller = new AbortController(); active.add(controller); if (job) job.requests.add(controller);
    const timeout = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(target.href, {
        method, headers, credentials: "same-origin", redirect: "error", cache: "no-store", signal: controller.signal,
        ...(write ? {body: JSON.stringify({...body, ...bound, idempotency_key: key || uid()})} : {})
      });
      let value; try { value = await response.json(); } catch (_) { value = {}; }
      if (!response.ok) {
        const detail = typeof value.detail === "string" ? value.detail : typeof value.error === "string" ? value.error : "";
        if (/invalid_evidence_capability|invalid_evidence|evidence_capability/i.test(detail))
          throw new ApiError(response.status, "上传链接无效或已失效。请从当前 PR Summary 重新打开附件页面。", detail);
        if ([401, 403].includes(response.status) && proofSecret && authMode === "proof") clearProof();
        else if (response.status === 401 && proofSecret && authMode !== "invite") clearProof();
        if (response.status === 401) {
          const expired = "上传链接可能已过期或失效。请刷新 PR Summary 并打开最新上传链接。";
          if (!(path === "/context" && authMode === "invite")) halt(expired);
          throw new ApiError(401, expired, detail);
        }
        if (response.status === 403) throw new ApiError(403, "权限不足：当前上传者无权执行此操作。", detail);
        if (response.status === 409 && ["round_changed", "context_not_writable", "pr_closed", "pr_closed_or_mismatched"].includes(detail))
          halt("HEAD/round 或 PR 状态已变化，已停止所有写入。请刷新 PR Summary 并打开最新上传链接。");
        if (response.status === 413) throw new ApiError(413, "超过服务端大小限制。 " + detail, detail);
        throw new ApiError(response.status, response.status + ": " + (detail || response.statusText), detail);
      }
      return value;
    } finally { clearTimeout(timeout); active.delete(controller); if (job) job.requests.delete(controller); }
  }
  function requestRaw(path, blob, job, onProgress) {
    return new Promise((resolve, reject) => {
      if (!writable()) return reject(new Error("当前轮不可写，请刷新并核对 HEAD/round。"));
      const target = new URL(api + path, window.location.href);
      const useProof = proofSecret && !(context && (context.auth_method === "signed_invite" || context.upload_invite === true));
      const transportAllowed = useProof ? proofTransportAllowed(target) : inviteTransportAllowed(target);
      if (target.origin !== window.location.origin || !transportAllowed)
        return reject(new Error("当前链接没有有效上传授权，或请求不是安全的同源 HTTPS 请求。"));
      const xhr = new XMLHttpRequest(); active.add(xhr); job.requests.add(xhr);
      let finished = false;
      const cleanup = () => { if (finished) return; finished = true; active.delete(xhr); job.requests.delete(xhr); };
      xhr.open("PUT", target.href); xhr.withCredentials = true; xhr.timeout = 120000;
      xhr.setRequestHeader("Accept", "application/json");
      xhr.setRequestHeader("Content-Type", "application/octet-stream");
      xhr.setRequestHeader("X-Evidence-Client-Key", clientKey);
      if (useProof) {
        xhr.setRequestHeader("X-Evidence-Secret", proofSecret);
        xhr.setRequestHeader("X-Evidence-Nonce", randomHex());
      }
      xhr.upload.onprogress = event => { if (event.lengthComputable) onProgress(event.loaded); };
      xhr.onload = () => {
        cleanup();
        let value = {}; try { value = JSON.parse(xhr.responseText || "{}"); } catch (_) {}
        let responseOrigin = ""; try { responseOrigin = new URL(xhr.responseURL).origin; } catch (_) {}
        if (xhr.status >= 200 && xhr.status < 300 && responseOrigin === target.origin) return resolve(value);
        const detail = typeof value.detail === "string" ? value.detail : "";
        if (/invalid_evidence_capability|invalid_evidence|evidence_capability/i.test(detail))
          return reject(new ApiError(xhr.status, "上传链接无效或已失效。请从当前 PR Summary 重新打开附件页面。", detail));
        if (xhr.status === 401) clearProof();
        if (xhr.status === 401) halt("上传链接可能已过期或失效。请刷新 PR Summary 并打开最新上传链接。");
        if (xhr.status === 403) return reject(new ApiError(403, "权限不足：当前上传者无权执行此操作。", detail));
        if (xhr.status === 409 && ["round_changed", "context_not_writable", "pr_closed", "pr_closed_or_mismatched"].includes(detail))
          halt("HEAD/round 或 PR 状态已变化，已停止所有写入。请刷新 PR Summary 并打开最新上传链接。");
        reject(new ApiError(xhr.status, xhr.status + ": " + (detail || "分块上传失败"), detail));
      };
      xhr.onerror = () => { cleanup(); reject(new Error("网络中断；将先查询 COS 分块状态再决定是否重试。")); };
      xhr.ontimeout = () => { cleanup(); reject(new Error("分块上传超时；将先查询 COS 分块状态再决定是否重试。")); };
      xhr.onabort = () => { cleanup(); reject(new Error("上传已暂停。")); };
      xhr.send(blob);
    });
  }
  function anchor(label, path) {
    const target = new URL(api + path, window.location.href);
    if (target.origin !== window.location.origin) throw new Error("拒绝跨站下载地址。");
    const a = document.createElement("a"); a.textContent = label; a.href = target.href;
    a.setAttribute("download", ""); a.rel = "noopener"; return a;
  }
  function attachmentDownloadUrl(item) {
    if (!item || item.status !== "ready" || item.deleted_at) return "";
    const attachmentId = item.attachment_id || item.id;
    if (!attachmentId || typeof item.download_capability !== "string" || !item.download_capability) return "";
    try {
      const path = "/attachments/" + id(attachmentId) + "/download?cap=" + encodeURIComponent(item.download_capability);
      const target = new URL(api + path, window.location.href);
      return target.origin === window.location.origin ? target.href : "";
    } catch (_) { return ""; }
  }
  function attachmentPreviewUrl(item) {
    if (!item || item.status !== "ready" || item.deleted_at) return "";
    const attachmentId = item.attachment_id || item.id;
    if (!attachmentId || typeof item.download_capability !== "string" || !item.download_capability) return "";
    try {
      const path = "/attachments/" + id(attachmentId) + "/preview?cap=" + encodeURIComponent(item.download_capability);
      const target = new URL(api + path, window.location.href);
      return target.origin === window.location.origin ? target.href : "";
    } catch (_) { return ""; }
  }
  function mediaType(item) {
    const type = String(item.media_type || item.content_type || "").toLowerCase().split(";")[0];
    const name = String(item.filename || "").toLowerCase();
    if (/^image\/(png|jpeg|gif|webp)$/.test(type)) return "image";
    if (/^video\/(mp4|webm)$/.test(type)) return "video";
    if (type && type !== "application/octet-stream") return "";
    if (/\.(png|jpe?g|gif|webp)$/.test(name)) return "image";
    if (/\.(mp4|webm)$/.test(name)) return "video";
    return "";
  }
  function appendMeta(list, label, value) {
    if (value === undefined || value === null || value === "") return;
    const term = document.createElement("dt"), description = document.createElement("dd");
    term.textContent = label; description.textContent = String(value);
    list.append(term, description);
  }
  function itemTimestamp(item) {
    const value = Date.parse(item && (item.created_at || item.uploaded_at || item.updated_at) || "");
    return Number.isFinite(value) ? value : 0;
  }
  function isDeletedAttachment(item) {
    return !!(item && (item.status === "deleted" || item.deleted_at));
  }
  function isTextRecord(item) {
    return !!(item && (item.status === "text" || item.evidence_kind === "note" || item.kind === "note"));
  }
  function displayNote(item) {
    if (!item || item.evidence_kind !== "note" || typeof item.display_note !== "string") return "";
    return item.display_note.trim();
  }
  function displayFilename(item) {
    const value = String(item && item.filename || "").trim();
    if (!value || /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(value))
      return item && item.status === "text" ? "附件说明" : "附件文件";
    return value;
  }
  function appendInlineMarkdown(parent, text) {
    const tick = String.fromCharCode(96);
    const tokenPattern = new RegExp("(\\*\\*[^*]+\\*\\*|\\*[^*]+\\*|" + tick + "[^" + tick + "]+" + tick + "|\\[[^\\]]+\\]\\([^)]+\\))", "g");
    let offset = 0, match;
    while ((match = tokenPattern.exec(text))) {
      if (match.index > offset) parent.append(document.createTextNode(text.slice(offset, match.index)));
      const token = match[0];
      let node = null;
      if (token.startsWith("**") && token.endsWith("**")) {
        node = document.createElement("strong"); node.textContent = token.slice(2, -2);
      } else if (token.startsWith("*") && token.endsWith("*")) {
        node = document.createElement("em"); node.textContent = token.slice(1, -1);
      } else if (token.startsWith(tick) && token.endsWith(tick)) {
        node = document.createElement("code"); node.textContent = token.slice(1, -1);
      } else {
        const linkMatch = token.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
        if (linkMatch) {
          try {
            const target = new URL(linkMatch[2], window.location.href);
            if (target.protocol === "https:" && !target.username && !target.password) {
              node = document.createElement("a"); node.textContent = linkMatch[1];
              node.href = target.href; node.rel = "noopener noreferrer";
            }
          } catch (_) {}
        }
      }
      parent.append(node || document.createTextNode(token));
      offset = tokenPattern.lastIndex;
    }
    if (offset < text.length) parent.append(document.createTextNode(text.slice(offset)));
  }
  function renderMarkdown(container, source) {
    const lines = String(source || "").split(/\r?\n/);
    let list = null;
    for (const line of lines) {
      if (!line.trim()) { list = null; continue; }
      const heading = line.match(/^(#{1,3})\s+(.+)$/);
      if (heading) {
        list = null;
        const node = document.createElement(heading[1].length === 1 ? "h3" : "h4");
        appendInlineMarkdown(node, heading[2]); container.append(node); continue;
      }
      const bullet = line.match(/^\s*[-*]\s+(.+)$/);
      if (bullet) {
        if (!list) { list = document.createElement("ul"); container.append(list); }
        const item = document.createElement("li"); appendInlineMarkdown(item, bullet[1]); list.append(item); continue;
      }
      list = null;
      const paragraph = document.createElement("p");
      appendInlineMarkdown(paragraph, line); container.append(paragraph);
    }
  }
  function syntheticBotNote(item, note) {
    if (!item || item.actor_verified !== true) return false;
    const actor = String(item.actor_login || item.actor || "");
    if (actor.toLowerCase() !== "4pd-pr-assistant[bot]") return false;
    const normalized = String(note || "").trim().replace(/\s+/g, " ").toLowerCase();
    return [
      "server-side cos multipart acceptance",
      "server-side 20mb h.264 cos transport acceptance"
    ].some(template => normalized === template ||
      normalized.startsWith(template) && /^[ :;,.!?…-]/.test(normalized.slice(template.length)));
  }
  function attachmentList(rows, parent, current) {
    const ordered = [...(rows || [])].filter(item => item && typeof item === "object")
      .sort((a, b) => itemTimestamp(b) - itemTimestamp(a));
    const latestVideo = current ? ordered.find(item => !item.deleted_at && item.status === "ready" &&
      mediaType(item) === "video" && attachmentPreviewUrl(item)) : null;
    if (latestVideo) ordered.unshift(ordered.splice(ordered.indexOf(latestVideo), 1)[0]);
    for (const item of ordered) {
      const deleted = isDeletedAttachment(item);
      const li = document.createElement("li");
      li.className = "attachment";
      const heading = document.createElement("div"), title = document.createElement("strong"), size = document.createElement("span");
      heading.className = "attachment-heading"; title.className = "attachment-title"; size.className = "file-size";
      const attachmentId = item.attachment_id || item.id;
      const independentNote = isTextRecord(item);
      title.textContent = independentNote ? "独立备注" : displayFilename(item);
      size.textContent = formatBytes(item.size);
      heading.append(title); if (size.textContent) heading.append(size);
      li.append(heading);
      const actor = item.actor_verified === true ? (item.actor_login || item.actor || "已验证用户") :
        item.upload_method === "signed_invite" ? "分享链接（身份未验证）" : "身份未验证";
      const meta = document.createElement("p");
      meta.className = "attachment-meta-inline";
      meta.textContent = actor + " · " + formatDate(item.created_at || item.uploaded_at, true);
      li.append(meta);
      if (deleted) {
        const badge = document.createElement("p"); badge.className = "muted";
        badge.textContent = "已删除，历史记录保留";
        li.append(badge);
      }
      const previewUrl = deleted ? "" : attachmentPreviewUrl(item);
      const kind = previewUrl ? mediaType(item) : "";
      if (kind) {
        const preview = document.createElement("div");
        const media = document.createElement(kind === "video" ? "video" : "img");
        preview.className = "media-preview";
        if (kind === "video") {
          media.controls = true; media.playsInline = true;
          media.preload = item === latestVideo ? "auto" : "metadata";
          if (item === latestVideo) { media.muted = true; media.autoplay = true; }
        } else {
          media.alt = displayFilename(item); media.loading = "lazy";
        }
        const hint = document.createElement("p"); hint.className = "media-hint";
        media.addEventListener("error", () => { hint.textContent = "暂时无法预览；请重新打开附件页面。"; });
        preview.append(media, hint); media.src = previewUrl; li.append(preview);
      }
      const note = independentNote ? displayNote(item) : String(item.note || "").trim();
      if (note && !syntheticBotNote(item, note)) {
        const noteBlock = document.createElement("div");
        noteBlock.className = "attachment-note";
        renderMarkdown(noteBlock, note);
        li.append(noteBlock);
      }
      const actions = document.createElement("div");
      actions.className = "attachment-actions";
      const url = deleted ? "" : attachmentDownloadUrl(item);
      if (url && kind !== "video") {
        actions.append(anchor("下载", "/attachments/" + id(item.attachment_id || item.id) +
          "/download?cap=" + encodeURIComponent(item.download_capability)));
      }
      const same = current && (!item.head_sha || item.head_sha === bound.head_sha) &&
        (!item.round_id || item.round_id === bound.round_id);
      if (!deleted && same && writable() && item.can_delete === true && attachmentId) {
        const remove = document.createElement("button"); remove.type = "button"; remove.textContent = "删除";
        remove.dataset.delete = attachmentId;
        remove.addEventListener("click", () => deleteAttachment(item)); actions.append(remove);
        if (item.status === "ready" && item.can_delete === true && !independentNote) {
          const option = document.createElement("option"); option.value = attachmentId;
          option.textContent = displayFilename(item); $("replace-targets").append(option);
        }
      }
      if (actions.childNodes.length) li.append(actions);
      parent.append(li);
    }
  }
  function render() {
    const state = typeof context.status === "string" ? context.status : "";
    const stageAvailable = ["testing", "succeeded", "failed"].includes(state);
    $("current-section").hidden = !stageAvailable;
    $("history-section").hidden = true;
    $("attachments").replaceChildren(); $("history").replaceChildren();
    if (!stageAvailable) {
      $("writes").hidden = true; $("writes").disabled = true;
      $("message").textContent = "";
      controls();
      return;
    }
    const oldRounds = (context.rounds || context.history || []).filter(round => !roundMatches(round));
    oldRounds.sort((a, b) => {
      const getTime = round => {
        const value = Date.parse(round && (round.archived_at || round.ended_at || round.round_started_at || round.started_at || round.created_at) || "");
        return Number.isFinite(value) ? value : 0;
      };
      return getTime(b) - getTime(a);
    });
    const roundNumber = Number.isSafeInteger(context.round_number) && context.round_number > 0
      ? context.round_number : null;
    $("round-title").textContent = (roundNumber ? "第 " + roundNumber + " 轮 · " : "当前轮 · ") + "附件";
    $("summary").textContent = state === "testing" ?
      "当前轮附件与说明；视频默认显示播放器。" : "本轮附件只读查看。";
    const replacement = $("replace").value;
    const replacementGroup = document.createElement("optgroup");
    replacementGroup.id = "replace-targets";
    replacementGroup.label = "替换指定附件（旧记录保留历史）";
    $("replace").replaceChildren(new Option("新增附件（保留已有附件）", ""), replacementGroup);
    attachmentList(context.attachments || [], $("attachments"), true);
    if (!$("attachments").childNodes.length) {
      const empty = document.createElement("li"); empty.className = "empty-state";
      empty.textContent = "本轮暂无附件；上传完成后会显示在这里。"; $("attachments").append(empty);
    }
    for (let index = 0; index < oldRounds.length; index++) {
      const round = oldRounds[index];
      const details = document.createElement("details"), summary = document.createElement("summary"), content = document.createElement("div");
      const rawRoundDate = round.created_at || round.started_at || round.round_started_at || round.archived_at || round.ended_at;
      const roundDate = rawRoundDate && Number.isFinite(Date.parse(rawRoundDate)) ? formatDate(rawRoundDate) : "";
      const hasAttachmentDetails = Array.isArray(round.attachments);
      const count = hasAttachmentDetails ? round.attachments.filter(item => item && !isDeletedAttachment(item)).length : null;
      const number = Number.isSafeInteger(round.round_number) && round.round_number > 0
        ? round.round_number : null;
      details.className = "history-record";
      const summaryParts = [number ? "第 " + number + " 轮" : "历史轮次"];
      if (roundDate) summaryParts.push(roundDate);
      const historyAttachmentLabel = !hasAttachmentDetails ? "附件明细请查看历史归档" :
        round.attachments.length === 0 ? "该轮未保存附件" :
        count ? count + " 个附件" : round.attachments.length + " 条历史附件记录（已删除）";
      summaryParts.push(historyAttachmentLabel);
      summary.textContent = summaryParts.join(" · ");
      details.append(summary);
      content.className = "history-content";
      const status = document.createElement("p"); status.className = "muted";
      status.textContent = "阶段：" + lifecycleLabel(round.status); content.append(status);
      if (hasAttachmentDetails) {
        const ul = document.createElement("ul"); attachmentList(round.attachments, ul, false); content.append(ul);
      } else {
        const missing = document.createElement("p"); missing.className = "muted";
        missing.textContent = "附件明细请查看历史归档。"; content.append(missing);
      }
      details.append(content);
      if (round.round_id && (round.cos_object_key || round.cos && round.cos.object_key || round.log_available || round.log_ready) &&
          context.enabled !== false && context.evidence_enabled !== false)
        details.append(anchor("下载本轮日志", "/rounds/" + id(round.round_id) + "/log/download?cap=" + encodeURIComponent(round.capability || "")));
      $("history").append(details);
    }
    let archiveCount = 0;
    for (const archive of context.archive_links || []) {
      try {
        const url = new URL(archive.url), prefix = "/" + app.dataset.repo;
        if (url.origin !== "https://github.com" || url.username || url.password || url.search ||
            ![prefix + "/issues/" + app.dataset.pr, prefix + "/pull/" + app.dataset.pr].includes(url.pathname) ||
            !/^#issuecomment-[0-9]+$/.test(url.hash)) continue;
        const a = document.createElement("a"), p = document.createElement("p");
        a.textContent = "查看完整历史记录"; a.href = url.href; a.rel = "noopener";
        p.append(a); p.className = "history-links"; $("history").append(p); archiveCount++;
      } catch (_) {}
    }
    $("history-section").hidden = oldRounds.length === 0 && archiveCount === 0;
    if ([...$("replace").options].some(o => o.value === replacement)) $("replace").value = replacement;
    controls();
  }
  async function refresh(job) {
    let next, useProofFallback = false;
    try {
      // Let the browser's HttpOnly upload cookie authenticate first. An old
      // proof is a one-time compatibility fallback only after invite auth fails.
      next = await request("/context", {job, authMode: "invite"});
    } catch (inviteError) {
      if (!proofSecret || ![401, 403].includes(inviteError.status)) throw inviteError;
      useProofFallback = true;
    }
    if (!useProofFallback && (!next || next.pr_state !== "open" || next.merged)) {
      context = next || null;
      halt("无法确认 PR 仍处于 Open 状态，已停止写入。");
      if (next) render();
      return next;
    }
    if (!useProofFallback && proofSecret && next.auth_method !== "signed_invite" &&
        roundMatches(next) && next.pr_state === "open" && !next.merged) {
      // A valid read-only page cookie is not an upload invitation. Preserve
      // the legacy proof path when it is available, but never replace stale
      // HEAD/round or closed-PR responses with a second auth result.
      useProofFallback = true;
    }
    if (useProofFallback) next = await request("/context", {job, authMode: "proof"});
    context = next;
    if (!roundMatches(next)) halt("当前 HEAD/round 已变化，已停止写入。请刷新 PR Summary 并打开最新上传链接。");
    else if (next.pr_state !== "open" || next.merged)
      halt("无法确认 PR 仍处于 Open 状态，或 PR 已关闭/合并；已停止写入。请从当前 PR Summary 重新打开。");
    else if (next.auth_method === "signed_invite") clearProof();
    render(); return next;
  }
  function persist(job) {
    if (!job.fingerprint) return;
    const key = job.fingerprint + ":" + (job.replacesId || "new") + ":" + job.noteDigest;
    savedUploads[key] = {upload_id: job.uploadId || "", init_key: job.initKey || "",
      fingerprint: job.fingerprint, filename: job.file.name, size: job.file.size,
      replaces_id: job.replacesId, note_digest: job.noteDigest, complete_unknown: job.completeUnknown === true};
    try { localStorage.setItem(storageKey, JSON.stringify({uploads: savedUploads})); } catch (_) {}
  }
  function restore() {
    try {
      const saved = JSON.parse(localStorage.getItem(storageKey) || "null");
      if (saved && saved.uploads && typeof saved.uploads === "object") savedUploads = saved.uploads;
      else if (saved && typeof saved.fingerprint === "string" && typeof saved.upload_id === "string") {
        savedUploads[saved.fingerprint + ":new:"] = {upload_id: saved.upload_id,
          fingerprint: saved.fingerprint, complete_unknown: saved.complete_unknown === true};
      }
    } catch (_) { savedUploads = {}; }
  }
  async function fileFingerprint(file, job) {
    let chain = new Uint8Array(32);
    for (let start = 0; start < file.size; start += 4 * 1024 * 1024) {
      if (job.paused || stopped) throw new Error("已暂停文件核对。");
      const chunk = await crypto.subtle.digest("SHA-256", await file.slice(start, start + 4 * 1024 * 1024).arrayBuffer());
      const both = new Uint8Array(64); both.set(chain); both.set(new Uint8Array(chunk), 32);
      chain = new Uint8Array(await crypto.subtle.digest("SHA-256", both));
      const checked = Math.min(file.size, start + 4 * 1024 * 1024);
      job.progress.value = Math.min(99, checked * 100 / file.size);
      job.detailNode.textContent = "正在检查文件 · " + job.progress.value.toFixed(1) +
        "% · " + formatBytes(checked) + " / " + formatBytes(file.size);
    }
    const metadata = new TextEncoder().encode(JSON.stringify([file.name, file.size, file.type, file.lastModified]));
    const last = new Uint8Array(chain.length + metadata.length); last.set(chain); last.set(metadata, chain.length);
    return [...new Uint8Array(await crypto.subtle.digest("SHA-256", last))].map(v => v.toString(16).padStart(2, "0")).join("");
  }
  const partsOf = value => (Array.isArray(value.uploaded_parts || value.parts) ? (value.uploaded_parts || value.parts) : [])
    .map(row => typeof row === "number" ? {part_number: row} : row);
  async function uploadState(job) {
    const value = await request("/uploads/" + id(job.uploadId), {job});
    if ((value.head_sha && value.head_sha !== bound.head_sha) || (value.round_id && value.round_id !== bound.round_id)) {
      halt("上传属于其他 HEAD/round，已停止。"); throw new Error("轮次不匹配");
    }
    if (value.upload_id && value.upload_id !== job.uploadId) throw new Error("上传标识不匹配");
    job.state = {...job.state, ...value, parts: partsOf(value)}; return job.state;
  }
  const terminalReady = value => value && value.status === "ready" && value.verified === true;
  function updateJob(job, text) {
    job.statusNode.textContent = jobStatusLabel(job.status);
    if (text !== undefined) job.detailNode.textContent = text;
    job.actionButton.textContent = job.running ? "暂停" : (job.status === "succeeded" ? "已完成" : "重试 / 继续");
    job.actionButton.dataset.jobAction = job.running ? "pause" : "retry";
    job.actionButton.disabled = !writable() || (busy && job.actionButton.dataset.jobAction !== "pause") || job.status === "succeeded";
    job.actionButton.onclick = () => {
      if (job.running) {
        job.paused = true;
        for (const request of job.requests) abortRequest(request);
        updateJob(job, "暂停中；续传前会重新查询 COS 状态。");
      } else if (!busy && job.status !== "succeeded") runSingle(job);
    };
  }
  function makeJob(file, replacesId = "", note = "") {
    const row = document.createElement("li"), heading = document.createElement("div"), title = document.createElement("strong");
    const size = document.createElement("span"), progress = document.createElement("progress");
    const footer = document.createElement("div"), status = document.createElement("span"), detail = document.createElement("span");
    const action = document.createElement("button");
    row.className = "job"; heading.className = "job-heading"; title.className = "job-title"; size.className = "file-size";
    title.textContent = file.name; size.textContent = formatBytes(file.size); heading.append(title, size);
    progress.max = 100; progress.value = 0; progress.setAttribute("aria-label", "文件处理进度");
    footer.className = "job-footer"; status.className = "job-state"; status.setAttribute("aria-live", "polite");
    detail.className = "job-detail"; action.type = "button"; action.textContent = "重试 / 继续";
    footer.append(status, action); row.append(heading, progress, footer, detail);
    $("queue").append(row); $("queue-heading").hidden = false;
    const job = {file, row, progress, statusNode: status, detailNode: detail, actionButton: action,
      requests: new Set(), parts: new Map(), paused: false, running: false, status: "排队中",
      completeUnknown: false, replacesId, note};
    jobs.push(job); updateJob(job); return job;
  }
  function convertLongNoteToTextAttachment(note) {
    const file = new File([note], "附件说明-" + Date.now() + ".txt", {
      type: "text/plain;charset=utf-8", lastModified: Date.now()
    });
    if (file.size > MAX_BYTES) throw new Error("说明转换后的 TXT 超过 512 MiB，未上传；请缩短说明后重试。");
    return file;
  }
  function currentRoundFileQuota(replacementId) {
    if (!context || !Array.isArray(context.attachments) || !Array.isArray(context.operations)) return null;
    const records = new Map();
    for (const row of [...context.operations, ...context.attachments]) {
      if (!row || typeof row !== "object") return null;
      if (!row.round_id || !row.head_sha) return null;
      if (row.round_id !== bound.round_id || row.head_sha !== bound.head_sha) continue;
      if (isTextRecord(row)) continue;
      const attachmentId = row.attachment_id || row.id;
      if (typeof attachmentId !== "string" || !attachmentId) return null;
      records.set(attachmentId, row);
    }
    const completedFiles = [], pendingFiles = [];
    for (const row of records.values()) {
      if (isDeletedAttachment(row) || row.status === "invalid" ||
          ["failed", "aborted", "expired"].includes(row.status)) continue;
      if (typeof row.status !== "string") return null;
      if (!["ready", "initializing", "uploading", "completing", "verifying", "processing"].includes(row.status))
        return null;
      const size = Number(row.size);
      if (!Number.isSafeInteger(size) || size < 1) return null;
      const item = {id: row.attachment_id || row.id, size, row};
      (row.status === "ready" ? completedFiles : pendingFiles).push(item);
    }
    const serverIds = new Set(records.keys());
    for (const job of jobs) {
      if (job.status === "succeeded") continue;
      if (job.uploadId && serverIds.has(job.uploadId)) continue;
      if (!job.file || !Number.isSafeInteger(job.file.size) || job.file.size < 1) return null;
      pendingFiles.push({id: "local:" + job.fingerprint + ":" + (job.replacesId || "new"), size: job.file.size, row: job});
    }
    const activeFiles = [...completedFiles, ...pendingFiles];
    let fileCount = activeFiles.length;
    let totalBytes = activeFiles.reduce((sum, row) => sum + row.size, 0);
    if (!Number.isSafeInteger(totalBytes)) return null;
    if (replacementId) {
      const target = activeFiles.find(row => row.id === replacementId);
      if (!target || target.row.status !== "ready" || target.row.can_delete !== true)
        return {error: "替换目标已删除、不可替换或状态已变化；请刷新附件页面后重选。"};
      fileCount -= 1;
      totalBytes -= target.size;
    }
    const configuredFiles = context.upload_max_files;
    const configuredBytes = context.upload_max_bytes;
    const invite = context.auth_method === "signed_invite" || context.upload_invite === true;
    if (!Number.isSafeInteger(configuredFiles) || configuredFiles < 0 ||
        !Number.isSafeInteger(configuredBytes) || configuredBytes < 0) {
      if (invite) return null;
    }
    return {
      fileCount, totalBytes,
      maxFiles: Number.isSafeInteger(configuredFiles) && configuredFiles >= 0
        ? Math.min(MAX_FILES, configuredFiles) : MAX_FILES,
      maxBytes: Number.isSafeInteger(configuredBytes) && configuredBytes >= 0
        ? Math.min(MAX_TOTAL_BYTES, configuredBytes) : MAX_TOTAL_BYTES
    };
  }
  function validateSelectedFiles(files, replacementId = "") {
    const selected = Array.from(files || []);
    if (!selected.length) return "请选择文件，或填写超过 5000 字的说明以生成 TXT 附件。";
    if (selected.length > MAX_FILES) return "一次最多上传 20 个文件（自动生成的 TXT 也计入数量）。";
    let totalBytes = 0;
    for (let index = 0; index < selected.length; index++) {
      const file = selected[index], size = Number(file && file.size);
      if (!Number.isFinite(size) || size < 1)
        return "第 " + (index + 1) + " 个文件为空或大小无效；每个文件必须至少为 1 B。";
      if (size > MAX_BYTES)
        return "第 " + (index + 1) + " 个文件超过 512 MiB 单文件上限。";
      totalBytes += size;
      if (totalBytes > MAX_TOTAL_BYTES) return "所选文件总大小超过 10 GiB 上限。";
    }
    const quota = currentRoundFileQuota(replacementId);
    if (!quota) return "无法确认当前轮附件及剩余配额，请刷新页面后重试。";
    if (quota.error) return quota.error;
    const maxFiles = Math.min(MAX_FILES, quota.maxFiles);
    const maxBytes = Math.min(MAX_TOTAL_BYTES, quota.maxBytes);
    if (quota.fileCount + selected.length > maxFiles)
      return "加上当前轮已有文件后将超过 " + maxFiles + " 个文件的上限。";
    if (quota.totalBytes + totalBytes > maxBytes)
      return "加上当前轮已有文件后将超过 " + formatBytes(maxBytes) + " 的总大小上限。";
    return "";
  }
  async function saveStandaloneNote() {
    if (busy || !writable()) return;
    const text = $("note").value;
    if (!text.trim()) { message("请输入要单独保存的备注。"); return; }
    if ([...text].length > MAX_NOTE_CHARS) {
      message("备注超过 5000 字，正在将原文完整转为 UTF-8 TXT 附件。");
      await runUpload({longNoteOnly: true});
      return;
    }
    busy = true; controls();
    try {
      if (!noteOperation || noteOperation.text !== text) {
        const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(
          JSON.stringify([bound.head_sha, bound.round_id, text])));
        const hash = [...new Uint8Array(digest)].map(value => value.toString(16).padStart(2, "0")).join("");
        noteOperation = {text, key: "note-" + hash, saved: false};
      }
      if (!noteOperation.saved) {
        const result = await request("/notes", {method: "POST", key: noteOperation.key, body: {text}});
        if (result.status !== "ready" || result.cos_content_verified !== true)
          throw new Error("服务端尚未确认备注内容已校验保存；可再次尝试，操作会保持幂等。");
        noteOperation.saved = true;
      }
      try { await refresh(); }
      catch (_) {
        message("备注已保存并通过内容校验；页面暂未刷新，请稍后重试查看。");
        return;
      }
      message(context.summary_updated === true ? "备注已保存，页面已更新。" : "备注已安全保存，Summary 正在等待自动更新。");
    } catch (error) {
      message(error.message || "备注保存失败；可重试，操作会保持幂等。");
    } finally { busy = false; controls(); }
  }
  function progressJob(job, parts, inflight, partBytes) {
    let bytes = 0;
    for (const part of parts.values()) bytes += Math.min(partBytes, Math.max(0, job.file.size - (part.part_number - 1) * partBytes));
    for (const amount of inflight.values()) bytes += amount;
    job.progress.value = job.file.size ? Math.min(100, bytes * 100 / job.file.size) : 0;
    job.detailNode.textContent = (job.statusText || "上传中") + " · " + job.progress.value.toFixed(1) +
      "% · " + formatBytes(bytes) + " / " + formatBytes(job.file.size);
  }
  async function waitReady(job, attachmentId = "") {
    if (!job.isNote) { job.status = "verifying"; updateJob(job, "COS 已接收；等待 SHA256 校验及 Summary 更新。"); }
    for (let count = 0; count < 30 && !stopped && !job.paused; count++) {
      const status = await uploadState(job), current = await refresh(job);
      const item = (current.attachments || []).find(row =>
        (attachmentId && (row.attachment_id === attachmentId || row.id === attachmentId)) ||
        row.upload_id === job.uploadId);
      if (terminalReady(status) && current.summary_updated === true && item && item.status === "ready") {
        if (!job.isNote) {
          job.progress.value = 100; job.completeUnknown = false; job.status = "succeeded";
          job.statusText = "成功 · COS ready/verified，Summary 已更新";
          const key = job.fingerprint + ":" + (job.replacesId || "new") + ":" + job.noteDigest;
          delete savedUploads[key]; try { localStorage.setItem(storageKey, JSON.stringify({uploads: savedUploads})); } catch (_) {}
          updateJob(job, job.statusText);
        }
        return true;
      }
      await delay(2000);
    }
    throw new Error("COS ready 与 Summary 更新尚未同时确认；可重试以先查询真实状态。");
  }
  async function startOrRestore(job) {
    job.status = "核对中"; updateJob(job, "正在以 4 MiB 有界分块核对文件指纹。");
    job.fingerprint = await fileFingerprint(job.file, job);
    job.noteDigest = [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(job.note)))]
      .map(v => v.toString(16).padStart(2, "0")).join("");
    const resumeKey = job.fingerprint + ":" + (job.replacesId || "new") + ":" + job.noteDigest;
    const legacyKey = job.fingerprint + ":" + (job.replacesId || "new") + ":";
    const saved = savedUploads[resumeKey] || savedUploads[legacyKey];
    if (saved) {
      if (saved.filename && saved.filename !== job.file.name || saved.size && Number(saved.size) !== job.file.size ||
          saved.replaces_id && saved.replaces_id !== job.replacesId || saved.note_digest && saved.note_digest !== job.noteDigest)
        throw new Error("保存的上传记录与所选文件或备注不匹配，拒绝混用。");
      job.uploadId = saved.upload_id || ""; job.initKey = saved.init_key || uid();
      job.completeUnknown = saved.complete_unknown === true;
    } else job.initKey = uid();
    persist(job);
    if (!job.uploadId) {
      const initialized = await request("/uploads", {method: "POST", key: job.initKey,
        body: {filename: job.file.name, size: job.file.size, content_type: job.file.type || "application/octet-stream",
          note: job.note, replaces_id: job.replacesId || null}, job});
      id(initialized.upload_id); job.uploadId = initialized.upload_id;
      job.state = {...initialized, parts: partsOf(initialized)}; persist(job);
    }
    let state = await uploadState(job);
    if (state.status === "initializing" || state.initialization_pending === true) {
      const resumed = await request("/uploads/" + id(job.uploadId) + "/resume", {
        method: "POST", key: job.uploadId + ":resume", body: {}, job
      });
      if (resumed.upload_id && resumed.upload_id !== job.uploadId) throw new Error("恢复结果上传标识不匹配");
      state = await uploadState(job);
    }
    if (terminalReady(state)) return waitReady(job, state.attachment_id || "");
    if (["completing", "verifying", "processing"].includes(state.status)) return waitReady(job, state.attachment_id || "");
    return state;
  }
  async function uploadParts(job, state) {
    const partBytes = Number(state.part_bytes || PART_BYTES);
    if (!Number.isSafeInteger(partBytes) || partBytes < 1 || partBytes > PART_BYTES) throw new Error("服务端分片大小无效。");
    if (state.filename && state.filename !== job.file.name || state.size !== undefined && Number(state.size) !== job.file.size)
      throw new Error("服务端文件元数据与所选文件不匹配。");
    const count = Math.ceil(job.file.size / partBytes), done = new Map();
    for (const part of state.parts || []) {
      const n = Number(part.part_number);
      if (!Number.isInteger(n) || n < 1 || n > count ||
          part.size !== undefined && Number(part.size) !== Math.min(partBytes, job.file.size - (n - 1) * partBytes))
        throw new Error("COS 分片状态与本文件不匹配。");
      done.set(n, {...part, part_number: n});
    }
    const inflight = new Map(), queue = Array.from({length: count}, (_, n) => n + 1).filter(n => !done.has(n));
    let cursor = 0;
    const update = () => progressJob(job, done, inflight, partBytes);
    async function worker() {
      while (cursor < queue.length && !job.paused && !stopped) {
        const number = queue[cursor++], blob = job.file.slice((number - 1) * partBytes, Math.min(number * partBytes, job.file.size));
        let complete = false;
        for (let attempt = 0; attempt < 3 && !job.paused && !stopped; attempt++) {
          try {
            const query = "?head_sha=" + encodeURIComponent(bound.head_sha) + "&round_id=" + encodeURIComponent(bound.round_id);
            const path = "/uploads/" + id(job.uploadId) + "/parts/" + number + "/data" + query;
            await requestRaw(path, blob, job, amount => { inflight.set(number, amount); update(); });
            done.set(number, {part_number: number, size: blob.size}); complete = true;
          } catch (error) {
            inflight.delete(number); update();
            if (job.paused || stopped) break;
            if ([401, 403, 409, 413].includes(error.status)) throw error;
            const checked = await uploadState(job);
            const found = (checked.parts || []).find(part => Number(part.part_number) === number);
            if (found) {
              if (Number(found.size) !== blob.size) throw new ApiError(409, "COS 已有分片大小冲突，已停止该文件。");
              done.set(number, found); complete = true;
            } else if (attempt < 2) await delay((attempt + 1) * 500);
            else throw error;
          }
          if (complete) { inflight.delete(number); update(); break; }
        }
        if (!complete && !job.paused && !stopped) throw new Error("分片重试次数已用完。");
      }
    }
    const results = await Promise.allSettled([worker(), worker()]);
    const failed = results.find(result => result.status === "rejected");
    job.parts = done; progressJob(job, done, inflight, partBytes);
    if (failed) throw failed.reason;
    if (job.paused || stopped) throw new Error("文件上传已暂停。");
    return done;
  }
  async function completeUpload(job, attachmentId = "") {
    let state = await uploadState(job);
    if (terminalReady(state)) return waitReady(job, attachmentId || state.attachment_id || "");
    if (["completing", "verifying", "processing"].includes(state.status)) return waitReady(job, attachmentId || state.attachment_id || "");
    if (job.completeUnknown && state.status !== "uploading") throw new Error("完成状态仍未明确；未重复提交 complete。");
    job.completeUnknown = true; persist(job);
    try {
      const result = await request("/uploads/" + id(job.uploadId) + "/complete", {
        method: "POST", key: job.uploadId + ":complete", body: {}, job
      });
      if (result.attachment_id) attachmentId = result.attachment_id;
      if (terminalReady(result)) return waitReady(job, attachmentId);
      state = await uploadState(job);
      if (terminalReady(state)) return waitReady(job, attachmentId || state.attachment_id || "");
      if (["completing", "verifying", "processing"].includes(state.status)) return waitReady(job, attachmentId || state.attachment_id || "");
      throw new Error("complete 后 COS 尚未 ready；已先查询状态，没有盲目重试。");
    } catch (error) {
      try {
        state = await uploadState(job);
        if (terminalReady(state)) return waitReady(job, attachmentId || state.attachment_id || "");
        if (["completing", "verifying", "processing"].includes(state.status)) return waitReady(job, attachmentId || state.attachment_id || "");
      } catch (_) { /* Preserve the uncertain marker for an explicit retry. */ }
      throw error;
    }
  }
  async function uploadOne(job) {
    if (!(job.file.size >= 1 && job.file.size <= MAX_BYTES)) throw new Error("单个文件必须为 1..512 MiB。");
    let state = await startOrRestore(job);
    if (job.status === "succeeded") return;
    if (["failed", "deleted", "aborted", "expired", "invalid"].includes(state.status))
      throw new Error("服务端上传状态不可继续：" + state.status);
    await uploadParts(job, state);
    const completed = await uploadState(job), partBytes = Number(completed.part_bytes || PART_BYTES);
    const count = Math.ceil(job.file.size / partBytes);
    if ((completed.parts || []).length < count) throw new Error("COS 尚未确认全部分片；请继续重试。");
    await completeUpload(job, completed.attachment_id || "");
  }
  async function processJob(job) {
    if (job.running || !writable()) return;
    job.running = true; job.paused = false; job.status = "running"; updateJob(job, "处理中。"); controls();
    try {
      await uploadOne(job);
      if (job.status !== "succeeded") { job.status = "waiting"; updateJob(job, "等待 Summary 更新；重试会先查询 COS 状态。"); }
    } catch (error) {
      if (job.paused) { job.status = "paused"; updateJob(job, "已暂停；重试前会查询 COS 状态。"); }
      else if (!stopped) { job.status = "failed"; updateJob(job, "失败：" + (error.message || String(error))); }
    } finally { job.running = false; controls(); }
  }
  async function runSingle(job) {
    if (busy || stopped) return;
    busy = true; controls(); await processJob(job); busy = false; controls();
  }
  async function runUpload({longNoteOnly = false} = {}) {
    if (busy || !writable()) return;
    const note = $("note").value;
    let files = longNoteOnly ? [] : Array.from($("file").files || []);
    const longNote = [...note].length > MAX_NOTE_CHARS;
    const replacementId = longNoteOnly ? "" : $("replace").value || "";
    if (longNote) {
      try { files.push(convertLongNoteToTextAttachment(note)); }
      catch (error) { message(error.message || "说明无法转换为附件。"); return; }
    }
    if (replacementId && (longNote || files.length !== 1)) {
      message("替换模式每次只能选择一个文件；不能与多文件或自动生成的 TXT 同时使用。请选择“新增附件”或调整选择。");
      return;
    }
    if (replacementId && jobs.some(job => job.replacesId === replacementId && job.status !== "succeeded")) {
      message("该附件已有替换任务；请在上传队列中继续或重试，避免重复替换同一附件。");
      return;
    }
    const validationError = validateSelectedFiles(files, replacementId);
    if (validationError) { message(validationError); return; }
    if (!longNoteOnly) $("file").value = "";
    for (const file of files) {
      const job = makeJob(file, replacementId, longNote ? "" : note);
    }
    busy = true; controls();
    for (const job of jobs) {
      if (job.status !== "排队中" || stopped) continue;
      await processJob(job);
    }
    busy = false; controls(); if (!stopped) message("队列已处理；各文件状态与恢复进度分别显示在队列中。");
  }
  async function deleteAttachment(item) {
    if (busy || !writable()) return;
    if (!window.confirm("确认删除当前轮附件“" + (item.filename || "附件") +
        "”？删除将保留审计记录，历史轮次不会自动删除。")) return;
    busy = true; controls();
    const key = bound.round_id + ":delete:" + (item.attachment_id || item.id);
    try { await request("/attachments/" + id(item.attachment_id || item.id), {method: "DELETE", key, body: {}}); await refresh(); message("删除结果已刷新，审计记录保留。"); }
    catch (error) { message(error.message); try { await refresh(); } catch (_) {} }
    finally { busy = false; controls(); }
  }
  $("save-note").addEventListener("click", saveStandaloneNote);
  $("upload").addEventListener("click", runUpload);
  const dropzone = $("dropzone");
  for (const eventName of ["dragenter", "dragover"]) dropzone.addEventListener(eventName, event => {
    event.preventDefault(); dropzone.classList.add("dragging");
  });
  for (const eventName of ["dragleave", "drop"]) dropzone.addEventListener(eventName, event => {
    event.preventDefault(); dropzone.classList.remove("dragging");
    if (eventName === "drop" && event.dataTransfer && event.dataTransfer.files.length) {
      try { $("file").files = event.dataTransfer.files; }
      catch (_) { message("浏览器无法读取拖入文件；请使用文件选择框。"); }
    }
  });
  restore(); refresh().catch(error => {
    const status = error && Number(error.status || 0);
    const detail = String(error && error.message || "");
    $("current-section").hidden = true; $("history-section").hidden = true;
    if ([401, 403].includes(status) || /invalid_evidence_capability|invalid_evidence/i.test(detail))
      message("此附件链接当前不可用。请从 PR Summary 重新打开；附件只在测试阶段开放。");
    else message("暂时无法读取附件，请稍后从 PR Summary 重新打开。");
  });
})();
'''
