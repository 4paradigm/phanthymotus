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
<meta name="referrer" content="no-referrer"><title>部署验证附件</title>
<style>
body{{font:16px system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#172432;background:#f6f8fa}}
section{{background:white;border:1px solid #ccd5dd;border-radius:10px;padding:1rem;margin:1rem 0}}
label{{display:block;margin:.7rem 0}}input,textarea,select,button{{font:inherit;max-width:100%}}
textarea{{display:block;width:95%;min-height:5rem}}button{{padding:.45rem .8rem;margin:.25rem}}progress{{width:100%}}
#dropzone{{border:2px dashed #8395a7;border-radius:8px;padding:1rem;text-align:center}}
#dropzone.dragging{{background:#eef6ff;border-color:#2463a6}}
pre,code{{overflow-wrap:anywhere;white-space:pre-wrap}}li{{padding:.6rem 0;border-bottom:1px solid #ddd}}
#message{{white-space:pre-wrap}}[hidden]{{display:none!important}}
</style><script src="/evidence/assets/app.js" defer></script></head>
<body><h1>部署验证附件</h1>
<main id="app" data-api="{esc(api)}" data-repo="{esc(repo)}" data-pr="{esc(pr_number)}" data-head="{esc(head_sha)}" data-round="{esc(round_id)}">
<p>项目 <strong>{esc(repo)}</strong> · PR #{esc(pr_number)}<br>HEAD <code>{esc(head_sha)}</code><br>round <code>{esc(round_id)}</code></p>
<p id="message" role="status" aria-live="polite"></p><button id="refresh" type="button">刷新状态</button>
<section><h2>当前轮</h2><p id="summary">等待服务端状态</p>
<fieldset id="writes" disabled hidden><legend>上传与备注</legend>
<label id="dropzone">选择或拖入文件（每个文件最大 512 MiB，可多选）<input id="file" type="file" multiple></label>
<label>备注<textarea id="note" maxlength="10000"></textarea></label>
<button id="save-note" type="button">保存备注</button>
<label>替换当前轮附件<select id="replace"><option value="">新增附件</option></select></label>
<button id="upload" type="button">开始上传</button>
</fieldset><ul id="queue" aria-live="polite"></ul>
<ul id="attachments"></ul></section>
<section><h2>历史轮次（只读）</h2><div id="history"></div></section>
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
  const PART_BYTES = 8 * 1024 * 1024, MAX_BYTES = 512 * 1024 * 1024;
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
      (context.pr_state === undefined || context.pr_state === "open") && !context.merged && !stopped;
  }
  function controls() {
    const canWrite = writable();
    $("writes").hidden = !canWrite; $("writes").disabled = !canWrite;
    $("upload").disabled = busy || !canWrite; $("file").disabled = busy || !canWrite;
    $("note").disabled = busy || !canWrite; $("replace").disabled = busy || !canWrite;
    $("save-note").disabled = busy || !canWrite;
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
    const a = document.createElement("a"); a.textContent = label; a.href = api + path;
    a.setAttribute("download", ""); a.rel = "noopener"; return a;
  }
  function attachmentList(rows, parent, current) {
    for (const item of rows || []) {
      const li = document.createElement("li"), deleted = item.status === "deleted" || !!item.deleted_at;
      const attachmentId = item.attachment_id || item.id, description = document.createElement("span");
      description.textContent = [item.filename || item.text || "文本证据", item.size === undefined ? "" : item.size + " B",
        item.media_type || item.content_type || "", item.actor_verified === true ? (item.actor_login || item.actor || "") :
          item.upload_method === "signed_invite" ? "分享链接上传，GitHub 身份未验证" : "未验证身份",
        item.created_at || "", item.head_sha || "", item.round_id || "", item.sha256 || "", item.note || "",
        deleted ? "已删除 " + (item.deleted_at || "") + " " + (item.deleted_by || "") : item.status || "pending",
        item.replaces_id ? "替换 " + item.replaces_id : ""].filter(Boolean).join(" · ");
      li.append(description);
      if (!deleted && item.status === "ready" && attachmentId && item.download_capability) {
        li.append(" ", anchor("COS 下载", "/attachments/" + id(attachmentId) +
          "/download?cap=" + encodeURIComponent(item.download_capability)));
      } else if (!deleted && item.status === "ready" && attachmentId && item.object_key) {
        const hint = document.createElement("span");
        hint.textContent = " · 下载链接暂不可用";
        li.append(hint);
      }
      const same = current && (!item.head_sha || item.head_sha === bound.head_sha) &&
        (!item.round_id || item.round_id === bound.round_id);
      if (!deleted && same && writable() && item.can_delete === true && attachmentId) {
        const remove = document.createElement("button"); remove.type = "button"; remove.textContent = "删除";
        remove.dataset.delete = attachmentId; remove.addEventListener("click", () => deleteAttachment(item));
        li.append(" ", remove);
        if (item.status === "ready" && item.can_delete === true) {
          const option = document.createElement("option"); option.value = attachmentId;
          option.textContent = item.filename || attachmentId; $("replace").append(option);
        }
      }
      parent.append(li);
    }
  }
  function render() {
    $("summary").textContent = typeof context.summary === "string" ? context.summary : JSON.stringify(context.summary || {status: context.status});
    $("attachments").replaceChildren(); $("history").replaceChildren();
    const replacement = $("replace").value; $("replace").replaceChildren(new Option("新增附件", ""));
    attachmentList(context.attachments || [], $("attachments"), true);
    for (const round of context.rounds || context.history || []) {
      if (roundMatches(round)) continue;
      const details = document.createElement("details"), summary = document.createElement("summary");
      summary.textContent = "HEAD " + (round.head_sha || "unknown") + " · round " + (round.round_id || "unknown") + " · " + (round.status || "unknown");
      details.append(summary);
      const ul = document.createElement("ul"); attachmentList(round.attachments || [], ul, false); details.append(ul);
      if (round.round_id && (round.cos_object_key || round.cos?.object_key || round.log_available || round.log_ready) &&
          context.enabled !== false && context.evidence_enabled !== false)
        details.append(anchor("下载本轮日志", "/rounds/" + id(round.round_id) + "/log/download?cap=" + encodeURIComponent(round.capability || "")));
      $("history").append(details);
    }
    for (const archive of context.archive_links || []) {
      try {
        const url = new URL(archive.url), prefix = "/" + app.dataset.repo;
        if (url.origin !== "https://github.com" || url.username || url.password || url.search ||
            ![prefix + "/issues/" + app.dataset.pr, prefix + "/pull/" + app.dataset.pr].includes(url.pathname) ||
            !/^#issuecomment-[0-9]+$/.test(url.hash)) continue;
        const a = document.createElement("a"), p = document.createElement("p");
        a.textContent = "历史索引 · " + (archive.round_id || ""); a.href = url.href; a.rel = "noopener";
        p.append(a); $("history").append(p);
      } catch (_) { /* Ignore untrusted archive redirect destinations. */ }
    }
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
    if (!useProofFallback && proofSecret && next.auth_method !== "signed_invite" &&
        roundMatches(next) && !next.merged) {
      // A valid read-only page cookie is not an upload invitation. Preserve
      // the legacy proof path when it is available, but never replace stale
      // HEAD/round or closed-PR responses with a second auth result.
      useProofFallback = true;
    }
    if (useProofFallback) next = await request("/context", {job, authMode: "proof"});
    context = next;
    if (!roundMatches(next)) halt("当前 HEAD/round 已变化，已停止写入。请刷新 PR Summary 并打开最新上传链接。");
    else if ((next.pr_state && next.pr_state !== "open") || next.merged)
      halt("PR 已关闭或已合并，已停止写入。请刷新 PR Summary 并打开最新上传链接。");
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
      job.progress.value = Math.min(99, (start + 4 * 1024 * 1024) * 100 / file.size);
      job.statusNode.textContent = "核对指纹 " + Math.min(file.size, start + 4 * 1024 * 1024) + " / " + file.size + " B";
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
    if (text) job.statusNode.textContent = text;
    job.actionButton.textContent = job.running ? "暂停" : (job.status === "succeeded" ? "完成" : "重试 / 继续");
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
  function makeJob(file) {
    const row = document.createElement("li"), title = document.createElement("strong");
    const progress = document.createElement("progress"), status = document.createElement("span"), action = document.createElement("button");
    title.textContent = file.name; progress.max = 100; progress.value = 0;
    status.textContent = "排队中"; action.type = "button"; action.textContent = "重试 / 继续";
    row.append(title, document.createElement("br"), progress, document.createElement("br"), status, " ", action);
    $("queue").append(row);
    const job = {file, row, progress, statusNode: status, actionButton: action,
      requests: new Set(), parts: new Map(), paused: false, running: false, status: "排队中", completeUnknown: false};
    jobs.push(job); updateJob(job); return job;
  }
  function progressJob(job, parts, inflight, partBytes) {
    let bytes = 0;
    for (const part of parts.values()) bytes += Math.min(partBytes, Math.max(0, job.file.size - (part.part_number - 1) * partBytes));
    for (const amount of inflight.values()) bytes += amount;
    job.progress.value = job.file.size ? Math.min(100, bytes * 100 / job.file.size) : 0;
    job.statusNode.textContent = (job.statusText || "上传中") + " · " + job.progress.value.toFixed(1) +
      "% · " + bytes + " / " + job.file.size + " B";
  }
  async function waitReady(job, attachmentId = "") {
    if (!job.isNote) updateJob(job, "COS 已接收；等待 SHA256 校验及 Summary 投影。");
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
    job.note = $("note").value; job.replacesId = $("replace").value || "";
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
  async function runUpload() {
    if (busy || !writable()) return;
    const files = Array.from($("file").files || []); $("file").value = "";
    if (!files.length) { message("请选择一个或多个文件。"); return; }
    for (const file of files) makeJob(file);
    busy = true; controls();
    for (const job of jobs) {
      if (job.status !== "排队中" || stopped) continue;
      await processJob(job);
    }
    busy = false; controls(); if (!stopped) message("队列已处理；各文件状态与恢复进度分别显示在队列中。");
  }
  async function deleteAttachment(item) {
    if (busy || !writable()) return;
    if (!window.confirm("确认删除当前轮附件“" + (item.filename || item.attachment_id || item.id) +
        "”？删除将保留审计记录，历史轮次不会自动删除。")) return;
    busy = true; controls();
    const key = bound.round_id + ":delete:" + (item.attachment_id || item.id);
    try { await request("/attachments/" + id(item.attachment_id || item.id), {method: "DELETE", key, body: {}}); await refresh(); message("删除结果已刷新，审计记录保留。"); }
    catch (error) { message(error.message); try { await refresh(); } catch (_) {} }
    finally { busy = false; controls(); }
  }
  async function saveNote() {
    if (busy || !writable()) return;
    const body = {text: $("note").value, note: ""};
    if (!body.text.trim()) { message("请输入备注。"); return; }
    const content = JSON.stringify(body);
    if (!noteOperation || noteOperation.content !== content) noteOperation = {content, key: uid()};
    busy = true; controls();
    try {
      const result = await request("/notes", {method: "POST", key: noteOperation.key, body});
      await refresh();
      const noteJob = {uploadId: result.upload_id, requests: new Set(), isNote: true};
      if (result.upload_id && (result.attachment_id || result.id)) await waitReady(noteJob, result.attachment_id || result.id);
      else message("备注已提交；请刷新服务端状态确认 Summary 更新。");
      noteOperation = null;
    } catch (error) { message(error.message); }
    finally { busy = false; controls(); }
  }
  $("upload").addEventListener("click", runUpload);
  $("save-note").addEventListener("click", saveNote);
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
  $("refresh").addEventListener("click", async () => {
    try { if (busy) { message("上传进行中；可暂停对应文件后刷新。"); return; } stopped = false; await refresh(); }
    catch (error) { message(error.message); }
  });
  restore(); refresh().catch(error => { message(error.message || String(error)); });
})();
'''
