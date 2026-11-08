"""Dependency-free evidence page. Authentication and storage belong to the API.

Serve javascript_source() at /evidence/assets/app.js with a JavaScript MIME type.
No credentials, signed URLs or CSRF tokens are persisted by this client.
"""
from __future__ import annotations

import html

from .comments import _evidence_path


def render_upload_page(repo, pr_number, head_sha, round_id, *, read_only=False) -> str:
    api = _evidence_path(repo, pr_number)
    if not api:
        raise ValueError("invalid repository or PR")
    esc = lambda value: html.escape(str(value), quote=True)
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="referrer" content="no-referrer"><title>上传测试证据</title>
<style>
*{{box-sizing:border-box}}
body{{margin:0;background:#f5f5f7;color:#1d1d1f;font:16px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif}}
main{{width:min(100% - 32px,680px);margin:48px auto 72px}}
h1{{margin:0;font-size:clamp(28px,5vw,36px);line-height:1.15;letter-spacing:-.035em}}
.subtitle{{margin:10px 0 26px;color:#6e6e73}}
section.panel{{padding:24px;background:#fff;border:1px solid #e5e5ea;border-radius:16px}}
button,input,textarea{{font:inherit}}
button{{min-height:44px;padding:9px 16px;border:1px solid #0071e3;border-radius:10px;background:#0071e3;color:#fff;font-weight:600;cursor:pointer}}
button:disabled{{cursor:not-allowed;opacity:.55}}
button:focus-visible,input:focus-visible,textarea:focus-visible{{outline:3px solid #77aaff;outline-offset:2px}}
#message{{margin:0 0 16px;color:#b42318;white-space:pre-wrap}}
#message:empty{{display:none}}
fieldset{{min-width:0;margin:0;padding:0;border:0}}
#dropzone{{display:flex;min-height:132px;align-items:center;justify-content:center;flex-direction:column;gap:8px;padding:20px;border:1px dashed #a1a1a6;border-radius:13px;text-align:center;cursor:pointer}}
#dropzone.dragging{{border-color:#0071e3;background:#f0f7ff}}
#dropzone input{{max-width:100%;font-size:14px}}
.file-hint{{margin:5px 0 0;color:#6e6e73;font-size:13px}}
label.control{{display:block;margin-top:18px;color:#424245;font-size:14px;font-weight:600}}
textarea{{display:block;width:100%;min-height:78px;margin-top:6px;padding:10px 12px;border:1px solid #d2d2d7;border-radius:9px;background:#fff;color:#1d1d1f;resize:vertical}}
.primary-row{{display:flex;justify-content:flex-end;margin-top:14px}}
.list-heading{{margin:24px 0 8px;font-size:16px}}
.queue{{list-style:none;margin:0;padding:0}}
.queue:empty{{display:none}}
.job{{margin:10px 0;padding:15px;border:1px solid #e5e5ea;border-radius:12px;background:#fff}}
.job-heading{{display:flex;align-items:flex-start;justify-content:space-between;gap:12px}}
.job-title{{min-width:0;overflow-wrap:anywhere;font-weight:600}}
.file-size{{flex-shrink:0;color:#6e6e73;font-size:14px;white-space:nowrap}}
progress{{display:block;width:100%;height:8px;margin:12px 0 7px;accent-color:#0071e3}}
.job-footer{{display:flex;align-items:center;justify-content:space-between;gap:12px}}
.job-state{{color:#424245;font-size:14px;font-weight:600}}
.job-detail{{color:#6e6e73;font-size:13px;overflow-wrap:anywhere}}
.job-footer button{{min-height:40px;background:#fff;color:#1d1d1f;border-color:#d2d2d7}}
[hidden]{{display:none!important}}
@media (max-width:560px){{
  main{{width:min(100% - 24px,680px);margin:28px auto 48px}}
  section.panel{{padding:18px;border-radius:14px}}
  .job-heading{{gap:6px;flex-direction:column}}
  .file-size{{white-space:normal}}
  .job-footer{{align-items:flex-start;flex-direction:column}}
  .job-footer button{{width:100%}}
}}
</style><script src="/evidence/assets/app.js" defer></script></head>
<body>
<main id="app" data-api="{esc(api)}" data-repo="{esc(repo)}" data-pr="{esc(pr_number)}" data-head="{esc(head_sha)}" data-round="{esc(round_id)}" data-read-only="{str(bool(read_only)).lower()}">
<header><h1>上传测试证据</h1><p class="subtitle">这里上传供验收人查看的测试效果视频、截图等证据。</p></header>
<p id="message" role="status" aria-live="polite"></p>
<section id="upload-section" class="panel">
<fieldset id="writes" disabled hidden>
<label id="dropzone" for="file"><strong>选择文件，或拖放到这里</strong><span class="file-hint">最多 20 个文件 · 单个文件最大 512 MiB · 总计最大 10 GiB</span><input id="file" type="file" multiple></label>
<label class="control" for="note">文件说明（可选）<textarea id="note" maxlength="5000" aria-describedby="note-help"></textarea><span id="note-help" class="file-hint">说明会随本次选择的文件一起提供给验收人。</span></label>
<div class="primary-row"><button id="upload" type="button">上传测试证据</button></div>
</fieldset>
<h2 class="list-heading" id="queue-heading" hidden>上传进度</h2><ul id="queue" class="queue" aria-labelledby="queue-heading" aria-live="polite"></ul>
</section>
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
  let context = null, stopped = false, pageLeaving = false, busy = false, proofSecret = "", clientKey = "", savedUploads = {};
  let contextAuthorized = false;
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
  const jobStatusLabel = status => ({
    排队中: "等待上传", 核对中: "检查文件中", running: "正在上传", verifying: "确认上传中",
    waiting: "等待恢复", succeeded: "上传成功", failed: "失败", paused: "已暂停"
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
    return contextAuthorized && context && roundMatches(context) && context.enabled !== false && context.evidence_enabled !== false &&
      context.can_write === true && context.status === "testing" &&
      (secureInvite || (invite && proofTransportAllowed(new URL(api, window.location.href))) || proofUser) &&
      context.readiness && context.readiness.COS_READY === true &&
      context.readiness.EVIDENCE_BACKEND_READY === true &&
      context.readiness.BROWSER_UPLOAD_READY === true &&
      !stopped && !pageLeaving;
  }
  function controls() {
    const canWrite = writable();
    $("writes").hidden = !canWrite; $("writes").disabled = !canWrite;
    $("upload").disabled = busy || !canWrite; $("file").disabled = busy || !canWrite;
    $("note").disabled = busy || !canWrite;
    for (const button of document.querySelectorAll("button[data-job-action]"))
      button.disabled = !canWrite || (busy && button.dataset.jobAction !== "pause");
  }
  function abortRequest(request, reason = "stopped") {
    try {
      request._evidenceAbortReason = reason;
      if (request instanceof AbortController)
        request.abort(new DOMException(reason, "AbortError"));
      else request.abort();
    } catch (_) {}
  }
  function halt(text) {
    stopped = true;
    for (const request of active) abortRequest(request, "halt");
    controls(); message(text);
  }
  window.addEventListener("pagehide", () => {
    pageLeaving = true;
    for (const request of active) abortRequest(request, "pagehide");
  });
  window.addEventListener("pageshow", event => {
    if (!event.persisted || !pageLeaving) return;
    contextAuthorized = false;
    pageLeaving = false;
    controls();
    refresh().catch(refreshFailure);
  });
  class ApiError extends Error {
    constructor(status, text, detail = "") { super(text); this.status = status; this.detail = detail; }
  }
  function clearProof() {
    proofSecret = "";
    try { sessionStorage.removeItem(proofKey); } catch (_) {}
  }
  function requestStage(path, method) {
    if (path === "/context") return "核对当前测试轮次";
    if (path === "/uploads" && method === "POST") return "准备上传";
    if (/\/resume$/.test(path)) return "恢复上传";
    if (/\/complete$/.test(path)) return "正在确认上传结果";
    if (/\/uploads\/[^/]+$/.test(path)) return "检查上传进度";
    return method === "GET" ? "检查上传进度" : "保存附件";
  }
  function requestTimeout(path, method) {
    if (path === "/context") return 20000;
    if (path === "/uploads" && method === "POST" || /\/resume$/.test(path)) return 60000;
    if (/\/complete$/.test(path)) return 120000;
    return 20000;
  }
  function evidenceRequestHeaders(target, method, authMode = "auto") {
    const headers = {Accept: "application/json", "X-Evidence-Client-Key": clientKey};
    const sendProof = proofSecret && (authMode === "proof" ||
      authMode === "auto" && !(context && (context.auth_method === "signed_invite" || context.upload_invite === true))) &&
      proofTransportAllowed(target);
    const transportAllowed = sendProof ? proofTransportAllowed(target) : inviteTransportAllowed(target);
    if (sendProof) {
      headers["X-Evidence-Secret"] = proofSecret;
      headers["X-Evidence-Nonce"] = randomHex();
    } else if (method !== "GET" && !transportAllowed) {
      throw new Error("当前链接没有有效上传授权，或请求不是安全的同源 HTTPS 请求。");
    }
    return headers;
  }
  async function request(path, {method = "GET", body, key, job, timeoutMs, authMode = "auto"} = {}) {
    const write = method !== "GET";
    if (write && !writable()) throw new Error("当前测试轮次已变化或无法上传，请刷新 PR Summary 后重新打开页面。");
    const target = new URL(api + path, window.location.href);
    if (path === "/context") {
      target.searchParams.set("head_sha", bound.head_sha);
      target.searchParams.set("round_id", bound.round_id);
      if (app.dataset.readOnly === "true") target.searchParams.set("read_view", "1");
    }
    if (target.origin !== window.location.origin) throw new Error("拒绝跨站 Evidence 请求。");
    const headers = evidenceRequestHeaders(target, method, authMode);
    if (write) headers["Content-Type"] = "application/json";
    let stage = requestStage(path, method);
    if (job && job.status === "verifying" && method === "GET" && /\/uploads\/[^/]+$/.test(path))
      stage = "正在确认上传结果";
    const boundedTimeoutMs = Number.isFinite(timeoutMs) && timeoutMs > 0
      ? timeoutMs : requestTimeout(path, method);
    if (job) {
      job.statusText = stage;
      job.detailNode.textContent = stage + "…";
    }
    const controller = new AbortController(); active.add(controller); if (job) job.requests.add(controller);
    let timedOut = false;
    const timeout = setTimeout(() => {
      timedOut = true; controller.abort(new DOMException("request timeout", "TimeoutError"));
    }, boundedTimeoutMs);
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
        if (response.status === 403) throw new ApiError(403, "当前链接没有此操作权限。请从 PR Summary 获取有效上传邀请；历史轮次仅支持查看。", detail);
        if (response.status === 503) throw new ApiError(503, "附件或 GitHub 服务暂时无法读取；当前操作未确认完成。请稍后重试，或重新打开 PR Summary 中的附件页面。", detail);
        if (response.status === 409 && ["round_changed", "context_not_writable", "pr_closed",
          "pr_closed_or_mismatched", "pr_draft", "state_unavailable"].includes(detail))
          halt("HEAD/round 或 PR 状态已变化，已停止所有写入。请刷新 PR Summary 并打开最新上传链接。");
        if (response.status === 413) throw new ApiError(413, "超过服务端大小限制。 " + detail, detail);
        throw new ApiError(response.status, response.status + ": " + (detail || response.statusText), detail);
      }
      return value;
    } catch (error) {
      if (error instanceof ApiError) throw error;
      if (job && job.paused || controller._evidenceAbortReason === "paused")
        throw new Error("已暂停；继续前会先查询服务端上传状态。");
      if (controller._evidenceAbortReason === "halt")
        throw new Error("HEAD/round 或权限状态已变化，写入已停止。");
      if (controller._evidenceAbortReason === "pagehide")
        throw new Error(stage + "因页面离开而中断；结果可能未知，重新打开后会先查询服务端状态。");
      if (controller._evidenceAbortReason === "stopped")
        throw new Error("请求已主动停止；结果可能未知，继续前会查询服务端状态。");
      const seconds = Math.ceil(boundedTimeoutMs / 1000);
      if (timedOut)
        throw new ApiError(0, stage + "超时（" + seconds + " 秒）；" +
          (write ? "操作结果未知，重试前会先查询服务端状态。" : "尚未取得服务端确认，请稍后重试。"));
      if (error && error.name === "AbortError")
        throw new Error("请求已停止；继续前会先查询服务端上传状态。");
      if (error instanceof TypeError || error && error.name === "NetworkError")
        throw new ApiError(0, stage + "网络中断；" +
          (write ? "操作结果未知，重试前会先查询服务端状态。" : "请检查连接后重试。"));
      throw error;
    } finally { clearTimeout(timeout); active.delete(controller); if (job) job.requests.delete(controller); }
  }
  function uploadControllerPart(uploadId, partNumber, blob, job, onProgress) {
    return new Promise((resolve, reject) => {
      if (!writable()) return reject(new Error("当前轮不可写，请刷新并核对 HEAD/round。"));
      const path = "/uploads/" + id(uploadId) + "/parts/" + partNumber + "/data";
      const target = new URL(api + path, window.location.href);
      if (target.origin !== window.location.origin || target.protocol !== "https:")
        return reject(new Error("分片上传只允许当前 HTTPS 来源。"));
      target.searchParams.set("head_sha", bound.head_sha);
      target.searchParams.set("round_id", bound.round_id);
      let headers;
      try {
        headers = evidenceRequestHeaders(target, "PUT");
        headers["Content-Type"] = "application/octet-stream";
      } catch (error) { return reject(error); }
      job.statusText = "文件发送中 · 分片 " + partNumber;
      job.detailNode.textContent = job.statusText + "…";
      const xhr = new XMLHttpRequest(); active.add(xhr); job.requests.add(xhr);
      let finished = false;
      const cleanup = () => {
        if (finished) return;
        finished = true; active.delete(xhr); job.requests.delete(xhr);
      };
      try {
        xhr.open("PUT", target.href); xhr.withCredentials = true; xhr.timeout = 120000;
        for (const [name, value] of Object.entries(headers)) xhr.setRequestHeader(name, value);
      } catch (_) {
        cleanup(); reject(new Error("同源分片请求无法启动。")); return;
      }
      xhr.upload.onprogress = event => { if (event.lengthComputable) onProgress(event.loaded); };
      xhr.onload = () => {
        cleanup();
        let value = {};
        try { value = JSON.parse(xhr.responseText || "{}"); } catch (_) {}
        if (xhr.status >= 200 && xhr.status < 300) {
          if (value.uploaded === true && Number(value.part_number) === partNumber && Number(value.size) === blob.size)
            return resolve(value);
          const error = new ApiError(503, "同源分片响应结果不明确。", "unclear_upload_result");
          error.resultUnclear = true; return reject(error);
        }
        const detail = typeof value.detail === "string" ? value.detail : "";
        if (xhr.status === 401) return reject(new ApiError(401, "上传授权无效或已过期。", detail));
        if (xhr.status === 403) return reject(new ApiError(403, "上传被服务端拒绝；权限未绕过。", detail));
        if (xhr.status === 409) {
          if (["round_changed", "context_not_writable", "pr_closed", "pr_closed_or_mismatched",
            "pr_draft", "state_unavailable"].includes(detail))
            halt("HEAD/round 或 PR 状态已变化，已停止所有写入。请刷新当前 PR Summary。");
          return reject(new ApiError(409, "服务端拒绝该分片；请先核对当前上传状态。", detail));
        }
        if (xhr.status === 413) return reject(new ApiError(413, "分片超过服务端大小限制。", detail));
        if (xhr.status === 503) {
          const error = new ApiError(503, "附件服务暂时不可用；分片状态尚待核对。", detail);
          error.resultUnclear = true; return reject(error);
        }
        const error = new ApiError(xhr.status || 503, "同源分片上传未确认成功。", detail);
        error.resultUnclear = xhr.status === 0 || xhr.status >= 500;
        reject(error);
      };
      xhr.onerror = () => {
        cleanup(); const error = new ApiError(0, "同源分片网络中断；结果未知。");
        error.resultUnclear = true; reject(error);
      };
      xhr.ontimeout = () => {
        cleanup(); const error = new ApiError(0, "同源分片超时（120 秒）；结果未知。");
        error.resultUnclear = true; reject(error);
      };
      xhr.onabort = () => {
        cleanup();
        reject(new Error(xhr._evidenceAbortReason === "halt"
          ? "HEAD/round 或权限状态已变化，上传已停止。"
          : xhr._evidenceAbortReason === "pagehide"
            ? "页面离开导致上传中断；重新打开后会先核对进度。"
            : "上传已暂停；继续前会先核对进度。"));
      };
      xhr.send(blob);
    });
  }
  function isDeletedAttachment(item) {
    return !!(item && (item.status === "deleted" || item.deleted_at));
  }
  function isTextRecord(item) {
    return !!(item && (item.status === "text" || item.evidence_kind === "note" || item.kind === "note"));
  }
  function render() {
    const degraded = context && context.evidence_degraded === true;
    const pageAvailable = !!context && (roundMatches(context) || typeof context.status === "string");
    if (degraded) {
      message("");
      const label = document.createTextNode("附件状态暂不可用，上传已暂停。 ");
      const retry = document.createElement("button"); retry.type = "button"; retry.textContent = "重试";
      retry.addEventListener("click", () => refresh().catch(refreshFailure));
      $("message").append(label, retry);
    } else if (!pageAvailable) {
      message("无法确认当前测试轮次，上传已暂停。请从 PR Summary 重新打开附件页。");
    } else if (!writable()) {
      message("当前页面没有本轮上传权限，请从 PR Summary 的上传入口重新打开。");
    } else {
      message("");
    }
    $("writes").hidden = !pageAvailable || degraded || !writable();
    $("writes").disabled = !pageAvailable || degraded || !writable();
    controls();
  }
  async function refresh(job, timeoutMs = 20000) {
    let next, useProofFallback = false;
    try {
      // Let the browser's HttpOnly upload cookie authenticate first. An old
      // proof is a one-time compatibility fallback only after invite auth fails.
      next = await request("/context", {job, authMode: "invite", timeoutMs});
    } catch (inviteError) {
      if (!proofSecret || ![401, 403].includes(inviteError.status)) throw inviteError;
      useProofFallback = true;
    }
    // The API owns the authoritative live-PR check. Do not repeat it here:
    // context responses may be served by compatible backends that omit the
    // presentation-only PR fields, which must not disable a valid open-PR invite.
    if (!useProofFallback && proofSecret && next.auth_method !== "signed_invite" &&
        roundMatches(next)) {
      // A valid read-only page cookie is not an upload invitation. Preserve
      // the legacy proof path when it is available, but never replace stale
      // HEAD/round or closed-PR responses with a second auth result.
      useProofFallback = true;
    }
    if (useProofFallback) next = await request("/context", {job, authMode: "proof", timeoutMs});
    context = next;
    contextAuthorized = next.authenticated === true && next.enabled === true;
    if (!roundMatches(next)) halt("当前 HEAD/round 已变化，已停止写入。请刷新 PR Summary 并打开最新上传链接。");
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
    if (typeof job.fingerprint === "string" && /^[0-9a-f]{64}$/.test(job.fingerprint))
      return job.fingerprint;
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
    job.fingerprint = [...new Uint8Array(await crypto.subtle.digest("SHA-256", last))]
      .map(v => v.toString(16).padStart(2, "0")).join("");
    return job.fingerprint;
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
  function temporaryNetworkFailure(error) {
    return error && (error.status === 0 || error.status === 429 ||
      Number(error.status) >= 500 || error instanceof TypeError ||
      error.name === "NetworkError" || error.name === "TimeoutError");
  }
  async function uploadStateWithRetry(job) {
    let lastError;
    for (let attempt = 0; attempt < 4 && !job.paused && !stopped && !pageLeaving; attempt++) {
      try { return await uploadState(job); }
      catch (error) {
        lastError = error;
        if (!temporaryNetworkFailure(error) || attempt === 3) break;
        await delay([300, 700, 1400][attempt]);
      }
    }
    if (lastError && temporaryNetworkFailure(lastError)) {
      lastError.recoveryRetryable = true;
      lastError.message = "暂时无法查询 COS 分片状态；上传标识和文件指纹已保留，点击“重试 / 继续”会先恢复状态。";
    }
    throw lastError || new Error("上传状态查询已暂停；重试时会重新核对 COS 状态。");
  }
  const terminalReady = value => value && value.status === "ready" &&
    value.github_record_persisted === true &&
    (value.verified === true || value.cos_content_verified === true);
  const verifiedReady = value => terminalReady(value) &&
    typeof value.sha256 === "string" && /^[0-9a-f]{64}$/.test(value.sha256) &&
    value.head_sha === bound.head_sha && value.round_id === bound.round_id;
  function updateJob(job, text) {
    job.statusNode.textContent = jobStatusLabel(job.status);
    if (text !== undefined) job.detailNode.textContent = text;
    job.actionButton.textContent = job.running ? "暂停" : (job.status === "succeeded" ? "已完成" : "重试 / 继续");
    job.actionButton.dataset.jobAction = job.running ? "pause" : "retry";
    job.actionButton.disabled = !writable() || (busy && job.actionButton.dataset.jobAction !== "pause") || job.status === "succeeded";
    job.actionButton.onclick = () => {
      if (job.running) {
        job.paused = true;
        job.pauseReason = "已暂停；继续前会重新查询 COS 状态。";
        for (const request of job.requests) abortRequest(request, "paused");
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
    if (!selected.length) return "请选择要上传的测试视频或图片。";
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
  function progressJob(job, parts, inflight, partBytes) {
    let bytes = 0;
    for (const part of parts.values()) bytes += Math.min(partBytes, Math.max(0, job.file.size - (part.part_number - 1) * partBytes));
    for (const amount of inflight.values()) bytes += amount;
    job.progress.value = job.file.size ? Math.min(100, bytes * 100 / job.file.size) : 0;
    job.detailNode.textContent = (job.statusText || "上传中") + " · " + job.progress.value.toFixed(1) +
      "% · " + formatBytes(bytes) + " / " + formatBytes(job.file.size);
  }
  async function waitReady(job, initialStatus = null, attachmentId = "") {
    job.status = "verifying";
    job.statusText = "上传完成，正在归档附件";
    updateJob(job, job.statusText + "。");
    let status = initialStatus;
    for (let count = 0; count < 30 && !stopped && !job.paused; count++) {
      if (!status) status = await uploadStateWithRetry(job);
      if (verifiedReady(status)) {
        job.progress.value = 100; job.completeUnknown = false; job.status = "succeeded";
        job.statusText = "上传成功，文件和 GitHub 记录已确认。";
        const key = job.fingerprint + ":" + (job.replacesId || "new") + ":" + job.noteDigest;
        delete savedUploads[key]; try { localStorage.setItem(storageKey, JSON.stringify({uploads: savedUploads})); } catch (_) {}
        updateJob(job, job.statusText);
        return true;
      }
      if (status && status.status === "ready") {
        if (status === initialStatus) {
          initialStatus = null; status = null;
          continue;
        }
        job.status = "waiting";
        updateJob(job, "上传记录尚未完成保存；请点“重试 / 继续”核对并恢复。文件和备注仍保留在本地队列中。");
        return false;
      }
      status = null;
      await delay(2000);
    }
    job.status = "waiting";
    updateJob(job, "上传尚未确认完成；请点“重试 / 继续”核对状态并恢复。若授权失效，请从 PR Summary 重新打开上传页。");
    return false;
  }
  async function startOrRestore(job) {
    job.status = "核对中"; updateJob(job, "正在检查文件并准备上传。");
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
    let state;
    if (!job.uploadId) {
      const initialized = await request("/uploads", {method: "POST", key: job.initKey,
        body: {filename: job.file.name, size: job.file.size, content_type: job.file.type || "application/octet-stream",
          note: job.note, replaces_id: job.replacesId || null}, job});
      if (initialized.head_sha !== bound.head_sha || initialized.round_id !== bound.round_id) {
        halt("上传属于其他 HEAD/round，已停止。请刷新后重试。");
        throw new Error("轮次不匹配");
      }
      id(initialized.upload_id); job.uploadId = initialized.upload_id;
      state = {...initialized, parts: partsOf(initialized)};
      job.state = state; persist(job);
    } else state = await uploadStateWithRetry(job);
    if (state.status === "initializing" || state.initialization_pending === true) {
      const resumed = await request("/uploads/" + id(job.uploadId) + "/resume", {
        method: "POST", key: job.uploadId + ":resume", body: {}, job
      });
      if (resumed.upload_id && resumed.upload_id !== job.uploadId) throw new Error("恢复结果上传标识不匹配");
      state = await uploadStateWithRetry(job);
    }
    if (state.status === "ready" && !verifiedReady(state) || state.record_pending === true) {
      state = await request("/uploads/" + id(job.uploadId) + "/resume", {
        method: "POST", key: job.uploadId + ":resume", body: {}, job
      });
      if (state.upload_id && state.upload_id !== job.uploadId) throw new Error("恢复结果上传标识不匹配");
    }
    if (state.status === "ready") return waitReady(job, state, state.attachment_id || "");
    if (["completing", "verifying", "processing"].includes(state.status))
      return waitReady(job, state, state.attachment_id || "");
    return state;
  }
  async function uploadParts(job, state) {
    const partBytes = Number(state.part_bytes || PART_BYTES);
    if (!Number.isSafeInteger(partBytes) || partBytes < 1 || partBytes > PART_BYTES) throw new Error("上传配置异常，请刷新页面后重试。");
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
    function acceptListedPart(checked, number, blob) {
      const found = (checked.parts || []).find(part => Number(part.part_number) === number);
      if (!found) return false;
      if (Number(found.size) !== blob.size) throw new ApiError(409, "上传进度与所选文件不匹配，请刷新页面后重试。");
      done.set(number, found); return true;
    }
    let workerFailure = null;
    async function runWorker() {
      try {
        while (!workerFailure && cursor < queue.length && !job.paused && !stopped && !pageLeaving) {
          const number = queue[cursor++], blob = job.file.slice((number - 1) * partBytes, Math.min(number * partBytes, job.file.size));
          let complete = false;
          for (let attempt = 0; attempt < 3 && !job.paused && !stopped && !pageLeaving; attempt++) {
            try {
              job.statusText = "正在上传文件";
              job.detailNode.textContent = job.statusText + "…";
              const uploaded = await uploadControllerPart(job.uploadId, number, blob, job,
                amount => { inflight.set(number, amount); update(); });
              done.set(number, uploaded); complete = true;
            } catch (error) {
              inflight.delete(number); update();
              if (job.paused || stopped || pageLeaving) break;
              if ([401, 403, 409, 413].includes(error.status)) throw error;
              if (error.resultUnclear === true) {
                const checked = await uploadStateWithRetry(job);
                if (acceptListedPart(checked, number, blob)) {
                  complete = true;
                } else if (attempt < 2) await delay((attempt + 1) * 500);
                else throw error;
              } else if (attempt < 2 && temporaryNetworkFailure(error)) await delay((attempt + 1) * 500);
              else throw error;
            }
            if (complete) { inflight.delete(number); update(); break; }
          }
          if (!complete && !job.paused && !stopped && !pageLeaving)
            throw new Error("上传中断，请点“重试 / 继续”后恢复。");
        }
      } catch (error) {
        if (!workerFailure) workerFailure = error;
      }
    }
    await Promise.all([runWorker(), runWorker()]);
    job.parts = done; progressJob(job, done, inflight, partBytes);
    if (workerFailure) throw workerFailure;
    if (job.paused || stopped || pageLeaving) throw new Error(pageLeaving
      ? "页面离开导致上传中断；重新打开后会先核对进度。"
      : "文件上传已暂停。继续前会先核对进度。");
    return done;
  }
  async function completeUpload(job, initialState = null, attachmentId = "") {
    let state = initialState || await uploadStateWithRetry(job);
    if (state.status === "ready") return waitReady(job, state, attachmentId || state.attachment_id || "");
    if (["completing", "verifying", "processing"].includes(state.status))
      return waitReady(job, state, attachmentId || state.attachment_id || "");
    if (job.completeUnknown && state.status !== "uploading") throw new Error("完成状态仍未明确；未重复提交 complete。");
    job.completeUnknown = true; persist(job);
    let result;
    try {
      result = await request("/uploads/" + id(job.uploadId) + "/complete", {
        method: "POST", key: job.uploadId + ":complete", body: {}, job
      });
    } catch (error) {
      if ([401, 403, 409, 413, 429].includes(error.status) || !temporaryNetworkFailure(error)) throw error;
      try {
        state = await uploadStateWithRetry(job);
        if (state.status === "ready" || ["completing", "verifying", "processing"].includes(state.status))
          return waitReady(job, state, attachmentId || state.attachment_id || "");
      } catch (_) { /* Preserve the uncertain marker for an explicit retry. */ }
      throw error;
    }
    if (result && result.attachment_id) attachmentId = result.attachment_id;
    if (verifiedReady(result) || result && (result.status === "ready" ||
        ["completing", "verifying", "processing"].includes(result.status)))
      return waitReady(job, result, attachmentId || result.attachment_id || "");
    state = await uploadStateWithRetry(job);
    if (state.status === "ready" || ["completing", "verifying", "processing"].includes(state.status))
      return waitReady(job, state, attachmentId || state.attachment_id || "");
    throw new Error("complete 后服务端尚未确认归档；已查询一次状态，没有盲目重试。");
  }
  async function uploadOne(job) {
    if (!(job.file.size >= 1 && job.file.size <= MAX_BYTES)) throw new Error("单个文件必须为 1..512 MiB。");
    let state = await startOrRestore(job);
    if (job.status === "succeeded") return;
    if (!state || typeof state !== "object") return;
    if (["failed", "deleted", "aborted", "expired", "invalid"].includes(state.status))
      throw new Error("服务端上传状态不可继续：" + state.status);
    const done = await uploadParts(job, state), partBytes = Number(state.part_bytes || PART_BYTES);
    const count = Math.ceil(job.file.size / partBytes);
    if (done.size !== count) throw new Error("本地分片记录不完整；请继续重试上传。");
    job.progress.value = 100;
    job.status = "verifying";
    job.statusText = "正在确认上传结果";
    updateJob(job, job.statusText + " · 100% · 正在保存文件");
    await completeUpload(job, state, state.attachment_id || "");
  }
  async function processJob(job) {
    if (job.running || !writable()) return;
    job.running = true; job.paused = false; job.pauseReason = "";
    job.status = "running"; updateJob(job, "处理中。"); controls();
    try {
      await uploadOne(job);
      if (job.status !== "succeeded" && job.status !== "waiting") {
        job.status = "waiting"; updateJob(job, "等待恢复；点“重试 / 继续”会先核对上传进度。");
      }
    } catch (error) {
      if (job.paused) { job.status = "paused"; updateJob(job, job.pauseReason || "已暂停；继续前会先核对进度。"); }
      else if (!stopped && (error && error.recoveryRetryable || temporaryNetworkFailure(error))) {
        job.status = "waiting";
        updateJob(job, (error.message || "网络暂时不可用。") + " 已保留上传进度，点“重试 / 继续”恢复。");
      }
      else if (!stopped) {
        job.status = "failed";
        const nextStep = error.status === 413
          ? "请选择符合页面大小限制的文件后重新上传。"
          : [401, 403].includes(error.status)
            ? "请从当前 PR Summary 重新打开上传页；仍无权限时联系 PR 负责人。"
            : error.status === 409
              ? "请刷新 PR Summary，确认最新测试轮次后再上传。"
              : "请点“重试 / 继续”再试一次；如果链接失效，请从 PR Summary 重新打开上传页。";
        updateJob(job, "上传失败：" + (error.message || String(error)) + " " + nextStep);
      }
    } finally { job.running = false; updateJob(job); controls(); }
  }
  async function runSingle(job) {
    if (busy || stopped) return;
    busy = true; controls(); await processJob(job); busy = false; controls();
  }
  async function runUpload() {
    if (busy || !writable()) return;
    const note = $("note").value;
    const files = Array.from($("file").files || []);
    if ([...note].length > MAX_NOTE_CHARS) { message("文件说明最多 5000 字，请缩短后重试。"); return; }
    const validationError = validateSelectedFiles(files);
    if (validationError) { message(validationError); return; }
    $("file").value = "";
    for (const file of files) makeJob(file, "", note);
    $("note").value = "";
    busy = true; controls();
    for (const job of jobs) {
      if (job.status !== "排队中" || stopped) continue;
      await processJob(job);
    }
    busy = false; controls();
  }
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
  function refreshFailure(error) {
    contextAuthorized = false; controls();
    const status = error && Number(error.status || 0);
    const text = status === 401 ? "授权已失效，请从可信 PR Summary 重新打开附件页。"
      : status === 403 ? "当前账号无权读取或修改附件；写入操作已禁用。"
      : status === 409 ? "PR 或轮次状态已变化，请重试以获取可信数据。"
      : status === 503 || status === 0 ? "服务暂不可用或网络超时；已保留已确认内容。"
      : "当前数据暂不可用；已保留已确认内容。";
    message("");
    const label = document.createTextNode(text + " ");
    const retry = document.createElement("button"); retry.type = "button"; retry.textContent = "重试";
    retry.addEventListener("click", () => refresh().catch(refreshFailure));
    $("message").append(label, retry);
  }
  restore(); refresh().catch(refreshFailure);
})();
'''
