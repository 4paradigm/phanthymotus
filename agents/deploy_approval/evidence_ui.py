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
pre,code{{overflow-wrap:anywhere;white-space:pre-wrap}}li{{padding:.6rem 0;border-bottom:1px solid #ddd}}
#message{{white-space:pre-wrap}}[hidden]{{display:none!important}}
</style><script src="/evidence/assets/app.js" defer></script></head>
<body><h1>部署验证附件</h1>
<main id="app" data-api="{esc(api)}" data-repo="{esc(repo)}" data-pr="{esc(pr_number)}" data-head="{esc(head_sha)}" data-round="{esc(round_id)}">
<p>项目 <strong>{esc(repo)}</strong> · PR #{esc(pr_number)}<br>HEAD <code>{esc(head_sha)}</code><br>round <code>{esc(round_id)}</code></p>
<p id="auth">正在检查当前 PR/HEAD/round capability。</p>
<p id="message" role="status" aria-live="polite"></p><button id="refresh" type="button">刷新状态</button>
<section><h2>当前轮</h2><p id="summary">等待服务端状态</p>
<fieldset id="writes" disabled><legend>上传与备注</legend>
<label>提交者<input id="submitter" value="browser tester" maxlength="100"></label>
<label>文件（支持视频和普通文件）<input id="file" type="file"></label>
<label>备注<textarea id="note" maxlength="10000"></textarea></label>
<label>替换当前轮附件<select id="replace"><option value="">新增附件</option></select></label>
<button id="upload" type="button">上传 / 继续</button><button id="pause" type="button" disabled>暂停</button>
<button id="retry" type="button" disabled>重试</button><button id="resubmit" type="button" disabled>重新提交完成请求</button>
<label>文本证据<textarea id="text" maxlength="50000"></textarea></label>
<button id="save-note" type="button">提交文本与备注</button>
</fieldset><progress id="progress" value="0" max="100" aria-label="上传进度"></progress><p id="progress-label">尚未上传</p>
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
  const active = new Set();
  let context = null, stopped = false, paused = true, busy = false;
  let session = null, chosen = null, fingerprint = "", completeUnknown = false, pollGeneration = 0;
  let noteOperation = null;
  const message = text => { $("message").textContent = String(text); };
  const id = value => {
    if (typeof value !== "string" || !/^[A-Za-z0-9_-]+$/.test(value)) throw new Error("服务端返回了无效标识符");
    return encodeURIComponent(value);
  };
  const uid = () => crypto.randomUUID();
  const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
  const roundMatches = value => value && value.head_sha === bound.head_sha && value.round_id === bound.round_id;
  function writable() {
    return context && roundMatches(context) && context.enabled !== false && context.evidence_enabled !== false &&
      context.authenticated === true && context.can_write === true && context.status === "testing" &&
      (context.pr_state === undefined || context.pr_state === "open") && !context.merged && !stopped;
  }
  function controls() {
    $("writes").disabled = !writable();
    $("upload").disabled = busy;
    $("file").disabled = busy;
    $("note").disabled = busy;
    $("replace").disabled = busy;
    $("save-note").disabled = busy;
    $("pause").disabled = !busy;
    $("retry").disabled = busy || !session;
    $("resubmit").disabled = busy || !session || !completeUnknown;
    for (const button of document.querySelectorAll("button[data-delete]")) button.disabled = busy || !writable();
  }
  function halt(text) {
    stopped = true; paused = true;
    for (const request of active) request.abort();
    controls(); message(text);
  }
  class ApiError extends Error {
    constructor(status, text) { super(text); this.status = status; }
  }
  async function request(path, {method = "GET", body, key} = {}) {
    const write = method !== "GET";
    if (write && !writable()) throw new Error("当前轮不可写，请刷新并核对 HEAD/round。");
    const controller = new AbortController(); active.add(controller);
    const timeout = setTimeout(() => controller.abort(), 30000);
    try {
      const headers = {Accept: "application/json"};
      if (write) {
        headers["Content-Type"] = "application/json";
        if (!context.csrf_token) throw new Error("缺少 CSRF token，请从 Summary capability 链接重新打开。");
        headers["X-CSRF-Token"] = context.csrf_token;
        headers["X-Evidence-Submitter"] = $("submitter").value || "browser tester";
      }
      const response = await fetch(api + path, {
        method, headers, credentials: "same-origin", redirect: "error", cache: "no-store", signal: controller.signal,
        ...(write ? {body: JSON.stringify({...body, ...bound, idempotency_key: key || uid()})} : {})
      });
      let value;
      try { value = await response.json(); } catch (_) { value = {}; }
      if (!response.ok) {
        const detail = typeof value.detail === "string" ? value.detail : typeof value.error === "string" ? value.error : "";
        if (response.status === 401) {
          halt("Capability 不可用，请从 PR Summary 的 Evidence 链接重新打开页面。");
          $("auth").textContent = "Capability 不可用，请从 PR Summary 的 Evidence 链接重新打开页面。";
        }
        if (response.status === 409) halt("HEAD/round 或上传状态已变化，已停止所有写入。请刷新当前轮。 " + detail);
        if (response.status === 413) throw new ApiError(413, "超过服务端大小限制。" + detail);
        throw new ApiError(response.status, `${response.status}: ${detail || response.statusText}`);
      }
      return value;
    } finally { clearTimeout(timeout); active.delete(controller); }
  }
  function anchor(label, path) {
    const a = document.createElement("a"); a.textContent = label; a.href = api + path;
    a.setAttribute("download", ""); a.rel = "noopener"; return a;
  }
  function attachmentList(rows, parent, current) {
    for (const item of rows || []) {
      const li = document.createElement("li");
      const deleted = item.status === "deleted" || !!item.deleted_at;
      const attachmentId = item.attachment_id || item.id;
      const description = document.createElement("span");
      description.textContent = [item.filename || item.text || "文本证据", item.size === undefined ? "" : `${item.size} B`,
        item.actor || item.submitted_by || "", item.created_at || "", item.note || "",
        deleted ? `已删除 ${item.deleted_at || ""} ${item.deleted_by || ""}` : item.status || "pending",
        item.replaces_id ? `替换 ${item.replaces_id}` : ""].filter(Boolean).join(" · ");
      li.append(description);
      if (!deleted && item.status === "ready" && attachmentId) {
        const capability = item.download_capability ? "?cap=" + encodeURIComponent(item.download_capability) : "";
        li.append(" ", anchor("下载", "/attachments/" + id(attachmentId) + "/download" + capability));
      }
      const same = current && (!item.head_sha || item.head_sha === bound.head_sha) &&
        (!item.round_id || item.round_id === bound.round_id);
      if (!deleted && same && writable() && attachmentId) {
        const remove = document.createElement("button"); remove.type = "button"; remove.textContent = "删除";
        remove.dataset.delete = attachmentId;
        remove.addEventListener("click", () => deleteAttachment(item, remove)); li.append(" ", remove);
        if (item.status === "ready") {
          const option = document.createElement("option"); option.value = attachmentId;
          option.textContent = item.filename || attachmentId; $("replace").append(option);
        }
      }
      parent.append(li);
    }
  }
  function render() {
    $("auth").textContent = context.authenticated ? (writable() ? "Capability 有效，可以向当前测试轮提交证据。" : "当前轮只读或不处于 testing。") : "Capability 无效。请从当前 PR Summary 重新打开。";
    $("summary").textContent = typeof context.summary === "string" ? context.summary : JSON.stringify(context.summary || {status: context.status});
    $("attachments").replaceChildren(); $("history").replaceChildren();
    const replacement = $("replace").value;
    $("replace").replaceChildren(new Option("新增附件", ""));
    attachmentList(context.attachments || [], $("attachments"), true);
    for (const round of context.rounds || context.history || []) {
      if (roundMatches(round)) continue;
      const details = document.createElement("details"), summary = document.createElement("summary");
      summary.textContent = `HEAD ${round.head_sha || "unknown"} · round ${round.round_id || "unknown"} · ${round.status || "unknown"}`;
      details.append(summary);
      const ul = document.createElement("ul"); attachmentList(round.attachments || [], ul, false); details.append(ul);
      if (round.round_id && (round.cos_object_key || round.cos?.object_key || round.log_available || round.log_ready) && context.enabled !== false && context.evidence_enabled !== false) details.append(anchor("下载本轮日志", "/rounds/" + id(round.round_id) + "/log/download?cap=" + encodeURIComponent(round.capability || "")));
      $("history").append(details);
    }
    for (const archive of context.archive_links || []) {
      // This is the GitHub comment discovered by the authenticated service,
      // restricted again here to this repository and PR.
      try {
        const url = new URL(archive.url);
        const prefix = "/" + app.dataset.repo;
        if (url.origin !== "https://github.com" || url.username || url.password || url.search ||
            ![prefix + "/issues/" + app.dataset.pr, prefix + "/pull/" + app.dataset.pr].includes(url.pathname) ||
            !/^#issuecomment-[0-9]+$/.test(url.hash)) continue;
        const a = document.createElement("a"), p = document.createElement("p");
        a.textContent = "历史索引 · " + (archive.round_id || ""); a.href = url.href; a.rel = "noopener";
        p.append(a); $("history").append(p);
      } catch (_) { /* Do not use arbitrary archive redirect destinations. */ }
    }
    if ([...$("replace").options].some(o => o.value === replacement)) $("replace").value = replacement;
    controls();
  }
  async function refresh() {
    const next = await request("/context");
    context = next;
    if (!roundMatches(next)) {
      halt("当前 HEAD/round 已变化。本页面绑定的历史轮次只读，请从新的生命周期评论进入当前轮。");
    }
    render(); return next;
  }
  function persist() {
    if (!session) return;
    try { localStorage.setItem(storageKey, JSON.stringify({upload_id: session.upload_id, fingerprint, parts: session.parts || [], complete_unknown: completeUnknown})); }
    catch (_) { message("浏览器无法保存恢复信息；当前上传仍可继续。"); }
  }
  function restore() {
    try {
      const saved = JSON.parse(localStorage.getItem(storageKey) || "null");
      if (saved && typeof saved.fingerprint === "string" && saved.fingerprint.length === 64) {
        id(saved.upload_id); session = {upload_id: saved.upload_id, parts: []}; fingerprint = saved.fingerprint;
        completeUnknown = saved.complete_unknown === true;
        message("发现未清除的上传记录。请重新选择相同文件，继续时将先核对文件指纹和服务端分块状态。");
      }
    } catch (_) { /* Invalid local hints never become server authority. */ }
  }
  async function fileFingerprint(file) {
    // Hash every byte using bounded WebCrypto calls. This is a file identity,
    // not a claim to be the object's SHA-256 checksum (the server verifies it).
    let chain = new Uint8Array(32);
    for (let start = 0; start < file.size; start += 4 * 1024 * 1024) {
      if (paused || stopped) throw new Error("已暂停文件核对。");
      const chunk = await crypto.subtle.digest("SHA-256", await file.slice(start, start + 4 * 1024 * 1024).arrayBuffer());
      const both = new Uint8Array(64); both.set(chain); both.set(new Uint8Array(chunk), 32);
      chain = new Uint8Array(await crypto.subtle.digest("SHA-256", both));
      $("progress-label").textContent = `正在核对文件：${Math.min(file.size, start + 4 * 1024 * 1024)} / ${file.size} B`;
    }
    const metadata = new TextEncoder().encode(JSON.stringify([file.name, file.size, file.type, file.lastModified]));
    const last = new Uint8Array(chain.length + metadata.length); last.set(chain); last.set(metadata, chain.length);
    return [...new Uint8Array(await crypto.subtle.digest("SHA-256", last))].map(v => v.toString(16).padStart(2, "0")).join("");
  }
  function partsOf(value) {
    const rows = value.uploaded_parts || value.parts || [];
    if (Array.isArray(rows)) return rows.map(row => typeof row === "number" ? {part_number: row} : row);
    return Object.entries(rows).map(([number, row]) => ({...(typeof row === "object" ? row : {etag: row}), part_number: Number(number)}));
  }
  async function uploadState() {
    const value = await request("/uploads/" + id(session.upload_id));
    if ((value.head_sha && value.head_sha !== bound.head_sha) || (value.round_id && value.round_id !== bound.round_id)) {
      halt("上传属于其他 HEAD/round，已停止。"); throw new Error("轮次不匹配");
    }
    if (value.upload_id && value.upload_id !== session.upload_id) throw new Error("上传标识不匹配");
    session = {...session, ...value, parts: partsOf(value)}; persist(); return session;
  }
  const terminalReady = value => value.status === "ready" && value.verified === true;
  async function resumeIfRequired(state) {
    if (state.resume_required !== true) return state;
    if (!chosen || !fingerprint) throw new Error("恢复上传前请重新选择原文件并完成指纹核对。");
    const resumed = await request("/uploads/" + id(session.upload_id) + "/resume", {
      method: "POST", key: session.upload_id + ":resume", body: {}
    });
    if ((resumed.head_sha && resumed.head_sha !== bound.head_sha) ||
        (resumed.round_id && resumed.round_id !== bound.round_id) ||
        (resumed.upload_id && resumed.upload_id !== session.upload_id)) {
      halt("恢复结果属于其他 HEAD/round 或上传，已停止。"); throw new Error("恢复结果不匹配");
    }
    const checked = await uploadState();
    if (checked.filename && chosen && checked.filename !== chosen.name ||
        checked.size !== undefined && chosen && Number(checked.size) !== chosen.size) {
      throw new Error("恢复后的文件元数据与所选文件不匹配");
    }
    return checked;
  }
  function progress(parts, partBytes, inflight = new Map()) {
    const total = chosen ? chosen.size : 0;
    let bytes = 0;
    for (const part of parts.values()) bytes += Math.min(partBytes, Math.max(0, total - (part.part_number - 1) * partBytes));
    for (const amount of inflight.values()) bytes += amount;
    const percentage = total ? Math.min(100, bytes * 100 / total) : 0;
    $("progress").value = percentage;
    $("progress-label").textContent = `${percentage.toFixed(1)}% · ${bytes} / ${total} B（分块已传输，最终状态以服务端 ready 为准）`;
  }
  function putPart(signed, blob, number, inflight, update) {
    return new Promise((resolve, reject) => {
      const target = new URL(signed.url);
      if (target.protocol !== "https:" || target.username || target.password || target.hash) return reject(new Error("无效的上传地址"));
      if (signed.method && signed.method !== "PUT") return reject(new Error("不支持的上传方法"));
      const xhr = new XMLHttpRequest(); active.add(xhr);
      const cleanup = () => { active.delete(xhr); inflight.delete(number); update(); };
      xhr.open("PUT", target.href); xhr.withCredentials = false; xhr.timeout = 120000;
      // Only storage headers returned by the authenticated API go to COS.
      for (const [name, value] of Object.entries(signed.headers || {})) {
        if (/^(cookie|authorization|x-csrf-token)$/i.test(name)) { cleanup(); return reject(new Error("拒绝向存储地址发送认证头")); }
        xhr.setRequestHeader(name, value);
      }
      xhr.upload.onprogress = event => { inflight.set(number, event.loaded); update(); };
      xhr.onload = () => {
        cleanup();
        if (xhr.status >= 200 && xhr.status < 300 && new URL(xhr.responseURL).href === target.href) {
          resolve({part_number: number, etag: xhr.getResponseHeader("ETag") || "", size: blob.size});
        } else reject(new ApiError(xhr.status, "分块上传失败：" + xhr.status));
      };
      xhr.onerror = () => { cleanup(); reject(new Error("网络中断，可继续上传")); };
      xhr.ontimeout = () => { cleanup(); reject(new Error("分块上传超时，可继续上传")); };
      xhr.onabort = () => { cleanup(); reject(new Error("上传已暂停")); };
      xhr.send(blob);
    });
  }
  async function waitReady(uploadId, attachmentId = session && session.attachment_id, clearUpload = true) {
    const generation = ++pollGeneration;
    message("文件已提交，等待服务端校验及摘要更新。");
    for (let count = 0; count < 30 && generation === pollGeneration && !stopped; count++) {
      const current = await refresh();
      const rows = current.attachments || [];
      const item = rows.find(row => (uploadId && row.upload_id === uploadId) ||
        (attachmentId && (row.attachment_id === attachmentId || row.id === attachmentId)));
      const ready = item && item.status === "ready";
      const summaryReady = current.summary_status === "ready" || current.summary_updated === true ||
        (current.summary && typeof current.summary === "object" && current.summary.status === "ready");
      if (ready && summaryReady) {
        message("服务端已确认附件 ready，摘要已更新。");
        $("progress").value = 100; $("progress-label").textContent = "服务端确认完成";
        if (clearUpload) {
          try { localStorage.removeItem(storageKey); } catch (_) {}
          session = null; fingerprint = ""; completeUnknown = false;
        }
        controls(); return;
      }
      if (item && ["failed", "rejected"].includes(item.status)) throw new Error(item.error || "服务端校验失败");
      await delay(2000);
    }
    message("仍在等待服务端 ready/summary 更新；请刷新状态后继续确认。");
  }
  async function complete(explicit = false) {
    const state = await uploadState();
    if (terminalReady(state) || ["completing", "verifying", "processing"].includes(state.status)) {
      await waitReady(session.upload_id); return;
    }
    if (state.status === "ready" || state.status === "complete_unknown" || state.status === "completion_unknown") {
      completeUnknown = true; persist(); controls();
      if (!explicit) {
        message("服务端完成状态尚未验证。请核对状态后点“重新提交完成请求”。"); return;
      }
    }
    if (["failed", "deleted", "aborted", "expired"].includes(state.status)) throw new Error(state.error || `上传状态：${state.status}`);
    if (completeUnknown && !explicit) {
      message("上次完成请求结果未知，已查询服务端状态。请核对后点“重新提交完成请求”。"); return;
    }
    completeUnknown = true; persist(); controls();
    const uploadId = session.upload_id;
    try {
      const result = await request("/uploads/" + id(uploadId) + "/complete", {
        method: "POST", key: uploadId + ":complete", body: {}
      });
      session = {...session, ...result};
      // A POST response without a verified ready state remains uncertain.
      completeUnknown = !terminalReady(result); persist(); controls();
      if (["failed", "rejected"].includes(result.status)) throw new Error(result.error || "服务端校验失败");
      await waitReady(uploadId);
    } catch (error) {
      // Never automatically replay an uncertain multipart completion.
      if (!stopped) {
        try { const checked = await uploadState(); if (terminalReady(checked)) { completeUnknown = false; persist(); await waitReady(uploadId); return; } }
        catch (_) { /* Keep uncertainty visible for an explicit user retry. */ }
      }
      throw error;
    }
  }
  async function runUpload() {
    if (busy || !writable()) return;
    chosen = $("file").files[0];
    if (!chosen) { message("请选择文件；恢复上传必须重新选择原文件。"); return; }
    if (!chosen.size) { message("请选择非空文件。"); return; }
    busy = true; paused = false; controls();
    try {
      message("正在核对完整文件指纹，尚未上传成功。");
      const selectedFingerprint = await fileFingerprint(chosen);
      if (session && selectedFingerprint !== fingerprint) throw new Error("文件与原上传不一致，已拒绝拼接。请重新选择原文件。");
      fingerprint = selectedFingerprint;
      if (!session) {
        await refresh();
        const result = await request("/uploads", {method: "POST", key: bound.round_id + ":" + fingerprint + ":" + ($("replace").value || "new") + ":" + uid(),
          body: {filename: chosen.name, size: chosen.size, content_type: chosen.type || "application/octet-stream", note: $("note").value, submitter: $("submitter").value || "browser tester", replaces_id: $("replace").value || null}});
        id(result.upload_id); session = {...result, parts: partsOf(result)}; persist();
      }
      let state = await uploadState();
      state = await resumeIfRequired(state);
      if (terminalReady(state) || ["completing", "verifying", "processing"].includes(state.status) || completeUnknown) { await complete(); return; }
      const partBytes = Number(state.part_bytes);
      if (!Number.isSafeInteger(partBytes) || partBytes <= 0) throw new Error("服务端未提供有效的 part_bytes，无法安全分块。");
      if (state.filename && state.filename !== chosen.name || state.size !== undefined && Number(state.size) !== chosen.size) throw new Error("服务端上传文件元数据不匹配");
      const count = Math.ceil(chosen.size / partBytes);
      const done = new Map();
      for (const part of state.parts) {
        const n = Number(part.part_number);
        if (!Number.isInteger(n) || n < 1 || n > count) throw new Error("服务端分块状态无效");
        if (part.size !== undefined && Number(part.size) !== Math.min(partBytes, chosen.size - (n - 1) * partBytes)) throw new Error("服务端分块大小与所选文件不匹配");
        done.set(n, {...part, part_number: n});
      }
      const inflight = new Map();
      const update = () => progress(done, partBytes, inflight);
      update();
      const queue = Array.from({length: count}, (_, n) => n + 1).filter(n => !done.has(n));
      let cursor = 0;
      async function worker() {
        while (cursor < queue.length && !paused && !stopped) {
          const number = queue[cursor++];
          const blob = chosen.slice((number - 1) * partBytes, Math.min(number * partBytes, chosen.size));
          let finished = false;
          for (let attempt = 0; attempt < 3 && !paused && !stopped; attempt++) {
            try {
              const signed = await request("/uploads/" + id(session.upload_id) + "/parts/" + number + "/url", {
                method: "POST", key: `${session.upload_id}:part:${number}:${attempt}:${uid()}`, body: {}
              });
              const part = await putPart(signed, blob, number, inflight, update);
              done.set(number, part); session.parts = [...done.values()]; persist(); update(); finished = true; break;
            } catch (error) {
              if (paused || stopped || [401, 409, 413].includes(error.status) || attempt === 2) throw error;
              // A successful PUT may have lost its response. Reconcile before
              // requesting a fresh signed URL; expiration never restarts upload.
              const checked = await uploadState();
              const found = checked.parts.find(part => Number(part.part_number) === number);
              if (found) { done.set(number, found); finished = true; update(); break; }
              await delay((attempt + 1) * 500);
            }
          }
          if (!finished && !paused) throw new Error("分块重试次数已用完，请手动继续。");
        }
      }
      const outcomes = await Promise.allSettled([worker(), worker()]);
      const failed = outcomes.find(outcome => outcome.status === "rejected");
      if (failed) throw failed.reason;
      if (paused || stopped) throw new Error("上传已暂停，可继续。");
      session.parts = [...done.values()].sort((a, b) => a.part_number - b.part_number);
      await complete();
    } catch (error) { message(error.message || String(error)); }
    finally { busy = false; paused = true; controls(); }
  }
  async function deleteAttachment(item, button) {
    if (busy || !writable()) return;
    if (!window.confirm(`确认删除当前轮附件“${item.filename || item.attachment_id || item.id}”？删除将保留审计记录，历史轮次不会自动删除。`)) return;
    busy = true; controls();
    // A deterministic operation key survives refresh after an unknown result.
    const key = bound.round_id + ":delete:" + (item.attachment_id || item.id);
    try {
      await request("/attachments/" + id(item.attachment_id || item.id), {method: "DELETE", key, body: {}});
      await refresh(); message("删除结果已从服务端刷新，审计记录保留。");
    } catch (error) { message(error.message); try { await refresh(); } catch (_) {} }
    finally { busy = false; controls(); }
  }
  $("upload").addEventListener("click", runUpload);
  $("retry").addEventListener("click", runUpload);
  $("pause").addEventListener("click", () => {
    paused = true; pollGeneration++;
    for (const request of active) request.abort(); message("已暂停；继续时先查询服务端分块状态。");
  });
  $("resubmit").addEventListener("click", async () => {
    if (busy || !writable()) return;
    busy = true; controls();
    try { await refresh(); await complete(true); } catch (error) { message(error.message); }
    finally { busy = false; controls(); }
  });
  $("save-note").addEventListener("click", async () => {
    if (busy || !writable()) return;
    const body = {text: $("text").value, note: $("note").value, submitter: $("submitter").value || "browser tester"};
    if (!body.text.trim() && !body.note.trim()) { message("请输入文本或备注。"); return; }
    const content = JSON.stringify(body);
    if (!noteOperation || noteOperation.content !== content) noteOperation = {content, key: uid()};
    busy = true; controls();
    try {
      const result = await request("/notes", {method: "POST", key: noteOperation.key, body});
      await refresh();
      const attachmentId = result.attachment_id || result.id;
      if (attachmentId) await waitReady(null, attachmentId, false);
      else message("文本已提交，正在等待摘要更新；请刷新服务端状态确认。");
      noteOperation = null;
    } catch (error) { message(error.message); }
    finally { busy = false; controls(); }
  });
  $("refresh").addEventListener("click", async () => {
    try {
      if (busy) { message("上传进行中，请暂停后刷新。"); return; }
      stopped = false; await refresh();
      // Refresh is read-only. Resume and completion require an explicit button action.
    } catch (error) { message(error.message); }
  });
  restore(); refresh().catch(error => message(error.message));
})();
'''
