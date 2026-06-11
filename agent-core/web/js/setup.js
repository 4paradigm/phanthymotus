/**
 * setup.js — Three-step setup wizard.
 * Calls onComplete() when config is saved.
 */

// Returns true if `available` format satisfies `required` format.
// Supports trailing wildcard: "audio/*" matches "audio/pcm-16k", "audio/opus", etc.
function _fmtMatch(required, available) {
  if (!required || !available) return false;
  if (required === available) return true;
  if (required.endsWith('/*')) {
    const prefix = required.slice(0, -1); // "audio/"
    return available.startsWith(prefix);
  }
  return false;
}
let _mcpList = [];  // { id, name, transport, url, render_hint }
let _currentStep = 0;
let _onComplete = null;
let _lastPingData = null;  // tools/resources/server_name from last successful ping

const _rand7 = () => Math.random().toString(36).slice(2, 9);

export function initSetup(onComplete) {
  _onComplete = onComplete;
  _currentStep = 0;
  _mcpList = [];
  _lastPingData = null;

  // Load existing config to pre-fill fields
  loadCurrentConfig();

  // Wire navigation
  document.getElementById('btn-next').onclick = onNext;
  document.getElementById('btn-prev').onclick = onPrev;

  // Reset button state (may be stale from previous session)
  goToStep(0);

  // Step tabs click
  document.querySelectorAll('.step-tab').forEach(btn => {
    btn.onclick = () => goToStep(parseInt(btn.dataset.step));
  });

  // MCP
  document.getElementById('btn-add-mcp').onclick    = openMcpDialog;
  document.getElementById('mcp-dialog-cancel').onclick  = closeMcpDialog;
  document.getElementById('mcp-dialog-ping').onclick    = pingMcp;
  document.getElementById('mcp-dialog-confirm').onclick = confirmAddMcp;

  // Service tests
  document.getElementById('btn-test-llm').onclick = () => testService('llm');
  document.getElementById('btn-test-tts').onclick = () => testService('tts');
  document.getElementById('btn-test-asr').onclick = () => testService('asr');

  // ASR provider switching
  document.getElementById('asr-provider').onchange = _onAsrProviderChange;
}

// ── Config loading ──────────────────────────────────────────────────────────

async function loadCurrentConfig() {
  try {
    const res = await fetch('/api/config');
    const json = await res.json();
    const { services = {}, mcp_list = [] } = json.data || {};

    document.getElementById('llm-url').value   = services.llm?.url   || '';
    document.getElementById('llm-key').value   = (services.llm?.key && services.llm.key !== '****') ? services.llm.key : '';
    document.getElementById('llm-model').value = services.llm?.model || '';
    document.getElementById('tts-url').value   = services.tts?.url   || '';

    // ASR
    const asr = services.asr || {};
    const provider = asr.provider || '';
    document.getElementById('asr-provider').value   = provider;
    document.getElementById('asr-url').value         = asr.url        || '';
    document.getElementById('asr-key').value         = (asr.key && asr.key !== '****') ? asr.key : '';
    document.getElementById('asr-model').value       = asr.model      || '';
    document.getElementById('asr-language').value    = asr.language   || 'zh-CN';
    _onAsrProviderChange();

    // Use /api/mcp for the live list — it includes topic_in/topic_out, online status, etc.
    // Fall back to mcp_list from config for MCPs not yet returned by the live endpoint.
    try {
      const mcpRes = await fetch('/api/mcp');
      const mcpJson = await mcpRes.json();
      const liveMcps = mcpJson.data || [];
      console.log('[setup] loadCurrentConfig liveMcps:', liveMcps.map(m => ({ id: m.id, name: m.name, topic_in: m.topic_in, topic_out: m.topic_out })));
      _mcpList = liveMcps.map(m => ({ depends_on: '', ...m }));
    } catch {
      _mcpList = mcp_list.map(m => ({ ...m }));
    }
    renderMcpTable();
  } catch (e) {
    // ignore — first run with no config
  }
}

// ── ASR provider switching ────────────────────────────────────────────────────

function _onAsrProviderChange() {
  const provider = document.getElementById('asr-provider').value;
  document.getElementById('asr-fields-openai').style.display = provider === 'openai' ? '' : 'none';
  document.getElementById('asr-fields-common').style.display = provider             ? '' : 'none';
}

// ── Step navigation ──────────────────────────────────────────────────────────

function goToStep(step) {
  _currentStep = step;

  document.querySelectorAll('.step-panel').forEach(p =>
    p.classList.toggle('active', parseInt(p.dataset.step) === step));
  document.querySelectorAll('.step-tab').forEach(t =>
    t.classList.toggle('active', parseInt(t.dataset.step) === step));
  document.querySelectorAll('.step-dots .dot').forEach(d =>
    d.classList.toggle('active', parseInt(d.dataset.step) === step));

  document.getElementById('btn-prev').disabled = step === 0;

  const isLast = step === 2;
  const btn = document.getElementById('btn-next');
  btn.disabled = false;
  btn.textContent = isLast ? '保存并启动' : '下一步';

  if (isLast) renderConfirmSummary();
}

function onNext() {
  if (_currentStep === 1) {
    const missing = _mcpList.filter(m => (m.topic_in || []).length > 0 && !m.depends_on);
    if (missing.length) {
      alert(`请为以下 MCP 选择依赖总线：${missing.map(m => m.name).join('、')}`);
      return;
    }
    const mismatched = _mcpList.filter(m => {
      if (!(m.topic_in || []).length || !m.depends_on) return false;
      const reqFormat = m.topic_in[0].format || '';
      const upstream = _mcpList.find(c => c.id === m.depends_on);
      if (!upstream) return true;
      return !(upstream.topic_out || []).some(t => _fmtMatch(reqFormat, t.format));
    });
    if (mismatched.length) {
      alert(`以下 MCP 的依赖总线数据类型不匹配，请重新选择：${mismatched.map(m => m.name).join('、')}`);
      return;
    }
  }
  if (_currentStep < 2) {
    goToStep(_currentStep + 1);
  } else {
    saveConfig();
  }
}

function onPrev() {
  if (_currentStep > 0) goToStep(_currentStep - 1);
}

// ── Confirm summary ──────────────────────────────────────────────────────────

function renderConfirmSummary() {
  const llmUrl   = document.getElementById('llm-url').value;
  const llmModel = document.getElementById('llm-model').value;
  const ttsUrl   = document.getElementById('tts-url').value;
  const asrProvider = document.getElementById('asr-provider').value;

  const row = (lbl, val) =>
    `<div class="confirm-row"><span class="lbl">${lbl}</span><span class="val">${val || '—'}</span></div>`;

  let html = '<h4>服务配置</h4>';
  html += row('LLM URL',  llmUrl);
  html += row('LLM 模型', llmModel);
  html += row('TTS',      ttsUrl || '跳过');

  if (!asrProvider) {
    html += row('ASR', '跳过');
  } else if (asrProvider === 'openai') {
    html += row('ASR', `OpenAI兼容  ${document.getElementById('asr-url').value || ''}`);
  } else if (asrProvider === 'openai_omni') {
    html += row('ASR', `OpenAI Omni  ${document.getElementById('asr-omni-url')?.value || document.getElementById('asr-url').value || ''}`);
  }

  if (_mcpList.length) {
    html += '<h4 style="margin-top:16px">MCP 硬件数据总线</h4>';
    _mcpList.forEach(m => {
      const dep = m.depends_on ? _mcpList.find(x => x.id === m.depends_on) : null;
      html += row(m.name, `${m.transport}  ${m.url}${dep ? '  ←  ' + dep.name : ''}`);
    });
  }

  document.getElementById('confirm-summary').innerHTML = html;
}

// ── Save config ───────────────────────────────────────────────────────────────

async function saveConfig() {
  const btn = document.getElementById('btn-next');
  btn.disabled = true;
  btn.textContent = '保存中…';

  const provider = document.getElementById('asr-provider').value;
  const payload = {
    services: {
      llm: {
        url:   document.getElementById('llm-url').value,
        key:   document.getElementById('llm-key').value,
        model: document.getElementById('llm-model').value,
      },
      tts: { url: document.getElementById('tts-url').value },
      asr: {
        provider,
        url:        provider === 'openai_omni'
                      ? (document.getElementById('asr-omni-url')?.value || document.getElementById('asr-url').value)
                      : document.getElementById('asr-url').value,
        key:        provider === 'openai_omni'
                      ? (document.getElementById('asr-omni-key')?.value || document.getElementById('asr-key').value)
                      : document.getElementById('asr-key').value,
        model:      provider === 'openai_omni'
                      ? (document.getElementById('asr-omni-model')?.value || document.getElementById('asr-model').value)
                      : document.getElementById('asr-model').value,
        language:   document.getElementById('asr-language').value || 'zh-CN',
      },
    },
    mcp_list: _mcpList,
  };

  try {
    const res = await fetch('/api/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const json = await res.json();
    if (json.code === 200) {
      _onComplete && _onComplete();
      return;
    }
    alert('保存失败：' + (json.message || '未知错误'));
  } catch (e) {
    alert('保存失败：' + e.message);
  }

  btn.disabled = false;
  btn.textContent = '保存并启动';
}

// ── Service tests ─────────────────────────────────────────────────────────────

async function testService(type) {
  const btn      = document.getElementById(`btn-test-${type}`);
  const resultEl = document.getElementById(`test-result-${type}`);

  btn.disabled    = true;
  btn.textContent = '测试中…';
  resultEl.className   = 'service-test-result';
  resultEl.textContent = '请稍候…';

  const payload = { type };
  if (type === 'llm') {
    payload.url   = document.getElementById('llm-url').value;
    payload.key   = document.getElementById('llm-key').value;
    payload.model = document.getElementById('llm-model').value;
  } else if (type === 'tts') {
    payload.url = document.getElementById('tts-url').value;
  } else {  // asr
    const provider = document.getElementById('asr-provider').value || 'openai';
    payload.provider  = provider;
    payload.url       = provider === 'openai_omni'
                          ? (document.getElementById('asr-omni-url')?.value || document.getElementById('asr-url').value)
                          : document.getElementById('asr-url').value;
    payload.key       = provider === 'openai_omni'
                          ? (document.getElementById('asr-omni-key')?.value || document.getElementById('asr-key').value)
                          : document.getElementById('asr-key').value;
  }

  try {
    const res  = await fetch('/api/config/test', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const json = await res.json();
    const { ok, info } = json.data || {};
    resultEl.className   = `service-test-result ${ok ? 'ok' : 'err'}`;
    resultEl.textContent = ok
      ? `✓ 成功${info ? '  ' + info : ''}`
      : `❌ 失败${info ? ': ' + info : ''}`;
  } catch (e) {
    resultEl.className   = 'service-test-result err';
    resultEl.textContent = '❌ ' + e.message;
  }

  btn.disabled    = false;
  btn.textContent = '测试';
}

// ── MCP table ─────────────────────────────────────────────────────────────────

function renderMcpTable() {
  const tbody = document.getElementById('mcp-table-body');
  console.log('[setup] renderMcpTable, _mcpList:', _mcpList.map(m => ({ id: m.id, name: m.name, topic_in: m.topic_in, topic_out: m.topic_out, depends_on: m.depends_on })));
  if (!_mcpList.length) {
    tbody.innerHTML = '<tr class="empty-row"><td colspan="6">暂无硬件数据总线，点击「添加总线」注册第一个 MCP 设备</td></tr>';
    return;
  }
  tbody.innerHTML = _mcpList.map((m, i) => {
    const tools = m.tools || [];
    const hasDetail = m.server_name || tools.length;
    const detailHtml = hasDetail ? `
      <tr class="mcp-detail-row hidden" id="mcp-detail-${i}">
        <td colspan="6">
          <div class="mcp-detail">
            ${m.server_name ? `<div class="mcp-detail-server">${m.server_name}</div>` : ''}
            ${tools.length ? `<div class="mcp-tools-list">${tools.map(t => {
              const name = typeof t === 'string' ? t : (t.name || '');
              const desc = typeof t === 'object' ? (t.description || '') : '';
              return `<div class="tool-item"><span class="tool-name">${name}</span>${desc ? `<span class="tool-desc">${desc}</span>` : ''}</div>`;
            }).join('')}</div>` : ''}
          </div>
        </td>
      </tr>` : '';

    // Dependency column
    const topicIn = m.topic_in || [];
    let depCell;
    if (topicIn.length > 0) {
      const reqFormat = topicIn[0].format || '';
      const candidates = _mcpList.filter((c, ci) => {
        if (ci === i) return false;
        return (c.topic_out || []).some(t => _fmtMatch(reqFormat, t.format));
      });
      const options = candidates.map(c =>
        `<option value="${c.id}" ${m.depends_on === c.id ? 'selected' : ''}>${c.name}</option>`
      ).join('');

      // Validate existing depends_on: must exist and format must match
      let depError = '';
      if (m.depends_on) {
        const upstream = _mcpList.find(c => c.id === m.depends_on);
        if (!upstream) {
          depError = `上游 MCP 已不存在`;
        } else {
          const fmtMatch = (upstream.topic_out || []).some(t => _fmtMatch(reqFormat, t.format));
          if (!fmtMatch) {
            const upFmts = (upstream.topic_out || []).map(t => t.format).join(', ') || '未知';
            depError = `格式不匹配：需要 ${reqFormat}，上游输出 ${upFmts}`;
          }
        }
      }

      depCell = `<select class="dep-select${depError ? ' dep-error' : ''}" data-dep="${i}">
        <option value="">— 请选择 —</option>
        ${options}
      </select>${depError ? `<div class="dep-error-msg">${depError}</div>` : ''}`;
    } else {
      depCell = `<span style="color:var(--text-muted,#666)">—</span>`;
    }

    return `
      <tr>
        <td>${m.name}</td>
        <td>${m.transport}</td>
        <td style="font-family:var(--font-mono);font-size:11px">${m.url || '—'}</td>
        <td>${m.render_hint || '自动'}</td>
        <td>${depCell}</td>
        <td>
          ${hasDetail ? `<button class="btn-icon" data-expand="${i}" title="详情">▶</button>` : ''}
          <button class="btn-icon" data-refresh="${i}" title="重新 ping">↻</button>
          <button class="btn-icon" data-remove="${i}" title="删除">✕</button>
        </td>
      </tr>
      ${detailHtml}`;
  }).join('');

  tbody.querySelectorAll('[data-refresh]').forEach(btn => {
    btn.onclick = async () => {
      const idx = parseInt(btn.dataset.refresh);
      const m = _mcpList[idx];
      if (m.transport !== 'http' || !m.url) return;
      btn.textContent = '…';
      btn.disabled = true;
      try {
        let pingData = null;
        if (m.id && !m.id.startsWith('tmp-')) {
          // Real id: use existing ping endpoint
          const res  = await fetch(`/api/mcp/${m.id}/ping`, { method: 'POST' });
          const json = await res.json();
          if (json.data?.online) pingData = json.data;
        } else {
          // Tmp id: add temp MCP, ping, delete
          const addRes = await fetch('/api/mcp', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: '__refresh__', transport: m.transport, url: m.url }),
          });
          const tempId = (await addRes.json()).data?.id;
          if (tempId) {
            const res  = await fetch(`/api/mcp/${tempId}/ping`, { method: 'POST' });
            const json = await res.json();
            await fetch(`/api/mcp/${tempId}`, { method: 'DELETE' });
            if (json.data?.online) pingData = json.data;
          }
        }
        if (pingData) {
          _mcpList[idx] = { ..._mcpList[idx],
            topic_in:    pingData.topic_in    || [],
            topic_out:   pingData.topic_out   || [],
            tools:       pingData.tools       || [],
            render_hint: pingData.render_hint || m.render_hint,
          };
          console.log('[setup] refresh', m.name, 'topic_out:', _mcpList[idx].topic_out);
        }
      } catch (e) {
        console.warn('[setup] refresh ping failed:', e);
      }
      renderMcpTable();
    };
  });

  tbody.querySelectorAll('[data-remove]').forEach(btn => {
    btn.onclick = async (e) => {
      e.stopPropagation();
      const idx = parseInt(btn.dataset.remove);
      const m = _mcpList[idx];
      console.log('[setup] delete clicked idx=', idx, 'id=', m?.id, 'name=', m?.name);
      const confirmed = confirm(`确认删除「${m.name}」？`);
      console.log('[setup] confirm result=', confirmed);
      if (!confirmed) return;
      if (m.id) {
        try {
          console.log('[setup] DELETE /api/mcp/' + m.id);
          const res = await fetch(`/api/mcp/${m.id}`, { method: 'DELETE' });
          console.log('[setup] DELETE response status=', res.status);
        } catch (e) {
          console.warn('[setup] delete mcp failed:', e);
        }
      } else {
        console.warn('[setup] no id, skipping backend delete');
      }
      _mcpList.splice(idx, 1);
      renderMcpTable();
    };
  });

  tbody.querySelectorAll('[data-expand]').forEach(btn => {
    btn.onclick = () => {
      const row = document.getElementById(`mcp-detail-${btn.dataset.expand}`);
      if (!row) return;
      const hidden = row.classList.toggle('hidden');
      btn.textContent = hidden ? '▶' : '▼';
    };
  });

  tbody.querySelectorAll('[data-dep]').forEach(sel => {
    sel.onchange = () => {
      const idx = parseInt(sel.dataset.dep);
      const depId = sel.value;
      _mcpList[idx].depends_on = depId;
      // Propagate upstream topic_out[0].topic → downstream topic_in[0].topic
      if (depId) {
        const upstream = _mcpList.find(c => c.id === depId);
        const upTopic  = (upstream?.topic_out || [])[0]?.topic || '';
        if (_mcpList[idx].topic_in?.length > 0) {
          _mcpList[idx].topic_in[0].topic = upTopic;
        }
      } else {
        if (_mcpList[idx].topic_in?.length > 0) {
          _mcpList[idx].topic_in[0].topic = '';
        }
      }
    };
  });
}

// ── MCP dialog ────────────────────────────────────────────────────────────────

function openMcpDialog() {
  document.getElementById('mcp-name').value = '';
  document.getElementById('mcp-transport').value = 'http';
  document.getElementById('mcp-url').value = '';
  document.getElementById('mcp-render').value = '';
  document.getElementById('mcp-dep-hint').textContent = '';
  _lastPingData = null;
  // Pre-fill dep select with all existing MCPs
  _refreshDepSelect('', '');
  hidePingResult();
  document.getElementById('mcp-dialog-overlay').classList.remove('hidden');
  document.getElementById('mcp-url').focus();
}

function _refreshDepSelect(reqFormat, selectedId) {
  const depSelect = document.getElementById('mcp-dep');
  const depHint   = document.getElementById('mcp-dep-hint');
  // All MCPs as candidates; after ping filter to format-compatible ones
  const candidates = reqFormat
    ? _mcpList.filter(c => (c.topic_out || []).some(t => _fmtMatch(reqFormat, t.format)))
    : _mcpList;
  depSelect.innerHTML = '<option value="">— 无依赖 —</option>' +
    candidates.map(c => `<option value="${c.id}" ${selectedId === c.id ? 'selected' : ''}>${c.name}</option>`).join('');
  if (reqFormat) {
    depHint.textContent = `需要 ${reqFormat} 输入${candidates.length ? '' : '（当前无格式匹配的上游总线）'}`;
  } else {
    depHint.textContent = _mcpList.length ? '（可选）选择上游数据总线' : '（暂无已注册总线）';
  }
}

function closeMcpDialog() {
  document.getElementById('mcp-dialog-overlay').classList.add('hidden');
}

function hidePingResult() {
  const el = document.getElementById('ping-result');
  el.classList.add('hidden');
  el.className = 'ping-result hidden';
}

async function pingMcp() {
  const url       = document.getElementById('mcp-url').value;
  const transport = document.getElementById('mcp-transport').value;

  if (!url || transport !== 'http') {
    showPingResult('stdio 协议暂不支持从界面 ping，请手动验证。', 'ok');
    return;
  }

  // Temporarily add to config and use the ping endpoint
  const pingBtn = document.getElementById('mcp-dialog-ping');
  pingBtn.disabled = true;
  pingBtn.textContent = '测试中…';

  try {
    // Add a temp MCP for ping
    const addRes = await fetch('/api/mcp', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: '__ping_test__', transport, url }),
    });
    const addJson = await addRes.json();
    const tempId  = addJson.data?.id;

    if (!tempId) throw new Error('添加临时 MCP 失败');

    const pingRes  = await fetch(`/api/mcp/${tempId}/ping`, { method: 'POST' });
    const pingJson = await pingRes.json();
    console.log('[setup] ping raw response:', JSON.stringify(pingJson, null, 2));

    // Clean up
    await fetch(`/api/mcp/${tempId}`, { method: 'DELETE' });

    const { online, tools = [], resources = [], render_hint = '', server_name = '', error,
            topic_in = [], topic_out = [] } = pingJson.data || {};
    console.log('[setup] ping extracted: online=%o topic_in=%o topic_out=%o render_hint=%o',
      online, topic_in, topic_out, render_hint);
    if (!online) {
      showPingResult(`❌ 连接失败${error ? ': ' + error : ''}`, 'err');
    } else {
      _lastPingData = { tools, resources, server_name, render_hint, topic_in, topic_out };
      console.log('[setup] _lastPingData set:', JSON.stringify(_lastPingData, null, 2));

      // Update dependency selector based on topic_in discovered from ping
      const reqFormat = (topic_in[0] || {}).format || '';
      const curDepId  = document.getElementById('mcp-dep').value;
      _refreshDepSelect(reqFormat, curDepId);
      if (render_hint && !document.getElementById('mcp-render').value) {
        document.getElementById('mcp-render').value = render_hint;
      }
      // Auto-fill name from serverInfo if user hasn't typed one
      if (server_name && !document.getElementById('mcp-name').value.trim()) {
        document.getElementById('mcp-name').value = server_name + '_' + _rand7();
      }
      const lines = [`✓ 在线  ${server_name ? 'name: ' + server_name + '  ' : ''}render_hint: ${render_hint || '未识别'}`];
      if (tools.length)     lines.push(`Tools: ${tools.map(t => typeof t === 'string' ? t : t.name).join(', ')}`);
      if (resources.length) lines.push(`Resources: ${resources.join(', ')}`);
      showPingResult(lines.join('\n'), 'ok');
    }
  } catch (e) {
    showPingResult('❌ ' + e.message, 'err');
  }

  pingBtn.disabled = false;
  pingBtn.textContent = '测试连接';
}

function showPingResult(text, type) {
  const el = document.getElementById('ping-result');
  el.textContent = text;
  el.className = `ping-result ${type}`;
}

function confirmAddMcp() {
  if (!_lastPingData) {
    alert('请先点击「测试连接」，验证设备在线并获取 topic 信息后再添加。');
    return;
  }

  let name      = document.getElementById('mcp-name').value.trim();
  const transport = document.getElementById('mcp-transport').value;
  const url       = document.getElementById('mcp-url').value.trim();
  const render    = document.getElementById('mcp-render').value;

  if (!name) name = 'device_' + _rand7();

  // Dependency: required when topic_in is present (discovered via ping)
  const topicIn = _lastPingData.topic_in || [];
  const depId   = document.getElementById('mcp-dep').value;

  if (topicIn.length > 0 && !depId) {
    alert(`该设备需要 ${topicIn[0].format} 输入，请选择依赖的上游总线。`);
    return;
  }

  // Validate format compatibility
  if (depId) {
    const upstream = _mcpList.find(c => c.id === depId);
    const reqFormat = topicIn[0]?.format || '';
    if (upstream && reqFormat && !(upstream.topic_out || []).some(t => _fmtMatch(reqFormat, t.format))) {
      const upFmts = (upstream.topic_out || []).map(t => t.format).join(', ') || '未知';
      alert(`数据类型不匹配：当前设备需要 ${reqFormat}，所选上游输出 ${upFmts}`);
      return;
    }
  }

  const entry = { id: 'tmp-' + _rand7(), name, transport, url, render_hint: render };
  entry.server_name = _lastPingData.server_name || '';
  entry.tools       = _lastPingData.tools       || [];
  entry.resources   = _lastPingData.resources   || [];
  entry.topic_in    = _lastPingData.topic_in    ? JSON.parse(JSON.stringify(_lastPingData.topic_in)) : [];
  entry.topic_out   = _lastPingData.topic_out   || [];
  entry.depends_on  = depId;
  // Fill in the upstream topic path into topic_in[0].topic
  if (depId && entry.topic_in.length > 0) {
    const upstream = _mcpList.find(c => c.id === depId);
    entry.topic_in[0].topic = (upstream?.topic_out || [])[0]?.topic || '';
  }
  console.log('[setup] confirmAddMcp entry:', JSON.stringify(entry, null, 2));
  _mcpList.push(entry);
  console.log('[setup] _mcpList after add:', _mcpList.map(m => ({ id: m.id, name: m.name, topic_in: m.topic_in, topic_out: m.topic_out, depends_on: m.depends_on })));
  renderMcpTable();
  closeMcpDialog();
}
