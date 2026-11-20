// A beginner-friendly, in-card guide for the existing live_stream MCP tool.
// Account authorization always stays on the platform's own site; no credentials are logged.
import { PLATFORMS, READINESS, getReadiness, getGuide, getGuideHelp, makeStreamUrl, makeXiaohongshuStreamUrl } from './platforms.js';
import { callLiveStream, saveLiveStreamConfig, authorizeStoredLiveStream, startApprovedLiveStream } from './api.js';

const ACTIVE_STATES = new Set(['starting', 'streaming', 'stopping']);
const liveLabel = { starting: '正在连接直播平台', streaming: 'Bumi 正在推流', stopping: '正在停止推流' };

export function mountLiveStreamCard(root, { mcpId, cardId, authorize }) {
  const state = {
    phase: 'platform', platform: 'xiaohongshu', readiness: 'new',
    guideIndex: 0, showHelp: false, rawInput: '', server: '',
    busy: false, error: '', notice: '',
    ready: false, connected: false, remote: null, destroyed: false,
  };
  // Remember non-sensitive guide progress; never persist stream URLs, codes or secrets.
  const progressKey = `bumi:live-guide:v3:${mcpId}:${cardId}`;
  const safePhases = new Set(['platform', 'readiness', 'guide', 'input']);
  try {
    const saved = JSON.parse(localStorage.getItem(progressKey) || 'null');
    if (saved && Object.hasOwn(PLATFORMS, saved.platform)
        && READINESS.some(item => item.id === saved.readiness)
        && safePhases.has(saved.phase)) {
      state.platform = saved.platform;
      state.readiness = saved.readiness;
      state.phase = saved.phase;
      const count = getGuide(state.platform, state.readiness).length;
      state.guideIndex = Math.min(Math.max(0, Number(saved.guideIndex) || 0), Math.max(0, count - 1));
    }
  } catch { /* storage can be unavailable in restricted browsers */ }
  let checking = false;

  function button(text, action, kind = '', disabled = false) {
    return `<button type="button" class="ls-btn ${kind}" data-action="${action}"${disabled ? ' disabled' : ''}>${text}</button>`;
  }
  function back(to) { return button('返回上一步', to, 'ls-back'); }

  function render() {
    if (state.destroyed) return;
    const platform = PLATFORMS[state.platform];
    const guide = getGuide(state.platform, state.readiness);
    let body = '';
    if (state.phase === 'platform') {
      body = `<div class="ls-title">你想在哪里直播？</div>
        <div class="ls-platforms">
          ${Object.entries(PLATFORMS).map(([id, item]) =>
            button(item.name, 'platform:' + id, 'ls-platform' + (id === state.platform ? ' selected' : ''))
          ).join('')}
        </div>
        ${button('下一步', 'to-readiness', 'ls-primary')}`;
    } else if (state.phase === 'readiness') {
      body = `<div class="ls-title">你现在准备到哪一步了？</div>
        <div class="ls-choices">${getReadiness(state.platform).map(item => button(item.title,
          'readiness:' + item.id, 'ls-choice' + (item.id === state.readiness ? ' selected' : '')
        )).join('')}</div>
        ${button('继续', 'start-guide', 'ls-primary')}
        ${back('to-platform')}`;
    } else if (state.phase === 'guide') {
      const current = guide[state.guideIndex];
      if (!current) {
        state.phase = 'input';
        render();
        return;
      }
      const label = `第 ${state.guideIndex + 1} 步，共 ${guide.length} 步`;
      const nextLabel = current.title === '申请 OBS 开播权限' ? '申请成功，继续下载' : '我完成了这一步';
      body = `<div class="ls-progress">${label} · ${current.stage || '直播准备'} · ${current.place || '官方平台'}</div>
        <div class="ls-meter"><div class="ls-meter-fill" style="width:${Math.round(100 * (state.guideIndex + 1) / guide.length)}%"></div></div>
        <div class="ls-title" data-guide-title></div>
        <p class="ls-guide-detail" data-guide-detail></p>
        ${current.url ? '<a class="ls-link" data-guide-url target="_blank" rel="noopener noreferrer">打开本步页面 ↗</a>' : '<p class="ls-hint">请按上面的提示在对应 App 中操作。</p>'}
        ${button(nextLabel, 'guide-next', 'ls-primary')}
        ${button('找不到页面或按钮', 'guide-help', 'ls-secondary')}
        ${state.showHelp ? '<p class="ls-help" data-guide-help></p>' : ''}
        ${state.showHelp && current.helpUrl ? '<a class="ls-link" data-guide-help-url target="_blank" rel="noopener noreferrer">查看官方操作说明 ↗</a>' : ''}
        ${back('guide-back')}`;
    } else if (state.phase === 'input') {
      body = state.platform === 'xiaohongshu'
        ? `<div class="ls-title">填写小红书推流信息</div>
          <label class="ls-label" for="ls-server-${cardId}">服务器地址</label>
          <input id="ls-server-${cardId}" class="ls-input" data-input="server"
            type="text" autocomplete="off" spellcheck="false" placeholder="粘贴服务器地址">
          <label class="ls-label" for="ls-key-${cardId}">直播推流码</label>
          <input id="ls-key-${cardId}" class="ls-input" data-input="secret"
            type="password" autocomplete="off" spellcheck="false" placeholder="粘贴直播推流码">
          ${state.remote?.configured && state.remote?.platform === state.platform ? button('使用已保存的信息继续', 'use-saved', 'ls-secondary') : ''}
          ${button('检查并保存', 'save', 'ls-primary', state.busy)}
          ${back('input-back')}`
        : `<div class="ls-title">填写虎牙推流信息</div>
          <label class="ls-label" for="ls-input-${cardId}">完整推流地址</label>
          <input id="ls-input-${cardId}" class="ls-input" data-input="secret"
            type="password" autocomplete="off" spellcheck="false" placeholder="粘贴完整推流地址">
          ${button('从剪贴板导入', 'paste', 'ls-secondary')}
          ${state.remote?.configured && state.remote?.platform === state.platform ? button('使用已保存的信息继续', 'use-saved', 'ls-secondary') : ''}
          ${button('检查并保存', 'save', 'ls-primary', state.busy)}
          ${back('input-back')}`;
    } else if (state.phase === 'review') {
      body = `<div class="ls-title">即将开始直播</div>
        <div class="ls-check">✓ 直播信息已保存</div>
        ${button('确认开始直播', 'start', 'ls-primary', state.busy)}
        ${back('to-input')}`;
    } else if (state.phase === 'live') {
      const info = state.remote || {};
      const label = liveLabel[info.state] || '直播状态未知';
      body = `<div class="ls-title" data-live-label></div>
        <div class="ls-stats">
          <span>已运行 <b data-stat="duration"></b></span>
          <span>视频帧 <b data-stat="frames"></b></span>
          <span>音频块 <b data-stat="audio"></b></span>
        </div>
        ${button('结束直播', 'stop', 'ls-danger', state.busy)}`;
    }
    root.innerHTML = `<div class="ls-panel${state.phase === 'input' ? ' ls-config' : ''}">
      ${state.phase === 'input' ? '' : '<div class="ls-heading">' + platform.name + ' · 直播助手</div>'}
      ${body}
      ${state.error ? '<p class="ls-error" role="alert" data-error></p>' : ''}
      ${state.notice ? '<p class="ls-notice" role="status" data-notice></p>' : ''}
    </div>`;
    const title = root.querySelector('[data-guide-title]');
    const detail = root.querySelector('[data-guide-detail]');
    const link = root.querySelector('[data-guide-url]');
    if (title && guide[state.guideIndex]) title.textContent = guide[state.guideIndex].title;
    if (detail && guide[state.guideIndex]) detail.textContent = guide[state.guideIndex].detail;
    if (link && guide[state.guideIndex]?.url) {
      link.href = guide[state.guideIndex].url;
      link.textContent = (guide[state.guideIndex].linkText || '打开官方页面') + ' ↗';
    }
    const helpLink = root.querySelector('[data-guide-help-url]');
    if (helpLink && guide[state.guideIndex]?.helpUrl) helpLink.href = guide[state.guideIndex].helpUrl;
    const help = root.querySelector('[data-guide-help]');
    if (help && guide[state.guideIndex]) help.textContent = getGuideHelp(state.platform, guide[state.guideIndex].title);
    if (state.phase === 'input') {
      const secret = root.querySelector('[data-input="secret"]');
      const server = root.querySelector('[data-input="server"]');
      if (secret) secret.value = state.rawInput;
      if (server) server.value = state.server;
    }
    if (state.phase === 'live') {
      const info = state.remote || {};
      root.querySelector('[data-live-label]').textContent = liveLabel[info.state] || '直播状态未知';
      root.querySelector('[data-stat="duration"]').textContent = Math.round(info.duration_s || 0) + ' 秒';
      root.querySelector('[data-stat="frames"]').textContent = String(info.video_frames || 0);
      root.querySelector('[data-stat="audio"]').textContent = String(info.audio_chunks || 0);
    }
    if (state.error) root.querySelector('[data-error]').textContent = state.error;
    if (state.notice) root.querySelector('[data-notice]').textContent = state.notice;
    if (safePhases.has(state.phase)) {
      try {
        localStorage.setItem(progressKey, JSON.stringify({
          phase: state.phase, platform: state.platform,
          readiness: state.readiness, guideIndex: state.guideIndex,
        }));
      } catch { /* storage can be disabled */ }
    }
  }

  function move(phase) {
    state.error = '';
    state.notice = '';
    state.phase = phase;
    render();
  }

  async function refresh() {
    if (checking || state.destroyed) return;
    checking = true;
    try {
      const result = await callLiveStream(mcpId, 'info');
      if (state.destroyed) return;
      state.connected = true;
      if (state.busy) return;
      const savedButtonChanged = state.phase === 'input' &&
        (Boolean(state.remote?.configured) !== Boolean(result.configured)
          || state.remote?.platform !== result.platform);
      state.remote = result;
      if (savedButtonChanged) render();
      if (state.phase === 'live' && Object.hasOwn(PLATFORMS, result.platform)) state.platform = result.platform;
      if (ACTIVE_STATES.has(result.state)) {
        state.phase = 'live';
        state.ready = false;
        render();
      } else if (state.phase === 'live') {
        state.ready = false;
        move('platform');
        state.notice = '推流已停止';
        render();
      } else if (state.phase === 'review' && !result.ready_to_go_live) {
        state.ready = Boolean(result.configured);
        state.notice = '';
        render();
      }
    } catch {
      // A temporarily unreachable Driver does not mean the platform stream stopped.
      state.connected = false;
      if (state.phase === 'live') {
        state.error = '暂时无法获取机器人状态，请先检查网络，不要重复开播。';
        render();
      }
    } finally { checking = false; }
  }

  async function act(action) {
    if (state.busy || state.destroyed) return;
    state.error = '';
    state.notice = '';
    if (action.startsWith('platform:')) {
      state.platform = action.split(':')[1];
      state.rawInput = '';
      state.server = '';
      render(); return;
    }
    if (action.startsWith('readiness:')) {
      state.readiness = action.split(':')[1];
      render(); return;
    }
    if (action === 'to-readiness') { move('readiness'); return; }
    if (action === 'to-platform') { move('platform'); return; }
    if (action === 'start-guide') {
      state.guideIndex = 0; state.showHelp = false;
      move(state.readiness === 'ready' ? 'input' : 'guide'); return;
    }
    if (action === 'guide-next') {
      state.showHelp = false; state.guideIndex += 1;
      move(state.guideIndex >= getGuide(state.platform, state.readiness).length ? 'input' : 'guide');
      return;
    }
    if (action === 'guide-back') {
      state.showHelp = false;
      if (state.guideIndex === 0) move('readiness');
      else { state.guideIndex -= 1; render(); }
      return;
    }
    if (action === 'guide-help') { state.showHelp = !state.showHelp; render(); return; }
    if (action === 'input-back') {
      const guides = getGuide(state.platform, state.readiness);
      state.guideIndex = Math.max(0, guides.length - 1);
      move(guides.length ? 'guide' : 'readiness'); return;
    }
    if (action === 'paste') {
      try {
        const clip = await navigator.clipboard.readText();
        if (!clip.trim()) throw new Error('剪贴板是空的');
        state.rawInput = clip.trim();
        render();
      } catch {
        state.error = '无法自动读取剪贴板，请按 Ctrl+V 手动粘贴。';
        render();
      }
      return;
    }
    if (action === 'use-saved') {
      if (state.remote?.configured && state.remote?.platform === state.platform) {
        state.ready = true;
        move('review');
      } else {
        state.error = '没有找到该平台已保存的直播信息，请重新填写。';
        render();
      }
      return;
    }
    if (action === 'to-input') { state.ready = false; move('input'); return; }
    if (!['save', 'start', 'stop'].includes(action)) return;
    state.busy = true;
    render();
    try {
      if (action === 'save') {
        const url = state.platform === 'xiaohongshu'
          ? makeXiaohongshuStreamUrl(state.server, state.rawInput)
          : makeStreamUrl(state.platform, state.rawInput);
        if (!(await authorize())) return;
        await saveLiveStreamConfig(mcpId, cardId, url);
        state.rawInput = '';
        state.server = '';
        state.ready = true;
        move('review');
      } else if (action === 'start') {
        if (!state.ready) throw new Error('请先保存本场直播信息');
        if (!(await authorize())) return;
        if (!window.confirm('确认现在公开 Bumi 摄像头画面和麦克风声音，并开始推流吗？')) return;
        await authorizeStoredLiveStream(mcpId, cardId);
        const result = await startApprovedLiveStream(mcpId, cardId);
        if (!ACTIVE_STATES.has(result.state)) throw new Error('尚未开始推流，请检查机器人状态。');
        state.remote = result;
        state.ready = false;
        move('live');
      } else {
        // Stopping stays available even if the Canvas edit lock was lost.
        const result = await callLiveStream(mcpId, 'end_live');
        state.remote = result;
        if (result.state === 'idle') {
          state.ready = false;
          move('platform');
          state.notice = '推流已停止';
          render();
        } else move('live');
      }
    } catch (error) {
      state.error = error.message || '操作失败，请重试。';
    } finally {
      state.busy = false;
      render();
    }
  }

  const handleInput = event => {
    if (event.target.dataset.input === 'secret') state.rawInput = event.target.value;
    if (event.target.dataset.input === 'server') state.server = event.target.value;
  };
  const handleClick = event => {
    const target = event.target.closest('button[data-action]');
    if (!target || !root.contains(target)) return;
    event.preventDefault();
    event.stopPropagation();
    void act(target.dataset.action);
  };
  const stopPointer = event => event.stopPropagation();
  root.addEventListener('input', handleInput);
  root.addEventListener('click', handleClick);
  root.addEventListener('pointerdown', stopPointer);
  render();
  void refresh();
  const interval = setInterval(refresh, 4000);
  return () => {
    state.destroyed = true;
    clearInterval(interval);
    root.removeEventListener('input', handleInput);
    root.removeEventListener('click', handleClick);
    root.removeEventListener('pointerdown', stopPointer);
    state.rawInput = '';
  };
}
