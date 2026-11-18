/** Manage QQ users, channel defaults and automatically discovered group permissions. */
function element(tag, text, cls = '') {
  const el = document.createElement(tag);
  el.textContent = text || '';
  if (cls) el.className = cls;
  return el;
}
function button(title, action) {
  const b = element('button', title, 'btn-ghost btn-sm');
  b.type = 'button';
  b.onclick = action;
  return b;
}
async function request(path, method = 'GET', body = null) {
  const res = await fetch('/api/channel' + path, {
    method, headers: {'Content-Type':'application/json'},
    ...(body ? {body:JSON.stringify(body)} : {}),
  });
  const out = await res.json().catch(() => ({}));
  if (!res.ok) throw Error(out.detail || '请求失败');
  return out;
}
export function initQQManagement() {
  const host = document.getElementById('channel-details-panel');
  if (!host || document.getElementById('qq-management')) return;
  const panel = document.createElement('section');
  panel.id = 'qq-management';
  panel.className = 'qq-management';
  panel.innerHTML = [
    '<div class="channel-users-header qq-console-header"><strong>QQ 机器人</strong><div class="qq-console-actions">',
    '<button class="btn-ghost btn-sm" type="button" data-manage-refresh>刷新</button>',
    '<button class="btn-ghost btn-sm" type="button" data-manage-close>返回渠道列表</button></div></div>',
    '<p class="channel-users-hint" data-manage-selected></p>',
    '<select data-manage-channel hidden aria-label="当前 QQ 渠道"></select>',
    '<p data-manage-message role="status"></p>',
    '<div data-qq-core-settings-slot></div>',
    '<details class="qq-console-invite" data-qq-invite-help><summary>邀请其他 QQ 用户添加机器人</summary>',
    '<p class="channel-users-hint">测试阶段：QQ 开放平台 → 我的机器人 → 开发设置 → 开发体验号设置，添加测试 QQ 用户。</p>',
    '<p class="channel-users-hint">在平台的「使用范围与人员」或测试配置中找到机器人添加二维码或资料卡，分享给允许使用的用户。正式发布后按实际开放范围添加。</p>',
    '<p class="channel-users-hint">对方添加机器人并私聊后，会自动出现在用户列表中，无需再次审批。</p>',
    '<p><a href="https://q.qq.com/qqbot/dashboard/" target="_blank" rel="noopener noreferrer">前往 QQ 开放平台</a></p></details>',
    '<div class="qq-console-list-section"><div class="qq-management-head"><strong>私聊用户</strong><span class="channel-users-hint">已与机器人互动的用户</span></div>',
    '<div class="qq-console-list-head"><span>用户</span><span>权限</span><span>操作</span></div>',
    '<div data-manage-users></div></div>',
    '<div class="qq-console-list-section"><div class="qq-management-head"><strong>已接入群聊</strong><span class="channel-users-hint">首次 @机器人后自动接入</span></div>',
    '<div class="qq-console-list-head"><span>群聊</span><span>权限</span></div>',
    '<div data-manage-groups></div></div>',
    '<section class="qq-console-list-section" data-qq-health><div class="qq-management-head"><strong>功能运行状态</strong><span class="channel-users-hint">基于当前进程实际收发记录</span></div>',
    '<p class="channel-users-hint" data-qq-diagnostic-summary>尚未检测</p><div data-qq-capabilities></div></section>',
    '<div data-qq-advanced-slot></div>',
  ].join('');
  host.appendChild(panel);
  const $ = selector => panel.querySelector(selector);
  const channel = () => $('[data-manage-channel]').value;
  const say = message => { $('[data-manage-message]').textContent = message; };
  const setQQDetailsMode = active => {
    const list = document.getElementById('channel-list');
    const add = document.getElementById('channel-add-btn');
    if (list) list.hidden = active;
    if (add) add.hidden = active;
  };
  let selectedAppId = '';

  const run = async task => {
    try { await task(); } catch(err) { say(err.message); }
  };
  async function loadChannels(requestedChannel) {
    const state = await request('/qq/setup/status');
    const select = $('[data-manage-channel]');
    const keep = select.value;
    select.textContent = '';
    for (const ch of (state.channels || [])) {
      const option = element('option', 'QQ · ' + ch.app_id);
      option.value = ch.channel_id;
      select.appendChild(option);
    }
    const requested = requestedChannel || keep;
    const selected = (state.channels || []).find(ch => ch.channel_id === requested);
    if (selected) select.value = selected.channel_id;
    $('[data-manage-selected]').textContent = selected ? '当前机器人 · ' + selected.app_id : '';
    selectedAppId = selected?.app_id || '';
    if (!selected) {
      host.hidden = true;
      setQQDetailsMode(false);
      return;
    }
    host.hidden = false;
    setQQDetailsMode(true);
    await refresh();
    host.scrollIntoView({block:'start', behavior:'smooth'});
  }
  async function loadUsers() {
    const result = await request('/users?platform=qq');
    const target = $('[data-manage-users]');
    target.textContent = '';
    const prefix = 'official:' + selectedAppId + ':';
    const users = (result.users || []).filter(user =>
      user.user_id?.startsWith(prefix) && user.user_id.split(':').length === 3);
    if (!users.length) {
      target.appendChild(element('p', '暂无已互动的私聊用户', 'channel-users-hint'));
      return;
    }
    for (const user of users) {
      const row = element('div', '', 'qq-management-row');
      row.appendChild(element('span', (user.display_name || 'QQ 用户') +
        ' · ' + user.user_id.slice(-6), 'qq-row-identity'));
      {
        const blocked = user.role === 'blocked';
        row.appendChild(element('span', blocked ? '已停用' : '仅聊天', 'qq-row-permission'));
        const action = button(blocked ? '恢复' : '停用', () => run(async () => {
          if (!blocked && !window.confirm('确定停用该 QQ 用户？')) return;
          if (blocked) {
            await request('/users/preset', 'POST', {
              platform:'qq', user_id:user.user_id, preset:'chat',
            });
          } else {
            await request('/users', 'POST', {
              platform:'qq', user_id:user.user_id,
              display_name:user.display_name || '', role:'blocked', tool_filter:'',
            });
          }
          say(blocked ? '用户已恢复仅聊天权限' : '用户已停用');
          await loadUsers();
        }));
        action.classList.add('qq-row-action');
        row.appendChild(action);
      }
      target.appendChild(row);
    }
  }
  async function loadGroups() {
    const result = await request('/qq/groups?channel_id=' + encodeURIComponent(channel()));
    const target = $('[data-manage-groups]');
    target.textContent = '';
    if (!result.groups?.length) {
      target.appendChild(element('p', '暂无已接入群聊。在已添加机器人的 QQ 群中 @机器人发消息后会自动出现在这里。', 'channel-users-hint'));
      return;
    }
    for (const item of result.groups) {
      const row = element('div', '', 'qq-management-row');
      row.appendChild(element('span', '群 · ' + item.group_id.slice(-6), 'qq-row-identity'));
      row.appendChild(element('span', item.status === 'approved' ? '仅聊天' : '已停用', 'qq-row-permission'));
      // Group @ access is automatic. There is no extra permission workflow
      // or per-group approval/disable action in the default product UI.
      target.appendChild(row);
    }
  }
  async function loadDiagnostics() {
    const report = await request('/qq/setup/diagnostics?channel_id=' + encodeURIComponent(channel()));
    const counts = report.events?.counts || {};
    $('[data-qq-diagnostic-summary]').textContent = [
      'Gateway：' + (report.gateway === 'connected' ? '已连接' : '未连接'),
      '已接收私聊 ' + (counts.private_inbound || 0),
      '已接收群聊 ' + (counts.group_inbound || 0),
      '平台接受回复 ' + (counts.platform_accepted || 0),
      '发送失败 ' + (counts.send_failed || 0),
    ].join('　·　') + '（重启后重新统计；平台接受不等于 QQ 客户端已收到）';
    const statuses = {
      observed: '已观测到实际事件', awaiting_real_test: '等待真实测试',
      configured: '已配置，待重启验证', configured_unverified: '配置已打开，尚未验收',
      not_enabled: '未启用', not_connected: '网关未连接',
      not_ready: '画布未连接', requires_platform_or_workflow: '待平台授权或业务接入',
    };
    const titles = {
      private_text:'私聊接收', group_at_text:'群聊 @ 接收',
      inbound_media:'接收图片/文件', outbound_media:'发送图片/文件',
      voice_transcription:'语音转写', typing:'输入状态提示',
      session_resume:'断线会话恢复', enable_markdown:'Markdown',
      enable_interactions:'交互按钮', enable_approvals:'审批功能',
      enable_guild_plain_messages:'频道普通消息',
    };
    const output = $('[data-qq-capabilities]');
    output.replaceChildren();
    for (const [key, title] of Object.entries(titles)) {
      const state = report.capabilities?.[key];
      if (!state) continue;
      const row = element('div', '', 'qq-management-row');
      row.append(
        element('span', title, 'qq-row-identity'),
        element('span', statuses[state.state] || state.state, 'qq-row-permission'),
      );
      output.appendChild(row);
    }
  }
  async function refresh() {
    await loadUsers();
    await loadGroups();
    await loadDiagnostics();
  }
  $('[data-manage-channel]').addEventListener('change', () => run(refresh));
  $('[data-manage-close]').onclick = () => {
    host.hidden = true;
    setQQDetailsMode(false);
  };
  document.addEventListener('qq-channel-manage', event => {
    const selected = event.detail?.channelId;
    if (selected) run(() => loadChannels(selected));
  });
  document.addEventListener('qq-settings-updated', event => {
    if (event.detail?.channelId === channel()) run(loadUsers);
  });
  $('[data-manage-refresh]').onclick = () => run(async () => {
    await refresh();
    document.dispatchEvent(new CustomEvent('qq-management-refresh', {detail:{channelId:channel()}}));
  });
  document.getElementById('btn-channels')?.addEventListener('click', () => {
    host.hidden = true;
    setQQDetailsMode(false);
  });
}
