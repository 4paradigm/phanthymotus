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
    '<p><a href="https://q.qq.com/" target="_blank" rel="noopener noreferrer">前往 QQ 开放平台</a></p></details>',
    '<div class="qq-console-list-section"><div class="qq-management-head"><strong>私聊用户</strong><span class="channel-users-hint">已与机器人互动的用户</span></div>',
    '<div class="qq-console-list-head"><span>用户</span><span>权限</span><span>操作</span></div>',
    '<div data-manage-users></div></div>',
    '<div class="qq-console-list-section"><div class="qq-management-head"><strong>已接入群聊</strong><span class="channel-users-hint">首次 @机器人后自动接入</span></div>',
    '<div class="qq-console-list-head"><span>群聊</span><span>权限</span><span>操作</span></div>',
    '<div data-manage-groups></div></div>',
    '<div data-qq-advanced-slot></div>',
    '<details class="qq-management-advanced" data-qq-rebind><summary>换绑管理员</summary>',
    '<p class="channel-users-hint">仅在转交管理权时解除当前绑定，随后需要重新验证管理员身份。</p>',
    '<button class="btn-ghost btn-sm" data-manage-revoke>解除管理员绑定</button></details>',
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
      if (user.role === 'owner') {
        row.appendChild(element('span', user.tool_filter === '*' ? '管理员 · 全部工具' : '管理员 · 自定义权限', 'qq-row-permission'));
        row.appendChild(element('span', '—', 'qq-row-action'));
      } else {
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
      const action = button(item.status === 'approved' ? '停用' : '恢复', () =>
        run(async () => {
          const enabling = item.status !== 'approved';
          if (!enabling && !window.confirm('确定停用这个 QQ 群？')) return;
          await request('/qq/groups/permission', 'PUT', {
            channel_id:channel(), group_id:item.group_id,
            default_permission:'chat',
            mention_only:true, enabled:enabling,
          });
          say(enabling ? '群聊已恢复（成员仅聊天）' : '群聊已停用');
          await loadGroups();
        }));
      action.classList.add('qq-row-action');
      row.appendChild(action);
      target.appendChild(row);
    }
  }
  async function refresh() {
    await loadUsers();
    await loadGroups();
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
  $('[data-manage-revoke]').onclick = () => run(async () => {
    if (!window.confirm('确定解除这个 QQ 机器人的管理员绑定？原管理员权限将立即撤销。')) return;
    await request('/qq/pair/revoke', 'POST', {
      channel_id:channel(), confirmation:'解除管理员绑定',
    });
    say('旧管理员权限已撤销。请在添加渠道向导中选择现有 QQ，再绑定新管理员。');
  });
  document.getElementById('btn-channels')?.addEventListener('click', () => {
    host.hidden = true;
    setQQDetailsMode(false);
  });
}
