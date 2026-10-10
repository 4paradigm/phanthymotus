/**
 * activity-summary.js — one activity event → the one line shown for it.
 *
 * Split out of activity-log.js, which cannot be imported under `node --test`:
 * it pulls in mobile.js and from there the whole renderer tree (three.js). This
 * file imports nothing, so every row type can be covered by a test — which is
 * the point, because a wrong branch here is invisible in review and shows up as
 * a row of raw JSON on a robot.
 */

export function summarizeEvent(event) {
  const p = event.payload || {};
  switch (event.type) {
    case 'mcp_call':       return `${p.tool || ''}(${_trunc(JSON.stringify(p.args || {}), 200)})`;
    case 'mcp_result':     return `← ${_trunc(JSON.stringify(p.result), 400)}`;
    case 'agent_thought':  return p.text || '';
    // 单独一类，不并进 agent_thought：本功能最坏的失败模式是汇报里编造一个不存在的
    // 发现，得能和模型自己的想法分开看才抓得住。
    // 轮数维度已删（纯时间触发），payload 里只剩秒数 —— 再读 silent_rounds 会渲染成
    // 「静默undefined轮」。in_turn 区分"turn 还在跑"和"turn 结束后子代理还在干"。
    case 'narration':      return `📢 ${p.text || ''} (静默${p.silent_seconds}s${p.in_turn ? '' : '·后台'})`;
    case 'asr_result':     return `"${p.text || ''}"`;
    // 旁边听到、没在跟机器人说的话。在活动流里它此前没有任何痕迹：bg subagent
    // 可能被节流、或被判为「没有内容」而根本不 spawn，那句话就只存在于 DDS 上。
    // 说话人/语种/情绪只在有值时出现 —— perception 侧已经把无信息的取值压掉了，
    // 这里再补一层空值过滤，是因为一行日志的宽度比 prompt 更金贵。
    case 'asr_background': {
      const tags = ['speaker_name', 'speaker_id', 'lang', 'emotion', 'audio_event']
        .filter((k) => p[k])
        .map((k) => (k === 'speaker_name' || k === 'speaker_id' ? p[k] : `${p[k]}`));
      return `👂 "${p.text || ''}"${tags.length ? ` · ${tags.join(' · ')}` : ''}`;
    }
    case 'trigger':        return p.text || _trunc(JSON.stringify(p), 400);
    case 'peer_pair_request': return `${p.display_name || p.peer_id?.slice(0, 12) || 'peer'} 请求配对 · 验证码 ${p.code}`;
    case 'peer_tool_call':   return `${p.peer || 'peer'} → ${p.tool}${p.action ? `(${p.action})` : ''}`;
    case 'peer_tool_result': return p.ok
      ? `${p.peer || 'peer'} ← ${p.tool} 完成${p.elapsed_ms != null ? ` ${p.elapsed_ms}ms` : ''}${p.action_id ? ` [${p.action_id}]` : ''}`
      : `${p.peer || 'peer'} ← ${p.tool} 失败: ${_trunc(String(p.error || ''), 200)}`;
    case 'render':         return `renderer=${p.renderer}`;
    case 'llm_usage':      return `tokens: in=${p.prompt_tokens} out=${p.completion_tokens} cached=${p.cached_tokens}`
                              + (p.elapsed_s != null ? ` · ${p.elapsed_s}s` : '');
    case 'turn_end':
      if (p.usage) return `✓ ${p.rounds}轮 ${p.duration_s}s | tokens: in=${p.usage.prompt_tokens} out=${p.usage.completion_tokens} cached=${p.usage.cached_tokens}`;
      return '✓ turn complete';
    case 'status':
      return 'connected' in p
        ? (p.connected ? '● 已连接' : '○ 断开')
        : (p.online ? '⬤ online' : '○ offline') + (p.mcp_id ? ` [${p.mcp_id}]` : '');
    default:               return _trunc(JSON.stringify(p), 300);
  }
}

// **Cuts out of the middle, not off the end.**
//
// Every payload here is a JSON blob whose boilerplate comes first and whose
// point comes last: an ACP completion spends its first 180 characters on
// `type` / `action_id` / `status` before reaching `reason`, so a head-only cut
// reliably kept the part that could be guessed and dropped the part that could
// not. On the robot this showed up as an activity row that always ended in
// `"reaso`. The middle of a blob is the cheapest thing to lose.
function _trunc(str, n) {
  str = str || '';
  if (str.length <= n) return str;
  const tail = Math.min(Math.floor(n / 3), 200);
  return `${str.slice(0, n - tail)}…（省略 ${str.length - n} 字）…${str.slice(-tail)}`;
}
