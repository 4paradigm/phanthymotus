"""
peer/backoff.py — 对「被对端明确拒绝」的 peer 做退避，只针对后台轮询。

为什么需要它，以及为什么只针对这一种失败：

`dds_state` 每 5s 推一次状态、`lan` adapter 探活、`mcp_bridge` 拉 `tools/list`，
三条都是没有人触发的后台轮询。它们原来对任何失败都一视同仁地按原节奏重试，而
transport 的注释把这解释成设计：「一台关掉的机器人本来就是不可达的常态」。对**连不上**
来说这是对的，代价也只落在发起方自己身上。

但 403 不是连不上。它是对端在说「我的表里没有你」，而这件事不会因为再问一次就变。
天轶实测：它的 `peers` 表空了之后，Orin5 和 R1 连着 7 天每 5s 敲一次门，累计

    POST /api/peer/inbox/state    12363
    POST /api/peer/inbox/ping      1779
    GET  /api/peer/tools/list      1067
    ────────────────────────────────────
                                  15209   全部 403 unknown_peer

每一次都让接收方做一次完整的 Ed25519 验签，并且把它的日志淹到真正的错误看不见 ——
排查那批 LLM 400 的时候，就是这些 403 把 `[decision] error in _one_turn` 冲散的。

所以这里的判据很窄：**只有 HTTP 401/403 才退避**。连接失败、超时、DNS —— 一律不碰，
保持原样。互动路径（发消息、`peer_call`、`peer_delegate`）也不走这个闸：那些是人或
LLM 明确要求的动作，应该真的试一次并拿到真实错误，而不是被一个后台状态机拒绝。
"""

import re
import time

# 第一次被拒后等多久，以及上限。60s 起步是因为重新配对是人在界面上点的，
# 分钟级的恢复延迟无感；15 分钟封顶，免得一次误判要等到天荒地老。
INITIAL_S = 60.0
MAX_S = 900.0

# peer_id → {'until': 单调时钟秒, 'delay': 当前退避长度, 'reason': 最后一次拒绝原因}
_gated: dict[str, dict] = {}

# transport 把每个 endpoint 的结果拼成 '<base> → HTTP <code> <body>'，多个用 '; ' 连接。
_AUTHORITATIVE = re.compile(r'HTTP (401|403)\b')


def _is_authoritative_rejection(reason: str) -> bool:
    """对端是否**明确**拒绝了我们 —— 而不是没答上来。

    多 endpoint 时要求每一段都是 401/403。只要有一段是连接错误，那这个 peer 至少还有
    一条路可能是活的，退避就会盖掉一条本来能用的链路。
    """
    if not reason:
        return False
    parts = [p.strip() for p in reason.split(';') if p.strip()]
    return bool(parts) and all(_AUTHORITATIVE.search(p) for p in parts)


def should_skip(peer_id: str) -> bool:
    """这一轮后台轮询要不要跳过这个 peer。"""
    entry = _gated.get(peer_id)
    if not entry:
        return False
    if time.monotonic() >= entry['until']:
        return False
    return True


def note_result(peer_id: str, ok: bool, reason: str = '') -> None:
    """记一次后台轮询的结果。成功会立刻解除退避。"""
    if ok:
        if _gated.pop(peer_id, None) is not None:
            print(f'[peer] {peer_id[:8]} accepted us again — backoff cleared')
        return
    if not _is_authoritative_rejection(reason):
        return
    entry = _gated.get(peer_id)
    delay = min(entry['delay'] * 2, MAX_S) if entry else INITIAL_S
    _gated[peer_id] = {'until': time.monotonic() + delay,
                       'delay': delay,
                       'reason': reason}
    print(f'[peer] {peer_id[:8]} rejected us ({reason[:120]}) — '
          f'backing off {delay:.0f}s before the next background poll')


def reset(peer_id: str) -> None:
    """配对/重新配对之后调用：立刻恢复正常节奏，不用等退避走完。"""
    _gated.pop(peer_id, None)


def state() -> dict[str, dict]:
    """当前处于退避中的 peer，供排查用。"""
    now = time.monotonic()
    return {pid: {'retry_in_s': round(max(0.0, e['until'] - now)),
                  'reason': e['reason']}
            for pid, e in _gated.items()}


def clear() -> None:
    _gated.clear()
