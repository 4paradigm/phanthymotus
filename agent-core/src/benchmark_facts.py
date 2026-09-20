"""benchmark_facts.py — 真机上的事实来源。

## 为什么需要它

裁判（`benchmark_case.py`）从一开始就是纯函数，它的模块文档写着：

> 将来一个跑在真机器人上的用例，只要能吐出同样形状的事件流，这同一套断言直接可用。

**那个「只要」一直没人实现。** 事实流的唯一生产者是仿真器驱动的 `sim_report`，于是
benchmark 名义上面向解决方案，实际上只能在装了仿真器的机器上跑。这个模块补上缺的那一半：
agent-core 自己把它派发出去的异步动作记成同样形状的事实。

仿真器在场时仍然优先用 `sim_report` —— 它看得见世界，能给出轨迹占用这种 agent-core
无从知道的量。这里产出的是**真机上能知道的那个子集**，少掉的部分由裁判记为「不可测」，
不是记为通过（见 `benchmark_case.check_never_occupied`）。

## 事实从哪来

两个订阅点，都是 `mcp_client` 已有的：`on_action_registered`（动作开始）与
`on_action_settled`（有结局了）。ACP 的异步动作正好覆盖需要判定的那些行为 —— 导航和
讲解都是异步的，因为它们要等物理世界。

**分类按驱动申报的 `x-resource` 通道，不按工具名。** `peer/tools.py` 的模块文档记着猜
名字的那次代价：它查 `move`/`grasp`/`speak`，而真机的执行器叫 `loco`、`led`、`speaker`、
`switch_mode`，一个都不匹配，于是 viewer 权限的 peer 能驱动底盘。这里同样只认申报：

| 事实 | 判据 |
|---|---|
| `nav_start` / `arrive` / `nav_cancelled` / `nav_failed` | 通道含 `base` |
| `speak_start` / `speak_end` | 通道含 `mouth` |

没申报 `x-resource` 的工具不产出事实。这是**有意让它缺**而不是猜一个：一条猜错的事实
会让裁判判错，而缺一条事实至少还能在预检里说出来。

## 站名：唯一真正难的一处

真机的 ACP result **不带站名**。天轶完成导航时回的是 `{pose, elapsed_s,
action_id_chassis}`，仿真器回的才是 `{label, ...}`。目标名只存在于**派发参数**里，而参数名
各家不同：天轶是 `tag_name`，仿真器是 `label`，下一家会是别的。

挑一个参数名来读，等于把某一家驱动的 schema 焊进判定层。所以反过来：

> 一段路的 label = 这次派发的参数里，**值**等于用例 `waypoint_order` 中某一项的那个字符串。

与驱动无关，不需要知道任何 schema。降级也是诚实的：模型若按坐标导航而不是按站名，
就没有 label，`waypoint_order` 判失败 —— 而那是**正确**的判定，因为用例要的就是按名字
去那些站。
"""

from __future__ import annotations

import threading
import time

import mcp_client

# 申报的通道 → 这台机器上哪个部位在动。
NAV_CHANNELS = frozenset({'base'})
SPEECH_CHANNELS = frozenset({'mouth'})

# 结算状态 → 事实名。`completed` 之外的都不是到达，分开记是因为裁判要区分
# 「被打断」和「撞停」—— 前者是用例设计的，后者是事故。
_NAV_END = {'completed': 'arrive', 'cancelled': 'nav_cancelled',
            'error': 'nav_failed', 'timeout': 'nav_failed'}

# 分类名 → 事件名前缀。**必须显式写出来**，不能拿分类名去拼。
#
# 第一版写的是 f'{kind}_start'，于是 speech 类产出 `speech_start`，而裁判读的是
# `speak_start` —— 事实流里一句讲解都没有，`announce_after_arrive` 在真机上恒为失败，
# 全程零报错。事件名是裁判那一侧的契约，得照抄，不该由这边的分类名决定。
_PREFIX = {'nav': 'nav', 'speech': 'speak'}


class Recorder:
    """一次运行期间的事实累积器。

    `mcp_client` 的两个回调在事件循环之外也可能触发（SSE 线程、超时清理），所以这里
    的状态都在锁里改。
    """

    def __init__(self, waypoints: list[str] | None = None):
        # 已知的站名。派发参数里的值匹配到其中之一，就是这一段的目标。
        self.waypoints = [str(w) for w in (waypoints or [])]
        self._lock = threading.RLock()
        self._started = time.time()
        self._events: list[dict] = []
        self._acp_posts: list[dict] = []
        # action_id → 派发时记下的东西。结算时查不到 mcp_client 那几张表了
        # （`_forget_pending` 先拆表再通知），所以自己留一份。
        self._inflight: dict[str, dict] = {}

    # ── 订阅点 ────────────────────────────────────────────────────────────────

    def on_registered(self, action_id: str, tool: str, args: dict, resource) -> None:
        kind = _kind(resource)
        if kind is None:
            return
        label = self.label_of(args)
        with self._lock:
            self._inflight[action_id] = {'kind': kind, 'tool': tool, 'label': label}
            self._append(f'{_PREFIX[kind]}_start', tool=tool, label=label,
                         action_id=action_id)

    def on_settled(self, action_id: str, _resource=None) -> None:
        with self._lock:
            entry = self._inflight.pop(action_id, None)
        if entry is None:
            return
        status = _status_of(action_id)
        with self._lock:
            if entry['kind'] == 'nav':
                name = _NAV_END.get(status, 'nav_failed')
                self._append(name, tool=entry['tool'], label=entry['label'],
                             action_id=action_id, status=status,
                             **({'reason': status} if name == 'nav_failed' else {}))
            else:
                self._append('speak_end', tool=entry['tool'], action_id=action_id,
                             status=status)

    def on_acp_post(self, body: dict) -> None:
        """`/api/acp/complete` 收到的原样 body。

        裁判的 `interrupted_leg` / `exactly_one_terminal_post` 读的就是这个形状，所以
        存原样，不重新组装。真机的 result 里没有 `label`，这里补上我们自己认出来的
        那个 —— 否则那两条断言在真机上没有任何可比的东西。
        """
        action_id = str(body.get('action_id') or '')
        with self._lock:
            entry = self._inflight.get(action_id) or {}
        post = dict(body)
        label = entry.get('label')
        if label and isinstance(post.get('result'), dict) and 'label' not in post['result']:
            post['result'] = {**post['result'], 'label': label}
        with self._lock:
            self._acp_posts.append(post)

    # ── 读出 ──────────────────────────────────────────────────────────────────

    def facts(self) -> dict:
        """和 `sim_report` 同形状的一份事实。

        **没有 `trail_occupied`** —— 它要拿轨迹对着占用栅格数，只有仿真器算得出。不放
        一个 0 进去是要紧的：裁判会把缺席读成「不可测」，而 0 会被读成「一个点都没压到」，
        于是安全维度报一个从没查过的满分。
        """
        with self._lock:
            return {'events': list(self._events), 'acp_posts': list(self._acp_posts),
                    'source': 'agent-core', 'waypoints': list(self.waypoints)}

    # ── 内部 ──────────────────────────────────────────────────────────────────

    def label_of(self, args: dict) -> str:
        """派发参数里能对上某个已知站名的那个值。对不上就是空串。

        按**值**找而不是按参数名找 —— 见模块文档。遍历顺序按参数名排序，好让同一份
        参数每次得到同一个答案（两个参数都对上时不该看字典顺序的脸色）。
        """
        for key in sorted(args or {}):
            value = args[key]
            if isinstance(value, str) and value in self.waypoints:
                return value
        return ''

    def _append(self, event: str, **fields) -> None:
        # `t` 是**相对这次运行开始**的秒数，和 `sim_report` 一致。裁判全程只做差值与
        # 比大小，混进绝对时间戳不会报错，只会让复盘时的数字看不懂。
        self._events.append({'event': event, 't': round(time.time() - self._started, 3),
                             **{k: v for k, v in fields.items() if v not in (None, '')}})


def _kind(resource) -> str | None:
    """申报的通道属于哪一类。没申报就是 None —— 不猜。"""
    if not resource:
        return None
    channels = set(resource)
    if channels & NAV_CHANNELS:
        return 'nav'
    if channels & SPEECH_CHANNELS:
        return 'speech'
    return None


def _status_of(action_id: str) -> str:
    """这个动作的结局。

    两条路各存各的：完成走 `mark_action_complete`，payload 留在 `_pending_results`；
    超时 / 取消走 `_forget_pending`，它**先拆表再通知**，所以这时只剩
    `action_outcome`。两个都问，才不会把一次超时读成没有结局。
    """
    payload = mcp_client.pending_result(action_id)
    if isinstance(payload, dict) and payload.get('status'):
        return str(payload['status'])
    outcome = mcp_client.action_outcome(action_id) or {}
    return str(outcome.get('status') or 'unknown')


# ── 当前运行 ──────────────────────────────────────────────────────────────────
#
# 同一时刻只有一次运行（`benchmark_runner` 的单例），所以这里也只留一个。订阅是
# 进程级、一次性的：回调本身按 `_current is None` 短路，比反复 subscribe/unsubscribe
# 少一类竞态 —— 后者在结算晚于停止记录时会丢事实。

_current: Recorder | None = None
_wired = False


def start(waypoints: list[str] | None = None) -> Recorder:
    global _current
    _wire()
    _current = Recorder(waypoints)
    return _current


def stop() -> None:
    global _current
    _current = None


def current() -> Recorder | None:
    return _current


def note_acp_post(body: dict) -> None:
    """`start.py` 的 `/acp/complete` 转过来的一份。"""
    recorder = _current
    if recorder is not None:
        recorder.on_acp_post(body)


def _wire() -> None:
    global _wired
    if _wired:
        return
    mcp_client.on_action_registered(
        lambda aid, tool, args, res: _current and _current.on_registered(aid, tool, args, res))
    mcp_client.on_action_settled(
        lambda aid, res=None: _current and _current.on_settled(aid, res))
    _wired = True
