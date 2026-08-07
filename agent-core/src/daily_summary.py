"""
daily_summary.py — 每日自动摘要。

每天凌晨定时 spawn 一个 subagent 做当天摘要：
- 用户交互摘要
- subagent 结论汇总
- 任务完成状态
- 异常事件记录
- 每日复盘（做得好/不好）
- 技能发现（重复模式→可抽象为 skill）

结果存入 subagent_conclusions（source_type='daily_summary'）。
"""

import asyncio
import datetime
import time


_DAILY_SUMMARY_GOAL = """[daily] 每日摘要：查询今天的所有记忆数据，生成一份结构化日报。

请使用 memory_recall 工具检索今天的数据（time_range='1d'），分别用不同关键词检索。

## 输出格式

### 1. 用户交互摘要
- 今天用户主要问了什么、要求做了什么

### 2. Subagent 结论汇总
- 后台监控发现了什么（电池/传感器状态变化）
- 研究/搜索任务的结果

### 3. 任务完成状态
- 完成了哪些任务
- 失败/未完成的任务及原因

### 4. 异常事件
- 电池告警、硬件异常、通信问题等

### 5. 每日复盘
- 做得好的：响应及时、信息准确、用户满意的地方
- 做得不好的：重复打扰、响应不准确、遗漏信息等

### 6. 技能发现
- 用户反复要求做的操作（可以抽象为自动化 skill）
- 发现的固定模式/流程

请用中文输出，保持简洁。每个部分 2-5 条要点即可。
"""


async def run():
    """常驻协程：每天凌晨 2:03 执行每日摘要。"""
    while True:
        # 计算距下一个 02:03 的等待时间
        now = datetime.datetime.now()
        target = now.replace(hour=2, minute=3, second=0, microsecond=0)
        if now >= target:
            target += datetime.timedelta(days=1)
        wait_seconds = (target - now).total_seconds()
        print(f'[daily_summary] next run at {target.strftime("%Y-%m-%d %H:%M")}, waiting {wait_seconds/3600:.1f}h')
        await asyncio.sleep(wait_seconds)

        # 执行摘要
        await _do_summary()


async def _do_summary():
    """Spawn subagent 执行每日摘要。"""
    try:
        from subagent import _manager_instance
        if not _manager_instance:
            print('[daily_summary] no subagent manager, skip')
            return

        from subagent.protocol import SubagentSpec, P_LOW
        spec = SubagentSpec(
            goal=_DAILY_SUMMARY_GOAL,
            priority=P_LOW,
            model=None,
            tool_deny=['mcp__*', 'Bash', 'Write', 'Edit'],
            max_rounds=10,
            timeout_s=120,
            context_seed='',
        )
        agent_id = await _manager_instance.spawn(spec)
        print(f'[daily_summary] spawned subagent: {agent_id}')
    except Exception as e:
        print(f'[daily_summary] failed to spawn: {e}')
