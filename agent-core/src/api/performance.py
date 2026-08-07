"""
api/performance.py — 性能分析 API（开放 Span 式）。
"""

import time
from fastapi import APIRouter, Query

import perf_log

router = APIRouter(prefix='/performance', tags=['performance'])


@router.get('/latest')
async def get_latest(n: int = Query(20, ge=1, le=200)):
    return perf_log.query_latest(n=n)


@router.get('/spans')
async def get_spans(trace_id: str = Query(...)):
    return perf_log.query_spans(trace_id=trace_id)


@router.get('/aggregate')
async def get_aggregate(
    start: float = Query(0),
    end: float = Query(0),
):
    return perf_log.aggregate(start=start, end=end)


@router.get('/usage')
async def get_usage(range: str = Query('7d')):
    """Token usage summary + daily/hourly breakdown."""
    import datetime
    now = time.time()

    # Calculate start time (UTC+8 day boundary for 'today')
    tz = datetime.timezone(datetime.timedelta(hours=8))
    now_dt = datetime.datetime.now(tz)

    if range == 'today':
        # Start of today in UTC+8
        start_of_day = now_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        start = start_of_day.timestamp()
    else:
        range_map = {'7d': 7, '30d': 30, '90d': 90}
        days = range_map.get(range, 7)
        start_of_day = (now_dt - datetime.timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
        start = start_of_day.timestamp()

    summary = perf_log.query_usage_summary(start=start)

    # today/7d → hourly, 30d/90d → daily
    if range in ('today', '7d'):
        breakdown = perf_log.query_usage_hourly(start=start)
    else:
        breakdown = perf_log.query_usage_daily(start=start)

    return {'summary': summary, 'breakdown': breakdown, 'granularity': 'hourly' if range in ('today', '7d') else 'daily'}


@router.delete('/clear')
async def clear_data():
    import config
    conn = config._get_conn()
    conn.execute('DELETE FROM perf_spans')
    conn.execute('DELETE FROM perf_turns')
    conn.execute('DELETE FROM token_usage')
    conn.commit()
    conn.close()
    return {'code': 200, 'message': 'cleared'}
