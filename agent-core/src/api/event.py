"""
api/event.py — MCP / 外部系统推送事件的 HTTP 入口。

POST /api/event
{
    "source":  "mcp:cam-head",     # 必填
    "text":    "检测到陌生人脸",    # 必填，自然语言描述
    "payload": { ... }              # 可选，原始数据
}

接收后：
  1. 写入 event_bus（唤醒 Agent Loop）
  2. 广播到 /ws/motus（前端可视化）
"""

import fastapi
from pydantic import BaseModel

import event_bus
from api.motus_stream import push_event

router = fastapi.APIRouter(prefix='/event', tags=['event'])


class EventRequest(BaseModel):
    source:  str
    text:    str
    payload: dict = {}


@router.post('')
async def receive_event(req: EventRequest):
    await event_bus.enqueue(
        source  = req.source,
        text    = req.text,
        payload = req.payload,
    )
    # 同时推到前端 Activity 面板
    await push_event({
        'type':    'trigger',
        'mcp_id':  req.source,
        'payload': {'text': req.text, **req.payload},
    })
    return {'code': 200}
