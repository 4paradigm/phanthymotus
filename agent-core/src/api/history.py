"""Chat history API endpoints."""

import fastapi
from pydantic import BaseModel

import chat_history

router = fastapi.APIRouter(prefix='/history', tags=['history'])


@router.get('/sessions')
def get_sessions(limit: int = 50, offset: int = 0):
    sessions, total = chat_history.list_sessions(limit, offset)
    return {'sessions': sessions, 'total': total}


@router.get('/sessions/{session_id}')
def get_session(session_id: str):
    turns = chat_history.get_session_turns(session_id)
    return {
        'session_id': session_id,
        'messages': [t['messages'] for t in turns],
        'turn_times': [
            {'started_at': t['started_at'], 'updated_at': t['updated_at']}
            for t in turns
        ],
    }


def _forget_if_current(session_ids: list[str]) -> None:
    """删除当前会话时，同时取消该轮并丢弃内存上下文。"""
    import event
    if event.llm._session_id and event.llm._session_id in session_ids:
        event.llm.forget_conversation()
    elif event.llm._channel_session_id and event.llm._channel_session_id in session_ids:
        event.llm.forget_conversation(channel_only=True)


@router.delete('/sessions/{session_id}')
def delete_session(session_id: str):
    _forget_if_current([session_id])
    chat_history.delete_session(session_id)
    return {'ok': True}


class BatchDeleteBody(BaseModel):
    ids: list[str]


@router.post('/sessions/batch-delete')
def batch_delete(body: BatchDeleteBody):
    _forget_if_current(body.ids)
    chat_history.delete_sessions(body.ids)
    return {'ok': True}


@router.delete('/sessions')
async def clear_all():
    # 和「重置 → 历史对话」走同一条路：只删库的话，内存里的对话、在跑的 turn、
    # 子代理都还在，模型清完照样接着做旧事。
    import event
    await event.llm.clear_history('history: clear all')
    return {'ok': True}
