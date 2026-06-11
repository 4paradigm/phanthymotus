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
    messages = chat_history.get_session_messages(session_id)
    return {'session_id': session_id, 'messages': messages}


@router.delete('/sessions/{session_id}')
def delete_session(session_id: str):
    chat_history.delete_session(session_id)
    return {'ok': True}


class BatchDeleteBody(BaseModel):
    ids: list[str]


@router.post('/sessions/batch-delete')
def batch_delete(body: BatchDeleteBody):
    chat_history.delete_sessions(body.ids)
    return {'ok': True}


@router.delete('/sessions')
def clear_all():
    chat_history.clear_all()
    return {'ok': True}
