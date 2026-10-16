"""Operator-only setup. End-user recall is exclusively through scoped LLM tools."""
from typing import Literal

import fastapi
from pydantic import BaseModel, Field

import auth
import personal_memory as memory


def _operator(request: fastapi.Request):
    # Unlike ordinary dashboard routes, these endpoints never run without auth.
    if not auth.is_enabled():
        raise fastapi.HTTPException(503, '请先配置 ACCESS_TOKEN 并重启服务。')
    if not auth.verify(auth._extract_token(request)):
        raise fastapi.HTTPException(401, '需要运营者认证。')


router = fastapi.APIRouter(prefix='/personal-memory', tags=['personal-memory'],
                          dependencies=[fastapi.Depends(_operator)])


class UserCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    remember: bool = False
    onboarding: bool = False


class Preferences(BaseModel):
    remember: bool
    onboarding: bool


class Binding(BaseModel):
    modality: Literal['face', 'voice']
    external_id: str = Field(min_length=1, max_length=200)


def _call(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except (ValueError, PermissionError) as error:
        raise fastapi.HTTPException(400, str(error)) from error


@router.get('/users')
def users():
    return memory.list_users()


@router.post('/users')
def create_user(body: UserCreate):
    return _call(memory.create_user, body.name, remember=body.remember, onboarding=body.onboarding)


@router.put('/users/{user_id}/preferences')
def preferences(user_id: str, body: Preferences):
    return _call(memory.update_user, user_id, body.remember, body.onboarding)


@router.post('/users/{user_id}/bindings')
def bind(user_id: str, body: Binding):
    _call(memory.bind_identity, user_id, body.modality, body.external_id)
    return {'ok': True}


@router.delete('/bindings/{modality}/{external_id}')
def unbind(modality: Literal['face', 'voice'], external_id: str):
    memory.unbind_identity(modality, external_id)
    return {'ok': True}


@router.delete('/users/{user_id}')
def delete_user(user_id: str):
    memory.delete_user(user_id)
    return {'ok': True}


@router.get('/public-candidates')
def candidates():
    return memory.public_candidates()


@router.post('/public-candidates/{memory_id}/approve')
def approve(memory_id: str):
    _call(memory.approve_public, memory_id)
    return {'ok': True}


@router.delete('/public-candidates/{memory_id}')
def delete_public(memory_id: str):
    memory.delete_public(memory_id)
    return {'ok': True}
