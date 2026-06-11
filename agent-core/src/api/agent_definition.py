"""
api/agent_definition.py — 智能体定义编辑 API。

提供 identity.md、prompt_system.md、prompt_memory.md 的读写接口，供前端 modal 使用。
"""

import pathlib

import fastapi
import pydantic

import config


router = fastapi.APIRouter(prefix='/agent', tags=['agent'])


_IDENTITY_PATH = pathlib.Path('./resource/memory/identity.md')
_SYSTEM_PATH = pathlib.Path('./resource/memory/prompt_system.md')


def _memory_path() -> pathlib.Path:
    return pathlib.Path(config.main.get('event', {}).get('llm', {}).get(
        'prompt_memory', './resource/memory/prompt_memory.md'))


class DefinitionSaveRequest(pydantic.BaseModel):
    identity: str = ''
    system: str = ''
    memory: str = ''


@router.get('/definition')
async def get_definition():
    identity = _IDENTITY_PATH.read_text() if _IDENTITY_PATH.exists() else ''
    system = _SYSTEM_PATH.read_text() if _SYSTEM_PATH.exists() else ''
    mem_path = _memory_path()
    memory = mem_path.read_text() if mem_path.exists() else ''
    return {'code': 200, 'data': {'identity': identity, 'system': system, 'memory': memory}}


@router.post('/definition')
async def save_definition(req: DefinitionSaveRequest):
    _IDENTITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _IDENTITY_PATH.write_text(req.identity)
    _SYSTEM_PATH.write_text(req.system)
    _memory_path().write_text(req.memory)
    return {'code': 200, 'message': '已保存'}
