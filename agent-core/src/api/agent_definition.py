"""
api/agent_definition.py — 智能体定义编辑 API。

提供 identity.md、prompt_system.md 和 Memory Core 长期记忆的读写接口，供前端 modal 使用。
"""

import pathlib

import fastapi
import pydantic

import agent_memory
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
    memory = agent_memory.snapshot()
    return {
        'code': 200,
        'data': {
            'identity': identity,
            'system': system,
            'memory': memory.text,
            'memoryStatus': agent_memory.status(memory),
        },
    }


@router.post('/definition')
async def save_definition(req: DefinitionSaveRequest):
    try:
        snapshot = await agent_memory.replace(
            req.memory,
            actor_key='api:agent_definition',
            reason='api_edit',
        )
    except agent_memory.AgentMemoryValidationError as error:
        raise fastapi.HTTPException(status_code=422, detail='长期记忆内容不能为空') from error
    except agent_memory.AgentMemoryCommitUncertainError as error:
        raise fastapi.HTTPException(
            status_code=503,
            detail='长期记忆写入结果无法确认，请检查存储状态',
        ) from error
    except agent_memory.AgentMemoryError as error:
        raise fastapi.HTTPException(status_code=503, detail='长期记忆存储暂不可用') from error

    _IDENTITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _IDENTITY_PATH.write_text(req.identity)
    _SYSTEM_PATH.write_text(req.system)
    warning = None if snapshot.fallback_ready else agent_memory.COMPATIBILITY_WARNING
    return {
        'code': 200,
        'message': '已保存',
        'warning': warning,
        'memoryStatus': {
            'backend': snapshot.backend,
            'revision': snapshot.revision,
            'fallback_ready': snapshot.fallback_ready,
            'warning': warning,
        },
    }
