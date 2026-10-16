"""Immutable built-in skill; source files ship with the application image."""
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def personal_memory_skill() -> dict:
    instruction = (Path(__file__).parent / 'builtin_skills' / 'personal-memory' / 'SKILL.md').read_text()
    return {'slug': 'personal-memory', 'name': '个人记忆管理',
            'oneLiner': '按本人授权分类、记忆和管理个人资料', 'active': True,
            'instruction': instruction, 'version': '1.0.0'}
