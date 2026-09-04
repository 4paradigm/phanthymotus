import typing

import agent_memory
import log


class Event():
    @log.function_(call=True)
    async def update(self,
        new_prompt: typing.Annotated[str, '完整的新记忆层文本，将完全替换当前的记忆内容。你需要保留你认为仍然需要的内容，并加入新的内容。'],
    ):
        """更新你的记忆（长期记忆）。调用此工具会永久改变你的身份、行为规则和记忆。请谨慎使用，确保新内容完整包含你仍需要的所有信息。"""
        try:
            snapshot = await agent_memory.replace(
                new_prompt,
                actor_key='agent:main',
                reason='llm_update',
            )
        except agent_memory.AgentMemoryValidationError:
            return '更新失败：记忆内容不能为空。'
        except agent_memory.AgentMemoryCommitUncertainError:
            return '更新结果无法确认：请检查长期记忆存储状态后再操作。'
        except agent_memory.AgentMemoryError:
            return '更新失败：长期记忆存储暂不可用，原有记忆已保留。'
        result = (
            f'已更新（共 {len(snapshot.text)} 字，revision={snapshot.revision or "file"}）。'
            '新的记忆将在下一轮对话生效。'
        )
        if not snapshot.fallback_ready:
            result += agent_memory.COMPATIBILITY_WARNING
        return result
