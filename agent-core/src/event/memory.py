import pathlib
import typing

import log


class Event():
    @log.function_(call=True)
    async def update(self,
        new_prompt: typing.Annotated[str, '完整的新记忆层文本，将完全替换当前的记忆内容。你需要保留你认为仍然需要的内容，并加入新的内容。'],
    ):
        """更新你的记忆（长期记忆）。调用此工具会永久改变你的身份、行为规则和记忆。请谨慎使用，确保新内容完整包含你仍需要的所有信息。"""
        if not new_prompt.strip():
            return '更新失败：记忆内容不能为空。'
        pathlib.Path('./resource/memory/prompt_memory.md').write_text(new_prompt)
        return f'已更新（共 {len(new_prompt)} 字）。新的记忆将在下一轮对话生效。'
