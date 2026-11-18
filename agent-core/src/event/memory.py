import asyncio
import pathlib
import typing

import log


class Event():
    @log.function_(call=True)
    async def channel_memory_remember(self,
        text: typing.Annotated[str, '当前 QQ 私聊或群聊中要记住的事实（最多 500 字）'],
    ):
        """只写入本次已验证 QQ 会话的独立长期记忆。"""
        from channel.manager import manager, qq_memory_scope
        scope = qq_memory_scope.get()
        if scope is None:
            return 'Error: no authorized QQ conversation'
        try:
            added = await asyncio.to_thread(manager.remember_chat_memory, *scope, text)
        except ValueError as exc:
            return f'Error: {exc}'
        return '已记入当前会话的长期记忆。' if added else '当前会话已保存这条记忆。'

    @log.function_(call=True)
    async def channel_memory_read(self):
        """只读取本次 QQ 私聊或群聊的长期记忆。"""
        from channel.manager import manager, qq_memory_scope
        scope = qq_memory_scope.get()
        if scope is None:
            return 'Error: no authorized QQ conversation'
        facts = await asyncio.to_thread(manager.get_chat_memory, *scope)
        return '\n'.join(f'- {item}' for item in facts) if facts else '当前会话没有长期记忆。'

    @log.function_(call=True)
    async def channel_memory_forget(self,
        text: typing.Annotated[str, '要删除的完整记忆文本；留空表示清除当前会话全部长期记忆'] = '',
    ):
        """删除当前 QQ 会话的指定记忆或全部记忆，不影响其他会话。"""
        from channel.manager import manager, qq_memory_scope
        scope = qq_memory_scope.get()
        if scope is None:
            return 'Error: no authorized QQ conversation'
        deleted = await asyncio.to_thread(manager.forget_chat_memory, *scope, text)
        return f'已从当前会话删除 {deleted} 条记忆。'

    @log.function_(call=True)
    async def update(self,
        new_prompt: typing.Annotated[str, '完整的新记忆层文本，将完全替换当前的记忆内容。你需要保留你认为仍然需要的内容，并加入新的内容。'],
    ):
        """更新你的记忆（长期记忆）。调用此工具会永久改变你的身份、行为规则和记忆。请谨慎使用，确保新内容完整包含你仍需要的所有信息。"""
        if not new_prompt.strip():
            return '更新失败：记忆内容不能为空。'
        pathlib.Path('./resource/memory/prompt_memory.md').write_text(new_prompt)
        return f'已更新（共 {len(new_prompt)} 字）。新的记忆将在下一轮对话生效。'
