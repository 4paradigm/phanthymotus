"""Track the Feishu context that produced a searchable record."""

from contextvars import ContextVar
import json


# Channel, chat, sender, and generation at the start of the originating turn.
FeishuScope = tuple[str, str, str, int]
current_feishu_scope: ContextVar[FeishuScope | None] = ContextVar('feishu_history_scope', default=None)
current_origin_unknown: ContextVar[bool] = ContextVar('history_origin_unknown', default=False)
UNSCOPED = '[]'


def encode(scope: FeishuScope | None) -> str:
    if scope is None:
        return UNSCOPED
    channel_id, _, user_id, generation = scope
    return json.dumps((channel_id, user_id, generation))


def searchable_scopes() -> tuple[str, str]:
    scope = current_feishu_scope.get()
    if scope is not None:
        from channel.manager import manager

        channel_id, chat_id, user_id, generation = scope
        if generation == manager.chat_generation(channel_id, chat_id, user_id):
            return UNSCOPED, encode(scope)
    return UNSCOPED, UNSCOPED


def from_metadata(metadata: dict) -> FeishuScope | None:
    scope = metadata.get('history_feishu_scope')
    if (isinstance(scope, (list, tuple)) and len(scope) == 4
            and all(isinstance(value, str) and value for value in scope[:3])
            and type(scope[3]) is int):
        return scope[0], scope[1], scope[2], scope[3]
    return None


def archived_scope(metadata: dict) -> str | None:
    if metadata.get('history_origin_unknown'):
        return None
    if 'history_feishu_scope' in metadata and metadata['history_feishu_scope'] is None:
        return UNSCOPED
    scope = from_metadata(metadata)
    return encode(scope) if scope is not None else None
