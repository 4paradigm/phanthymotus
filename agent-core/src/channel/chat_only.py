"""Execution policy for Channels that may only answer their current chat.

A Channel-level policy (``chat_only`` on the channel config), never a platform
one: it narrows what a turn may *do*, not what the model is told. Such turns
share the system prompt, environment and conversation history with every
other turn.
"""


def enabled(channel: dict | None) -> bool:
    return bool(channel and channel.get('chat_only'))


REPLY_TOOL = 'mcp__channel__channel_reply'
REPLY_SCHEMA = {
    'name': REPLY_TOOL,
    'description': ('Send a text reply to the Channel message that triggered this turn. '
                    'Set action="send" and source_message_id to the current message_id; '
                    'finish when done. No other action is permitted.'),
    'parameters': {
        'type': 'object',
        'properties': {'action': {'type': 'string', 'enum': ['send']},
                       'text': {'type': 'string'},
                       'source_message_id': {'type': 'string'}},
        'required': ['action', 'text', 'source_message_id'],
        'additionalProperties': False,
    },
}

async def dispatch(name, args, trigger):
    if name == 'finish' and args == {}:
        return 'Done'
    if (name != REPLY_TOOL or not isinstance(args, dict)
            or set(args) - {'action', 'text', 'source_message_id'}
            or args.get('action') != 'send'
            or not isinstance(args.get('text'), str)
            or args.get('source_message_id') not in trigger['_channel_message_ids']):
        return 'Error: only a text reply to the current chat message is allowed.'
    from channel.manager import manager
    from channel.qq_canvas import reply_card

    channel_id = trigger['_chat_only_channel']
    card = reply_card(channel_id)
    if not card:
        return 'Error: the current channel has no bound reply card.'
    return await manager.send_to_channel(
        channel_id, text=args['text'], source_message_id=args['source_message_id'],
    )
