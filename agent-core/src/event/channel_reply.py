"""
event/channel_reply.py — channel_reply system tool.

When a trigger comes from a messaging channel, the LLM can call this tool
to reply to the user. The LLM decides whether and what to reply
(rather than forcibly sending all output back to the channel).
"""

import typing

import log


class Tools:
    @log.function_(call=True)
    async def channel_reply(self,
        text: typing.Annotated[str, 'Reply text to send to the channel user'],
    ):
        """Send a reply to the user who triggered this event via a messaging channel. Only available when the current event originates from a channel (channel:*). Use this tool to actively choose what to reply to the user."""
        if not text.strip():
            return 'Reply text cannot be empty'

        # Actual send logic is handled in llm.py's _dispatch_channel_reply
        # This return is just a confirmation placeholder
        return f'Reply sent ({len(text)} chars)'
