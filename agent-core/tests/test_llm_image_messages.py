"""Image tool outputs must preserve Chat Completions tool-result ordering."""

import copy
import importlib

import pytest


llm = importlib.import_module('client.llm')


@pytest.mark.parametrize('image_url', [
    'data:image/png;base64,cGl4ZWw=',
    {'url': 'data:image/png;base64,cGl4ZWw='},
])
def test_image_outputs_follow_the_complete_tool_batch(image_url):
    messages = [
        {'role': 'assistant', 'tool_calls': [
            {'id': 'read-image', 'type': 'function',
             'function': {'name': 'Read', 'arguments': '{}'}},
            {'id': 'read-status', 'type': 'function',
             'function': {'name': 'mcp__sensor__status', 'arguments': '{}'}},
        ]},
        {'role': 'tool', 'tool_call_id': 'read-image', 'content': [
            {'type': 'text', 'text': 'User attachment'},
            {'type': 'image_url', 'image_url': image_url},
        ]},
        {'role': 'tool', 'tool_call_id': 'read-status', 'content': 'Battery 80%'},
        {'role': 'assistant', 'content': 'Image description'},
    ]
    original = copy.deepcopy(messages)

    prepared = llm._prepare_messages(messages)

    assert [item['role'] for item in prepared] == ['assistant', 'tool', 'tool', 'user', 'assistant']
    assert [item['tool_call_id'] for item in prepared[1:3]] == ['read-image', 'read-status']
    assert prepared[1]['content'] == [{'type': 'text', 'text': 'User attachment'}]
    assert prepared[2]['content'] == 'Battery 80%'
    assert prepared[3]['content'][-1] == {
        'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,cGl4ZWw='},
    }
    assert messages == original
