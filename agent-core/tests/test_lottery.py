"""The display and the tool caller must receive one authoritative winner."""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import lottery  # noqa: E402


class LotteryConsistencyTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        lottery._state = {
            'state': 'idle', 'mode': 'dice',
            'options': list(lottery.DEFAULT_OPTIONS),
        }

    async def test_result_frame_and_tool_return_are_the_same_decision(self):
        published = []

        async def capture(payload):
            published.append(dict(payload))

        with mock.patch.object(lottery, '_settings', return_value=(
                'options', ['Alice', 'Bob', 'Carol'], 500)), \
                mock.patch.object(lottery, '_publish', side_effect=capture), \
                mock.patch.object(lottery.asyncio, 'sleep', new=mock.AsyncMock()), \
                mock.patch.object(lottery.secrets, 'randbelow', return_value=1), \
                mock.patch.object(lottery.secrets, 'randbits', return_value=12345):
            result = await lottery.handle({'action': 'draw'})

        self.assertEqual([event['state'] for event in published],
                         ['spinning', 'result'])
        self.assertNotIn('result', published[0], 'winner leaked before reveal')
        self.assertEqual(published[1], result)
        self.assertEqual(result['result'], 'Bob')
        self.assertEqual(result['result_index'], 1)

    async def test_overlapping_draw_returns_busy_instead_of_second_winner(self):
        await lottery._draw_lock.acquire()
        try:
            result = await lottery.handle({'action': 'draw'})
        finally:
            lottery._draw_lock.release()
        self.assertEqual(result['state'], 'busy')


if __name__ == '__main__':
    unittest.main()
