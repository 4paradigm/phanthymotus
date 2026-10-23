"""A failed initial DDS subscription must remain retryable on registration."""
import asyncio
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from api import inspection


@pytest.mark.parametrize('failure', [False, RuntimeError('DDS unavailable')])
def test_registration_retries_failed_subscription(failure):
    topic = '/navigation/test/map'

    async def exercise():
        with patch.object(inspection, '_topic_registry', {}), \
                patch.object(inspection, '_active_primary_subs', set()), \
                patch.object(inspection.ros2_bridge, 'subscribe',
                             side_effect=[failure, True]) as subscribe:
            if isinstance(failure, Exception):
                with pytest.raises(RuntimeError, match='DDS unavailable'):
                    await inspection.register_topic_internal(topic, 'sensor/occupancy-grid', 'navigation')
            else:
                await inspection.register_topic_internal(topic, 'sensor/occupancy-grid', 'navigation')
            assert topic not in inspection._active_primary_subs

            await inspection.register_topic_internal(topic, 'sensor/occupancy-grid', 'navigation')
            assert topic in inspection._active_primary_subs
            await inspection.register_topic_internal(topic, 'sensor/occupancy-grid', 'navigation')
            assert subscribe.call_count == 2
            assert subscribe.call_args.args[:3] == (
                f'__primary__#{topic}', topic, 'sensor/occupancy-grid')

    asyncio.run(exercise())
