"""Display pacing, bounded work and cancellation; no robot or authority path."""
import asyncio
import threading
import time
import pytest
from teleop.capture_server import visualization_stream


def test_slow_fk_does_not_block_event_loop_or_spawn_backlog():
    async def run():
        lock=threading.Lock();counts={'active':0,'max':0,'calls':0};ticks=[]
        def provider():
            with lock:
                counts['active']+=1;counts['max']=max(counts['max'],counts['active']);counts['calls']+=1
            time.sleep(.08)
            with lock:counts['active']-=1
            return {'sequence':counts['calls']}
        class Socket:
            closed=False
            values=[]
            async def send_json(self,value):
                self.values.append(value)
                if len(self.values)==3:self.closed=True
        ws=Socket()
        task=asyncio.create_task(visualization_stream(ws,provider))
        while not task.done():
            ticks.append(time.monotonic());await asyncio.sleep(.005)
        await task
        assert counts['max']==1 and counts['calls']==3
        assert len(ticks)>20
        assert [x['visualization']['sequence'] for x in ws.values]==[1,2,3]
    asyncio.run(run())


def test_backpressured_socket_stops_display_without_replay():
    async def run():
        class Socket:
            closed=False
            calls=0
            async def send_json(self,value):
                self.calls+=1;await asyncio.sleep(1)
        ws=Socket();await visualization_stream(ws,lambda:{'available':True})
        assert ws.calls==1
    asyncio.run(run())


def test_provider_failure_is_diagnostic_and_cancellation_propagates():
    async def run():
        class Socket:
            closed=False
            values=[]
            async def send_json(self,value):self.values.append(value)
        ws=Socket()
        def fail():raise ValueError('bad FK')
        task=asyncio.create_task(visualization_stream(ws,fail))
        while not ws.values:await asyncio.sleep(.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):await task
        assert ws.values[0]['visualization']['reason']=='visualization_unavailable'
    asyncio.run(run())
