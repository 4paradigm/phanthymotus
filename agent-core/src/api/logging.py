import typing
import dataclasses
import base64
import json
import asyncio
import fastapi
import fastapi.responses

import logging
logger = logging.getLogger("main")


router = fastapi.APIRouter(prefix="/logging")


def _record_stream(handler):
    """ndjson generator over a LoggingHandler's ring buffer.

    Tracks a monotonic `seq` rather than a list index. The buffer is a bounded
    deque, so once it saturates its length stops growing — an index-based reader
    would silently stop delivering records at that point. Also snapshots with
    list() because iterating a deque that another thread appends to raises
    RuntimeError.
    """
    async def generate():
        last_seq = 0
        while True:
            new_records = [r for r in list(handler.record_list)
                           if getattr(r, 'seq', 0) > last_seq]
            if new_records:
                last_seq = new_records[-1].seq
                for record in new_records:
                    data = {
                        'time': record.created,
                        'level': record.levelname,
                        'name': record.name,
                        'message': record.getMessage(),
                    }
                    yield (json.dumps(data, ensure_ascii=False) + '\n').encode('utf-8')
            await asyncio.sleep(0.1)

    return generate


@router.get("/logging_debug_stream")
async def logging_debug_stream():
    # 从 "main" logger 上定位 start.py 挂载的 LoggingHandler
    handler = logger.handlers[0]
    return fastapi.responses.StreamingResponse(
        _record_stream(handler)(),
        media_type='application/x-ndjson',
    )


@router.get("/logging_show_stream")
async def logging_show_stream():
    # 从 "main" logger 上定位 start.py 挂载的 LoggingHandler
    handler = logger.handlers[0]
    return fastapi.responses.StreamingResponse(
        _record_stream(handler)(),
        media_type='application/x-ndjson',
    )
