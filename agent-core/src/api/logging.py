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


@router.get("/logging_debug_stream")
async def endpoint():
    # 从 "main" logger 上定位 start.py 挂载的 LoggingHandler
    handler = logger.handlers[0]

    async def generate():
        last_index = 0
        while True:
            current_len = len(handler.record_list)
            if current_len > last_index:
                new_records = handler.record_list[last_index:current_len]
                last_index = current_len
                for record in new_records:
                    data = {
                        'time': record.created,
                        'level': record.levelname,
                        'name': record.name,
                        'message': record.getMessage(),
                    }
                    yield (json.dumps(data, ensure_ascii=False) + '\n').encode('utf-8')
            await asyncio.sleep(0.1)

    return fastapi.responses.StreamingResponse(
        generate(),
        media_type='application/x-ndjson',
    )



@router.get("/logging_show_stream")
async def endpoint():
    # 从 "main" logger 上定位 start.py 挂载的 LoggingHandler
    handler = logger.handlers[0]

    async def generate():
        last_index = 0
        while True:
            current_len = len(handler.record_list)
            if current_len > last_index:
                new_records = handler.record_list[last_index:current_len]
                last_index = current_len
                for record in new_records:
                    data = {
                        'time': record.created,
                        'level': record.levelname,
                        'name': record.name,
                        'message': record.getMessage(),
                    }
                    yield (json.dumps(data, ensure_ascii=False) + '\n').encode('utf-8')
            await asyncio.sleep(0.1)

    return fastapi.responses.StreamingResponse(
        generate(),
        media_type='application/x-ndjson',
    )
