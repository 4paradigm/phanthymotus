"""Read-only OneBot v11 login probe. Does not print QQ account ID or tokens.

Run inside a network-host container with /run/qq-token mounted read-only.
"""
import asyncio
import json
from pathlib import Path

import aiohttp


async def main():
    token = Path('/run/qq-token').read_text(encoding='utf-8').strip()
    assert len(token) >= 32, 'Missing OneBot token'
    timeout = aiohttp.ClientTimeout(total=12)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        try:
            async with s.ws_connect('ws://127.0.0.1:3001/',
                                    headers={'Authorization': 'Bearer ' + token},
                                    max_msg_size=65536) as ws:
                await ws.send_json({'action': 'get_login_info',
                                    'params': {}, 'echo': 'qq-health-probe'})
                for _ in range(15):
                    frame = await asyncio.wait_for(ws.receive(), timeout=5)
                    if frame.type != aiohttp.WSMsgType.TEXT:
                        break
                    data = json.loads(frame.data)
                    if data.get('echo') != 'qq-health-probe':
                        continue
                    ok = data.get('status') == 'ok' and data.get('retcode', 0) == 0
                    uid = (data.get('data') or {}).get('user_id')
                    print('onebot_api_ok:', bool(ok))
                    print('qq_logged_in:', bool(ok and str(uid or '').isdigit()))
                    print('account_id_redacted:', True)
                    return
                print('onebot_api_ok: false (no matching response)')
        except Exception as e:
            print('onebot_api_ok: false')
            print('error_type:', type(e).__name__)


if __name__ == '__main__':
    asyncio.run(main())
