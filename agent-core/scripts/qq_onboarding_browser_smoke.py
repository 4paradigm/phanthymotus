"""Verify QQ onboarding in the ORIGINAL Agent Core index.html and Channels UI.

Only the QQ Gateway/LLM and heavyweight full-app bootstrap are simulated.
Actual web/index.html, web/js/channels.js, setup modules and Channel routers run.
SQLite is isolated to a temporary database. This is NOT a separate product UI.
"""
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from playwright.sync_api import sync_playwright
import uvicorn


def run():
    with tempfile.TemporaryDirectory(prefix='qq-native-browser-', ignore_cleanup_errors=True) as tmp:
        os.environ['DB_PATH'] = str(Path(tmp)/'data.db')
        from api import channel, channel_setup
        from channel.manager import manager
        from channel.qq_onboarding import receive as qq_pairing_receive
        import auth
        auth.is_enabled = lambda: True

        async def simulated_gateway_restart(cid, **kwargs):
            manager._health[cid] = (True, 'gateway-simulated-for-test')
        manager.restart_adapter = simulated_gateway_restart

        app = FastAPI()
        app.include_router(channel.router, prefix='/api')
        app.include_router(channel_setup.router, prefix='/api')

        @app.post('/test/qq-message')
        def fake_sdk_event(payload:dict):
            return {'accepted':qq_pairing_receive(SimpleNamespace(
                platform='qq', chat_type='p2p', sender_type='user',
                channel_id='qq_official_1900000000',
                user_id='official:1900000000:ABCDEFGHIJKLMNOP940FBB',
                text='绑定 ' + str(payload.get('code','')),
            ))}

        @app.post('/api/config/start-project')
        def fake_llm(): return {'ok':True, 'mock_llm':True}

        app.mount('/', StaticFiles(directory=str(ROOT/'web'), html=True), name='native-dashboard')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(
            app, host='127.0.0.1', port=port, log_level='error', lifespan='off'))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        for _ in range(80):
            if server.started:
                break
            time.sleep(.1)
        assert server.started

        n = 0
        def check(value, label):
            nonlocal n
            assert value, label
            n+=1
            print('PASS:',label)

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page(viewport={'width':1440,'height':900})
                js_errors = []
                page.on('pageerror', lambda exc:js_errors.append(str(exc)))

                # Heavy app bootstrap uses ROS/motus/three. Replace ONLY the
                # app bootstrap in this unit-level browser test; actual Channels
                # and QQ modules are served unchanged from original repository.
                page.route('**/js/app.js*', lambda route:route.fulfill(
                    status=200,content_type='application/javascript',
                    body=("import { initChannels } from '/js/channels.js';"
                          "document.getElementById('app').style.display='';"
                          "initChannels();")))
                page.goto(f'http://127.0.0.1:{port}/',wait_until='domcontentloaded')
                check(page.title()=='PhanthyMotus','Original Agent Core Dashboard HTML loaded')
                page.evaluate("document.getElementById('btn-channels').click()")
                page.locator('#channel-overlay:not(.hidden)').wait_for(timeout=8000)
                check(page.locator('#channel-add-btn').is_visible(),'Original Channels modal opened')
                page.locator('#channel-add-btn').click()
                page.locator('#channel-form-platform').select_option('qq')
                wizard=page.locator('#qq-guided-setup')
                wizard.wait_for(state='visible')
                check('步骤 1 / 5' in wizard.inner_text(),'QQ wizard embedded inside original modal')
                check(wizard.locator('a[href="https://q.qq.com/"]').count()==1,'Official QQ signup link')

                page.locator('[data-qq-next]').click()
                page.locator('[data-qq-appid]').fill('1900000000')
                page.locator('[data-qq-secret]').fill('x'*32)
                page.locator('[data-qq-connect]').click()
                page.locator('[data-qq-to-pair]').wait_for(state='visible',timeout=9000)
                check('已连接' in page.locator('[data-qq-connection-result]').inner_text(),'Gateway status in QQ wizard')
                check('已配置' in page.locator('[data-qq-flow-result]').inner_text(),'Default Canvas created')
                check(page.locator('[data-qq-secret]').input_value()=='','Credential input cleared')

                page.locator('[data-qq-to-pair]').click()
                page.locator('[data-qq-pair-start]').click()
                page.locator('[data-qq-pair-check]').wait_for(state='visible')
                code=page.locator('[data-qq-pair-code]').inner_text().strip().split()[-1]
                import httpx
                accepted=httpx.post(f'http://127.0.0.1:{port}/test/qq-message',
                                    json={'code':code},timeout=8).json()['accepted']
                check(accepted,'SDK-like verified QQ owner message processed')
                page.locator('[data-qq-pair-check]').click()
                page.locator('[data-qq-pair-confirm]').wait_for(state='visible')
                page.locator('[data-qq-pair-confirm]').click()
                page.locator('[data-qq-save-policy]').wait_for(state='visible')
                check('步骤 4 / 5' in wizard.inner_text(),'Admin binding confirmed')

                check(page.locator('[name="qq-access-policy"]').count()==0,
                      'Private QQ approval controls removed')
                check(page.locator('[name="qq-default-permission"]').count()==0,
                      'Ordinary QQ users have fixed chat-only permission')
                page.locator('[data-qq-save-policy]').click()
                page.locator('[data-qq-start-project]').wait_for(state='visible')
                page.locator('[data-qq-recheck]').click()
                page.wait_for_timeout(200)
                check('已绑定' in page.locator('[data-qq-diagnostics]').inner_text(),
                      'Original dashboard diagnostics shows admin bound')
                page.locator('[data-qq-finish]').click()
                page.locator('#qq-guided-setup').wait_for(state='detached')
                page.locator('#channel-list .channel-item').first.wait_for(timeout=5000)
                check(page.locator('#channel-list .channel-item').count()==1,
                      'Original channel list refreshes to show configured QQ')
                check(page.locator('#qq-management').count()==1,
                      'QQ permissions management embedded under original Users')
                check(not js_errors,'No JavaScript errors: '+repr(js_errors))
                print('RESULT',n,'checks passed')
                browser.close()
        finally:
            server.should_exit=True
            thread.join(timeout=5)


if __name__=='__main__':
    run()
