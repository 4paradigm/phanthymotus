"""Actual offline preview interactions in local Chromium; no production Canvas."""
import argparse
import json
import os
from pathlib import Path
import threading
from plugins.face_effects_live.server import make_server


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--evidence',required=True)
    parser.add_argument('--browser-libs',help='optional isolated runtime library directory')
    args=parser.parse_args()
    if args.browser_libs:
        os.environ['LD_LIBRARY_PATH']=os.pathsep.join(filter(None,[args.browser_libs,os.environ.get('LD_LIBRARY_PATH')]))
    from playwright.sync_api import sync_playwright
    root=Path(args.evidence)
    cfg=json.loads((root/'local-config.json').read_text())
    server,plugin=make_server(cfg)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    errors=[]
    try:
        with sync_playwright() as p:
            env=dict(os.environ)
            if (root/'fonts.conf').exists():
                env['FONTCONFIG_FILE']=str(root/'fonts.conf')
            browser=p.chromium.launch(headless=True,env=env)
            page=browser.new_page(viewport={'width':1280,'height':1100})
            page.on('pageerror',lambda exc:errors.append(str(exc)))
            page.goto(f'http://127.0.0.1:{server.server_port}',wait_until='domcontentloaded')
            page.fill('#path',str(root/'fixtures/synthetic-720p30.mp4'))
            page.click('#start')
            page.wait_for_function("()=>{try{return JSON.parse(document.getElementById('status').textContent).result?.tracked===1}catch{return false}}",timeout=15000)
            page.wait_for_function("()=>document.getElementById('view').getContext('2d').getImageData(620,300,1,1).data[3]>0")
            dimensions=page.locator('#view').evaluate('(c)=>[c.width,c.height]')
            assert dimensions==[1280,720]
            page.screenshot(path=str(root/'canvas-preview.png'),full_page=True)
            page.select_option('#effect','glasses')
            page.wait_for_function("()=>{try{return JSON.parse(document.getElementById('status').textContent).result?.effects.join(',')==='glasses'}catch{return false}}")
            page.select_option('#effect','none')
            page.wait_for_function("()=>{try{return JSON.parse(document.getElementById('status').textContent).result?.effects.length===0}catch{return false}}")
            status=json.loads(page.locator('#status').text_content())
            page.click('#stop')
            page.wait_for_function("()=>document.getElementById('notice').textContent==='已停止'")
            page.wait_for_function("()=>document.getElementById('view').getContext('2d').getImageData(620,300,1,1).data[3]===0")
            assert not errors
            report={'ok':True,'browser':browser.version,'dimensions':dimensions,
                    'checks':['actual HTML/JS loaded','file source start','decoded JPEG painted to Canvas',
                              'three demo effects active','switch to glasses','disable effects','stop clears Canvas'],
                    'page_errors':errors,'status_sample':status,
                    'limits':'standalone offline preview; production Core ROS2 Canvas not validated'}
            (root/'browser-acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps({k:v for k,v in report.items() if k!='status_sample'},indent=2))
            browser.close()
    finally:
        plugin.shutdown();server.shutdown();server.server_close();thread.join(2)


if __name__=='__main__':main()
