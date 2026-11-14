"""Isolated Monitor fixture. Run with --serve for http://127.0.0.1:15681.
Uses real WebSocket binary JPEG messages, no robot or Agent Core backend.
"""
import base64, hashlib, io, json, socket, struct, sys, threading, time
from pathlib import Path
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from PIL import Image, ImageDraw
WEB = Path(__file__).resolve().parents[1] / 'web'
state = {'a': True, 'b': True}
img = Image.new('RGB', (480, 270), '#285b69')
ImageDraw.Draw(img).text((30, 100), 'LOCAL SIMULATED CAMERA - NO HARDWARE', fill='white')
buf=io.BytesIO(); img.save(buf,format='JPEG'); FRAME=buf.getvalue()
BOOT = """
export function fixture() {
 _grid=document.getElementById('monitor-dashboard-grid');
 _createCard('/a','image/jpeg','online',0,0,2,2);
 _createCard('/b','image/jpeg','online',2,0,2,2);
 _createCard('/log','data/json','online',0,2,2,1);
}
"""
HTML = """<!doctype html><meta charset="utf-8"><link rel="stylesheet" href="/css/style.css">
<style>body{display:block;padding:20px}#monitor-dashboard-grid{display:grid;grid-template-columns:repeat(4,1fr);grid-auto-rows:160px;gap:12px;height:700px}button{padding:8px;margin:8px}</style>
<h2>Monitor 本地模拟视频（无后台/无硬件）</h2>
<button onclick="fetch('/control/a/stop')">停止 A 帧（保留心跳）</button>
<button onclick="fetch('/control/a/start')">恢复 A 帧</button>
<button onclick="location.reload()">重建卡片</button>
<div id="monitor-dashboard-grid"></div>
<script type="module">import {fixture,deactivate} from '/js/monitor-dashboard.js';fixture();window.stopMonitor=deactivate;window.ready=true;</script>"""
class Handler(SimpleHTTPRequestHandler):
 def __init__(self,*a,**k):super().__init__(*a,directory=str(WEB),**k)
 def log_message(self,*a):pass
 def do_GET(self):
  if self.path.startswith('/ws/bus/'):
   key=self.path.rsplit('/',1)[-1]
   accept=base64.b64encode(hashlib.sha1((self.headers['Sec-WebSocket-Key']+'258EAFA5-E914-47DA-95CA-C5AB0DC85B11').encode()).digest()).decode()
   self.send_response(101);self.send_header('Upgrade','websocket');self.send_header('Connection','Upgrade');self.send_header('Sec-WebSocket-Accept',accept);self.end_headers()
   try:
    for _ in range(3600):
     data=(b'corrupt-jpeg' if state.get(key)=='bad' else FRAME) if state.get(key,False) else b'{"type":"ping"}'
     opcode=2 if state.get(key,False) else 1
     if key=='log':data=b'{"value":42}';opcode=1
     header=bytes([128|opcode,len(data)]) if len(data)<126 else bytes([128|opcode,126])+struct.pack('!H',len(data))
     self.wfile.write(header+data);self.wfile.flush();time.sleep(1)
   except (OSError,ConnectionError):pass
   self.close_connection=True;return
  if self.path.startswith('/control/'):
   _,_,key,action=self.path.split('/');state[key]=action if action=='bad' else action=='start';data=b'ok';mime='text/plain'
  elif self.path=='/':data=HTML.encode();mime='text/html'
  elif self.path=='/js/monitor-dashboard.js':data=((WEB/'js/monitor-dashboard.js').read_text(encoding='utf-8')+BOOT).encode();mime='text/javascript'
  elif self.path in ['/js/renderers/'+x+'.js' for x in ['mapping','skeleton','pointcloud']]:
   name={'mapping':'Mapping','skeleton':'Skeleton','pointcloud':'PointCloud'}[Path(self.path).stem];data=('export const '+name+'Renderer={canRender:()=>false};').encode();mime='text/javascript'
  else:return super().do_GET()
  self.send_response(200);self.send_header('Content-Type',mime+'; charset=utf-8');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
def browser_test():
 from playwright.sync_api import sync_playwright, expect
 import os
 artifacts=Path(os.environ.get('TEMP','.'))/'monitor-stale-evidence';artifacts.mkdir(exist_ok=True)
 with sync_playwright() as p:
  browser=p.chromium.launch(executable_path=os.environ.get('MOTUS_TEST_BROWSER',r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'))
  page=browser.new_page(viewport={'width':1200,'height':900})
  page.request.get('http://127.0.0.1:15681/control/a/stop')
  page.goto('http://127.0.0.1:15681/');page.wait_for_function('window.ready')
  a=page.locator('[data-topic="/a"]');b=page.locator('[data-topic="/b"]')
  page.wait_for_timeout(11000)
  expect(a.locator('.monitor-card-empty')).to_be_visible()
  expect(a.locator('.monitor-video-stale')).to_be_hidden()
  print('PASS before first frame and heartbeats')
  page.request.get('http://127.0.0.1:15681/control/a/start')
  page.wait_for_timeout(2000)
  page.screenshot(path=str(artifacts/'normal.png'))
  page.request.get('http://127.0.0.1:15681/control/a/stop')
  page.wait_for_timeout(11000)
  expect(a.locator('.monitor-video-stale')).to_be_visible()
  expect(b.locator('.monitor-video-stale')).to_be_hidden()
  assert a.locator('img').evaluate('(e)=>e.naturalWidth')>0
  assert page.locator('[data-topic="/log"] .monitor-video-stale').count()==0
  page.screenshot(path=str(artifacts/'stale.png'))
  print('PASS last frame retained, independent cards, heartbeat does not renew, JSON unaffected')
  page.request.get('http://127.0.0.1:15681/control/a/start')
  expect(a.locator('.monitor-video-stale')).to_be_hidden(timeout=5000)
  print('PASS recovery')
  page.request.get('http://127.0.0.1:15681/control/a/bad')
  page.wait_for_timeout(11000)
  expect(a.locator('.monitor-video-stale')).to_be_visible()
  assert a.locator('img').evaluate('(e)=>e.naturalWidth')>0
  page.request.get('http://127.0.0.1:15681/control/a/start')
  expect(a.locator('.monitor-video-stale')).to_be_hidden(timeout=5000)
  print('PASS corrupt frames retain last image, do not renew, then recover')
  page.evaluate('window.stopMonitor()')
  page.wait_for_timeout(11000)
  assert page.locator('.monitor-card').count()==0
  page.reload();page.wait_for_function('window.ready')
  page.wait_for_timeout(2000)
  expect(page.locator('[data-topic="/a"] .monitor-video-stale')).to_be_hidden()
  print('PASS deactivate and rebuild; screenshots:',artifacts)
  browser.close()

if __name__=='__main__' and '--test' in sys.argv:
 browser_test();raise SystemExit
if __name__=='__main__':
 server=ThreadingHTTPServer(('127.0.0.1',15681),Handler)
 print('http://127.0.0.1:15681/',flush=True)
 server.serve_forever()
