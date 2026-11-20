"""Real canvas renderer + CSS browser regression, with isolated HTTP fixtures.
Run: python agent-core/tests/test_canvas_field_help_browser.py
Requires Playwright and a Chromium browser; MOTUS_TEST_BROWSER may select Edge.
Preview only: append --serve (127.0.0.1:15680). No backend or database is used.
"""
import functools
import json
import os
from pathlib import Path
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import sys
import threading
import unittest

WEB = Path(__file__).resolve().parents[1] / 'web'
EN = 'Forward velocity vx in metres per second. Allowed range: -0.5 to 0.5. Use a small value when testing; duration is measured in seconds.\nThis is the complete second line.'
ZH = '持续时间 duration 的单位为秒，建议取值范围为 0.1 至 30。此说明用于帮助用户了解参数含义、数值范围以及使用方法，不会改变参数的实际行为。请先检查当前动作需要的参数，再输入合适的数值；切换动作之后，只显示该动作所需的配置字段。\n这是第二行中文说明，应完整换行展示，不应被卡片边框或页面边缘裁切。'
PROPS = {
    'action': {'enum': ['move', 'wait'], 'description': 'Choose an action.'},
    'vx': {'type': 'number', 'description': EN},
    'duration': {'type': 'number', 'description': ZH},
    'vy': {'type': 'number', 'description': 'Lateral velocity: -0.3 to 0.3 m/s.'},
    'note': {'type': 'string'},
    'file': {'type': 'string', 'format': 'file', 'description': 'Choose a local file.'},
    'unsafe': {'type': 'string', 'description': '<img src=x onerror="window.injected=true"> & "quoted"'},
}
TOOLS = [{'name': kind + '_example', 'type': kind, 'inputSchema': {'properties': PROPS}} for kind in ['actuator', 'sensor', 'processor']]
TOOLS[0]['inputSchema']['x-action-params'] = {'move': {'params': list(PROPS)[1:]}, 'wait': {'params': ['duration']}}
BOOT = '''
export function mountHelpFixture(tools) {
  _canvasEl = document.querySelector('#canvas-area');
  _viewport = document.querySelector('#canvas-viewport');
  _isEditor = true;
  _allMcps = [{id:'fixture', tools}];
  _setupFieldTooltip();
  _setupFieldTooltip(); // Initialization is idempotent.
  tools.forEach((tool, index) => {
    _viewport.appendChild(_buildCardEl({id:'card-'+index, mcpId:'fixture', toolName:tool.name,
      driverName:'Local UI fixture', x:24+index*310, y:24}));
  });
}
'''
HTML = '''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Dashboard parameter help — isolated preview</title>
<link rel="stylesheet" href="/css/style.css">
<style>body{display:block;padding:24px;overflow:auto}h1{font-size:22px;margin-bottom:8px}p{color:var(--text-muted);margin-bottom:16px}#canvas-area{position:relative;height:780px;overflow:hidden;border:1px solid var(--border)}#canvas-viewport{position:relative;transform:none}.canvas-card{width:280px}.canvas-card-body-wrap{overflow:hidden}</style>
<h1>Dashboard 参数说明</h1><p>独立本地预览 · 悬停 / Tab / 点击 ⓘ · ↓ 进入说明滚动 · Esc 关闭 · 无机器人或后台连接</p>
<div id="canvas-area"><div id="canvas-viewport"></div></div>
<script type="module">import {mountHelpFixture} from '/js/canvas.js';mountHelpFixture(TOOLS);window.ready=true;</script>
'''.replace('TOOLS', json.dumps(TOOLS, ensure_ascii=False))

class Handler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == '/':
            data, mime = HTML, 'text/html'
        elif self.path == '/js/canvas.js':
            data, mime = (WEB / 'js/canvas.js').read_text(encoding='utf-8') + BOOT, 'text/javascript'
        # Unused 3D monitors import CDN-only three.js; keep this UI test offline.
        elif self.path in ('/js/renderers/skeleton.js', '/js/renderers/pointcloud.js', '/js/renderers/mapping.js'):
            name = {'skeleton': 'Skeleton', 'pointcloud': 'PointCloud', 'mapping': 'Mapping'}[Path(self.path).stem]
            data, mime = 'export const ' + name + 'Renderer = {};', 'text/javascript'
        elif self.path.startswith('/api/'):
            data, mime = '{}', 'application/json'
        else:
            return super().do_GET()
        data = data.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', mime + '; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def server(port=0):
    return ThreadingHTTPServer(('127.0.0.1', port), functools.partial(Handler, directory=str(WEB)))


class FieldHelpBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise unittest.SkipTest("Optional browser test requires Playwright")
        cls.server = server()
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.pw = sync_playwright().start()
        executable = os.environ.get('MOTUS_TEST_BROWSER')
        cls.browser = cls.pw.chromium.launch(**({'executable_path': executable} if executable else {}))
        cls.artifacts = Path(os.environ.get('MOTUS_TEST_ARTIFACTS', str(Path(os.environ.get('TEMP', '/tmp')) / 'motus-dashboard-help')))
        cls.artifacts.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.page = self.browser.new_page(viewport={'width': 1200, 'height': 960})
        self.errors = []
        self.page.on('pageerror', lambda error: (self.errors.append(str(error)), print('PAGE ERROR:', error, flush=True)))
        self.page.goto('http://127.0.0.1:%d/' % self.server.server_port)
        self.page.wait_for_function('window.ready === true', timeout=8000)
        self.tip = self.page.locator('#canvas-field-description')

    def tearDown(self):
        self.page.close()
        self.assertEqual(self.errors, [])

    def field(self, key, card=0):
        return self.page.locator('[data-card-id="card-%d"] .canvas-field' % card).filter(has=self.page.locator('[data-key="%s"]' % key))

    def help(self, key, card=0):
        return self.field(key, card).locator('.canvas-field-help')

    def visible_text(self, text):
        from playwright.sync_api import expect
        expect(self.tip).to_be_visible()
        expect(self.tip).to_have_text(text)

    def test_01_short_and_all_render_paths(self):
        self.page.screenshot(path=str(self.artifacts / 'normal.png'))
        for card in range(3):
            self.help('action', card).hover()
            self.visible_text('Choose an action.')
            self.assertEqual(self.field('note', card).locator('.canvas-field-help').count(), 0)
            self.help('file', card).hover()
            self.visible_text('Choose a local file.')

    def test_02_full_english_and_no_layout_shift(self):
        before = self.page.locator('[data-card-id="card-0"]').bounding_box()
        self.help('vx').hover()
        self.visible_text(EN)
        self.assertEqual(before, self.page.locator('[data-card-id="card-0"]').bounding_box())
        self.assertEqual(self.field('vx').locator('input').get_attribute('placeholder'), 'vx')
        self.page.screenshot(path=str(self.artifacts / 'hover-english.png'))

    def test_03_full_chinese_multiline(self):
        self.help('duration').hover()
        self.visible_text(ZH)
        self.assertGreater(len(ZH), 100)
        self.assertEqual(self.tip.evaluate('(e)=>getComputedStyle(e).whiteSpace'), 'pre-wrap')
        self.page.screenshot(path=str(self.artifacts / 'hover-chinese.png'))

    def test_04_input_and_action_switch(self):
        from playwright.sync_api import expect
        self.field('vx').locator('input').fill('0.25')
        self.help('vx').click()
        self.field('action').locator('select').select_option('wait')
        expect(self.field('vx')).to_be_hidden()
        expect(self.tip).to_be_hidden()
        self.field('action').locator('select').select_option('move')
        expect(self.field('vx').locator('input')).to_have_value('0.25')

    def test_05_keyboard(self):
        from playwright.sync_api import expect
        self.field('action').locator('select').focus()
        self.page.keyboard.press('Tab')
        expect(self.help('vx')).to_be_focused()
        self.visible_text(EN)
        self.page.keyboard.press('Escape')
        expect(self.tip).to_be_hidden()
        self.page.keyboard.press('Tab')
        expect(self.field('vx').locator('input')).to_be_focused()
        self.help('duration').focus()
        self.page.keyboard.press('ArrowDown')
        expect(self.tip).to_be_focused()
        self.page.keyboard.press('Escape')
        expect(self.help('duration')).to_be_focused()
        expect(self.tip).to_be_hidden()

    def test_06_viewport_and_overflow(self):
        for width, height in [(1200,960), (768,600), (375,480), (240,320)]:
            self.page.set_viewport_size({'width':width,'height':height})
            self.help('duration').evaluate('(e)=>{e.style.position="fixed";e.style.right="9px";e.style.bottom="9px"}')
            self.help('duration').hover()
            self.visible_text(ZH)
            self.page.wait_for_timeout(50)
            box = self.tip.bounding_box()
            self.assertGreaterEqual(box['x'], 0)
            self.assertGreaterEqual(box['y'], 0)
            self.assertLessEqual(box['x']+box['width'], width)
            self.assertLessEqual(box['y']+box['height'], height)
            self.assertEqual(self.tip.evaluate('(e)=>e.parentElement.tagName'), 'BODY')
            self.assertTrue(self.tip.evaluate('(e)=>{const r=e.getBoundingClientRect();return e.contains(document.elementFromPoint(r.x+5,r.y+5))}'))

    def test_07_click_dismiss_and_removed_anchor(self):
        from playwright.sync_api import expect
        self.help('vx').click()
        self.page.mouse.move(1100,10)
        self.visible_text(EN)
        self.help('vx').click()
        expect(self.tip).to_be_hidden()
        self.help('vx').click()
        self.page.locator('h1').click()
        expect(self.tip).to_be_hidden()
        self.help('vx').hover()
        self.help('vx').evaluate('(e)=>e.remove()')
        expect(self.tip).to_be_hidden()

    def test_08_html_is_text(self):
        self.help('unsafe').hover()
        self.visible_text(PROPS['unsafe']['description'])
        self.assertEqual(self.tip.locator('img').count(), 0)
        self.assertIsNone(self.page.evaluate('window.injected'))

    def test_10_touch(self):
        from playwright.sync_api import expect
        context = self.browser.new_context(viewport={'width': 390, 'height': 700}, has_touch=True, is_mobile=True)
        page = context.new_page()
        try:
            page.goto('http://127.0.0.1:%d/' % self.server.server_port)
            page.wait_for_function('window.ready === true')
            button = page.locator('[data-card-id="card-0"] .canvas-field-help').nth(1)
            button.tap()
            tip = page.locator('#canvas-field-description')
            expect(tip).to_have_text(EN)
            expect(tip).to_be_visible()
            button.tap()
            expect(tip).to_be_hidden()
        finally:
            context.close()

    def test_09_hover_tooltip_and_long_scroll(self):
        from playwright.sync_api import expect
        self.help('duration').evaluate('(e)=>e.dataset.description="长说明\\n".repeat(600)')
        self.help('duration').hover()
        self.tip.hover()
        self.page.wait_for_timeout(250)
        expect(self.tip).to_be_visible()
        self.tip.evaluate('(e)=>e.scrollTop=1000')
        self.assertGreater(self.tip.evaluate('(e)=>e.scrollTop'), 0)
        self.page.mouse.move(1190,5)
        expect(self.tip).to_be_hidden()


if __name__ == '__main__':
    if '--serve' in sys.argv:
        print('Isolated preview: http://127.0.0.1:15680', flush=True)
        server(15680).serve_forever()
    else:
        unittest.main(verbosity=2)
