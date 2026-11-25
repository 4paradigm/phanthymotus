"""`control/*` must reach ControlRenderer, in every view that picks one.

There is no build step, and a renderer that exists but is never reachable fails
silently and looks exactly like the feature not being written: the topic falls
through to ActivityRenderer or the KV panel and shows a `values` array as one
long line of numbers.

The renderer list itself now lives in `web/js/renderers/registry.js` — three views
each used to keep their own array, and the one that gets forgotten was the one
nobody had open while testing. This file pins the seam from the view side: each
view must route its picks through `renderers/registry.js` (`pickRenderer`), so a
view regressing to a private array — or a renderer added only to a view's old
list — fails here. Ordering within the find list and channel declarations are
pinned in `web/js/renderer-registry.test.mjs`.

Run: cd agent-core && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_control_renderer_registered.py
"""
import pathlib
import re
import unittest

WEB = pathlib.Path(__file__).resolve().parents[1] / 'web' / 'js'
VIEWS = ('detail-panel.js', 'dashboard.js', 'monitor-dashboard.js')


class ControlRendererReachability(unittest.TestCase):
    def test_every_view_routes_through_the_registry(self):
        for view in VIEWS:
            source = (WEB / view).read_text()
            with self.subTest(view=view):
                self.assertIn('renderers/registry.js', source,
                              f'{view} 不再从 renderers/registry.js 取渲染器——'
                              f'渲染器清单只允许存在于 registry.js 一处')
                self.assertIn('pickRenderer(', source,
                              f'{view} 没有走 pickRenderer —— 私有 find/列表会让'
                              f'某张卡只在被打开过的视图里能渲染')

    def test_registry_actually_lists_control(self):
        source = (WEB / 'renderers' / 'registry.js').read_text()
        self.assertIn('ControlRenderer', source,
                      '注册表里没有 ControlRenderer')

    def test_it_claims_the_whole_control_family_and_nothing_else(self):
        source = (WEB / 'renderers' / 'control.js').read_text()
        match = re.search(r"canRender:\s*\(hint\)\s*=>(.*)", source)
        self.assertIsNotNone(match, 'control.js has no canRender')
        predicate = match.group(1)
        # Prefix match, not an enumerated list: control/joint-torque and
        # control/joint-velocity exist in agent-core's format table too, and an
        # enumeration would silently miss the next one added.
        self.assertIn("startsWith('control/')", predicate)


if __name__ == '__main__':
    unittest.main()
