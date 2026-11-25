"""`control/*` must reach ControlRenderer, in every view that picks one.

There is no build step and no JS test runner here, so a renderer that exists but
was never added to a RENDERERS array fails silently and looks exactly like the
feature not being written: the topic falls through to ActivityRenderer or the KV
panel and shows a `values` array as one long line of numbers.

The detail panel, dashboard and monitor dashboard can declare their own array
or import a shared one. Check the array each view actually uses.

The ordering assertion matters as much as the membership one: `RENDERERS.find`
takes the first match, and TextRenderer answers to a broad enough hint that a
ControlRenderer placed after it would never be reached.

Run: cd agent-core && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_control_renderer_registered.py
"""
import pathlib
import re
import unittest

WEB = pathlib.Path(__file__).resolve().parents[1] / 'web' / 'js'
VIEWS = ('detail-panel.js', 'dashboard.js', 'monitor-dashboard.js')


def _renderer_source(view: str) -> str:
    path = WEB / view
    source = path.read_text()
    shared = re.search(r"""^import\s*\{\s*RENDERERS\s*\}\s*from\s*['"]([^'"]+)['"]""",
                       source, re.M)
    return (path.parent / shared.group(1)).read_text() if shared else source


def _renderers_array(source: str) -> list:
    match = re.search(r'const RENDERERS = \[(.*?)\]', source, re.S)
    if not match:
        return []
    return [name.strip() for name in match.group(1).split(',') if name.strip()]


class ControlRendererRegistration(unittest.TestCase):
    def test_every_view_registers_it(self):
        for view in VIEWS:
            with self.subTest(view=view):
                source = _renderer_source(view)
                self.assertRegex(
                    source,
                    r"""(?m)^import\s*\{\s*ControlRenderer\s*\}\s*from\s*"""
                    r"""['"]\./(?:renderers/)?control\.js['"]""",
                    f'{view} renderer list does not import ControlRenderer',
                )
                self.assertIn('ControlRenderer', _renderers_array(source),
                              f'{view} renderer list omits ControlRenderer')

    def test_it_is_tried_before_the_catch_all_renderers(self):
        """`find` takes the first match; a broad renderer ahead of it wins."""
        for view in VIEWS:
            with self.subTest(view=view):
                names = _renderers_array(_renderer_source(view))
                self.assertIn('ControlRenderer', names, f'{view} renderer list omits ControlRenderer')
                index = names.index('ControlRenderer')
                for catch_all in ('TextRenderer', 'ActivityRenderer'):
                    if catch_all in names:
                        self.assertLess(index, names.index(catch_all),
                                        f'{view}: ControlRenderer must precede {catch_all}')

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
