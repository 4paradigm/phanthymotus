"""`control/*` must reach ControlRenderer, in every view that picks one.

The detail panel and monitor dashboard import the shared renderer registry;
the legacy dashboard still declares its own RENDERERS array. Check each view's
actual registration source so a missing renderer cannot silently fall through
to ActivityRenderer or the KV panel. Shared views must keep importing the
registry rather than maintaining another copy of its list.

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
SHARED_VIEWS = ('detail-panel.js', 'monitor-dashboard.js')


def _renderers_array(source: str) -> list:
    match = re.search(r'const RENDERERS = \[(.*?)\]', source, re.S)
    if not match:
        return []
    return [name.strip() for name in match.group(1).split(',') if name.strip()]


class ControlRendererRegistration(unittest.TestCase):
    def _registration_source(self, view):
        path = WEB / view
        source = path.read_text()
        if view in SHARED_VIEWS:
            self.assertRegex(
                source,
                r"import\s*\{\s*RENDERERS\s*\}\s*from\s*"
                r"['\"]\./renderers/registry\.js['\"]",
                f'{view} must import the shared RENDERERS registry',
            )
            path = WEB / 'renderers' / 'registry.js'
            source = path.read_text()
        return path, source

    def test_every_view_imports_and_lists_it(self):
        for view in VIEWS:
            with self.subTest(view=view):
                path, source = self._registration_source(view)
                control_import = re.search(
                    r"import\s*\{\s*ControlRenderer\s*\}\s*from\s*"
                    r"['\"]([^'\"]+)['\"]", source,
                )
                self.assertIsNotNone(
                    control_import, f'{view}: {path.name} does not import ControlRenderer',
                )
                self.assertEqual(
                    (path.parent / control_import.group(1)).resolve(),
                    (WEB / 'renderers' / 'control.js').resolve(),
                    f'{view}: ControlRenderer must come from renderers/control.js',
                )
                self.assertIn('ControlRenderer', _renderers_array(source),
                              f'{view} imports ControlRenderer but never lists it, '
                              f'so control/* still falls through')

    def test_it_is_tried_before_the_catch_all_renderers(self):
        """`find` takes the first match; a broad renderer ahead of it wins."""
        for view in VIEWS:
            with self.subTest(view=view):
                _, source = self._registration_source(view)
                names = _renderers_array(source)
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
