"""
test_canvas_toolbar_touch.py — 工具条上的控件在手机上必须点得动、够得着。

两条约定,违反了都**不报错、不报警、看上去完全正常**,只是在手上失效:

1. **`pointer-events` 要自己 opt in。** 768px 以下 `.canvas-top-control` 是
   `pointer-events: none`（让触摸穿过去落在画布上），所以它的每个可交互子元素都得
   自己声明 `auto`。设备收尾芯片第一版漏了这一行:390x844 真机上 Playwright 的
   `tap()` 直接超时 —— 而那时它正是屏幕上唯一的提示（活动流在手机上是默认关着的
   抽屉）。桌面鼠标一切正常,所以这个 bug 只在手机上存在。

2. **触摸目标有 44px 地板,而这份样式表用 `pointer: coarse` 划线、不用宽度。**
   平板是宽屏 + 粗指针:按 768px 判断会让它拿到桌面尺寸的命中区。「重试」按钮第
   一版是 42x19px,不到地板的一半。

这里不渲染 CSS（没有浏览器），只断言规则**存在**。真正的验证是 Playwright 在
Orin 5 上量到的 44px 和成功的 tap；这个文件守的是「以后别人加控件时不要再忘」。

Run: cd agent-core && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_canvas_toolbar_touch.py
"""

from __future__ import annotations

import pathlib
import re

import pytest

WEB = pathlib.Path(__file__).resolve().parents[1] / 'web'
CSS = (WEB / 'css/style.css').read_text(encoding='utf-8')
INDEX = (WEB / 'index.html').read_text(encoding='utf-8')


def _media_blocks(query: str, source: str = CSS) -> str:
    """把某个 media query 的所有块体拼起来（花括号配平，不是正则贪婪）。

    同一个 query 在这份样式表里出现多次（coarse 有四处，768px 有十几处），而声明
    可能在任意一个里。只取第一个块会让断言看起来通过/失败都不可信。
    """
    out = []
    needle = f'@media {query}'
    at = source.find(needle)
    while at != -1:
        start = source.index('{', at)
        depth, i = 0, start
        while i < len(source):
            if source[i] == '{':
                depth += 1
            elif source[i] == '}':
                depth -= 1
                if depth == 0:
                    break
            i += 1
        out.append(source[start + 1:i])
        at = source.find(needle, i)
    assert out, f'样式表里找不到 @media {query}'
    return '\n'.join(out)


def _rule_bodies(selector: str, source: str = CSS) -> list[str]:
    """某个选择器的**每一条**规则体 —— 基础层和各断点的覆盖都算。"""
    bodies = re.findall(
        r'(?:^|[,{}\s])' + re.escape(selector) + r'\s*(?:,[^{]*)?\{([^}]*)\}',
        source, re.MULTILINE)
    assert bodies, f'样式表里找不到 {selector}'
    return bodies


# ── 工具条里每个可交互控件都要能收到触摸 ─────────────────────────────────────

def test_the_toolbar_disables_pointer_events_on_mobile():
    """这是前提。哪天它不再是 none，下面两条就不再是必要条件（但也无害）。"""
    mobile = _media_blocks('(max-width: 768px)')
    bodies = _rule_bodies('.canvas-top-control', mobile)
    assert any('pointer-events: none' in b for b in bodies), bodies


@pytest.mark.parametrize('selector', [
    '.canvas-main-btn',        # 已有
    '.auto-start-toggle',      # 已有
    '.teardown-chip',          # 设备收尾芯片 —— 第一版漏的就是这个
])
def test_every_interactive_toolbar_child_opts_into_pointer_events(selector):
    # 声明在基础层还是在手机断点里都行（现有两个控件一个在基础层、一个在断点里），
    # 要紧的是**手机上生效**。
    bodies = _rule_bodies(selector)
    assert any('pointer-events: auto' in b for b in bodies), (
        f'{selector} 没有 pointer-events: auto —— 它在手机上会完全点不动，'
        f'而桌面上一切正常，所以这个 bug 只会在手上出现')


def test_the_toolbar_children_in_the_html_are_the_ones_covered():
    """HTML 里新增一个工具条控件时，这个列表要跟着更新。"""
    section = INDEX[INDEX.index('id="canvas-top-control"'):]
    section = section[:section.index('id="canvas-controls"')]
    ids = set(re.findall(r'id="([a-z0-9-]+)"', section))
    assert {'canvas-project-toggle', 'teardown-chip'} <= ids, ids


# ── 触摸尺寸 ─────────────────────────────────────────────────────────────────

def test_touch_sizing_is_keyed_on_pointer_coarse_not_width():
    """平板是宽屏 + 粗指针。按宽度划线会让它拿到桌面命中区。"""
    coarse = _media_blocks('(pointer: coarse)')
    for selector in ('.teardown-summary', '.teardown-retry'):
        bodies = _rule_bodies(selector, coarse)
        assert any('min-height: 44px' in b for b in bodies), \
            f'{selector} 没有 44px 地板（实测第一版「重试」是 42x19px）'


def test_the_mobile_dropdown_cannot_overflow_the_screen():
    """工具条在那个断点是 left/right 定位且会换行，下拉不能按自己的左边缘定位。"""
    mobile = _media_blocks('(max-width: 768px)')
    bodies = _rule_bodies('.teardown-detail', mobile)
    assert any('left: 0' in b and 'right: 0' in b and 'min-width: 0' in b
               for b in bodies), \
        '手机断点里缺少 left/right/min-width 覆盖 —— 桌面的 min-width: 260px 会把' \
        '它顶出屏幕,而工具条在这个断点是 left/right 定位且会换行'


# ── 坏消息在手机上不能只写进抽屉 ─────────────────────────────────────────────

def test_a_failed_teardown_also_toasts_on_mobile():
    """活动流在 768px 以下是默认关着的抽屉，写进去等于没写。"""
    canvas = (WEB / 'js/canvas.js').read_text(encoding='utf-8')
    section = canvas[canvas.index("project_stop_done"):]
    section = section[:section.index('canvas_editor')]
    assert "matchMedia('(max-width: 768px)')" in section, section[:400]
    assert '_showToast' in section
