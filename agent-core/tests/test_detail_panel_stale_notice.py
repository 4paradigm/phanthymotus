"""「已暂停，画面是最后一帧」只对画布成立。

这条提示存在的理由写在它自己的常量注释里：canvas 会把最后一帧永远留在屏幕上，
停掉的流和活着的流长得一模一样，不说一声就分不出来。

日志不是这样。`data/json` 面板每一行自带时间戳，停没停它自己就说明了；再压一层
提示，唯一的效果是盖住一行数据 —— 现场截图里就是一条红字横在一行 JSON 上，红字
和那行数据都读不成。

没有 JS 测试框架、没有构建步骤（同 test_control_renderer_registered.py 的理由），
所以这里做文本检查。而文本检查恰好抓得住这个 bug：它不是逻辑错，是一条只对某类
渲染器成立的提示被用在了所有渲染器上。

Run: cd agent-core && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_detail_panel_stale_notice.py
"""
import pathlib
import re
import unittest

SOURCE = (pathlib.Path(__file__).resolve().parents[1]
          / 'web' / 'js' / 'detail-panel.js').read_text()


def _fn(name: str) -> str:
    match = re.search(rf'function {name}\(.*?\n\}}', SOURCE, re.S)
    assert match, f'{name} 不见了'
    return match.group(0)


class StaleNoticeTest(unittest.TestCase):

    def test_text_panels_never_arm_the_timer(self):
        """不是"提示换个位置"，是根本不提示 —— 定时器都不武装。"""
        body = _fn('_armStaleTimer')
        self.assertIn('if (_textLike) return;', body)
        # 早退必须在 setTimeout 之前，否则等于没写
        self.assertLess(body.index('if (_textLike) return;'), body.index('setTimeout'))

    def test_a_canvas_still_gets_the_notice(self):
        """图会停在最后一帧上而不自明，这正是它需要被告知的原因。"""
        self.assertIn('画面是最后一帧', _fn('_armStaleTimer'))

    def test_the_kind_comes_from_the_renderer_not_the_format_string(self):
        """格式字符串的写法会变（data/json、text/plain、application/json…），
        渲染器的身份不会。"""
        self.assertIn('Renderer === TextRenderer || Renderer === ActivityRenderer',
                      SOURCE)

    def test_the_flag_is_reset_between_topics(self):
        """留着上一个话题的判断，下一个面板会拿到错的行为。"""
        self.assertIn('_textLike = false;', _fn('_cleanup'))

    def test_no_banner_variant_survives(self):
        """上一版把提示改成顶部横幅，那是把位置当成了问题。真正的问题是它
        在日志上不该出现 —— 横幅那套连同它的样式一起删掉，不留死代码。"""
        self.assertNotIn('_staleNotice', SOURCE)
        self.assertNotIn('以下是最后收到的内容', SOURCE)


if __name__ == '__main__':
    unittest.main()
