"""
test_websearch_multimedia.py — WebSearch 必须按结果类型解析，不能只认网页那一种形状。

踩过的坑：`WebSearch` 的签名从一开始就声明了 `search_type: web|image|video`，但响应解析写死了
网页结果的五个字段（title/url/content/website/date）。搜索后端对三种类型返回**三种不同形状**：

- image 的图在 `image.url`，而 `content` 恒为空串 —— 老代码于是输出"一个标题 + 一个百度中转
  落地页"，真正的图片 URL 整个被丢掉。搜图功能看着能跑，实际一张图都拿不到。
- video 的元信息在 `video.{duration,hover_pic,width,height}`，且 `video.url` 恒为空 ——
  能播的是上一级的落地页 `url`。
- 网页结果的配图在 `web_extensions.images[]`。

本文件里的响应片段取自对 router.phanthy.com 的实测（2026-09-14，query=故宫），不是编的；
top_k 的上限（web 50 / image 30 / video 10）也是实测出来的。

Run: cd agent-core && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_websearch_multimedia.py
"""

import asyncio
import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

from event import desktop  # noqa: E402

# ── 实测响应片段 ────────────────────────────────────────────────────────────────

WEB_RESULT = {
    'id': '1',
    'title': '毛主席为何终生未踏故宫',
    'url': 'https://baijiahao.baidu.com/s?id=1876307421973612384',
    'icon': 'https://baijiahao.baidu.com/favicon.ico',
    'web_anchor': '',
    'website': '百家号',
    'content': '1954年5月，北京故宫城墙上……',
    'rerank_score': 1.0,
    'authority_score': 0.9,
    'date': '2026-09-14 19:43:18',
    'type': 'web',
    'is_aladdin': False,
    'web_extensions': {
        'images': [
            {'url': 'https://pic.rmb.bdstatic.com/a.jpeg', 'height': '607', 'width': '576'},
            {'url': 'https://pic.rmb.bdstatic.com/b.jpeg', 'height': '334', 'width': '576'},
            {'url': 'https://pic.rmb.bdstatic.com/c.jpeg', 'height': '679', 'width': '576'},
            {'url': 'https://pic.rmb.bdstatic.com/d.jpeg', 'height': '100', 'width': '100'},
        ],
    },
}

IMAGE_RESULT = {
    'id': '1',
    'title': '北京故宫夏日游：避暑与下午茶全攻略',
    'url': 'http://mbd.baidu.com/newspage/data/dtlandingsuper?nid=dt_5256952985203586790',
    'icon': 'https://psstatic.cdn.bcebos.com/video/wiseindex/aa6.png',
    'web_anchor': '',
    'website': '',
    'content': '',                      # 图片结果的 content 恒为空
    'rerank_score': 1.0,
    'date': '2025-03-01 14:40:28',
    'type': 'image',
    'image': {
        'url': 'http://gips2.baidu.com/it/u=2704664748,2854719008&fm=3074',
        'height': '1250',
        'width': '960',
    },
    'is_aladdin': False,
}

VIDEO_RESULT = {
    'id': '1',
    'title': '沈阳故宫',
    'url': 'http://weibo.com/tv/show/1034:5342792495595562',
    'icon': 'https://b.bdstatic.com/searchbox/x.png',
    'web_anchor': '',
    'website': '微博',
    'content': '沈阳故宫',
    'rerank_score': 1.0,
    'date': '2026-09-13 21:00:58',
    'type': 'video',
    'video': {
        'url': '',                      # 实测恒为空，可播地址是上一级的 url
        'height': '720',
        'width': '1280',
        'size': '',
        'duration': '164',
        'hover_pic': 'http://t13.baidu.com/it/u=1612456892,2599712163&fm=225',
    },
    'is_aladdin': False,
}


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status = 200
        self.reason = 'OK'
        self.headers = {'content-type': 'application/json'}

    async def json(self):
        return self._payload

    async def text(self):
        return json.dumps(self._payload)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    """记录最后一次 POST 的 body，好断言请求侧的 top_k 限流。"""

    last_payload = None

    def __init__(self, payload):
        self._payload = payload

    def post(self, url, json=None, **kwargs):
        _FakeSession.last_payload = json
        return _FakeResponse(self._payload)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _run_search(results, **kwargs) -> str:
    """跑一次 WebSearch，upstream 用给定的 results 伪造。"""
    content = json.dumps({'query': kwargs.get('query', '故宫'), 'results': results})
    body = {'choices': [{'message': {'content': content}}]}
    cfg = {'desktop_tools': {'search': {
        'type': 'baidu_search', 'base_url': 'https://example.invalid/v1', 'api_key': 'k',
    }}}
    tools = desktop.DesktopTools()
    with mock.patch.object(desktop.config, 'main', cfg), \
         mock.patch('aiohttp.ClientSession', lambda *a, **k: _FakeSession(body)):
        return asyncio.run(tools.WebSearch(**kwargs))


class ImageResultsTest(unittest.TestCase):
    def test_image_url_reaches_the_model(self):
        out = _run_search([IMAGE_RESULT], query='故宫', search_type='image')
        self.assertIn(IMAGE_RESULT['image']['url'], out,
                      'the actual picture URL must survive formatting')
        self.assertIn('960x1250', out)
        # 落地页要标成 Page，不能顶替图片位置。
        self.assertIn(f'Page: {IMAGE_RESULT["url"]}', out)

    def test_header_says_image(self):
        out = _run_search([IMAGE_RESULT], query='故宫', search_type='image')
        self.assertTrue(out.startswith('Image search results for "故宫"'), out[:80])

    def test_empty_content_is_not_printed(self):
        out = _run_search([IMAGE_RESULT], query='故宫', search_type='image')
        # 空 content 曾经被原样 append 成一个空行；正文块之间只应有格式化产生的空行。
        self.assertNotIn('\n\n\n', out)


class VideoResultsTest(unittest.TestCase):
    def test_watch_link_falls_back_to_landing_page(self):
        out = _run_search([VIDEO_RESULT], query='故宫', search_type='video')
        self.assertIn(f'Watch: {VIDEO_RESULT["url"]}', out,
                      'video.url is empty in practice — the landing page is the playable link')

    def test_cover_duration_and_resolution(self):
        out = _run_search([VIDEO_RESULT], query='故宫', search_type='video')
        self.assertIn(VIDEO_RESULT['video']['hover_pic'], out)
        self.assertIn('Duration: 2:44', out)
        self.assertIn('Resolution: 1280x720', out)

    def test_duration_hours(self):
        self.assertEqual(desktop._fmt_duration('3841'), '1:04:01')
        self.assertEqual(desktop._fmt_duration(''), '')
        self.assertEqual(desktop._fmt_duration(None), '')
        self.assertEqual(desktop._fmt_duration('0'), '')


class WebResultsTest(unittest.TestCase):
    def test_article_figures_are_surfaced(self):
        out = _run_search([WEB_RESULT], query='故宫', search_type='web')
        for img in WEB_RESULT['web_extensions']['images'][:3]:
            self.assertIn(img['url'], out)
        # 只带前 3 张，避免一条结果就灌一屏 URL。
        self.assertNotIn(WEB_RESULT['web_extensions']['images'][3]['url'], out)

    def test_web_fields_still_render(self):
        out = _run_search([WEB_RESULT], query='故宫', search_type='web')
        self.assertIn(f'URL: {WEB_RESULT["url"]}', out)
        self.assertIn('百家号 | 2026-09-14 19:43:18', out)
        self.assertIn('Authority: high', out)

    def test_result_type_wins_over_requested_type(self):
        """混排响应里每条结果按自己的 type 渲染。"""
        out = _run_search([WEB_RESULT, IMAGE_RESULT], query='故宫', search_type='web')
        self.assertIn(IMAGE_RESULT['image']['url'], out)
        self.assertIn(WEB_RESULT['url'], out)

    def test_missing_type_falls_back_to_web(self):
        stripped = {k: v for k, v in WEB_RESULT.items() if k != 'type'}
        out = _run_search([stripped], query='故宫', search_type='web')
        self.assertIn(f'URL: {WEB_RESULT["url"]}', out)


class TopKLimitsTest(unittest.TestCase):
    def _sent_top_k(self, **kwargs) -> int:
        _run_search([WEB_RESULT], query='故宫', **kwargs)
        return _FakeSession.last_payload['search_parameters']['top_k']

    def test_per_type_caps(self):
        self.assertEqual(self._sent_top_k(search_type='web', top_k=99), 50)
        self.assertEqual(self._sent_top_k(search_type='image', top_k=99), 30)
        self.assertEqual(self._sent_top_k(search_type='video', top_k=99), 10)

    def test_per_type_defaults(self):
        self.assertEqual(self._sent_top_k(search_type='web'), 10)
        self.assertEqual(self._sent_top_k(search_type='image'), 8)
        self.assertEqual(self._sent_top_k(search_type='video'), 5)

    def test_explicit_small_value_is_respected(self):
        self.assertEqual(self._sent_top_k(search_type='image', top_k=3), 3)

    def test_unknown_type_falls_back_to_web(self):
        # 后端对 news/aladdin 直接 422，先在本地收敛成 web 而不是把 422 丢给模型。
        _run_search([WEB_RESULT], query='故宫', search_type='news')
        self.assertEqual(_FakeSession.last_payload['search_parameters']['search_type'], 'web')


class DownloadTest(unittest.TestCase):
    """WebFetch(save_to=...) —— 远程图片变成本地文件，才谈得上发给用户。"""

    def test_path_outside_allowlist_is_refused(self):
        tools = desktop.DesktopTools()
        out = asyncio.run(tools.WebFetch('https://example.invalid/a.jpg',
                                         save_to='/etc/passwd_probe'))
        self.assertTrue(out.startswith('Error:'), out)
        self.assertIn('outside allowed directories', out)
        self.assertFalse(pathlib.Path('/etc/passwd_probe').exists())


if __name__ == '__main__':
    unittest.main()
