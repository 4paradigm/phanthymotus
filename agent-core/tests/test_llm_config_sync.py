"""
test_llm_config_sync.py — LLM 配置两个入口必须写同一组 key。

存在的原因：真机上发现 client.llm 有值而 services.llm 是空的，用户明确表示
只在 dashboard 里配置过、没碰过别的地方 —— 事实证明确实如此，是代码不对称。

dashboard 有两个地方能配 LLM：

  * 设置页          → api/config.py        写 services.llm，再派生 client.llm
  * 画布 decision_core 卡片 → api/mcp_manage.py   只写 client.llm

从卡片配置后，运行时正常（client/llm.py 只读 client.llm），但设置页读的是
services.llm，于是显示空白 —— 而在那个空白表单上点保存，会把空值写回并
派生进 client.llm，直接搞挂 LLM。

这里断言两条写入路径覆盖同一组 key。纯静态检查，不需要跑起服务。
"""

import os
import pathlib
import re
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

SRC = pathlib.Path(__file__).resolve().parents[1] / 'src'


class TestLlmConfigSync(unittest.TestCase):
    def test_card_config_writes_both_keys(self):
        """decision_core 卡片保存时必须同时写 client.llm 和 services.llm。"""
        src = (SRC / 'api' / 'mcp_manage.py').read_text()
        # 定位 action == 'config' 分支
        idx = src.find("elif action == 'config'")
        self.assertGreater(idx, 0, "decision_core 的 config 分支不见了")
        branch = src[idx:idx + 3000]

        self.assertIn("client_cfg['llm']", branch,
                      'card config no longer writes client.llm')
        self.assertIn("services_cfg['llm']", branch,
                      'card config writes client.llm but not services.llm — '
                      'the Settings page will show blank and can wipe the config')

    def test_settings_page_writes_both_keys(self):
        """设置页保存时必须同时写 services.llm 和派生 client.llm。"""
        src = (SRC / 'api' / 'config.py').read_text()
        idx = src.find('async def config_save')
        self.assertGreater(idx, 0)
        branch = src[idx:idx + 3000]

        self.assertIn("services['llm']", branch)
        self.assertIn("client_cfg['llm']", branch,
                      'settings page writes services.llm but no longer syncs client.llm')

    def test_runtime_reads_client_llm_only(self):
        """运行时只读 client.llm —— 这是「配置错位不影响机器人运行」的原因，
        也是为什么 services.llm 脱节时问题只在 dashboard 显现。"""
        src = (SRC / 'client' / 'llm.py').read_text()
        self.assertIn("config.main['client']['llm']", src)
        self.assertNotIn("services", src,
                         'client/llm.py should not depend on services.llm')

    def test_both_paths_cover_the_same_fields(self):
        """两条路径写入的字段集合必须一致，否则从一边配完到另一边就缺字段。"""
        card = (SRC / 'api' / 'mcp_manage.py').read_text()
        idx = card.find("elif action == 'config'")
        card_branch = card[idx:idx + 3000]

        # services.llm 字面量里出现的字段名
        m = re.search(r"services_cfg\['llm'\]\s*=\s*\{(.*?)\}", card_branch, re.S)
        self.assertIsNotNone(m, 'could not locate the services.llm assignment')
        card_fields = set(re.findall(r"'(\w+)':", m.group(1)))

        for required in ('url', 'key', 'model'):
            self.assertIn(required, card_fields,
                          f'card config omits {required!r} from services.llm')


if __name__ == '__main__':
    unittest.main()
