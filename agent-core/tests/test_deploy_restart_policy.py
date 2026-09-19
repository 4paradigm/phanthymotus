"""
test_deploy_restart_policy.py — 机器人重启后必须自己回来。

G1 在一次重启之后，所有容器仍然是 exited：agent-core、驱动、perception 一个都没起来。
要有人 SSH 进去手动 `docker start` 才恢复 —— 在那之前机器人就是没了，而且从外面看
「栈没起来」和「部署坏了」完全一样。

`unless-stopped` 与 `always` 的区别正好落在这件事上：前者不会去启动在守护进程停止前
就处于停止态的容器。代价是手动停掉的容器在重启后也会回来 —— 在这些机器上这是更小的
意外，因为「该在跑却没在跑」的代价明显更大。

这条测试盯住部署片段，不是盯某一台机器：策略写在仓库的 compose 片段里，控制台把它们
合并进主机的 docker-compose.yml，所以在机器上 `docker update` 改掉的那份会在下次部署
时被还原。

Run: cd agent-core && python3 -m pytest tests/test_deploy_restart_policy.py
"""

import pathlib
import unittest

import yaml

REPO = pathlib.Path(__file__).resolve().parents[2]

# 机器人栈的部署片段。pr-review 不在其列：那是构建机上的 PR 审查工具，不是机器人的
# 一部分，它在重启后自己回来没有意义。
FRAGMENTS = [
    REPO / 'agent-core' / 'deploy' / 'docker-compose.yml',
    REPO / 'perception' / 'deploy' / 'service.yml',
    REPO / 'actucore' / 'deploy' / 'service.yml',
]


def _services(path: pathlib.Path) -> dict:
    """片段可能是完整的 compose（带 services:），也可能是单个服务。"""
    doc = yaml.safe_load(path.read_text(encoding='utf-8'))
    if isinstance(doc, dict) and 'services' in doc:
        return doc['services']
    return doc or {}


class RestartPolicyTests(unittest.TestCase):
    def test_every_fragment_exists(self):
        """路径写错会让下面每条都空转通过。"""
        for f in FRAGMENTS:
            self.assertTrue(f.is_file(), f'{f} 不存在')

    def test_every_service_restarts_always(self):
        for f in FRAGMENTS:
            for name, svc in _services(f).items():
                self.assertEqual(
                    svc.get('restart'), 'always',
                    f'{f.relative_to(REPO)} 的 {name}: '
                    f'{svc.get("restart")!r} —— 重启后机器人不会自己回来')

    def test_no_fragment_still_says_unless_stopped(self):
        """文本层面再查一遍：注释里提到它是可以的，配置里不行。"""
        for f in FRAGMENTS:
            for line in f.read_text(encoding='utf-8').splitlines():
                stripped = line.strip()
                if stripped.startswith('#'):
                    continue
                self.assertNotIn('unless-stopped', stripped,
                                 f'{f.relative_to(REPO)}: {line}')

    def test_the_console_deploy_path_also_uses_always(self):
        """机器人上大多数容器其实是控制台建的，不是 compose 建的。

        `api/drivers.py` 直接调 docker SDK 创建容器，所以它那处硬编码的策略才是
        真正决定「重启后回不回来」的那一个 —— 只改 compose 片段会留下这条更常用的
        路径不动。
        """
        src = (REPO / 'agent-core' / 'src' / 'api' / 'drivers.py').read_text(encoding='utf-8')
        code = '\n'.join(ln for ln in src.splitlines() if not ln.lstrip().startswith('#'))
        self.assertIn("restart_policy={'Name': 'always'}", code)
        self.assertNotIn("'unless-stopped'", code)

    def test_install_sh_inherits_rather_than_hardcodes(self):
        """install.sh 从镜像里取 compose，所以它没有自己的一份策略要维护。

        这条断言的是那个契约本身：Dockerfile 必须把 deploy/ 放进镜像，
        否则 install.sh 的 `docker cp .../deploy/docker-compose.yml` 会取到别的东西。
        """
        dockerfile = (REPO / 'agent-core' / 'Dockerfile').read_text(encoding='utf-8')
        self.assertIn('COPY deploy/', dockerfile)

    def test_the_reason_is_written_down(self):
        """没有理由的话，下一个人会觉得 unless-stopped 更稳妥而改回去。"""
        for f in FRAGMENTS:
            text = f.read_text(encoding='utf-8')
            self.assertIn('unless-stopped', text,
                          f'{f.relative_to(REPO)} 没有说明为什么不用 unless-stopped')


if __name__ == '__main__':
    unittest.main()
