from pathlib import Path
import unittest


class DockerfileMirrorTest(unittest.TestCase):
    def test_core_rewrites_inherited_apt_source_before_update(self):
        dockerfile = (Path(__file__).parents[1] / "Dockerfile").read_text()

        rewrite = dockerfile.index('ARG PYPI_MIRROR APT_MIRROR')
        apt_update = dockerfile.index(
            'RUN apt-get -o Acquire::AllowInsecureRepositories=true update'
        )

        self.assertLess(rewrite, apt_update)
        for host in (
            r"archive\\.ubuntu\\.com",
            r"security\\.ubuntu\\.com",
            r"ports\\.ubuntu\\.com",
            r"mirrors\\.tencentyun\\.com",
        ):
            self.assertIn(host, dockerfile)
        self.assertIn('https://${APT_MIRROR}', dockerfile)
