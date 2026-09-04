from pathlib import Path
import unittest


class DockerfileMirrorTest(unittest.TestCase):
    def test_core_rewrites_inherited_apt_source_before_update(self):
        dockerfile = (Path(__file__).parents[1] / "Dockerfile").read_text()

        install_layer = dockerfile.split("# Install ffmpeg", 1)[1].split(
            "# Install uv", 1
        )[0]
        rewrite = install_layer.index('RUN if [ -n "${APT_MIRROR}" ]')
        apt_update = install_layer.index(
            "apt-get -o Acquire::AllowInsecureRepositories=true update"
        )

        self.assertLess(rewrite, apt_update)
        self.assertEqual(install_layer.count("\nRUN "), 1)
        for host in (
            r"archive\.ubuntu\.com",
            r"security\.ubuntu\.com",
            r"ports\.ubuntu\.com",
            r"mirrors\.tencentyun\.com",
        ):
            self.assertIn(host, install_layer)
        self.assertIn("https://${APT_MIRROR}", install_layer)
