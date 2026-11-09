"""Exercise the Docker recipe's Python-path setup without ROS or Docker."""
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


DOCKERFILE = Path(__file__).resolve().parents[1] / "Dockerfile.jetson"


class RosPythonPathTests(unittest.TestCase):
    def test_mixed_install_layouts_import_without_sourcing_ros(self):
        recipe = DOCKERFILE.read_text().replace("\\\n", " ")
        script = next(command for command in re.findall(r"^RUN (.*)$", recipe, re.M)
                      if "ros_ws.pth" in command)

        # These are directory layouts, not claims to run either target ABI here.
        for abi in ("3.8", "3.10"):
            with self.subTest(python_layout=abi), tempfile.TemporaryDirectory(
                    prefix="ros-python-path-") as directory:
                root = Path(directory)
                site_dir = root / f"usr/local/lib/python{abi}/dist-packages"
                site_dir.mkdir(parents=True)
                env_file = root / "etc/jetpack.env"
                env_file.parent.mkdir()
                env_file.write_text(f"PY_ABI={abi}\n")

                prefixes = ("opt/rtabmap_ros", "ros_ws/install/audio_msgs",
                            "opt/ros/humble/install")
                package_dirs, modules = [], []
                for layer, prefix in enumerate(prefixes):
                    for kind, relative in (
                            ("site", f"lib/python{abi}/site-packages"),
                            ("dist", f"local/lib/python{abi}/dist-packages")):
                        path = root / prefix / relative
                        path.mkdir(parents=True)
                        package_dirs.append(path)
                        module = f"_ros_path_{layer}_{kind}"
                        (path / f"{module}.py").write_text("VALUE = 'installed'\n")
                        modules.append(module)

                # A duplicate exposes an underlay accidentally overriding the overlay.
                for path, origin in ((package_dirs[1], "overlay"),
                                     (package_dirs[4], "underlay")):
                    (path / "_ros_path_precedence.py").write_text(f"ORIGIN = {origin!r}\n")
                (root / prefixes[0] / "share").mkdir()

                command = script
                for absolute in ("/etc/jetpack.env", "/usr/local/lib",
                                 "/opt/rtabmap_ros", "/ros_ws/install",
                                 "/opt/ros/humble/install"):
                    command = command.replace(absolute, str(root) + absolute)
                result = subprocess.run(["bash", "-c", command], capture_output=True,
                                        text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

                # A fresh interpreter cannot inherit pytest's ROS stubs or PYTHONPATH.
                # addsitedir processes the generated .pth as Python startup would.
                probe = (
                    "import importlib, json, site, sys\n"
                    "site.addsitedir(sys.argv[1])\n"
                    "for name in json.loads(sys.argv[2]):\n"
                    "    importlib.import_module(name)\n"
                    "import _ros_path_precedence\n"
                    "assert _ros_path_precedence.ORIGIN == 'overlay'\n"
                )
                imported = subprocess.run(
                    [sys.executable, "-I", "-S", "-c", probe,
                     str(site_dir), json.dumps(modules)],
                    capture_output=True, text=True, timeout=10)
                self.assertEqual(imported.returncode, 0, imported.stdout + imported.stderr)

                paths = (site_dir / "ros_ws.pth").read_text().splitlines()
                self.assertCountEqual(paths, [str(path) for path in package_dirs])
                for earlier, later in ((package_dirs[:2], package_dirs[2:4]),
                                       (package_dirs[2:4], package_dirs[4:])):
                    self.assertLess(max(paths.index(str(path)) for path in earlier),
                                    min(paths.index(str(path)) for path in later))


if __name__ == "__main__":
    unittest.main()
