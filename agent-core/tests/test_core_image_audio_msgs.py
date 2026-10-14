"""The Core image must carry the ROS message used by sound direction."""

import os
from pathlib import Path
import subprocess

import pytest


REPO = Path(__file__).resolve().parents[2]


def test_core_build_context_contains_sound_direction_source(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "uname").write_text("#!/bin/sh\necho aarch64\n")
    (bin_dir / "docker").write_text(
        '#!/bin/sh\nprintf "%s\\n" "$@" > "$DOCKER_CAPTURE"\n')
    # build_core.sh reads the commit it tags the image with *before* it reaches
    # do_build(). The harness mounts this worktree detached, so its .git points
    # at a path that only exists in the review container and real git exits 128
    # there — killing the run before the docker invocation this test exists to
    # check. Answer the tag lookup, the way uname and docker are answered above;
    # build_core.sh runs nothing else against git.
    (bin_dir / "git").write_text("#!/bin/sh\necho abcdef0\n")
    for command in ("uname", "docker", "git"):
        (bin_dir / command).chmod(0o755)
    capture = tmp_path / "docker-args"
    env = os.environ.copy()
    for key in ("REGISTRY", "REGISTRY_USER", "REGISTRY_PASSWORD",
                "IMAGE_NAMESPACE", "RESOURCE_CENTER_API_KEY"):
        env.pop(key, None)
    env.update(PATH=f"{bin_dir}:{env['PATH']}", DOCKER_CAPTURE=str(capture))

    subprocess.run(["bash", str(REPO / "deploy" / "build_core.sh"),
                    "--mirror", "none"], cwd=REPO, env=env, check=True,
                   capture_output=True, text=True)

    context = Path(capture.read_text().splitlines()[-1])
    assert (context / "deploy/ros-base/audio_msgs/msg/SoundDirection.msg").is_file()


def test_core_image_imports_sound_direction_message():
    if not Path("/ros_ws/install/audio_msgs").exists():
        pytest.skip("ROS image is needed for this build smoke test")
    result = subprocess.run(
        ["/work/.venv/bin/python", "-c",
         "from audio_msgs.msg import AudioChunk, SoundDirection"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
