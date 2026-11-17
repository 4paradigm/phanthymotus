"""Package a previously validated G1 model for a clean runtime, without motion data."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def package(profile_path, output):
    profiles = json.loads(Path(profile_path).read_text())
    if len(profiles) != 1:
        raise ValueError("安装包需要恰好一个已验证 G1 配置")
    profile = profiles[0]
    model = Path(profile["model_path"])
    data = model.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if (profile.get("model_sha256") != digest or profile.get("compiled_model_sha256") != digest
            or profile.get("mujoco_version") != "3.16.0" or model.suffix != ".mjb"
            or profile.get("robot_type") != "unitree_g1"):
        raise ValueError("请使用 MuJoCo 3.16.0 已验证的完整 G1 MJB 配置")
    profile["model_path"] = "/profiles/scene.mjb"
    files = {"scene.mjb": data,
             "profiles.json": json.dumps([profile], ensure_ascii=False).encode()}
    files["manifest.json"] = json.dumps({"recipe_id": "supermimic-g1-v1", "robot": "unitree_g1",
                                        "mujoco": "3.16.0", "source": "official-task-runtime",
                                        "contains_training_data": False,
                                        "sha256": {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}}).encode()
    with tarfile.open(output, "w:gz") as archive:
        for name, content in files.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(content)
            entry.mode = 0o644
            archive.addfile(entry, io.BytesIO(content))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    package(args.profile, args.output)
