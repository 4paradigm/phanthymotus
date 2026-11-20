"""Local immutable assets and bounded resumable upload transport."""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import uuid
import zipfile

from .contracts import PROTOCOL, identifier, integer, public_profile, validate_motion


def read_json(path):
    return json.loads(Path(path).read_text())


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_artifacts(folder, hashes):
    folder = Path(folder).resolve()
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("模型缺少文件完整性记录")
    for relative, expected in hashes.items():
        path = (folder / relative).resolve(strict=True)
        if not path.is_relative_to(folder) or file_sha256(path) != expected:
            raise ValueError("模型文件已改变或损坏")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with tmp.open("x") as f:
            json.dump(value, f, ensure_ascii=False, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


class AssetStore:
    def __init__(self, root, profiles, max_upload_bytes=64 * 1024 * 1024):
        self.root, self.profiles = Path(root).resolve(), profiles
        self.max_upload_bytes = max_upload_bytes
        self.lock = threading.RLock()
        for name in ("uploads", "datasets", "models", "jobs", "packages"):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def profile(self, id):
        identifier(id)
        if id not in self.profiles:
            raise ValueError("机器人配置不存在")
        return self.profiles[id]

    def get(self, kind, id):
        return read_json(self.root / kind / (identifier(id) + ".json"))

    def list(self, kind):
        return [read_json(p) for p in sorted((self.root / kind).glob("*.json"))]

    def dataset(self, id, profile_id=None):
        d = self.get("datasets", id)
        p = self.profile(profile_id or d["profile_id"])
        if d["profile_hash"] != p["profile_hash"]:
            raise ValueError("数据集与当前机器人配置不匹配")
        path = self.root / "datasets" / (d["id"] + ".npz")
        if file_sha256(path) != d["sha256"]:
            raise ValueError("动作数据文件已改变或损坏")
        return d, path

    def verified_model(self, id):
        model = self.get("models", id)
        folder = self.root / "jobs" / identifier(model["job_id"])
        hashes = model.get("artifact_sha256", {})
        required = {model[k] for k in ("actor", "world_model", "onnx") if model.get(k)} | {"run.json"}
        if not required <= hashes.keys():
            raise ValueError("模型缺少必要文件校验记录")
        verify_artifacts(folder, hashes)
        if model.get("evaluation_job_id"):
            verify_artifacts(self.root / "jobs" / identifier(model["evaluation_job_id"]),
                             model.get("evaluation_artifact_sha256"))
        return model, folder

    def catalog(self):
        return {"protocol": PROTOCOL, "profiles": [public_profile(p) for p in self.profiles.values()],
                "datasets": self.list("datasets"), "models": self.list("models")}

    def preview(self, args):
        import numpy as np
        dataset, path = self.dataset(args.get("dataset_id"))
        limit = integer(args.get("limit", 100), 2, 200, "预览帧数")
        with np.load(path, allow_pickle=False) as data:
            indices = np.unique(np.linspace(0, dataset["frames"] - 1, min(limit, dataset["frames"]), dtype=int))
            frames = [{"time": int(i) / dataset["fps"], "root_pos": data["root_pos_w"][i].tolist(),
                       "joint_pos": data["joint_pos"][i].tolist()} for i in indices]
        return {"dataset": dataset, "frames": frames}

    def upload_begin(self, args):
        profile = self.profile(args.get("profile_id"))
        size = integer(args.get("size"), 1, self.max_upload_bytes, "文件大小")
        name = args.get("name", "")
        if not isinstance(name, str) or not name.lower().endswith(".npz") or len(name) > 200:
            raise ValueError("请选择 .npz 动作文件")
        with self.lock:
            active = 0
            for path in (self.root / "uploads").glob("*.json"):
                old = read_json(path)
                if time.time() - old["created_at"] > 3600:
                    path.unlink()
                    path.with_suffix(".part").unlink(missing_ok=True)
                else:
                    active += 1
            if active >= 4:
                raise ValueError("同时上传数量已达上限")
            id = uuid.uuid4().hex
            record = {"id": id, "name": Path(name).name, "size": size, "offset": 0,
                      "profile_id": profile["id"], "profile_hash": profile["profile_hash"],
                      "created_at": time.time()}
            (self.root / "uploads" / (id + ".part")).touch(exist_ok=False)
            atomic_json(self.root / "uploads" / (id + ".json"), record)
            return {"upload_id": id, "offset": 0}

    def upload_chunk(self, args):
        id = identifier(args.get("upload_id"))
        offset = integer(args.get("offset"), 0, self.max_upload_bytes, "offset")
        encoded = args.get("data")
        if not isinstance(encoded, str) or len(encoded) > 32768:
            raise ValueError("上传分片超过限制")
        try:
            blob = base64.b64decode(encoded, validate=True)
        except Exception as e:
            raise ValueError("无效上传编码") from e
        if not blob:
            raise ValueError("空上传分片")
        with self.lock:
            record = self.get("uploads", id)
            path = self.root / "uploads" / (id + ".part")
            actual = path.stat().st_size
            if offset < actual:
                with path.open("rb") as f:
                    f.seek(offset)
                    if f.read(len(blob)) != blob:
                        raise ValueError("重复分片内容不同")
                return {"offset": actual}
            if offset != actual or actual + len(blob) > record["size"]:
                raise ValueError("上传偏移或大小不匹配")
            with path.open("ab") as f:
                f.write(blob)
                f.flush()
            return {"offset": actual + len(blob)}

    def upload_finish(self, args):
        id = identifier(args.get("upload_id"))
        with self.lock:
            if (self.root / "datasets" / (id + ".json")).exists():
                return {"dataset": self.get("datasets", id)}
            record = self.get("uploads", id)
            path = self.root / "uploads" / (id + ".part")
            if path.stat().st_size != record["size"]:
                raise ValueError("文件尚未上传完整")
            p = self.profile(record["profile_id"])
            if record["profile_hash"] != p["profile_hash"]:
                raise ValueError("上传期间机器人配置已改变")
            metadata = validate_motion(path, p)
            d = {k: record[k] for k in ("id", "name", "size", "profile_id", "profile_hash", "created_at")}
            d.update(metadata, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            os.replace(path, self.root / "datasets" / (id + ".npz"))
            atomic_json(self.root / "datasets" / (id + ".json"), d)
            (self.root / "uploads" / (id + ".json")).unlink()
            return {"dataset": d}

    def export(self, model_id):
        model, jobdir = self.verified_model(model_id)
        profile = self.profile(model["profile_id"])
        if profile["profile_hash"] != model["profile_hash"]:
            raise ValueError("模型与当前机器人配置不匹配")
        if not model.get("onnx"):
            raise ValueError("此训练结果没有可部署的 ONNX 模型")
        # A package can be exported before qualification; its status must survive export.
        manifest = {"schema": PROTOCOL, "model": model, "profile": public_profile(profile),
                    "qualification": model.get("qualification", "unverified"), "files": {}}
        sources = {"policy.onnx": jobdir / model["onnx"],
                   "run.json": jobdir / "run.json",
                   "reference.npz": self.dataset(model["dataset_id"])[1]}
        if model.get("evaluation_job_id"):
            sources["evaluation.json"] = self.root / "jobs" / model["evaluation_job_id"] / "evaluation.json"
        id = uuid.uuid4().hex
        path = self.root / "packages" / (id + ".zip")
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, source in sources.items():
                manifest["files"][name] = hashlib.sha256(source.read_bytes()).hexdigest()
                archive.write(source, name)
            archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False))
        atomic_json(self.root / "packages" / (id + ".json"), {"id": id, "size": path.stat().st_size})
        return {"package_id": id, "size": path.stat().st_size, "manifest": manifest}

    def artifact_read(self, args):
        id = identifier(args.get("package_id"))
        offset = integer(args.get("offset", 0), 0, 2**40, "offset")
        limit = integer(args.get("limit", 24576), 1, 24576, "limit")
        path = self.root / "packages" / (id + ".zip")
        total = path.stat().st_size
        if offset > total:
            raise ValueError("读取偏移超出文件大小")
        with path.open("rb") as f:
            f.seek(offset)
            blob = f.read(limit)
        return {"data": base64.b64encode(blob).decode(), "offset": offset + len(blob), "total": total}
