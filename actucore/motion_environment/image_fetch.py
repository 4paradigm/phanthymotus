"""Fetch one pinned official Python image without changing Docker daemon proxies.

The index, platform manifest, compressed blobs and expanded layer digests are
verified before making a Docker-load archive. No registry credentials are saved.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import urllib.parse
import urllib.request

INDEX = "sha256:b685a4fa58bb19d1814d78a1ec0f0208f351452724f78b20212c984d6e124a34"
REPOSITORY = "library/python"
REGISTRY = "https://registry-1.docker.io/v2/" + REPOSITORY
ACCEPT = ", ".join(["application/vnd.oci.image.index.v1+json", "application/vnd.oci.image.manifest.v1+json",
                    "application/vnd.docker.distribution.manifest.list.v2+json",
                    "application/vnd.docker.distribution.manifest.v2+json"])


def digest_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return "sha256:" + value.hexdigest()


def select_manifest(index):
    matches = [p for p in index.get("manifests", [])
               if p.get("platform", {}).get("os") == "linux"
               and p.get("platform", {}).get("architecture") == "amd64"]
    if len(matches) != 1:
        raise ValueError("官方镜像缺少唯一 Linux amd64 清单")
    return matches[0]["digest"]


class Fetcher:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.opener = urllib.request.build_opener()
        self.token = None

    def auth(self):
        url = "https://auth.docker.io/token?" + urllib.parse.urlencode(
            {"service": "registry.docker.io", "scope": "repository:library/python:pull"})
        with self.opener.open(url, timeout=60) as response:
            self.token = json.load(response)["token"]

    def blob(self, kind, digest):
        if not digest.startswith("sha256:") or len(digest) != 71:
            raise ValueError("官方镜像摘要格式无效")
        path = self.root / digest.replace(":", "-")
        if path.is_file() and digest_file(path) == digest:
            print("复用已校验镜像文件 " + digest[:23], flush=True)
            return path
        if self.token is None:
            self.auth()
        request = urllib.request.Request(REGISTRY + "/" + kind + "/" + digest,
                                         headers={"Accept": ACCEPT})
        # Blob requests may redirect to signed object-storage URLs. Do not copy
        # registry Authorization to the redirected host.
        request.add_unredirected_header("Authorization", "Bearer " + self.token)
        temporary = path.with_suffix(".part")
        total = 0
        value = hashlib.sha256()
        with self.opener.open(request, timeout=120) as response, temporary.open("wb") as stream:
            expected = response.headers.get("Content-Length", "未知")
            print(f"下载官方镜像文件 {digest[:23]}，字节数 {expected}", flush=True)
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
                value.update(chunk)
                total += len(chunk)
                if total % (16 * 1024 * 1024) == 0:
                    print(f"已下载 {digest[:23]} {total} 字节", flush=True)
        if "sha256:" + value.hexdigest() != digest:
            temporary.unlink(missing_ok=True)
            raise ValueError("官方镜像下载摘要校验失败")
        os.replace(temporary, path)
        print(f"摘要校验通过 {digest[:23]}，实际 {total} 字节", flush=True)
        return path

    def build_archive(self):
        index = json.loads(self.blob("manifests", INDEX).read_bytes())
        manifest_digest = select_manifest(index)
        manifest = json.loads(self.blob("manifests", manifest_digest).read_bytes())
        config_digest = manifest["config"]["digest"]
        config = self.blob("blobs", config_digest)
        metadata = json.loads(config.read_bytes())
        diffs = metadata["rootfs"]["diff_ids"]
        if len(diffs) != len(manifest["layers"]):
            raise ValueError("镜像层清单不一致")
        layers = []
        for layer, expected in zip(manifest["layers"], diffs):
            compressed = self.blob("blobs", layer["digest"])
            expanded = self.root / (expected.replace(":", "-") + ".tar")
            if not expanded.exists() or digest_file(expanded) != expected:
                if not layer["mediaType"].endswith("gzip"):
                    raise ValueError("固定镜像包含不支持的压缩格式")
                temp = expanded.with_suffix(".part")
                with gzip.open(compressed, "rb") as source, temp.open("wb") as output:
                    while True:
                        block = source.read(1024 * 1024)
                        if not block:
                            break
                        output.write(block)
                if digest_file(temp) != expected:
                    temp.unlink(missing_ok=True)
                    raise ValueError("解压后的镜像层摘要不匹配")
                os.replace(temp, expanded)
            layers.append(expanded)
        image = "local/phanthymotus-python-base:" + config_digest[7:23]
        archive_path = self.root / "python-3.13.9-amd64.tar"
        temporary = archive_path.with_suffix(".part")
        config_name = config_digest[7:] + ".json"
        entries = [{"Config": config_name, "RepoTags": [image],
                    "Layers": [p.name for p in layers]}]
        with tarfile.open(temporary, "w") as archive:
            archive.add(config, arcname=config_name)
            for path in layers:
                archive.add(path, arcname=path.name)
            data = json.dumps(entries).encode()
            entry = tarfile.TarInfo("manifest.json")
            entry.size = len(data)
            archive.addfile(entry, io.BytesIO(data))
        os.replace(temporary, archive_path)
        result = {"image": image, "image_id": config_digest, "index_digest": INDEX,
                  "manifest_digest": manifest_digest, "archive": str(archive_path),
                  "archive_sha256": digest_file(archive_path)}
        (self.root / "image.json").write_text(json.dumps(result))
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    args = parser.parse_args()
    print(json.dumps(Fetcher(args.cache).build_archive()), flush=True)
