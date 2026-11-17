"""Prepare original, SHA-verified CUDA wheels on a faster administrator network.

Only the fixed recipe's Linux x86_64 / Python 3.13 packages are accepted. PyPI
provides authoritative digests; the optional mirror transports identical bytes.
No installed environment or site-packages directory is copied.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time
import urllib.parse
import urllib.request

PACKAGES = {
    "nvidia-cudnn-cu12": "9.10.2.21", "nvidia-cublas-cu12": "12.6.4.1",
    "nvidia-cufft-cu12": "11.3.0.4", "nvidia-cusolver-cu12": "11.7.1.2",
    "nvidia-cusparse-cu12": "12.5.4.2", "nvidia-nccl-cu12": "2.29.3",
    "nvidia-nvshmem-cu12": "3.4.5", "nvidia-cusparselt-cu12": "0.7.1",
    "nvidia-curand-cu12": "10.3.7.77", "nvidia-cuda-nvrtc-cu12": "12.6.85",
    "nvidia-nvjitlink-cu12": "12.8.93", "nvidia-cuda-cupti-cu12": "12.6.80",
    "nvidia-cuda-runtime-cu12": "12.6.77", "nvidia-cufile-cu12": "1.11.1.6",
    "nvidia-nvtx-cu12": "12.6.77", "cuda-bindings": "12.9.4",
    "cuda-pathfinder": "1.3.3", "cuda-toolkit": "12.6.3", "triton": "3.8.0",
    "numpy": "2.4.2", "scipy": "1.17.0", "mujoco": "3.16.0",
    "mujoco-warp": "3.16.0", "onnx": "1.20.1", "onnxscript": "0.6.1",
    "onnxruntime": "1.23.2", "tensordict": "0.11.0", "Pillow": "12.3.0",
}
CHUNK = 8 * 1024 * 1024


def wheel_rank(filename):
    """Rank explicit target tags, independently of the downloader's macOS tags."""
    if not filename.endswith(".whl"):
        return None
    python, abi, platform = filename[:-4].rsplit("-", 3)[-3:]
    stable = re.fullmatch(r"cp3(\d+)", python)
    abi3 = abi == "abi3" and stable is not None and 2 <= int(stable[1]) <= 13
    if not abi3 and (python not in {"py3", "py2.py3", "cp313"} or abi not in {"none", "cp313"}):
        return None
    if platform == "any":
        return (2 if python == "cp313" else int(abi3), 0, 0, filename)
    compatible = []
    for tag in platform.split("."):
        if not tag.endswith("_x86_64"):
            continue
        match = re.fullmatch(r"manylinux_(\d+)_(\d+)_x86_64", tag)
        minimum = tuple(map(int, match.groups())) if match else {
            "manylinux2014_x86_64": (2, 17), "manylinux2010_x86_64": (2, 12),
            "manylinux1_x86_64": (2, 5), "linux_x86_64": (0, 0)}.get(tag)
        if minimum is not None and minimum <= (2, 36):  # Debian bookworm image.
            compatible.append(minimum)
    if not compatible:
        return None
    major, minor = max(compatible)
    return (2 if python == "cp313" else int(abi3), major, minor, filename)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pypi_entry(name, version, mirror):
    with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=45) as response:
        metadata = json.load(response)
    matches = []
    for entry in metadata["urls"]:
        filename = entry["filename"]
        if wheel_rank(filename) is not None:
            matches.append(entry)
    if not matches:
        raise ValueError(f"{name} 没有匹配 Linux x86_64 / Python 3.13 / glibc 2.36 的官方 wheel")
    entry = max(matches, key=lambda value: wheel_rank(value["filename"]))
    url = entry["url"]
    if mirror:
        parsed = urllib.parse.urlparse(url)
        if parsed.hostname != "files.pythonhosted.org" or not parsed.path.startswith("/packages/"):
            raise ValueError("官方 PyPI wheel 来源无效")
        url = "https://mirrors.aliyun.com/pypi" + parsed.path
    return {"name": name, "version": version, "filename": entry["filename"], "size": entry["size"],
            "sha256": entry["digests"]["sha256"], "official_url": entry["url"], "download_url": url}


def torch_entry(mirror=False):
    index = "https://download.pytorch.org/whl/cu126/torch/"
    with urllib.request.urlopen(index, timeout=45) as response:
        value = response.read().decode()
    links = [html.unescape(link) for link in re.findall(r'href=["\x27]([^"\x27]+)', value)]
    matches = [urllib.parse.urljoin(index, link) for link in links
               if "torch-2.14.1+cu126-cp313-cp313-manylinux_2_28_x86_64.whl" in urllib.parse.unquote(link)]
    if len(matches) != 1:
        raise ValueError("官方 PyTorch cu126 wheel 清单不唯一")
    url, _, fragment = matches[0].partition("#")
    if not re.fullmatch(r"sha256=[a-f0-9]{64}", fragment):
        raise ValueError("官方 PyTorch wheel 缺少摘要")
    with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-0"}), timeout=45) as response:
        size = int(response.headers["Content-Range"].rsplit("/", 1)[1])
        response.read(1)
    download_url = ("https://mirrors.nju.edu.cn/pytorch/whl/cu126/" + url.rsplit("/", 1)[1]) if mirror else url
    return {"name": "torch", "version": "2.14.1+cu126",
            "filename": urllib.parse.unquote(urllib.parse.urlparse(url).path.rsplit("/", 1)[1]),
            "size": size, "sha256": fragment[7:], "official_url": url, "download_url": download_url}


def warp_entry():
    with urllib.request.urlopen("https://api.github.com/repos/NVIDIA/warp/releases/tags/v1.18.0", timeout=45) as response:
        release = json.load(response)
    filename = "warp_lang-1.18.0+cu12-py3-none-manylinux_2_28_x86_64.whl"
    assets = [entry for entry in release["assets"] if entry["name"] == filename]
    if len(assets) != 1 or not re.fullmatch(r"sha256:[a-f0-9]{64}", assets[0].get("digest", "")):
        raise ValueError("官方 Warp release 缺少唯一 wheel 或摘要")
    entry = assets[0]
    return {"name": "warp-lang", "version": "1.18.0+cu12", "filename": filename,
            "size": entry["size"], "sha256": entry["digest"][7:],
            "official_url": entry["browser_download_url"], "download_url": entry["browser_download_url"]}


def download(entry, directory, workers=8):
    if (not isinstance(entry.get("sha256"), str) or not re.fullmatch(r"[a-f0-9]{64}", entry["sha256"])
            or not isinstance(entry.get("filename"), str) or Path(entry["filename"]).name != entry["filename"]
            or not entry["filename"].endswith(".whl") or type(entry.get("size")) is not int or entry["size"] <= 0):
        raise ValueError("wheel 文件名、长度或摘要无效")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / entry["filename"]
    if target.exists() and target.stat().st_size == entry["size"] and digest(target) == entry["sha256"]:
        print(json.dumps({"package": entry["name"], "state": "verified_cache"}), flush=True)
        return target
    parts = directory / ".parts" / entry["sha256"]
    parts.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    completed = 0
    began = time.monotonic()

    def part(index):
        nonlocal completed
        start = index * CHUNK
        end = min(entry["size"], start + CHUNK) - 1
        size = end - start + 1
        path = parts / f"{index:05d}"
        if not path.exists() or path.stat().st_size != size:
            temp = path.with_suffix(".part")
            for attempt in range(4):
                try:
                    request = urllib.request.Request(entry["download_url"],
                                                     headers={"Range": f"bytes={start}-{end}"})
                    with urllib.request.urlopen(request, timeout=90) as response, temp.open("wb") as stream:
                        expected = f"bytes {start}-{end}/{entry['size']}"
                        if response.status != 206 or response.headers.get("Content-Range") != expected:
                            if not (response.status == 200 and start == 0 and size == entry["size"]):
                                raise ValueError("下载源不支持校验过的分块范围")
                        total = 0
                        while True:
                            block = response.read(1024 * 1024)
                            if not block:
                                break
                            total += len(block)
                            if total > size:
                                raise ValueError("下载分块超出预期大小")
                            stream.write(block)
                    if temp.stat().st_size != size:
                        raise ValueError("下载分块不完整")
                    os.replace(temp, path)
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(.5 * (attempt + 1))
        with lock:
            completed += size
            print(json.dumps({"package": entry["name"], "verified_ranges_bytes": completed,
                              "total_bytes": entry["size"], "elapsed_s": round(time.monotonic() - began, 2)}), flush=True)
        return path

    count = (entry["size"] + CHUNK - 1) // CHUNK
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        paths = list(pool.map(part, range(count)))
    assembled = target.with_suffix(".assembling")
    with assembled.open("wb") as output:
        for path in paths:
            with path.open("rb") as source:
                shutil.copyfileobj(source, output, 4 * 1024 * 1024)
    actual = digest(assembled)
    if actual != entry["sha256"]:
        assembled.unlink(missing_ok=True)
        shutil.rmtree(parts)
        raise ValueError(f"{entry['name']} 完整 SHA 与官方摘要不符，拒绝安装")
    os.replace(assembled, target)
    shutil.rmtree(parts)
    print(json.dumps({"package": entry["name"], "state": "sha256_verified", "sha256": actual}), flush=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--mirror", action="store_true", help="Use the mirror only as transport; keep official SHA checks")
    parser.add_argument("--workers", type=int, default=8, choices=range(1, 9))
    args = parser.parse_args()
    root = Path(args.directory)
    root.mkdir(parents=True, exist_ok=True)
    records = []
    def save_manifest():
        temporary = root / "manifest.json.tmp"
        temporary.write_text(json.dumps({"recipe_id": "supermimic-g1-v1", "wheels": records}, indent=2))
        os.replace(temporary, root / "manifest.json")
    for name, version in PACKAGES.items():
        entry = pypi_entry(name, version, args.mirror)
        download(entry, root, args.workers)
        records.append(entry)
        save_manifest()
    for entry in (torch_entry(args.mirror), warp_entry()):
        download(entry, root, args.workers)
        records.append(entry)
        save_manifest()


if __name__ == "__main__":
    main()
