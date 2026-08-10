"""Verify both Android OpenXR Capture APKs without installing a headset."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
EXPECTED_PERMISSIONS = {
    "android.permission.ACCESS_NETWORK_STATE",
    "android.permission.INTERNET",
    "android.permission.WAKE_LOCK",
    "org.khronos.openxr.permission.OPENXR",
    "org.khronos.openxr.permission.OPENXR_SYSTEM",
}
EXPECTED_LIBRARIES = {
    "lib/arm64-v8a/libc++_shared.so",
    "lib/arm64-v8a/libmotus_quest_capture.so",
    "lib/arm64-v8a/libopenxr_loader.so",
}
ANDROID_LIBRARIES = {
    "libandroid.so",
    "libc.so",
    "libdl.so",
    "libEGL.so",
    "libGLESv3.so",
    "liblog.so",
    "libm.so",
}
FLAVORS = {
    "meta": {
        "package": "com.phanthymotus.questcapture",
        "label": "PhanthyMotus Meta Capture",
        "pico_metadata": False,
    },
    "pico": {
        "package": "com.phanthymotus.picocapture",
        "label": "PhanthyMotus PICO Capture",
        "pico_metadata": True,
    },
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def run(*arguments: str) -> str:
    completed = subprocess.run(
        arguments,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return completed.stdout


def android_tools() -> tuple[Path, Path]:
    sdk = Path(os.environ.get("ANDROID_SDK_ROOT", ""))
    require(sdk.is_dir(), "ANDROID_SDK_ROOT must point to the pinned Android SDK")
    build_tools = sdk / "build-tools" / "35.0.1"
    prebuilt = (
        sdk
        / "ndk"
        / "27.0.12077973"
        / "toolchains"
        / "llvm"
        / "prebuilt"
    )
    readelf_candidates = list(prebuilt.glob("*/bin/llvm-readelf"))
    for tool in ("aapt2", "apksigner", "zipalign"):
        require((build_tools / tool).is_file(), f"missing Android tool: {tool}")
    require(len(readelf_candidates) == 1, "missing or ambiguous NDK llvm-readelf")
    return build_tools, readelf_candidates[0]


def verify_elf(readelf: Path, library: Path, packaged_names: set[str]) -> None:
    header = run(str(readelf), "-h", str(library))
    require("Class:                             ELF64" in header, f"{library.name} is not ELF64")
    require("Machine:                           AArch64" in header, f"{library.name} is not AArch64")
    dynamic = run(str(readelf), "-d", str(library))
    needed = set(re.findall(r"Shared library: \[([^]]+)]", dynamic))
    unexpected = needed - packaged_names - ANDROID_LIBRARIES
    require(not unexpected, f"{library.name} has unbundled dependencies: {sorted(unexpected)}")


def verify_apk(
    flavor: str,
    expected: dict[str, object],
    build_tools: Path,
    readelf: Path,
) -> dict[str, object]:
    apk = PROJECT / "app" / "build" / "outputs" / "apk" / flavor / "debug" / f"app-{flavor}-debug.apk"
    require(apk.is_file(), f"missing {flavor} APK: {apk}")

    badging = run(str(build_tools / "aapt2"), "dump", "badging", str(apk))
    for fragment in (
        f"package: name='{expected['package']}'",
        "versionCode='2'",
        "versionName='0.2.0'",
        "minSdkVersion:'29'",
        "targetSdkVersion:'35'",
        f"application-label:'{expected['label']}'",
        "native-code: 'arm64-v8a'",
    ):
        require(fragment in badging, f"{flavor} badging is missing {fragment!r}")

    permissions = run(str(build_tools / "aapt2"), "dump", "permissions", str(apk))
    actual_permissions = set(re.findall(r"uses-permission: name='([^']+)'", permissions))
    require(
        actual_permissions == EXPECTED_PERMISSIONS,
        f"{flavor} permission mismatch: {sorted(actual_permissions)}",
    )

    manifest = run(
        str(build_tools / "aapt2"),
        "dump",
        "xmltree",
        "--file",
        "AndroidManifest.xml",
        str(apk),
    )
    require('="android.app.lib_name"' in manifest, f"{flavor} lacks NativeActivity library metadata")
    require('="org.khronos.openxr.intent.category.IMMERSIVE_HMD"' in manifest, f"{flavor} lacks OpenXR launch category")
    pico_metadata = re.search(
        r'="pvr\.app\.type".*\n\s+A: .*="vr"',
        manifest,
    ) is not None
    require(
        pico_metadata == expected["pico_metadata"],
        f"{flavor} pvr.app.type presence is incorrect",
    )

    run(str(build_tools / "zipalign"), "-c", "-P", "16", "-v", "4", str(apk))
    signature = run(
        str(build_tools / "apksigner"),
        "verify",
        "--verbose",
        "--print-certs",
        str(apk),
    )
    require(
        "Verified using v2 scheme (APK Signature Scheme v2): true" in signature,
        f"{flavor} APK lacks a valid v2 signature",
    )

    with zipfile.ZipFile(apk) as archive:
        libraries = {name for name in archive.namelist() if name.startswith("lib/")}
        require(libraries == EXPECTED_LIBRARIES, f"{flavor} native library set mismatch: {sorted(libraries)}")
        with tempfile.TemporaryDirectory(prefix=f"motus-{flavor}-apk-") as directory:
            extracted = Path(directory)
            packaged_names = {Path(name).name for name in libraries}
            for name in libraries:
                archive.extract(name, extracted)
                verify_elf(readelf, extracted / name, packaged_names)

    digest = hashlib.sha256(apk.read_bytes()).hexdigest()
    return {
        "bytes": apk.stat().st_size,
        "package": expected["package"],
        "pvr_app_type": pico_metadata,
        "sha256": digest,
    }


def main() -> None:
    build_tools, readelf = android_tools()
    results = {
        flavor: verify_apk(flavor, expected, build_tools, readelf)
        for flavor, expected in FLAVORS.items()
    }
    require(
        results["meta"]["sha256"] != results["pico"]["sha256"],
        "Meta and PICO APKs must be distinct artifacts",
    )
    print(json.dumps({"artifacts": results, "result": "PASS"}, sort_keys=True))


if __name__ == "__main__":
    main()
