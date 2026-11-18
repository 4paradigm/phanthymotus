"""QQ official SDK administrator onboarding tool.

Run only from a trusted local administrator terminal. No bot message, Canvas
tool or remote user can initiate this workflow. Existing credentials are
never overwritten and decrypted credentials are never printed to stdout.
"""
from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path


def save_new_credential(path: Path, credential: str) -> None:
    if not credential or not credential.strip():
        raise ValueError("QQ onboarding returned an empty credential")
    if path.exists():
        raise FileExistsError("The QQ credential target already exists")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write(credential.strip())
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


async def run_onboard(target: Path):
    from qqbot_agent_sdk import start_onboard
    if target.exists():
        raise FileExistsError("QQ credential exists; refusing to rotate it")
    print("Use a trusted QQ client to open the onboarding link:")
    def show_url(url: str) -> None:
        print(url)
    result = await start_onboard(on_qr_ready=show_url)
    save_new_credential(target, result.client_secret)
    print("QQ credentials saved with owner-only access.")
    print("AppID:", result.app_id)
    print("Configure the Core with the corresponding secret-file mount.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Onboard an official QQ bot")
    parser.add_argument(
        "--target", type=Path, required=True,
        help="The protected local path where a new credential file is created")
    args = parser.parse_args()
    asyncio.run(run_onboard(args.target))


if __name__ == "__main__":
    main()
