"""Live vision smoke test for the already-configured DeepSeek Flash API.

Synthetic solid red PNG only; never prints keys, URLs or raw model messages.
Does not alter config; returns nonzero if the API declines image input.
"""
import asyncio
import os
from pathlib import Path
import struct
import tempfile
import sys
import zlib

sys.path.insert(0, "/work/src")


def png_red_square(size=64):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(
            ">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    rows = b"".join(b"\x00" + (b"\xff\x00\x00" * size)
                    for _ in range(size))
    return b"\x89PNG\r\n\x1a\n" + (
        chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b""))


async def main():
    import config
    from openai import AsyncOpenAI
    svc = config.main.get("services", {}).get("llm", {})
    if svc.get("model") != "deepseek-flash":
        print("NOT_TESTED: configured model is not deepseek-flash")
        return 2
    if not svc.get("key"):
        print("NOT_TESTED: no configured key")
        return 2
    from channel import store
    from channel.qq_vision import add_qq_images
    channel_id = "qq_official_v2_acceptance"
    image_dir = store.dir_for(channel_id).resolve()
    descriptor, image_path = tempfile.mkstemp(suffix=".png", dir=image_dir)
    os.fchmod(descriptor, 0o600)
    with os.fdopen(descriptor, "wb") as output:
        output.write(png_red_square())
    path = Path(image_path)
    messages = [{"role": "user", "content":
                "What is the dominant color in this image? Answer one English color word."}]
    trigger = {
        "_qq_channel_turn": True,
        "payload": {"channel_ids": [channel_id]},
        "_qq_image_files": [{"kind": "image", "path": str(path)}],
    }
    attached = await add_qq_images(messages, trigger, vision_enabled=True)
    print("QQ_BRIDGE_ATTACHED", attached)
    if not attached:
        path.unlink(missing_ok=True)
        return 4
    client = AsyncOpenAI(api_key=svc["key"], base_url="https://api.deepseek.com",
                         timeout=30.0, max_retries=0)
    try:
        result = await client.chat.completions.create(
            model="deepseek-flash",
            messages=messages,
            max_tokens=256,
        )
        answer = (result.choices[0].message.content or "").strip().lower()
        print("VISION_REQUEST_ACCEPTED", bool(result.choices))
        print("COLOR_IS_RED", "red" in answer)
        print("MODEL_REPLY_LENGTH", len(answer))
        print("FINISH_REASON", result.choices[0].finish_reason)
        usage = getattr(result, "usage", None)
        detail = getattr(usage, "completion_tokens_details", None)
        print("REASONING_TOKENS", getattr(detail, "reasoning_tokens", None))
        return 0 if "red" in answer else 3
    except Exception as exc:
        print("VISION_REQUEST_FAILED", type(exc).__name__)
        print("HTTP_STATUS", getattr(exc, "status_code", None))
        payload = getattr(exc, "body", None)
        if isinstance(payload, dict):
            info = payload.get("error", payload)
            if isinstance(info, dict):
                print("ERROR_CODE", str(info.get("code", ""))[:80])
                print("ERROR_TYPE", str(info.get("type", ""))[:80])
                print("ERROR_MESSAGE", str(info.get("message", ""))[:190])
        return 1
    finally:
        await client.close()
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
