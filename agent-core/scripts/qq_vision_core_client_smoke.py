"""Exercise QQ attachment -> shared image reader -> actual Agent Core LLM client.

Uses a synthetic red image in shared Channel storage and deletes it afterwards.
Never sends QQ messages or prints credentials.
"""
import asyncio
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, "/work/src")
sys.path.insert(0, "/work/scripts")
from qq_vision_model_smoke import png_red_square


async def main():
    from channel import store
    from channel.qq_vision import add_qq_images
    from client import llm
    channel_id = "qq_official_v2_acceptance"
    folder = store.dir_for(channel_id).resolve()
    fd, filename = tempfile.mkstemp(suffix=".png", dir=folder)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(png_red_square())
    img = Path(filename)
    try:
        messages = [{
            "role": "system", "content": "Reply concisely to the user's image question.",
        }, {
            "role": "user", "content":
            "What is the dominant color in the image? Say one English word.",
        }]
        trigger = {
            "_qq_channel_turn": True,
            "payload": {"channel_ids": [channel_id]},
            "_qq_image_files": [{"kind": "image", "path": str(img)}],
        }
        attached = await add_qq_images(messages, trigger, vision_enabled=True)
        print("QQ_BRIDGE_ATTACHED", attached)
        if not attached:
            return 2
        try:
            result = await llm(messages, [])
        except Exception as exc:
            print("CORE_LLM_FAILED", type(exc).__name__)
            return 3
        answer = (result.get("content") or "").strip().lower()
        print("CORE_LLM_RETURNED", isinstance(result, dict))
        print("COLOR_IS_RED", "red" in answer)
        print("ANSWER_LENGTH", len(answer))
        return 0 if "red" in answer else 4
    finally:
        img.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
