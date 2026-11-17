"""Launch the same browser checks on macOS and Linux review machines."""
import os
from pathlib import Path


def chromium_options():
    executable = os.environ.get("MOTION_TEST_CHROMIUM_EXECUTABLE")
    if executable:
        return {"executable_path": executable}
    mac_chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if mac_chrome.is_file():
        return {"executable_path": str(mac_chrome)}
    # Playwright's installed Chromium is the portable default. Install it with
    # `python -m playwright install chromium` when running the browser suite.
    return {}
