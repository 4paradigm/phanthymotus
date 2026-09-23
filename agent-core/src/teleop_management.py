"""Local teleop management capability shared by every Core HTTP call path."""
import os
from pathlib import Path
from urllib.parse import urlsplit


class TeleopManagementError(ValueError):
    """A safe, credential-free failure before making the HTTP request."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


def management_headers(url: str, tool: str, arguments: dict) -> dict:
    """Never send the local capability to a registry-selected remote endpoint."""
    if tool != 'teleop' or arguments.get('action', 'info') == 'info':
        return {}
    try:
        endpoint = urlsplit(url)
        allowed = (
            url == os.environ.get('TELEOP_MANAGEMENT_URL', '')
            and endpoint.scheme == 'http'
            and endpoint.hostname in ('localhost', '127.0.0.1', '::1')
            and not (endpoint.username or endpoint.password or endpoint.query or endpoint.fragment)
            and endpoint.path == '/mcp'
        )
    except ValueError:
        allowed = False
    if not allowed:
        raise TeleopManagementError(403, 'Teleop management endpoint is not configured')
    try:
        key = Path(os.environ['TELEOP_MANAGEMENT_KEY_FILE']).read_text().strip()
    except (KeyError, OSError, UnicodeError):
        raise TeleopManagementError(503, 'Teleop management key is unavailable') from None
    # Reject control characters before aiohttp can include a bad header in an error.
    if len(key) < 32 or not key.isascii() or any(ord(c) <= 32 or ord(c) == 127 for c in key):
        raise TeleopManagementError(503, 'Teleop management key is invalid')
    return {'X-Teleop-Management': key}
