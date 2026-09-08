"""
tests/test_mcp_file_proxy.py — the cross-container file upload proxy.

Why it exists: agent-core serves the browser, but the tool that needs a file
runs in another container and shares no filesystem with it. A path minted here
is meaningless there — the first face-enrolment attempt failed with an
`image_path` that really existed, in *this* container. base64 through the tool
call failed too (43 800 characters does not survive an LLM's context).

So `POST /api/mcp/{mcp_id}/file/upload` streams the bytes to the owning service,
which writes them where it can see them and returns **its own** absolute path.
The address comes from the MCP registry, which is what makes one route cover
perception, actucore and every driver despite their differing ports.

Run: python3 -m pytest agent-core/tests/test_mcp_file_proxy.py -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import fastapi  # noqa: E402

from api.mcp_manage import _file_intake_url  # noqa: E402


REGISTRY = [
    {'id': 'mcp-perception', 'url': 'http://localhost:15720/mcp', 'transport': 'http'},
    {'id': 'mcp-g1',         'url': 'http://localhost:15701/mcp', 'transport': 'http'},
    {'id': 'mcp-tianyi',     'url': 'http://localhost:15707/mcp', 'transport': 'http'},
    {'id': 'agentcore',      'url': '',                           'transport': 'internal'},
    {'id': 'mcp-noport',     'url': '',                           'transport': 'http'},
    {'id': 'mcp-prefixed',   'url': 'http://host:9000/api/v2/mcp', 'transport': 'http'},
    {'id': 'mcp-nosuffix',   'url': 'http://host:9100/',          'transport': 'http'},
]


@pytest.fixture(autouse=True)
def _registry(monkeypatch):
    import api.mcp_manage as mm
    monkeypatch.setattr(mm, '_get_mcp_list', lambda: REGISTRY)


def test_the_port_comes_from_the_registry_not_a_hardcoded_table():
    """Each service reports its url when it registers, so one route covers
    perception and every driver even though the ports all differ."""
    assert _file_intake_url('mcp-perception') == 'http://localhost:15720/file/upload'
    assert _file_intake_url('mcp-g1')         == 'http://localhost:15701/file/upload'
    assert _file_intake_url('mcp-tianyi')     == 'http://localhost:15707/file/upload'


def test_a_url_with_a_path_prefix_keeps_it():
    """rsplit('/mcp') rather than urljoin, which would discard the prefix."""
    assert _file_intake_url('mcp-prefixed') == 'http://host:9000/api/v2/file/upload'


def test_a_url_without_the_mcp_suffix_still_resolves():
    assert _file_intake_url('mcp-nosuffix') == 'http://host:9100/file/upload'


def test_an_unknown_mcp_is_a_404():
    with pytest.raises(fastapi.HTTPException) as caught:
        _file_intake_url('mcp-does-not-exist')
    assert caught.value.status_code == 404


def test_an_internal_mcp_points_at_the_local_endpoint_instead():
    """agentcore and channel are served in-process; there is nothing to proxy
    to, and saying so beats a confusing connection error."""
    with pytest.raises(fastapi.HTTPException) as caught:
        _file_intake_url('agentcore')
    assert caught.value.status_code == 400
    assert '/api/file/upload' in caught.value.detail


def test_a_service_that_has_not_reported_a_url_yet_is_503():
    """Distinct from 404: the service exists but has not registered its address,
    so retrying is reasonable."""
    with pytest.raises(fastapi.HTTPException) as caught:
        _file_intake_url('mcp-noport')
    assert caught.value.status_code == 503
