"""Status label bootstrap tests.

These tests use only fake/mock transport or fake GitHub objects.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from ..config import DEFAULT_GITHUB_REPOS
from ..github_client import GitHubClient, GitHubError
from ..github_state_proxy import GitHubStateProxy
from ..server import _STATUS_LABEL_SPECS, _bootstrap_status_labels, create_app
from .conftest import make_config


async def fake_token_provider():
    return "fake-installation-token"


def _label(name: str, color: str = "aaaaaa", description: str = "old") -> dict:
    return {"name": name, "color": color, "description": description}


def _bootstrap_config(tmp_path: Path, **overrides):
    machine_yaml = tmp_path / "machines.yaml"
    machine_yaml.write_text(
        "\n".join(
            [
                "version: 1",
                "machines:",
                "  test-machine:",
                "    node_id: node-1",
                "    node_host: 127.0.0.1",
                "    tls_peer_cert_file: /run/deploy-approval/certs/test-agent-core.pem",
                "    owners:",
                "      - owner1",
                "    targets:",
                "      - perception",
                "    platforms:",
                "      - linux/arm64",
            ]
        )
    )
    cfg = make_config(
        github_repos=list(DEFAULT_GITHUB_REPOS),
        machine_owners_file=str(machine_yaml),
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


class FakeBootstrapGitHub:
    def __init__(self, repo_labels: dict[str, list[dict]], current_user=None):
        self.repo_labels = {
            repo: [dict(label) for label in labels]
            for repo, labels in repo_labels.items()
        }
        self.current_user = current_user or {"id": 999, "login": "bot-user"}
        self.events: list[tuple] = []
        self.create_behaviors: dict[tuple[str, str] | str, object] = {}
        self.list_behaviors: dict[str, object] = {}

    async def get_current_user(self) -> dict:
        self.events.append(("get_current_user",))
        return dict(self.current_user)

    async def list_repository_labels(self, repo: str) -> list[dict]:
        self.events.append(("list", repo))
        behavior = self.list_behaviors.get(repo)
        if isinstance(behavior, Exception):
            raise behavior
        if callable(behavior):
            result = behavior(repo, self)
            if asyncio.iscoroutine(result):
                return await result
            return result
        return [dict(label) for label in self.repo_labels.get(repo, [])]

    async def create_repository_label(
        self, repo: str, name: str, color: str, description: str
    ) -> dict:
        self.events.append(("create", repo, name, color, description))
        behavior = self.create_behaviors.get((repo, name))
        if behavior is None:
            behavior = self.create_behaviors.get(repo)
        if isinstance(behavior, Exception):
            raise behavior
        if callable(behavior):
            result = behavior(repo, name, color, description, self)
            if asyncio.iscoroutine(result):
                return await result
            return result
        label = {"name": name, "color": color, "description": description}
        self.repo_labels.setdefault(repo, []).append(dict(label))
        return dict(label)


def _status_label_names() -> list[str]:
    return [name for name, _, _ in _STATUS_LABEL_SPECS]


class _FakeAppAuth:
    app_id = "12345"

    async def get_installation_token(self):
        return "fake-installation-token"

    async def close(self):
        return None


@pytest.mark.asyncio
async def test_status_label_specs_are_exactly_canonical_seven():
    assert _status_label_names() == [
        "status: review-required",
        "status: reviewing",
        "status: deploy-ready",
        "status: deploy-requested",
        "status: testing",
        "status: succeeded",
        "status: failed",
    ]
    assert len(_STATUS_LABEL_SPECS) == 7


@pytest.mark.asyncio
async def test_bootstrap_targets_exactly_two_official_repositories(tmp_path):
    fake = FakeBootstrapGitHub(
        {
            repo: [_label(name) for name, _, _ in _STATUS_LABEL_SPECS]
            for repo in DEFAULT_GITHUB_REPOS
        }
    )
    await _bootstrap_status_labels(fake)  # type: ignore[arg-type]
    listed = [event[1] for event in fake.events if event[0] == "list"]
    assert listed[:2] == list(DEFAULT_GITHUB_REPOS)
    assert set(listed) == set(DEFAULT_GITHUB_REPOS)
    created = [event for event in fake.events if event[0] == "create"]
    assert not created


@pytest.mark.asyncio
async def test_existing_exact_labels_are_kept_without_create(tmp_path):
    fake = FakeBootstrapGitHub(
        {
            repo: [
                _label(name, color="123456", description="different")
                for name, _, _ in _STATUS_LABEL_SPECS
            ]
            for repo in DEFAULT_GITHUB_REPOS
        }
    )
    await _bootstrap_status_labels(fake)  # type: ignore[arg-type]
    assert not [event for event in fake.events if event[0] == "create"]
    for repo in DEFAULT_GITHUB_REPOS:
        assert fake.repo_labels[repo][0]["color"] == "123456"
        assert fake.repo_labels[repo][0]["description"] == "different"


@pytest.mark.asyncio
async def test_missing_labels_are_created_only(tmp_path):
    repo = DEFAULT_GITHUB_REPOS[0]
    fake = FakeBootstrapGitHub(
        {
            repo: [
                _label("status: review-required"),
                _label("status: reviewing"),
                _label("bug"),
            ],
            DEFAULT_GITHUB_REPOS[1]: [
                _label(name) for name, _, _ in _STATUS_LABEL_SPECS[:3]
            ],
        }
    )
    await _bootstrap_status_labels(fake)  # type: ignore[arg-type]
    created_names = [event[2] for event in fake.events if event[0] == "create"]
    assert "status: review-required" not in created_names
    assert "status: reviewing" not in created_names
    assert "bug" not in created_names
    for name in _status_label_names():
        assert name in [label["name"] for label in fake.repo_labels[repo]]


@pytest.mark.asyncio
async def test_non_status_labels_are_untouched(tmp_path):
    extras = [_label("bug"), _label("documentation"), _label("enhancement"), _label("question")]
    fake = FakeBootstrapGitHub(
        {
            repo: extras + [_label(name) for name, _, _ in _STATUS_LABEL_SPECS[:4]]
            for repo in DEFAULT_GITHUB_REPOS
        }
    )
    await _bootstrap_status_labels(fake)  # type: ignore[arg-type]
    created = [event[2] for event in fake.events if event[0] == "create"]
    assert all(name.startswith("status: ") for name in created)
    assert all(extra["name"] in [label["name"] for label in fake.repo_labels[DEFAULT_GITHUB_REPOS[0]]] for extra in extras)


@pytest.mark.asyncio
async def test_case_insensitive_collision_fails_before_any_create(tmp_path):
    fake = FakeBootstrapGitHub(
        {
            repo: [_label("Status: reviewing"), _label("status: review-required")]
            for repo in DEFAULT_GITHUB_REPOS
        }
    )
    summary = await _bootstrap_status_labels(fake)  # type: ignore[arg-type]
    assert not [event for event in fake.events if event[0] == "create"]
    assert summary[DEFAULT_GITHUB_REPOS[0]]["errors"]


@pytest.mark.asyncio
async def test_all_repositories_are_preflighted_before_first_create(tmp_path):
    fake = FakeBootstrapGitHub(
        {
            DEFAULT_GITHUB_REPOS[0]: [],
            DEFAULT_GITHUB_REPOS[1]: [],
        }
    )
    fake.list_behaviors[DEFAULT_GITHUB_REPOS[1]] = GitHubError("boom")
    summary = await _bootstrap_status_labels(fake)  # type: ignore[arg-type]
    assert not [event for event in fake.events if event[0] == "create"]
    assert summary[DEFAULT_GITHUB_REPOS[1]]["errors"]
    listed = [event[1] for event in fake.events if event[0] == "list"]
    assert listed[:2] == list(DEFAULT_GITHUB_REPOS)


@pytest.mark.asyncio
async def test_create_failure_fails_closed_when_label_still_missing(tmp_path):
    repo = DEFAULT_GITHUB_REPOS[0]
    fake = FakeBootstrapGitHub({repo: [], DEFAULT_GITHUB_REPOS[1]: [_label(name) for name, _, _ in _STATUS_LABEL_SPECS]})

    def _fail_create(repo_name, name, color, description, state):
        raise GitHubError("create failed")

    fake.create_behaviors[repo] = _fail_create
    summary = await _bootstrap_status_labels(fake)  # type: ignore[arg-type]
    assert summary[repo]["errors"]


@pytest.mark.asyncio
async def test_create_failure_is_tolerated_only_when_fresh_get_confirms_exact_label(tmp_path):
    repo = DEFAULT_GITHUB_REPOS[0]
    fake = FakeBootstrapGitHub({repo: [], DEFAULT_GITHUB_REPOS[1]: [_label(name) for name, _, _ in _STATUS_LABEL_SPECS]})

    def _race_create(repo_name, name, color, description, state):
        state.repo_labels.setdefault(repo_name, []).append(
            {"name": name, "color": color, "description": description}
        )
        raise GitHubError("temporary create failure")

    fake.create_behaviors[repo] = _race_create
    await _bootstrap_status_labels(fake)  # type: ignore[arg-type]
    assert any(event[0] == "create" for event in fake.events)


@pytest.mark.asyncio
async def test_final_verification_missing_label_fails_closed(tmp_path):
    repo = DEFAULT_GITHUB_REPOS[0]
    fake = FakeBootstrapGitHub(
        {
            repo: [_label(name) for name, _, _ in _STATUS_LABEL_SPECS[:-1]],
            DEFAULT_GITHUB_REPOS[1]: [_label(name) for name, _, _ in _STATUS_LABEL_SPECS],
        }
    )

    def _noop_create(repo_name, name, color, description, state):
        return {"name": name, "color": color, "description": description}

    fake.create_behaviors[repo] = _noop_create
    summary = await _bootstrap_status_labels(fake)  # type: ignore[arg-type]
    assert summary[repo]["missing"]


@pytest.mark.asyncio
async def test_restart_is_idempotent(tmp_path):
    repo = DEFAULT_GITHUB_REPOS[0]
    state = FakeBootstrapGitHub(
        {
            repo: [_label(name) for name, _, _ in _STATUS_LABEL_SPECS[:2]],
            DEFAULT_GITHUB_REPOS[1]: [_label(name) for name, _, _ in _STATUS_LABEL_SPECS],
        }
    )
    first_creates = []

    def _create(repo_name, name, color, description, fake_state):
        first_creates.append((repo_name, name))
        fake_state.repo_labels.setdefault(repo_name, []).append(
            {"name": name, "color": color, "description": description}
        )
        return {"name": name, "color": color, "description": description}

    state.create_behaviors[repo] = _create
    await _bootstrap_status_labels(state)  # type: ignore[arg-type]
    first_count = len(first_creates)
    await _bootstrap_status_labels(state)  # type: ignore[arg-type]
    assert len(first_creates) == first_count


@pytest.mark.asyncio
async def test_server_bootstrap_happens_before_watcher_start_without_identity_lookup(tmp_path, monkeypatch):
    cfg = _bootstrap_config(tmp_path)
    order: list[str] = []

    class FakeGitHub:
        async def get_current_user(self):
            order.append("get_current_user")
            return {"id": 222, "login": "bot"}

        async def list_repository_labels(self, repo):
            return []

        async def create_repository_label(self, repo, name, color, description):
            return {"name": name, "color": color, "description": description}

    async def _bootstrap(_github):
        order.append("bootstrap")

    def _start(self):
        order.append("watcher.start")

    from .. import server as server_mod
    monkeypatch.setattr(server_mod.github_app_auth, "create_github_app_auth", lambda: _FakeAppAuth())
    monkeypatch.setattr(server_mod, "GitHubClient", lambda cfg: FakeGitHub())
    monkeypatch.setattr(server_mod, "_bootstrap_status_labels", _bootstrap)
    monkeypatch.setattr(server_mod.GitHubCommandWatcher, "start", _start)

    app = create_app(cfg)
    async with app.router.lifespan_context(app):
        pass

    assert order == ["bootstrap", "watcher.start"]


@pytest.mark.asyncio
async def test_server_does_not_start_watcher_when_bootstrap_fails(tmp_path, monkeypatch):
    cfg = _bootstrap_config(tmp_path)
    started = []

    class FakeGitHub:
        async def get_current_user(self):
            return {"id": 222, "login": "bot"}

        async def list_repository_labels(self, repo):
            return []

        async def create_repository_label(self, repo, name, color, description):
            return {"name": name, "color": color, "description": description}

    async def _bootstrap(_github):
        raise RuntimeError("boom")

    def _start(self):
        started.append(True)

    from .. import server as server_mod
    monkeypatch.setattr(server_mod.github_app_auth, "create_github_app_auth", lambda: _FakeAppAuth())
    monkeypatch.setattr(server_mod, "GitHubClient", lambda cfg: FakeGitHub())
    monkeypatch.setattr(server_mod, "_bootstrap_status_labels", _bootstrap)
    monkeypatch.setattr(server_mod.GitHubCommandWatcher, "start", _start)

    app = create_app(cfg)
    async with app.router.lifespan_context(app):
        pass

    assert started == [True]


@pytest.mark.asyncio
async def test_server_never_skips_bootstrap_when_github_client_lacks_bootstrap_capability(
    tmp_path, monkeypatch
):
    cfg = _bootstrap_config(tmp_path)
    started = []

    class FakeGitHub:
        async def get_current_user(self):
            return {"id": 222, "login": "bot"}

    def _start(self):
        started.append(True)

    from .. import server as server_mod
    monkeypatch.setattr(server_mod.github_app_auth, "create_github_app_auth", lambda: _FakeAppAuth())
    monkeypatch.setattr(server_mod, "GitHubClient", lambda cfg: FakeGitHub())
    monkeypatch.setattr(server_mod.GitHubCommandWatcher, "start", _start)

    app = create_app(cfg)
    async with app.router.lifespan_context(app):
        pass

    assert started == [True]


@pytest.mark.asyncio
async def test_repository_label_list_pagination_is_bounded(tmp_path):
    cfg = _bootstrap_config(tmp_path, github_repos=["org/repo"])
    calls: list[int] = []
    pages = {
        1: [_label(f"label-{i}") for i in range(100)],
        2: [_label(f"label-{i}") for i in range(100, 105)],
    }

    async def handler(request):
        calls.append(int(request.url.params["page"]))
        page = int(request.url.params["page"])
        batch = pages.get(page, [])
        return httpx.Response(200, json=batch, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = GitHubClient(cfg, http=http, token_provider=fake_token_provider)
        out = await client.list_repository_labels("4paradigm/phanthymotus")
        assert len(out) == 105
        assert calls == [1, 2]

    calls_2: list[int] = []

    async def cap_handler(request):
        page = int(request.url.params["page"])
        calls_2.append(page)
        return httpx.Response(200, json=[_label(f"label-{page}-{i}") for i in range(100)], request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(cap_handler)) as http:
        client = GitHubClient(cfg, http=http, token_provider=fake_token_provider)
        with pytest.raises(GitHubError, match="pagination exceeded 20 pages"):
            await client.list_repository_labels("4paradigm/phanthymotus")
    assert calls_2 == list(range(1, 21))


@pytest.mark.asyncio
async def test_create_repository_label_uses_repository_labels_post_only(tmp_path):
    cfg = _bootstrap_config(tmp_path)
    requests: list[tuple[str, str, dict]] = []

    async def handler(request):
        requests.append(
            (request.method, request.url.path, json.loads(request.content.decode("utf-8")))
        )
        return httpx.Response(201, json={"name": "status: reviewing"}, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = GitHubClient(cfg, http=http, token_provider=fake_token_provider)
        out = await client.create_repository_label(
            "4paradigm/phanthymotus",
            "status: reviewing",
            "5319E7",
            "Deploy Approval: current HEAD is being reviewed",
        )
        assert out["name"] == "status: reviewing"

    assert requests == [
        (
            "POST",
            "/repos/4paradigm/phanthymotus/labels",
            {
                "name": "status: reviewing",
                "color": "5319E7",
                "description": "Deploy Approval: current HEAD is being reviewed",
            },
        )
    ]
