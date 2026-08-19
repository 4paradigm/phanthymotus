"""Reviewer — rule-based checks + LLM-powered code review."""

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import httpx

from .config import Config

logger = logging.getLogger(__name__)

# An empty or malformed completion is usually transient, so the call is retried
# before giving up and reporting on the PR.
LLM_ATTEMPTS = 3
LLM_RETRY_BACKOFF = 4


@dataclass
class Finding:
    severity: str  # "error", "warning", "info"
    file: str
    message: str


# ── Rule-based checks ──────────────────────────────────────────────────────────


# Files at or above this are reported separately; overridable via Config.
DEFAULT_LARGE_FILE_KB = 500

# The wrong *kind* of file to commit, whatever its size. Every real offender in
# these repos is under 1 MB — a committed .zip of message definitions, an x86_64
# .so in an ARM64-only project — so a size threshold alone would miss them.
BINARY_ARTIFACT_SUFFIXES = {
    ".tar", ".gz", ".tgz", ".bz2", ".xz", ".zip", ".7z", ".rar",
    ".so", ".a", ".o", ".dylib", ".dll",
    ".pt", ".pth", ".onnx", ".tflite", ".engine", ".bin", ".safetensors",
    ".whl", ".jar", ".deb", ".rpm",
}

# Shared bases: a change here reaches far past the PR's own component, and
# ros-base reaches across repositories.
SHARED_BASE_PATHS = {
    "deploy/ros-base/Dockerfile",
    "dji/base/Dockerfile",
    "dji/base/build.sh",
}
SHARED_BASE_PREFIXES = ("common/",)

INFRA_NAMES = {
    "requirements.txt", "pyproject.toml", "uv.lock", "driver.yaml",
    "service.yml", "docker-compose.yml", ".dockerignore",
}


def run_rule_checks(
    changed_files: list[str],
    diff_stat: str = "",
    worktree: Path | None = None,
    large_file_kb: int = DEFAULT_LARGE_FILE_KB,
) -> list[Finding]:
    """Run fast, deterministic checks on the changeset.

    `worktree` enables exact file sizes. Without it the size check is skipped
    rather than guessed at — the previous version inferred sizes from
    `git diff --stat`, which only reports bytes for *binary* files, so a 2 MB
    generated JSON or a vendored .py passed silently.
    """
    findings = []
    findings.extend(_check_infrastructure(changed_files))
    findings.extend(_check_file_sizes(changed_files, worktree, large_file_kb))
    findings.extend(_check_binary_artifacts(changed_files))
    findings.extend(_check_sensitive_files(changed_files))
    return findings


def large_files(
    changed_files: list[str],
    worktree: Path | None,
    large_file_kb: int = DEFAULT_LARGE_FILE_KB,
) -> list[tuple[str, int]]:
    """(path, bytes) for changed files at or above the threshold, largest first."""
    if worktree is None:
        return []
    out = []
    limit = large_file_kb * 1024
    for f in changed_files:
        size = _size_of(worktree, f)
        if size >= limit:
            out.append((f, size))
    return sorted(out, key=lambda t: -t[1])


def infra_files(changed_files: list[str]) -> tuple[list[str], list[str]]:
    """(all infrastructure files touched, those that are shared bases)."""
    infra, shared = [], []
    for f in changed_files:
        name = Path(f).name
        # Anything under a shared path counts as infrastructure whatever it is
        # called — `common/` is ordinary .py that every driver imports, so a
        # filename test alone would miss the highest-blast-radius changes.
        is_shared = f in SHARED_BASE_PATHS or f.startswith(SHARED_BASE_PREFIXES)
        is_infra = is_shared or (
            name == "Dockerfile"
            or name.startswith("Dockerfile.")
            or name in INFRA_NAMES
            or (name.startswith("build") and name.endswith(".sh"))
        )
        if not is_infra:
            continue
        infra.append(f)
        if is_shared:
            shared.append(f)
    return infra, shared


def _size_of(worktree: Path, rel: str) -> int:
    try:
        p = worktree / rel
        # A deleted file is not a size finding.
        return p.stat().st_size if p.is_file() else 0
    except OSError:
        return 0


def _check_infrastructure(changed_files: list[str]) -> list[Finding]:
    """Flag infrastructure changes, escalating for shared bases."""
    infra, shared = infra_files(changed_files)
    findings = []
    for f in infra:
        if f in shared:
            findings.append(Finding(
                severity="error",
                file=f,
                message=(
                    "Shared build infrastructure — every component that builds "
                    "on this is affected, across both repositories. Needs an "
                    "explicit justification."
                ),
            ))
        else:
            findings.append(Finding(
                severity="warning",
                file=f,
                message=(
                    "Infrastructure change — confirm it is necessary and does "
                    "not grow the image (minimal-change principle)"
                ),
            ))
    return findings


def _check_file_sizes(
    changed_files: list[str], worktree: Path | None, large_file_kb: int
) -> list[Finding]:
    """Exact sizes from disk, for text and binary alike."""
    findings = []
    for f, size in large_files(changed_files, worktree, large_file_kb):
        findings.append(Finding(
            severity="error",
            file=f,
            message=(
                f"{size / 1024:.0f}KB — over the {large_file_kb}KB limit. Large "
                "assets belong in COS "
                "(agi-phanthy-dev-1252788780.cos.ap-beijing.myqcloud.com/public/) "
                "and should be fetched by URL, e.g. a Dockerfile ARG as "
                "unitree/g1 does for cyclonedds."
            ),
        ))
    return findings


def _check_binary_artifacts(changed_files: list[str]) -> list[Finding]:
    """Flag committed archives and binaries regardless of size."""
    findings = []
    for f in changed_files:
        p = Path(f)
        suffixes = {s.lower() for s in p.suffixes[-2:]} or {p.suffix.lower()}
        hit = suffixes & BINARY_ARTIFACT_SUFFIXES
        if hit:
            findings.append(Finding(
                severity="warning",
                file=f,
                message=(
                    f"Committed binary/archive ({', '.join(sorted(hit))}) — these "
                    "belong in COS and should be fetched at build time. Note the "
                    "repo's existing offenders are all under 1MB, so size alone "
                    "does not catch them."
                ),
            ))
    return findings


# Filename fragments that mean "this file may carry a credential".
#
# Shared with the review sandbox (tools.py), which refuses to *read* anything
# matching — the check below reports a committed secret, and the sandbox makes
# sure the reviewer is not the thing that republishes it into a public PR
# comment or the dashboard.
#
# Split in two because the two halves need different strictness:
#
# - Suffix/whole-name matches are credential *file formats*. Nothing legitimate
#   is called `.pem` or `id_rsa`, so these are refused unconditionally.
# - Substring matches ("secret", "credentials") describe a file's *subject*, and
#   plenty of ordinary source code is about credentials without containing any.
#   Applying them to source would refuse `secret_manager.py` and the vendored
#   CycloneDDS header `dds_security_shared_secret.h`, i.e. make real code
#   unreviewable, so these skip files with a source or docs extension.
SENSITIVE_NAME_PARTS = frozenset({
    ".env", "credentials", "secret",
    ".pem", ".key", "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519",
    ".p12", ".pfx", ".keystore", ".jks", ".ppk",
    ".netrc", ".htpasswd", ".pgpass", ".kdbx",
})

# The subject-matter half — only these get the source-extension exemption.
_SUBJECT_PARTS = frozenset({"credentials", "secret"})

# Extensions where a real credential file is implausible but review value is
# high. A hardcoded token in one of these is still visible: it shows up in the
# diff, which `file_diff` reads, and the rule checks flag the file by name.
REVIEWABLE_SUFFIXES = (
    ".py", ".pyi", ".js", ".mjs", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java",
    ".c", ".cc", ".cpp", ".h", ".hpp", ".hh", ".sh", ".zsh", ".bash",
    ".md", ".rst", ".proto", ".urdf", ".xacro", ".dockerfile",
)

# Suffixes that make a sensitive-looking name a template rather than a secret.
# `deploy/*/.env.example` is how every deployment here is documented, so these
# stay readable and reviewable.
TEMPLATE_SUFFIXES = (".example", ".sample", ".template", ".dist")


def is_sensitive_name(name: str) -> bool:
    """Whether a bare filename looks like it carries a credential."""
    lowered = name.lower()
    if lowered.endswith(TEMPLATE_SUFFIXES):
        return False
    reviewable = lowered.endswith(REVIEWABLE_SUFFIXES)
    for part in SENSITIVE_NAME_PARTS:
        if part not in lowered:
            continue
        if reviewable and part in _SUBJECT_PARTS:
            continue
        return True
    return False


def _check_sensitive_files(changed_files: list[str]) -> list[Finding]:
    """Check for potentially sensitive files being committed."""
    findings = []
    for f in changed_files:
        name_lower = Path(f).name.lower()
        if name_lower.endswith(TEMPLATE_SUFFIXES):
            continue
        for pattern in sorted(SENSITIVE_NAME_PARTS):
            if pattern in name_lower:
                findings.append(Finding(
                    severity="error",
                    file=f,
                    message=(
                        f"File may contain secrets (matched: {pattern}) — "
                        "verify nothing sensitive is committed"
                    ),
                ))
                break
    return findings


# ── LLM Review ─────────────────────────────────────────────────────────────────

class TransientLLMError(RuntimeError):
    """An LLM call that failed for a reason that may not recur.

    Separated from a permanent failure because the two want opposite handling:
    a bad key or a wrong model name will fail identically forever and should
    surface immediately, while a gateway hiccup on round 9 of a review throws
    away everything the reviewer had learned. Callers that can retry catch this;
    callers that cannot still see a RuntimeError with the same message.
    """


# Statuses worth retrying. 5xx and above is the broad case, and it deliberately
# covers non-standard codes: `router.phanthy.com` answers 666 with
# `bad_response_status_code` wrapping an upstream `openai_error`, which is a
# transient upstream fault wearing a status code no client would special-case.
# 408/429 are the sub-500 ones that mean "try again", and 529 is Anthropic's
# overloaded signal.
RETRYABLE_STATUSES = frozenset({408, 425, 429, 529})


def is_retryable_status(status: int) -> bool:
    return status >= 500 or status in RETRYABLE_STATUSES


def describe_http_failure(resp: "httpx.Response", endpoint: str) -> dict:
    """Parse a completion response, raising errors that say what happened.

    Parsing is attempted regardless of content-type: gateways commonly return a
    valid completion labelled `text/plain`, and rejecting those on the header
    alone throws away a perfectly good response. Content-type is used only to
    *explain* a failure — HTML almost always means the URL is wrong.

    The original version raised for status and called .json(), so a gateway
    answering 200 with its web front-end surfaced as "Expecting value: line 1
    column 1" — a symptom that hid the cause and took a shell session to find.
    Shared by the review loop, which needs the whole message (tool calls
    included) rather than just its text.
    """
    ctype = resp.headers.get("content-type", "unknown")
    snippet = resp.text[:200].replace("\n", " ").strip()

    if resp.status_code >= 400:
        message = f"HTTP {resp.status_code} ({ctype}) from {endpoint}: {snippet}"
        if is_retryable_status(resp.status_code):
            raise TransientLLMError(message)
        raise RuntimeError(message)

    try:
        data = json.loads(resp.text)
    except ValueError as e:
        hint = ""
        if "html" in ctype.lower() or snippet.lstrip().startswith("<"):
            hint = (
                " The response is HTML, so this URL is probably serving a web "
                "page rather than the API — an OpenAI-compatible endpoint lives "
                "at /v1/chat/completions."
            )
        raise RuntimeError(
            f"HTTP {resp.status_code} ({ctype}) from {endpoint} was not JSON "
            f"({e}).{hint} Body: {snippet}"
        ) from e

    return data


def explain_empty_completion(data: dict) -> str:
    """Why a completion came back with neither text nor tool calls.

    Kept from the single-call reviewer, where an empty completion posted a blank
    "Code Review" comment on a PR — worse than saying nothing, because it reads
    as "the reviewer had no comments". In the loop this becomes the recorded
    error when the model keeps returning nothing.
    """
    try:
        choice = data["choices"][0]
    except (KeyError, IndexError, TypeError):
        return "response contained no choices"
    finish = choice.get("finish_reason")
    usage = data.get("usage", {}) or {}
    hint = {
        "length": " The token budget was exhausted before any text was produced "
                  "— raise LLM_MAX_TOKENS.",
        "content_filter": " The response was filtered.",
    }.get(finish, "")
    return (
        f"the model returned no content (finish_reason={finish!r}, "
        f"prompt_tokens={usage.get('prompt_tokens')}, "
        f"completion_tokens={usage.get('completion_tokens')}).{hint}"
    )


def chat_completions_url(base: str) -> str:
    """Build the chat-completions endpoint from a configured base URL.

    Accepts the three forms people actually configure, because requiring one
    exact spelling is how this broke: `https://router.phanthy.com` produced
    `/chat/completions`, which on that host is the web UI, not the API.

        https://host                     -> https://host/v1/chat/completions
        https://host/v1                  -> https://host/v1/chat/completions
        https://host/v1/chat/completions -> unchanged
    """
    url = (base or "").rstrip("/")
    if not url:
        return ""
    if url.endswith("/chat/completions"):
        return url
    if re.search(r"/v\d+$", url):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"

