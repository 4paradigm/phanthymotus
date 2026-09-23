"""Shared fixtures and import-time stubs for the agent-core suite.

The docker SDK ships in the agent-core image but is not on a dev host, and
several modules under `api/` import it at module scope. Until this file
existed, one test module (`test_deploy_pull_errors.py`) installed a stub into
`sys.modules` as a side effect of being imported, and every other docker-touching
test worked only because pytest happened to collect that file first — an
ordering nobody declared and `-k`, `-p no:randomly` or a renamed file could
break. `test_deploy_progress.py` was in exactly that position: green in a full
run, `ModuleNotFoundError` on its own.

Stubbing here happens once, before any test module is imported, and only when
the real SDK is absent — so the image still tests against the real thing.
"""

from __future__ import annotations

import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def _install_docker_stub() -> None:
    try:
        import docker  # noqa: F401
        return
    except ImportError:
        pass

    docker = types.ModuleType('docker')
    errors = types.ModuleType('docker.errors')

    class DockerException(Exception):
        pass

    class APIError(DockerException):
        pass

    class NotFound(APIError):
        pass

    class ImageNotFound(NotFound):
        pass

    # Distinct classes, unlike the ad-hoc stub this replaces: tests that assert
    # a missing *container* and a missing *image* are handled differently
    # cannot do so if both names point at one exception.
    errors.DockerException = DockerException
    errors.APIError = APIError
    errors.NotFound = NotFound
    errors.ImageNotFound = ImageNotFound

    def from_env(*_args, **_kwargs):
        raise DockerException(
            'docker SDK is stubbed in this test run; patch '
            '`<module>.docker_sdk.from_env` if this call is meant to succeed'
        )

    docker.errors = errors
    docker.from_env = from_env
    sys.modules['docker'] = docker
    sys.modules['docker.errors'] = errors


_install_docker_stub()
