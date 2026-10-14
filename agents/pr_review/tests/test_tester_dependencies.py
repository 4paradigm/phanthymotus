"""Perception's schema tests need their test-only dependency in both indexes."""

from agents.pr_review.config import Config
from agents.pr_review.tester import _inner_script


def test_perception_installs_schema_validator_before_pytest_with_fallback():
    script = _inner_script('perception', Config(test_pypi_index='https://mirror.example/simple/'))
    installer, pytest = script.splitlines()[1:]
    assert installer.count('jsonschema==4.19.2') == 2
    assert 'https://pypi.org/simple/' in installer
    assert '|| exit 97' in installer
    assert '-m pytest tests' in pytest


def test_core_does_not_receive_perception_only_test_dependency():
    script = _inner_script('agent-core', Config())
    assert 'jsonschema' not in script
