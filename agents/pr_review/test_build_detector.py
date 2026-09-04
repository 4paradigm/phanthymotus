from agents.pr_review.build_detector import detect_targets
from agents.pr_review.components import build_context
from agents.pr_review.models import BuildTarget


def test_memory_core_source_change_builds_core():
    targets, driver_paths = detect_targets(
        '4paradigm/phanthymotus',
        ['memory-core/src/memory_core/repository.py'],
    )

    assert targets == [BuildTarget.CORE]
    assert driver_paths == []


def test_memory_core_readme_only_does_not_build():
    targets, driver_paths = detect_targets(
        '4paradigm/phanthymotus',
        ['memory-core/README.md'],
    )

    assert targets == []
    assert driver_paths == []


def test_memory_core_keeps_mixed_target_order_stable():
    targets, _ = detect_targets(
        '4paradigm/phanthymotus',
        [
            'perception/src/server.py',
            'memory-core/pyproject.toml',
            'actucore/src/server.py',
        ],
    )

    assert targets == [
        BuildTarget.CORE,
        BuildTarget.PERCEPTION,
        BuildTarget.ACTUCORE,
    ]


def test_memory_core_change_uses_agent_core_review_context():
    changed_files = ['memory-core/src/memory_core/repository.py']
    context = build_context(
        '4paradigm/phanthymotus',
        [BuildTarget.CORE],
        [],
        changed_files,
    )

    assert context.name == 'agent-core'
    assert context.rule_files == ['common.md', 'core.md']
    assert context.docs == [
        'CONTRIBUTING.md',
        'README.md',
        'memory-core/README.md',
    ]
