"""A skill can carry files; nothing in the declaration says who reads them.

The LLM-facing half of this is a **read primitive on a robot** whose path
argument ultimately comes from text in a prompt, so the confinement tests below
are the point of this file. Two path bugs were already found in this feature
from the other direction (a `..` check a zip library had normalised away, and a
separator check that missed `..\\`), which is why the checks here are duplicated
at declaration time and at read time rather than trusting either one.

Run: cd agent-core && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \\
         python3 -m pytest tests/test_skill_files.py -q
"""
import hashlib
import os
import pathlib
import sys
import tempfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'skill-files.db'))

import config                                    # noqa: E402
import skill_files                                # noqa: E402
from api.skills import validate_local_skill       # noqa: E402

TOOL = 'example_recorder'
SLUG = 'wave-hand'


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def definition(files, required=(TOOL,)):
    return {'slug': SLUG, 'name': '挥手', 'oneLiner': '让机器人挥个手',
            'instruction': '判断场合时读 reference/etiquette.md', 'category': 'robotics',
            'requiredTools': list(required), 'files': files}


REFERENCE = '握手前先确认对方伸手。'.encode()
TEXT_FILE = {'path': 'reference/etiquette.md',
             'bytes': len(REFERENCE), 'sha256': sha(REFERENCE)}
TRAJECTORY = b'{"t":0,"q":[0,0]}\n'
BINARY_FILE = {'path': 'trajectory.jsonl',
               'bytes': len(TRAJECTORY), 'sha256': sha(TRAJECTORY)}


@pytest.fixture
def store(monkeypatch, tmp_path):
    monkeypatch.setattr(config, 'main', {'skills': {'installed': []}})
    monkeypatch.setattr(skill_files, 'root', lambda: tmp_path / 'skill_files')
    return tmp_path / 'skill_files'


def test_a_declaration_says_what_it_ships_and_nothing_about_who_reads_it(store):
    """No consumer, no protocol, no tool name -- by design.

    Earlier drafts had all three. `tools` duplicated what `schema` carried and
    forced a marketplace author to enumerate driver tool names they cannot know;
    `schema`/`consumer` moved a per-call routing decision into an authoring-time
    binding, which nothing else in a skill does (`requiredTools` is enforced
    nowhere -- one display label is its only reader).
    """
    out = validate_local_skill(definition([TEXT_FILE, BINARY_FILE]))
    assert out['files'] == [TEXT_FILE, BINARY_FILE]
    for rejected in ('consumers', 'consumer', 'schema', 'tools', 'tool'):
        with pytest.raises(Exception) as caught:
            validate_local_skill(definition([{**TEXT_FILE, rejected: ['x']}]))
        assert rejected in caught.value.detail, rejected


def test_a_skill_can_carry_files_without_declaring_any_tool(store):
    """The marketplace case: the author does not know the driver's tool names."""
    out = validate_local_skill(definition([TEXT_FILE], required=()))
    assert out['files'][0]['path'] == TEXT_FILE['path']
    assert out['requiredTools'] == []


@pytest.mark.parametrize('bad', [
    {'path': '../escape.md'}, {'path': '/abs.md'}, {'path': 'a/../../escape.md'},
    {'path': '..\\escape.md'}, {'path': 'a/./b.md'}, {'path': '.hidden'},
    {'path': 'a/b/c/d/e/f.md'}, {'path': ''}, {'path': 'x' * 201},
])
def test_a_declared_path_that_could_escape_is_refused(store, bad):
    with pytest.raises(Exception) as caught:
        validate_local_skill(definition([{**TEXT_FILE, **bad}]))
    assert caught.value.status_code == 422


@pytest.mark.parametrize('bad', [
    {'consumers': 'model'}, {'consumers': []}, {'consumers': ['model', 'model']},
    {'consumers': ['nobody']}, {'bytes': 0}, {'bytes': '12'}, {'sha256': 'nope'},
])
def test_a_malformed_entry_is_refused(store, bad):
    with pytest.raises(Exception) as caught:
        validate_local_skill(definition([{**TEXT_FILE, **bad}]))
    assert caught.value.status_code == 422


def test_duplicate_paths_are_refused(store):
    with pytest.raises(Exception):
        validate_local_skill(definition([TEXT_FILE, dict(TEXT_FILE)]))


def test_an_unsupported_field_is_refused(store):
    with pytest.raises(Exception) as caught:
        validate_local_skill(definition([{**TEXT_FILE, 'url': 'http://x/y'}]))
    assert 'url' in caught.value.detail


# ── storage and confinement ──────────────────────────────────────────────────

def test_content_must_match_the_declaration(store):
    skill = validate_local_skill(definition([TEXT_FILE]))
    skill_files.write(SLUG, skill, TEXT_FILE['path'], REFERENCE)
    text, info = skill_files.read_text(SLUG, skill, TEXT_FILE['path'])
    assert text == REFERENCE.decode() and info['bytes'] == len(REFERENCE)
    # Declaration first, content second: neither a different length nor
    # different bytes of the same length can be substituted.
    for wrong in (REFERENCE + b'!', bytes(len(REFERENCE))):
        with pytest.raises(ValueError):
            skill_files.write(SLUG, skill, TEXT_FILE['path'], wrong)


def test_a_binary_file_is_handed_over_by_path_not_read_as_text(store, monkeypatch):
    """The device half, and it needs no new transport.

    `skill_file(..., as_path=True)` returns a local path; the LLM puts it in a
    tool's `format: file` parameter and `mcp_client._transfer_file_args` pushes
    it to that tool's service. Which tool the model called *is* the destination
    -- the skill declares none.
    """
    skill = validate_local_skill(definition([BINARY_FILE]))
    skill_files.write(SLUG, skill, BINARY_FILE['path'], TRAJECTORY)
    handed = skill_files.local_path(SLUG, skill, BINARY_FILE['path'])
    assert os.path.isabs(handed) and os.path.isfile(handed)
    # `_transfer_file_args` keys off exactly this: a local file at that path.
    import mcp_client
    schema = {'properties': {'trajectory': {'type': 'string', 'format': 'file'}}}
    sent = {}

    async def fake_push(mcp_id, url, local):
        sent['local'] = local
        return '/work/uploads/trajectory.jsonl'
    monkeypatch.setattr(mcp_client, '_push_file', fake_push)
    import asyncio
    args, error = asyncio.run(mcp_client._transfer_file_args(
        'robot-a', 'http://x/mcp', schema, {'trajectory': handed}))
    assert error is None
    assert sent['local'] == handed
    assert args['trajectory'] == '/work/uploads/trajectory.jsonl'


def test_a_file_too_large_for_text_says_to_hand_it_over_instead(store, monkeypatch):
    monkeypatch.setattr(skill_files, 'MAX_TEXT_BYTES', 4)
    skill = validate_local_skill(definition([BINARY_FILE]))
    skill_files.write(SLUG, skill, BINARY_FILE['path'], TRAJECTORY)
    with pytest.raises(ValueError, match='as_path'):
        skill_files.read_text(SLUG, skill, BINARY_FILE['path'])
    # But handing it over still works -- that is the point of the message.
    assert skill_files.local_path(SLUG, skill, BINARY_FILE['path'])


def test_an_undeclared_file_has_no_storage_and_no_reader(store):
    skill = validate_local_skill(definition([TEXT_FILE]))
    with pytest.raises(ValueError, match='未声明'):
        skill_files.write(SLUG, skill, 'reference/other.md', b'x')
    with pytest.raises(ValueError, match='未声明'):
        skill_files.read_text(SLUG, skill, 'reference/other.md')
    with pytest.raises(ValueError, match='未声明'):
        skill_files.local_path(SLUG, skill, 'reference/other.md')


@pytest.mark.parametrize('escape', [
    '../../etc/passwd', '/etc/passwd', 'reference/../../../etc/passwd',
    '..\\..\\windows', 'reference/./../../x',
])
def test_resolve_refuses_to_leave_the_skill_directory(store, escape):
    with pytest.raises(ValueError):
        skill_files.resolve(SLUG, escape)


def test_resolve_stays_inside_even_for_a_lookalike_sibling_directory(store):
    """`wave-hand-evil` starts with the same string as `wave-hand`.

    A containment check written as `str(target).startswith(str(base))` passes
    for `.../skill_files/wave-hand-evil/x` -- the sibling escape that prefix
    comparison always misses. This asserts the parents check instead.
    """
    inside = skill_files.resolve(SLUG, 'reference/etiquette.md')
    assert skill_files._slug_dir(SLUG).resolve() in inside.parents
    sibling = skill_files._slug_dir(SLUG + '-evil').resolve()
    assert sibling not in inside.parents and inside != sibling


@pytest.mark.parametrize('slug', ['../other', 'a/b', '/abs', '', 'UPPER', 'x' * 101])
def test_a_malformed_slug_cannot_address_a_directory(store, slug):
    with pytest.raises(ValueError):
        skill_files.resolve(slug, 'reference/etiquette.md')


def test_a_symlink_planted_in_the_store_is_not_followed(store):
    """The store directory is writable by this process, so a symlink there must
    not turn a skill-file read into a read of something else."""
    skill = validate_local_skill(definition([TEXT_FILE]))
    skill_files.write(SLUG, skill, TEXT_FILE['path'], REFERENCE)
    target = skill_files.resolve(SLUG, TEXT_FILE['path'])
    secret = store.parent / 'secret.txt'
    secret.write_text('not for the model')
    target.unlink()
    os.symlink(secret, target)
    with pytest.raises(ValueError):
        skill_files.read_text(SLUG, skill, TEXT_FILE['path'])


def test_a_long_file_is_read_in_bounded_chunks(store, monkeypatch):
    monkeypatch.setattr(skill_files, 'MAX_READ_BYTES', 8)
    body = b'0123456789abcdef'
    entry = {'path': 'reference/long.md', 'bytes': len(body), 'sha256': sha(body)}
    skill = validate_local_skill(definition([entry]))
    skill_files.write(SLUG, skill, entry['path'], body)
    text, info = skill_files.read_text(SLUG, skill, entry['path'])
    assert text == '01234567' and info['truncated'] and info['next_offset'] == 8
    text, info = skill_files.read_text(SLUG, skill, entry['path'], offset=8)
    assert text == '89abcdef' and not info['truncated']


# ── the LLM-facing tool ──────────────────────────────────────────────────────

def test_the_tool_requires_the_skill_to_be_activated(store):
    import asyncio
    # `event/__init__.py` rebinds the attribute to a `Tools()` instance, so
    # `event.skills` is the instance and sys.modules holds the real module —
    # the module globals (`_runtime_activated`) only exist on the latter.
    import event.skills                       # noqa: F401 - ensure it is imported
    skills_mod = sys.modules['event.skills']
    skill = validate_local_skill(definition([TEXT_FILE]))
    config.main['skills'] = {'installed': [{**skill, 'active': True}]}
    skill_files.write(SLUG, skill, TEXT_FILE['path'], REFERENCE)
    tools = skills_mod.Tools()

    # Visible but not activated: activation is the gate, so this cannot be used
    # to browse installed skills.
    skills_mod._runtime_activated.discard(SLUG)
    answer = asyncio.run(tools.skill_file(SLUG, TEXT_FILE['path']))
    assert '未激活' in answer and REFERENCE.decode() not in answer

    asyncio.run(tools.activate_skill(SLUG))
    assert REFERENCE.decode() in asyncio.run(tools.skill_file(SLUG, TEXT_FILE['path']))
    skills_mod._runtime_activated.discard(SLUG)


def test_the_tool_reports_what_is_readable_when_a_path_is_wrong(store):
    """A refusal has to say what *is* available, or the model retries blindly."""
    import asyncio
    # `event/__init__.py` rebinds the attribute to a `Tools()` instance, so
    # `event.skills` is the instance and sys.modules holds the real module —
    # the module globals (`_runtime_activated`) only exist on the latter.
    import event.skills                       # noqa: F401 - ensure it is imported
    skills_mod = sys.modules['event.skills']
    skill = validate_local_skill(definition([TEXT_FILE, BINARY_FILE]))
    config.main['skills'] = {'installed': [{**skill, 'active': True}]}
    skills_mod._runtime_activated.add(SLUG)
    answer = asyncio.run(skills_mod.Tools().skill_file(SLUG, 'reference/missing.md'))
    assert 'reference/etiquette.md' in answer and BINARY_FILE['path'] in answer
    skills_mod._runtime_activated.discard(SLUG)


def test_the_tool_does_not_rebuild_the_system_prompt(store):
    """`skill_file` returns a tool result; it changes nothing *in* the prompt.

    Listing it alongside activate/deactivate would rebuild the frozen system
    prompt on every file read and discard the prompt cache for nothing.
    """
    source = pathlib.Path(__file__).resolve().parents[1] / 'src/event/llm.py'
    body = source.read_text(encoding='utf-8')
    line = next(l for l in body.splitlines() if 'skill_tools = {' in l)
    assert 'skill_file' not in line
    assert "('skill_file', event.skills.skill_file)" in body, '但它必须注册为系统工具'
