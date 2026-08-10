from __future__ import annotations

import asyncio
import importlib
from pathlib import Path

import pytest

import auth
import config
from event.desktop import DesktopTools


event_llm = importlib.import_module('event.llm')


def _desktop_tool_names() -> set[str]:
    return {
        name
        for name, _function in event_llm._desktop_tool_functions(DesktopTools())
    }


def test_authenticated_deployment_exposes_only_read_only_local_file_tools():
    auth.init({
        'ACCESS_TOKEN': 'owner-token',
        'MOTUS_DRIVER_TOKEN': 'driver-token',
        'MOTUS_TELEOP_TICKET_SECRET': 't' * 32,
    })

    names = _desktop_tool_names()

    assert {'Read', 'Glob', 'Grep'} <= names
    assert {'Bash', 'PythonExec', 'Write', 'Edit', 'WebFetch'}.isdisjoint(names)


def test_unsafe_code_and_mutation_tools_require_explicit_secret_free_opt_in():
    auth.init({'MOTUS_ENABLE_UNSAFE_DESKTOP_CODE_TOOLS': 'true'})
    assert {
        'Bash', 'PythonExec', 'Write', 'Edit', 'WebFetch',
    } <= _desktop_tool_names()
    auth.init({})

    for secret_settings in (
        {'ACCESS_TOKEN': 'owner-token'},
        {'MOTUS_DRIVER_TOKEN': 'driver-token'},
        {'MOTUS_TELEOP_TICKET_SECRET': 't' * 32},
    ):
        with pytest.raises(ValueError, match='cannot be enabled'):
            auth.init({
                **secret_settings,
                'MOTUS_ENABLE_UNSAFE_DESKTOP_CODE_TOOLS': 'true',
            })
        assert auth.unsafe_desktop_code_tools_enabled() is False


def test_file_tools_hide_dotenv_runtime_database_and_symlink_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    public = tmp_path / 'notes.txt'
    dotenv = tmp_path / '.env'
    database = tmp_path / 'runtime.db'
    dotenv_link = tmp_path / 'innocent-looking.txt'
    tls_key = tmp_path / 'server-material'
    tls_key_link = tmp_path / 'server-material-link'
    public.write_text('public marker', encoding='utf-8')
    dotenv.write_text('DRIVER_SENTINEL=must-not-leak', encoding='utf-8')
    database.write_text('DATABASE_SENTINEL=must-not-leak', encoding='utf-8')
    dotenv_link.symlink_to(dotenv)
    tls_key.write_text('TLS_KEY_SENTINEL=must-not-leak', encoding='utf-8')
    tls_key_link.hardlink_to(tls_key)
    monkeypatch.setattr(config, 'DB_PATH', str(database))
    monkeypatch.setenv('MOTUS_TLS_KEY_FILE', str(tls_key))

    tools = DesktopTools()
    monkeypatch.setattr(
        importlib.import_module('event.desktop'),
        '_ALLOWED_DIRS',
        [str(tmp_path)],
    )

    assert 'protected credential' in asyncio.run(tools.Read(str(dotenv)))
    assert 'protected credential' in asyncio.run(tools.Read(str(database)))
    assert 'protected credential' in asyncio.run(tools.Read(str(dotenv_link)))
    assert 'protected credential' in asyncio.run(tools.Read(str(tls_key_link)))

    grep = asyncio.run(tools.Grep('SENTINEL|public marker', str(tmp_path)))
    assert 'public marker' in grep
    assert 'DRIVER_SENTINEL' not in grep
    assert 'DATABASE_SENTINEL' not in grep
    assert 'TLS_KEY_SENTINEL' not in grep

    glob = asyncio.run(tools.Glob('*', str(tmp_path)))
    assert str(public) in glob
    assert str(dotenv) not in glob
    assert str(database) not in glob
    assert str(dotenv_link) not in glob
    assert str(tls_key) not in glob
    assert str(tls_key_link) not in glob


def test_python_exec_does_not_publish_open_builtin_even_in_unsafe_mode():
    tools = DesktopTools()

    result = asyncio.run(tools.PythonExec("print('open' in __builtins__)"))

    assert result.strip() == 'False'
