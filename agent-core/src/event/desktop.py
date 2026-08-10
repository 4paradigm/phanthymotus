"""
event/desktop.py — 通用桌面 Agent 工具。

提供文件操作、Shell 执行、Python 沙盒、内容搜索、URL 抓取等基础能力，
工具命名与 Claude Code 保持一致以最大化 LLM 调用准确率。
"""

import asyncio
import fnmatch
import io
import os
import pathlib
import re
import sys
import typing

import log
import config

# ── 安全配置 ─────────────────────────────────────────────────────────────────────

_MAX_OUTPUT = 51200  # 50 KB output cap

_ALLOWED_DIRS = ['/work', '/tmp']

_BASH_BLOCKED = [
    'rm -rf /', 'rm -rf /*', 'mkfs', 'reboot', 'shutdown', 'poweroff',
    'dd if=', 'dd of=/dev',
]

_PYTHON_ALLOWED_MODULES = {
    'math', 'json', 're', 'datetime', 'collections', 'itertools',
    'struct', 'pathlib', 'numpy', 'hashlib', 'base64', 'urllib.parse',
    'string', 'textwrap', 'functools', 'operator', 'copy', 'time',
    'random', 'statistics', 'fractions', 'decimal', 'enum', 'typing',
    'dataclasses', 'csv', 'io', 'os.path',
}

_PROTECTED_FILE_SUFFIXES = {'.key', '.pem', '.p12', '.pfx'}


def _path_is_protected(p: pathlib.Path) -> bool:
    """Keep deployment credentials and the runtime config DB out of file tools."""

    name = p.name.lower()
    if name == '.env' or name.startswith('.env.'):
        return True
    if (
        name in {'id_rsa', 'id_ed25519', 'credentials.json', 'secrets.json'}
        or 'privkey' in name
        or 'private-key' in name
    ):
        return True
    if p.suffix.lower() in _PROTECTED_FILE_SUFFIXES:
        return True

    protected_paths = {
        pathlib.Path('/opt/phanthy-motus/.env'),
        pathlib.Path('.env'),
    }
    tls_key_path = os.environ.get('MOTUS_TLS_KEY_FILE', '').strip()
    if tls_key_path:
        protected_paths.add(pathlib.Path(tls_key_path))
    try:
        config_db = pathlib.Path(config.DB_PATH).resolve()
        protected_paths.update({
            config_db,
            pathlib.Path(f'{config_db}-shm'),
            pathlib.Path(f'{config_db}-wal'),
            pathlib.Path(f'{config_db}-journal'),
        })
    except (OSError, RuntimeError, TypeError):
        pass
    for protected in protected_paths:
        try:
            protected = protected.resolve()
            if p == protected:
                return True
            if p.exists() and protected.exists() and os.path.samefile(p, protected):
                return True
        except (OSError, RuntimeError):
            continue
    return False


def _resolve_path(path: str) -> pathlib.Path:
    """Resolve path, default relative to /work."""
    p = pathlib.Path(path)
    if not p.is_absolute():
        p = pathlib.Path('/work') / p
    return p.resolve()


def _check_path_allowed(p: pathlib.Path, dirs: list[str] | None = None) -> str | None:
    """Return error message if path is not within allowed dirs, else None."""
    try:
        p = p.resolve()
    except (OSError, RuntimeError):
        return f'Access denied: cannot safely resolve path {p}'
    if _path_is_protected(p):
        return f'Access denied: {p} is a protected credential or runtime config path'
    allowed = dirs or _ALLOWED_DIRS
    s = str(p)
    for d in allowed:
        if s == d or s.startswith(d + '/'):
            return None
    return f'Access denied: {p} is outside allowed directories {allowed}'


def _truncate(text: str, max_bytes: int = _MAX_OUTPUT) -> str:
    """Truncate output to max bytes."""
    if len(text.encode('utf-8', errors='replace')) <= max_bytes:
        return text
    # Truncate by characters (approximate)
    encoded = text.encode('utf-8', errors='replace')[:max_bytes]
    truncated = encoded.decode('utf-8', errors='ignore')
    return truncated + f'\n\n... [output truncated at {max_bytes} bytes]'


# ── Tools class ──────────────────────────────────────────────────────────────────

class DesktopTools:
    """General-purpose desktop agent tools."""

    def __init__(self):
        self._python_namespace: dict = {}  # persists within turn

    def reset_python_namespace(self):
        """Reset Python namespace between turns."""
        self._python_namespace = {}

    # ── 1. Bash ──────────────────────────────────────────────────────────────

    @log.function_(call=True)
    async def Bash(self,
        command: typing.Annotated[str, 'The shell command to execute'],
        timeout: typing.Annotated[int, 'Timeout in seconds (default 30, max 120)'] = 30,
        cwd: typing.Annotated[str, 'Working directory (default /work)'] = '/work',
    ) -> str:
        """Execute a shell command and return stdout+stderr. Use for system monitoring (top/df/ps), package management, process control, network debugging. For file operations prefer Read/Write/Edit; for file search prefer Glob/Grep."""
        # Security: check blocked patterns
        cmd_lower = command.lower().strip()
        for blocked in _BASH_BLOCKED:
            if blocked in cmd_lower:
                return f'Error: command blocked for safety — contains "{blocked}"'

        if 'sudo ' in cmd_lower:
            return 'Error: sudo is not allowed'

        timeout = max(1, min(timeout, 120))

        # Resolve cwd
        cwd_path = pathlib.Path(cwd)
        if not cwd_path.exists():
            cwd_path = pathlib.Path('/work')

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(cwd_path),
                env={**os.environ, 'TERM': 'dumb'},
            )
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return f'Error: command timed out after {timeout}s and was killed'

            output = stdout.decode('utf-8', errors='replace')
            result = _truncate(output)
            if proc.returncode != 0:
                result = f'[exit code {proc.returncode}]\n{result}'
            return result or '(no output)'

        except Exception as e:
            return f'Error: {e}'

    # ── 2. PythonExec ────────────────────────────────────────────────────────

    @log.function_(call=True)
    async def PythonExec(self,
        code: typing.Annotated[str, 'Python code to execute'],
        timeout: typing.Annotated[int, 'Timeout in seconds (default 30)'] = 30,
    ) -> str:
        """Execute Python code in a sandboxed environment. Useful for calculations, data processing, JSON manipulation. Variables persist across calls within the same turn. Available: math, json, re, datetime, collections, itertools, struct, pathlib, numpy, hashlib, base64."""
        timeout = max(1, min(timeout, 60))

        # Build restricted builtins
        import builtins as _builtins
        safe_builtins = {
            k: getattr(_builtins, k)
            for k in dir(_builtins)
            if not k.startswith('_')
            and k not in ('exec', 'eval', 'compile', 'exit', 'quit', 'open')
        }

        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def _restricted_import(name, *args, **kwargs):
            top = name.split('.')[0]
            if top not in _PYTHON_ALLOWED_MODULES and name not in _PYTHON_ALLOWED_MODULES:
                raise ImportError(f"Module '{name}' is not allowed. Allowed: {sorted(_PYTHON_ALLOWED_MODULES)}")
            return original_import(name, *args, **kwargs)

        safe_builtins['__import__'] = _restricted_import
        safe_builtins['__builtins__'] = safe_builtins

        if not self._python_namespace:
            self._python_namespace = {'__builtins__': safe_builtins}
        else:
            self._python_namespace['__builtins__'] = safe_builtins

        # Capture stdout
        captured = io.StringIO()

        def _exec_code():
            old_stdout = sys.stdout
            sys.stdout = captured
            try:
                # Try as expression first (to capture result)
                try:
                    result = eval(compile(code, '<agent>', 'eval'), self._python_namespace)
                    if result is not None:
                        print(repr(result))
                except SyntaxError:
                    exec(compile(code, '<agent>', 'exec'), self._python_namespace)
            finally:
                sys.stdout = old_stdout

        try:
            # Run in thread pool with timeout
            loop = asyncio.get_event_loop()
            await asyncio.wait_for(
                loop.run_in_executor(None, _exec_code),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return f'Error: execution timed out after {timeout}s'
        except Exception as e:
            output = captured.getvalue()
            err_msg = f'{type(e).__name__}: {e}'
            return _truncate(f'{output}\n{err_msg}' if output else err_msg)

        output = captured.getvalue()
        return _truncate(output) if output else '(no output)'

    # ── 3. Read ──────────────────────────────────────────────────────────────

    @log.function_(call=True)
    async def Read(self,
        file_path: typing.Annotated[str, 'Absolute file path to read'],
        offset: typing.Annotated[int, 'Line number to start reading from (1-indexed)'] = 1,
        limit: typing.Annotated[int, 'Maximum number of lines to return (default 200)'] = 200,
    ) -> str:
        """Read file contents with line numbers. Supports reading specific line ranges for large files."""
        p = _resolve_path(file_path)

        err = _check_path_allowed(p)
        if err:
            return err

        if not p.exists():
            return f'Error: file not found: {p}'
        if not p.is_file():
            return f'Error: not a file: {p}'

        # Check if binary
        try:
            raw = p.read_bytes()
            if b'\x00' in raw[:8192]:
                hex_preview = raw[:256].hex(' ')
                return f'[Binary file, {len(raw)} bytes]\nHex preview (first 256 bytes):\n{hex_preview}'
            text = raw.decode('utf-8', errors='replace')
        except Exception as e:
            return f'Error reading file: {e}'

        lines = text.splitlines(keepends=True)
        total_lines = len(lines)
        file_size = len(raw)

        # Apply offset/limit
        offset = max(1, offset)
        limit = max(1, min(limit, 1000))
        selected = lines[offset - 1: offset - 1 + limit]

        # Format with line numbers
        header = f'[{p} | {file_size} bytes | {total_lines} lines | showing {offset}-{offset + len(selected) - 1}]\n'
        numbered = ''.join(f'{offset + i:>4}| {line}' for i, line in enumerate(selected))

        return _truncate(header + numbered)

    # ── 4. Write ─────────────────────────────────────────────────────────────

    @log.function_(call=True)
    async def Write(self,
        file_path: typing.Annotated[str, 'Absolute file path to write'],
        content: typing.Annotated[str, 'The content to write to the file'],
    ) -> str:
        """Create a new file or overwrite an existing file. Creates parent directories automatically. For modifying existing files, prefer Edit for surgical changes."""
        p = _resolve_path(file_path)

        err = _check_path_allowed(p)
        if err:
            return err

        # Size check (1 MB max)
        if len(content.encode('utf-8')) > 1_048_576:
            return 'Error: content exceeds 1 MB limit'

        try:
            p.parent.mkdir(parents=True, exist_ok=True)

            # Backup existing file > 1KB
            if p.exists() and p.stat().st_size > 1024:
                bak = p.with_suffix(p.suffix + '.bak')
                bak.write_bytes(p.read_bytes())

            p.write_text(content, encoding='utf-8')
            return f'Written {len(content)} bytes to {p}'
        except Exception as e:
            return f'Error: {e}'

    # ── 5. Edit ──────────────────────────────────────────────────────────────

    @log.function_(call=True)
    async def Edit(self,
        file_path: typing.Annotated[str, 'Absolute file path to modify'],
        old_string: typing.Annotated[str, 'The exact text to find and replace (must be unique in file)'],
        new_string: typing.Annotated[str, 'The replacement text'],
    ) -> str:
        """Replace a unique string in a file. The old_string must appear exactly once in the file. Use for surgical modifications to code and config files."""
        p = _resolve_path(file_path)

        err = _check_path_allowed(p)
        if err:
            return err

        if not p.exists():
            return f'Error: file not found: {p}'

        try:
            text = p.read_text(encoding='utf-8')
        except Exception as e:
            return f'Error reading file: {e}'

        count = text.count(old_string)
        if count == 0:
            return f'Error: old_string not found in {p}'
        if count > 1:
            return f'Error: old_string found {count} times (must be unique). Provide more context to make it unique.'

        new_text = text.replace(old_string, new_string, 1)

        # Backup
        if p.stat().st_size > 1024:
            bak = p.with_suffix(p.suffix + '.bak')
            bak.write_text(text, encoding='utf-8')

        p.write_text(new_text, encoding='utf-8')
        return f'Edited {p}: replaced {len(old_string)} chars with {len(new_string)} chars'

    # ── 6. Glob ──────────────────────────────────────────────────────────────

    @log.function_(call=True)
    async def Glob(self,
        pattern: typing.Annotated[str, 'Glob pattern (e.g. "**/*.py", "src/**/*.ts")'],
        path: typing.Annotated[str, 'Root directory to search in'] = '/work',
    ) -> str:
        """Find files matching a glob pattern. Returns sorted file paths, most recently modified first."""
        root = _resolve_path(path)

        err = _check_path_allowed(root)
        if err:
            return err

        if not root.exists() or not root.is_dir():
            return f'Error: directory not found: {root}'

        try:
            matches = sorted(
                (
                    match
                    for match in root.glob(pattern)
                    if _check_path_allowed(match) is None
                ),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        except Exception as e:
            return f'Error: {e}'

        if not matches:
            return f'No files matching "{pattern}" in {root}'

        limit = 200
        lines = [str(m) for m in matches[:limit]]
        result = '\n'.join(lines)
        if len(matches) > limit:
            result += f'\n\n... ({len(matches) - limit} more files not shown)'
        return result

    # ── 7. Grep ──────────────────────────────────────────────────────────────

    @log.function_(call=True)
    async def Grep(self,
        pattern: typing.Annotated[str, 'Regex pattern to search for'],
        path: typing.Annotated[str, 'File or directory to search in'] = '/work',
        include: typing.Annotated[str, 'File glob filter (e.g. "*.py")'] = '',
    ) -> str:
        """Search file contents using regex. Returns matching lines with file path and line number. Use for finding function definitions, config values, error messages."""
        root = _resolve_path(path)

        err = _check_path_allowed(root)
        if err:
            return err

        # A Python traversal makes the per-file protected-path check explicit;
        # recursive grep subprocesses can otherwise read hidden .env or SQLite
        # credential stores even when their root directory itself is allowed.
        return await self._grep_python(pattern, root, include)

    async def _grep_python(self, pattern: str, root: pathlib.Path, include: str) -> str:
        """Pure Python grep fallback."""
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f'Error: invalid regex — {e}'

        results = []
        max_results = 50

        def _search_file(fp: pathlib.Path):
            try:
                resolved = fp.resolve()
            except (OSError, RuntimeError):
                return
            if _check_path_allowed(resolved) is not None:
                return
            try:
                text = resolved.read_text(encoding='utf-8', errors='ignore')
                for i, line in enumerate(text.splitlines(), 1):
                    if regex.search(line):
                        results.append(f'{resolved}:{i}: {line.rstrip()}')
                        if len(results) >= max_results:
                            return
            except (OSError, UnicodeDecodeError):
                pass

        if root.is_file():
            _search_file(root)
        else:
            for fp in root.rglob('*'):
                if len(results) >= max_results:
                    break
                if not fp.is_file():
                    continue
                if include and not fnmatch.fnmatch(fp.name, include):
                    continue
                _search_file(fp)

        if not results:
            return f'No matches for pattern "{pattern}" in {root}'
        return _truncate('\n'.join(results))

    # ── 8. WebFetch ──────────────────────────────────────────────────────────

    @log.function_(call=True)
    async def WebFetch(self,
        url: typing.Annotated[str, 'URL to fetch'],
        prompt: typing.Annotated[str, 'What information to extract from the page (optional)'] = '',
        timeout: typing.Annotated[int, 'Timeout in seconds (default 30)'] = 30,
    ) -> str:
        """Fetch URL content. HTML is automatically converted to Markdown for readability. Optionally specify a prompt to describe what information you want to extract."""
        timeout = max(5, min(timeout, 60))

        try:
            import aiohttp
        except ImportError:
            return 'Error: aiohttp not installed. Run: pip install aiohttp'

        try:
            import html2text
            h2t = html2text.HTML2Text()
            h2t.ignore_links = False
            h2t.ignore_images = True
            h2t.body_width = 0
        except ImportError:
            h2t = None

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout),
                                       headers={'User-Agent': 'Mozilla/5.0 (compatible; AgentBot/1.0)'}) as resp:
                    if resp.status >= 400:
                        return f'Error: HTTP {resp.status} {resp.reason}'
                    content_type = resp.headers.get('content-type', '')
                    raw = await resp.read()

                    # Limit to 500KB raw
                    if len(raw) > 512_000:
                        raw = raw[:512_000]

                    text = raw.decode('utf-8', errors='replace')

                    # Convert HTML to markdown
                    if 'html' in content_type and h2t:
                        text = h2t.handle(text)

        except asyncio.TimeoutError:
            return f'Error: request timed out after {timeout}s'
        except Exception as e:
            return f'Error fetching URL: {e}'

        result = _truncate(text, max_bytes=_MAX_OUTPUT)

        if prompt:
            result = f'[Extracted from {url} — user asked: "{prompt}"]\n\n{result}'
        else:
            result = f'[Content from {url}]\n\n{result}'

        return result

    # ── 9. WebSearch ─────────────────────────────────────────────────────────

    @log.function_(call=True)
    async def WebSearch(self,
        query: typing.Annotated[str, 'Search query keywords'],
        search_type: typing.Annotated[str, 'Type of search: web, image, or video (default web)'] = 'web',
        top_k: typing.Annotated[int, 'Maximum number of results (default 10)'] = 10,
        time_range: typing.Annotated[str, 'Time filter: day/week/month/year, empty for no limit'] = '',
    ) -> str:
        """Search the web for current information. Returns structured results with titles, URLs, and content summaries. Use for finding news, documentation, research, or any information that may have changed recently."""
        import json as _json

        # Load search config (prefer desktop_tools.search, fallback to tool_config)
        dt_config = config.main.get('desktop_tools', {})
        search_cfg = dt_config.get('search', {})
        search_provider = search_cfg.get('type', 'none')

        if search_provider == 'none' or not search_provider:
            return 'Error: Web search is not configured. Ask the administrator to configure search in Settings.'

        base_url = search_cfg.get('base_url', '').rstrip('/')
        api_key = search_cfg.get('api_key', '')

        if not base_url:
            return 'Error: Search base_url is not configured.'

        # Ensure base_url ends with /v1
        if not base_url.endswith('/v1'):
            base_url = base_url + '/v1'

        # Build request payload
        search_params = {
            'search_type': search_type if search_type in ('web', 'image', 'video') else 'web',
            'top_k': max(1, min(top_k, 20)),
        }
        if time_range and time_range in ('day', 'week', 'month', 'year'):
            search_params['time_range'] = time_range

        payload = {
            'model': 'baidu-search',
            'messages': [{'role': 'user', 'content': query}],
            'stream': False,
            'search_parameters': search_params,
        }

        try:
            import aiohttp
        except ImportError:
            return 'Error: aiohttp not installed'

        try:
            headers = {'Content-Type': 'application/json'}
            if api_key:
                headers['Authorization'] = f'Bearer {api_key}'

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f'{base_url}/chat/completions',
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status == 401:
                        return 'Error: Search API key is invalid (401 Unauthorized)'
                    if resp.status == 422:
                        body = await resp.text()
                        return f'Error: Invalid search parameters (422): {body[:200]}'
                    if resp.status >= 400:
                        body = await resp.text()
                        return f'Error: Search API returned HTTP {resp.status}: {body[:200]}'

                    data = await resp.json()

        except asyncio.TimeoutError:
            return 'Error: Search request timed out after 30s'
        except Exception as e:
            return f'Error: Search request failed: {e}'

        # Parse response
        try:
            content_str = data['choices'][0]['message']['content']
            results_data = _json.loads(content_str)
            results = results_data.get('results', [])
        except (KeyError, IndexError, _json.JSONDecodeError) as e:
            return f'Error: Failed to parse search results: {e}\nRaw: {str(data)[:500]}'

        if not results:
            return f'No results found for "{query}"'

        # Format results for LLM consumption
        lines = [f'Search results for "{query}" ({len(results)} results):']
        lines.append('')
        for r in results:
            title = r.get('title', 'Untitled')
            url = r.get('url', '')
            content = r.get('content', '')
            website = r.get('website', '')
            date = r.get('date', '')

            lines.append(f'### {title}')
            if url:
                lines.append(f'URL: {url}')
            meta_parts = []
            if website:
                meta_parts.append(website)
            if date:
                meta_parts.append(date)
            if meta_parts:
                lines.append(f'Source: {" | ".join(meta_parts)}')
            if content:
                lines.append(content[:500])
            lines.append('')

        return _truncate('\n'.join(lines))
