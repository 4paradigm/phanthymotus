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

# Read 直接返回给 LLM 的图片：超过此大小先用 ffmpeg 缩放重编码，避免 context 爆掉
_IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}
_IMAGE_INLINE_MAX = 1_500_000   # 1.5 MB
_IMAGE_MAX_EDGE = 1280          # 缩放后最长边

# WebFetch(save_to=...) 落盘上限。与 channel/media.py:LIMITS 的平台附件上限同量级——
# 下载一个发不出去的文件没有意义。注意这**不是** WebFetch 转 markdown 那条路径的 512KB raw cap：
# 那条路的内容要进 context，这条不进。
_DOWNLOAD_MAX = 20 * 1024 * 1024

# WebSearch 每种搜索类型的 (默认条数, 上限)。上限是对 router.phanthy.com 实测出来的：
# web 给 50 就回 50，image 给 30 回 23（够了），video 给 20 也只回 10。
# 默认值压得比上限低得多——一次 WebSearch 的结果单独就能撑爆历史压缩阈值
# （见 config.py 的 compress 阈值注释与 subagent/context.py 的实测记录）。
_SEARCH_LIMITS = {
    'web':   (10, 50),
    'image': (8, 30),
    'video': (5, 10),
}

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


def _resolve_path(path: str) -> pathlib.Path:
    """Resolve path, default relative to /work."""
    p = pathlib.Path(path)
    if not p.is_absolute():
        p = pathlib.Path('/work') / p
    return p.resolve()


def _check_path_allowed(p: pathlib.Path, dirs: list[str] | None = None) -> str | None:
    """Return error message if path is not within allowed dirs, else None."""
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


# ── WebSearch 结果格式化 ─────────────────────────────────────────────────────────
#
# 搜索后端对三种 search_type 返回**三种不同形状**的结果，只有 web 那种是
# {title,url,content,website,date}。image 的图在 `image.url`（`content` 恒为空串），
# video 的元信息在 `video.*`（`video.url` 恒为空，能播的是落地页 `url`），网页结果的配图在
# `web_extensions.images[]`。按一种形状去读三种结果，就等于把图片搜索的图整个丢掉。


def _fmt_meta(r: dict) -> str:
    """website | date，两者都可能缺。"""
    parts = [r.get(k, '') for k in ('website', 'date')]
    return ' | '.join(p for p in parts if p)


def _fmt_duration(seconds) -> str:
    """'164' → '2:44'。后端给的是字符串秒数，也可能是空串。"""
    try:
        s = int(float(seconds))
    except (TypeError, ValueError):
        return ''
    if s <= 0:
        return ''
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f'{h}:{m:02d}:{sec:02d}' if h else f'{m}:{sec:02d}'


def _fmt_wh(d: dict) -> str:
    w, h = d.get('width', ''), d.get('height', '')
    return f'{w}x{h}' if w and h else ''


def _fmt_web_result(r: dict) -> list[str]:
    lines = [f'### {r.get("title", "Untitled")}']
    if r.get('url'):
        lines.append(f'URL: {r["url"]}')
    meta = _fmt_meta(r)
    if meta:
        lines.append(f'Source: {meta}')
    try:
        if float(r.get('authority_score') or 0) >= 0.8:
            lines.append('Authority: high')
    except (TypeError, ValueError):
        pass
    if r.get('content'):
        lines.append(r['content'][:500])
    # 网页自带的配图——要发图给用户时这些往往比 image 搜索更贴题。
    images = (r.get('web_extensions') or {}).get('images') or []
    for img in images[:3]:
        if img.get('url'):
            wh = _fmt_wh(img)
            lines.append(f'- Figure: {img["url"]}' + (f' ({wh})' if wh else ''))
    return lines


def _fmt_image_result(r: dict) -> list[str]:
    lines = [f'### {r.get("title", "Untitled")}']
    img = r.get('image') or {}
    if img.get('url'):
        wh = _fmt_wh(img)
        lines.append(f'Image: {img["url"]}' + (f' ({wh})' if wh else ''))
    if r.get('url'):
        lines.append(f'Page: {r["url"]}')
    meta = _fmt_meta(r)
    if meta:
        lines.append(f'Source: {meta}')
    # content 对图片结果恒为空，不打印空行。
    if r.get('content'):
        lines.append(r['content'][:200])
    return lines


def _fmt_video_result(r: dict) -> list[str]:
    lines = [f'### {r.get("title", "Untitled")}']
    vid = r.get('video') or {}
    # video.url 实测恒为空——可播地址是上一级的落地页 url。
    watch = vid.get('url') or r.get('url', '')
    if watch:
        lines.append(f'Watch: {watch}')
    if vid.get('hover_pic'):
        lines.append(f'Cover: {vid["hover_pic"]}')
    dur = _fmt_duration(vid.get('duration'))
    wh = _fmt_wh(vid)
    facts = [x for x in (f'Duration: {dur}' if dur else '', f'Resolution: {wh}' if wh else '') if x]
    if facts:
        lines.append(' | '.join(facts))
    meta = _fmt_meta(r)
    if meta:
        lines.append(f'Source: {meta}')
    # 视频结果的 content 往往是整段字幕/文案，比网页摘要长得多，单独截短。
    if r.get('content'):
        lines.append(r['content'][:300])
    return lines


_RESULT_FORMATTERS = {
    'web': _fmt_web_result,
    'image': _fmt_image_result,
    'video': _fmt_video_result,
}


def vision_input_enabled() -> bool:
    """主模型是否接受图片输入（decision_core 配置项 vision_input，默认关）。

    默认关是因为「发出去才知道不支持」的代价不小：一张手机照片是 ~420KB base64，
    换来的是一个 400 和一轮失败。开关由人显式打开，不做能力自动推断。
    """
    return bool(config.main.get('event', {}).get('llm', {}).get('vision_input', False))


def _image_unavailable(p: pathlib.Path, mime: str, size: int, origin: str = '') -> str:
    """模型不接受图片输入时，图片读取按**失败**返回。

    这里刻意用 `Error:` 而不是「成功但内容为空」：实测过三种成功形态的返回
    （解释性占位 / 纯客观说明行 / 文件信息行），模型都把「成功读取」当成「我拿到了这张图」，
    然后凭文件名编出斜拉桥、灯光秀、猫。唯一一次如实回答，恰恰是那一轮直接失败、
    工具结果压根不存在的时候。文件信息照旧附上 —— 转发、存档、放进文档都还用得到。
    """
    return (f'Error: cannot parse image contents — this model does not accept image input. '
            f'File: [{origin}{p} | {mime} | {size} bytes]')


def _image_origin(p: pathlib.Path) -> str:
    """入站附件目录里的文件是用户发来的，标注出来；其它（如相机截图）不标注。"""
    try:
        from channel import store as _store
        if _store.is_inbound_media(p):
            return 'user-uploaded image | '
    except Exception:
        pass
    return ''


def _image_mime(p: pathlib.Path) -> str:
    return {
        '.png': 'image/png', '.gif': 'image/gif',
        '.webp': 'image/webp', '.bmp': 'image/bmp',
    }.get(p.suffix.lower(), 'image/jpeg')


async def _read_image(p: pathlib.Path):
    """把图片文件读成 OpenAI multi-modal 内容块。

    返回值形如 [{'type':'text',...},{'type':'image_url','image_url':'data:...'}]，
    与 mcp_client.call_tool 处理 MCP 图片结果的格式一致；event/llm.py::_trim 已能
    处理 list 型 tool content（超出 max_images 时替换为占位符）。

    大图先用 ffmpeg 缩放重编码为 JPEG（镜像内自带 ffmpeg，见 start.py::publish_image_file），
    否则一张手机照片就能占满上下文。
    """
    import base64

    raw = p.read_bytes()
    mime = _image_mime(p)

    note = ''
    if len(raw) > _IMAGE_INLINE_MAX:
        proc = await asyncio.create_subprocess_exec(
            'ffmpeg', '-y', '-i', str(p), '-frames:v', '1',
            '-vf', f'scale=\'min({_IMAGE_MAX_EDGE},iw)\':-2',
            '-q:v', '4', '-f', 'mjpeg', 'pipe:1',
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, _err = await asyncio.wait_for(proc.communicate(), timeout=30)
        except asyncio.TimeoutError:
            proc.kill()
            out = b''
        if out:
            note = f' (downscaled from {len(raw) / 1e6:.1f}MB for the model)'
            raw, mime = out, 'image/jpeg'
        else:
            return (f'[{p} | {len(raw)} bytes] Image too large to inline and ffmpeg '
                    f'downscaling failed. Use Bash + ffmpeg to shrink it first.')

    b64 = base64.b64encode(raw).decode('ascii')
    # 说明行体现来源：入站附件目录里的文件是**用户发来的**，不是机器人自己取得的图像。
    return [
        {'type': 'text', 'text': f'[{_image_origin(p)}{p} | {mime} | {len(raw)} bytes{note}]'},
        {'type': 'image_url', 'image_url': f'data:{mime};base64,{b64}'},
    ]


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
        safe_builtins = {k: getattr(_builtins, k) for k in dir(_builtins)
                        if not k.startswith('_') and k not in ('exec', 'eval', 'compile', 'exit', 'quit')}

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
        """Read file contents with line numbers. Supports reading specific line ranges for large files. Images (jpg/png/gif/webp/bmp) are returned as viewable images — use this to look at photos received from messaging channels or captured by the robot."""
        p = _resolve_path(file_path)

        err = _check_path_allowed(p)
        if err:
            return err

        if not p.exists():
            return f'Error: file not found: {p}'
        if not p.is_file():
            return f'Error: not a file: {p}'

        # 图片：开关打开才把图像内联给模型（多模态）；否则按读取失败返回
        if p.suffix.lower() in _IMAGE_EXT:
            if vision_input_enabled():
                return await _read_image(p)
            return _image_unavailable(p, _image_mime(p), p.stat().st_size, _image_origin(p))

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
            matches = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
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

        # Try using grep/rg subprocess for speed
        cmd_parts = []
        if os.path.exists('/usr/bin/grep') or os.path.exists('/bin/grep'):
            cmd_parts = ['grep', '-rn', '--color=never', '-E']
            if include:
                cmd_parts.extend(['--include', include])
            cmd_parts.extend([pattern, str(root)])
        else:
            # Fallback: pure Python
            return await self._grep_python(pattern, root, include)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd_parts,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            output = stdout.decode('utf-8', errors='replace')
        except asyncio.TimeoutError:
            return 'Error: grep timed out after 30s'
        except Exception as e:
            return await self._grep_python(pattern, root, include)

        if not output.strip():
            return f'No matches for pattern "{pattern}" in {root}'

        # Limit output
        lines = output.splitlines()
        if len(lines) > 50:
            result = '\n'.join(lines[:50])
            result += f'\n\n... ({len(lines) - 50} more matches not shown)'
            return _truncate(result)
        return _truncate(output)

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
                text = fp.read_text(encoding='utf-8', errors='ignore')
                for i, line in enumerate(text.splitlines(), 1):
                    if regex.search(line):
                        results.append(f'{fp}:{i}: {line.rstrip()}')
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
        save_to: typing.Annotated[str, "Save the raw bytes to this local path instead of extracting text. Use it to download an image or video URL returned by WebSearch. Must be under /work or /tmp; the extension is filled in from the response if omitted."] = '',
    ) -> str:
        """Fetch URL content. HTML is automatically converted to Markdown for readability. Optionally specify a prompt to describe what information you want to extract.

        Pass save_to to download instead: the raw bytes go to that local path and the tool returns the
        saved path, size and type. This is how a remote image/video from WebSearch becomes something you
        can send with the channel reply tool's files parameter or look at with Read()."""
        timeout = max(5, min(timeout, 60))

        try:
            import aiohttp
        except ImportError:
            return 'Error: aiohttp not installed. Run: pip install aiohttp'

        if save_to:
            return await self._download(url, save_to, timeout)

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
                                       headers={'User-Agent': 'Mozilla/5.0 (compatible; AgentBot/1.0)'},
                                       ssl=False) as resp:
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

    async def _download(self, url: str, save_to: str, timeout: int) -> str:
        """WebFetch(save_to=...) 的下载分支：把响应体原样写到本地白名单目录。

        独立成一个方法，是因为它和取文本那条路几乎没有共同点：不解码、不转 markdown、
        不进 context，大小上限也不同（那条路的 512KB 是"别撑爆上下文"，这条是"别撑爆磁盘"）。
        """
        import mimetypes

        import aiohttp

        p = _resolve_path(save_to)
        err = _check_path_allowed(p)
        if err:
            return f'Error: {err}'
        if p.is_dir():
            return f'Error: {p} is a directory'

        try:
            p.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return f'Error: cannot create {p.parent}: {e}'

        written = 0
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout),
                                       headers={'User-Agent': 'Mozilla/5.0 (compatible; AgentBot/1.0)'},
                                       ssl=False) as resp:
                    if resp.status >= 400:
                        return f'Error: HTTP {resp.status} {resp.reason}'
                    content_type = (resp.headers.get('content-type', '') or '').split(';')[0].strip()

                    # 没有扩展名时按 Content-Type 补一个：channel/media.py::infer_kind 先看扩展名，
                    # 一个没后缀的 JPEG 会被当成普通文件发出去（走文件接口而不是图片接口）。
                    if not p.suffix and content_type:
                        ext = mimetypes.guess_extension(content_type)
                        if ext == '.jpe':      # mimetypes 对 image/jpeg 的第一个答案，平台不认
                            ext = '.jpg'
                        if ext:
                            p = p.with_suffix(ext)

                    # 流式写盘：超限即中止，不把 20MB+ 先攒进内存。
                    with open(p, 'wb') as fh:
                        async for chunk in resp.content.iter_chunked(65536):
                            written += len(chunk)
                            if written > _DOWNLOAD_MAX:
                                fh.close()
                                p.unlink(missing_ok=True)
                                return (f'Error: response exceeds the {_DOWNLOAD_MAX // 1024 // 1024}MB '
                                        f'download limit. Nothing was saved.')
                            fh.write(chunk)
        except asyncio.TimeoutError:
            p.unlink(missing_ok=True)
            return f'Error: request timed out after {timeout}s'
        except Exception as e:
            p.unlink(missing_ok=True)
            return f'Error downloading URL: {e}'

        if written == 0:
            p.unlink(missing_ok=True)
            return f'Error: {url} returned an empty body. Nothing was saved.'

        desc = f'Saved {written} bytes to {p}'
        if content_type:
            desc += f' ({content_type})'
        return desc

    # ── 9. WebSearch ─────────────────────────────────────────────────────────

    @log.function_(call=True)
    async def WebSearch(self,
        query: typing.Annotated[str, 'Search query keywords'],
        search_type: typing.Annotated[str, "Type of search: 'web' (pages), 'image' (pictures), or 'video' (clips). Default 'web'."] = 'web',
        top_k: typing.Annotated[int, 'Maximum number of results. 0 = per-type default (web 10, image 8, video 5). Caps: web 50, image 30, video 10.'] = 0,
        time_range: typing.Annotated[str, 'Time filter: day/week/month/year, empty for no limit'] = '',
    ) -> str:
        """Search the web for current information. Use for news, documentation, research, or anything that may have changed recently.

        search_type='image' returns picture URLs ("Image: https://..."), search_type='video' returns watch
        links, cover images and durations, and web results may list article figures ("Figure: https://...").
        Those are remote URLs — to show one to the user or look at it yourself, first download it with
        WebFetch(url, save_to='/tmp/pic.jpg'), then send it via the channel reply tool's files parameter
        (it only accepts local paths) or inspect it with Read(path)."""
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
        if search_type not in _SEARCH_LIMITS:
            search_type = 'web'
        default_k, max_k = _SEARCH_LIMITS[search_type]
        search_params = {
            'search_type': search_type,
            'top_k': default_k if top_k <= 0 else min(top_k, max_k),
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
                    ssl=False,
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

        # Format results for LLM consumption. Each result carries its own `type`; dispatch on it
        # rather than on the requested search_type, so a mixed response still renders correctly.
        label = {'web': 'Web', 'image': 'Image', 'video': 'Video'}[search_type]
        lines = [f'{label} search results for "{query}" ({len(results)} results):']
        lines.append('')
        for r in results:
            fmt = _RESULT_FORMATTERS.get(r.get('type') or search_type, _fmt_web_result)
            lines.extend(fmt(r))
            lines.append('')

        return _truncate('\n'.join(lines))
