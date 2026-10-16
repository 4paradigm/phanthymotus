"""Files a skill carries.

A skill definition can declare `files`: a manifest of what it ships, each entry
just `{path, bytes, sha256}`. There is deliberately **nothing in the
declaration about who reads a file** -- no consumer, no protocol, no tool name.

That is not minimalism for its own sake; it is how every other tool-to-skill
relationship here already works. `requiredTools` is enforced nowhere -- one
display label in the skill panel is its only reader -- because a skill says in
prose "use tool X" and the LLM routes the call. A file is the same. The two
readers reach it through mechanisms that already exist:

| | the model reads it | a device consumes it |
|---|---|---|
| how | `skill_file(slug, path)` -> text | `skill_file(..., as_path=True)` -> a local path, passed to any `format: file` parameter |
| who decides | the LLM, from the instruction | the LLM, by **which tool it calls** |
| transport | this module | `mcp_client._transfer_file_args` pushes it to that service and rewrites the argument |
| new machinery | this module | **none** |

Earlier drafts had each entry name its consumer, its protocol `schema`, and the
`tools` allowed to receive it. All three were wrong in the same direction:

- `tools` duplicated what `schema` already carried (`targets()` filters on the
  declared schema; the tool name was only ever an optional extra narrowing) and
  forced a marketplace author to enumerate driver tool names they cannot know --
  "works on quadrupeds" does not mean "knows this one calls it `go2_nav`".
- `schema` and `consumer` moved a decision the LLM makes per call into a binding
  fixed at authoring time. Nothing else in a skill binds that way.

What stays Core's job is confinement: a declared path cannot escape the skill's
own directory, stored content must match its declaration, and reads are
bounded. That part carries the weight here, because this is an LLM-facing read
primitive on a robot whose path argument ultimately comes from text in a prompt.
Two path bugs in this feature were already found from the other side -- a `..`
check a zip library had normalised away (dead code reading as protection) and a
separator check that missed `..\\` -- so the checks below run at declaration
time *and* against the resolved filesystem path.

The chunked transfer in `api/skill_packages` is the **special** case, not the
general one: it exists because one device protocol is "take this whole archive,
then address it by a handle I mint". The general case is `format: file`.
"""
import hashlib
import os
import pathlib
import re
import stat

import config

MAX_FILES = 32
MAX_FILE_BYTES = 16 * 1024 * 1024       # one file, as stored
MAX_TOTAL_BYTES = 32 * 1024 * 1024      # everything one skill carries
MAX_READ_BYTES = 64 * 1024              # one `skill_file` read; enters the context
MAX_TEXT_BYTES = 1024 * 1024            # refuse to text-read anything larger

# One path component: no separators, no `..`, no leading dot, no control chars.
_COMPONENT = re.compile(r'[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}')
_SLUG = re.compile(r'[a-z0-9][a-z0-9_-]{0,99}')
MAX_DEPTH = 4


def validate_path(path) -> str:
    """A declared file path, or raise ValueError."""
    if not isinstance(path, str) or not path or len(path) > 200:
        raise ValueError('文件路径无效')
    parts = path.split('/')
    if not 1 <= len(parts) <= MAX_DEPTH:
        raise ValueError(f'文件路径最多 {MAX_DEPTH} 层')
    for part in parts:
        if not _COMPONENT.fullmatch(part):
            raise ValueError(f'文件路径含非法成分：{part!r}')
    return path


def validate_files(files) -> list:
    """Normalise a definition's `files`, or raise ValueError.

    Takes no tool list and consults none: a file declaration says what the skill
    ships, not who may read it.
    """
    if not isinstance(files, list) or len(files) > MAX_FILES:
        raise ValueError(f'files 必须是最多 {MAX_FILES} 项的列表')
    seen, result, total = set(), [], 0
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError('files 的每一项必须是对象')
        path = validate_path(entry.get('path'))
        if path in seen:
            raise ValueError(f'重复的文件路径：{path}')
        seen.add(path)
        size = entry.get('bytes')
        if type(size) is not int or not 1 <= size <= MAX_FILE_BYTES:
            raise ValueError(f'{path} 的 bytes 必须是 1..{MAX_FILE_BYTES} 的整数')
        total += size
        if total > MAX_TOTAL_BYTES:
            raise ValueError(f'技能随附文件合计超过 {MAX_TOTAL_BYTES // (1024 * 1024)} MiB')
        digest = entry.get('sha256')
        if not isinstance(digest, str) or not re.fullmatch(r'[a-f0-9]{64}', digest):
            raise ValueError(f'{path} 的 sha256 无效')
        row = {'path': path, 'bytes': size, 'sha256': digest}
        extra = set(entry) - set(row)
        if extra:
            raise ValueError(f'{path} 含不支持的字段：{", ".join(sorted(extra))}')
        result.append(row)
    return result


# ── storage ──────────────────────────────────────────────────────────────────

def root() -> pathlib.Path:
    """Where skill files live.

    Beside the config DB, so it shares the deployment's persistent mount
    (`/opt/phanthy-motus/data` on a robot) and survives a container restart the
    same way installed skills do.
    """
    return pathlib.Path(config.DB_PATH).resolve().parent / 'skill_files'


def _slug_dir(slug: str) -> pathlib.Path:
    # `validate_local_skill` already constrains a slug to this shape, but this
    # is the directory an LLM-reachable read resolves against -- re-check rather
    # than inherit the guarantee from a caller.
    if not _SLUG.fullmatch(slug or ''):
        raise ValueError('技能标识无效')
    return root() / slug


def resolve(slug: str, path: str) -> pathlib.Path:
    """The on-disk path for a declared file, confined to the skill's directory.

    `resolve()` first, then verify containment via `parents` -- not a string
    prefix, which would accept a sibling directory whose name merely starts the
    same way (`<slug>-evil/x` passes `startswith(<slug>)`). Never open the
    supplied value directly.
    """
    base = _slug_dir(slug).resolve()
    target = (base / validate_path(path)).resolve()
    if target != base and base not in target.parents:
        raise ValueError('文件路径越出技能目录')
    return target


def declared(skill: dict, path: str) -> dict:
    """The skill's own declaration for `path`, or raise ValueError.

    Content can only be written, and only be read, where the definition said it
    would be. A file the skill never declared has no storage and no reader.
    """
    for entry in skill.get('files') or []:
        if entry.get('path') == path:
            return entry
    declared_paths = ', '.join(e['path'] for e in (skill.get('files') or []))
    raise ValueError(f'技能未声明文件 {path}'
                     + (f'；已声明：{declared_paths}' if declared_paths else '；该技能没有随附文件'))


def write(slug: str, skill: dict, path: str, data: bytes) -> dict:
    """Store a declared file, verified against its declaration."""
    entry = declared(skill, path)
    if len(data) != entry['bytes']:
        raise ValueError(f'{path} 长度与声明不符：声明 {entry["bytes"]}，实际 {len(data)}')
    if hashlib.sha256(data).hexdigest() != entry['sha256']:
        raise ValueError(f'{path} 校验和与声明不符')
    target = resolve(slug, path)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Write-then-rename: a reader never sees half a file, and a failed write
    # leaves the previous content intact.
    staged = target.with_name(target.name + '.part')
    staged.write_bytes(data)
    os.replace(staged, target)
    return {'path': path, 'bytes': len(data)}


def _open_regular(target: pathlib.Path, path: str):
    """Open a stored file, refusing a symlink or anything not a regular file.

    `O_NOFOLLOW` because this process can write the directory: a symlink
    planted there must not turn a skill-file read into a read of something else.
    """
    try:
        fd = os.open(target, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, 'O_NONBLOCK', 0))
    except FileNotFoundError:
        raise ValueError(f'{path} 尚未写入内容')
    except OSError:
        raise ValueError(f'无法读取 {path}')
    stream = os.fdopen(fd, 'rb')
    if not stat.S_ISREG(os.stat(stream.fileno()).st_mode):
        stream.close()
        raise ValueError(f'{path} 不是普通文件')
    return stream


def local_path(slug: str, skill: dict, path: str) -> str:
    """The absolute local path of a declared, stored file.

    This is what the LLM hands to a tool's `format: file` parameter;
    `mcp_client._transfer_file_args` then pushes it to whichever service that
    tool belongs to and rewrites the argument. No skill-side declaration of the
    destination is involved -- the tool the model called *is* the destination.
    """
    declared(skill, path)
    target = resolve(slug, path)
    with _open_regular(target, path):
        pass
    return str(target)


def read_text(slug: str, skill: dict, path: str, offset: int = 0) -> tuple[str, dict]:
    """Read a declared file as text, bounded. Returns `(text, info)`."""
    entry = declared(skill, path)
    target = resolve(slug, path)
    if not isinstance(offset, int) or offset < 0:
        raise ValueError('offset 不能为负')
    with _open_regular(target, path) as stream:
        size = os.stat(stream.fileno()).st_size
        if size > MAX_TEXT_BYTES:
            raise ValueError(f'{path} 有 {size} 字节，过大无法按文本读取；'
                             f'改用 as_path=true 把它交给接受文件参数的工具')
        stream.seek(min(offset, size))
        chunk = stream.read(MAX_READ_BYTES)
    return chunk.decode('utf-8', errors='replace'), {
        'bytes': size, 'offset': offset, 'next_offset': offset + len(chunk),
        'truncated': offset + len(chunk) < size, 'declared_sha256': entry['sha256']}
