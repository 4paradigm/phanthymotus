"""Pinned Xiaoyou model contract, verification and installation.

Production downloads use a fixed ModelScope repository and immutable revision.
An internal HTTP mirror is available only through the explicitly named test opt-in.
Existing bundles are rehashed before use; unknown/corrupt destinations are preserved.
"""
from contextlib import contextmanager, suppress
import fcntl
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import shutil
import tarfile
import tempfile
import time
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen
from zipfile import ZipFile

SOURCE_SHA = '440346a39553bccbe65b1afe10bbc1b8f8b9fa9ce2e9fe23902578d44b9243b7'

FRONTEND_FILE = 'frontend/phonemizer.py'
FRONTEND_SHA = 'e11045eeda1ed090dfa907f454a941ac2a1305454f5ffa2e169389ae890ac8ea'
_RELEASE_FRONTEND_SHA = '650f908b1f9ef473656e537630b8c2823b7a3633ff8a562745cc5d7ce1db0dc5'

NORMALIZATION_REVISION = '01yuewen-contextual-polyphones-compact-20261118'

NORMALIZATION_MODULE_SHAS = {
    'frontend/numbers.py': '7bd2edc9672b3057fd3a4ff17c1c5e8cb0fbf6d45dd4326d06fdac37fa1271a4',
}
# The immutable model manifest records the source from its original export.
# The installed frontend is verified against the current hashes above.
_RELEASE_NORMALIZATION_MODULE_SHAS = {
    'frontend/normalization.py': '839d57f1c99a2ccd04780f117fc377f523e5447cb901f772b1da6b8cbec39047',
    'frontend/numbers.py': 'b89d92f17e977f58e0889d6f77050724e5307b15a05d35b3b3c4bb2984a68d57',
    'frontend/pronunciation.py': '37f97358f67e9f7d57706640045a0a07b04e4399057953f46e5c2b4a8351ba23',
    'frontend/segmentation.py': 'd94d38877734783911b10063215e0e8a80ece7729031c0dc2ac3b0e85dcfbd8d',
}

TERMS_FILE = 'frontend/terms.json'
TERMS_SHA = '424c489cabc9bfec347aa1f15b9b5886cbd4d227649efe789a06333c01e2981c'

PHONE_ABI = 'kokoro-v1.0-misaki-0.9.4-xiaoyou-v6'

VOCAB_SHA = '248ee3192acfc27a9907180354a0efaa63389047a6ff78c3f75e2e19466d0b76'

CONFIG_SHA = '5abb01e2403b072bf03d04fde160443e209d7a0dad49a423be15196b9b43c17f'

GLOBAL_STEP = 50195

VOICE_ID = '01yuewen'

NATIVE_WEIGHTS_SHA = '9e9e01ac451018269a7ee371e8ac186ad7d7458b1f92713700787362e9f40d07'

# The published archive records the pre-merge excitation source.
PHASE_DSP_SHA = '82e542dc0d002ff5214779ae568b66407cd0f2770495bb7614fc58bc3804a4b4'
SYNTHESIS_SOURCE_SHA = '435c10ace9cc5d27e6a14d65fae19791cda7ebb6f400f7d66e53770eb1d253fd'

STYLE_FILE_SHA = 'd6ee65a6740975b2d289e6427be1ce54b629d9af55214f5e2a1d7e1ec745a571'

STYLE_RAW_SHA = '654b411021bb18f83c49c68701f11aaa49199d784b7a5850f8ad71cbcd704b67'

GRAPH_SHAS = {'acoustic': '4934fe01f188c3b8958996d5c5b9b857b19312553f88b02ce28f21a704fb779f',
 'encoder': '6db844eb75ff6e523ed4844d1bca0aac43adc30f7c85796c58142a0ccaff193f',
 'vocoder': 'f524a67bf359ce87938723549a67502aa30d775275633f27b3cab8a8178a8d68'}

PLAN_SHAS = {'acoustic': '718a3340da8ccbc460a5e16209e77556e6916f31acd9dd40a31d052a8b9df84b',
 'encoder': 'd0f61501c8fc8d5e4bae73f6c92b8cfeed7ce67d7b097add4b86bf355a237dff',
 'vocoder': '432953b40f087cc50ac989ec3478bd28266951a0f9ca52ab90ec9a26b2d069b8'}

EXPECTED_PROFILES = {'acoustic': {'d': [[1, 640, 3], [1, 640, 128], [1, 640, 512]],
              'indices': [[3], [512], [832]],
              'style': [[1, 256], [1, 256], [1, 256]],
              't_en': [[1, 512, 3], [1, 512, 128], [1, 512, 512]]},
 'encoder': {'speed': [[1], [1], [1]],
             'style': [[1, 256], [1, 256], [1, 256]],
             'tokens': [[1, 3], [1, 128], [1, 512]]},
 'vocoder': {'excitation_noise': [[1, 1800, 9], [1, 153600, 9], [1, 438000, 9]],
             'f0': [[1, 6], [1, 512], [1, 1460]],
             'features': [[1, 512, 6], [1, 512, 512], [1, 512, 1460]],
             'source_phase': [[1, 1800, 9], [1, 153600, 9], [1, 438000, 9]],
             'style': [[1, 256], [1, 256], [1, 256]]}}

JP5_GRAPH_SHAS = {'acoustic': '2c5ff484e0e5c5d5f46a237cfc7a4509612837b1ed00c4a2d39e8c04281aaec1',
 'encoder': '5ab1c3cfe10c4db5fd1bb7c401434e6347d266cbe3afaa4d19580a1eaf57ef34',
 'vocoder': 'acbf05eb2fb0949d5d4bdc7b2c2b12253f4b577368a47d4beb021d9a1bc9335d'}
JP5_PLAN_SHAS = {'acoustic': '020732b58e5f6d62d6669209d132fb32c4685efeba222620268c80b0788bc1a6',
 'encoder': '0a65d7719aeb08a487a9aef81edba2f0549fc66423b3a7614e1017a2f8f2c6cb',
 'vocoder': 'c996b39d196844919c0db13b39a83395e50ce57d68d7e1616b0635bf28cf29d0'}
JP5_EXPECTED_PROFILES = {'acoustic': {'d': [[1, 640, 3], [1, 640, 128], [1, 640, 512]],
              'indices': [[3], [512], [832]],
              'style': [[1, 256], [1, 256], [1, 256]],
              't_en': [[1, 512, 3], [1, 512, 128], [1, 512, 512]]},
 'encoder': {'speed': [[1], [1], [1]],
             'style': [[1, 256], [1, 256], [1, 256]],
             'tokens': [[1, 3], [1, 128], [1, 512]]},
 'vocoder': {'excitation_noise': [[1, 1800, 9], [1, 153600, 9], [1, 438000, 9]],
             'f0': [[1, 6], [1, 512], [1, 1460]],
             'features': [[1, 512, 6], [1, 512, 512], [1, 512, 1460]],
             'source_phase': [[1, 1800, 9], [1, 153600, 9], [1, 438000, 9]],
             'style': [[1, 256], [1, 256], [1, 256]]}}
TRT_VERSION_PREFIX = {5: '8.5.2', 6: '10.4'}

STREAM_POLICY = {'activation_allocation': {'default': 'fixed',
                           'release': 'after_confirmed_original_stream_sync',
                           'shape_mode': 'per_stage_lease_after_input_shape_validation',
                           'supported': ['fixed', 'shape']},
 'allow_frame_truncation': False,
 'allowed_first_output_policies': ['clause', 'sentence'],
 'default_first_output_policy': 'clause',
 'join_padding': {'kind': 'PCM16_exact_zero_only',
                  'max_leading_samples': 2000,
                  'max_trailing_samples': 2000,
                  'sample_rate': 16000,
                  'scope': 'selected_first_output_to_full_tail_only'},
 'max_T': 730,
 'mode': 'natural_text_streaming',
 'pcm_postprocessing': {'ambiguous_join_fallback': 'complete_reference',
                        'complete_reference_PCM_preserved': True,
                        'default_prefix_enabled': True,
                        'native_context_samples': 12023,
                        'prefix_samples': 8000,
                        'sample_rate': 16000},
 'preserve_full_text': True,
 'proactive_span_limit': 2,
 'speed': 1.0,
 'split_strategy': 'safe_first_punctuation_clause_then_full_tail_with_lossless_semantic_fallback',
 'supported_F_range': [6, 1460],
 'first_clause_selection': {'keep_complete_sentence_spoken_units_max': 24,
                            'minimum_prefix_spoken_units': 12,
                            'spoken_units': 'CJK ideographs plus twice ASCII word/number token '
                                            'count',
                            'punctuation': 'first safe comma, colon or enumeration boundary after two complete items inside the first complete sentence; retain original punctuation',
                            'short_or_ambiguous_fallback': 'complete_sentence'}}

OPERATING_F_RANGE = [6, 1460]



def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def unpack_jp6_english_model(model_dir, destination):
    """Make the pinned wheel importable without altering the verified model tree."""
    wheel = Path(model_dir) / 'frontend/en_core_web_sm-3.8.0-py3-none-any.whl'
    size = 12_806_118
    with wheel.open('rb') as stream:
        data = stream.read(size + 1)
    expected = '1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85'
    if len(data) != size or hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('Xiaoyou JP6 English model wheel differs from the pinned resource')
    with ZipFile(BytesIO(data)) as archive:
        archive.extractall(destination)


def validate_frontend_files(front, directory):
    """Bind model metadata, the fixed Git contract and real package files."""
    if not isinstance(front, dict) or (
            front.get('abi') != PHONE_ABI
            or front.get('source_sha256') != _RELEASE_FRONTEND_SHA
            or front.get('terms_file') != TERMS_FILE
            or front.get('terms_sha256') != TERMS_SHA
            or front.get('vocab_sha256') != VOCAB_SHA
            or front.get('normalization_revision') != NORMALIZATION_REVISION
            or front.get('normalization_modules') != _RELEASE_NORMALIZATION_MODULE_SHAS):
        raise ValueError('Xiaoyou frontend metadata differs from the fixed source contract')
    root = Path(directory).resolve()
    files = {FRONTEND_FILE: FRONTEND_SHA, **NORMALIZATION_MODULE_SHAS}
    # Shared V4 helpers are provided by the application revision.
    for name, expected in files.items():
        path = root / name
        if (path.is_symlink() or not path.is_file()
                or root not in path.resolve().parents or sha(path) != expected):
            raise ValueError('Xiaoyou frontend source missing, linked or SHA-mismatched: ' + name)
    return dict(files)



MANIFEST = 'runtime_manifest.json'
MAX_FILE_BYTES = 512 * 1024 * 1024
MAX_BUNDLE_BYTES = 2 * 1024 * 1024 * 1024

_MODEL_RELEASE = {'asset_bytes': 318220890,
 'asset_sha256': '5c22ae5407c125afb6a780265d5b3373af074cb678d475506b7698331971c644',
 'modelscope_file': '01yuewen-jp6-fp32-50195-20261125-v1.tar.gz',
 'modelscope_repo': 'lanxijiang/xiaoyou01',
 'modelscope_revision': '8fab2acf1b99296f7d152b9a1d23a7f0b2660958',
 'runtime_manifest_sha256': '7f35ca329602852609f072dfb00c23881aa0aebd342dbb6ae3d95bed73e73521'}

_JP5_MODEL_RELEASE = {'asset_bytes': 318416119,
 'asset_sha256': 'cd7cf76cae81d196ae3ead00b92c208a4b521bbcf69f155ed7e1c690cc3b3af6',
 'modelscope_file': '01yuewen-jp511-fp32-50195-20261125-v1.tar.gz',
 'modelscope_repo': 'lanxijiang/xiaoyou01',
 'modelscope_revision': '0498b3238fe60e1b2f8cc51ac77bbea4e5a3528d',
 'runtime_manifest_sha256': 'd87d4834c79a63ee8202d4c5e21237c1f869aa22bc2dea807161e1764dea882e'}


def detect_jetpack_major():
    image_major = {'511': 5, '61': 6}.get(os.environ.get('XIAOYOU_TTS_IMAGE_JP'))
    try:
        release = Path('/etc/nv_tegra_release').read_text()
    except FileNotFoundError:
        return image_major
    except OSError:
        return None
    match = re.match(r'^#\s*R(35|36)\b', release)
    release_major = {'35': 5, '36': 6}[match[1]] if match else None
    if release_major is not None and image_major is not None and release_major != image_major:
        raise RuntimeError('Xiaoyou JetPack image and L4T release differ')
    return release_major


def _model_release(jetpack_major):
    major = jetpack_major if jetpack_major is not None else detect_jetpack_major()
    return _JP5_MODEL_RELEASE if major == 5 else _MODEL_RELEASE


def _digest(value, name):
    value = str(value or '').lower()
    if not re.fullmatch(r'[0-9a-f]{64}', value):
        raise ValueError('A fixed SHA256 is required for ' + name)
    return value


def _relative(name):
    from utils.model_downloader import _check_bundle_relpath

    if not isinstance(name, str):
        raise ValueError('Invalid bundle relative path')
    _check_bundle_relpath(name)
    if name.count('/') >= 8 or len(name) > 512:
        raise ValueError('Bundle relative path is too long')
    return name


def _verified_bundle(root, expected_manifest_sha256):
    from utils.model_downloader import _verify_pinned_file

    root = Path(root)
    manifest_path = root / MANIFEST
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileNotFoundError('Missing V6 runtime_manifest.json')
    if sha(manifest_path) != expected_manifest_sha256:
        raise ValueError('V6 runtime manifest SHA mismatch')
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('ready') is not True or manifest.get('schema_version') != 1
            or manifest.get('project_version') != 'xiaoyou_v6'
            or manifest.get('model_id') != 'xiaoyou01'
            or manifest.get('voice_id') != VOICE_ID
            or manifest.get('backend') != 'tensorrt_fp32'
            or manifest.get('precision') != 'FP32'
            or manifest.get('sample_rate') != 24000
            or manifest.get('global_step') != GLOBAL_STEP):
        raise ValueError('Wrong V6 GPU runtime bundle identity')
    files = manifest.get('files')
    if not isinstance(files, dict) or not files or MANIFEST in files:
        raise ValueError('Missing or circular V6 file manifest')
    terms = files.get(TERMS_FILE)
    if not isinstance(terms, dict) or terms.get('sha256') != TERMS_SHA:
        raise ValueError('Missing or incorrect pinned V6 lexicon in bundle manifest')
    declared = {MANIFEST}
    total = 0
    for name, item in files.items():
        name = _relative(name)
        if not isinstance(item, dict):
            raise ValueError('Invalid V6 file declaration')
        size = item.get('bytes')
        if type(size) is not int or not 0 < size <= MAX_FILE_BYTES:
            raise ValueError('Invalid V6 file size')
        total += size
        expected = _digest(item.get('sha256'), name)
        path = root / name
        if path.is_symlink() or not path.is_file() or root.resolve() not in path.resolve().parents:
            raise ValueError('Missing or escaping V6 file')
        try:
            _verify_pinned_file(path, {'size': size, 'sha256': expected})
        except ValueError as error:
            raise ValueError('V6 file integrity mismatch: ' + name) from error
        declared.add(name)
    if total > MAX_BUNDLE_BYTES:
        raise ValueError('V6 bundle exceeds the fixed extraction bound')
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('Symlinks are not accepted in V6 bundles')
        if path.is_file() and path.relative_to(root).as_posix() not in declared:
            raise ValueError('Undeclared file in V6 bundle')
    # Schema/weights/frontend/typed IO and GPU long-profile acceptance are checked
    # again by the GPU backend; a successful download is not numerical acceptance.
    return root


@contextmanager
def _lock(path):
    with path.open('a+b') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)



def resolve_runtime_manifest_pin(explicit=None, *, jetpack_major=None):
    """Resolve only an external cfg/env/fixed-Git release pin, never bundle content."""
    release = _model_release(jetpack_major)
    supplied = [value for value in (explicit, os.environ.get('XIAOYOU_V6_RUNTIME_MANIFEST_SHA256'),
                                   release['runtime_manifest_sha256']) if value]
    pins = {_digest(value, 'runtime manifest') for value in supplied}
    if not pins:
        raise ValueError('An external fixed V6 runtime manifest SHA is required')
    if len(pins) != 1:
        raise ValueError('Configured runtime manifest SHA declarations disagree')
    return pins.pop()


def _release_from_environment(jetpack_major=None):
    release = _model_release(jetpack_major)
    expected = _digest(os.environ.get('XIAOYOU_V6_ASSET_SHA256') or release['asset_sha256'], 'model archive')
    try:
        size = int(os.environ.get('XIAOYOU_V6_ASSET_BYTES') or release['asset_bytes'])
    except (TypeError, ValueError):
        raise ValueError('A fixed model archive byte count is required') from None
    if not 0 < size <= MAX_BUNDLE_BYTES:
        raise ValueError('Model archive byte count is outside the supported bound')
    mirror = os.environ.get('XIAOYOU_V6_INTERNAL_TEST_ASSET_URL')
    if mirror:
        if os.environ.get('XIAOYOU_V6_ALLOW_INTERNAL_TEST_ASSETS') != '1':
            raise ValueError('Internal model mirrors require the explicit test opt-in')
        url = mirror
    else:
        repo = os.environ.get('XIAOYOU_V6_MODELSCOPE_REPO') or release['modelscope_repo']
        revision = os.environ.get('XIAOYOU_V6_MODELSCOPE_REVISION') or release['modelscope_revision'] or ''
        name = os.environ.get('XIAOYOU_V6_MODELSCOPE_FILE') or release['modelscope_file']
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*', repo):
            raise ValueError('A fixed ModelScope repository is required')
        if not re.fullmatch(r'[0-9a-fA-F]{40,64}', revision):
            raise ValueError('ModelScope revision must be an immutable commit, never a branch')
        name = _relative(name)
        url = ('https://modelscope.cn/api/v1/models/' + quote(repo, safe='/')
               + '/repo?' + urlencode({'Revision': revision, 'FilePath': name}))
    parsed = urlsplit(url)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('Model archive URL must be HTTP(S) without embedded credentials')
    return url, size, expected


def _download(url, path, expected_size, expected_sha256, *, progress_cb=None):
    request = Request(url, headers={'User-Agent': 'xiaoyou-v6-pinned-assets/1'})
    last_error = None
    mb_total = expected_size / (1024 * 1024)
    if progress_cb is not None:
        with suppress(Exception):
            progress_cb(0, 0, mb_total)
    for attempt in range(3):
        digest = hashlib.sha256()
        count = 0
        last_pct = 0
        try:
            with urlopen(request, timeout=30) as response, path.open('xb') as stream:
                for chunk in iter(lambda: response.read(1024 * 1024), b''):
                    count += len(chunk)
                    if count > expected_size:
                        raise ValueError('Archive exceeds its pinned byte count')
                    stream.write(chunk)
                    digest.update(chunk)
                    if progress_cb is not None:
                        pct = count * 100 // expected_size
                        if last_pct + 10 <= pct < 100:
                            last_pct = pct
                            with suppress(Exception):
                                progress_cb(pct, count / (1024 * 1024), mb_total)
                stream.flush()
                os.fsync(stream.fileno())
            if count != expected_size or digest.hexdigest() != expected_sha256:
                raise ValueError('Downloaded V6 archive size/SHA mismatch')
            if progress_cb is not None:
                with suppress(Exception):
                    progress_cb(100, mb_total, mb_total)
            return
        except Exception as exc:
            last_error = type(exc).__name__
            if path.exists():
                path.unlink()  # Only this invocation's exclusive temporary path.
            if attempt < 2:
                time.sleep(attempt + 1)
    # Never expose signed URLs, credentials, or HTTP request details in exceptions.
    raise RuntimeError('V6 archive download failed after 3 attempts: ' + str(last_error))


def _extract(archive_path, destination, verified_sha=None):
    if verified_sha == _MODEL_RELEASE['asset_sha256']:
        from utils.model_downloader import _extract_verified_tar

        # The published archive has reviewed, bounded regular members. Only
        # use its digest from the completed size- and SHA-checked download.
        _extract_verified_tar(str(archive_path), str(destination))
        for path in destination.rglob('*'):
            if path.is_file():
                with path.open('rb') as stream:
                    os.fsync(stream.fileno())
        return

    # An alternate archive can use a different container layout and headers.
    # Enforce the member/count/expanded-size bounds before writing each file.
    total = 0
    seen = set()
    with tarfile.open(archive_path, 'r:*') as archive:
        for member_number, member in enumerate(archive, 1):
            if member_number > 4096:
                raise ValueError('Too many archive members')
            name = member.name.rstrip('/') if member.isdir() else member.name
            name = _relative(name)
            if name in seen:
                raise ValueError('Duplicate archive member')
            seen.add(name)
            target = destination / name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError('Only ordinary V6 files/directories are accepted')
            if not 0 < member.size <= MAX_FILE_BYTES:
                raise ValueError('Archive member size is outside the fixed bound')
            total += member.size
            if total > MAX_BUNDLE_BYTES:
                raise ValueError('Archive uncompressed size exceeds the fixed bound')
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError('Missing archive member data')
            with source, target.open('xb') as stream:
                shutil.copyfileobj(source, stream, length=1024 * 1024)
                stream.flush()
                os.fsync(stream.fileno())


def ensure_runtime_bundle(model_dir=None, expected_manifest_sha256=None, *, jetpack_major=None,
                          progress_cb=None, stage_cb=None):
    """Return a fully rehashed local bundle; fail closed on any missing pin or mismatch.

    ModelScope repo/revision/archive size/SHA are required only for a cold download.
    A preprovisioned offline bundle is accepted only with the same external manifest
    SHA and complete per-file verification. Existing unknown content is never erased.
    """
    expected = resolve_runtime_manifest_pin(expected_manifest_sha256, jetpack_major=jetpack_major)
    root = Path(model_dir or os.environ.get('XIAOYOU_V6_MODEL_DIR') or '/models/01yuewen').expanduser()
    if root.is_symlink():
        raise ValueError('V6 model root must not be a symlink')
    root = root.resolve()
    if root == root.parent or root.name in ('models', '', '.', '..'):
        raise ValueError('A dedicated V6 model subdirectory is required')
    if root.exists() and any(root.iterdir()):
        return _verified_bundle(root, expected)
    url, size, archive_sha = _release_from_environment(jetpack_major)
    root.parent.mkdir(parents=True, exist_ok=True)
    with _lock(root.parent / ('.' + root.name + '.xiaoyou-v6.lock')):
        if root.exists() and any(root.iterdir()):
            return _verified_bundle(root, expected)
        with tempfile.TemporaryDirectory(prefix='.' + root.name + '.install-', dir=root.parent) as temp:
            temp = Path(temp)
            archive = temp / 'model.tar'
            staging = temp / 'bundle'
            staging.mkdir()
            _download(url, archive, size, archive_sha, progress_cb=progress_cb)
            if stage_cb is not None:
                with suppress(Exception):
                    stage_cb('extract')
            _extract(archive, staging, archive_sha)
            _verified_bundle(staging, expected)
            os.replace(staging, root)  # Complete verified tree, same filesystem.
    return root
