"""Product regressions for X-ASR device selection and optional LM settings."""
import hashlib
from pathlib import Path
import re
import sys
import types
import urllib.request

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plugins.x_asr import XASRAdapter
from utils import onnx_provider


@pytest.fixture
def bundle(tmp_path):
    for name in ('encoder-epoch-99-avg-1.int8.onnx', 'encoder-epoch-99-avg-1.onnx',
                 'decoder-epoch-99-avg-1.onnx', 'joiner-epoch-99-avg-1.int8.onnx',
                 'joiner-epoch-99-avg-1.onnx', 'tokens.txt', 'bpe.model',
                 'bpe.vocab', 'hotwords.txt', 'lm.onnx'):
        (tmp_path/name).touch()
    (tmp_path/'hotwords.bpe.txt').write_text('语 音 :2.5\n测 试 :2.5\n', encoding='utf-8')
    return tmp_path


@pytest.fixture
def runtime(monkeypatch):
    calls = []
    module = types.SimpleNamespace(
        XASR_PREFIX_LM_VERSION=1, XASR_EARLY_HOTWORD_VERSION=1,
        OfflineRecognizer=types.SimpleNamespace(from_transducer=lambda **kw: calls.append(kw) or object()))
    monkeypatch.setitem(sys.modules, 'sherpa_onnx', module)
    monkeypatch.setattr(onnx_provider, 'provider_for_device', lambda device, *a: 'cuda' if device == 'gpu' else 'cpu')
    monkeypatch.setenv('SHERPA_ONNX_EARLY_HOTWORD_MIN_SCORE', '3.9')
    return module, calls


@pytest.mark.parametrize('device,infix', [('cpu', '.int8'), ('gpu', '')])
def test_weight_precision_follows_device(bundle, runtime, device, infix):
    _, calls = runtime
    XASRAdapter(str(bundle), device)
    assert calls[0]['encoder'].endswith(f'encoder-epoch-99-avg-1{infix}.onnx')
    assert calls[0]['joiner'].endswith(f'joiner-epoch-99-avg-1{infix}.onnx')


def test_stock_runtime_rejects_prefix_lm(bundle, runtime):
    module, _ = runtime
    module.XASR_PREFIX_LM_VERSION = 0
    with pytest.raises(RuntimeError, match='prefix-enabled'):
        XASRAdapter(str(bundle), prefix_lm_path=str(bundle/'lm.onnx'), prefix_lm_scale=.05)


def test_entity_boost_requires_prefix_lm(bundle, runtime):
    with pytest.raises(ValueError, match='requires prefix LM'):
        XASRAdapter(str(bundle), entity_boost={'语音':4.0})


def test_entity_boost_requires_matching_runtime(bundle, runtime):
    module, _ = runtime
    module.XASR_EARLY_HOTWORD_VERSION = 0
    with pytest.raises(RuntimeError, match='entity-enabled'):
        XASRAdapter(str(bundle), prefix_lm_path=str(bundle/'lm.onnx'),
                    prefix_lm_scale=.05, entity_boost={'语音':4.0})


def test_entity_setting_is_adapter_local_and_packaged_hotwords_are_restored(bundle, runtime):
    import os
    _, calls = runtime
    options = dict(prefix_lm_path=str(bundle/'lm.onnx'), prefix_lm_scale=.05)
    XASRAdapter(str(bundle), entity_boost={'语音':4.0}, **options)
    assert os.environ['SHERPA_ONNX_EARLY_HOTWORD_MIN_SCORE'] == '3.9'
    assert '语 音 :4.0' in Path(calls[-1]['hotwords_file']).read_text()
    XASRAdapter(str(bundle), entity_boost={}, **options)
    assert os.environ['SHERPA_ONNX_EARLY_HOTWORD_MIN_SCORE'] == '3.9'
    assert calls[-1]['hotwords_file'] == str(bundle/'hotwords.bpe.txt')
    assert '语 音 :2.5' in Path(calls[-1]['hotwords_file']).read_text()


def _wheel_download(monkeypatch, tmp_path, checksum):
    dockerfile = (ROOT/'Dockerfile.jetson').read_text()
    source = re.search(r'python3 -c "(import urllib\.request, urllib\.parse, sys, hashlib, pathlib;.*?)" \\\n',
                       dockerfile, re.S).group(1).replace('\\\n', '')
    payload = b'wheel fixture'
    calls = []
    def download(url, destination):
        calls.append(url)
        Path(destination).write_bytes(payload)
    monkeypatch.setattr(urllib.request, 'urlretrieve', download)
    monkeypatch.setattr(sys, 'argv', ['download', 'https://example.invalid/custom.whl',
                                    str(tmp_path/'custom.whl'), checksum])
    exec(source, {})
    return calls


def test_custom_wheel_is_verified_with_explicit_checksum(monkeypatch, tmp_path):
    checksum = hashlib.sha256(b'wheel fixture').hexdigest()
    assert _wheel_download(monkeypatch, tmp_path, checksum) == ['https://example.invalid/custom.whl']


def test_custom_wheel_wrong_checksum_fails(monkeypatch, tmp_path):
    with pytest.raises(AssertionError, match='checksum mismatch'):
        _wheel_download(monkeypatch, tmp_path, '0'*64)


def test_custom_wheel_missing_checksum_fails(monkeypatch, tmp_path):
    with pytest.raises(AssertionError, match='SHA256'):
        _wheel_download(monkeypatch, tmp_path, '')


def test_custom_words_merge_chinese_english_and_keep_packaged_source(bundle):
    from plugins.x_asr import _custom_hotwords_file
    source = bundle/'hotwords.bpe.txt'
    original = source.read_bytes()
    result = _custom_hotwords_file(source, '语音\n星河展厅\nFancy Robot\nFancy Robot\n\n')
    try:
        text = result.read_text()
        assert '语 音 :4.0' in text
        assert '测 试 :2.5' in text
        assert '星 河 展 厅 :4.0' in text
        assert text.count('Fancy Robot :4.0') == 1
        assert source.read_bytes() == original
    finally:
        result.unlink()
    assert _custom_hotwords_file(source, ' \n') == source


def test_custom_hotwords_enable_early_scoring_and_release_temporary_file(bundle, runtime):
    import os
    module, calls = runtime
    def build(**kwargs):
        calls.append(kwargs)
        assert 'Fancy Robot :4.0' in Path(kwargs['hotwords_file']).read_text()
        return object()
    module.OfflineRecognizer.from_transducer = build
    XASRAdapter(str(bundle), prefix_lm_path=str(bundle/'lm.onnx'),
                prefix_lm_scale=.05, custom_hotwords='Fancy Robot')
    assert os.environ['SHERPA_ONNX_EARLY_HOTWORD_MIN_SCORE'] == '3.9'
    assert not Path(calls[-1]['hotwords_file']).exists()


def _wav_bytes():
    import io
    import wave
    data = io.BytesIO()
    with wave.open(data, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b'\0\0' * 160)
    return data.getvalue()


@pytest.mark.parametrize('previous', [None, '7.1'])
@pytest.mark.parametrize('fails', [False, True])
def test_decode_scopes_and_restores_environment(bundle, runtime, monkeypatch, previous, fails):
    import os
    name = 'SHERPA_ONNX_EARLY_HOTWORD_MIN_SCORE'
    if previous is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, previous)
    module, _ = runtime
    seen = []
    class Recognizer:
        def create_stream(self):
            return types.SimpleNamespace(accept_waveform=lambda *a: None,
                                         result=types.SimpleNamespace(text='hello'))
        def decode_streams(self, streams):
            seen.append(os.environ.get(name))
            if fails:
                raise RuntimeError('decode failed')
    module.OfflineRecognizer.from_transducer = lambda **kw: Recognizer()
    adapter = XASRAdapter(str(bundle), prefix_lm_path=str(bundle/'lm.onnx'),
                         prefix_lm_scale=.05, custom_hotwords='hello')
    assert os.environ.get(name) == previous
    if fails:
        with pytest.raises(RuntimeError, match='decode failed'):
            adapter.transcribe(_wav_bytes(), '')
    else:
        assert adapter.transcribe(_wav_bytes(), '') == 'hello'
    assert seen == ['3.9']
    assert os.environ.get(name) == previous


def test_mixed_adapters_decode_concurrently_without_leaking_state(bundle, runtime, monkeypatch):
    import os
    import threading
    from concurrent.futures import ThreadPoolExecutor
    name = 'SHERPA_ONNX_EARLY_HOTWORD_MIN_SCORE'
    monkeypatch.setenv(name, '7.1')
    module, _ = runtime
    entered, release, second_started = threading.Event(), threading.Event(), threading.Event()
    observations = []
    class Recognizer:
        def __init__(self, custom):
            self.custom = custom
        def create_stream(self):
            return types.SimpleNamespace(accept_waveform=lambda *a: None,
                                         result=types.SimpleNamespace(text='ok'))
        def decode_streams(self, streams):
            before = os.environ.get(name)
            if self.custom:
                entered.set()
                assert release.wait(3)
            observations.append((self.custom, before, os.environ.get(name)))
    module.OfflineRecognizer.from_transducer = lambda **kw: Recognizer(
        ':4.0' in Path(kw['hotwords_file']).read_text())
    custom = XASRAdapter(str(bundle), prefix_lm_path=str(bundle/'lm.onnx'),
                        prefix_lm_scale=.05, custom_hotwords='hello')
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(custom.transcribe, _wav_bytes(), '')
        try:
            assert entered.wait(3)
            # A WebSocket recognizer can be constructed during plugin decoding.
            plain = XASRAdapter(str(bundle))
            def second():
                second_started.set()
                return plain.transcribe(_wav_bytes(), '')
            other = pool.submit(second)
            assert second_started.wait(3)
        finally:
            release.set()
        assert first.result(timeout=3) == 'ok'
        assert other.result(timeout=3) == 'ok'
    assert sorted(observations) == [(False, None, None), (True, '3.9', '3.9')]
    assert os.environ.get(name) == '7.1'
