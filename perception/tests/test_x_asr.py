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


def test_disabling_entity_boost_clears_setting_and_restores_hotwords(bundle, runtime):
    import os
    _, calls = runtime
    options = dict(prefix_lm_path=str(bundle/'lm.onnx'), prefix_lm_scale=.05)
    XASRAdapter(str(bundle), entity_boost={'语音':4.0}, **options)
    assert os.environ['SHERPA_ONNX_EARLY_HOTWORD_MIN_SCORE'] == '3.9'
    assert '语 音 :4.0' in Path(calls[-1]['hotwords_file']).read_text()
    XASRAdapter(str(bundle), entity_boost={}, **options)
    assert os.environ.get('SHERPA_ONNX_EARLY_HOTWORD_MIN_SCORE') is None
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
