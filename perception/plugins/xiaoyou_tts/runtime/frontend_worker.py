"""Run Xiaoyou's pinned G2P dependencies outside the Perception interpreter."""

import os
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from pathlib import Path
import platform
import subprocess
import sys
import time

from .model_bundle import TERMS_FILE, detect_jetpack_major


PYTHON = '/opt/xiaoyou-tts/bin/python3'


def require_runtime():
    """Check local prerequisites before loading a model or stopping another engine."""
    if platform.machine() != 'aarch64':
        raise RuntimeError('xiaoyou-zh-en requires JetPack on aarch64')
    major = detect_jetpack_major()
    if major is None:
        raise RuntimeError('xiaoyou-zh-en requires JetPack 5 or 6 (L4T R35 or R36)')
    if not Path(PYTHON).is_file() or not os.access(PYTHON, os.X_OK):
        raise RuntimeError('The Xiaoyou frontend environment is not installed in this image')
    return major


class FrontendWorker:
    INITIAL_TIMEOUT = 60
    REQUEST_TIMEOUT = 15

    def __init__(self, model_dir):
        self._model_dir = model_dir
        self._connection = None
        self._process = None
        self._closed = False
        self._failure = None
        self.metadata, self.legal_ids, self.dictionary_seconds = self._start(self.INITIAL_TIMEOUT)

    @property
    def failure_reason(self):
        process = self._process
        if self._failure is not None:
            return self._failure
        if process is not None and process.poll() is not None:
            return 'Xiaoyou frontend worker stopped'
        return None

    def _start(self, timeout):
        env = os.environ.copy()
        for name in ('PYTHONHOME', 'PYTHONPATH'):
            env.pop(name, None)
        self._connection, child = Pipe()
        try:
            with child:
                self._process = subprocess.Popen(
                    [PYTHON, '-Bs', '-m', __name__, str(child.fileno()), str(self._model_dir)],
                    pass_fds=(child.fileno(),), env=env, cwd=Path(__file__).resolve().parents[3],
                )
            hello = self._receive(timeout)
            if hasattr(self, 'metadata') and hello[:2] != (self.metadata, self.legal_ids):
                raise RuntimeError('Xiaoyou frontend worker restarted with different metadata or token IDs')
        except BaseException as error:
            self._retire(str(error))
            raise
        self._failure = None
        return hello

    def _receive(self, timeout):
        try:
            if not self._connection.poll(timeout):
                raise TimeoutError('Xiaoyou frontend worker timed out')
            status, value = self._connection.recv()
        except EOFError as error:
            raise BrokenPipeError('Xiaoyou frontend worker stopped') from error
        if status == 'ok':
            return value
        if status == 'invalid':
            raise ValueError(value)
        raise RuntimeError(value)

    def normalize_text(self, *args, **options):
        return self._call('normalize', args, options)

    def __call__(self, *args, **options):
        return self._call('phonemize', args, options)

    def phonemize_normalized(self, *args, **options):
        return self._call('normalized_span', args, options)

    def _call(self, operation, args, options):
        # The backend serializes RPC and close. Recover before sending a new
        # operation so an interrupted or partial reply is never replayed.
        if self._closed:
            raise RuntimeError('Xiaoyou frontend worker is closed')
        deadline = time.monotonic() + self.REQUEST_TIMEOUT
        process = self._process
        if process is None or process.poll() is not None:
            self._retire(self._failure or 'Xiaoyou frontend worker stopped')
            self._start(max(0.0, deadline - time.monotonic()))
        try:
            self._connection.send((operation, args, options))
            return self._receive(max(0.0, deadline - time.monotonic()))
        except TimeoutError as error:
            self._retire(str(error))
            raise
        except OSError as error:
            message = 'Xiaoyou frontend worker stopped'
            self._retire(message)
            raise RuntimeError(message) from error

    def _retire(self, reason=None):
        self._failure = reason
        connection, self._connection = self._connection, None
        process, self._process = self._process, None
        if connection is not None:
            connection.close()
        if process is None:
            return
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def close(self):
        self._closed = True
        self._retire()


def run(connection, model_dir):
    # Thinc supports a NumPy-only frontend if Torch is unavailable. Keep this
    # import policy inside the dedicated process.
    sys.modules['torch'] = None
    try:
        from importlib.metadata import version

        import jieba

        from ..frontend.phonemizer import Frontend

        if version('en-core-web-sm') != '3.8.0':
            raise RuntimeError('Install the pinned Xiaoyou spaCy model in the frontend environment')
        started = time.perf_counter()
        jieba.initialize()
        if not jieba.dt.initialized:
            raise RuntimeError('Xiaoyou Jieba dictionary initialization did not complete')
        dictionary_seconds = time.perf_counter() - started
        frontend = Frontend(config_path=model_dir / 'config.json', terms_path=model_dir / TERMS_FILE)
        metadata = {key: frontend.metadata[key] for key in (
            'frontend_sha256', 'terms_sha256', 'normalization_module_sha256',
        )}
        connection.send(('ok', (metadata, set(frontend.vocab.values()) | {0}, dictionary_seconds)))
    except Exception as error:
        connection.send(('error', f'{type(error).__name__}: {error}'))
        return

    methods = {'normalize': frontend.normalize_text, 'phonemize': frontend,
               'normalized_span': frontend.phonemize_normalized}
    while True:
        try:
            operation, args, options = connection.recv()
        except EOFError:
            return
        try:
            value = methods[operation](*args, **options)
        except ValueError as error:
            connection.send(('invalid', str(error)))
        except Exception as error:
            connection.send(('error', f'{type(error).__name__}: {error}'))
        else:
            connection.send(('ok', value))


if __name__ == '__main__':
    with Connection(int(sys.argv[1])) as channel:
        run(channel, Path(sys.argv[2]))
