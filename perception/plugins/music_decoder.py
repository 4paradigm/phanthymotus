"""Bounded, cancellable HTTPS → ffmpeg → PCM stream. Audio never goes to disk."""
import http.client
import ipaddress
import queue
import socket
import subprocess
import threading
import urllib.error
import urllib.request
from urllib.parse import urlsplit

FRAME_BYTES = 3200  # 100 ms, PCM S16LE / 16 kHz / mono
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024
MAX_PCM_BYTES = 1800 * 32000


def validate_url(url):
    if not isinstance(url, str) or len(url) > 8192:
        raise ValueError('Audio URL must be an HTTPS URL of at most 8192 characters')
    try:
        parsed = urlsplit(url)
        valid = (parsed.scheme == 'https' and parsed.hostname and not parsed.username
                 and not parsed.password and not parsed.fragment)
        _ = parsed.port
    except ValueError:
        valid = False
    if not valid:
        raise ValueError('Audio URL must use HTTPS without embedded credentials or a fragment')
    return url


def _public_connection(address, timeout=10, source_address=None):
    """Resolve once, check every address, connect to that IP (no DNS rebinding)."""
    host, port = address
    addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
        raise ValueError('Audio hosts must resolve only to public addresses')
    last_error = None
    for family, socktype, proto, _, sockaddr in addresses:
        sock = socket.socket(family, socktype, proto)
        try:
            sock.settimeout(timeout)
            if source_address:
                sock.bind(source_address)
            sock.connect(sockaddr)
            return sock
        except OSError as error:
            last_error = error
            sock.close()
    raise last_error or OSError('No usable audio address')


class _PublicHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._create_connection = _public_connection


class _PublicHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, request):
        return self.do_open(_PublicHTTPSConnection, request, context=self._context)


class _HTTPSRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def open_audio(url):
    # No environment proxy: validation/pinning must cover the actual destination.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}),
                                        _PublicHTTPSHandler(), _HTTPSRedirect())
    return opener.open(urllib.request.Request(validate_url(url), headers={
        'Accept': 'audio/mpeg, audio/wav', 'User-Agent': 'MotusMusic/1'}), timeout=10)


class Decoder:
    def __init__(self, url, audio_format='mp3', *, ffmpeg='ffmpeg'):
        self.url = validate_url(url)
        if audio_format not in ('mp3', 'wav'):
            raise ValueError('Only mp3 and wav audio are supported')
        self.audio_format, self.ffmpeg = audio_format, ffmpeg
        self.frames = queue.Queue(maxsize=20)  # two seconds of decoded audio
        self.done = threading.Event()
        self.cancelled = threading.Event()
        self.error = ''
        self._process = None
        self._thread = None
        self._feed_thread = None
        self._lock = threading.Lock()

    def start(self):
        self._thread = threading.Thread(target=self._run, name='music-decode', daemon=True)
        self._thread.start()

    def _feed(self, process):
        try:
            with open_audio(self.url) as response:
                total = 0
                while not self.cancelled.is_set():
                    chunk = response.read(32768)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_DOWNLOAD_BYTES:
                        raise ValueError('Audio exceeds 50 MiB')
                    process.stdin.write(chunk)
        except Exception as exc:
            if not self.cancelled.is_set():
                # Do not put signed URLs or remote response bodies in status/logs.
                self.error = ('Audio URL expired or unavailable; refresh the track by ID'
                              if isinstance(exc, urllib.error.HTTPError) and exc.code in (401, 403, 404)
                              else 'Audio download failed or exceeded its limit')
                self._kill()
            if isinstance(exc, urllib.error.HTTPError):
                exc.close()
        finally:
            try:
                process.stdin.close()
            except (OSError, ValueError):
                pass

    def _kill(self):
        with self._lock:
            process = self._process
            if process and process.poll() is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass

    def cancel(self):
        self.cancelled.set()
        self._kill()

    def close(self):
        self.cancel()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=2)

    def is_closed(self):
        return (self.done.is_set() and not (self._thread and self._thread.is_alive())
                and not (self._feed_thread and self._feed_thread.is_alive()))

    def _run(self):
        process = None
        try:
            # ffmpeg sees only bytes on stdin. It cannot open nested playlist URLs,
            # local files, or another protocol named by malicious media content.
            process = subprocess.Popen([
                self.ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error',
                '-protocol_whitelist', 'pipe', '-f', self.audio_format, '-i', 'pipe:0',
                '-vn', '-ac', '1', '-ar', '16000', '-f', 's16le', 'pipe:1',
            ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            with self._lock:
                self._process = process
            if self.cancelled.is_set():
                self._kill()
                return
            self._feed_thread = threading.Thread(target=self._feed, args=(process,),
                                                 name='music-download', daemon=True)
            self._feed_thread.start()
            total = 0
            while not self.cancelled.is_set():
                frame = process.stdout.read(FRAME_BYTES)
                if not frame:
                    break
                total += len(frame)
                if total > MAX_PCM_BYTES:
                    raise ValueError('Audio exceeds 30 minutes')
                while not self.cancelled.is_set():
                    try:
                        self.frames.put(frame, timeout=0.1)
                        break
                    except queue.Full:
                        continue
            if not self.cancelled.is_set():
                code = process.wait(timeout=2)
                if code and not self.error:
                    self.error = 'Audio decoding failed'
                elif total == 0 and not self.error:
                    self.error = 'Audio contains no playable samples'
        except FileNotFoundError:
            self.error = 'ffmpeg is not installed in this Perception image'
        except Exception:
            if not self.cancelled.is_set():
                self.error = 'Audio decoder failed or exceeded its duration limit'
        finally:
            self._kill()
            if process:
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
                process.stdout.close()
                if not self._feed_thread:
                    process.stdin.close()
            self.done.set()
