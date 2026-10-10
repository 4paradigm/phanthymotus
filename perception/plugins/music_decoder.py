"""Bounded, cancellable HTTPS → ffmpeg → PCM stream. Audio never goes to disk."""
import http.client
import ipaddress
import json
import queue
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlencode, urlsplit

FRAME_BYTES = 3200  # 100 ms, PCM S16LE / 16 kHz / mono
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024
MAX_PCM_BYTES = 1800 * 32000
_FAKE_IP_RANGE = ipaddress.ip_network('198.18.0.0/15')
_DNS_RESPONSE_LIMIT = 16384


class AudioAddressError(ValueError):
    """Safe, actionable transport messages: never contain a URL or response body."""


class _DNSNoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise AudioAddressError('Public DNS redirect refused; configure real DNS for audio')


def _query_public_dns(endpoint, host, timeout):
    deadline = time.monotonic() + timeout
    url = endpoint + '?' + urlencode({
        'name': host.encode('idna').decode('ascii'), 'type': 'A',
        'edns_client_subnet': '0.0.0.0/0'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _DNSNoRedirect())
    request = urllib.request.Request(url, headers={'Accept': 'application/dns-json'})
    with opener.open(request, timeout=timeout) as response:
        body = bytearray()
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError
            chunk = response.read1(4096)
            if not chunk:
                break
            body.extend(chunk)
            if len(body) > _DNS_RESPONSE_LIMIT:
                raise ValueError
    data = json.loads(body)
    if not isinstance(data, dict):
        raise ValueError
    answers = data.get('Answer')
    if (type(data.get('Status')) is not int or data['Status'] != 0
            or data.get('TC') or not isinstance(answers, list)
            or not 1 <= len(answers) <= 32):
        raise ValueError
    addresses = []
    for record in answers:
        if not isinstance(record, dict):
            raise ValueError
        if record.get('type') == 1:
            ip = ipaddress.ip_address(record['data'])
            if ip.version != 4 or not ip.is_global:
                raise ValueError
            addresses.append(str(ip))
    if not addresses:
        raise ValueError
    return list(dict.fromkeys(addresses))


def _public_dns_addresses(host, timeout):
    """Recover only proxy fake-IP answers using fixed, verified HTTPS resolvers.

    Send only the hostname, never a media path/query or catalogue credential.
    Network failures may try the second resolver; an invalid/private answer or
    redirect fails closed instead of shopping for a more permissive answer.
    """
    deadline = time.monotonic() + min(8.0, timeout)
    for endpoint in ('https://cloudflare-dns.com/dns-query', 'https://dns.google/resolve'):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            return _query_public_dns(endpoint, host, min(3.0, remaining))
        except (OSError, http.client.HTTPException) as exc:
            if isinstance(exc, urllib.error.HTTPError):
                exc.close()
                if exc.code != 429 and exc.code < 500:
                    break
        except (ValueError, TypeError, KeyError):
            break
    raise AudioAddressError('Audio DNS returned a proxy fake-IP, but public DNS lookup failed; '
                            'configure real DNS or exclude audio domains from proxy fake-IP') from None


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
    """Validate/pin DNS results; replace only an all-fake-IP answer, never private IPs."""
    host, port = address
    deadline = time.monotonic() + timeout
    addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    ips = [ipaddress.ip_address(item[4][0]) for item in addresses]
    if ips and all(ip in _FAKE_IP_RANGE for ip in ips):
        # A literal non-public IP is never a DNS compatibility case.
        try:
            ipaddress.ip_address(host)
        except ValueError:
            resolved = _public_dns_addresses(host, deadline - time.monotonic())
            addresses = [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, '', (ip, port))
                         for ip in resolved]
    if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
        raise AudioAddressError('Audio host must resolve only to public addresses; private/local audio is not allowed')
    last_error = None
    for family, socktype, proto, _, sockaddr in addresses:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Audio connection timed out')
        sock = socket.socket(family, socktype, proto)
        try:
            sock.settimeout(remaining)
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
                self.error = (str(exc) if isinstance(exc, AudioAddressError) else
                              'Audio URL expired or unavailable; refresh the track by ID'
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
