"""Private COS multipart boundary; lifecycle records are owned by EvidenceService.

All keys passed here must come from server-owned persisted records. No method
accepts a browser URL. SDK I/O runs in bounded worker threads, never the loop.
"""
from __future__ import annotations

import asyncio
import hashlib
import re
import time
import unicodedata
import zlib
from datetime import datetime, timezone, timedelta
from urllib.parse import parse_qsl, quote, unquote, urlsplit
from zoneinfo import ZoneInfo

from .config import validate_repository_names
from .cos_client import EVIDENCE_PRESIGNED_URL_TTL_SECONDS


class EvidenceStoreError(Exception):
    """An object is unavailable, incomplete, or failed integrity verification."""


def safe_filename(filename: str) -> str:
    if (not isinstance(filename, str) or not filename or filename in {'.', '..'}
            or '/' in filename or '\\' in filename
            or any(unicodedata.category(c).startswith('C') for c in filename)
            or len(filename.encode('utf-8')) > 255):
        raise ValueError('invalid filename')
    return filename


def _segment(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', value):
        raise ValueError('invalid identity')
    return value


def build_attachment_key(repo, pr_number, head_sha, round_id, attachment_id, uploaded_at, filename) -> str:
    canonical = validate_repository_names([repo])[0]
    if isinstance(pr_number, bool) or not isinstance(pr_number, int) or pr_number < 1:
        raise ValueError('invalid PR')
    if not isinstance(head_sha, str) or not re.fullmatch(r'[0-9a-fA-F]{40}', head_sha):
        raise ValueError('full HEAD SHA required')
    stamp = datetime.fromisoformat(uploaded_at.replace('Z', '+00:00'))
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError('uploaded_at must include timezone')
    stamp = stamp.astimezone(ZoneInfo('Asia/Shanghai'))
    # Use the existing injective repository directory convention: legacy
    # repository names keep short paths; other owners cannot collide.
    from .cos_client import _repo_directory
    name = _repo_directory(canonical)
    identity = _segment(attachment_id)
    return (f'phanthymotus_pr/{name}/{stamp:%Y-%m}/{stamp:%Y-%m-%d}/'
            f'pr-{pr_number}/head-{head_sha.lower()}/round-{_segment(round_id)}/'
            f'attachments/{identity}/{safe_filename(filename)}')


class CosAttachmentStore:
    def __init__(self, config, client=None):
        self.config = config
        self._client = client
        self._slots = asyncio.Semaphore(4)

    def _sdk(self):
        if self._client is None:
            from qcloud_cos import CosConfig, CosS3Client
            import requests

            class EvidenceSession(requests.Session):
                def request(self, method, url, **kwargs):
                    # The SDK's CompleteMultipartUpload supplies a 20-minute
                    # override. Keep every socket operation bounded instead.
                    kwargs['timeout'] = (10, 30)
                    kwargs['verify'] = True
                    return super().request(method, url, **kwargs)

            session = EvidenceSession()
            session.trust_env = False
            self._client = CosS3Client(CosConfig(
                Region=self.config.cos_region, SecretId=self.config.cos_secret_id,
                SecretKey=self.config.cos_secret_key, Scheme='https',
                Timeout=30, VerifySSL=True), retry=0, session=session)
        return self._client

    def _key(self, key):
        if (not isinstance(key, str) or not key.startswith('phanthymotus_pr/')
                or any(c in key for c in ('\\', '\r', '\n', '\x00'))
                or any(p in ('', '.', '..') for p in key.split('/'))):
            raise ValueError('invalid evidence key')
        return key

    @staticmethod
    def _part_ttl(ttl):
        if isinstance(ttl, bool) or not isinstance(ttl, int) or not 1 <= ttl <= 900:
            raise ValueError('TTL must be 1..900 seconds')
        return ttl

    @staticmethod
    def _download_ttl(ttl):
        from .cos_client import EVIDENCE_PRESIGNED_URL_TTL_SECONDS
        if (isinstance(ttl, bool) or not isinstance(ttl, int)
                or not 1 <= ttl <= EVIDENCE_PRESIGNED_URL_TTL_SECONDS):
            raise ValueError('download TTL exceeds the configured evidence lifetime')
        return ttl

    def issue_part_url(self, *, key, upload_id, part_number, ttl=900) -> dict:
        self._key(key)
        if not isinstance(upload_id, str) or not upload_id or len(upload_id) > 512:
            raise ValueError('invalid upload ID')
        if isinstance(part_number, bool) or not isinstance(part_number, int) or not 1 <= part_number <= 10000:
            raise ValueError('invalid part number')
        ttl = self._part_ttl(ttl)
        url = self._sdk().get_presigned_url(
            Method='PUT', Bucket=self.config.cos_bucket, Key=key, Expired=ttl,
            Params={'uploadId': upload_id, 'partNumber': str(part_number)})
        return {'url': url, 'method': 'PUT', 'headers': {},
                'expires_at': (datetime.now(timezone.utc) + timedelta(seconds=ttl)).isoformat()}

    def issue_download_url(self, *, key, filename, ttl=None) -> str:
        from .cos_client import EVIDENCE_PRESIGNED_URL_TTL_SECONDS
        if ttl is None:
            ttl = EVIDENCE_PRESIGNED_URL_TTL_SECONDS
        disposition = "attachment; filename=download; filename*=UTF-8''" + quote(safe_filename(filename), safe='')
        return self._sdk().get_presigned_url(
            Method='GET', Bucket=self.config.cos_bucket, Key=self._key(key), Expired=self._download_ttl(ttl),
            Params={'response-content-disposition': disposition,
                    'response-content-type': 'application/octet-stream'})

    def issue_preview_url(
        self, *, key, filename, ttl=EVIDENCE_PRESIGNED_URL_TTL_SECONDS,
    ) -> str:
        filename = safe_filename(filename)
        media_types = {
            '.mp4': 'video/mp4', '.webm': 'video/webm', '.png': 'image/png',
            '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.gif': 'image/gif',
            '.webp': 'image/webp',
        }
        suffix = '.' + filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
        media_type = media_types.get(suffix)
        if media_type is None:
            raise EvidenceStoreError('inline preview media type is not allowed')
        ttl = self._download_ttl(ttl)
        return self._sdk().get_presigned_url(
            Method='GET', Bucket=self.config.cos_bucket, Key=self._key(key), Expired=ttl,
            Params={'response-content-disposition': 'inline',
                    'response-content-type': media_type})

    def download_url_expiry(self, url, *, key, filename=None, attachment_id=None):
        """Return a trusted signed-URL expiry, or None for any mismatch.

        Reuse is limited to links previously projected in the trusted lifecycle
        comment. The exact configured bucket host, object path, attachment
        identity, COS signing fields, response disposition, and one-year window
        must all match before a prior URL is retained.
        """
        from .cos_client import EVIDENCE_PRESIGNED_URL_TTL_SECONDS
        try:
            key = self._key(key)
            if attachment_id is not None and f'/attachments/{_segment(attachment_id)}/' not in f'/{key}/':
                return None
            parsed = urlsplit(url)
            expected_host = f'{self.config.cos_bucket}.cos.{self.config.cos_region}.myqcloud.com'
            if (parsed.scheme != 'https' or parsed.hostname != expected_host
                    or parsed.username is not None or parsed.password is not None
                    or parsed.port is not None or parsed.fragment
                    or unquote(parsed.path.lstrip('/')) != key):
                return None
            pairs = parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=True)
            query = {}
            for name, value in pairs:
                if name in query:
                    return None
                query[name] = value
            required = {'q-sign-algorithm', 'q-ak', 'q-sign-time', 'q-key-time',
                        'q-header-list', 'q-url-param-list', 'q-signature',
                        'response-content-disposition'}
            if (not required <= query.keys() or not query['q-ak']
                    or query['q-ak'] != self.config.cos_secret_id):
                return None
            if query['q-sign-algorithm'] not in {'sha1', 'sha256'}:
                return None
            signature = query['q-signature']
            if not re.fullmatch(r'[0-9a-fA-F]{40,128}', signature) or len(signature) % 2:
                return None
            time_match = re.fullmatch(r'([0-9]+);([0-9]+)', query['q-sign-time'])
            if not time_match or query['q-key-time'] != query['q-sign-time']:
                return None
            start, expiry = map(int, time_match.groups())
            signed_lifetime = expiry - start
            if (expiry <= start
                    or not EVIDENCE_PRESIGNED_URL_TTL_SECONDS <= signed_lifetime
                    <= EVIDENCE_PRESIGNED_URL_TTL_SECONDS + 120):
                return None
            signed_params = {part.lower() for part in query['q-url-param-list'].split(';') if part}
            if 'response-content-disposition' not in signed_params:
                return None
            if ('response-content-type' in query
                    and ('response-content-type' not in signed_params
                         or query['response-content-type'] != 'application/octet-stream')):
                return None
            if filename is not None:
                disposition = query['response-content-disposition']
                safe_name = safe_filename(filename)
                if ("filename*=UTF-8''" + quote(safe_name, safe='') not in disposition
                        and f'filename="{safe_name}"' not in disposition
                        and f'filename={safe_name}' not in disposition):
                    return None
            now = int(time.time())
            if start > now or expiry <= now:
                return None
            return expiry
        except (TypeError, ValueError, UnicodeError):
            return None

    async def _io(self, function, **kwargs):
        async with self._slots:
            # The SDK has finite connection/read timeouts and retries. Await its
            # worker to completion so a timeout never starts a duplicate writer.
            attempts = 2 if getattr(function, '__name__', '') in {
                '_verify', 'read', 'find', 'list_parts', 'head_object', 'get_object'} else 1
            for attempt in range(attempts):
                try:
                    return await asyncio.to_thread(function, **kwargs)
                except (ValueError, EvidenceStoreError):
                    raise
                except Exception:
                    if attempt + 1 == attempts:
                        raise EvidenceStoreError('COS operation unavailable') from None
                    await asyncio.sleep(0.25)

    @staticmethod
    def _metadata(metadata):
        required = {'repo', 'pr_number', 'head_sha', 'round_id', 'attachment_id'}
        if not isinstance(metadata, dict) or not required <= metadata.keys():
            raise ValueError('object identity metadata required')
        result = {}
        for name in required:
            value = str(metadata[name])
            if not value or any(ord(c) < 32 or ord(c) > 126 for c in value):
                raise ValueError('invalid object metadata')
            result['x-cos-meta-' + name.replace('_', '-')] = value
        return result

    async def init_upload(self, *, key, metadata, content_type='application/octet-stream', **_):
        response = await self._io(self._sdk().create_multipart_upload,
            Bucket=self.config.cos_bucket, Key=self._key(key),
            ContentType='application/octet-stream', Metadata=self._metadata(metadata))
        return {'upload_id': response['UploadId'], 'uploaded_parts': []}

    async def list_parts(self, *, key, upload_id):
        def read():
            parts, marker = [], 0
            for _ in range(101):
                response = self._sdk().list_parts(Bucket=self.config.cos_bucket,
                    Key=self._key(key), UploadId=upload_id, PartNumberMarker=marker, MaxParts=1000)
                page = response.get('Part', [])
                if isinstance(page, dict):
                    page = [page]
                for item in page:
                    parts.append({'part_number': int(item['PartNumber']),
                                  'etag': item['ETag'], 'size': int(item['Size'])})
                if str(response.get('IsTruncated', 'false')).lower() != 'true':
                    return parts
                next_marker = int(response['NextPartNumberMarker'])
                if next_marker <= marker or len(parts) > 10000:
                    break
                marker = next_marker
            raise EvidenceStoreError('invalid COS part listing')
        return await self._io(read)

    async def put_part(self, *, operation, part_number, body, content_length):
        """Store one bounded raw browser part against a trusted operation."""
        key = self._operation_key(operation)
        upload_id = operation.get('upload_id')
        chunk = operation.get('part_bytes', self.config.evidence_part_bytes)
        size = operation.get('size')
        if (not isinstance(upload_id, str) or not upload_id
                or type(part_number) is not int or not 1 <= part_number <= 10000
                or type(chunk) is not int or not 1 <= chunk <= 8 * 1024 * 1024
                or type(size) is not int or not 1 <= size <= 512 * 1024 * 1024
                or type(content_length) is not int or not 1 <= content_length <= chunk
                or not hasattr(body, 'read')):
            raise ValueError('invalid multipart part')
        count = (size + chunk - 1) // chunk
        expected_size = min(chunk, size - (part_number - 1) * chunk)
        if part_number > count or content_length != expected_size:
            raise ValueError('multipart part size mismatch')

        digest = hashlib.md5(usedforsecurity=False)
        read = 0
        while read < content_length:
            block = await asyncio.to_thread(body.read, min(1024 * 1024, content_length - read))
            if not block:
                raise EvidenceStoreError('multipart part ended early')
            read += len(block)
            digest.update(block)
        if await asyncio.to_thread(body.read, 1):
            raise EvidenceStoreError('multipart part exceeded declared size')
        expected_etag = digest.hexdigest()
        await asyncio.to_thread(body.seek, 0)

        def matching_part(parts):
            existing = next((p for p in parts if p['part_number'] == part_number), None)
            if existing is None:
                return None
            etag = str(existing.get('etag', '')).strip('"').lower()
            return existing['size'] == content_length and etag == expected_etag

        parts = await self.list_parts(key=key, upload_id=upload_id)
        existing = next((p for p in parts if p['part_number'] == part_number), None)
        if existing is not None:
            return {'conflict': not matching_part(parts), 'uploaded': matching_part(parts) is True,
                    'part_number': part_number, 'size': content_length}
        try:
            result = await self._io(self._sdk().upload_part,
                Bucket=self.config.cos_bucket, Key=key, UploadId=upload_id,
                PartNumber=part_number, Body=body)
        except EvidenceStoreError:
            # The response may be lost after COS stored the part. Reconcile by
            # listing it and comparing both size and the COS part MD5 ETag.
            reconciled = await self.list_parts(key=key, upload_id=upload_id)
            present = next((p for p in reconciled if p['part_number'] == part_number), None)
            if present is None:
                raise
            matches = matching_part(reconciled)
            return {'conflict': not matches, 'uploaded': bool(matches),
                    'part_number': part_number, 'size': content_length}
        etag = result.get('ETag') if isinstance(result, dict) else None
        if not isinstance(etag, str) or not etag:
            reconciled = await self.list_parts(key=key, upload_id=upload_id)
            present = next((p for p in reconciled if p['part_number'] == part_number), None)
            if present is None:
                raise EvidenceStoreError('COS multipart part was not confirmed')
            matches = matching_part(reconciled)
            return {'conflict': not matches, 'uploaded': bool(matches),
                    'part_number': part_number, 'size': content_length}
        return {'uploaded': True, 'part_number': part_number, 'size': content_length}

    async def complete_upload(self, *, key, upload_id, size, metadata, sha256=None, part_bytes=None, **_):
        """Reconcile uncertain complete via real HEAD/GET; never recreate/overwrite."""
        self._key(key)
        if isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= self.config.evidence_max_file_bytes:
            raise ValueError('invalid object size')
        # A prior complete may have succeeded while its response was lost.
        # Verify the final object before touching multipart state again.
        try:
            return await self.verify_object(key=key, size=size, metadata=metadata, sha256=sha256)
        except EvidenceStoreError:
            pass
        try:
            parts = await self.list_parts(key=key, upload_id=upload_id)
        except EvidenceStoreError:
            # The object probe above failed and this exact UploadId is no
            # longer observable; surface a recoverable storage error.
            return await self.verify_object(key=key, size=size, metadata=metadata, sha256=sha256)
        chunk = part_bytes or self.config.evidence_part_bytes
        expected_count = (size + chunk - 1) // chunk
        if (len(parts) != expected_count or
                [p['part_number'] for p in parts] != list(range(1, expected_count + 1)) or
                any(p['size'] != min(chunk, size - i * chunk) for i, p in enumerate(parts))):
            raise EvidenceStoreError('multipart upload is incomplete')
        try:
            await self._io(self._sdk().complete_multipart_upload,
                Bucket=self.config.cos_bucket, Key=key, UploadId=upload_id,
                MultipartUpload={'Part': [{'PartNumber': p['part_number'], 'ETag': p['etag']} for p in parts]},
                Metadata={'x-cos-forbid-overwrite': 'true'})
        except EvidenceStoreError:
            pass
        return await self.verify_object(key=key, size=size, metadata=metadata, sha256=sha256)

    async def verify_object(self, *, key, size, metadata, sha256=None):
        return await self._io(self._verify, key=key, size=size, metadata=metadata, sha256=sha256)

    async def put_native_file(self, *, key, metadata, size, sha256, body):
        """Create a deterministic native-attachment object, then read-back verify it.

        A retry first observes the exact key. COS's no-overwrite header prevents a
        duplicate comment retry from replacing an existing audit object.
        """
        key = self._key(key)
        self._metadata(metadata)
        if (isinstance(size, bool) or not isinstance(size, int)
                or not 1 <= size <= min(self.config.evidence_max_file_bytes, 100 * 1024 * 1024)
                or not isinstance(sha256, str) or not re.fullmatch(r'[0-9a-f]{64}', sha256)):
            raise ValueError('invalid native attachment identity')
        try:
            return await self.verify_object(key=key, size=size, metadata=metadata, sha256=sha256)
        except EvidenceStoreError:
            pass
        body.seek(0)
        try:
            await self._io(self._sdk().put_object, Bucket=self.config.cos_bucket,
                Key=key, Body=body, ContentLength=str(size), ContentType='application/octet-stream',
                Metadata={**self._metadata(metadata), 'x-cos-forbid-overwrite': 'true'})
        except EvidenceStoreError:
            # The PUT response may be lost after COS committed the object.
            # Only an exact read-back is considered success.
            return await self.verify_object(key=key, size=size, metadata=metadata, sha256=sha256)
        return await self.verify_object(key=key, size=size, metadata=metadata, sha256=sha256)

    def _verify(self, *, key, size, metadata, sha256=None, gzip_required=False, log_identity=None):
        cap = 10 * 1024 * 1024 if gzip_required else self.config.evidence_max_file_bytes
        if isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= cap:
            raise ValueError('invalid verification size')
        if sha256 is not None and (not isinstance(sha256, str) or not re.fullmatch('[0-9a-f]{64}', sha256)):
            raise ValueError('invalid verification digest')
        client = self._sdk()
        args = {'Bucket': self.config.cos_bucket, 'Key': self._key(key)}
        head = client.head_object(**args)
        lowered = {k.lower(): str(v) for k, v in head.items()}
        if int(lowered.get('content-length', '-1')) != size:
            raise EvidenceStoreError('object size mismatch')
        expected = self._metadata(metadata) if not gzip_required else metadata
        if any(lowered.get(k.lower()) != str(v) for k, v in expected.items()):
            raise EvidenceStoreError('object identity mismatch')
        etag = head.get('ETag', head.get('etag'))
        if etag:
            args['IfMatch'] = etag
        response = client.get_object(**args)
        stream = response['Body'].get_raw_stream()
        digest, count = hashlib.sha256(), 0
        decoder = zlib.decompressobj(16 + zlib.MAX_WBITS) if gzip_required else None
        log_chunks = [] if gzip_required else None
        expanded = 0
        started = time.monotonic()
        try:
            while True:
                data = stream.read(1024 * 1024)
                if not data:
                    break
                count += len(data)
                if count > size or time.monotonic() - started > 600:
                    raise EvidenceStoreError('object exceeds verification bounds')
                digest.update(data)
                if log_chunks is not None:
                    log_chunks.append(data)
                if decoder:
                    pending = data
                    while pending:
                        expanded += len(decoder.decompress(pending, 1024 * 1024))
                        if expanded > 10 * 1024 * 1024:
                            raise EvidenceStoreError('log decompression exceeds bounds')
                        pending = decoder.unconsumed_tail
                        if time.monotonic() - started > 600:
                            raise EvidenceStoreError('gzip verification timeout')
            if decoder and (not decoder.eof or decoder.unused_data):
                raise EvidenceStoreError('invalid gzip object')
        finally:
            stream.close()
        actual = digest.hexdigest()
        if count != size or (sha256 is not None and actual != sha256):
            raise EvidenceStoreError('object digest mismatch')
        if log_identity:
            from .cos_client import CosClient
            error = CosClient._validate_evidence_identity(key, b''.join(log_chunks),
                expected_sha256=sha256, expected_size=size, **log_identity)
            if error is not None:
                raise EvidenceStoreError('archived log identity mismatch')
        return {'status': 'ready', 'verified': True, 'size': count, 'sha256': actual, 'object_key': key}

    async def verify_log(self, *, repo, pr_number, head_sha, key, size, sha256):
        # Reuse the established immutable log layout and its independent 10 MiB cap.
        from .cos_client import CosClient, EVIDENCE_MAX_ARCHIVE_BYTES
        CosClient.validate_object_key(repo, pr_number, head_sha, key)
        if (isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= EVIDENCE_MAX_ARCHIVE_BYTES
                or not isinstance(sha256, str) or not re.fullmatch('[0-9a-f]{64}', sha256)):
            raise ValueError('invalid archived log metadata')
        return await self._io(self._verify, key=key, size=size, sha256=sha256,
            metadata={}, gzip_required=True,
            log_identity={'repo': repo, 'pr_number': pr_number, 'head_sha': head_sha})

    def _operation_key(self, operation):
        expected = build_attachment_key(operation['repo'], operation['pr_number'],
            operation['head_sha'], operation['round_id'], operation['attachment_id'],
            operation['created_at'], operation['filename'])
        key = operation.get('object_key', expected)
        if key != expected:
            raise EvidenceStoreError('persisted object identity mismatch')
        return expected

    def _identity_metadata(self, operation):
        return {name: operation[name] for name in
                ('repo', 'pr_number', 'head_sha', 'round_id', 'attachment_id')}

    async def _find_upload(self, key, *, allow_missing=False):
        """Find a lost create response without issuing a second create."""
        def find():
            key_marker, upload_marker, found = '', '', []
            for _ in range(20):
                response = self._sdk().list_multipart_uploads(
                    Bucket=self.config.cos_bucket, Prefix=key, MaxUploads=1000,
                    KeyMarker=key_marker, UploadIdMarker=upload_marker)
                uploads = response.get('Upload', [])
                if isinstance(uploads, dict):
                    uploads = [uploads]
                found.extend(u['UploadId'] for u in uploads if u.get('Key') == key)
                if str(response.get('IsTruncated', 'false')).lower() != 'true':
                    if len(found) == 1:
                        return found[0]
                    if not found and allow_missing:
                        return None
                    raise EvidenceStoreError('upload initialization unresolved')
                marker = (response.get('NextKeyMarker', ''), response.get('NextUploadIdMarker', ''))
                if marker == (key_marker, upload_marker):
                    break
                key_marker, upload_marker = marker
            raise EvidenceStoreError('upload initialization unresolved')
        return await self._io(find)

    async def dispatch(self, action, *, operation, payload):
        """A's store adapter: operation is loaded exclusively from trusted GitHub."""
        if action == 'log_download':
            key = operation['object_key']
            await self.verify_log(repo=operation['repo'], pr_number=operation['pr_number'],
                head_sha=operation['head_sha'], key=key, size=operation['size'], sha256=operation['sha256'])
            return {'url': await self._io(self.issue_download_url, key=key, filename=key.rsplit('/', 1)[-1])}
        key = self._operation_key(operation)
        metadata = self._identity_metadata(operation)
        size = operation['size']
        sha = operation.get('sha256') or None
        if action == 'status':
            # This path only observes durable facts. It never creates an
            # UploadId, completes a multipart upload, or writes an object.
            upload_id = operation.get('upload_id')
            if upload_id:
                parts = await self.list_parts(key=key, upload_id=upload_id)
                return {'status': operation.get('status', 'uploading'),
                        'uploaded_parts': parts, 'size': size,
                        'filename': operation['filename'], 'head_sha': operation['head_sha'],
                        'round_id': operation['round_id']}
            # Lost CreateMultipartUpload responses are resolved by probing
            # the final object and then listing existing multipart uploads.
            try:
                verified = await self.verify_object(key=key, size=size, metadata=metadata, sha256=sha)
            except EvidenceStoreError:
                verified = None
            if verified:
                return {'status': 'ready', 'uploaded_parts': [], 'size': verified['size'],
                        'filename': operation['filename'], 'head_sha': operation['head_sha'],
                        'round_id': operation['round_id'], 'verified': True,
                        'sha256': verified['sha256'], 'object_key': key}
            upload_id = await self._find_upload(key, allow_missing=True)
            if upload_id is None:
                return {'status': operation.get('status', 'initializing'), 'uploaded_parts': [],
                        'size': size, 'filename': operation['filename'],
                        'head_sha': operation['head_sha'], 'round_id': operation['round_id']}
            parts = await self.list_parts(key=key, upload_id=upload_id)
            return {'status': operation.get('status', 'uploading'),
                    'uploaded_parts': parts, 'size': size,
                    'filename': operation['filename'], 'head_sha': operation['head_sha'],
                    'round_id': operation['round_id']}
        if action == 'download':
            await self.verify_object(key=key, size=size, metadata=metadata, sha256=sha)
            return {'url': await self._io(self.issue_download_url, key=key, filename=operation['filename'])}
        if action == 'preview':
            await self.verify_object(key=key, size=size, metadata=metadata, sha256=sha)
            return {'url': await self._io(self.issue_preview_url,
                key=key, filename=operation['filename'])}
        if action == 'note':
            text = payload.get('text', '')
            if not isinstance(text, str):
                raise ValueError('invalid note')
            body = text.encode('utf-8')
            if not 1 <= len(body) <= self.config.evidence_max_note_bytes:
                raise ValueError('invalid note size')
            actual = hashlib.sha256(body).hexdigest()
            if len(body) != size or (sha and sha != actual):
                raise EvidenceStoreError('note identity mismatch')
            try:
                await self._io(self._sdk().put_object, Bucket=self.config.cos_bucket,
                    Key=key, Body=body, ContentType='application/octet-stream',
                    Metadata={**self._metadata(metadata), 'x-cos-forbid-overwrite': 'true'})
            except EvidenceStoreError:
                pass
            return await self.verify_object(key=key, size=size, metadata=metadata, sha256=actual)
        if action == 'init':
            result = await self.init_upload(key=key, metadata=metadata)
            return {**result, 'object_key': key, 'part_bytes': self.config.evidence_part_bytes}
        if action == 'resume':
            if operation.get('status') in {'completing', 'verifying', 'processing', 'ready'}:
                try:
                    return await self.verify_object(key=key, size=size, metadata=metadata, sha256=sha)
                except EvidenceStoreError:
                    if not operation.get('upload_id'):
                        raise
                    # An uncertain complete may be retried only against the
                    # same UploadId, after this real object probe.
                    return await self.complete_upload(key=key, upload_id=operation['upload_id'],
                        size=size, metadata=metadata, sha256=sha,
                        part_bytes=operation.get('part_bytes', self.config.evidence_part_bytes))
            upload_id = operation.get('upload_id')
            if not upload_id:
                try:
                    return await self.verify_object(key=key, size=size, metadata=metadata, sha256=sha)
                except EvidenceStoreError:
                    upload_id = await self._find_upload(key)
            parts = await self.list_parts(key=key, upload_id=upload_id)
            return {'object_key': key, 'upload_id': upload_id, 'uploaded_parts': parts,
                    'part_bytes': operation.get('part_bytes', self.config.evidence_part_bytes),
                    'attachment_id': operation['attachment_id'], 'head_sha': operation['head_sha'],
                    'round_id': operation['round_id'], 'size': size, 'filename': operation['filename'],
                    'status': operation.get('status', 'uploading')}
        if action == 'sign_part':
            part = payload['part_number']
            chunk = operation.get('part_bytes', self.config.evidence_part_bytes)
            if not 1 <= part <= (size + chunk - 1) // chunk:
                raise ValueError('part outside declared object')
            return await self._io(self.issue_part_url, key=key, upload_id=operation['upload_id'], part_number=part)
        if action == 'put_part':
            return await self.put_part(operation=operation,
                part_number=payload['part_number'], body=payload['body'],
                content_length=payload['content_length'])
        if action == 'complete':
            return await self.complete_upload(key=key, upload_id=operation['upload_id'], size=size,
                metadata=metadata, sha256=sha, part_bytes=operation.get('part_bytes', self.config.evidence_part_bytes))
        raise ValueError('unsupported store action')
