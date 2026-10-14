"""Robot-independent JPEG QR decoding and bounded content deduplication."""
from __future__ import annotations

class QrDecoder:
    """Decode JPEG frames. Pixel coordinates refer to the original image."""

    def __init__(self):
        import cv2
        self.cv2 = cv2
        self.detector = cv2.QRCodeDetector()
        self.undecoded = []

    def decode(self, jpeg: bytes) -> list[dict]:
        import numpy as np
        self.undecoded = []
        if not jpeg or len(jpeg) > 5_000_000:
            raise ValueError("empty or oversized JPEG")
        image = self.cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), self.cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("invalid JPEG")
        _, texts, points, _ = self.detector.detectAndDecodeMulti(image)
        # Retain undecoded multi-code candidates, including partial successes.
        if points is None or (len(points) == 1 and not any(texts)):
            text, single, _ = self.detector.detectAndDecode(image)
            if points is None or text:
                points = single
                texts = [text] if single is not None else []
        if points is None:
            return []
        candidates = np.asarray(points).reshape(-1, 4, 2)
        codes = []
        retry_budget = 8  # Across all candidates; blank frames do no retries.
        for index, corners in enumerate(candidates):
            if not np.isfinite(corners).all():
                continue
            text = texts[index] if index < len(texts) else ''
            if not text and retry_budget:
                text, used = self._retry_crop(image, corners, retry_budget)
                retry_budget -= used
            record = {
                "corners_px": [[float(x), float(y)] for x, y in corners],
                "image_width": int(image.shape[1]),
                "image_height": int(image.shape[0]),
            }
            if text:
                codes.append({"text": text, **record})
            else:
                self.undecoded.append(record)
        return codes

    def _retry_crop(self, image, corners, budget):
        """Re-detect at bounded scales, retaining coordinates in the source frame."""
        import numpy as np
        low, high = corners.min(axis=0), corners.max(axis=0)
        margin = max(12, int(max(high - low) * 0.12))
        height, width = image.shape[:2]
        x0, y0 = np.maximum(0, np.floor(low - margin)).astype(int)
        x1, y1 = np.minimum([width, height], np.ceil(high + margin)).astype(int)
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            return '', 0
        used = 0
        sizes = set()
        for scale in (0.75, 1.0, 2.0, 1.5):
            if used >= budget:
                break
            # Bound allocation even if the source is a high-resolution frame.
            factor = min(scale, 1600 / max(crop.shape[:2]))
            w, h = max(1, round(crop.shape[1] * factor)), max(1, round(crop.shape[0] * factor))
            if (w, h) in sizes:
                continue
            sizes.add((w, h))
            resized = self.cv2.resize(crop, (w, h), interpolation=self.cv2.INTER_CUBIC)
            used += 1
            text, found, _ = self.detector.detectAndDecode(resized)
            if not text or found is None:
                continue
            # An adjacent QR may enter the padded crop. Only accept a decoded
            # quadrangle overlapping this candidate, not its neighbour.
            mapped = np.asarray(found).reshape(4, 2) / [w / crop.shape[1], h / crop.shape[0]] + [x0, y0]
            rlow, rhigh = mapped.min(axis=0), mapped.max(axis=0)
            intersection = np.maximum(0, np.minimum(high, rhigh) - np.maximum(low, rlow)).prod()
            union = (high - low).prod() + (rhigh - rlow).prod() - intersection
            if union > 0 and intersection / union >= 0.5:
                return text, used
        return '', used


class QrTracker:
    """Emit once per content until absent for rearm_s. Bounded memory."""

    def __init__(self, rearm_s: float):
        self.rearm_s = rearm_s
        self.last_seen: dict[str, float] = {}
        self.sequence = 0

    def update(self, codes: list[dict], now: float) -> list[dict]:
        self.last_seen = {k: t for k, t in self.last_seen.items()
                          if now - t < self.rearm_s}
        events = []
        for code in codes:
            text = code["text"]
            if text not in self.last_seen:
                self.sequence += 1
                events.append({"sequence": self.sequence, "text": text})
            self.last_seen[text] = now
        # Detection input is physically bounded by image size; also bound history.
        if len(self.last_seen) > 256:
            self.last_seen = dict(sorted(self.last_seen.items(),
                                         key=lambda item: item[1])[-256:])
        return events


def self_check():
    """Catch OpenCV builds with detection but without QR decoding support."""
    import base64
    import numpy as np
    import cv2
    png = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAASIAAAEiCAAAAAB41YBqAAAQV0lEQVR4Ae3BMW4kWg7AQPL+h+ZmUiL4oY3Gnw1cZfz5mfHnZ8afnxl/fmb8+Znx52fGn58Zf35m/PmZ8ednxp+fGX9+Zvz5mfHnZ8afnxl/fmb8+Znx52fGn58Zf35m/PmZ8ednxp+fGX9+Ztzky2LILU4yYsmXxcm4yZfFkFucZMSSL4uTcZMviyG3OMmIJV8WJ+MmXxZDbnGSEUu+LE7GTb4shtziJCOWfFmcjJt8WQy5xUlGLPmyOBk3+bIYcouTjFjyZXEybvJlMeQWJxmx5MviZNzky2LILU4yYsmXxcm4yZfFkFucZMSSL4uTcZMviyG3OMmIJV8WJ+MmI35LVpxkxUlGLFnxWzLiZNxkxG/JipOsOMmIJSt+S0acjJuM+C1ZcZIVJxmxZMVvyYiTcZMRvyUrTrLiJCOWrPgtGXEybjLit2TFSVacZMSSFb8lI07GTUb8lqw4yYqTjFiy4rdkxMm4yYjfkhUnWXGSEUtW/JaMOBk3GfFbsuIkK04yYsmK35IRJ+MmI35LVpxkxUlGLFnxWzLiZNxkxG/JipOsOMmIJSt+S0acjJuM+C1ZcZIVJxmxZMVvyYiTcZMRvyUrTrLiJCOWrPgtGXEybjJiyVMMWTFkxZIRS0YsWTHkKZaMOBk3GbHkKYasGLJiyYglI5asGPIUS0acjJuMWPIUQ1YMWbFkxJIRS1YMeYolI07GTUYseYohK4asWDJiyYglK4Y8xZIRJ+MmI5Y8xZAVQ1YsGbFkxJIVQ55iyYiTcZMRS55iyIohK5aMWDJiyYohT7FkxMm4yYglTzFkxZAVS0YsGbFkxZCnWDLiZNxkxJKnGLJiyIolI5aMWLJiyFMsGXEybjJiyVMMWTFkxZIRS0YsWTHkKZaMOBk3GbHkKYasGLJiyYglI5asGPIUS0acjJuMWPIUQ1YMWbFkxJIRS1YMeYolI07GTUYseYohK4asOMmKk6wY8hRLRpyMm4xY8hRDVgxZcZIVJ1kx5CmWjDgZNxmx5CmGrBiy4iQrTrJiyFMsGXEybjJiyVMMWTFkxUlWnGTFkKdYMuJk3GTEkqcYsmLIipOsOMmKIU+xZMTJuMmIJU8xZMWQFSdZcZIVQ55iyYiTcZMRS55iyIohK06y4iQrhjzFkhEn4yYjljzFkBVDVpxkxUlWDHmKJSNOxk1GLHmKISuGrDjJipOsGPIUS0acjJuMWPIUQ1YMWXGSFSdZMeQplow4GTcZseQphqwYsuIkK06yYshTLBlxMm4yYslTDFkxZMVJVpxkxZCnWDLiZNxkxJKnGLJiyFM8yYohT7FkxMm4yYglTzFkxZCneJIVQ55iyYiTcZMRS55iyIohT/EkK4Y8xZIRJ+MmI5Y8xZAVQ57iSVYMeYolI07GTUYseYohK4Y8xZOsGPIUS0acjJuMWPIUQ1YMeYonWTHkKZaMOBk3GbHkKYasGPIUT7JiyFMsGXEybjJiyVMMWTHkKZ5kxZCnWDLiZNxkxJKnGLJiyFM8yYohT7FkxMm4yYglTzFkxZCneJIVQ55iyYiTcZMRS55iyIohT/EkK4Y8xZIRJ+MmI5Y8xZAVQ57iSVYMeYolI07GTUb8lqz4MlnxWzLiZNxkxG/Jii+TFb8lI07GTUb8lqz4MlnxWzLiZNxkxG/Jii+TFb8lI07GTUb8lqz4MlnxWzLiZNxkxG/Jii+TFb8lI07GTUb8lqz4MlnxWzLiZNxkxG/Jii+TFb8lI07GTUb8lqz4MlnxWzLiZNxkxG/Jii+TFb8lI07GTUb8lqz4MlnxWzLiZNzky2LIiiUjloxY8mVxMm7yZTFkxZIRS0Ys+bI4GTf5shiyYsmIJSOWfFmcjJt8WQxZsWTEkhFLvixOxk2+LIasWDJiyYglXxYn4yZfFkNWLBmxZMSSL4uTcZMviyErloxYMmLJl8XJuMmXxZAVS0YsGbHky+Jk3OTLYsiKJSOWjFjyZXEybvJlMWTFkhFLRiz5sjgZN/myGLJiyYglI5Z8WZyMm3xZDFmxZMSSEUu+LE7GvyWn+L9h/Ftyiv8bxr8lp/i/Yfxbcor/G8a/Jaf4v2H8W3KK/xvGvyWn+L9h/Ftyiv8bxr8lp/i/Yfxbcor/G8a/Jaf4v2Hc5BNxkhVDViwZcZIVJ3mKJSNOxk0+ESdZMWTFkhEnWXGSp1gy4mTc5BNxkhVDViwZcZIVJ3mKJSNOxk0+ESdZMWTFkhEnWXGSp1gy4mTc5BNxkhVDViwZcZIVJ3mKJSNOxk0+ESdZMWTFkhEnWXGSp1gy4mTc5BNxkhVDViwZcZIVJ3mKJSNOxk0+ESdZMWTFkhEnWXGSp1gy4mTc5BNxkhVDViwZcZIVJ3mKJSNOxk0+ESdZMWTFkhEnWXGSp1gy4mTc5BNxkhVDViwZcZIVJ3mKJSNOxk0+ESdZMWTFkhEnWXGSp1gy4mTc5BRDbvEitzjJiI/IiCUrXoybnGLILV7kFicZ8REZsWTFi3GTUwy5xYvc4iQjPiIjlqx4MW5yiiG3eJFbnGTER2TEkhUvxk1OMeQWL3KLk4z4iIxYsuLFuMkphtziRW5xkhEfkRFLVrwYNznFkFu8yC1OMuIjMmLJihfjJqcYcosXucVJRnxERixZ8WLc5BRDbvEitzjJiI/IiCUrXoybnGLILV7kFicZ8REZsWTFi3GTUwy5xYvc4iQjPiIjlqx4MW5yiiG3eJFbnGTER2TEkhUvxk1GnOQj8SQjloxYcoolT/Fi3GTEST4STzJiyYglp1jyFC/GTUac5CPxJCOWjFhyiiVP8WLcZMRJPhJPMmLJiCWnWPIUL8ZNRpzkI/EkI5aMWHKKJU/xYtxkxEk+Ek8yYsmIJadY8hQvxk1GnOQj8SQjloxYcoolT/Fi3GTEST4STzJiyYglp1jyFC/GTUac5CPxJCOWjFhyiiVP8WLcZMRJPhJPMmLJiCWnWPIUL8ZNRpzkI/EkI5aMWHKKJU/xYtzkC+IbZMRNTjHkKU7GTb4gvkFG3OQUQ57iZNzkC+IbZMRNTjHkKU7GTb4gvkFG3OQUQ57iZNzkC+IbZMRNTjHkKU7GTb4gvkFG3OQUQ57iZNzkC+IbZMRNTjHkKU7GTb4gvkFG3OQUQ57iZNzkC+IbZMRNTjHkKU7GTb4gvkFG3OQUQ57iZNzkC+IbZMRNTjHkKU7GTb4gvkFG3OQUQ57iZNxkxEk+EidZMWTFJ2TFk4w4GTcZcZKPxElWDFnxCVnxJCNOxk1GnOQjcZIVQ1Z8QlY8yYiTcZMRJ/lInGTFkBWfkBVPMuJk3GTEST4SJ1kxZMUnZMWTjDgZNxlxko/ESVYMWfEJWfEkI07GTUac5CNxkhVDVnxCVjzJiJNxkxEn+UicZMWQFZ+QFU8y4mTcZMRJPhInWTFkxSdkxZOMOBk3GXGSj8RJVgxZ8QlZ8SQjTsZNRpzkI3GSFUNWfEJWPMmIk/Hfk1t8gZziJiNOxn9PbvEFcoqbjDgZ/z25xRfIKW4y4mT89+QWXyCnuMmIk/Hfk1t8gZziJiNOxn9PbvEFcoqbjDgZ/z25xRfIKW4y4mT89+QWXyCnuMmIk/Hfk1t8gZziJiNOxn9PbvEFcoqbjDgZ/z25xRfIKW4y4mT89+QWXyCnuMmIk3GTL4uTnOIktxiyYsiKJSNOxk2+LE5yipPcYsiKISuWjDgZN/myOMkpTnKLISuGrFgy4mTc5MviJKc4yS2GrBiyYsmIk3GTL4uTnOIktxiyYsiKJSNOxk2+LE5yipPcYsiKISuWjDgZN/myOMkpTnKLISuGrFgy4mTc5MviJKc4yS2GrBiyYsmIk3GTL4uTnOIktxiyYsiKJSNOxk2+LE5yipPcYsiKISuWjDgZN/myOMkpTnKLISuGrFgy4mTc5MviJKc4yS2GrBiyYsmIk3GTEb8lK06yYsgpbnKKJxlxMm4y4rdkxUlWDDnFTU7xJCNOxk1G/JasOMmKIae4ySmeZMTJuMmI35IVJ1kx5BQ3OcWTjDgZNxnxW7LiJCuGnOImp3iSESfjJiN+S1acZMWQU9zkFE8y4mTcZMRvyYqTrBhyipuc4klGnIybjPgtWXGSFUNOcZNTPMmIk3GTEb8lK06yYsgpbnKKJxlxMm4y4rdkxUlWDDnFTU7xJCNOxk1G/JasOMmKIae4ySmeZMTJuMmIJU8xZMWTnOIkK07yiTgZNxmx5CmGrHiSU5xkxUk+ESfjJiOWPMWQFU9yipOsOMkn4mTcZMSSpxiy4klOcZIVJ/lEnIybjFjyFENWPMkpTrLiJJ+Ik3GTEUueYsiKJznFSVac5BNxMm4yYslTDFnxJKc4yYqTfCJOxk1GLHmKISue5BQnWXGST8TJuMmIJU8xZMWTnOIkK07yiTgZNxmx5CmGrHiSU5xkxUk+ESfjJiOWPMWQFU9yipOsOMkn4mTcZMSSpxiy4klOcZIVJ/lEnIybjFjyFENWDFnxCVnxW7LixbjJiCVPMWTFkBWfkBW/JStejJuMWPIUQ1YMWfEJWfFbsuLFuMmIJU8xZMWQFZ+QFb8lK16Mm4xY8hRDVgxZ8QlZ8Vuy4sW4yYglTzFkxZAVn5AVvyUrXoybjFjyFENWDFnxCVnxW7LixbjJiCVPMWTFkBWfkBW/JStejJuMWPIUQ1YMWfEJWfFbsuLFuMmIJU8xZMWQFZ+QFb8lK16Mm4xY8hRDVgxZ8QlZ8Vuy4sW4yYglTzFkxZAVS07xTxk3GbHkKYasGLJiySn+KeMmI5Y8xZAVQ1YsOcU/ZdxkxJKnGLJiyIolp/injJuMWPIUQ1YMWbHkFP+UcZMRS55iyIohK5ac4p8ybjJiyVMMWTFkxZJT/FPGTUYseYohK4asWHKKf8q4yYglTzFkxZAVS07xTxk3GbHkKYasGLJiySn+KeMmI5Y8xZAVQ1YsOcU/ZdxkxJKnGLJiyIolp/injJuM+C1ZMWTFkpdYsmLIiiErPmHcZMRvyYohK5a8xJIVQ1YMWfEJ4yYjfktWDFmx5CWWrBiyYsiKTxg3GfFbsmLIiiUvsWTFkBVDVnzCuMmI35IVQ1YseYklK4asGLLiE8ZNRvyWrBiyYslLLFkxZMWQFZ8wbjLit2TFkBVLXmLJiiErhqz4hHGTEb8lK4asWPISS1YMWTFkxSeMm4z4LVkxZMWSl1iyYsiKISs+YdxkxG/JiiErlrzEkhVDVgxZ8QnjJiN+S1YMWbHkJZasGLJiyIpPGDcZ8VuyYsiKJS+xZMWQFUNWfMK4yZfFkFsMeYolL7FkxYtxky+LIbcY8hRLXmLJihfjJl8WQ24x5CmWvMSSFS/GTb4shtxiyFMseYklK16Mm3xZDLnFkKdY8hJLVrwYN/myGHKLIU+x5CWWrHgxbvJlMeQWQ55iyUssWfFi3OTLYsgthjzFkpdYsuLFuMmXxZBbDHmKJS+xZMWLcZMviyG3GPIUS15iyYoX4yZfFkNuMeQplrzEkhUvxp+fGX9+Zvz5mfHnZ8afnxl/fmb8+Znx52fGn58Zf35m/PmZ8ednxp+fGX9+Zvz5mfHnZ8afnxl/fmb8+Znx52fGn58Zf35m/PnZ/wD1Ym9bnqEm5gAAAABJRU5ErkJggg==')
    image = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
    ok, jpeg = cv2.imencode('.jpg', image)
    assert ok, 'OpenCV JPEG encoding unavailable'
    result = QrDecoder().decode(jpeg.tobytes())
    assert [c['text'] for c in result] == ['QR-SELF-CHECK'], 'OpenCV QR decoding unavailable'
