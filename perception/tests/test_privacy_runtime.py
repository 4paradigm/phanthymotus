"""Real pixels and JPEGs, using synthetic images only."""

import hashlib

import numpy as np
import pytest
from plugins import privacy_runtime as pr


@pytest.fixture
def image():
    y, x = np.indices((120, 160))
    return np.stack(
        ((x % 2) * 255, (y % 2) * 255, (x * 7 + y * 3) % 256), axis=2
    ).astype(np.uint8)


def entry(box, score=0.9):
    return {"bbox": box, "det_score": score}


@pytest.mark.parametrize("mode", ["opaque", "pixelate", "gaussian"])
@pytest.mark.parametrize(
    "boxes",
    [
        [],
        [[20, 20, 60, 70]],
        [[20, 20, 60, 70], [100, 40, 140, 90]],
        [[-2, -4, 20, 30], [140, 100, 170, 130]],
        [[20, 20, 70, 80], [50, 40, 100, 90]],
        [[30, 30, 30.1, 30.2]],
    ],
)
def test_pixels_and_jpeg(image, mode, boxes):
    cfg = pr.PrivacyConfig().updated({"mode": mode, "bbox_padding": 0})
    before = image.copy()
    regions = pr.validated_regions([entry(b) for b in boxes], 160, 120, 0.5, 0)
    out = pr.redact(image, regions, cfg)
    assert np.array_equal(image, before)
    assert out is not image
    covered = np.zeros(image.shape[:2], bool)
    for x1, y1, x2, y2 in regions:
        covered[y1:y2, x1:x2] = True
    assert np.array_equal(out[~covered], image[~covered])
    if regions:
        assert not np.array_equal(out[covered], image[covered])
        if mode == "opaque":
            assert np.all(out[covered] == 0)
        elif covered.sum() > 100:
            assert out[covered].var() < image[covered].var()
    else:
        assert np.array_equal(out, image)
    encoded = pr.encode_jpeg(out)
    assert pr.decode_jpeg(encoded).shape == image.shape


@pytest.mark.parametrize(
    "box",
    [
        [1, 2, 1, 5],
        [5, 4, 1, 8],
        [0, 0, float("nan"), 8],
        [0, 0, float("inf"), 8],
        [0, 0, 2],
        [True, 0, 2, 3],
        ["0", 0, 5, 6],
        [160, 2, 170, 8],
        [-10, 2, -1, 8],
        [0, 120, 10, 130],
    ],
)
def test_reject_invalid_box(box):
    with pytest.raises(pr.PrivacyError):
        pr.validated_regions([entry([1, 1, 5, 5]), entry(box)], 160, 120, 0.5, 0.25)


@pytest.mark.parametrize(
    "faces",
    [
        None,
        {},
        [None],
        [{"bbox": [0, 0, 4, 4]}],
        [entry([0, 0, 4, 4], float("nan"))],
        [entry([0, 0, 4, 4], 0.4)],
        [entry([0, 0, 4, 4], 1.1)],
        [entry([0, 0, 4, 4], True)],
    ],
)
def test_reject_broken_detection(faces):
    with pytest.raises(pr.PrivacyError):
        pr.validated_regions(faces, 160, 120, 0.5, 0.25)


def test_padding_clipping():
    assert pr.validated_regions([entry([2.2, 3.2, 10.1, 11.1])], 12, 12, 0.5, 0.5) == [
        (0, 0, 12, 12)
    ]


@pytest.mark.parametrize(
    "size", [(1, 1), (1, 1000), (2000, 1), (320, 240), (1920, 1080)]
)
def test_resolutions(size):
    w, h = size
    image = np.full((h, w, 3), 200, np.uint8)
    regions = pr.validated_regions([entry([0, 0, w, h])], w, h, 0.5, 0.25)
    out = pr.redact(image, regions, pr.PrivacyConfig())
    assert not out.any()
    assert pr.decode_jpeg(pr.encode_jpeg(out)).shape == image.shape


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"invalid",
        b"\xff\xd8xx\xff\xd9",
        b"\x89PNG\r\n",
        b"\xff\xd8" + b"x" * (pr.MAX_JPEG_BYTES + 1),
    ],
)
def test_bad_jpeg(data):
    with pytest.raises(pr.PrivacyError):
        pr.decode_jpeg(data)


def test_reject_bomb_before_cv_decode(image, monkeypatch):
    data = pr.encode_jpeg(image)
    monkeypatch.setattr(pr, "MAX_IMAGE_PIXELS", 10)
    with pytest.raises(pr.PrivacyError, match="dimensions"):
        pr.decode_jpeg(data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("mode", "blur"),
        ("detect_fps", 0),
        ("detect_fps", float("nan")),
        ("detect_fps", True),
        ("bbox_padding", -1),
        ("bbox_padding", "0.2"),
        ("confidence_threshold", 0.99),
        ("gaussian_strength", 4),
        ("gaussian_strength", 3.0),
        ("pixelate_strength", 1),
        ("pixelate_strength", True),
        ("max_frame_age_s", 100),
        ("inference_timeout_s", float("inf")),
        ("unknown", 1),
    ],
)
def test_invalid_config(field, value):
    with pytest.raises(pr.PrivacyError):
        pr.PrivacyConfig().updated({field: value})


def test_reply_binding(image):
    data = pr.encode_jpeg(image)
    reply = {
        "ok": True,
        "complete": True,
        "truncated": False,
        "frame_token": "abc",
        "sha256": hashlib.sha256(data).hexdigest(),
        "width": 160,
        "height": 120,
        "faces": [],
    }
    assert pr.validate_reply(reply, data, "abc", image.shape, pr.PrivacyConfig()) == []
    for key, value in (
        ("ok", False),
        ("complete", False),
        ("truncated", True),
        ("frame_token", "old"),
        ("sha256", "other"),
        ("width", 161),
        ("height", 120.0),
        ("faces", None),
    ):
        with pytest.raises(pr.PrivacyError):
            pr.validate_reply(
                {**reply, key: value}, data, "abc", image.shape, pr.PrivacyConfig()
            )


def test_encoder_failure_never_returns_input(image, monkeypatch):
    class Broken:
        IMWRITE_JPEG_QUALITY = 1

        def imencode(self, *a):
            return False, None

    monkeypatch.setattr(pr, "load_cv2", lambda: Broken())
    with pytest.raises(pr.PrivacyError, match="encode"):
        pr.encode_jpeg(image)
