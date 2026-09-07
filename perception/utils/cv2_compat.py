"""
utils/cv2_compat.py — Import cv2 on Jetson, where the system build is broken.

The L4T images ship OpenCV as an apt package whose Python wrapper hits a
circular import in `cv2/mat_wrapper` under some interpreter/numpy
combinations: `import cv2` raises, or succeeds with the module half-built so
that `cv2.IMREAD_COLOR` is missing. Loading the extension `.so` directly
sidesteps the wrapper package entirely.

`plugins/vop.py` (`_ensure_model`, around line 273) carries this workaround
inline and is the origin of this code; it is left as-is deliberately — it is
working in production and the copy here exists so new plugins do not each
reinvent it.

`imshow` and friends are stubbed for the same reason vop stubs them: the
container is headless, and some OpenCV code paths call them unconditionally.
"""

from __future__ import annotations

import glob
import importlib.util
import logging
import sys

log = logging.getLogger(__name__)


def load_cv2():
    """Return a usable `cv2` module, or raise ImportError."""
    module = None
    try:
        import cv2 as module  # noqa: PLC0415
        _ = module.IMREAD_COLOR      # half-built module raises AttributeError
    except (ImportError, AttributeError):
        module = _load_from_extension()

    for name in ("imshow", "waitKey", "destroyAllWindows"):
        if not hasattr(module, name):
            setattr(module, name, _headless_stub(name))
    return module


def _load_from_extension():
    candidates = sorted(
        glob.glob("/usr/lib/python*/dist-packages/cv2/python-*/cv2.cpython-*.so")
    )
    if not candidates:
        import cv2  # noqa: PLC0415 - let the original error surface
        return cv2
    path = candidates[0]
    log.warning("[cv2_compat] system cv2 package is broken; loading %s directly",
                path)
    spec = importlib.util.spec_from_file_location("cv2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["cv2"] = module
    return module


def _headless_stub(name: str):
    if name == "waitKey":
        return lambda *args, **kwargs: 0
    return lambda *args, **kwargs: None


__all__ = ["load_cv2"]
