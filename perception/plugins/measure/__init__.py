"""Generic object measurement card.

The package is split the same way as `plugins/navi/`:

    plugin.py    card lifecycle, topics and the publish timer
    geometry.py  bbox + depth -> metres, and the short-window stabilizer
    camera.py    the consumer side of motus.camera/1
"""

from .plugin import MeasurePlugin

__all__ = ["MeasurePlugin"]
