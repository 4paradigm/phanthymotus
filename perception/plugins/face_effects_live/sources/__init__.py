from .capture import FileSource, USBSource, RTSPSource
from .push import PushSource


def create_source(cfg, executor=None):
    kind = cfg.get("kind")
    if kind == "file":
        return FileSource(cfg["path"], cfg.get("paced", True))
    if kind == "usb":
        return USBSource(cfg["device"], cfg.get("width"), cfg.get("height"), cfg.get("fps"))
    if kind == "rtsp":
        return RTSPSource(cfg["url"])
    if kind in ("ros_image", "ros_compressed"):
        from .ros import ROSSource
        return ROSSource(cfg["topic"], kind, executor)
    if kind == "push":
        return PushSource()
    raise ValueError("source.kind must be file/usb/rtsp/ros_image/ros_compressed/push")
