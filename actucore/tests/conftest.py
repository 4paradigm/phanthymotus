"""Stubs for the ROS pieces the cards touch, so the suite runs on a laptop.

These cards are deliberately ROS-free in their logic and only reach for rclpy
at the edges — creating a publisher, choosing a QoS profile. Requiring a ROS
install to test a dictionary lookup or a reliability constant would mean the
laptop suite could not cover the one bug that has now been found twice in this
directory: a RELIABLE subscriber silently receiving nothing from a BEST_EFFORT
publisher. `test_observations_are_subscribed_best_effort` and
`test_subscriptions_use_best_effort_qos` both assert against these constants.

Installed via conftest so every module in the suite sees them, regardless of
collection order — putting them in one test file made the others depend on
being collected second, which is not something a test file should rely on.
"""
import os
import sys
import types

# Explicit DDS integration runs must use the installed ROS implementation.
# Import before setdefault so these laptop stubs cannot shadow real packages.
# Missing ROS dependencies intentionally fail an explicitly requested run.
if os.environ.get("RUN_MOTION_CONTROL_ROS") == "1":
    import rclpy
    import rclpy.qos
    import sensor_msgs.msg
    import std_msgs.msg

_STUBS = {
    "sensor_msgs.msg": ("CompressedImage", "Image"),
    "std_msgs.msg": ("String",),
}

for _path, _names in _STUBS.items():
    _pkg = _path.split(".")[0]
    sys.modules.setdefault(_pkg, types.ModuleType(_pkg))
    _mod = sys.modules.setdefault(_path, types.ModuleType(_path))
    for _name in _names:
        if not hasattr(_mod, _name):
            setattr(_mod, _name, type(_name, (), {}))
    setattr(sys.modules[_pkg], "msg", _mod)

_rclpy = sys.modules.setdefault("rclpy", types.ModuleType("rclpy"))
_qos = sys.modules.setdefault("rclpy.qos", types.ModuleType("rclpy.qos"))
for _enum, _members in (
    ("ReliabilityPolicy", {"BEST_EFFORT": "best_effort", "RELIABLE": "reliable"}),
    ("HistoryPolicy", {"KEEP_LAST": "keep_last", "KEEP_ALL": "keep_all"}),
    ("DurabilityPolicy", {"VOLATILE": "volatile", "TRANSIENT_LOCAL": "transient_local"}),
):
    if not hasattr(_qos, _enum):
        setattr(_qos, _enum, type(_enum, (), dict(_members)))
if not hasattr(_qos, "QoSProfile"):
    class _QoSProfile:
        """Readable as attributes or as items.

        The real QoSProfile exposes attributes; some tests here reach for
        `profile["reliability"]` because they are checking a dict the card
        built. Supporting both keeps the stub out of the way of whichever
        style a test finds clearer.
        """

        def __init__(self, **kw):
            self.__dict__.update(kw)

        def __getitem__(self, key):
            return self.__dict__[key]

        def __repr__(self):
            return f"QoSProfile({self.__dict__})"

    setattr(_qos, "QoSProfile", _QoSProfile)
setattr(_rclpy, "qos", _qos)
