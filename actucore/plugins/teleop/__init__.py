"""ActuCore PICO/Tianyi card; lazy import keeps protocol tests independent of ROS."""
def __getattr__(name):
    if name == 'TeleopPlugin':
        from .plugin import TeleopPlugin
        return TeleopPlugin
    raise AttributeError(name)
