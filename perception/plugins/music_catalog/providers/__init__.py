"""Discover providers without constructing them or performing network I/O."""
import importlib
import pkgutil


def discover():
    found = {}
    for module in pkgutil.iter_modules(__path__):
        if not module.name.startswith('_'):
            factory = getattr(importlib.import_module(f'{__name__}.{module.name}'), 'PROVIDER', None)
            if factory and all(callable(getattr(factory, name, None))
                               for name in ('capabilities', 'search', 'track')):
                found[module.name] = factory
    return found
