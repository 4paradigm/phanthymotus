"""Provider discovery — one file per inference backend, found by scanning.

The card must not know which backends exist. There will be a steady stream of
them (openpi's WebSocket server, LeRobot's policy server and everything behind
it, a vendor's HTTP endpoint, an in-process small model), and a hard-coded enum
means every new one edits the card, its schema and its tests.

So: drop a module in this directory that exposes `PROVIDER`, and it is
available. Same duck typing the cards themselves use — no base class, no
registry decorator.

Two shapes live here, and the asymmetry is deliberate:

- **One file per local model**, named after the model — `smolvla.py`, and
  whatever comes next. Each owns its weights, its download and its load, the
  way `perception/plugins/` is organised. A single `local` provider would have
  become a switch over model families with every quirk piled up behind it.

- **One file for everything remote** — `vla_cloud.py`. Off the robot, the model
  brings none of its own problems here: what arrives is an action chunk, and
  the only thing that varies is the address. Configuring it is
  `{endpoint, key, model}`, the same shape agent-core already uses for LLMs.

A provider implements four methods:

    capabilities() -> dict
        {model, action_dim, chunk_size, control_hz, needs_state, n_cameras,
         image_size, supports_rtc}. Read once at start and reconciled against
        the downstream driver's action space before anything moves.

    infer(observation, inference_delay=0) -> list[list[float]]
        An action chunk, shape (T, action_dim), already in engineering units.
        A single-step model returns T=1.

    health() -> bool
    close() -> None

`capabilities()` is not optional and not advisory. It is the only thing that
can catch "the model emits 32 values and this arm takes 14" before the first
command rather than at 30 Hz afterwards.
"""

from __future__ import annotations

import importlib
import pkgutil

# Methods every provider must have. Checked at discovery rather than at first
# use: a provider that turns out to be missing `close` halfway through a run
# leaks whatever it holds, and the operator finds out from a card that will not
# restart.
REQUIRED = ("capabilities", "infer", "health", "close")


def discover() -> dict:
    """`{name: provider factory}` for every importable module in this package.

    A module that fails to import is skipped with its error, not raised: one
    backend whose optional dependency is absent must not take the card down,
    and `mock` in particular has to stay available on a machine with no GPU,
    no network and no torch.
    """
    found, errors = {}, {}
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_"):
            continue
        try:
            module = importlib.import_module(f"{__name__}.{info.name}")
        except Exception as error:      # noqa: BLE001 — see docstring
            errors[info.name] = f"{type(error).__name__}: {error}"
            continue
        factory = getattr(module, "PROVIDER", None)
        if factory is None:
            errors[info.name] = "module has no PROVIDER"
            continue
        missing = [m for m in REQUIRED if not callable(getattr(factory, m, None))
                   and not hasattr(factory, m)]
        if missing:
            errors[info.name] = f"missing {', '.join(missing)}"
            continue
        found[info.name] = factory
    discover.errors = errors
    return found


discover.errors = {}


def names() -> list:
    """Provider names, for a schema enum built at runtime rather than written down."""
    return sorted(discover())
