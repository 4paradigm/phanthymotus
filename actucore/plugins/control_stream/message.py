"""Build one motus.control/1 message.

Separate from the card so the wire format can be tested without ROS, and so the
two fields that are easy to get subtly wrong are in one readable place:

  `stamp_ms` is when this command was generated.
  `obs_stamp_ms` is when the observation it was computed from was captured.

They are not the same number and the difference is the whole point. A command
can be freshly generated from an 800 ms old picture — the characteristic
failure of remote inference — and only the second timestamp catches it. Passing
`stamp_ms` for both looks correct in every test that does not involve latency,
and disables the receiver's staleness check in production.

Spec: phanthymotus-driver/README_dev.md § "Continuous Control".
"""

from __future__ import annotations

SCHEMA = "motus.control/1"


def build(*, seq: int, values, mode: str, dof: int, source: str,
          stamp_ms: int, obs_stamp_ms: int, ttl_ms: int, priority: int = 50,
          gripper=None, chunk_index=None, chunk_size=None) -> dict:
    """One command. Every field the receiver checks is required here.

    Nothing is defaulted that the receiver treats as safety-relevant: a missing
    `obs_stamp_ms` or `ttl_ms` would be rejected downstream, and a default
    invented here would only move the mistake somewhere harder to see.
    """
    message = {
        "schema": SCHEMA,
        "seq": int(seq),
        "stamp_ms": int(stamp_ms),
        "obs_stamp_ms": int(obs_stamp_ms),
        "ttl_ms": int(ttl_ms),
        "source": source,
        "priority": int(priority),
        "mode": mode,
        "dof": int(dof),
        "values": [float(v) for v in values],
    }
    if gripper is not None:
        message["gripper"] = float(gripper)
    if chunk_index is not None:
        message["chunk"] = {"index": int(chunk_index),
                            "size": int(chunk_size) if chunk_size is not None else None}
    return message
