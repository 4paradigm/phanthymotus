"""
peer/naming.py — how a peer is named in a prompt, and how that name resolves back.

The environment snapshot carries names rather than `peer_id`s on purpose: a
32-char hex fingerprint costs more tokens than everything else on the line put
together, and the model has to read it every turn.

Names, though, are chosen by humans and collide — two robots called "Orin6" would
render as two identical lines with nothing to tell them apart, and a delegation
to "Orin6" could only be refused. So the disambiguating suffix is paid for
**only when there is a collision**: unique names stay bare, colliding ones get
`name#<8 hex>` appended — the same prefix length already used to stand in for a
peer with no name, so there is one convention rather than two. If two peers
somehow shared those 8 characters the suffix grows until it separates them,
which a fingerprint makes vanishingly unlikely but costs nothing to handle.

Rendering and resolving live together here because they are one contract. When
they drifted apart the failure would be silent in the worst way: the snapshot
would show a label the tool refuses to accept.
"""

_SUFFIX_LEN = 8


def _base(peer: dict) -> str:
    name = (peer.get('display_name') or '').strip()
    return name or peer['peer_id'][:8]


def labels(peers: list[dict]) -> dict[str, str]:
    """`peer_id → label` for a set of peers, disambiguated only where needed."""
    counts: dict[str, int] = {}
    for p in peers:
        counts[_base(p).lower()] = counts.get(_base(p).lower(), 0) + 1

    out = {}
    for p in peers:
        base = _base(p)
        if counts[base.lower()] == 1:
            out[p['peer_id']] = base
            continue
        clashing = [q for q in peers if _base(q).lower() == base.lower()]
        n = _SUFFIX_LEN
        while n < len(p['peer_id']) and sum(
                1 for q in clashing if q['peer_id'][:n] == p['peer_id'][:n]) > 1:
            n += 4
        out[p['peer_id']] = f'{base}#{p["peer_id"][:n]}'
    return out


def resolve(token: str, peers: list[dict]) -> tuple[dict | None, str]:
    """Find the peer a model meant. Returns `(peer, '')` or `(None, reason)`.

    Accepts, in order: an exact `peer_id`, the rendered label, a bare name, and
    a `peer_id` prefix of at least 4 characters. Ambiguity is reported rather
    than guessed — picking one of two robots for the caller is not a recoverable
    kind of wrong.
    """
    token = (token or '').strip()
    if not token:
        return None, 'no peer given'

    by_id = {p['peer_id']: p for p in peers}
    if token in by_id:
        return by_id[token], ''

    lab = labels(peers)
    for pid, label in lab.items():
        if label.lower() == token.lower():
            return by_id[pid], ''

    named = [p for p in peers if _base(p).lower() == token.lower()]
    if len(named) == 1:
        return named[0], ''
    if len(named) > 1:
        opts = ', '.join(sorted(lab[p['peer_id']] for p in named))
        return None, f'"{token}" matches {len(named)} peers — use one of: {opts}'

    # A shorter prefix than a label carries is still worth accepting: it is
    # unambiguous or it is refused, and typing fewer characters is not an error.
    if len(token) >= 4:
        pref = [p for p in peers if p['peer_id'].startswith(token.lower())]
        if len(pref) == 1:
            return pref[0], ''
        if len(pref) > 1:
            return None, f'peer_id prefix "{token}" is ambiguous'

    known = ', '.join(sorted(lab.values())) or '(none)'
    return None, f'unknown peer "{token}". Paired peers: {known}'
