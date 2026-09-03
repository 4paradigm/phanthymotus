"""Regression tests for the policy-route toggle in api/network.py.

The trap these guard, observed on Bumi 2026-09-02: the toggle was on for wifi and
the robot had no internet at all. The old implementation set `ipv4.route-table`,
which does not copy a connection's routes into the private table — it *relocates*
them, DHCP default route included. So the wifi stopped being usable as an egress
path, every outbound connection fell to a dead `10.42.0.1` default route on another
NIC, and NetworkManager reported the wifi link to systemd-resolved as
not-a-default-route, which dropped its DNS servers too (`Current Scopes: none`) and
left resolution riding the same dead link. It could not even be patched by hand:
with the subnet route relocated as well, the main table had no route to the wifi
gateway, so `ip route add default via 10.100.128.1 dev wlx…` was rejected with
"Nexthop has invalid gateway".

The fix copies the routes into the private table via the per-route `table`
attribute and leaves `route-table` unset, so the main table stays whatever DHCP
says. Verified on Bumi (NM 1.36.6): main regained the wifi default route,
`resolvectl` went back to `+DefaultRoute` with the wifi's own DNS servers,
`ip route get 8.8.8.8 from 10.100.129.141` still resolved via table 205, and an
SSH session over the very device being reconfigured survived Device.Reapply.

Second trap, same host later that day: the dead dongle was replaced. Interface names are
MAC-derived, so `wlx001f058a5d81` became `wlx001f058a4a58` and the table number derived
from it went 205 -> 218 — the rule and routes written for 205 were stranded in the
profile, and cleanup keyed on "the current table" could never reap them. The number is
now allocated (reuse what the connection already has, else the preferred number, else
the lowest free one) and cleanup covers the whole 200-249 range. That also removes the
collision: crc32 % 50 can put two NICs on one host in the same table, where their rules
silently merge.

Run: cd agent-core && python3 -m pytest tests/test_policy_route_settings.py
"""
import os
import pathlib
import sys
import tempfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))


# ── dbus stub ─────────────────────────────────────────────────────────────────
# network.py imports dbus inside each function, so injecting a stub module is
# enough — no dbus-python needed to test the settings we hand to NM.

class _Dict(dict):
    def __init__(self, value=None, signature=None):
        super().__init__(value or {})
        self.signature = signature


class _Array(list):
    def __init__(self, value=None, signature=None):
        super().__init__(value or [])
        self.signature = signature


class _FakeDbus:
    Dictionary = _Dict
    Array = _Array
    String = str
    UInt32 = int
    UInt64 = int
    Int32 = int
    Byte = int
    Boolean = bool
    ObjectPath = str

    class Interface:  # pragma: no cover - unused by these tests
        def __init__(self, obj, iface):
            pass


sys.modules.setdefault('dbus', _FakeDbus())

from api.network import (  # noqa: E402
    _POLICY_TABLE_MAX,
    _POLICY_TABLE_MIN,
    _allocate_policy_table,
    _policy_route_data,
    _policy_route_table_for,
    _tables_of,
    _without_policy_routes,
)

DEVICE = 'wlx001f058a5d81'
NETWORK = '10.100.128.0'
PREFIX = 19
GATEWAY = '10.100.128.1'


def _table():
    return _policy_route_table_for(DEVICE)


def test_table_number_is_stable_and_in_private_range():
    """The Bumi profile in the field carries table 205; a different preferred number
    here would hand every existing single-NIC host a pointless renumber."""
    table = _policy_route_table_for(DEVICE)
    assert table == 205
    assert _POLICY_TABLE_MIN <= table <= _POLICY_TABLE_MAX
    assert _policy_route_table_for(DEVICE) == table  # deterministic across calls


def test_user_static_routes_survive():
    user_route = {'dest': '10.9.0.0', 'prefix': 16, 'next-hop': '10.100.128.9'}
    existing = [user_route, *_policy_route_data(_table(), NETWORK, PREFIX, GATEWAY)]

    assert _without_policy_routes(existing) == [user_route]


def test_cleanup_also_reaps_a_previous_table_number():
    """The card-swap residue: Bumi 2026-09-02 went from table 205 to 218 because the
    replacement dongle had a different MAC. Stripping only the current number left
    the 205 pair in the profile forever."""
    stale = _policy_route_data(205, NETWORK, PREFIX, GATEWAY)
    fresh = _policy_route_data(218, NETWORK, PREFIX, GATEWAY)

    assert _without_policy_routes(stale + fresh) == []


def test_reuses_the_number_this_connection_already_has():
    """A card swap must not renumber: the rule and routes already written for 205 keep
    working, and nothing is stranded."""
    own = {'routing-rules': [{'table': 205}],
           'route-data': _policy_route_data(205, NETWORK, PREFIX, GATEWAY)}

    # preferred is what the *new* device name hashes to; it must lose to the existing one
    assert _allocate_policy_table(own, [], 218) == 205


def test_avoids_a_number_another_connection_uses():
    """50 slots and a crc32 collide. Two connections sharing one table would silently
    merge their rules and fight over it."""
    other = {'routing-rules': [{'table': 205}]}

    table = _allocate_policy_table({}, [other], 205)

    assert table != 205
    assert _POLICY_TABLE_MIN <= table <= _POLICY_TABLE_MAX


def test_gives_up_the_shared_number_rather_than_keeping_a_collision():
    """If our own number is already claimed by someone else, reuse is not safe."""
    own = {'routing-rules': [{'table': 205}]}
    other = {'route-data': [{'dest': '0.0.0.0', 'prefix': 0, 'table': 205}]}

    assert _allocate_policy_table(own, [other], 205) != 205


def test_preferred_number_is_taken_when_free():
    assert _allocate_policy_table({}, [{'routing-rules': [{'table': 207}]}], 205) == 205


def test_exhausted_range_raises_instead_of_reusing():
    others = [{'routing-rules': [{'table': t}]}
              for t in range(_POLICY_TABLE_MIN, _POLICY_TABLE_MAX + 1)]

    with pytest.raises(RuntimeError):
        _allocate_policy_table({}, others, 205)


def test_only_private_range_tables_count_as_ours():
    """Table 254 is main and 1000 might be someone else's scheme. Claiming either would
    make us delete routes we do not own, or hand out a number already in real use."""
    assert _tables_of({'route-data': [{'table': 254}, {'table': 1000}, {'table': 0}]}) == set()
    assert _tables_of({'route-data': [{'table': 205}]}) == {205}


def test_disable_then_enable_does_not_accumulate_copies():
    """The canvas-style repeat: toggling several times must not stack duplicates."""
    routes = []
    for _ in range(3):
        routes = (_without_policy_routes(routes)
                  + _policy_route_data(_table(), NETWORK, PREFIX, GATEWAY))
        assert len(routes) == 2
        routes = _without_policy_routes(routes)  # disable
        assert routes == []


def test_untabled_route_is_never_mistaken_for_ours():
    """A missing attribute means main, or a user's plain static route would be deleted
    on the first toggle."""
    plain = [{'dest': '10.9.0.0', 'prefix': 16}]
    assert _without_policy_routes(plain) == plain


def test_copies_are_pinned_to_the_private_table():
    routes = _policy_route_data(_table(), NETWORK, PREFIX, GATEWAY)
    assert [r['table'] for r in routes] == [_table(), _table()]


def test_subnet_copy_stays_on_link():
    """A table holding a default route never falls through to main, so without an
    on-link subnet route replies to a same-subnet peer detour via the gateway."""
    subnet = _policy_route_data(_table(), NETWORK, PREFIX, GATEWAY)[0]
    assert subnet['dest'] == NETWORK
    assert subnet['prefix'] == PREFIX
    assert 'next-hop' not in subnet


def test_default_copy_goes_via_the_gateway():
    default = _policy_route_data(_table(), NETWORK, PREFIX, GATEWAY)[1]
    assert default['dest'] == '0.0.0.0'
    assert default['prefix'] == 0
    assert default['next-hop'] == GATEWAY


def test_no_gateway_yields_subnet_copy_only():
    """A device with an address but no gateway can still answer its own subnet.
    Emitting a default route with an empty next-hop would be rejected by NM."""
    routes = _policy_route_data(_table(), NETWORK, PREFIX, '')
    assert len(routes) == 1
    assert routes[0]['dest'] == NETWORK


def test_nothing_lands_in_the_main_table():
    """The whole point: not one emitted route may be left for table main. An entry
    without a `table` attribute is exactly what breaks the host's connectivity."""
    for route in _policy_route_data(_table(), NETWORK, PREFIX, GATEWAY):
        assert route.get('table') == _table()


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
