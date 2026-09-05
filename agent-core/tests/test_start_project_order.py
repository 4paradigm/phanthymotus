"""Regression: a card must not start before the cards that feed it.

Observed on R1 2026-09-06. Speaking to the robot did nothing; typing the same
instruction through remote_control worked. Perception was fine — the ASR log
shows the wake word matched and `'小范小范，现在几点了？' → '现在几点了？'`
published on `/ubuntu/mic/audio/asr`, and asking the driver `action: info`
returns that exact topic. But the agent loop's own prompt listed one input:

    <subscribed_sensors><source name="dds:/remote_control/message" /></subscribed_sensors>

`_do_start_project` split the cards into sources (no inbound connection) and
processors (some inbound connection) and started the two groups in turn. That is
a topological order only for a graph two levels deep. The R1 canvas is three:
`mic → asr → decision_core`. asr and decision_core both fell in the processor
group, which was ordered by position in the layout — decision_core had been on
the canvas since 09-02, asr was added at 01:27 that morning — so decision_core
started first, found no topic for its ASR input, and subscribed to nothing.

ASR's output topic is derived (`{input_topic}/asr`), so it exists nowhere until
the plugin is running: no fallback could have covered for the order. The
persisted `fromTopic` on the connection was `''` and the card's persisted
`topicOut` was format-only, which is what a multiInstance tool's schema
declares by design.

Nothing failed and nothing was logged. Both halves are fixed here: the order,
and the silence — an inbound connection carrying no topic now fails its card
instead of starting it deaf.

Run: cd agent-core && python3 -m pytest tests/test_start_project_order.py
"""
import os
import pathlib
import sys
import tempfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

from api.config import order_cards_by_dependency  # noqa: E402


def card(cid, tool):
    return {'id': cid, 'toolName': tool, 'mcpId': 'mcp-1'}


def conn(src, dst, topic='', port='0'):
    return {'fromCardId': src, 'toCardId': dst, 'fromPortIdx': port,
            'toPortIdx': '0', 'fromTopic': topic}


def order(cards, connections):
    ordered, cyclic = order_cards_by_dependency(cards, connections)
    assert cyclic == []
    return [c['toolName'] for c in ordered]


# ── the reported layout ──────────────────────────────────────────────────────

# Card order as saved on R1: decision_core was built four days before asr, and
# the layout lists cards in creation order.
R1_CARDS = [
    card('card-mtjr82bgdxo9', 'tts'),
    card('card-mtjr86idgact', 'speaker'),
    card('card-mtju428pljp9', 'decision_core'),
    card('card-mtju5os2wysg', 'switch_mode'),
    card('card-mtju62eb0v8z', 'remote_message'),
    card('card-mtju68d9au22', 'loco'),
    card('card-mtlb92gin0yq', 'joints'),
    card('card-mtlbv2htotvc', 'mic'),
    card('card-mtonohvr64vl', 'asr'),
]
R1_CONNS = [
    conn('card-mtjr82bgdxo9', 'card-mtjr86idgact', '/perception/tts'),
    conn('card-mtlbv2htotvc', 'card-mtonohvr64vl', '/ubuntu/mic/audio'),
    conn('card-mtonohvr64vl', 'card-mtju428pljp9', ''),   # the empty one
    conn('card-mtju62eb0v8z', 'card-mtju428pljp9', '/remote_control/message'),
]


def test_asr_starts_before_the_agent_loop_that_reads_it():
    got = order(R1_CARDS, R1_CONNS)
    assert got.index('mic') < got.index('asr') < got.index('decision_core')


def test_the_old_split_is_what_this_replaces():
    """Guard the premise: the previous grouping really did invert the pair."""
    with_inbound = {c['toCardId'] for c in R1_CONNS}
    old = ([c['toolName'] for c in R1_CARDS if c['id'] not in with_inbound]
           + [c['toolName'] for c in R1_CARDS if c['id'] in with_inbound])
    assert old.index('decision_core') < old.index('asr')


def test_every_card_still_starts_exactly_once():
    got = order(R1_CARDS, R1_CONNS)
    assert sorted(got) == sorted(c['toolName'] for c in R1_CARDS)


# ── order properties ────────────────────────────────────────────────────────

def test_a_two_level_graph_keeps_the_order_it_had():
    """The common case must not change: sources in layout order, then the rest."""
    cards = [card('a', 'mic'), card('b', 'speaker'), card('c', 'tts')]
    conns = [conn('a', 'b', '/mic'), conn('c', 'b', '/tts')]
    assert order(cards, conns) == ['mic', 'tts', 'speaker']


def test_cards_that_do_not_depend_on_each_other_keep_layout_order():
    cards = [card('a', 'loco'), card('b', 'joints'), card('c', 'switch_mode')]
    assert order(cards, []) == ['loco', 'joints', 'switch_mode']


def test_a_four_level_chain_resolves():
    cards = [card('d', 'speaker'), card('c', 'tts'), card('b', 'asr'), card('a', 'mic')]
    conns = [conn('a', 'b'), conn('b', 'c'), conn('c', 'd')]
    assert order(cards, conns) == ['mic', 'asr', 'tts', 'speaker']


def test_a_card_fed_by_two_levels_waits_for_the_deeper_one():
    """decision_core's shape: one source is a source, the other is derived."""
    cards = [card('core', 'decision_core'), card('rm', 'remote_message'),
             card('mic', 'mic'), card('asr', 'asr')]
    conns = [conn('mic', 'asr'), conn('asr', 'core'), conn('rm', 'core')]
    got = order(cards, conns)
    assert got.index('asr') < got.index('decision_core')
    assert got.index('remote_message') < got.index('decision_core')


# ── degenerate graphs must still start something ─────────────────────────────

def test_a_cycle_is_reported_rather_than_dropped_or_hung():
    cards = [card('a', 'tts'), card('b', 'asr'), card('c', 'mic')]
    ordered, cyclic = order_cards_by_dependency(cards, [conn('a', 'b'), conn('b', 'a')])
    assert [c['toolName'] for c in ordered] == ['mic']
    assert sorted(c['toolName'] for c in cyclic) == ['asr', 'tts']


def test_a_self_loop_does_not_strand_its_own_card():
    cards = [card('a', 'vop')]
    assert order(cards, [conn('a', 'a')]) == ['vop']


def test_a_connection_from_a_deleted_card_does_not_strand_the_survivor():
    """Layout writes and connection deletes are separate saves; either can lag."""
    cards = [card('b', 'asr')]
    assert order(cards, [conn('card-gone', 'b')]) == ['asr']


def test_no_cards_and_no_connections():
    assert order_cards_by_dependency([], []) == ([], [])
