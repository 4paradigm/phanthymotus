/**
 * topic-derive.js — resolve canvas cards' derived output topics from the driver.
 *
 * A multiInstance tool's output topic is *derived*: the driver infers it from the
 * input topic the card is connected to (`/remote_control/mic` + asr →
 * `/remote_control/mic/asr`). Nothing in the saved layout can be relied on to
 * hold it — the canvas page asks for it as a side effect of being open, so a
 * card's topic reaching the layout depends on someone having visited that page.
 * The monitor dashboard builds its panels from those topics, which is why the ASR
 * panel "isn't there from the start".
 *
 * `action: info` is a read, so asking is cheap and safe. Kept free of DOM and
 * renderer imports so it can be tested directly:
 *   node --test "agent-core/web/js/*.test.mjs"
 */

/**
 * The topic a card publishes on `portIdx`, or '' if not known yet.
 *
 * A port that exists but has no topic yet resolves to '' and never to another
 * port's topic. The blanket `|| list[0]?.topic` fallback this replaces made a
 * link out of port 1 silently carry port 0's data whenever port 1 had not
 * resolved — a depth output feeding a panel that then showed the colour stream.
 * Worse, the answer was non-empty, so `unresolved` below counted the card as
 * done and nothing ever corrected it. Same doctrine as inputTopicOf: an
 * unresolved source is waited for, not guessed at.
 *
 * The fallback survives for a genuinely out-of-range index on a *single*-output
 * card, which is what a layout saved before the card's ports changed looks
 * like. There "port 3" can only have meant the one port there is. On a
 * multi-output card it could mean any of them, so '' is the honest answer and
 * start-project's unresolved-input check reports the broken link.
 */
export function topicOfPort(card, portIdx) {
  const list = card?.topicOut || [];
  if (portIdx < list.length) return list[portIdx]?.topic || '';
  return list.length === 1 ? (list[0]?.topic || '') : '';
}

/**
 * Does anything feed this card?
 *
 * The distinction that matters: `inputTopicOf` returns '' both for a card that
 * nothing is connected to and for one whose source exists but has not resolved
 * yet. Those need opposite handling — the first should be asked about right away
 * (an input-less card publishes the driver's default), the second must be waited
 * for. Conflating them is what let a stale topic survive: TTS lost its inbound
 * connection, so it fell into the "waiting" case forever and was never re-asked,
 * while the driver had long since fallen back to /perception/tts.
 */
export function hasInboundConnection(card, connections) {
  return connections.some(c => c.toCardId === card.id);
}

/** Prefer the tool-declared preview output, otherwise the first resolved topic. */
export function selectPreviewTopic(candidates, declaredTopicOut = []) {
  const resolved = (candidates || []).filter(t => t?.topic);
  const preferred = (declaredTopicOut || []).find(t => t.default_preview === true);
  return (preferred && resolved.find(t =>
    (preferred.port && t.port === preferred.port) || t.topic === preferred.topic
  )) || resolved[0] || null;
}

/**
 * The topic feeding `card`, resolved through the graph.
 *
 * The source card's own topic wins, and if the source has no topic yet this
 * returns '' rather than the connection's `fromTopic`. That field is written when
 * the connection is drawn, so it holds whatever was known then — often empty, and
 * sometimes a leftover from a link that has since been deleted. Deriving from a
 * leftover is how a card ends up publishing to a topic nothing feeds, so a source
 * that is merely *not resolved yet* must be waited for, not guessed at.
 *
 * `allowStale` is for the last resort: a source whose topic cannot be derived at
 * all (not on the canvas), where the persisted value is the only evidence there is.
 */
export function inputTopicOf(card, cards, connections, allowStale = false) {
  return inputTopicsOf(card, cards, connections, allowStale)[0] || '';
}

/** The topic one connection carries, per the rules in inputTopicOf's comment. */
function topicOfConnection(conn, cards, allowStale) {
  const src = cards.find(c => c.id === conn.fromCardId);
  if (src) {
    const topic = topicOfPort(src, parseInt(conn.fromPortIdx, 10) || 0);
    if (topic) return topic;
    if (!allowStale) return '';
  }
  return conn.fromTopic || '';
}

/**
 * Every topic feeding `card`, in the order the connections were drawn.
 *
 * The canvas lets several connections into one card — decision_core declares a
 * single `data/json` input and is normally fed by three — and api/config.py's
 * _resolve_input_topics has always walked all of them. This side walked one:
 * `connections.find(...)`, so the card's derived topic depended on which link
 * happened to be first in the array, and redrawing a link changed the answer.
 * The two sides then disagreed about what the card consumes, which is how a
 * monitor panel ends up subscribed to a topic the start never used.
 *
 * Returns [] if *any* connection is unresolved, rather than a partial set — the
 * same "wait, don't guess" rule inputTopicOf applies to a single source, for the
 * same reason: deriving from half the inputs produces an answer that has to be
 * thrown away, and on the start path it would bind a node to half a graph.
 * Duplicates are dropped, so two links carrying the same topic ask once.
 */
export function inputTopicsOf(card, cards, connections, allowStale = false) {
  const topics = [];
  for (const conn of connections.filter(c => c.toCardId === card.id)) {
    const topic = topicOfConnection(conn, cards, allowStale);
    if (!topic) return [];
    if (!topics.includes(topic)) topics.push(topic);
  }
  return topics;
}

/**
 * The arguments that tell a tool what to consume.
 *
 * Mirrors api/config.py's _start_and_resolve: the plural form when there is
 * more than one, and the singular alongside it because no driver reads the
 * plural — sending only `input_topics` reaches them as no input at all.
 */
export function inputArgs(topics) {
  if (topics.length > 1) return { input_topics: topics, input_topic: topics[0] };
  return { input_topic: topics[0] || '' };
}

/** A stable cache key for a set of input topics. */
export function inputKey(topics) {
  return topics.join('\n');
}

/**
 * Fill in the output topics the layout does not know, by asking each driver.
 *
 * Mutates `cards[].topicOut` in place and returns the cards it resolved. Rounds
 * let a chain settle (mic → asr → tts): each round only asks about cards that
 * gained a known input in the previous one, and it stops as soon as a round learns
 * nothing, so an offline driver costs one round rather than `maxRounds`.
 *
 * Then one final round allows the persisted `fromTopic`, for a card whose source
 * is not on the canvas and therefore never resolvable — by which point anything
 * derivable has been derived, so a stale value can no longer win over a real one.
 */
export async function resolveDerivedTopics(cards, connections, opts = {}) {
  const doFetch = opts.fetchImpl || ((...a) => fetch(...a));
  // Callers that can tell a derived (multiInstance) tool from a static one pass
  // this. Default is "no card is derived", which keeps the conservative
  // behaviour: only cards with no topic at all are asked about.
  const isDerived = opts.isDerived || (() => false);
  const maxRounds = opts.maxRounds ?? 4;
  const resolved = [];

  const ask = async (card, inputTopics) => {
    try {
      const res = await doFetch(`/api/mcp/${encodeURIComponent(card.mcpId)}/call`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          tool: card.toolName,
          arguments: { action: 'info', instance_id: card.id, ...inputArgs(inputTopics) },
        }),
      });
      const json = await res.json();
      let data = json.data;
      // tools/call answers either as a parsed dict or as MCP content items.
      if (Array.isArray(data) && data[0]?.text) data = JSON.parse(data[0].text);
      return data?.topic_out;
    } catch {
      return null;
    }
  };

  // A card counts as resolved only when *every* port has a topic. The old test
  // was `!some(t => t.topic)` — any one resolved port marked the whole card
  // done, so a two-output card whose second port came back empty kept that port
  // empty for good, and every link out of it stayed dead.
  const unresolved = (c) => {
    if (!c.mcpId || !c.toolName) return false;
    const list = c.topicOut || [];
    return !list.length || list.some(t => !t.topic);
  };
  // What each card has already been asked. The last-resort pass asks a different
  // question (a different input topic), but only where it *is* different —
  // otherwise an offline or can't-infer driver would be asked the same thing twice.
  const asked = new Map();
  const alreadyAsked = (card, input) => asked.get(card.id)?.has(input);
  const noteAsked = (card, input) => {
    if (!asked.has(card.id)) asked.set(card.id, new Set());
    asked.get(card.id).add(input);
  };

  for (let round = 0; round <= maxRounds; round++) {
    const allowStale = round === maxRounds;   // final pass only
    const pending = cards.filter((c) => {
      if (!c.mcpId || !c.toolName) return false;
      // Every *derived-topic* card is asked once, whether or not it already
      // holds a topic. A topic in the saved layout is not evidence — it is a
      // snapshot of what some browser derived at some point, and this file
      // exists because that snapshot cannot be trusted.
      //
      // This used to be narrower: only a derived card with *no inbound
      // connection* was re-asked, on the grounds that its topic may have
      // outlived a deleted link. A connected one that already had a topic was
      // skipped, so nothing here could ever notice the driver renaming its
      // output. When perception's visual_depth stopped deriving `{input}/depth`
      // and started deriving `{input}/visual_depth`, the dashboard went on
      // subscribing to the old name and showed empty panels with nothing to say
      // why. The canvas page does re-ask on every load and had the new names in
      // its ports, but a viewer is deliberately barred from persisting them
      // (canvas.js _saveLayout requires the edit lock), so the layout the
      // monitor reads stayed wrong indefinitely.
      //
      // The cost is one `info` per derived card per run, which is a read.
      // `asked` keeps it to once, and an answer is only taken when it is
      // non-empty and actually different, so a driver that cannot infer cannot
      // blank out a good topic.
      //
      // Restricted to cards the caller marks as derived (multiInstance). A card
      // with a static topic gets it from the MCP schema, and asking its driver
      // instead would let an `info` answer overwrite the declared value.
      const derived = isDerived(c);
      const inputless = derived && !hasInboundConnection(c, connections);
      if (!unresolved(c) && !derived) return false;
      const input = inputTopicsOf(c, cards, connections, allowStale);
      // Empty means either nothing feeds this card, or some source has not
      // resolved yet — only the first is answerable now.
      if (!input.length && !inputless) return false;
      return !alreadyAsked(c, inputKey(input));
    });
    if (!pending.length) break;

    const inputs = pending.map(c => inputTopicsOf(c, cards, connections, allowStale));
    pending.forEach((c, i) => noteAsked(c, inputKey(inputs[i])));
    const answers = await Promise.all(pending.map((c, i) => ask(c, inputs[i])));

    let progressed = false;
    pending.forEach((card, i) => {
      const out = answers[i];
      if (out?.some(t => t.topic)) {
        // Only count it as progress when it actually differs — re-confirming an
        // input-less card's existing topic must not keep the loop spinning.
        if (JSON.stringify(out) !== JSON.stringify(card.topicOut)) {
          card.topicOut = out;
          resolved.push(card);
          progressed = true;
        }
      }
    });
    if (!progressed && !allowStale) {
      // Nothing more is derivable; skip straight to the last-resort pass.
      round = maxRounds - 1;
    }
  }
  return resolved;
}

/**
 * Rewrite each connection's `fromTopic` to what its source card actually publishes.
 *
 * `fromTopic` is written when the link is drawn and never revisited, so it holds
 * whatever was known then. The monitor dashboard subscribes to it directly, so a
 * leftover value becomes a panel watching a topic nothing publishes on — the same
 * empty-panel symptom as a stale card topic, from the same cause one step along.
 *
 * Only rewrites when the source has a real topic: an unresolved source would
 * otherwise blank out the one piece of evidence the last-resort pass relies on.
 */
export function syncConnectionTopics(cards, connections) {
  const fixed = [];
  for (const conn of connections) {
    const src = cards.find(c => c.id === conn.fromCardId);
    if (!src) continue;
    const topic = topicOfPort(src, parseInt(conn.fromPortIdx, 10) || 0);
    if (topic && topic !== conn.fromTopic) {
      conn.fromTopic = topic;
      fixed.push(conn);
    }
  }
  return fixed;
}
