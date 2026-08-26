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

/** The topic a card publishes on `portIdx`, or '' if not known yet. */
export function topicOfPort(card, portIdx) {
  const list = card?.topicOut || [];
  return list[portIdx]?.topic || list[0]?.topic || '';
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
  const conn = connections.find(c => c.toCardId === card.id);
  if (!conn) return '';
  const src = cards.find(c => c.id === conn.fromCardId);
  if (src) {
    const topic = topicOfPort(src, parseInt(conn.fromPortIdx, 10) || 0);
    if (topic) return topic;
    if (!allowStale) return '';
  }
  return conn.fromTopic || '';
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
  const maxRounds = opts.maxRounds ?? 4;
  const resolved = [];

  const ask = async (card, inputTopic) => {
    try {
      const res = await doFetch(`/api/mcp/${encodeURIComponent(card.mcpId)}/call`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          tool: card.toolName,
          arguments: { action: 'info', instance_id: card.id, input_topic: inputTopic },
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

  const unresolved = (c) => c.mcpId && c.toolName && !(c.topicOut || []).some(t => t.topic);
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
      if (!unresolved(c)) return false;
      const input = inputTopicOf(c, cards, connections, allowStale);
      return input && !alreadyAsked(c, input);
    });
    if (!pending.length) break;

    const inputs = pending.map(c => inputTopicOf(c, cards, connections, allowStale));
    pending.forEach((c, i) => noteAsked(c, inputs[i]));
    const answers = await Promise.all(pending.map((c, i) => ask(c, inputs[i])));

    let progressed = false;
    pending.forEach((card, i) => {
      const out = answers[i];
      if (out?.some(t => t.topic)) {
        card.topicOut = out;
        resolved.push(card);
        progressed = true;
      }
    });
    if (!progressed && !allowStale) {
      // Nothing more is derivable; skip straight to the last-resort pass.
      round = maxRounds - 1;
    }
  }
  return resolved;
}
