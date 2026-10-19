# audio_mixer — one speaker, several sources, speech always on top

A speaker driver subscribes to exactly **one** topic and does not mix. That is
true of every driver in the tree (G1, R1, Go2, T800, Bumi, U1 Pro): `start`
takes a single `input_topic` and creates one subscription. So the moment a robot
has both a voice and a background bed, something upstream has to sum them.

That used to be `phanthy_music`, which is why it took TTS in on its own
`topic_in` and published the combined stream. This card is that job, alone.

## Wiring

```
TTS            ─┐
                ├─► audio_mixer ──► Speaker
phanthy_music  ─┘
```

Both sources land on the card's single in-port. Speech is passed at unity gain
and is never attenuated or interrupted; everything else is background and is
ducked under it.

## Why one in-port and not two

`api/config.py:_resolve_input_topics` collects a card's inbound connections by
filtering on `toCardId` and **never reads `toPortIdx`**. The topics arrive as a
role-less list in draw order. So a declared "speech" port and "background" port
could not be told apart at `start` — the assignment would be a coin flip.
`perception/plugins/echo_cancellation.py` refused a second in-port for exactly
this reason.

Roles therefore come from config, not from which wire went where.

## Settings

| Key | Default | Meaning |
|---|---|---|
| `speech_topics` | `auto` | Which **of this card's inputs** count as speech. `auto` resolves by asking the in-process TTS plugin what it publishes. Everything else connected is background. |
| `duck_gain` | `0.15` | What the background drops to while speech is present. 0 = silent, 1 = no ducking. |
| `duck_ramp_ms` | `200` | Ramp length. Too short is audible as a click. |
| `duck_hold_ms` | `300` | How long the duck is held after speech ends, so sentence gaps do not pump. |
| `background_gain` | `100` | Overall background trim. Per-track volume belongs to `phanthy_music`. |

`speech_topics` is the only setting most deployments touch, and `auto` is right
in every in-tree configuration. Set it explicitly only when TTS runs in another
container, where this process cannot ask it anything.

The dropdown lists the topics **actually wired into this card**, so the choice is
made among real inputs. It used to list whatever the local TTS plugin published,
which was a guess at who the speech source is — and a TTS in another container, a
driver's own voice card, or a second audio source never appeared in that guess.
Because the list comes from resolved connections, it is only complete once the
card has started.

**Check `info()` after wiring.** It reports every bound topic with its resolved
role, and sets `degraded` when no speech source was identified. That matters:
with no speech source every input is background, so the robot's own voice would
be ducked by its own music, and nothing else would say why.

## The speech jitter buffer

Before starting an utterance the mixer accumulates `speech_prefill_ms`
(default 200 ms).

This is not tuning — it fixes a bug observed on a rig: **with music playing,
speech breaks up.** The cause is neither decoding nor gain, it is *what drives
the output cadence*. With a background source present the mixer publishes on
every frame, so each late speech frame becomes a 100 ms silent hole in the
output. Without background the same lateness produces no hole: that tick has
nothing from either source, so nothing is published and the cadence follows
speech naturally. Which is why the bug appears **only** when both play at once,
and why testing each alone showed nothing. Reproduced with one late frame:

```
with music:    [3000, 3000, 0, 3000, 3000, 3000]   ← hole
without music: [3000, 3000, 3000, 3000, 3000]      ← none
```

Speech arrival is inherently jittery (ROS callbacks plus synthesis pacing), so
this is not a rare race.

200 ms and not more because `tts.py` already bursts 5 frames before pacing, so
the buffer fills almost immediately and the first syllable is not delayed —
and that delay is paid directly in barge-in responsiveness. An EOF releases
immediately, so a short utterance is never held back by the prefill.

The music is already ducked while the buffer fills: `_speech_open` is one of the
duck triggers, so by the time speech is emitted the music has long since moved
out of the way.

## Two tiers, not a priority matrix

One speaker carries one foreground voice over one background bed. A general
N-way priority lattice would be a setting nobody can reason about, and the
hardware cannot honour it anyway — there is one mix bus and one output topic.

## Duck, not pause

Speech over music always ducks. Pausing would stall the mp3 decoder and cost an
audible seam at every utterance, and a robot in conversation speaks constantly.
`phanthy_music`'s `pause`/`resume` exist only for an explicit user request.

## What is not wired

`duck` / `unduck` / `flush_speech` are dispatchable but deliberately absent from
`x-action-params` — mixing is wiring, not a decision for an agent.

`flush_speech` is bound to `on_interrupt_speak` and `on_interrupt_all`, and both
are fired by agent-core, so barge-in drops speech already buffered here (up to
`tts.py`'s 500 ms prebuffer). `duck`/`unduck` have **no** hook that fires them
today: there is no `on_hearing` binding path into this card. They are reachable
from the dashboard and from a driver that knows about speech agent-core cannot
see. Do not read them as an automatic mechanism.
