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
| `speech_topics` | `auto` | Which inputs count as speech. `auto` asks the in-process TTS plugin what it publishes. Everything else connected is background. |
| `duck_gain` | `0.15` | What the background drops to while speech is present. 0 = silent, 1 = no ducking. |
| `duck_ramp_ms` | `200` | Ramp length. Too short is audible as a click. |
| `duck_hold_ms` | `300` | How long the duck is held after speech ends, so sentence gaps do not pump. |
| `background_gain` | `100` | Overall background trim. Per-track volume belongs to `phanthy_music`. |

`speech_topics` is the only setting most deployments touch, and `auto` is right
in every in-tree configuration. Set it explicitly only when TTS runs in another
container, where this process cannot ask it anything.

**Check `info()` after wiring.** It reports every bound topic with its resolved
role, and sets `degraded` when no speech source was identified. That matters:
with no speech source every input is background, so the robot's own voice would
be ducked by its own music, and nothing else would say why.

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
