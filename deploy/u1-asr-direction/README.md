# U1 single-speaker direction experiment deployment

This overlay reuses the installed driver and perception services. It is not a standalone deployment. Use only the matching final PR build images, filling a local images.env based on images.env.example. It contains no credentials. Preserve the U1 B baseline: minimum_dbfs=-65, margin_db=2.5, settle=.5s, pause hold=.8s. Driver PR #449 includes the operator-selected defaults and the deployed idle-action fixes through 08e03ce.

Before updating, inspect the live image/project/compose path again and coordinate with other developers. Back up the two actual compose files, existing activity tuning config and saved canvas. Keep all existing mounts, networking, audio gain and other environment settings. Stop the canvas first. Remove ONLY the temporary /work/plugins/asr.py override installed by this experiment, if present; otherwise it would mask the new perception image source. Keep activity config mounts and the owned native DOA worker unchanged when their config has not changed. No firmware or behavior switches are required.

Example for the currently observed split projects (adjust if ownership changes):

```sh
docker compose --env-file images.env -p phanthy-motus -f /opt/phanthy-motus/docker-compose.yml -f compose.experiment.yml up -d --no-deps perception
docker compose --env-file images.env -p pr-image-ubtrobot-u1-pro -f /tmp/pr-image/ubtrobot-u1-pro/docker-compose.yml -f compose.experiment.yml up -d --no-deps ubtrobot-u1-pro
```

Do not use --remove-orphans or start the overlay alone. Subsequent recreation must include the same overlay or persist its two image values and experiment variable in the existing compose files. Roll back using the saved compose/config and previous image versions.

Canvas: mic -> ASR audio input; sound_direction -> ASR direction input; ASR main result -> decision_core. Preserve existing ASR settings including asr_kws and 小范小范. An old ASR card can lack the second saved port even if the service supports it; restore only the missing direction slot, retaining the audio slot and instance ID. Refresh the driver's tools metadata if the card is absent. For the voice-only experiment, back up original tool bindings and temporarily keep only TTS available to the agent; do not grant head/expression actions for this test.

Say “小范小范，你知道我在哪吗？” from one fixed position. Inspect both the ASR result and actual Agent Core reply: estimated direction should be explicit, conflict/no new data -> unknown, and no movement. This is not a distance estimate, verified capture association, or multi-speaker attribution. See U1_ASR_DIRECTION_EXPERIMENT.md. The gate itself is not speech classification; phone music failing to trigger does not establish speech-only capability.
