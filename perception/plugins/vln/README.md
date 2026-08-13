# Vision-and-Language Navigation (vln) card

The card has two public actions:

- `capture` has no parameters. It waits for a new camera/odometry pair, asks the
  VLM for a stable visual description, and records a visual waypoint at the
  current `map -> base_link` pose.
- `navigate` requires `query`. It compares the query with every point recorded
  in the current FAST-LIVO2 map session. A confident match is published to
  Nav2; a miss publishes nothing.

The underlying MCP arguments are `{"action":"capture"}` and
`{"action":"navigate","query":"去有白板的办公室"}`. Agent Core exposes these as a
zero-argument `capture` action and a `navigate(query)` action through
`x-action-params`.

## VLM configuration

Before starting the card, open its gear menu and configure the shared VLM
settings:

- `VLM API URL`: the service API URL.
- `VLM API Key`: the service credential (shown as a password field).
- `VLM model`: the provider's model name.
- `VLM timeout`: one to 120 seconds.

The API key has no source-code or YAML default and is never included in vln MCP
action responses, card status, or Perception request logs. For headless
startup, `VISION_AND_LANGUAGE_NAVIGATION_VLM_BASE_URL`,
`VISION_AND_LANGUAGE_NAVIGATION_VLM_API_KEY`,
`VISION_AND_LANGUAGE_NAVIGATION_VLM_MODEL`, and
`VISION_AND_LANGUAGE_NAVIGATION_VLM_TIMEOUT_SEC` are supported as fallbacks
until a gear configuration is applied. Explicit gear configuration always
takes precedence.

The Agent Core configuration framework stores shared card settings in
its local SQLite database and transports them over its control-plane HTTP API;
protect access to that database and API. `capture` sends the camera JPEG to the
configured external VLM service.

MCP is only the control plane from `decision_core` to these two actions. All
card-to-card data uses ROS2:

| Direction | Topic | ROS type | QoS |
| --- | --- | --- | --- |
| `camera_rgb -> vln` | `/ubuntu/camera/rgb` | `sensor_msgs/msg/CompressedImage` | best effort, volatile, depth 1 |
| `fast_livo2 -> vln` | `/ubuntu/navigation/odom` | `nav_msgs/msg/Odometry` | best effort, volatile, depth 5 |
| `fast_livo2 -> vln` | `/ubuntu/navigation/fast_livo2/status` | `std_msgs/msg/String` | reliable, transient local, depth 10 |
| `vln -> nav2` | `/ubuntu/navigation/goal_pose` | `std_msgs/msg/String` | reliable, volatile, depth 10 |

End-to-end robot deployment requires compatible FAST-LIVO2, vln, and Nav2
revisions that implement these ROS2 topic contracts, followed by coordinated
integration testing of all three cards.

The goal message data is compact JSON:

```json
{
  "schema": "phanthy.navigation.goal.v1",
  "goal_id": "vln-<unique id>",
  "x": 1.2,
  "y": -0.4,
  "yaw": 0.75,
  "speed": 0.3
}
```

`navigate` always publishes a matched goal, including when ROS discovery has
not found a subscriber yet. Its result reports `downstream_subscriber_ready`
separately so integration tests can distinguish publication from delivery.

## Canvas wiring

1. Connect `camera_rgb` to vln port `rgb`.
2. Connect FAST-LIVO2 port `livo_odom` to vln port `livo_odom`.
3. Optionally connect FAST-LIVO2 `status` to vln `livo_status`. When the canvas
   omits this optional binding, vln subscribes to the configured fixed status
   topic.
4. Connect vln port `goal_pose` to the Nav2 `goal_pose` input.
5. Add an execution connection from `decision_core` to `vln`.

The card supports both flat, unordered `input_topics` and port-aware
`input_bindings`, but rejects partial or ambiguous wiring.

## Limitations

- FAST-LIVO2 coordinates are session-local. Capture and publish are guarded by
  the status heartbeat plus a process-local map token. A bridge restart, stale
  heartbeat, non-mapping state, or observed session change blocks navigation.
- The upstream status contract has no true session UUID. A fast, same-name map
  restart that is invisible on the status topic cannot be detected perfectly;
  FAST-LIVO2 should eventually publish a per-run session UUID.
- Sensor pairing defaults to ROS receive time because the upstream cards do not
  yet guarantee a shared header clock. `sensor_sync_mode: source_timestamp` is
  available after that clock relationship is verified.
