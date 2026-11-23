# Shared QL_printer actuator

QL_printer is registered in the perception MCP bundle alongside barcode and
phanthy_music. It has no U1 SDK, device or ROS topic dependency. Any robot with
this perception release and network access to the Windows bridge can use it.
It does not make this Jetson image compatible with every CPU/platform: each
robot must use a supported platform build.

The everyday card form only asks for the frontdesk computer IP, Windows SSH
user and target printer. It derives the bridge URL on port 8765, generates a
token for a clean installation, reads existing tokens through authorized SSH on
upgrade, and uses the standard timeouts, SSH port and container key path.
Non-standard values remain under advanced settings for troubleshooting. `list` takes no visitor
fields and returns every unadmitted visitor for
the current local date without exposing bridge tickets. `print` accepts name
and host, matches exactly one entry from that daily list, and retains the bridge
ticket/idempotency protocol internally. Ambiguous, admitted and departed visits
are rejected.

`print` (including the legacy `find_and_print` action) returns `accepted` with an
`action_id` immediately. This acknowledges submission, not a printed badge.
The existing ACP callback reports `completed` with `result.status=printed`, or
`failed` with the original business error. Wait for that result before announcing
success. The completion timeout is 180 seconds; only one print job per card is
accepted at a time, and each job keeps the configuration captured at submission.
Perception must be able to reach Agent Core through its configured
`AGENT_CORE_URL` for completion delivery. Do not automatically resubmit on a
callback timeout: the frontdesk job may already have printed.

Browser authentication capture listens before the first navigation, so a
successful initial query needs no extra reload. Navigation is limited to 5
seconds, automatic query capture to 3 seconds, and the optional search fallback
to 2 seconds (button click limit: 1 second). No fixed settling sleeps are used.
These limits concern page/query capture, not printing or subsequent API calls.

Failures retain their stage-specific code through ACP: `page_load_timeout`,
`page_http_error` (with HTTP status), `login_required` (observed login redirect),
`query_button_missing`, `query_click_timeout`, `query_request_timeout`,
`browser_closed`, or `browser_operation_failed`. `query_method_mismatch` is used
only when a request to the configured endpoint was actually observed with a
different method. An absent request does not prove login expiry or an endpoint
mismatch; the returned message describes the missing observation instead.

## Optional remote startup

Remote startup is enabled after initialization. Configure the Windows SSH user;
the standard port is 22. Initialization requires administrator SSH command
execution, not a key restricted to a forced startup command.
Windows must remain powered on with the interactive user logged in.

The published perception service template mounts a dedicated host secret
directory into the same container path on every robot:

```yaml
volumes:
  - /opt/phanthy-motus/secrets/ql_printer:/run/ql-printer-auth:ro
```

Do not manually edit a generated `docker-compose.yml`; deploying a perception
image containing this service template merges the mount into the host compose.
The host directory must contain `frontdesk_bridge_ed25519` (private key, readable only by the
service identity) and `known_hosts`. Verify the Windows host fingerprint before
installing known_hosts. Host checking is strict; no first-contact trust prompt
or automatic host-key acceptance occurs. Keys are never included in the image.
Each robot receives its own key with the same filename, so the card path remains
`/run/ql-printer-auth/frontdesk_bridge_ed25519` on every robot. Add each robot's
public key to the frontdesk Windows authorization file. Do not place private
keys in this repository.

## Migration sequence

1. Provision this robot's secret directory, then deploy the perception release;
   its service template applies the mount automatically.
2. Verify tools/list advertises QL_printer with type actuator in perception.
3. Stop the canvas project. Disable QL_printer in the U1 driver configuration
   (`QL_printer.enabled: false`) and restart that driver. Do not remove other
   U1 plugins. Confirm its tools/list no longer advertises QL_printer.
4. Remove the old U1 canvas card and add the perception QL_printer card. Reapply
   its bridge/token/SSH configuration and reconnect the required canvas links.
5. Test list, remote startup with bridge stopped, then one controlled print.
   Verify physical paper and visitor status before considering migration done.

If rollback is needed, disable the perception card and reenable the U1 card.
Avoid two active tools named QL_printer during cutover.
