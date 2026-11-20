# Shared QL_printer actuator

QL_printer is registered in the perception MCP bundle alongside barcode and
phanthy_music. It has no U1 SDK, device or ROS topic dependency. Any robot with
this perception release and network access to the Windows bridge can use it.
It does not make this Jetson image compatible with every CPU/platform: each
robot must use a supported platform build.

Configure bridge_url, token, timeout_s (8) and print_timeout_s (60) in the
canvas. `list` takes no visitor fields and returns every unadmitted visitor for
the current local date without exposing bridge tickets. `print` accepts name
and host, matches exactly one entry from that daily list, and retains the bridge
ticket/idempotency protocol internally. Ambiguous, admitted and departed visits
are rejected.

## Optional remote startup

Enable remote_start_enabled and configure the SSH user and port. The Windows
restricted authorized key must invoke the existing bridge scheduled task.
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
