# Shared QL_printer actuator

QL_printer is registered in the perception MCP bundle alongside barcode and
phanthy_music. It has no U1 SDK, device or ROS topic dependency. Any robot with
this perception release and network access to the Windows bridge can use it.
It does not make this Jetson image compatible with every CPU/platform: each
robot must use a supported platform build.

Configure bridge_url, token, timeout_s (8) and print_timeout_s (60) in the
canvas. `find` accepts name and host. `print` accepts name and host and retains
the bridge ticket/idempotency protocol internally. An ambiguous or already
admitted visit is rejected. Windows bridge configuration and its database are
unchanged.

## Optional remote startup

Enable remote_start_enabled and configure the SSH user and port. The Windows
restricted authorized key must invoke the existing bridge scheduled task.
Windows must remain powered on with the interactive user logged in.

Mount a dedicated host secret directory into the perception container:

```yaml
volumes:
  - /opt/phanthy-motus/secrets/ql_printer:/run/ql-printer-auth:ro
```

It must contain `frontdesk_bridge_ed25519` (private key, readable only by the
service identity) and `known_hosts`. Verify the Windows host fingerprint before
installing known_hosts. Host checking is strict; no first-contact trust prompt
or automatic host-key acceptance occurs. Keys are never included in the image.
Existing U1 credentials can be securely provisioned into this dedicated
directory on that same robot; another robot should receive its own authorized
key. Do not place private keys in this repository.

## Migration sequence

1. Build/deploy the perception release and configure the optional secret mount.
2. Verify tools/list advertises QL_printer with type actuator in perception.
3. Stop the canvas project. Disable QL_printer in the U1 driver configuration
   (`QL_printer.enabled: false`) and restart that driver. Do not remove other
   U1 plugins. Confirm its tools/list no longer advertises QL_printer.
4. Remove the old U1 canvas card and add the perception QL_printer card. Reapply
   its bridge/token/SSH configuration and reconnect the required canvas links.
5. Test find, remote startup with bridge stopped, then one controlled print.
   Verify physical paper and visitor status before considering migration done.

If rollback is needed, disable the perception card and reenable the U1 card.
Avoid two active tools named QL_printer during cutover.
