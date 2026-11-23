# Shared QL_printer actuator

QL_printer is registered in the perception MCP bundle alongside barcode and
phanthy_music. It has no U1 SDK, device or ROS topic dependency. Any robot with
this perception release and network access to the Windows bridge can use it.
It does not make this Jetson image compatible with every CPU/platform: each
robot must use a supported platform build.

The card asks for the frontdesk computer IP, Windows SSH username/password,
visitor list URL and target printer. SSH uses port 22 by default; SCP uses the
same password credentials for installation. No HTTP bridge, shared
token or client private key is used. Runtime jobs run through Windows scheduled
tasks and local request/result files under D:/ParadigmVisitorAutomation/app.
Non-standard ports and timeouts remain under advanced settings. `list` takes no visitor
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
The card retains the same direct-print request ID for the same destination,
page, printer, visitor and host after an uncertain result; a retry retrieves the
Windows stored result rather than submitting a second badge. Definitive results
release the key. This in-memory association survives configuration edits but not
a Perception restart; after a restart, verify uncertain jobs at the front desk
before submitting another print.

Browser authentication capture listens before the first navigation, so a
successful initial query needs no extra reload. Navigation waits for document
markup (`DOMContentLoaded`) with a 5-second ceiling, without waiting on images
and other load-blocking resources. Requests are observed during navigation;
there is no separate automatic-query wait. Without a captured request, a visible
application loading mask gets at most 2 seconds to clear (no mask returns
immediately), then query buttons are checked immediately. Missing buttons return
`query_button_missing`; hidden buttons return `query_button_unavailable`.
A visible button is clicked with a 1-second limit, and only then is a matching
request awaited for at most 2 seconds. No fixed settling sleeps are used.
These limits concern page/query capture, not printing or subsequent API calls.

Failures retain their stage-specific code through ACP: `page_load_failed`,
`page_load_timeout`, `page_render_timeout`,
`page_http_error` (with HTTP status), `login_required` (observed login redirect),
`query_button_missing`, `query_click_timeout`, `query_request_timeout`,
`browser_closed`, or `browser_operation_failed`. `query_method_mismatch` is used
only when a request to the configured endpoint was actually observed with a
different method. An absent request does not prove login expiry or an endpoint
mismatch; the returned message describes the missing observation instead.
An absent query button is not classified as a timeout or proof of a wrong URL.
Overall tool-call time also includes SSH, Windows scheduling and browser startup.

## Optional remote startup

Remote startup is enabled after initialization. Configure the Windows SSH user;
the standard port is 22. Initialization requires administrator SSH command
execution, not a key restricted to a forced startup command.
Windows must remain powered on with the interactive user logged in.

Enter the computer IP and Windows username/password, then initialize.
No manual public-key registration or additional secret mount is required.
SSH and SCP keep their existing first-contact host-key acceptance policy;
this does not independently verify the first connection's host identity.
Do not place private keys in this repository.

Windows provisioning dependencies remain bundled in the Perception image,
so initialization does not add a robot or Windows network requirement.
Versions, official wheel URLs, byte sizes and SHA256 hashes are pinned in
`offline-wheels.json`; the Python installer SHA256 is pinned in `build_offline.py`.
Mirrors must match these hashes. The image-size cost of offline initialization
remains; moving dependencies to runtime download would change the supported flow.

## Migration sequence

1. Deploy the perception release matching the robot's JetPack version.
2. Verify tools/list advertises QL_printer with type actuator in perception.
3. Stop the canvas project. Disable QL_printer in the U1 driver configuration
   (`QL_printer.enabled: false`) and restart that driver. Do not remove other
   U1 plugins. Confirm its tools/list no longer advertises QL_printer.
4. Remove the old U1 canvas card and add the perception QL_printer card. Reapply
   its IP/Windows credentials/page/printer configuration and reconnect links.
5. Run initialization, test list, then one controlled print.
   Verify physical paper and visitor status before considering migration done.

If rollback is needed, disable the perception card and reenable the U1 card.
Avoid two active tools named QL_printer during cutover.
