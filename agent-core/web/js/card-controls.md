# Optional card presentation and confirmation metadata

Existing drivers keep their direct execution path. A tool may provide a `title` for its human card name,
property `title` values, and `oneOf: [{const, title}]` labels while retaining normal `enum` protocol values.
`x-ui-hidden` omits an internal field from the form. `x-visible-when: {object: "arm"}` conditions a field
on another selector; fields hidden by action/conditions are not included in the request.

A schema can opt into `x-confirmation` per action:

```json
{
  "x-confirmation": {
    "move": {
      "preview": {"action": "move", "execute": false},
      "execute": {"execute": true, "workspace_clear": true, "take_control": true},
      "message": "Confirm the target, clear workspace and current operator control."
    }
  }
}
```

The browser calls the preview with the entered parameters, shows actual targets/changes, and requires
operator confirmation before submitting identical parameters with that returned `preview_id`.
A blocked or missing preview, or cancelling the confirmation, never issues the execution call.
An already-ready `{confirmed: true, executed: false}` result does not send another command.
A rule without `preview` (e.g. one camera snapshot) asks confirmation before its sole call.

This metadata is presentation, not caller authentication. Drivers must enforce permission, current
operator confirmations, one-use preview binding/expiry, state freshness and mutual exclusion on the server.
A long operation still follows its driver's ACP completion contract; submitting is not completion.

Test: `node --test agent-core/web/js/card-controls.test.mjs`.
