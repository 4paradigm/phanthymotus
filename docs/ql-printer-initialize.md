# Remote Windows initialization

Update both AgentCore and perception. The perception image includes `/deploy/windows_bridge`.
In QL_printer settings enter the bridge URL, SSH account, port and mounted private-key path.
The account must have administrator SSH execution permissions; a forced start-only key cannot provision a computer.
Provision and verify `/run/ql-printer-auth/known_hosts` first. Private keys are never shipped in images.

Click Initialize/Upgrade. Blank tokens are generated in the browser; when upgrading enter the existing token.
The background job transfers the bundled files over SSH standard input and runs a PowerShell bootstrap.
It checks administrator permissions, installs signed Python 3.12 if missing, prepares a virtual environment,
installs Playwright, backs up code, preserves existing config/database/profile, and registers the interactive watchdog.
Existing task WorkingDirectory is honored; new installs use ProgramData/ParadigmVisitorBridge/app.
After success refresh printers, select a printer and SAVE the settings.

Prerequisites: Windows x64, enabled SSH, administrator key access, verified host key, Microsoft Edge,
and a robot image containing the offline Python installer and Windows wheels. Downloads from
python.org/PyPI happen at image build time; frontdesk pip uses --no-index. Transferred assets
are SHA256-checked and the Python installer signature is verified before execution.
The desktop task user must remain logged in.
Install printer drivers and sign into the visitor website separately. This is not a universal website adapter.
Bootstrap logs are under ProgramData/ParadigmVisitorBridge/bootstrap.log. A successful initialization
checks the HTTP service, not website login or physical printing. Test list and a controlled print afterward.

Runtime default-printer selection changes the Windows user's default printer and does not restore it.
Paper size and printer preferences must be configured on Windows for the selected printer.
Source tests use simulated SSH; clean-machine bootstrap requires on-site acceptance testing.
