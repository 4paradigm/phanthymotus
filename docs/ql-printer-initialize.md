# Remote Windows initialization

Use an Agent Core version supporting the visitor-card configuration UI and
deploy the matching JetPack Perception image. It includes deployment scripts
and offline Windows dependencies under `/deploy/windows_bridge`.

In Visitor_Badge_Printer settings enter the frontdesk IP, Windows administrator
SSH username and password. The default port is 22. No shared HTTP token or SSH
client private key is required. Windows x64 must have SSH and Microsoft Edge
installed, a D drive, and an interactive desktop session for the same account.

Click Install/Upgrade and keep the dialog open. The existing flow uses the
Windows dependencies already bundled in the robot image. Initialization needs
no new network downloads and no manual public-key registration or secret mount.
Artifact hashes are fixed in source; wheel URLs and sizes are pinned in
`offline-wheels.json`. Dependencies are downloaded and checked at image build
time; Windows installs from the uploaded files with `--no-index`.

The robot uploads files in chunks with SCP and runs `bootstrap_ssh.ps1` over
password-authenticated SSH. Bootstrap verifies transferred SHA256 hashes and the Python
installer signature, installs Python if absent, creates a virtual environment,
and installs wheels with `--no-index`. It deploys to
`D:/ParadigmVisitorAutomation/app`, preserves the existing config and data there,
and registers `Paradigm Visitor Automation` and `Paradigm Visitor Login`
interactive scheduled tasks. The obsolete HTTP bridge tasks are removed.

After installation, sign in to the visitor platform in the frontdesk Edge
window, select the correct visitor list URL and printer, and save. A successful
initialization confirms bootstrap completion; it does not confirm platform
login, correct page selection or physical printing. Test list and one controlled
print. Initialization errors are recorded at
`D:/ParadigmVisitorAutomation/install.log`; runtime diagnostics are at
`D:/ParadigmVisitorAutomation/app/data/worker.log`.

Windows must remain powered on with that user logged in. Install printer
drivers, paper size and preferences separately. Runtime selection changes the
Windows user's default printer and does not restore it. The platform adapter
supports the configured visitor query/detail flow, not arbitrary websites.
Automated tests simulate SSH and browser behavior; clean-machine installation
and physical printing still require on-site verification.
