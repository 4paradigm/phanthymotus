# ROI PR delivery — 2026-11-13

- Workspace: `/home/utopus/phanthy-work/phanthymotus-roi-goal-verifier`
- Branch: `feat/tabletop-roi-goal-verifier`; target: `4paradigm/phanthymotus:main`.
- Initial checkpoint: `651a463893b377e1869d7419590cf4248c0835ce`.
- Reviewed runtime-fix commit: `72b493989af253e3b40a1288a1831f424ede4565`. Documentation/package closure is committed separately; `git rev-parse HEAD` identifies the complete candidate. The immutable final HEAD, file list and package SHA256 are in the task artifact RELEASE/DELIVERY records.
- Code Ready / Draft PR Ready: YES. Regular PR Ready: NO pending actual D435/source/trust evidence and review. No known introduced unresolved P0/P1 finding remains.

PR text: PR.md. Functional guide: README.md / API.md. Actual tests and retained failures: TESTING.md. Findings/trust boundaries: REVIEW.md. Operator field steps: deploy/field/ONSITE.md.

The Card remains independent of Object #374 and of VOP/YOLOE. The only public file is perception/main.py (registration and token-log redaction). No shared config, Core/Web, camera driver or robot-control code is changed. The Object branch and its original dirty files remain intact; compatibility was tested in a disposable trial merge.

Task artifact root: `/home/utopus/phanthy-work/artifacts/dual-card-pr-ready-20261113/roi/`. It contains current regression/merge evidence, the rebuilt independent D435 kit, source patch/archive, SHA256 and final Git audit. The original `roi-independent-20261113` artifact remains historical and is not overwritten.

No push, PR creation, publication, shared-service changes, laboratory access or real robot motion was performed. Manual work remains: deploy to a new personal directory, verify actual camera/topic/frame/capture time/fixed view, run FAIL→PASS→INCONCLUSIVE with real frames and review the saved evidence. All current automated image evidence is synthetic.
