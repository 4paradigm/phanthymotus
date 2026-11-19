# Helios joint and dual-battery display

The Helios Driver introduces two sensor cards (`joints`, `battery`) and a `model` resource. This Core branch supplies the current display behavior; it is needed alongside the Driver branch, rather than being installed by a Driver image.

Driver branch: https://github.com/jiekaren/phanthymotus-driver/tree/feat/helios-driver

## Shared renderer fixes

Canvas details pass the originating MCP ID through the detail panel to model loading. URDF static roll/pitch/yaw uses `Rz * Ry * Rx` (Three.js ZYX). Joint frames received while the model is loading are replayed, and the loaded model is framed for the card aspect ratio. Live frames preserve the user's view. Latest-value rendering removes keys absent from a new snapshot so old joint values do not survive an invalid/empty state.

## Helios display

Only the URDF robot name `Rokae_Robot_2.1_AR5_v1.1` enables the schematic shoulder line, short neck and head sphere. All 34 kinematic frames and 20 actual joint rotations remain intact; the upper-body drawing follows the real shoulder positions and head orientation. This is a schematic, not calibrated head/shoulder geometry. No STL loading is included.

Dual-battery data matching the Driver contract uses native English log markup, colors and font sizes. Both packs share one log timestamp per received frame. It displays SOC, pack voltage, cell-voltage extrema, the latest maximum pack sensor temperature and cycle count. Capacities, BMS versions and diagnostics are folded under Details. Current, unavailable SOH and unverified charging state are hidden.

Opening Details pins the historical entry and reading position while new frames arrive. Closing all details resumes following the latest frame. Retention is bounded to 50 frames; if every retained frame is pinned, new log rows are discarded until a detail closes. Historical values and timestamps do not update in place. Latest mode separately clears stale readings after 3 seconds of silence and does not invent receive timestamps.

## Validation and delivery

Run the suite described in README.md. Existing local robot debug deployment was exercised, but this branch is not a bot-built Core image and has not completed browser or Master/teacher acceptance. Driver and Core need separate PRs and review, with their versions recorded and both deployed before formal acceptance. No robot deployment or movement is performed by preparing this branch.
