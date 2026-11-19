# Helios renderer regression tests

Run with Node.js 22+ from this directory:

```bash
npm ci --ignore-scripts
npm test
```

Tests execute the actual canvas, detail-panel and renderer source in a JavaScript VM, with DOM/WebGL boundaries stubbed. Three.js and xmldom are development dependencies only; there is no frontend build step or new production dependency.

Coverage:

- Live and cached canvas detail paths preserve the real MCP ID, and model fetch uses that ID.
- Independent URDF `Rz * Ry * Rx` forward kinematics for 20 movable joints and 34 links.
- Frames received before model loading are replayed; the whole model fits narrow and wide cards without resetting the camera on every frame.
- Helios shoulder line, head symbol and hidden sensor branches preserve all real link transforms; other models retain their markers.
- Battery units, zero/missing values, per-frame timestamps, temperature and hidden unsupported fields.
- Expanded Details and reading position survive 120 incoming frames; history stays bounded to 50 entries and following resumes on close.
- Escaping and latest-view freshness, plus a native G1 log snapshot to catch unrelated display changes.

Fixtures contain fixed historical protocol examples and are never used as live data. `battery.json` contains only the Driver's public battery payload. `g1-log.html` is the unmodified native text renderer's output for synthetic G1 data at a fixed clock.

`helios-kinematics.urdf` is a test-only projection of the Helios Driver model: all link/joint names, origins, axes and limits are retained; inertial, visual and collision elements are removed. Original source SHA-256: `490202573e9bc6bb54668440a4fcf39c4b4f5b0af71548806be76de695bf726f`, installed manufacturer model `Rokae_Robot_2.1_AR5_v1.1` (2026-11-18). No mesh data is included. Manufacturer model rights remain with their owners; applicable release terms require confirmation before upstream acceptance.

These checks do not replace a browser WebGL visual check, bot-image deployment or physical-motion acceptance.
