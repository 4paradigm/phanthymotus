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

`synthetic-model.mjs` generates original abstract test geometry with arbitrary dimensions, axes and compound rotations. It retains only the joint/link identifiers needed to exercise the renderer's interface. No manufacturer transforms, limits, inertial values, visual geometry or model files are copied. The 20-axis/34-link counts exercise the same code paths without bundling a manufacturer model.

The manufacturer-derived fixture in the initial PR commit has been deleted from the current source tree in response to review. This does not claim to erase earlier public Git history or establish redistribution permission for the original model.

These checks do not replace a browser WebGL visual check, bot-image deployment or physical-motion acceptance.
