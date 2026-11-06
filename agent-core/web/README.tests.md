# Frontend regression tests

From the repository root:

```sh
npm ci
npm test
```

CI/review runners must install the locked development dependency before running
`node --test agent-core/web/js/*.test.mjs`. Tests then run without network access.
`three@0.164.1` matches the r164 series loaded by the page import map. It is a
test-only dependency installed in the repository-root `node_modules/`. The Core
Docker build context is `agent-core/`, so neither this directory nor the root npm
manifests are sent to Docker. No `.dockerignore` change is needed. The browser
import map and production runtime dependencies are unchanged.

The URDF regression invokes the renderer's `_parseAndBuild` with real Three.js
Group/Euler/Quaternion/Matrix4 objects. Only XML input and bone visual creation
are stubbed; no WebGL context is needed. It compares the resulting local joint
matrix and quaternion against independently composed `Rz(yaw) * Ry(pitch) * Rx(roll)`.
Cases include both thumb origins and nonzero roll, pitch and yaw together.
It does not infer a matrix by interpreting the Euler order string itself.

Three.js uses intrinsic Euler orders. In this API, `ZYX` matches that URDF
composition; `XYZ` gives the other composition. Switching the renderer to XYZ
must fail these tests. This is a rendering check, not a hardware calibration test.
