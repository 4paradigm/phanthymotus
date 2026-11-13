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
matrix and quaternion against the closed-form fixed-axis RPY matrix, calculated
from scalar sine/cosine terms independently of the renderer's multiplication chain.
Cases include both thumb origins and nonzero roll, pitch and yaw together.
It does not infer a matrix by interpreting the Euler order string itself.

The renderer explicitly constructs `Rz(yaw) * Ry(pitch) * Rx(roll)` instead of
relying on an Euler order label. An intrinsic XYZ result is included as a
negative control and must differ for these mixed-angle cases. Reversing the
renderer matrix product must fail these tests. This is a rendering check,
not a hardware calibration test.

Frontend validation is separate from the Python `agent-core` suite reported by
the image review bot. A Python-suite pass does not establish that these Node.js
tests ran. Record `npm ci` / `npm test` results separately in PR validation notes.
