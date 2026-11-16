# SLAM visual recovery evidence — 2026-11-17

## Scope and initial findings

Changes target **PR371 only**, from `9a268b84e8d5ac4516b90b93f5734284d3a92fed`
in `.worktrees/pr371-current-review` (local branch alias
`feat/generic-rgbd-odometry-20261111`). PR371 remains open/draft at that remote
head. PR395 remains open at `f914ab6c5b48062cceff5abe26cc94efdd8c811a`; it is an
earlier independent extraction with different admission/build scope. Its worktree
was clean and was not edited. No remote push, deployment, robot service restart
or motion command was performed.

The initial 62 RGB/gateway/native-adapter tests passed. Existing image input,
projection and fallback did not establish a real loop or pose-loss recovery:
null poses never reached native mapping; the pinned core disallows null odometry
in incremental mode; saved snapshots incorrectly used raw `Node.pose`; and
proximity could masquerade as a global visual edge. Historical smoke evidence
was retained as historical, not reused as current acceptance.

Implementation and reproducible commands are in [visual recovery](slam-visual-recovery.md).
The native extension uses existing localization/registration/graph/OdometryF2M
APIs, preserving the original database and graph. It adds pose-less forwarding,
geometric recovery, separate sustained visual tracking, numerical external-origin
alignment, saved-map continuation, final optimized reconstruction and bounded
asynchronous image ownership/waiting. Admission remains numerical-format-only.

## Test levels and environment

- Python tests include ROS/native-process doubles: these verify gateway joins,
  bounds, publication, faults/stop, archive handling and actual regressions.
- Native core tests execute the pinned C++ implementation on generated metric
  RGB/depth/scan observations. They do not use a simulated RTAB-Map process.
- ROS replay executes production Python gateway, JPEG codec, SensorData converter,
  patched CoreWrapper and native graph optimization on the same generated scene.
  ROS replay no longer waits for JPEG decoding before sending the cloud.
- None of these are recorded moving-robot datasets or physical robot tests.

Native execution used an isolated local Lima Ubuntu22.04 arm64 VM, 2 CPUs,
3GiB RAM and 2GiB swap, localhost-only DDS with nonzero domains. Exact core pin
`df6300e0ba3e90058f90b09c4d646279366d3516` and ROS pin
`2a31e433d2b072795ae90a62abcdd0c00691e04d` were source-built with the repository
patches. OpenCV4.5.4/PCL1.12 and ROS Humble dependencies were used. The source
overlay explicitly selected `/usr/local/lib/rtabmap-0.22` to avoid a mixed
system-0.23 ABI. Final Release/O3 wrapper build passed in 6m10s; GCC heap-GC
flags `--param=ggc-min-expand=10 --param=ggc-min-heapsize=32768` limited compiler
memory without changing algorithm thresholds. This is not a Jetson image build.

## RGB on/off: true loops and map alignment

The same 62-frame departure/revisit and drifted external odometry were used.
The deterministic wall has fixed random texture and non-planar metric geometry;
the endpoint deliberately repeats the initial rendered view. Endpoint precision
therefore is not a physical accuracy claim.

| Saved native result | RGB enabled | Scan/external odometry only |
|---|---:|---:|
| Verified historical revisit global edges | 28 | 0 |
| 3D trajectory RMS error | 0.0986m | 0.3221m |
| 3D trajectory maximum error | 0.1814m | 0.6061m |
| Endpoint 3D error | <0.001m | 0.6061m |
| Surface alignment RMS | 0.0386m | 0.1871m |
| Surface alignment p95 | 0.0909m | 0.3924m |

“Verified” requires native visual inliers >=20, a persisted type-1 edge and a
historical temporal separation, not a loop counter. Surface residuals compare
transformed native scans against the known metric wall and expose displaced
duplicate surfaces. `rgb-comparison.png` plots both saved trajectories and map
residual distributions. Scan-only may have native geometric/proximity edges;
none are credited as visual recognition.

Image frames retain supplied motion covariance and use visual initialization
before ICP; they also enable the native, default-off `RGBD/LoopCovLimited`
weight bound. This is a deliberate algorithm-weight change. Feature matching,
ICP and `RGBD/OptimizeMaxError` thresholds were not relaxed. The new core
136-frame recovery run passed all positive/negative checks; its final maximum
linear constraint ratio was 3.715 on an existing neighbor edge (absolute error
0.0188m). Final global optimization is not itself a second native edge-admission
pass; the evaluator reports this residual rather than calling every saved
constraint perfect. The final ROS map below has lower measured graph residuals.

## Full production ROS replay

Original world origin was deliberately `(7,-3,0.5)`, not zero. All 136 frames
remained in one native process/database/segment; no restart was used to recover.

- First recovery: frame71, native ref73 -> historical node62, **436 visual
  inliers**, persisted global edge. Second recovery: frame125, ref128 -> node73,
  **636 visual inliers**, persisted global edge. Both begin without external pose.
- Recovery after valid revisit observations took **1.872s / 1.670s** monotonic
  wall time (two observations each). Successful-observation-to-Info delays were
  **567ms / 369ms**; native reported processing was **147ms / 153ms**.
- With external pose continuously absent, the next phases retained **37 and 8
  newly tracked nodes**. Localization alone was not counted as sustained tracking.
- Four appearance-identical/incompatible-depth and four novel-place negatives
  produced **zero new connected optimized nodes** and no accepted recovery.
  Temporary raw query IDs are not global mapped nodes.
- External poses returned with a +100m origin, then changed to -100m without
  relying on a lost label. Their relation to the original world stayed aligned.
  Maximum saved 3D error across positive frames was **0.0713m**. A corrupt-JPEG
  external-only frame mapped safely before the following reset; paired VO/native
  anchors were retained. Three pose-less RGB/depth-envelope gaps added no map nodes.
- Final native optimized graph contained **117 nodes**; original root remained
  `(7,-3,0.5)`. Maximum saved linear constraint ratio **1.646**, absolute **0.00570m**.
  Info counters were 37 global +3 proximity; their sum40 is not the acceptance test.
- Actual saved `.mapping` preview: 10,000 points, marker XY difference from final
  optimized pose **0**, known-surface median residual **0.0155m**. Reopened archive
  preview bytes were identical. Live mapGraph retained the original root.
- Optional preparation kept the **20ms** budget: **131/132 usable images** published,
  **1 timeout**. Last64 preparation samples p95 **16.74ms**, max **19.17ms**.
  This window does not erase the earlier timeout or claim a hard real-time bound.

## Saved-map cold recovery, repeated twice

Each run privately loaded the new ROS archive, then received 16 observations:
the first ten external poses were null; frames4/5 stopped depth/cloud delivery
while RGB continued; later external poses used a new +100m origin.

Both independent runs matched historical node2 at frame1 (new ref141), with
**646 visual inliers**, persisted global connection and no external pose. Both
grew the original **117 ->130 optimized nodes**, including **6 newly tracked
VO-only nodes**. Depth gaps emitted no current Info and no synthetic/global nodes;
geometry return resumed tracking. Max 3D error was **0.0261m /0.0401m**. Root and
archive roundtrip were preserved; maximum final linear ratios **2.330 /1.826**.

Cold-start-to-recovery monotonic wall times were **4.449s /4.430s**, including
runtime startup; successful observation-to-Info **678ms /555ms**. Both had one
initial optional-image timeout among14 metric observations (13 images published).
Preparation maxima **47.25ms /69.75ms** show remaining cold-start tails; the scan
was forwarded within its existing wait budget and recovery used the next image.
This is not evidence of zero latency tails. Saved preview markers matched final
optimized XY exactly; surface medians were **0.0080m /0.0135m**.

The final native low-memory test reduced local graph to40 nodes and still saved
61 connected optimized nodes (raw DB62). This verifies finalization includes
evicted historical geometry without treating an ignored orphan as mapped.

## Regressions and rejected experiments

Failing regressions preceded fixes for optimized-vs-raw geometry/scan transform,
final marker, silent-odom image ownership, inflight JPEG decode, paired VO/native
anchor across an RGB gap, bound-map rename/revision identity, disk-copy budget
and missing native capability before segment registration. The focused real core
origin-reset/RGB-gap error changed from about100m to0.018m.

Earlier cold replay `ros-resume-timed-rejected` failed checked graph attachment:
reoptimization with a new constraint produced a 27sigma error on a proximity
edge, although that edge's saved residual had been near zero. The map was retained
and unlocated additions were paused. Disabling image proximity alone also failed
repeated recovery and was discarded. The native covariance-weight bound was then
validated using a newly generated map and two cold restarts. It does not modify
old archived constraints; recovery of every historical/ill-conditioned archive
is not established. No threshold was reduced to accept the rejected graph.

## Commands and evidence

Evidence root (outside Git, databases retained locally):
`/Users/4paradigm/Desktop/shixi/diagnostics/slam-visual-recovery-20261117`.

| Evidence under that root | Contents |
|---|---|
| `core-cov-evaluation.json`, `core/{rgb-cov,no-rgb-v1,recovery-cov}.{db,jsonl}` | Actual core comparison/recovery |
| `ros-cov-evaluation.json`, `ros-cov-final/ros-cov-final/` | Final full gateway/native Info, graph, log, DB, params, archive |
| `ros-cov-resume-evaluation.json`, `ros-cov-resume/ros-cov-resume/` | Cold continuation with source interruption |
| `ros-cov-resume-repeat-evaluation.json`, `ros-cov-resume-repeat/ros-cov-resume-repeat/` | Independent cold repeat |
| `core/ltm-cov.{db,jsonl}` | Native finalization after memory eviction |
| `reset-gap-{red,green}.jsonl` | Real core failing/passing paired-anchor regression |
| `build-ros-cov.log`, `perception-cov-final.log` | Build and full test outputs |
| `evidence-manifest.json`, `rgb-comparison.png` | SHA256/SQLite checks and plotted geometry comparison |

Final Python command (Python3.12 venv, NumPy/OpenCV/PyYAML installed):

```bash
AGENT_CORE_URL=https://127.0.0.1:1 /tmp/slam-visual-recovery-venv/bin/python \
  -m pytest -q perception/tests
# 3530 passed, 46 skipped, 6 warnings, 71 subtests; 188.58s
```

The isolated URL prevents existing TTS tests from POSTing completion callbacks to
the local app; ordinary execution reproduced three unrelated callback-timeout
failures. TTS code was not modified. Focused final adapter/gateway/snapshot/library
suite: **75 passed**. Native build, all three final ROS geometry evaluators,
core evaluator, SQLite quick checks, syntax checks and `git diff --check` passed.
Native children exited cleanly before evidence copying.

## Not verified / remaining limits

No moving Bumi/Tianyi/other robot, real recorded-route dataset, actual canvas
browser rendering, updated Jetson image build/deployment, rotational external
origin-reset replay, long-duration resource soak or exhaustive perceptual-alias
test was performed. Mathematical SE(3) handling and format compatibility do not
constitute those physical validations. Synthetic texture, overlap and depth are
more controlled than real sensors. Cold preparation tails and native optimization
rejections remain observable failure modes. Failed geometry retains the map and
pauses unlocalized growth, rather than promising a recovery.

During pose loss, different RGB/cloud numerical times still require unavailable
inter-capture motion; projection reports inability instead of fabricating it.
Coincident captures and supplied numerical mounts work. Physical timing,
rectification, calibration and coordinate conventions remain provider duties;
source/status/frame/clock/time-semantics labels do not become input gates.
