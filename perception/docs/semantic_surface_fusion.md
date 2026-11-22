# Semantic Surface Fusion

Enable `plugins.semantic_surface_fusion` in Perception and refresh the Dashboard tool catalog. The card consumes the existing segmentation outputs and the camera's registered depth. It uses the Perception bundle's ROS executor and a bounded CPU worker per card instance.

## Canvas wiring

1. Connect the camera's RGB JPEG to a `segment_*` preset.
2. Connect that instance's `mask` (`image/png`) and `classes` (`data/json`) outputs to `semantic_surface_fusion`.
3. Connect the corresponding registered `depth` (`image/depth-zlib`) camera output. Start the project so the Dashboard forwards `motus.camera/1` declarations. The camera must supply matching rectified pinhole K and dimensions for its RGB and registered depth, the same nonempty camera ID, and `vendor.depth_aligned_to: color` on depth. Restart the project after a calibration change or after camera intrinsics become available.
4. Connect the fusion `state` (`data/json`) to Monitor. Set `max_points` to a positive value in card configuration to expose the additional ROS `sensor_msgs/msg/PointCloud2` output. Inspect it with a PointCloud2 subscriber or RViz using its optical frame as fixed frame if no corresponding TF exists.

The mask is a grayscale 8-bit PNG with background ID 0. The class table matches the mask's **exact** RGB stamp and frame ID. Depth uses the Bumi driver wire encoding `16UC1; compressedDepth zlib`, little-endian Z16. The default scale is 0.001 metres per unit. Set the camera's actual scale if it differs. A nonempty depth frame ID must match the RGB frame ID exactly. For drivers that omit both headers' frame IDs, the card requires sibling `/color` and `/depth` source topics under the same camera namespace. It labels the shared coordinate system `<camera namespace with slashes replaced by underscores>_color_optical_frame`. This label does not create a TF transform.

For direct MCP calls use `semantic_surface_fusion` with `action: start`, a stable `instance_id`, the three full topic names in `input_topics`, and `camera_info` keyed by the mask and depth topic names. `info` returns the resolved ports and settings. `stop` releases that instance's ROS resources and worker. `config` validates settings for the next start. Defaults are 25 ms depth sync tolerance, 1500 ms stale age, 200 ms processing timeout, 0.001 m depth scale and zero output points. The runtime assumes compatible ROS capture clocks and increasing timestamps. Replay must publish fresh stamps or advance the same clock used by the card.

## Output

The JSON state reports `valid`, `degraded`, `insufficient_evidence` or `stale`, with `reason` when quality is limited or evidence is unavailable. A synchronized frame includes an `observations` list with one entry per foreground class, including empty classes. The top-level geometry is the highest-quality class observation, with ties in class-table order. Each observation has `class_name`, `camera_id`, `frame_id`, capture `timestamp`, `visible_center_m`, `visible_bounds_m`, `valid_depth_fraction`, `point_count` and `quality`. Bounds contain per-axis 2nd and 98th percentile endpoints and their difference. The center is the per-axis median. The valid-depth fraction counts finite in-range depths before robust outlier rejection. The point count is the number of inliers before optional downsampling. Depth is accepted from 0.15 m through 10 m. Geometry requires at least 8 inliers covering 10% of the mask. Good quality requires at least 32 inliers covering 60%.

Geometry uses optical axes (+X right, +Y down, +Z forward). It describes an observed surface of a semantic class. Separate instances of the same class can contribute to a single estimate. The card computes no hidden volume, physical full-object dimensions or world-frame position. The optional cloud concatenates per-class sampled XYZ points under a single frame and capture stamp.

The worker accepts up to 1920 脳 1080 pixels and at most 20,000 output points. A bounded per-stream cache retains matching candidates while segmentation catches up. Only the newest complete candidate is scheduled when the worker is available. Unavailable records contain null geometry. Timeout suppresses late geometry. The worker checks the deadline between classes and finishes the current bounded native operation before shutdown. Dense 1080p input may need a higher processing timeout than the default.

## Verification

Run from `perception/` with NumPy, OpenCV, PyYAML and pytest installed. Source the ROS 2 environment for actual DDS tests.

```sh
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
PYTHONPATH=.:$PYTHONPATH python3 -m pytest --noconftest -q tests/test_semantic_surface_fusion.py
PYTHONPATH=.:$PYTHONPATH python3 -m pytest -q tests/test_semantic_masks.py
PYTHONPATH=.:$PYTHONPATH python3 -m pytest -q tests/test_measure_geometry.py tests/test_measure_card.py tests/test_measure_view.py
PYTHONPATH=.:$PYTHONPATH python3 tools/benchmark_surface_fusion.py
```

Run the suites in separate Python processes. The fusion suite disables the shared conftest so it does not install ROS stubs. The fusion ROS test carries the real message types over DDS, including synthetic masks and depth. For full acceptance, replay synchronized RGB and registered depth with topic declarations, start with the existing segmentation model available, inspect JSON and cloud, interrupt each input, and confirm stale output and graceful recovery. Stop the project and verify topic endpoints disappear. This procedure requires no robot motion.

The existing `perception/Dockerfile.jetson` copies the plugin directory and config and supplies ROS, OpenCV and NumPy. From the repository root, use the existing `deploy/build_perception.sh --jp-version 5.11` or `--jp-version 6.1` with registry credentials unset for a local-only build. On x86, use a Buildx builder with ARM64 support.

### Class surfaces and instance attribution

The upstream mask has one class ID per pixel. Each observation declares
`geometry_scope: class_visible_surface` and `instance_attribution: unknown`.
Even with a single connected visible region, the reported center describes
the class surface. Two substantial disconnected regions trigger
`degraded` with `reason: multiple_class_regions` and a null center.
Bounds and optional cloud still describe the class collection. Occlusion
can split a single object, and touching objects can form a single region.
Connectivity therefore supplies evidence of spatial ambiguity only.

### Local browser and model-backed acceptance

The Perception change is based on segmentation PR #407 at `a328a1e`.
For the browser acceptance below, pair it with Core's
`codex/monitor-png-and-latest-fields` at `e91ba05`, based on upstream main.
That Core change provides PNG display and removes stale nested values when
the parent field is replaced. The ROS fusion path can run independently.

For isolated browser testing, run the matching Dashboard in a disposable
ROS 2 Humble environment with loopback-only DDS. Expose its HTTPS port
15678 on host loopback 15729 and leave production credentials and model
caches out of this fixture. In the same network namespace, start:

```sh
export PYTHONPATH=/perception
source /opt/ros/humble/setup.bash
python3 /perception/tools/fusion_dashboard_fixture.py
```

The fixture registers a synthetic RGB-D replay tool and a #407 segmentation
card with a deterministic model head. ROS messages, segmentation postprocessing,
fusion, MCP transport, and Dashboard rendering use their actual implementations.
The deterministic head lets the browser test run without a GPU. It supplies no
TensorRT evidence.

With Playwright Core available, drive Edge against the local Dashboard:

```text
node perception/tools/fusion_dashboard_browser.cjs <playwright-core-directory> <edge-executable> <screenshot-directory> [dashboard-url]
```

The default Dashboard URL is `https://127.0.0.1:15729/`. The browser
script wires the cards on the canvas, checks that Monitor decodes the PNG
and displays JSON geometry, interrupts the depth source, then checks
refresh, Monitor WebSocket reconnection, and project stop/start.

For a model-backed walkthrough, use a GPU environment compatible with an
existing YOLOE segmentation engine. A standard deployment can reuse the
pinned bundle in `VOP_MODEL_DIR` and enable `plugins.segment` and
`plugins.semantic_surface_fusion` in the same Perception bundle. Use a
recorded JPEG stream and its registered Z16 depth with fresh synchronized
timestamps and matching pinhole camera declarations.

For a self-contained lab replay, export the paired recording under
`/replay`. Each pair consists of a rectified color JPEG and a NumPy
`.npy` array of registered unsigned 16-bit depth, in matching pixel grids.
The depth unit is configured on the fusion card. Supply `manifest.json`
with the measured optical calibration and a list of matched pairs:

```json
{
  "id": "lab_rgbd",
  "width": 640,
  "height": 480,
  "K": [600, 0, 320, 0, 600, 240, 0, 0, 1],
  "frames": [{"rgb": "0001.jpg", "depth": "0001.npy"}]
}
```

Substitute the recording's measured matrix and dimensions. Start the split
Core Dashboard on HTTPS port 15678. In an isolated Jetson container sharing
the Dashboard network namespace, mount this checkout's `perception/`
directory at `/perception`, the existing engine and its adjacent
`vocab.json` read-only under `/cached`, and the recording at `/replay`.
Use the Perception JetPack image with its supported NVIDIA runtime, and the
same isolated ROS domain and loopback settings as the Dashboard. For
JetPack 5.11 run inside that container:

```sh
export ROS_DOMAIN_ID=96 ROS_LOCALHOST_ONLY=1 PYTHONPATH=/perception
source /opt/ros/humble/install/setup.bash
source /ros_ws/install/setup.bash
python3 -c 'import tensorrt as trt; print(trt.__version__)'
python3 /perception/tools/fusion_dashboard_fixture.py \
  --recording /replay --engine /cached/yoloe-26s-seg.engine
```

This mode uses production TensorRT session creation and segmentation. Its
local-only resolver bypasses the pinned bundle downloader and vocabulary
prefetch. It fails early if the supplied engine or vocabulary is missing.
It republishes
each recorded pair with one fresh ROS timestamp. Use the replay tool
`fusion_test_replay` and a production segmentation preset present in the
engine vocabulary. The deterministic browser script uses a different
fixture preset and its expected numeric assertions are for synthetic input.

1. Search for the replay, matching `segment_*` preset, and fusion cards.
2. Connect RGB JPEG to the segmentation input. Connect its PNG mask and JSON
   classes to fusion. Connect registered depth to fusion. Starting the project
   forwards the camera declarations from the replay and segmentation tools.
3. Start the project and open Monitor. Check the class, frame ID, timestamps,
   center, and bounds against known geometry in the recording.
4. Pause depth while continuing RGB replay. Confirm that Monitor reports stale
   geometry with a null center. Resume depth and check a new valid result.
5. Refresh the browser, stop the project, and restart it. Confirm fresh
   results after reconnect. Keep the selected ROS domain isolated from robots.

Profile the fusion stages with
`PYTHONPATH=. python3 -m cProfile -s cumulative tools/benchmark_surface_fusion.py`.
The hot path keeps full-resolution measurements and exact percentiles.
`max_points` controls only the published point sample count.
