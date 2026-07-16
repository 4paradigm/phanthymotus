# OCR Large Image Timeout Implementation Plan

> **For AI agent workers:** Required sub-skill: use superpowers:test-driven-development to execute each task. Track progress with the checkboxes below.

**Goal:** Reliably process large OCR images while bounding Jetson memory and preserving source-image bbox coordinates.

**Architecture:** Fix DDS delivery at the ROS boundary, bound decoded image size in the RapidOCR adapter, and make plugin configuration and node lifecycle resource-safe. Keep the existing MCP and ROS message contracts.

**Tech Stack:** Python 3, ROS2/rclpy, Fast DDS, OpenCV, RapidOCR, ONNX Runtime, unittest.

---

### Task 1: Reliable bounded ROS transport

**Files:**
- Modify: `perception/plugins/ocr.py`
- Modify: `perception/config/fastdds_large_message.xml`
- Test: `perception/tests/test_ocr_contract.py`
- Test: `perception/tests/test_ocr_packaging.py`

- [ ] Change the camera and result QoS tests to require `RELIABLE` and depth 1.
- [ ] Run the focused tests and confirm they fail against best-effort/depth 10.
- [ ] Set both QoS profiles to reliable keep-last-one.
- [ ] Change the packaging test to require built-in transports to remain enabled.
- [ ] Update the Fast DDS profile and confirm focused tests pass.

### Task 2: Bounded decode with source-coordinate boxes

**Files:**
- Modify: `perception/plugins/ocr_runtime.py`
- Modify: `perception/plugins/ocr.py`
- Modify: `perception/config.yaml`
- Test: `perception/tests/test_ocr_contract.py`

- [ ] Add tests for reduced JPEG decode, non-JPEG resize, and bbox scaling.
- [ ] Run the tests and confirm the new API is missing.
- [ ] Add a configurable `max_side_len` adapter parameter with a default of 1600.
- [ ] Select OpenCV reduced JPEG decode flags before allocation where possible.
- [ ] Resize remaining oversized images and map normalized boxes to source pixels.
- [ ] Serialize calls to the RapidOCR engine.
- [ ] Run focused tests and confirm they pass.

### Task 3: Adapter and node lifecycle

**Files:**
- Modify: `perception/plugins/ocr.py`
- Test: `perception/tests/test_ocr_contract.py`

- [ ] Add tests proving empty/equivalent config reuses the adapter.
- [ ] Add tests proving stopped nodes are removed and destroyed.
- [ ] Add a test requiring a one-frame queue.
- [ ] Run tests and confirm failures describe current rebuilding/leaking behavior.
- [ ] Normalize and retain shared config, rebuilding only for model-affecting changes.
- [ ] Centralize node teardown, clear pending frames, and destroy stopped nodes.
- [ ] Run focused tests and confirm they pass.

### Task 4: OCR leaderboard resource defaults

**Files:**
- Modify: `perception/config.yaml`
- Test: `perception/tests/test_ocr_packaging.py`

- [ ] Change packaging expectations to ASR disabled, OCR thread count 1, and max side 1600.
- [ ] Run the test and confirm it fails against current defaults.
- [ ] Apply the OCR-only resource defaults.
- [ ] Run packaging tests and confirm they pass.

### Task 5: Verification and local commit

**Files:**
- Review all changed files.

- [ ] Run all perception unit tests in the isolated dependency environment.
- [ ] Compile changed Python modules with `py_compile`.
- [ ] Inspect the complete diff for model files, unrelated changes, and contract drift.
- [ ] Commit the verified changes locally without pushing.
