# OpenCV Container Dependencies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make both perception container images select a compatible OpenCV package and fail during construction when `cv2` is unavailable or shadowed.

**Architecture:** The CPU image pins a headless PyPI OpenCV wheel and validates it after all dependency installation. The Jetson image uses Ubuntu's `python3-opencv`, removes every PyPI OpenCV variant and stale `/usr/local` files, preserves the existing Jetson loader workaround, and validates that the final module is not loaded from `/usr/local`.

**Tech Stack:** Dockerfile, Python 3.8, OpenCV 4.7.0.72, JetPack 5 / Ubuntu 20.04

## Global Constraints

- CPU OpenCV package must be `opencv-python-headless==4.7.0.72`.
- Jetson OpenCV must come from the system package `python3-opencv`.
- Jetson must remove all four PyPI OpenCV variants and stale `/usr/local/lib/python3.8/dist-packages` OpenCV files.
- Both builds must execute a final `import cv2` validation after Python dependencies are installed.
- Jetson validation must fail if `cv2.__file__` starts with `/usr/local/`.
- Do not change OCR implementation code or unrelated perception dependencies.

---

### Task 1: Pin and Validate CPU OpenCV

**Files:**
- Modify: `perception/Dockerfile`

**Interfaces:**
- Consumes: Python 3 and pip from `ROS_BASE_IMAGE`.
- Produces: A CPU perception image with importable headless OpenCV 4.7.0.72.

- [ ] **Step 1: Run the static requirement check and verify it fails**

Run:

```bash
rg -q 'opencv-python-headless==4\\.7\\.0\\.72' perception/Dockerfile &&
rg -q 'import cv2; print.*cv2.__version__.*cv2.__file__' perception/Dockerfile
```

Expected: non-zero exit because the version is not pinned and no final validation exists.

- [ ] **Step 2: Pin the CPU package and add final validation**

Change the OCR dependency to:

```dockerfile
    onnxruntime pyclipper "opencv-python-headless==4.7.0.72"
```

After all Python dependency installation commands, add:

```dockerfile
# Fail the build early if the OCR OpenCV dependency cannot be loaded.
RUN python3 -c "import cv2; print('OpenCV', cv2.__version__, cv2.__file__)"
```

- [ ] **Step 3: Run the static requirement check and verify it passes**

Run:

```bash
rg -q 'opencv-python-headless==4\\.7\\.0\\.72' perception/Dockerfile &&
rg -q 'import cv2; print.*cv2.__version__.*cv2.__file__' perception/Dockerfile
```

Expected: exit 0.

### Task 2: Harden and Validate Jetson OpenCV

**Files:**
- Modify: `perception/Dockerfile.jetson`

**Interfaces:**
- Consumes: `python3-opencv` from the JP511 Ubuntu package repositories.
- Produces: A Jetson perception image whose `cv2` module is not shadowed by `/usr/local`.

- [ ] **Step 1: Run the static requirement check and verify it fails**

Run:

```bash
rg -q 'opencv-contrib-python-headless' perception/Dockerfile.jetson &&
rg -q 'startswith\\(\"/usr/local/\"\\)' perception/Dockerfile.jetson
```

Expected: non-zero exit because cleanup covers only two PyPI variants and no final path assertion exists.

- [ ] **Step 2: Expand cleanup and add final path validation**

Replace the current PyPI uninstall command with:

```dockerfile
RUN pip3 uninstall -y \
        opencv-python opencv-python-headless \
        opencv-contrib-python opencv-contrib-python-headless || true && \
    rm -rf \
        /usr/local/lib/python3.8/dist-packages/cv2 \
        /usr/local/lib/python3.8/dist-packages/opencv_python*.dist-info \
        /usr/local/lib/python3.8/dist-packages/opencv_contrib_python*.dist-info
```

After all Python dependency installation commands, add:

```dockerfile
# Fail the build if a PyPI cv2 copy still shadows Jetson's system OpenCV.
RUN python3 -c "import cv2; path = cv2.__file__; assert not path.startswith('/usr/local/'), path; print('OpenCV', cv2.__version__, path)"
```

- [ ] **Step 3: Run the static requirement check and verify it passes**

Run:

```bash
rg -q 'opencv-contrib-python-headless' perception/Dockerfile.jetson &&
rg -q "startswith\\('/usr/local/'\\)" perception/Dockerfile.jetson
```

Expected: exit 0.

### Task 3: Verify, Commit, and Push

**Files:**
- Modify: `perception/Dockerfile`
- Modify: `perception/Dockerfile.jetson`

**Interfaces:**
- Consumes: Completed CPU and Jetson Dockerfile changes.
- Produces: A clean commit pushed to `origin/feat_offline_ocr`.

- [ ] **Step 1: Check patch formatting**

Run:

```bash
git diff --check
```

Expected: exit 0 with no output.

- [ ] **Step 2: Review the exact Dockerfile diff**

Run:

```bash
git diff -- perception/Dockerfile perception/Dockerfile.jetson
```

Expected: only the OpenCV pin, Jetson cleanup, and final validation commands differ.

- [ ] **Step 3: Record local build limitation**

Run:

```bash
docker version
```

Expected in the current environment: Docker daemon unavailable. Do not claim either image was built locally.

- [ ] **Step 4: Commit the Dockerfile changes**

Run:

```bash
git add perception/Dockerfile perception/Dockerfile.jetson
git commit -m "fix(perception): validate OpenCV container dependencies"
```

Expected: one commit containing only the two Dockerfiles.

- [ ] **Step 5: Push the branch**

Run:

```bash
git push origin feat_offline_ocr
```

Expected: `origin/feat_offline_ocr` advances through the local commits.
