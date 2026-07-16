# OpenCV Container Dependency Design

## Goal

Ensure both perception images can import OpenCV during image construction and at runtime, while preserving the platform-specific dependency strategy required by CPU and Jetson environments.

## CPU Image

`perception/Dockerfile` will use the headless PyPI package because the CPU image does not need GUI support or Jetson-specific OpenCV integration.

- Pin `opencv-python-headless` to `4.7.0.72` for Python 3.8 and older Linux compatibility.
- Install it explicitly with the other OCR dependencies.
- Run a final `python3 -c` validation after all Python dependencies are installed.
- Print the OpenCV version and module path so build logs identify the selected package.

## Jetson Image

`perception/Dockerfile.jetson` will use the OpenCV package supplied by the Jetson/Ubuntu package repositories.

- Install `python3-opencv` with the system dependencies.
- Uninstall all PyPI OpenCV variants:
  - `opencv-python`
  - `opencv-python-headless`
  - `opencv-contrib-python`
  - `opencv-contrib-python-headless`
- Remove stale `/usr/local/lib/python3.8/dist-packages/cv2` files and matching PyPI metadata that could shadow the system package.
- Preserve the existing Jetson `cv2` initialization workaround.
- Run a final validation after all Python dependencies are installed.
- Require the imported module path to be outside `/usr/local`; fail the image build if a pip-installed copy still shadows the system package.

## Error Handling

Both Docker builds must fail immediately when `cv2` cannot be imported. The Jetson build must additionally fail when `cv2.__file__` resolves under `/usr/local`.

## Verification

- Run repository whitespace validation with `git diff --check`.
- Inspect both Dockerfiles to confirm their dependency and validation commands.
- If a Docker daemon and matching architecture are available, build both images.
- Jetson runtime verification remains required on the target device because the local development machine cannot execute the JP511 ARM64 image.

## Scope

Only OpenCV dependency selection, cleanup, and build-time validation are included. OCR implementation code and unrelated perception dependencies remain unchanged.
