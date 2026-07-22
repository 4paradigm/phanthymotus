# OCR Low-Memory MNN Implementation Plan

> **For AI agent workers:** Required sub-skill: use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task by task. Track work
> with the checkboxes below.

**Goal:** Keep the current OCR interface and PP-OCRv6 tiny accuracy while
replacing full-image OpenCV/Float32 processing with bounded libvips decode and
direct MNN preprocessing.

**Architecture:** Make the change inside the existing OCR runtime and tiled
strategy. MNN owns detector and recognizer inference; RapidOCR utilities remain
responsible for DB postprocessing, CTC decoding, and perspective crops. Models
stay on JuiceFS and are downloaded by `Dockerfile.jetson`.

**Tech stack:** Python 3.10, RapidOCR 3.9.1 utilities, MNN 3.6.0, libvips,
pyvips 3.1.0, ROS2 Humble, `unittest`.

---

## Files

- Modify `perception/plugins/ocr_runtime.py`: MNN sessions, direct uint8
  preprocessing, ORT startup fallback, and ordinary-image inference.
- Modify `perception/plugins/ocr_tiled_strategy.py`: libvips overview and
  sequential region decode while retaining tile merge logic.
- Modify `perception/plugins/ocr.py`: pass backend settings and fail clearly
  when the configured local backend cannot initialize.
- Modify `perception/utils/ocr_model_downloader.py`: explicit MNN file list and
  model-size validation.
- Modify `perception/Dockerfile.jetson`: install MNN/libvips and download MNN
  models.
- Modify `perception/config.yaml`: enable MNN and remove dynamic rejection.
- Modify `perception/tests/test_ocr_contract.py`: runtime and plugin behavior.
- Modify `perception/tests/test_ocr_tiled_strategy.py`: bounded libvips decode.
- Modify `perception/tests/test_ocr_model_downloader.py`: MNN bundle download.
- Modify `perception/tests/test_ocr_packaging.py`: Docker/config assertions.
- Delete `perception/plugins/ocr_memory_guard.py`: remove request rejection.
- Delete `perception/tests/test_ocr_memory_guard.py`: remove obsolete guard
  tests.

## Task 1: Model Bundle and Jetson Image

- [ ] **Step 1: Add failing downloader and packaging tests**

Add tests proving that the downloader accepts exactly
`("det.mnn", "rec.mnn", "keys.txt")`, the default config selects MNN, and the
Jetson image installs `libvips42`, `pyvips==3.1.0`, and `MNN==3.6.0`.

```python
def test_downloads_explicit_mnn_bundle(self):
    files = ("det.mnn", "rec.mnn", "keys.txt")
    with mock.patch("utils.ocr_model_downloader.download_file") as download:
        download_model("http://models/mnn", self.output, filenames=files)
    self.assertEqual(
        [call.args[0] for call in download.call_args_list],
        [f"http://models/mnn/{name}" for name in files],
    )
```

- [ ] **Step 2: Run the focused tests and confirm failure**

Run:

```bash
python3 -m unittest \
  perception.tests.test_ocr_model_downloader \
  perception.tests.test_ocr_packaging -v
```

Expected: failures for missing MNN/libvips Docker entries and old ONNX-only
configuration.

- [ ] **Step 3: Update downloader, Dockerfile, and config**

Add CLI support without changing the existing Python API:

```python
parser.add_argument("--filenames", nargs="+", default=list(MODEL_FILES))
download_model(
    args.base_url,
    args.output_dir,
    filenames=tuple(args.filenames),
)
```

Install the bounded decoder and MNN wheel, then download the three MNN assets:

```dockerfile
RUN apt-get install -y --no-install-recommends libvips42
RUN pip3 install --no-cache-dir -i ${PYPI_MIRROR} \
    "pyvips==3.1.0" "MNN==3.6.0"

ARG OCR_MNN_MODEL_BASE_URL=http://172.28.4.81:34567/zengzhitao/embodied-ai/ppocrv6-tiny-mnn
RUN python3 /tmp/ocr_model_downloader.py \
    --base-url "${OCR_MNN_MODEL_BASE_URL}" \
    --output-dir /models/ocr/ppocrv6-tiny-mnn \
    --filenames det.mnn rec.mnn keys.txt
```

Set `backend: mnn`, `model_dir: /models/ocr/ppocrv6-tiny-mnn`,
`use_angle_cls: false`, and remove `memory_guard`, `max_input_mb`, and
`max_decode_mb` from the default OCR config. Keep ORT code available for a
debug image, but do not download ORT models in the leaderboard image.

- [ ] **Step 4: Run packaging tests**

Run the command from Step 2. Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add perception/Dockerfile.jetson perception/config.yaml \
  perception/utils/ocr_model_downloader.py \
  perception/tests/test_ocr_model_downloader.py \
  perception/tests/test_ocr_packaging.py
git commit -m "build(ocr): package MNN runtime and models"
```

## Task 2: Replace Float32 RapidOCR Inference with MNN

- [ ] **Step 1: Add failing MNN runtime tests**

Mock `MNN.Interpreter`, `MNN.CVImageProcess`, and RapidOCR postprocessors. Prove
that only `det.mnn` and `rec.mnn` are loaded, session config uses one CPU thread
with low precision/memory, `CVImageProcess.convert()` receives the address of a
contiguous `uint8` array, and recognition crops run one at a time.

```python
self.assertEqual(loaded_models, ["det.mnn", "rec.mnn"])
self.assertEqual(session_config["backend"], "CPU")
self.assertEqual(session_config["numThread"], 1)
self.assertEqual(session_config["precision"], "low")
self.assertEqual(session_config["memory"], 2)
self.assertEqual(converted_input.dtype, np.uint8)
self.assertEqual(max_live_recognition_batch, 1)
```

Also replace the old test that allowed silent adapter failure:

```python
def test_local_adapter_initialization_failure_is_fatal(self):
    with mock.patch(
        "plugins.ocr._build_ocr_adapter", side_effect=RuntimeError("load failed")
    ):
        with self.assertRaisesRegex(RuntimeError, "load failed"):
            self.ocr.OCRPlugin({"provider": "rapidocr"}, object())
```

- [ ] **Step 2: Run the contract tests and confirm failure**

```bash
python3 -m unittest perception.tests.test_ocr_contract -v
```

Expected: failures because `RapidOCRAdapter` still constructs three ORT engines
and `OCRPlugin` still swallows initialization errors.

- [ ] **Step 3: Add the MNN path inside `ocr_runtime.py`**

Keep the public adapter class and put the small MNN helper in the same file:

```python
class _MNNModelSession:
    def __init__(self, model_path: Path, *, num_threads: int, mean, normal):
        import MNN

        self._mnn = MNN
        self._net = MNN.Interpreter(str(model_path))
        self._session = self._net.createSession({
            "backend": "CPU",
            "numThread": num_threads,
            "precision": "low",
            "memory": 2,
            "power": 0,
        })
        self._input = self._net.getSessionInput(self._session)
        self._process = MNN.CVImageProcess({
            "sourceFormat": MNN.CV_ImageFormat_RGB,
            "destFormat": MNN.CV_ImageFormat_BGR,
            "filterType": MNN.CV_Filter_BILINEAL,
            "mean": (*mean, 0.0),
            "normal": (*normal, 1.0),
        })

    def run_uint8(self, image, shape):
        image = np.ascontiguousarray(image, dtype=np.uint8)
        self._net.resizeTensor(self._input, shape)
        self._net.resizeSession(self._session)
        ptr = image.__array_interface__["data"][0]
        self._process.convert(ptr, image.shape[1], image.shape[0], image.strides[0], self._input)
        self._net.runSession(self._session)
        output = self._net.getSessionOutput(self._session)
        return output.getNumpyData().copy()
```

Use RapidOCR's `DBPostProcess`, `CTCLabelDecode`, and
`get_rotate_crop_image`. Resize detector input to a multiple of 32 as `uint8`;
normalize only inside `CVImageProcess`. For detection use mean
`(123.675, 116.28, 103.53)` and normal
`(0.01712475, 0.017507, 0.017429)`; for recognition use mean
`(127.5, 127.5, 127.5)` and normal
`(0.00784314, 0.00784314, 0.00784314)`. Resize and zero-pad one recognition crop
at a time as `uint8`. Do not import or instantiate `TextDetector`,
`TextRecognizer`, `RapidOCR`, or a classifier on the MNN path.

Add a small warm-up for detector and recognizer during adapter construction.
If MNN construction fails, close partial state and use ORT only when
`fallback_model_dir` exists and `fallback_backend` is configured.

- [ ] **Step 4: Make local adapter initialization fatal**

Remove the broad exception handler from `OCRPlugin.__init__`. A configured
local adapter that cannot load must raise through `main.py`, so Kubernetes sees
a failed container instead of an unavailable OCR service that looks healthy.

- [ ] **Step 5: Run the contract tests**

Run the command from Step 2. Expected: all tests pass, with no test expecting a
classifier or dynamic memory guard.

- [ ] **Step 6: Commit**

```bash
git add perception/plugins/ocr_runtime.py perception/plugins/ocr.py \
  perception/tests/test_ocr_contract.py
git commit -m "feat(ocr): run detector and recognizer with MNN"
```

## Task 3: Bound Decode Memory with libvips

- [ ] **Step 1: Add failing libvips decode tests**

Mock pyvips images so tests do not require the native library on the Mac. Prove
that ordinary images call `thumbnail_buffer(..., 960)`, large images decode one
1280-pixel region at a time, and no list stores decoded tile arrays.

```python
decoder.thumbnail.assert_called_once_with(image_bytes, 960)
self.assertLessEqual(max_materialized_width, 1280)
self.assertLessEqual(max_materialized_height, 1280)
self.assertEqual(max_simultaneous_tiles, 1)
```

Keep the existing coordinate, deterministic tile selection, and deduplication
tests unchanged.

- [ ] **Step 2: Run tiled and contract tests and confirm failure**

```bash
python3 -m unittest \
  perception.tests.test_ocr_tiled_strategy \
  perception.tests.test_ocr_contract -v
```

Expected: new tests fail because both paths still use OpenCV full decode.

- [ ] **Step 3: Replace decode internals in existing files**

For ordinary images, use `pyvips.Image.thumbnail_buffer()` and materialize only
the bounded RGB result:

```python
overview = pyvips.Image.thumbnail_buffer(
    image_bytes,
    max_side_len,
    size="down",
    no_rotate=False,
    fail_on="error",
)
image = np.ascontiguousarray(overview.numpy(), dtype=np.uint8)
```

For a large image, create one lazy source image with
`pyvips.Image.new_from_buffer(image_bytes, "", access="sequential")`. Build the
960-side overview from that source. For each selected source-coordinate tile,
call `crop(left, top, width, height)`, materialize that tile, infer it, and drop
the array before the next tile. Retain the existing offset, scale, dedup, and
reading-order code.

Delete calls to `OCRMemoryGuard`, then delete
`perception/plugins/ocr_memory_guard.py` and
`perception/tests/test_ocr_memory_guard.py`. Valid image dimensions never
produce `ImageTooLargeError`; corrupt or unsupported bytes still produce a
completed error payload.

- [ ] **Step 4: Run OCR tests**

```bash
python3 -m unittest discover -s perception/tests -p 'test_ocr*.py' -v
```

Expected: all OCR tests pass.

- [ ] **Step 5: Commit**

```bash
git add perception/plugins/ocr_runtime.py \
  perception/plugins/ocr_tiled_strategy.py \
  perception/plugins/ocr_memory_guard.py \
  perception/tests/test_ocr_contract.py \
  perception/tests/test_ocr_tiled_strategy.py \
  perception/tests/test_ocr_memory_guard.py
git commit -m "fix(ocr): bound image decode memory with libvips"
```

## Task 4: Verify on Jetson and Prepare Submission

- [ ] **Step 1: Convert and place MNN models on JuiceFS**

Run on the internal model host where the ONNX files are readable:

```bash
python3 -m pip install "MNN==3.6.0"
mkdir -p /mnt/data/zengzhitao/embodied-ai/ppocrv6-tiny-mnn
mnnconvert -f ONNX --keepInputFormat \
  --modelFile /mnt/data/zengzhitao/embodied-ai/ppocrv6-tiny/det.onnx \
  --MNNModel /mnt/data/zengzhitao/embodied-ai/ppocrv6-tiny-mnn/det.mnn \
  --bizCode ppocrv6-tiny-det
mnnconvert -f ONNX --keepInputFormat \
  --modelFile /mnt/data/zengzhitao/embodied-ai/ppocrv6-tiny/rec.onnx \
  --MNNModel /mnt/data/zengzhitao/embodied-ai/ppocrv6-tiny-mnn/rec.mnn \
  --bizCode ppocrv6-tiny-rec
cp /mnt/data/zengzhitao/embodied-ai/ppocrv6-tiny/keys.txt \
  /mnt/data/zengzhitao/embodied-ai/ppocrv6-tiny-mnn/keys.txt
```

Confirm each file can be read through port 34567 before building.

- [ ] **Step 2: Run the complete local test suite**

```bash
python3 -m unittest discover -s perception/tests -p 'test_*.py' -v
```

Expected: zero failures and zero errors.

- [ ] **Step 3: Build the Jetson image without cache**

```bash
docker build --no-cache --network=host \
  -f perception/Dockerfile.jetson \
  -t phanthymotus-perception-ocr:mnn .
```

Expected: dependency import checks, model download, and model-size check pass.

- [ ] **Step 4: Compare runtime behavior**

Run the same ordinary, 4000 by 3000, receipt, and empty images used for the ORT
baseline. Record startup RSS, peak RSS, latency, output text, and boxes. Then
run 250 sequential cases and the leaderboard container concurrency.

Pass conditions:

- no case is rejected for memory headroom or dimensions;
- no exit 137, restart, or 120-second incomplete response;
- aggregate concurrent RSS stays below 80 percent of 8 GB;
- model assets stay at or below 15 MiB;
- sample F1 is not below the existing Tiny baseline.

- [ ] **Step 5: Commit any Jetson-only compatibility fixes separately**

```bash
git add perception perception/tests
git commit -m "fix(ocr): align MNN runtime with Jetson"
```

Skip this commit when the worktree is already clean.
