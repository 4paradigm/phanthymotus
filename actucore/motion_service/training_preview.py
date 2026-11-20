"""Latest real training states, rendered outside the training process.

The producer asynchronously copies only its allotted environments at a bounded rate.
It never opens a graphics context. The consumer owns a bounded renderer cache
and uses private MuJoCo data; viewing a job never steps its physics.
"""
from __future__ import annotations

import base64
from collections import OrderedDict
from contextlib import nullcontext
import io
import hashlib
import json
import math
from pathlib import Path
import threading
import queue
import time

from .contracts import identifier, integer, load_model_fresh, model_fingerprint
from .store import atomic_json, file_sha256

PROTOCOL = "motus.training-preview/1"
MAX_SAMPLES = 8
MAX_RANKS = 64
MAX_SNAPSHOT_BYTES = 512 * 1024
STALE_SECONDS = 5.0


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"invalid training preview {name}")
    return float(value)


def _read(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_SNAPSHOT_BYTES + 1)
    if len(raw) > MAX_SNAPSHOT_BYTES:
        raise ValueError("training preview metadata is too large")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("invalid training preview metadata")
    return value


class PreviewRecorder:
    def __init__(self, folder, rank, world_size, max_total=MAX_SAMPLES, *,
                 sample_hz=4.0, history_frames=120, background=True):
        self.world_size = integer(world_size, 1, MAX_RANKS, "world_size")
        self.rank = integer(rank, 0, world_size - 1, "rank")
        self.max_total = integer(max_total, 1, MAX_SAMPLES, "max_total")
        self.sample_hz = _finite(sample_hz, "sample_hz")
        if not 1 <= self.sample_hz <= 8:
            raise ValueError("training preview sample_hz must be between 1 and 8")
        self.history_frames = integer(history_frames, 0, 120, "history_frames")
        self.quota = max_total // world_size + int(rank < max_total % world_size)
        self.path = Path(folder) / "ranks" / f"rank-{rank}" / "preview.json"
        self._next_capture = 0.0
        self.last_error = None
        self._sequence = 0
        self._history = []
        self._background = background
        self._queue = queue.Queue(maxsize=1)
        self._busy = threading.Event()
        self._closed = False
        self._thread = None
        self._staging = None
        self._stats = {"published": 0, "skipped_busy": 0, "capture_total_ms": 0.,
                       "capture_max_ms": 0., "write_max_ms": 0., "cuda_async": False}

    def stats(self):
        return {**self._stats, "sample_hz": self.sample_hz, "history_frames": self.history_frames,
                "max_samples": self.max_total, "error": self.last_error}

    def _publish(self, packet):
        import numpy as np

        began = time.monotonic()
        event = packet.pop("event", None)
        if event is not None:
            # Query on the writer thread. Training never waits for the GPU copy.
            while not event.query():
                time.sleep(.002)
            values = packet.pop("host").numpy().copy()
            packet.pop("device_values", None)  # Keep its storage alive through the copy.
        else:
            values = packet.pop("values")
        count, nq, nv = packet.pop("shape")
        indices = packet.pop("indices")
        if values.shape != (count, nq + nv + 1) or not np.isfinite(values).all():
            raise ValueError("training preview state contains nonfinite values")
        iteration = packet.pop("iteration")
        samples = [{"env_id": env_id, "iteration": iteration, "time": float(row[-1]),
                    "qpos": row[:nq].tolist(), "qvel": row[nq:nq + nv].tolist()}
                   for env_id, row in zip(indices, values, strict=True)]
        self._stats["published"] += 1
        value = {"protocol": PROTOCOL, "rank": self.rank, "world_size": self.world_size,
                 "max_total": self.max_total, "sample_hz": self.sample_hz,
                 "samples": samples, "producer": self.stats(), **packet}
        if self.history_frames:
            slot = value["sequence"] % self.history_frames
            frame_path = self.path.parent / "preview-frames" / f"{slot}.json"
            atomic_json(frame_path, value)
            self._history = [item for item in self._history if item["slot"] != slot]
            self._history.append({"slot": slot, "sequence": value["sequence"],
                                  "iteration": iteration, "captured_at": value["captured_at"]})
            atomic_json(self.path.parent / "preview-history.json",
                        {"protocol": PROTOCOL, "rank": self.rank, "frames": self._history})
        atomic_json(self.path, value)
        self.last_error = None
        self._stats["write_max_ms"] = max(self._stats["write_max_ms"], (time.monotonic() - began) * 1000)

    def _failure(self, exc):
        self.last_error = str(exc)[:500]
        try:
            atomic_json(self.path, {"protocol": PROTOCOL, "rank": self.rank,
                                   "world_size": self.world_size, "captured_at": time.time(),
                                   "error": self.last_error, "samples": []})
        except OSError:
            pass

    def _write_loop(self):
        while True:
            packet = self._queue.get()
            try:
                if packet is None:
                    return
                self._publish(packet)
            except Exception as exc:
                self._failure(exc)
            finally:
                self._busy.clear()
                self._queue.task_done()
            if self._closed and self._queue.empty():
                return

    def capture(self, env, iteration):
        """Sample only completed real environment steps; never step or render.

        CUDA uses one pinned, nonblocking D2H transfer. A single writer owns the
        pending frame; if it is busy, this call skips instead of stalling learning.
        Sampling is an upper bound: world-model updates do not create new physics
        states and therefore cannot generate new preview frames.
        """
        now = time.monotonic()
        if self._closed or not self.quota or now < self._next_capture:
            return False
        if self._busy.is_set():
            self._stats["skipped_busy"] += 1
            return False
        self._busy.set()
        self._next_capture = now + 1 / self.sample_hz
        try:
            import numpy as np

            iteration = integer(iteration, 0, 2**53 - 1, "iteration")
            data = env.unwrapped.sim.data
            qpos, qvel = data.qpos, data.qvel
            if len(qpos.shape) != 2 or len(qvel.shape) != 2 or qpos.shape[0] != qvel.shape[0]:
                raise ValueError("training state must contain batched qpos and qvel")
            num_envs = integer(int(qpos.shape[0]), 1, 1_000_000, "num_envs")
            nq = integer(int(qpos.shape[1]), 1, 512, "nq")
            nv = integer(int(qvel.shape[1]), 1, 512, "nv")
            count = min(self.quota, num_envs)
            indices = [i * num_envs // count for i in range(count)]
            self._sequence += 1
            packet = {"num_envs": num_envs, "captured_at": time.time(),
                      "sequence": self._sequence, "iteration": iteration,
                      "shape": (count, nq, nv), "indices": indices}
            if hasattr(qpos, "detach"):
                import torch

                owner_stream = None
                if qpos.is_cuda:
                    if hasattr(qpos, "wp_array"):
                        # mjlab exposes Warp-backed tensors. Queue gathering and
                        # D2H on their producer stream so both preceding and next
                        # physics steps are ordered without a CPU synchronize.
                        import warp as wp
                        owner_stream = torch.cuda.ExternalStream(
                            wp.get_stream(qpos.wp_array.device).cuda_stream, device=qpos.device)
                    else:
                        owner_stream = torch.cuda.current_stream(qpos.device)
                with torch.cuda.stream(owner_stream) if owner_stream is not None else nullcontext():
                    ids = torch.tensor(indices, device=qpos.device, dtype=torch.long)
                    clock = data.time
                    if not torch.is_tensor(clock):
                        clock = torch.as_tensor(clock, device=qpos.device, dtype=qpos.dtype)
                    clock = clock.reshape(-1)
                    if clock.numel() == 1:
                        times = clock.expand(count)
                    elif clock.numel() == num_envs:
                        times = clock.index_select(0, ids)
                    else:
                        raise ValueError("training clock does not match environment count")
                    packed = torch.cat((qpos.index_select(0, ids), qvel.index_select(0, ids),
                                        times.to(device=qpos.device, dtype=qpos.dtype).reshape(count, 1)), dim=1).detach()
                    if packed.is_cuda:
                        if (self._staging is None or self._staging.shape != packed.shape
                                or self._staging.dtype != packed.dtype):
                            self._staging = torch.empty(packed.shape, dtype=packed.dtype, device="cpu", pin_memory=True)
                        host = self._staging
                        host.copy_(packed, non_blocking=True)
                        event = torch.cuda.Event()
                        event.record(torch.cuda.current_stream(packed.device))
                        packet.update(host=host, event=event, device_values=packed)
                        self._stats["cuda_async"] = True
                    else:
                        packet["values"] = packed.numpy().copy()
            else:
                clock = np.asarray(data.time).reshape(-1)
                if clock.size not in (1, num_envs):
                    raise ValueError("training clock does not match environment count")
                times = np.repeat(clock, count) if clock.size == 1 else clock[indices]
                packet["values"] = np.concatenate((np.asarray(qpos)[indices], np.asarray(qvel)[indices],
                                                    times.reshape(count, 1)), axis=1).copy()
            elapsed = (time.monotonic() - now) * 1000
            self._stats["capture_total_ms"] += elapsed
            self._stats["capture_max_ms"] = max(self._stats["capture_max_ms"], elapsed)
            if self._background:
                if self._thread is None:
                    self._thread = threading.Thread(target=self._write_loop, name=f"preview-writer-{self.rank}", daemon=True)
                    self._thread.start()
                self._queue.put_nowait(packet)
            else:
                self._publish(packet)
                self._busy.clear()
            return True
        except Exception as exc:
            self._busy.clear()
            self._failure(exc)
            return False

    def flush(self, timeout=5):
        deadline = time.monotonic() + timeout
        while self._busy.is_set() and time.monotonic() < deadline:
            time.sleep(.002)
        return not self._busy.is_set()

    def close(self):
        if self._closed:
            return
        self._closed = True
        drained = self.flush()
        if self._thread is not None:
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass  # Writer exits after its pending packet when closed.
            self._thread.join(timeout=1)
        if not drained:
            self.last_error = "training preview writer did not drain before shutdown"
        try:
            atomic_json(self.path.parent / "preview-stats.json", self.stats())
        except OSError as exc:
            self.last_error = str(exc)[:500]


class _PreviewCamera:
    """Fit the articulated robot, excluding fixed terrain and scene bodies."""

    def __init__(self, model):
        import mujoco
        import numpy as np

        self.model = model
        self.data = mujoco.MjData(model)
        roots = np.unique(model.body_rootid[model.jnt_bodyid])
        roots = roots[roots != 0]
        self.geoms = np.flatnonzero(np.isin(model.body_rootid[model.geom_bodyid], roots))
        if not len(self.geoms):
            raise ValueError("训练机器人模型没有可取景的运动几何体")
        from .camera_fit import RobotCameraFit
        self.fit = RobotCameraFit(model, self.geoms)

    def view(self, sample, width, height):
        import mujoco
        import numpy as np

        # This private kinematic copy never steps or changes the training state.
        self.data.qpos[:] = sample["qpos"]
        self.data.qvel[:] = sample["qvel"]
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise ValueError("训练预览状态包含非有限数值")
        mujoco.mj_forward(self.model, self.data)
        return self.fit.view(self.data, width, height)


class TrainingPreview:
    def __init__(self, store):
        self.store = store
        self._lock = threading.Lock()
        self._renderers = OrderedDict()
        self._asset_signatures = OrderedDict()
        self._frame_cache = OrderedDict()
        self._closed = False

    def _renderer(self, profile, width, height):
        import mujoco
        from .rendering import FrameRenderer

        path = Path(profile.get("model_path", ""))
        if not path.is_absolute() or not path.is_file():
            raise ValueError("训练机器人配置没有可用模型")
        if (not profile.get("compiled_model_sha256")
                or profile.get("mujoco_version") != mujoco.__version__):
            raise ValueError("训练机器人模型或 MuJoCo 版本已改变")
        stat = path.stat()
        signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
                     profile.get("model_sha256"))
        # A G1 MJB is ~68 MB. Rehash only after file identity/metadata changes;
        # ctime also catches in-place edits whose mtime was restored.
        if self._asset_signatures.get(str(path)) != signature:
            if file_sha256(path) != profile.get("model_sha256"):
                raise ValueError("训练机器人模型或 MuJoCo 版本已改变")
            self._asset_signatures[str(path)] = signature
            self._asset_signatures.move_to_end(str(path))
            while len(self._asset_signatures) > 2:
                self._asset_signatures.popitem(last=False)
        key = (profile["profile_hash"], str(path), width, height)
        if key in self._renderers:
            self._renderers.move_to_end(key)
            return self._renderers[key]
        model = load_model_fresh(path)
        if model_fingerprint(model) != profile["compiled_model_sha256"]:
            raise ValueError("训练机器人模型资产已改变")
        while len(self._renderers) >= 2:
            oldest = next(iter(self._renderers))
            entry = self._renderers[oldest]
            entry[0].close()
            del self._renderers[oldest]
        camera = _PreviewCamera(model)
        renderer = FrameRenderer(model, max_width=width, max_height=height)
        entry = (renderer, model.nq, model.nv, camera)
        self._renderers[key] = entry
        return entry

    def _history(self, folder, rank):
        path = folder / "ranks" / f"rank-{rank}" / "preview-history.json"
        if not path.exists():
            return []
        value = _read(path)
        frames = value.get("frames")
        if (value.get("protocol") != PROTOCOL or value.get("rank") != rank
                or not isinstance(frames, list) or len(frames) > 120):
            raise ValueError("训练采样回放索引无效")
        seen = set()
        for frame in frames:
            if not isinstance(frame, dict):
                raise ValueError("训练采样回放索引元素无效")
            slot = integer(frame.get("slot"), 0, 119, "history slot")
            integer(frame.get("sequence"), 1, 2**53 - 1, "history sequence")
            _finite(frame.get("captured_at"), "history timestamp")
            if slot in seen:
                raise ValueError("训练采样回放索引重复")
            seen.add(slot)
        return frames

    def render(self, job_id, width=768, height=432, frame=None, sample=None,
               sample_start=None, sample_limit=None):
        """Compose up to eight sampled environments, without advancing physics."""
        job_id = identifier(job_id)
        width = integer(width, 320, 960, "width")
        height = integer(height, 160, 640, "height")
        if frame is not None:
            frame = integer(frame, 0, 119, "frame")
        if sample is not None:
            sample = integer(sample, 0, MAX_SAMPLES - 1, "sample")
            if sample_start is not None or sample_limit is not None:
                raise ValueError("单实例与分组观看不能同时指定")
        if sample_start is not None:
            sample_start = integer(sample_start, 0, MAX_SAMPLES - 1, "sample_start")
        if sample_limit is not None:
            sample_limit = integer(sample_limit, 1, MAX_SAMPLES, "sample_limit")
        if not self._lock.acquire(blocking=False):
            return {"available": False, "reason": "训练预览正在渲染"}
        try:
            began = time.monotonic()
            deadline = began + 5.5
            if self._closed:
                raise RuntimeError("training preview is closed")
            folder = self.store.root / "jobs" / job_id
            request = _read(folder / "request.json")
            if request.get("kind") != "train":
                return {"available": False, "reason": "此任务不是训练任务"}
            saved_profile = request.get("profile") or {}
            profile = self.store.profile(saved_profile.get("id"))
            if saved_profile.get("profile_hash") != profile.get("profile_hash"):
                raise ValueError("训练任务与当前机器人配置不匹配")
            history = self._history(folder, 0)
            if frame is not None and frame >= len(history):
                raise ValueError("训练采样回放帧不存在")
            anchor = history[frame] if frame is not None else None
            snapshots = []
            unavailable = False
            for rank in range(MAX_RANKS):
                path = folder / "ranks" / f"rank-{rank}" / "preview.json"
                expected = None
                if anchor is not None:
                    frames = history if rank == 0 else self._history(folder, rank)
                    if not frames:
                        continue
                    expected = min(frames, key=lambda entry: abs(entry["captured_at"] - anchor["captured_at"]))
                    path = path.parent / "preview-frames" / f"{expected['slot']}.json"
                if not path.is_file():
                    continue
                item = _read(path)
                if expected is not None and item.get("sequence") != expected["sequence"]:
                    return {"available": False, "reason": "训练采样回放窗口正在更新，请重试"}
                if item.get("protocol") != PROTOCOL or item.get("rank") != rank:
                    raise ValueError("训练预览来源无效")
                if item.get("error"):
                    unavailable = True
                    continue
                world_size = integer(item.get("world_size"), 1, MAX_RANKS, "world_size")
                integer(rank, 0, world_size - 1, "rank")
                integer(item.get("num_envs"), 1, 1_000_000, "num_envs")
                maximum = integer(item.get("max_total"), 1, MAX_SAMPLES, "max_total")
                quota = maximum // world_size + int(rank < maximum % world_size)
                samples = item.get("samples")
                if not isinstance(samples, list) or len(samples) > min(quota, item["num_envs"]):
                    raise ValueError("训练预览样本数量无效")
                _finite(item.get("captured_at"), "timestamp")
                snapshots.append(item)
            if not snapshots or not any(item["samples"] for item in snapshots):
                return {"available": False, "reason": "尚无可用训练状态快照"}
            first = snapshots[0]
            for item in snapshots:
                if any(item[key] != first[key] for key in ("world_size", "num_envs", "max_total")):
                    raise ValueError("训练预览各进程配置不一致")
            selected = [(item, sample) for item in snapshots for sample in item["samples"]]
            if len(selected) > MAX_SAMPLES:
                raise ValueError("训练预览样本超过限制")
            sample_count = len(selected)
            sources = [{"rank": item["rank"], "env_id": integer(state.get("env_id"), 0,
                        item["num_envs"] - 1, "env_id")} for item, state in selected]
            if sample is not None:
                if sample >= sample_count:
                    raise ValueError("所选真实训练实例暂不可用")
                visible = [sample]
            else:
                start = sample_start if sample_start is not None else 0
                limit = sample_limit if sample_limit is not None else MAX_SAMPLES
                if start >= sample_count:
                    raise ValueError("所选训练实例分组暂不可用")
                visible = list(range(start, min(sample_count, start + limit)))
            selected = [selected[index] for index in visible]
            # The responsive workbench pages through four portrait samples.
            # Keep that group in one row instead of leaving two empty cells
            # with the square-root layout's three-plus-one arrangement.
            columns = 4 if len(selected) in (4, 8) else min(len(selected), math.ceil(math.sqrt(len(selected) * width / height)))
            rows = math.ceil(len(selected) / columns)
            tile_width, tile_height = width // columns, height // rows
            renderer, nq, nv, camera = self._renderer(profile, tile_width, tile_height)
            frame_id = hashlib.sha256(json.dumps([snapshots, visible], allow_nan=False,
                                                 separators=(",", ":")).encode()).hexdigest()[:24]
            key = (job_id, profile["profile_hash"], width, height, frame_id)
            expected_ranks = min(first["world_size"], first["max_total"])
            captured_at = max(p["captured_at"] for p in snapshots)
            age = max(0., time.time() - captured_at)
            stale = (unavailable or len(snapshots) != expected_ranks
                     or (frame is None and any(time.time() - p["captured_at"] > STALE_SECONDS for p in snapshots)))
            recent = history[-20:]
            span = recent[-1]["captured_at"] - recent[0]["captured_at"] if len(recent) > 1 else 0
            dynamic = {"stale": stale, "age_s": age, "captured_at": captured_at,
                       "sample_hz": first.get("sample_hz", 1.),
                       "sample_count": sample_count, "selected_sample": sample,
                       "visible_sample_indices": visible,
                       "grid_columns": columns, "grid_rows": rows,
                       "sample_sources": sources,
                       "actual_sample_hz": (len(recent) - 1) / span if span > 0 else None,
                       "playback": {"frame_count": len(history), "selected_frame": frame,
                                    "selected_sequence": anchor["sequence"] if anchor is not None else first.get("sequence"),
                                    "first_sequence": history[0]["sequence"] if history else None,
                                    "last_sequence": history[-1]["sequence"] if history else None,
                                    "first_captured_at": history[0]["captured_at"] if history else None,
                                    "last_captured_at": history[-1]["captured_at"] if history else None,
                                    "mode": "replay" if frame is not None else "live"}}
            if key in self._frame_cache:
                self._frame_cache.move_to_end(key)
                return {**self._frame_cache[key], **dynamic, "cached": True, "render_ms": 0.}
            from PIL import Image

            canvas = Image.new("RGB", (width, height), (23, 27, 32))
            metadata = []
            seen = set()
            for i, (item, sample) in enumerate(selected):
                if not isinstance(sample, dict):
                    raise ValueError("训练预览样本无效")
                env_id = integer(sample.get("env_id"), 0, item["num_envs"] - 1, "env_id")
                source = (item["rank"], env_id)
                if source in seen:
                    raise ValueError("训练预览环境重复")
                seen.add(source)
                iteration = integer(sample.get("iteration"), 0, 2**53 - 1, "iteration")
                stamp = _finite(sample.get("time"), "simulation time")
                if len(sample.get("qpos", [])) != nq or len(sample.get("qvel", [])) != nv:
                    raise ValueError("训练预览状态与绑定模型维度不匹配")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("训练预览渲染超时")
                rendered = renderer.render(sample, camera=camera.view(sample, tile_width, tile_height),
                                           width=tile_width, height=tile_height,
                                           jpeg_quality=70, timeout=min(5, remaining))
                with Image.open(io.BytesIO(rendered["data"])) as picture:
                    canvas.paste(picture.convert("RGB"), ((i % columns) * tile_width, (i // columns) * tile_height))
                metadata.append({"rank": item["rank"], "env_id": env_id, "time": stamp,
                                 "iteration": iteration, "captured_at": item["captured_at"]})
            output = io.BytesIO()
            canvas.save(output, format="JPEG", quality=80)
            result = {"available": True, "image": {"mime": "image/jpeg", "width": width,
                    "height": height, "data": base64.b64encode(output.getvalue()).decode("ascii")},
                    "samples": metadata, "total_envs": first["num_envs"] * first["world_size"],
                    "world_size": first["world_size"], "frame_id": frame_id,
                    "render_ms": (time.monotonic() - began) * 1000, "cached": False, **dynamic}
            self._frame_cache[key] = result
            while len(self._frame_cache) > 2:
                self._frame_cache.popitem(last=False)
            return result
        finally:
            self._lock.release()

    def close(self):
        with self._lock:
            self._closed = True
            failure = None
            for key, (renderer, _, _, _) in list(self._renderers.items()):
                try:
                    renderer.close()
                except Exception as exc:
                    failure = failure or exc
                else:
                    del self._renderers[key]
            if failure is not None:
                raise failure
