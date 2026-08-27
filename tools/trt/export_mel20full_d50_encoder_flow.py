#!/usr/bin/env python3
"""
Export Mel20fullD50 model Encoder+Duration and Flow to ONNX.

Encoder: ph, to, la, xl → m_p, logs_p, logw, x_mask
Flow:     z_p, y_mask → z

Encoder/Flow weights are frozen from v8 base (G_final_4gpu.pth),
so they should be identical to v8 outputs.
"""

import sys, os, types, json
import numpy as np
import torch
import torch.nn as nn

# ── monotonic_align fallback ──
def _maximum_path(value, mask, max_neg_val=-np.inf):
    dtype = value.dtype; value = value.astype(np.float64); mask = mask.astype(np.float64)
    B, Tx, Ty = value.shape
    Q = np.full((B, Tx, Ty), max_neg_val, np.float64); Q[:,0,0] = value[:,0,0]
    for t in range(1, Ty): Q[:,0,t] = Q[:,0,t-1] + value[:,0,t] if mask[:,0,t] else max_neg_val
    for tx in range(1, Tx): Q[:,tx,0] = value[:,tx,0] + max(Q[:,tx-1,0], max_neg_val if not mask[:,tx,0] else Q[:,tx-1,0])
    for tx in range(1, Tx):
        for ty in range(1, Ty):
            if mask[:,tx,ty]: Q[:,tx,ty] = value[:,tx,ty] + max(Q[:,tx-1,ty-1], Q[:,tx-1,ty])
    path = np.zeros((B, Tx, Ty), np.float64); path[:,Tx-1,Ty-1] = 1.0
    for tx in range(Tx-1, -1, -1):
        for ty in range(Ty-1, -1, -1):
            if tx == 0 and ty == 0: continue
            if tx == 0: path[:,tx,ty-1] = 1.0
            elif ty == 0: path[:,tx-1,ty] = 1.0
            else:
                best = np.argmax(np.array([Q[:,tx-1,ty-1], Q[:,tx-1,ty]]), axis=0)
                for b in range(B): path[b,tx-1,ty-(1 if best[b]==0 else 0)] = 1.0
    return path.astype(dtype)

ma = types.ModuleType("monotonic_align"); ma.maximum_path = _maximum_path
sys.modules["monotonic_align"] = ma

sys.path.insert(0, "/home/zhangjinghong/vits2-mix")
from vits2 import models
from vits2.text import symbols
import vits2.utils as vits2_utils

# ── Config ────────────────────────────────────────────────────────────
MODEL_PATH = "/mnt/disk1/zhangjinghong/vits2_finetune/G_v12_bznsyp_mel20full_d50_ep50_merged.pth"
CONFIG_PATH = "/home/zhangjinghong/vits2-mix/config.json"
OUTDIR = "/mnt/disk1/zhangjinghong/trt_engines_mel20full_d50"
os.makedirs(OUTDIR, exist_ok=True)

# ── Load model ────────────────────────────────────────────────────────
print("Loading Mel20fullD50 model...")
hps = vits2_utils.get_hparams_from_file(CONFIG_PATH)
# Override for Mel20fullD50 model
hps.model['gen_istft_n_fft'] = 128
hps.model['gen_istft_hop_size'] = 4

net = models.SynthesizerTrn(
    len(symbols), hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=1, mas_noise_scale_initial=0.01,
    noise_scale_delta=2e-6, **hps.model)

ckpt = torch.load(MODEL_PATH, map_location="cpu")
if "model" in ckpt: ckpt = ckpt["model"]
ms = net.state_dict()
for k, v in ckpt.items():
    if k in ms and ms[k].shape == v.shape: ms[k] = v
net.load_state_dict(ms, strict=False)
net.eval()
print(f"Model loaded: {sum(p.numel() for p in net.parameters()):,} params")
print(f"  gen_istft_n_fft={net.dec.gen_istft_n_fft}, hop={net.dec.gen_istft_hop_size}")

# ── 1. Export Encoder + Duration ─────────────────────────────────────
print("\n=== Exporting Encoder+Duration ===")

class EncoderDPWrapper(nn.Module):
    """TextEncoder + DurationPredictor → m_p, logs_p, logw, x_mask"""
    def __init__(self, net):
        super().__init__()
        self.enc_p = net.enc_p
        self.dp = net.dp
        self.emb_g = net.emb_g

    def forward(self, ph, to, la, xl):
        # Speaker embedding g (same as net.infer())
        g = self.emb_g(torch.zeros(1, dtype=torch.long, device=ph.device)).unsqueeze(-1)
        # TextEncoder WITH speaker conditioning (matches net.infer())
        x, m_p, logs_p, x_mask = self.enc_p(ph, xl, to, la, None, None, g=g)
        # DurationPredictor WITH speaker conditioning
        logw = self.dp(x, x_mask, g=g)
        return m_p, logs_p, logw, x_mask

enc_wrapper = EncoderDPWrapper(net)
enc_wrapper.eval()

# Test with dummy input
T = 50
test_ph = torch.randint(0, 112, (1, T), dtype=torch.int32)
test_to = torch.zeros((1, T), dtype=torch.int32)
test_la = torch.zeros((1, T), dtype=torch.int32)
test_xl = torch.tensor([T], dtype=torch.int32)

with torch.no_grad():
    m_p, logs_p, logw, x_mask = enc_wrapper(test_ph, test_to, test_la, test_xl)
    print(f"  Encoder output: m_p={m_p.shape}, logs_p={logs_p.shape}, logw={logw.shape}, x_mask={x_mask.shape}")

encoder_onnx_path = f"{OUTDIR}/encoder_duration.onnx"
torch.onnx.export(
    enc_wrapper,
    (test_ph, test_to, test_la, test_xl),
    encoder_onnx_path,
    input_names=["ph", "to", "la", "xl"],
    output_names=["m_p", "logs_p", "logw", "x_mask"],
    dynamic_axes={
        "ph": {0: "B", 1: "T_phone"},
        "to": {0: "B", 1: "T_phone"},
        "la": {0: "B", 1: "T_phone"},
        "xl": {0: "B"},
        "m_p": {0: "B", 2: "T_phone"},
        "logs_p": {0: "B", 2: "T_phone"},
        "logw": {0: "B", 2: "T_phone"},
        "x_mask": {0: "B", 2: "T_phone"},
    },
    # NOTE: encoder 必须用 opset 16 —— opset 17 会导出融合算子 LayerNormalization，
    # 而 TRT 8.5.x（JP5）的 ONNX parser 不支持该算子，会报
    # "No importer registered for op: LayerNormalization"。opset 16 会把它拆成
    # ReduceMean/Sub/Pow/Sqrt/Div 原语，TRT 8.5 与 10.x 都能解析。
    opset_version=16,
    do_constant_folding=True,
)
size_mb = os.path.getsize(encoder_onnx_path) / 1024 / 1024
print(f"  ✅ encoder_duration.onnx ({size_mb:.1f} MB)")

# ── 2. Export Flow ────────────────────────────────────────────────────
print("\n=== Exporting Flow ===")

# Get speaker embedding for flow
g_flow = net.emb_g(torch.tensor([0])).unsqueeze(-1).detach() if net.n_speakers > 0 else None

class FlowONNX(nn.Module):
    def __init__(self, net, g):
        super().__init__()
        self.flow = net.flow
        self.g = g

    def forward(self, z_p, y_mask):
        z = self.flow(z_p, y_mask, g=self.g, reverse=True)
        return z

flow_wrapper = FlowONNX(net, g_flow)
flow_wrapper.eval()

test_z_p = torch.randn(1, 256, 100)
test_y_mask = torch.ones(1, 1, 100)

with torch.no_grad():
    z = flow_wrapper(test_z_p, test_y_mask)
    print(f"  Flow output: z={z.shape}")

flow_onnx_path = f"{OUTDIR}/flow.onnx"
torch.onnx.export(
    flow_wrapper,
    (test_z_p, test_y_mask),
    flow_onnx_path,
    input_names=["z_p", "y_mask"],
    output_names=["z"],
    dynamic_axes={
        "z_p": {0: "B", 2: "T_frame"},
        "y_mask": {0: "B", 2: "T_frame"},
        "z": {0: "B", 2: "T_frame"},
    },
    opset_version=17,
    do_constant_folding=True,
)
size_mb = os.path.getsize(flow_onnx_path) / 1024 / 1024
print(f"  ✅ flow.onnx ({size_mb:.1f} MB)")

# ── Verify with onnxruntime ───────────────────────────────────────────
print("\n=== Verifying with onnxruntime ===")
try:
    import onnxruntime as ort

    # Verify encoder
    sess = ort.InferenceSession(encoder_onnx_path, providers=['CPUExecutionProvider'])
    ort_out = sess.run(None, {
        "ph": test_ph.numpy(), "to": test_to.numpy(),
        "la": test_la.numpy(), "xl": test_xl.numpy()
    })
    for name, arr, pt in zip(["m_p", "logs_p", "logw", "x_mask"], ort_out,
                             [m_p, logs_p, logw, x_mask]):
        diff = np.abs(arr - pt.numpy()).max()
        print(f"  Encoder {name}: max diff vs PT = {diff:.8f}")

    # Verify flow
    sess = ort.InferenceSession(flow_onnx_path, providers=['CPUExecutionProvider'])
    ort_z = sess.run(None, {"z_p": test_z_p.numpy(), "y_mask": test_y_mask.numpy()})[0]
    diff = np.abs(ort_z - z.numpy()).max()
    print(f"  Flow z: max diff vs PT = {diff:.8f}")

    print("  ✅ ONNX Runtime verification passed!")
except Exception as e:
    print(f"  ⚠️  onnxruntime verify failed: {e}")

print(f"\n=== Done ===")
print(f"Files in {OUTDIR}:")
for f in sorted(os.listdir(OUTDIR)):
    sz = os.path.getsize(os.path.join(OUTDIR, f)) / 1024 / 1024
    print(f"  {f} ({sz:.1f} MB)")
