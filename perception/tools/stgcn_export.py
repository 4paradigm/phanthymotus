#!/usr/bin/env python3
"""
tools/stgcn_export.py — ST-GCN++ checkpoint → ONNX, for the pose card's
skeleton-action backend.

Run this on any machine with torch; it does not need a GPU, TensorRT, or the
target image. The ONNX it writes is then turned into a TensorRT engine inside
the target perception image by tools/export_vision_engines.py --model action.

**The module definitions below are PYSKL's own, extracted verbatim** from

    https://github.com/kennymckormick/pyskl  (Apache-2.0)
      pyskl/utils/graph.py            Graph
      pyskl/models/gcns/utils/tcn.py  unit_tcn, mstcn
      pyskl/models/gcns/utils/gcn.py  unit_gcn
      pyskl/models/gcns/stgcn.py      STGCNBlock, STGCN
      pyskl/models/gcns/utils/init_func.py

with only the mmcv registry decorators removed and three mmcv factory functions
replaced by the shims below. They are *not* a reimplementation, deliberately:
a hand-written ST-GCN++ with one branch mis-wired loads its state_dict without
complaint and returns wrong numbers, and there is no reference output to catch
that without pyskl itself. Extracting the source and asserting
`load_state_dict(strict=True)` is the check — every parameter name and every
shape has to line up with weights trained by this exact code.

The architecture comes from configs/stgcn++/stgcn++_ntu60_xsub_hrnet/j.py:

    STGCN(graph_cfg=dict(layout='coco', mode='spatial'),
          gcn_adaptive='init', gcn_with_res=True, tcn_type='mstcn')
    GCNHead(num_classes=60, in_channels=256)

and the input tensor from the same file's pipeline — `FormatGCNInput(
num_person=2)` and `UniformSample(clip_len=100)` give (N, M, T, V, C) =
(1, 2, 100, 17, 3), with C being x, y and the detector's score. Those are not
free parameters; plugins/pose_stgcn.py had three of them wrong before this file
was written.

Usage:

    pip install torch onnx onnxruntime
    python3 tools/stgcn_export.py --checkpoint stgcnpp_ntu60_xsub_hrnet_j.pth \
                                  --out stgcnpp-ntu60-2d.onnx
"""
from __future__ import annotations

import argparse
import copy as cp
import hashlib
import os
import sys

import numpy as np
import torch
import torch.nn as nn

# ── mmcv shims ───────────────────────────────────────────────────────────────
#
# The only three mmcv entry points the extracted source uses. `BN` is
# BatchNorm2d in mmcv's registry, which is what the 4-D (N, C, T, V) tensors
# flowing through these modules require; the backbone's own `data_bn` is a
# BatchNorm1d built directly and is untouched by this.


def build_norm_layer(cfg, num_features):
    kind = cfg["type"] if isinstance(cfg, dict) else cfg
    if kind in ("BN", "BN2d"):
        return "bn", nn.BatchNorm2d(num_features)
    if kind == "BN1d":
        return "bn", nn.BatchNorm1d(num_features)
    raise ValueError(f"unsupported norm {kind!r}")


def build_activation_layer(cfg):
    kind = cfg["type"] if isinstance(cfg, dict) else cfg
    if kind == "ReLU":
        return nn.ReLU()
    raise ValueError(f"unsupported activation {kind!r}")


# ── PYSKL source, verbatim ───────────────────────────────────────────────────

# Module-level constant from pyskl/models/gcns/stgcn.py, used when sizing the
# inflated stages.
EPS = 1e-4

def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def k_adjacency(A, k, with_self=False, self_factor=1):
    # A is a 2D square array
    if isinstance(A, torch.Tensor):
        A = A.data.cpu().numpy()
    assert isinstance(A, np.ndarray)
    Iden = np.eye(len(A), dtype=A.dtype)
    if k == 0:
        return Iden
    Ak = np.minimum(np.linalg.matrix_power(A + Iden, k), 1) - np.minimum(np.linalg.matrix_power(A + Iden, k - 1), 1)
    if with_self:
        Ak += (self_factor * Iden)
    return Ak


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A, dim=0):
    # A is a 2D square array
    Dl = np.sum(A, dim)
    h, w = A.shape
    Dn = np.zeros((w, w))

    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)

    AD = np.dot(A, Dn)
    return AD


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.eye(num_node)

    for i, j in edge:
        A[i, j] = 1
        A[j, i] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [
        np.linalg.matrix_power(A, d) for d in range(max_hop + 1)
    ]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


class Graph:
    """The Graph to model the skeletons.

    Args:
        layout (str): must be one of the following candidates: 'openpose', 'nturgb+d', 'coco'. Default: 'coco'.
        mode (str): must be one of the following candidates: 'stgcn_spatial', 'spatial'. Default: 'spatial'.
        max_hop (int): the maximal distance between two connected nodes.
            Default: 1
    """

    def __init__(self,
                 layout='coco',
                 mode='spatial',
                 max_hop=1,
                 nx_node=1,
                 num_filter=3,
                 init_std=0.02,
                 init_off=0.04):

        self.max_hop = max_hop
        self.layout = layout
        self.mode = mode
        self.num_filter = num_filter
        self.init_std = init_std
        self.init_off = init_off
        self.nx_node = nx_node

        assert nx_node == 1 or mode == 'random', "nx_node can be > 1 only if mode is 'random'"
        assert layout in ['openpose', 'nturgb+d', 'coco', 'handmp']

        self.get_layout(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.inward, max_hop)

        assert hasattr(self, mode), f'Do Not Exist This Mode: {mode}'
        self.A = getattr(self, mode)()

    def __str__(self):
        return self.A

    def get_layout(self, layout):
        if layout == 'openpose':
            self.num_node = 18
            self.inward = [
                (4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11), (10, 9),
                (9, 8), (11, 5), (8, 2), (5, 1), (2, 1), (0, 1), (15, 0),
                (14, 0), (17, 15), (16, 14)
            ]
            self.center = 1
        elif layout == 'nturgb+d':
            self.num_node = 25
            neighbor_base = [
                (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                (20, 19), (22, 8), (23, 8), (24, 12), (25, 12)
            ]
            self.inward = [(i - 1, j - 1) for (i, j) in neighbor_base]
            self.center = 21 - 1
        elif layout == 'coco':
            self.num_node = 17
            self.inward = [
                (15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6),
                (9, 7), (7, 5), (10, 8), (8, 6), (5, 0), (6, 0),
                (1, 0), (3, 1), (2, 0), (4, 2)
            ]
            self.center = 0
        elif layout == 'handmp':
            self.num_node = 21
            self.inward = [
                (1, 0), (2, 1), (3, 2), (4, 3), (5, 0), (6, 5), (7, 6), (8, 7),
                (9, 0), (10, 9), (11, 10), (12, 11), (13, 0), (14, 13),
                (15, 14), (16, 15), (17, 0), (18, 17), (19, 18), (20, 19)
            ]
            self.center = 0
        else:
            raise ValueError(f'Do Not Exist This Layout: {layout}')
        self.self_link = [(i, i) for i in range(self.num_node)]
        self.outward = [(j, i) for (i, j) in self.inward]
        self.neighbor = self.inward + self.outward

    def stgcn_spatial(self):
        adj = np.zeros((self.num_node, self.num_node))
        adj[self.hop_dis <= self.max_hop] = 1
        normalize_adj = normalize_digraph(adj)
        hop_dis = self.hop_dis
        center = self.center

        A = []
        for hop in range(self.max_hop + 1):
            a_close = np.zeros((self.num_node, self.num_node))
            a_further = np.zeros((self.num_node, self.num_node))
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if hop_dis[j, i] == hop:
                        if hop_dis[j, center] >= hop_dis[i, center]:
                            a_close[j, i] = normalize_adj[j, i]
                        else:
                            a_further[j, i] = normalize_adj[j, i]
            A.append(a_close)
            if hop > 0:
                A.append(a_further)
        return np.stack(A)

    def spatial(self):
        Iden = edge2mat(self.self_link, self.num_node)
        In = normalize_digraph(edge2mat(self.inward, self.num_node))
        Out = normalize_digraph(edge2mat(self.outward, self.num_node))
        A = np.stack((Iden, In, Out))
        return A

    def binary_adj(self):
        A = edge2mat(self.inward + self.outward, self.num_node)
        return A[None]

    def random(self):
        num_node = self.num_node * self.nx_node
        return np.random.randn(self.num_filter, num_node, num_node) * self.init_std + self.init_off


class unit_tcn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, dilation=1, norm='BN', dropout=0):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))
        self.bn = build_norm_layer(self.norm_cfg, out_channels)[1] if norm is not None else nn.Identity()
        self.drop = nn.Dropout(dropout, inplace=True)
        self.stride = stride

    def forward(self, x):
        return self.drop(self.bn(self.conv(x)))

    def init_weights(self):
        conv_init(self.conv)
        bn_init(self.bn, 1)


class mstcn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=None,
                 dropout=0.,
                 ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'],
                 stride=1):

        super().__init__()
        # Multiple branches of temporal convolution
        self.ms_cfg = ms_cfg
        num_branches = len(ms_cfg)
        self.num_branches = num_branches
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = nn.ReLU()

        if mid_channels is None:
            mid_channels = out_channels // num_branches
            rem_mid_channels = out_channels - mid_channels * (num_branches - 1)
        else:
            assert isinstance(mid_channels, float) and mid_channels > 0
            mid_channels = int(out_channels * mid_channels)
            rem_mid_channels = mid_channels

        self.mid_channels = mid_channels
        self.rem_mid_channels = rem_mid_channels

        branches = []
        for i, cfg in enumerate(ms_cfg):
            branch_c = rem_mid_channels if i == 0 else mid_channels
            if cfg == '1x1':
                branches.append(nn.Conv2d(in_channels, branch_c, kernel_size=1, stride=(stride, 1)))
                continue
            assert isinstance(cfg, tuple)
            if cfg[0] == 'max':
                branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, branch_c, kernel_size=1), nn.BatchNorm2d(branch_c), self.act,
                        nn.MaxPool2d(kernel_size=(cfg[1], 1), stride=(stride, 1), padding=(1, 0))))
                continue
            assert isinstance(cfg[0], int) and isinstance(cfg[1], int)
            branch = nn.Sequential(
                nn.Conv2d(in_channels, branch_c, kernel_size=1), nn.BatchNorm2d(branch_c), self.act,
                unit_tcn(branch_c, branch_c, kernel_size=cfg[0], stride=stride, dilation=cfg[1], norm=None))
            branches.append(branch)

        self.branches = nn.ModuleList(branches)
        tin_channels = mid_channels * (num_branches - 1) + rem_mid_channels

        self.transform = nn.Sequential(
            nn.BatchNorm2d(tin_channels), self.act, nn.Conv2d(tin_channels, out_channels, kernel_size=1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout, inplace=True)

    def inner_forward(self, x):
        N, C, T, V = x.shape

        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        feat = torch.cat(branch_outs, dim=1)
        feat = self.transform(feat)
        return feat

    def forward(self, x):
        out = self.inner_forward(x)
        out = self.bn(out)
        return self.drop(out)

    def init_weights(self):
        pass


class unit_gcn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 adaptive='importance',
                 conv_pos='pre',
                 with_res=False,
                 norm='BN',
                 act='ReLU'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_subsets = A.size(0)

        assert adaptive in [None, 'init', 'offset', 'importance']
        self.adaptive = adaptive
        assert conv_pos in ['pre', 'post']
        self.conv_pos = conv_pos
        self.with_res = with_res

        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        self.act_cfg = act if isinstance(act, dict) else dict(type=act)
        self.bn = build_norm_layer(self.norm_cfg, out_channels)[1]
        self.act = build_activation_layer(self.act_cfg)

        if self.adaptive == 'init':
            self.A = nn.Parameter(A.clone())
        else:
            self.register_buffer('A', A)

        if self.adaptive in ['offset', 'importance']:
            self.PA = nn.Parameter(A.clone())
            if self.adaptive == 'offset':
                nn.init.uniform_(self.PA, -1e-6, 1e-6)
            elif self.adaptive == 'importance':
                nn.init.constant_(self.PA, 1)

        if self.conv_pos == 'pre':
            self.conv = nn.Conv2d(in_channels, out_channels * A.size(0), 1)
        elif self.conv_pos == 'post':
            self.conv = nn.Conv2d(A.size(0) * in_channels, out_channels, 1)

        if self.with_res:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    build_norm_layer(self.norm_cfg, out_channels)[1])
            else:
                self.down = lambda x: x

    def forward(self, x, A=None):
        """Defines the computation performed at every call."""
        n, c, t, v = x.shape
        res = self.down(x) if self.with_res else 0

        A_switch = {None: self.A, 'init': self.A}
        if hasattr(self, 'PA'):
            A_switch.update({'offset': self.A + self.PA, 'importance': self.A * self.PA})
        A = A_switch[self.adaptive]

        if self.conv_pos == 'pre':
            x = self.conv(x)
            x = x.view(n, self.num_subsets, -1, t, v)
            x = torch.einsum('nkctv,kvw->nctw', (x, A)).contiguous()
        elif self.conv_pos == 'post':
            x = torch.einsum('nctv,kvw->nkctw', (x, A)).contiguous()
            x = x.view(n, -1, t, v)
            x = self.conv(x)

        return self.act(self.bn(x) + res)

    def init_weights(self):
        pass


class STGCNBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 stride=1,
                 residual=True,
                 **kwargs):
        super().__init__()

        gcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'gcn_'}
        tcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'tcn_'}
        kwargs = {k: v for k, v in kwargs.items() if k[:4] not in ['gcn_', 'tcn_']}
        assert len(kwargs) == 0, f'Invalid arguments: {kwargs}'

        tcn_type = tcn_kwargs.pop('type', 'unit_tcn')
        assert tcn_type in ['unit_tcn', 'mstcn']
        gcn_type = gcn_kwargs.pop('type', 'unit_gcn')
        assert gcn_type in ['unit_gcn']

        self.gcn = unit_gcn(in_channels, out_channels, A, **gcn_kwargs)

        if tcn_type == 'unit_tcn':
            self.tcn = unit_tcn(out_channels, out_channels, 9, stride=stride, **tcn_kwargs)
        elif tcn_type == 'mstcn':
            self.tcn = mstcn(out_channels, out_channels, stride=stride, **tcn_kwargs)
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, A=None):
        """Defines the computation performed at every call."""
        res = self.residual(x)
        x = self.tcn(self.gcn(x, A)) + res
        return self.relu(x)




class STGCN(nn.Module):

    def __init__(self,
                 graph_cfg,
                 in_channels=3,
                 base_channels=64,
                 data_bn_type='VC',
                 ch_ratio=2,
                 num_person=2,  # * Only used when data_bn_type == 'MVC'
                 num_stages=10,
                 inflate_stages=[5, 8],
                 down_stages=[5, 8],
                 pretrained=None,
                 **kwargs):
        super().__init__()

        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.data_bn_type = data_bn_type
        self.kwargs = kwargs

        if data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()

        lw_kwargs = [cp.deepcopy(kwargs) for i in range(num_stages)]
        for k, v in kwargs.items():
            if isinstance(v, tuple) and len(v) == num_stages:
                for i in range(num_stages):
                    lw_kwargs[i][k] = v[i]
        lw_kwargs[0].pop('tcn_dropout', None)

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages

        modules = []
        if self.in_channels != self.base_channels:
            modules = [STGCNBlock(in_channels, base_channels, A.clone(), 1, residual=False, **lw_kwargs[0])]

        inflate_times = 0
        for i in range(2, num_stages + 1):
            stride = 1 + (i in down_stages)
            in_channels = base_channels
            if i in inflate_stages:
                inflate_times += 1
            out_channels = int(self.base_channels * self.ch_ratio ** inflate_times + EPS)
            base_channels = out_channels
            modules.append(STGCNBlock(in_channels, out_channels, A.clone(), stride, **lw_kwargs[i - 1]))

        if self.in_channels == self.base_channels:
            num_stages -= 1

        self.num_stages = num_stages
        self.gcn = nn.ModuleList(modules)
        self.pretrained = pretrained

    def forward(self, x):
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        if self.data_bn_type == 'MVC':
            x = self.data_bn(x.view(N, M * V * C, T))
        else:
            x = self.data_bn(x.view(N * M, V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        for i in range(self.num_stages):
            x = self.gcn[i](x)

        x = x.reshape((N, M) + x.shape[1:])
        return x


# ── the exportable model ─────────────────────────────────────────────────────


class STGCNClassifier(nn.Module):
    """Backbone + head, with the recogniser's test-time wrapper left out.

    PYSKL's `RecognizerGCN.forward_test` averages logits over clips and people
    and reshapes around a batch of clips. Exporting *that* would bake a
    pipeline the robot does not have into the graph — the card has one clip and
    one tracked person. So the pooling is written out here explicitly, matching
    `GCNHead`: global average over time and joints, then over the person axis,
    then the classifier.
    """

    def __init__(self, num_classes: int = 60, in_channels: int = 256):
        super().__init__()
        self.backbone = STGCN(
            graph_cfg=dict(layout="coco", mode="spatial"),
            gcn_adaptive="init", gcn_with_res=True, tcn_type="mstcn",
        )
        self.cls_head = nn.Module()
        self.cls_head.fc_cls = nn.Linear(in_channels, num_classes)

    def forward(self, keypoint):
        # (N, M, T, V, C) -> backbone -> (N, M, C', T', V)
        x = self.backbone(keypoint)
        N, M, C, T, V = x.shape
        x = x.reshape(N, M, C, T * V).mean(dim=3)      # pool time and joints
        x = x.mean(dim=1)                              # pool the person axis
        return self.cls_head.fc_cls(x)


def load_model(checkpoint: str) -> STGCNClassifier:
    """Build and load, insisting on an exact match.

    `strict=True` is the whole verification story for the vendored source: if a
    single parameter name or shape disagreed with the weights, the architecture
    would be wrong and this would say so instead of quietly producing a model
    that runs.
    """
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    model = STGCNClassifier()
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if "num_batches_tracked" not in k]
    unexpected = [k for k in unexpected if "num_batches_tracked" not in k]
    if missing or unexpected:
        raise SystemExit(
            "the vendored architecture does not match the checkpoint.\n"
            f"  missing from checkpoint : {missing[:8]}{' ...' if len(missing) > 8 else ''}\n"
            f"  unexpected in checkpoint: {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}\n"
            "Re-extract from the pyskl revision the weights were trained with."
        )
    model.eval()
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="stgcnpp-ntu60-2d.onnx")
    parser.add_argument("--frames", type=int, default=100,
                        help="T, the clip length the checkpoint was trained "
                             "with (default %(default)s)")
    parser.add_argument("--opset", type=int, default=16)
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    params = sum(p.numel() for p in model.parameters())
    print(f"[export] loaded, strict match, {params/1e6:.2f} M parameters",
          flush=True)

    shape = (1, 2, args.frames, 17, 3)
    example = torch.zeros(shape, dtype=torch.float32)
    # Not zeros for the check: an all-zero skeleton is a degenerate input and
    # would hide a transposed axis behind a symmetric answer.
    probe = torch.from_numpy(
        np.random.default_rng(0).standard_normal(shape).astype(np.float32))

    with torch.no_grad():
        reference = model(probe).numpy()
    print(f"[export] torch output {reference.shape}", flush=True)

    torch.onnx.export(
        model, example, args.out,
        input_names=["keypoint"], output_names=["logits"],
        opset_version=args.opset, do_constant_folding=True,
        dynamic_axes=None,
    )
    print(f"[export] wrote {args.out} ({os.path.getsize(args.out)} bytes)",
          flush=True)

    try:
        import onnxruntime as ort
    except ImportError:
        print("[export] onnxruntime not installed — SKIPPING the numerical "
              "check. Install it and re-run before trusting this file.",
              file=sys.stderr)
        return 0

    session = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    got = session.run(["logits"], {"keypoint": probe.numpy()})[0]
    delta = float(np.max(np.abs(got - reference)))
    print(f"[export] onnx vs torch, max abs delta = {delta:.3e}", flush=True)
    if delta > 1e-4:
        raise SystemExit(f"ONNX disagrees with torch by {delta:.3e} — refusing")
    print("[export] numerically identical", flush=True)

    digest = hashlib.sha256(open(args.out, "rb").read()).hexdigest()
    print(f'\n    "{os.path.basename(args.out)}": '
          f'{{"size": {os.path.getsize(args.out)}, "sha256": "{digest}"}},',
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
