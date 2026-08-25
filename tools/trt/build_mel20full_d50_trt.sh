#!/usr/bin/env bash
# 在 Jetson Orin 上构建 mel20full+d50 冠军模型的 TRT 引擎（encoder + flow + decoder）
# 用法: bash build_mel20full_d50_trt.sh <workspace_dir> [trt_image]
# 前置: workspace_dir 内含 encoder_duration.onnx（opset 16）、flow.onnx、decoder_spec.onnx
set -euo pipefail

WS="${1:-/home/develop/trt_engines_mel20full_d50}"
IMG="${2:-bj-warehouse.tencentcloudcr.com/phanthy-motus/jetson-base:jp511-torch}"
TRTEXEC=/usr/src/tensorrt/bin/trtexec

[ -f "$WS/encoder_duration.onnx" ] || { echo "缺少 $WS/encoder_duration.onnx（需用 export_mel20full_d50_encoder_flow.py 以 opset 16 导出）"; exit 1; }
[ -f "$WS/flow.onnx" ] || { echo "缺少 $WS/flow.onnx"; exit 1; }
[ -f "$WS/decoder_spec.onnx" ] || { echo "缺少 $WS/decoder_spec.onnx"; exit 1; }

echo "== 构建 encoder.trt (FP16, ph/to/la 动态 1..1000) =="
docker run --rm --runtime=nvidia \
    -v "$WS":/workspace \
    --entrypoint "$TRTEXEC" \
    "$IMG" \
    --onnx=/workspace/encoder_duration.onnx --saveEngine=/workspace/encoder.trt --fp16 \
    --minShapes=ph:1x1,to:1x1,la:1x1,xl:1 \
    --optShapes=ph:1x200,to:1x200,la:1x200,xl:1 \
    --maxShapes=ph:1x1000,to:1x1000,la:1x1000,xl:1

echo "== 构建 flow.trt (FP16) =="
docker run --rm --runtime=nvidia \
    -v "$WS":/workspace \
    --entrypoint "$TRTEXEC" \
    "$IMG" \
    --onnx=/workspace/flow.onnx --saveEngine=/workspace/flow.trt --fp16 \
    --minShapes=z_p:1x256x1,y_mask:1x1x1 \
    --optShapes=z_p:1x256x200,y_mask:1x1x200 \
    --maxShapes=z_p:1x256x1500,y_mask:1x1x1500

echo "== 构建 decoder.trt (FP16) =="
docker run --rm --runtime=nvidia \
    -v "$WS":/workspace \
    --entrypoint "$TRTEXEC" \
    "$IMG" \
    --onnx=/workspace/decoder_spec.onnx --saveEngine=/workspace/decoder.trt --fp16 \
    --minShapes=z:1x256x1 \
    --optShapes=z:1x256x200 \
    --maxShapes=z:1x256x1500

ls -lh "$WS"/encoder.trt "$WS"/flow.trt "$WS"/decoder.trt
echo "== 完成 =="
