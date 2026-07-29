#!/usr/bin/env bash
set -euo pipefail

export TTS_PLUGIN=${TTS_PLUGIN:-vits2_tts_trt}
export DEFAULT_TTS_PLUGIN=${DEFAULT_TTS_PLUGIN:-vits2_tts_trt}

MODEL_VARIANT="${MIX_VITS_MODEL_VARIANT:-vits256}"
BUILT_VARIANT="$(cat /models/vits2-model.variant 2>/dev/null || true)"
if [[ "${MODEL_VARIANT}" != "custom" && "${BUILT_VARIANT}" != "${MODEL_VARIANT}" ]]; then
    echo "Requested model ${MODEL_VARIANT}, but image contains ${BUILT_VARIANT:-unknown}" >&2
    echo "Rebuild with --build-arg VITS_MODEL_VARIANT=${MODEL_VARIANT}" >&2
    exit 2
fi
case "${MODEL_VARIANT}" in
    lc1)
        MODEL_ROOT="${MIX_VITS_MODEL_ROOT:-/models/vits2-mix}"
        export MIX_VITS_CONFIG_PATH="${MODEL_ROOT}/config.json"
        export MIX_VITS_TRT_ENGINE_DIR="${MODEL_ROOT}/engines"
        ;;
    vits256)
        MODEL_ROOT="${MIX_VITS_MODEL_ROOT:-/models/vits2-vits256}"
        export MIX_VITS_CONFIG_PATH="${MODEL_ROOT}/config.json"
        export MIX_VITS_TRT_ENGINE_DIR="${MODEL_ROOT}/engines"
        ;;
    custom)
        : "${MIX_VITS_CONFIG_PATH:?custom model requires MIX_VITS_CONFIG_PATH}"
        : "${MIX_VITS_TRT_ENGINE_DIR:?custom model requires MIX_VITS_TRT_ENGINE_DIR}"
        MODEL_ROOT="<custom-paths>"
        ;;
    *)
        echo "Unsupported MIX_VITS_MODEL_VARIANT=${MODEL_VARIANT}; expected lc1, vits256, or custom" >&2
        exit 2
        ;;
esac

test -s "${MIX_VITS_CONFIG_PATH}" || {
    echo "Missing VITS config: ${MIX_VITS_CONFIG_PATH}" >&2
    exit 2
}
test -s "${MIX_VITS_TRT_ENGINE_DIR}/manifest.json" || {
    echo "Missing TensorRT manifest: ${MIX_VITS_TRT_ENGINE_DIR}/manifest.json" >&2
    exit 2
}
test -d "${NLTK_DATA:-/models/vits2-mix/nltk_data}" || {
    echo "Missing shared NLTK data: ${NLTK_DATA:-/models/vits2-mix/nltk_data}" >&2
    exit 2
}
echo "VITS2 model variant=${MODEL_VARIANT} root=${MODEL_ROOT}"
echo "VITS2 config=${MIX_VITS_CONFIG_PATH} engines=${MIX_VITS_TRT_ENGINE_DIR}"

set +u
source /opt/ros/jazzy/setup.bash
source /ros_ws/install/setup.bash
set -u

cd /work
exec /usr/bin/python3 /work/main.py
