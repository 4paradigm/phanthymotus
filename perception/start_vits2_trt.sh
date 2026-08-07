#!/usr/bin/env bash
set -euo pipefail

export TTS_PLUGIN=${TTS_PLUGIN:-vits2_tts_trt}
export DEFAULT_TTS_PLUGIN=${DEFAULT_TTS_PLUGIN:-vits2_tts_trt}

BUILT_RELEASE="$(cat /models/vits2-model.release 2>/dev/null || true)"
MODEL_RELEASE="${MIX_VITS_RELEASE:-${BUILT_RELEASE}}"
if [[ -z "${BUILT_RELEASE}" || "${MODEL_RELEASE}" != "${BUILT_RELEASE}" ]]; then
    echo "Requested release ${MODEL_RELEASE:-unknown}, but image contains ${BUILT_RELEASE:-unknown}" >&2
    echo "Rebuild with --build-arg VITS_MODEL_RELEASE=${MODEL_RELEASE:-<release>}" >&2
    exit 2
fi
MODEL_ROOT="/models/releases/${MODEL_RELEASE}"
export MIX_VITS_CONFIG_PATH="${MODEL_ROOT}/config.json"
export MIX_VITS_TRT_ENGINE_DIR="${MODEL_ROOT}/engines"

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
echo "VITS2 release=${MODEL_RELEASE} root=${MODEL_ROOT}"
echo "VITS2 config=${MIX_VITS_CONFIG_PATH} engines=${MIX_VITS_TRT_ENGINE_DIR}"

set +u
source /opt/ros/jazzy/setup.bash
source /ros_ws/install/setup.bash
set -u

cd /work
exec /usr/bin/python3 /work/main.py
