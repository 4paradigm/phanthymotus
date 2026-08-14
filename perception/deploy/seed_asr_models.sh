#!/usr/bin/env bash
set -euo pipefail

seed_root="${ASR_MODEL_SEED_ROOT:-/opt/phanthy-motus/model-seed/asr}"
target_root="${ASR_MODEL_TARGET_ROOT:-/models/sherpa-onnx}"

if [ ! -d "${seed_root}" ]; then
    exit 0
fi

mkdir -p "${target_root}"
for source_bundle in "${seed_root}"/*; do
    [ -d "${source_bundle}" ] || continue
    bundle_name="${source_bundle##*/}"
    target_bundle="${target_root}/${bundle_name}"
    mkdir -p "${target_bundle}"
    for source_file in "${source_bundle}"/*; do
        [ -f "${source_file}" ] || continue
        filename="${source_file##*/}"
        target_file="${target_bundle}/${filename}"
        if [ ! -s "${target_file}" ] || ! cmp -s "${source_file}" "${target_file}"; then
            temporary_file="$(mktemp "${target_file}.tmp.XXXXXX")"
            trap 'rm -f "${temporary_file}"' EXIT
            cp -- "${source_file}" "${temporary_file}"
            chmod 0644 "${temporary_file}"
            mv -f -- "${temporary_file}" "${target_file}"
            trap - EXIT
        fi
    done
done
