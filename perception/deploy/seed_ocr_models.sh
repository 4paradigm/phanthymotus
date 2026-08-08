#!/usr/bin/env bash
set -euo pipefail

seed_root="/opt/phanthy-motus/model-seed/ocr"
target_root="/models/ocr"

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
        if [ ! -s "${target_file}" ]; then
            temporary_file="${target_file}.tmp.$$"
            cp "${source_file}" "${temporary_file}"
            chmod 0644 "${temporary_file}"
            mv -f "${temporary_file}" "${target_file}"
        fi
    done
done
