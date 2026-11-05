/** Decode diagnostic odometry previews without accepting SLAM claims. */
export function decodeSlamPreview(buffer) {
  if (!(buffer instanceof ArrayBuffer) || buffer.byteLength < 21) return null;
  const length = new DataView(buffer).getUint32(0, true);
  if (length === 0 || length > 2048 || 4 + length + 17 > buffer.byteLength) return null;
  try {
    const meta = JSON.parse(new TextDecoder('utf-8', {fatal: true}).decode(new Uint8Array(buffer, 4, length)));
    const odometry = meta.kind === 'unvalidated_odometry_preview' && ['raw', 'gtsam'].includes(meta.backend);
    const geometry = meta.kind === 'unvalidated_geometry_preview' && meta.backend === 'geometry_experimental' &&
      Number.isSafeInteger(meta.candidate_count) && meta.candidate_count >= 0;
    if (meta.schema !== 'motus.slam-preview/1' || (!odometry && !geometry) ||
        meta.accepted_loops !== 0 ||
        typeof meta.segment_id !== 'string' || !meta.segment_id || meta.segment_id.length > 256 ||
        typeof meta.frame_id !== 'string' || !meta.frame_id || meta.frame_id.length > 256 ||
        typeof meta.stamp_ns !== 'string' || !/^[0-9]+$/.test(meta.stamp_ns) || /^0+$/.test(meta.stamp_ns)) return null;
    const payload = buffer.slice(4 + length);
    const view = new DataView(payload);
    const count = view.getUint32(13, true);
    if (view.getUint8(12) !== 3 || count > 80000 || payload.byteLength !== 17 + count * 12) return null;
    return {meta, payload};
  } catch (_) {
    return null;
  }
}
