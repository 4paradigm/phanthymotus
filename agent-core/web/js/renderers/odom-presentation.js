/** Standard motus.odom/1 fields shared by current snapshots and retained logs. */
import { plainValue } from './json-util.js';

const TRACKING = { path: 'vendor.motus_slam.status', equals: 'tracking' };
const ODOM_PRESENTATION = {
  fields: [
    { path: 'vendor.motus_slam.status', label: '定位状态', group: '状态', missing: 'unknown' },
    { path: 'vendor.motus_slam.source_available', label: '来源可用', group: '状态', missing: 'unknown' },
    { path: 'vendor.motus_slam.reason', label: '原因', group: '状态' },
    { path: 'pose.position', label: '位置', group: '位置与朝向', unit: 'm', valid_when: TRACKING },
    { path: 'pose.frame_id', label: '参考坐标系', group: '位置与朝向' },
    { path: 'pose.child_frame_id', label: '位姿参考点', group: '位置与朝向' },
    { path: 'frame', label: '速度坐标系', group: '运动' },
    { path: 'twist.0', label: 'vx', group: '运动', unit: 'm/s' },
    { path: 'twist.1', label: 'vy', group: '运动', unit: 'm/s' },
    { path: 'twist.5', label: 'wz', group: '运动', unit: 'rad/s' },
    { path: 'stamp_ms', label: '采样时间', group: '时间与连续性', unit: 'ms', missing: '未知',
      valid_when: { path: 'vendor.motus_slam.stamp_source', equals: 'measurement' } },
    { path: 'vendor.motus_slam.stamp_source', label: '时间来源', group: '时间与连续性' },
    { path: 'vendor.motus_slam.measurement_age_ms', label: '测量年龄', group: '时间与连续性', unit: 'ms', missing: '未知' },
    { path: 'vendor.motus_slam.segment_id', label: '分段', group: '时间与连续性' },
    { path: 'vendor.motus_slam.clock_id', label: '时钟', group: '时间与连续性' },
  ],
  details: [
    { path: 'pose', label: '完整位姿、四元数与协方差' },
    { path: 'twist', label: '完整运动轴' },
    { path: 'vendor.motus_slam', label: '时间映射与连续性' },
  ],
};

export const ODOM_DETAILS = ODOM_PRESENTATION.details;

export function valueAtPath(value, path) {
  if (typeof path !== 'string') return undefined;
  if (!path) return value;
  for (const key of path.split('.')) {
    if (value === null || typeof value !== 'object' || !Object.hasOwn(value, key)) return undefined;
    value = value[key];
  }
  return value;
}

export function projectOdom(value) {
  const finite = number => typeof number === 'number' && Number.isFinite(number);
  const fields = ODOM_PRESENTATION.fields.map(field => {
    const valid = !field.valid_when || valueAtPath(value, field.valid_when.path) === field.valid_when.equals;
    let current = valid ? valueAtPath(value, field.path) : undefined;
    if (field.path === 'pose.position') {
      current = Array.isArray(current) && current.length === 3
        ? current.map(axis => finite(axis) ? plainValue(axis) : '—').join(', ') : undefined;
    } else if (field.unit && (!finite(current) ||
        (field.path.startsWith('twist.') && !Array.isArray(value.twist)))) current = undefined;
    const missing = current === null || current === undefined;
    return { label: field.label, group: field.group,
      text: missing ? field.missing ?? '—' : plainValue(current) + (field.unit ? ' ' + field.unit : '') };
  });
  const yaw = orientationYaw(value);
  fields.push({ label: 'yaw', group: '位置与朝向',
    text: yaw === undefined ? '未知' : plainValue(yaw) + ' rad' });
  return fields;
}

function orientationYaw(value) {
  if (valueAtPath(value, TRACKING.path) !== TRACKING.equals) return undefined;
  const axes = valueAtPath(value, 'vendor.motus_slam.pose_axes');
  if (axes?.quaternion_order !== 'xyzw' || axes.up_axis !== 'z') return undefined;
  const q = valueAtPath(value, 'pose.quaternion');
  if (!Array.isArray(q) || q.length !== 4 || !q.every(Number.isFinite)) return undefined;
  const norm = Math.hypot(...q);
  if (!Number.isFinite(norm) || norm === 0) return undefined;
  const [x, y, z, w] = q.map(component => component / norm);
  return Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
}
