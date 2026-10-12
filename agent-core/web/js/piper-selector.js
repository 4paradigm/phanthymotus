/** Explicitly named device bundles; selection changes visibility, never card routing. */
export function piperDevices(mcps) {
  const devices = (mcps || []).flatMap(m => {
    const match = /^机械臂\s*([12])$/.exec(m.name || '');
    const names = new Set((m.tools || []).map(t => typeof t === 'string' ? t : t.name));
    return match && names.has('piper_joint') && names.has('piper_photo')
      ? [{id: m.id, number: Number(match[1]), label: `机械臂 ${match[1]}`}]
      : [];
  }).sort((a, b) => a.number - b.number);
  // Ambiguous names or missing bundles must not silently select another robot.
  return devices.length === 2 && new Set(devices.map(d => d.number)).size === 2
    && new Set(devices.map(d => d.id)).size === 2 ? devices : [];
}

export async function switchPiperDevice(devices, targetId, {call, projectStopped}) {
  if (!devices.some(d => d.id === targetId)) throw new Error('目标机械臂不存在，请刷新设备列表');
  if (!await projectStopped()) throw new Error('请先关闭智能控制再切换机械臂');
  for (const device of devices) {
    await call(device.id, 'piper_status', {action: 'start'});
    const status = await call(device.id, 'piper_status', {action: 'read'});
    if (status.busy !== false || status.feedback_fresh !== true) {
      throw new Error(`${device.label} 正忙或反馈不新鲜，请先检查状态`);
    }
  }
  for (const device of devices) {
    const result = await call(device.id, 'piper_permission', {action: 'block'});
    if (result.motion_enabled !== false || result.operation_stopping) {
      throw new Error(`${device.label} 许可关闭尚未完成，请读取状态后重试`);
    }
  }
  // Recheck after closing permission; another client may have started work.
  for (const device of devices) {
    const status = await call(device.id, 'piper_status', {action: 'read'});
    if (status.busy !== false || status.motion_enabled !== false || status.feedback_fresh !== true) {
      throw new Error(`${device.label} 状态已变化，未切换操作对象`);
    }
  }
  if (!await projectStopped()) throw new Error('智能控制已开启，未切换操作对象');
  return targetId;
}
