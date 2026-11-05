/**
 * browser-source.js — the part of starting `remote_mic` / `remote_camera` that
 * is about the *card*, not about the media API.
 *
 * Both cards declare instance-scoped config (`device_id`, and `fps`/`width` for
 * the camera), the config dialog saves it, and for as long as these cards have
 * existed nothing read it back: both start paths — start-project and the
 * post-refresh re-arm — called the streamers with no options at all. Choosing a
 * microphone changed nothing; the browser opened its default, which hid the bug
 * whenever the default happened to be the one chosen. Found on a U1 Pro, where
 * it was not.
 *
 * Three states, kept apart on purpose:
 *
 *   no device chosen     → the browser default, as before
 *   chosen, unavailable  → an error asking to re-pick it. Never a silent switch
 *                          to another device: a robot hearing the wrong room
 *                          is worse than one that says it cannot hear.
 *   config unreadable    → an error. Never treated as "nothing chosen", or a
 *                          401 would quietly open the default microphone.
 *
 * Pure apart from `fetchInstanceConfig`, so the decisions are testable in node.
 */

/**
 * Read one card instance's saved config, fresh from the server.
 *
 * Not the sidebar's cache: that one is filled at page load and updated only by
 * this tab's own saves, so a device picked in another tab — or another
 * browser — would not be seen.
 *
 * @returns {Promise<object>} the saved config, `{}` when none was saved
 * @throws when the request fails, so the caller cannot mistake it for `{}`
 */
export async function fetchInstanceConfig(mcpId, toolName, instanceId, fetchFn = fetch) {
  const url = `/api/canvas/tool-config/${encodeURIComponent(mcpId)}`
    + `/${encodeURIComponent(toolName)}/${encodeURIComponent(instanceId)}`;
  const resp = await fetchFn(url, { cache: 'no-store' });
  if (!resp.ok) throw new Error(`读取 ${toolName} 配置失败（HTTP ${resp.status}）`);
  const body = await resp.json();
  if (body && body.code !== undefined && body.code !== 200) {
    throw new Error(`读取 ${toolName} 配置失败（code=${body.code}）`);
  }
  const data = body ? body.data : null;
  return data && typeof data === 'object' ? data : {};
}

function _deviceId(cfg) {
  const id = cfg && cfg.device_id;
  return typeof id === 'string' ? id.trim() : '';
}

function _positiveNumber(value) {
  // `null` and `''` need excluding by hand: Number() turns both into 0.
  if (value === null || value === undefined || value === '') return undefined;
  const n = Number(value);
  return Number.isFinite(n) && n > 0 ? n : undefined;
}

/** `remote_mic` instance config → `startMicStream` options. */
export function micOptions(cfg) {
  return { deviceId: _deviceId(cfg) };
}

/** `remote_camera` instance config → `startCameraStream` options.
 *
 * Unset or unusable values are left out rather than zeroed, so the streamer's
 * own defaults apply. */
export function cameraOptions(cfg) {
  const opts = { deviceId: _deviceId(cfg) };
  const fps = _positiveNumber(cfg && cfg.fps);
  const width = _positiveNumber(cfg && cfg.width);
  if (fps !== undefined) opts.fps = fps;
  if (width !== undefined) opts.width = width;
  return opts;
}

/** The `getUserMedia` constraint that selects a device, or none. `exact`,
 * because a plain `deviceId` is only a preference and the browser may quietly
 * hand back a different device — the very failure being fixed here. */
export function deviceConstraint(deviceId) {
  return deviceId ? { deviceId: { exact: deviceId } } : {};
}

/**
 * Turn a media error into something an operator can act on.
 *
 * Without this every failure surfaced as "no sound" — a denied permission, an
 * unplugged device, a device another app holds and a dead WebSocket all look
 * the same from the robot's side.
 *
 * @param {Error} err
 * @param {'microphone'|'camera'} kind
 * @param {string} deviceId  the device that was asked for, '' for the default
 */
export function describeMediaError(err, kind, deviceId = '') {
  const noun = kind === 'camera' ? '摄像头' : '麦克风';
  const card = kind === 'camera' ? 'remote_camera' : 'remote_mic';
  const name = err && err.name;
  const detail = (err && err.message) || String(err);
  if (name === 'NotAllowedError' || name === 'SecurityError') {
    return `浏览器${noun}权限被拒绝：请在地址栏允许${noun}后重试`;
  }
  if (deviceId && (name === 'NotFoundError' || name === 'OverconstrainedError')) {
    // A device id is per browser and per site: one saved from another
    // computer, another browser, or before the device was unplugged is not
    // valid here. Same error either way, same remedy.
    return `所选${noun}在这个浏览器里不可用（已拔出，或是在别的电脑/浏览器上选的）：`
      + `请在 ${card} 配置里重新选择并保存`;
  }
  if (name === 'NotFoundError') return `这台电脑上没有可用的${noun}`;
  if (name === 'NotReadableError' || name === 'AbortError') {
    return `${noun}被其他程序占用或无法读取：关闭占用它的程序后重试（${detail}）`;
  }
  return `${noun}启动失败：${detail}`;
}
