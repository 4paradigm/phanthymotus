import { clearToken, getAuthStatus, getToken, setToken } from '../auth.js';
import {
  buildRtcFrameV1,
  createRtcFrameState,
  isDualSqueezeDeadmanRequested,
  markRtcFrameDeadmanReleased,
} from './webxr-frame.js';
import {
  captureAssignmentForSession,
  captureAttachEligibility,
  captureBootstrapFields,
} from './capture-console.js';

const notice = document.getElementById('global-notice');
const loginPanel = document.getElementById('login-panel');
const loginForm = document.getElementById('login-form');
const loginError = document.getElementById('login-error');
const tokenInput = document.getElementById('token-input');
const content = document.getElementById('content');
const logoutButton = document.getElementById('logout-button');
const refreshButton = document.getElementById('refresh-button');
const webxrSupportElement = document.getElementById('webxr-support');
const capturePanel = document.getElementById('capture-panel');
const captureRefreshButton = document.getElementById('capture-refresh-button');
const capturePairButton = document.getElementById('capture-pair-button');
const captureLabelInput = document.getElementById('capture-label');
const captureBootstrap = document.getElementById('capture-bootstrap');
const captureWssUrl = document.getElementById('capture-wss-url');
const capturePairingId = document.getElementById('capture-pairing-id');
const capturePairingCode = document.getElementById('capture-pairing-code');
const captureCaBase64 = document.getElementById('capture-ca-base64');
const capturePairingExpiry = document.getElementById('capture-pairing-expiry');
const captureDeviceSelect = document.getElementById('capture-device-select');
const captureSessionSelect = document.getElementById('capture-session-select');
const captureDeviceFacts = document.getElementById('capture-device-facts');
const captureAttachButton = document.getElementById('capture-attach-button');
const captureRevokeButton = document.getElementById('capture-revoke-button');
const captureStatus = document.getElementById('capture-status');

const STATUS_POLL_MS = 1000;
const SLOW_POLL_MS = 5000;
const CORE_HEARTBEAT_MS = 5000;
// Must finish before the next 5s ownership heartbeat opportunity.
const REQUEST_TIMEOUT_MS = 4000;
const RTC_SIGNAL_TIMEOUT_MS = 12000;
const RTC_ICE_GATHER_TIMEOUT_MS = 5000;
const RTC_PEER_PING_MS = 1000;
const RTC_PEER_PING_TIMEOUT_MS = 5000;
const RTC_RTT_SAMPLE_LIMIT = 256;
const LOGOUT_RELEASE_WAIT_MS = 1500;
const XR_NORMAL_SEND_HZ = 60;
const XR_SEND_INTERVAL_MS = 1000 / XR_NORMAL_SEND_HZ;
const XR_UI_REFRESH_MS = 250;
const POSE_MESSAGE_LIMIT_BYTES = 64 * 1024;
const POSE_BUFFER_HIGH_WATER_BYTES = 16 * 1024;
const CLIENT_ID_KEY = 'motus.teleop.client-id.v1';
const CANONICAL_UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
const RELEASABLE_SESSION_STATES = new Set([
  'awaiting_confirmation', 'preparing', 'active', 'paused', 'hold',
]);
const TERMINAL_STATES = new Set(['released', 'expired', 'faulted']);
const RECOVERABLE_DRIVER_HOLD_REASONS = new Set([
  'command_timeout',
  'deadman_released',
  'intent_expired',
  'pose_timeout',
  'rtc_closed',
  'rtc_disconnected',
  'rtc_failed',
  'rtc_not_ready',
  'tracking_lost',
]);

function generateClientId() {
  if (typeof crypto.randomUUID === 'function') return crypto.randomUUID().toLowerCase();
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = Array.from(bytes, (value) => value.toString(16).padStart(2, '0')).join('');
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

function tabClientId() {
  // Browsers may clone sessionStorage into a tab opened by an opener or by
  // "Duplicate tab". Never read or reuse that inherited value: rotate before
  // the first API call so two live documents cannot share browser authority.
  const generated = generateClientId();
  if (!CANONICAL_UUID.test(generated)) throw new Error('client UUID generation failed');
  try {
    sessionStorage.setItem(CLIENT_ID_KEY, generated);
  } catch {
    // A privacy-restricted browser still receives a document-lifetime ID.
  }
  return generated;
}

const TELEOP_CLIENT_ID = tabClientId();

let principal = null;
let robots = [];
let statusTimer = null;
let slowTimer = null;
let heartbeatTimer = null;
let requestGeneration = 0;
let directoryInFlight = false;
let statusInFlight = false;
let eventsInFlight = false;
let heartbeatInFlight = false;
let actionInFlight = false;
let captureListInFlight = false;
let captureActionInFlight = false;
let captures = [];
let capturePairing = null;
let selectedCaptureId = '';
let selectedCaptureSessionId = '';
let captureStatusMessage = '尚未读取采集设备。';
let captureStatusKind = '';
let peerPingSequence = 0;
const activeControllers = new Set();
const sessionViews = new Map();
const lastSessionByRobot = new Map();
const sessionEvents = new Map();
const pendingRobots = new Set();
const browserHeartbeatFailures = new Map();
const rtcViews = new Map();
const rtcInFlight = new Set();
const rtcFrameStates = new Map();
const xrSessions = new Map();
const xrSessionStats = new Map();

const webxrSupport = {
  secureContext: window.isSecureContext === true,
  apiAvailable: Boolean(navigator.xr),
  immersiveVr: 'checking',
  detail: '正在检测 immersive-vr 与 local-floor…',
};

const REASON_TEXT = {
  ready: '可信且声明遥操能力',
  driver_registration_not_trusted: '注册未受信任，禁止遥操',
  teleop_session_not_declared: '未显式声明 x-teleop',
  teleop_descriptor_invalid: 'x-teleop 描述不符合通用遥操协议',
  driver_transport_invalid: 'Driver 不是安全且受支持的 HTTP 传输目标',
  driver_runtime_not_trusted: 'Driver 运行时注册尚未通过可信校验',
  driver_runtime_target_mismatch: 'Driver 运行时目标或遥操能力指纹与配置不一致',
  driver_offline: 'Driver 当前离线',
  authority_recovery_required: 'Core 重启后仍保持写入锁定，等待 owner 验证 Driver 已安全停止',
};

const ERROR_TEXT = {
  network_error: '网络连接失败；不会自动获取新会话。',
  request_timeout: '请求超时；控制台不会自动重试控制操作。',
  request_cancelled: '请求已取消。',
  teleop_client_required: 'Core 未识别本标签页客户端身份。',
  robot_busy: '机器人已被其他会话占用。',
  session_forbidden: '你无权操作这个会话。',
  session_client_mismatch: '该会话属于另一个浏览器标签页。',
  session_state_conflict: '当前会话状态不允许执行该操作。',
  session_not_found: '会话不存在或已不再占用机器人。',
  session_released: '会话已经释放。',
  session_expired: '浏览器到 Core 的租约已过期。',
  session_faulted: '会话因 Driver 或协议故障终止。',
  session_prepare_stale: '会话准备结果已过期，请重新获取。',
  driver_not_found: '没有找到对应的 Driver。',
  driver_not_ready: 'Driver 尚未满足可信遥操条件。',
  teleop_mode_unavailable: 'Driver 没有提供所选 Shadow/Live 模式。',
  live_confirmation_mismatch: 'Live 确认与当前会话、标签页或能力配置不一致。',
  driver_authority_busy: 'Driver 已持有另一份控制权。',
  robot_recovery_required: '机器人仍处于 Core 重启安全恢复锁定中。',
  authority_guard_not_found: '这台机器人没有待恢复的安全锁。',
  authority_guard_target_changed: 'Driver 目标或能力已变化；安全锁保持，禁止自动解除。',
  authority_guard_not_safe: 'Driver 尚未提供严格的安全停机证明；安全锁保持。',
  authority_guard_persistence_error: 'Core 无法可靠更新恢复锁；普通写入继续保持锁定。',
  driver_command_busy: 'Driver 仍有普通控制命令在执行；遥操未获取控制权。',
  driver_response_invalid: 'Driver 返回了不符合协议的状态。',
  driver_lease_unsafe: 'Driver 租约不在安全范围内。',
  driver_epoch_exhausted: 'Driver 防重放 epoch 已耗尽；Core 已拒绝创建新会话。',
  driver_identity_changed: 'Driver 身份或启动实例已经变化。',
  driver_prepare_rejected: 'Driver 拒绝准备遥操会话。',
  driver_session_lost: 'Core 已失去 Driver 会话，当前会话已故障终止。',
  driver_pause_rejected: 'Driver 未确认 Pause。',
  driver_soft_stop_rejected: 'Driver 未确认 HOLD。',
  driver_release_invalid: 'Driver 的 Release 确认不符合协议。',
  driver_secret_reflected: 'Driver 响应包含私密控制凭据，Core 已拒绝。',
  driver_timeout: 'Driver 响应超时。',
  driver_auth_rejected: 'Driver 拒绝了 Core 身份。',
  driver_unreachable: '当前无法连接 Driver。',
  driver_protocol_error: 'Driver 协议调用失败。',
  coordinator_stopping: 'Core 遥操协调器正在停止。',
  rtc_ice_timeout: 'ICE candidate 收集超时。',
  rtc_offer_invalid: '浏览器未生成有效的 SDP offer。',
  rtc_answer_invalid: 'Core 返回的 SDP answer 无效。',
  teleop_signaling_unavailable: 'Core 尚未配置可用的 WebRTC 遥操信令。',
  signaling_content_type_required: 'WebRTC offer 必须使用 application/json。',
  signaling_offer_too_large: 'WebRTC SDP offer 超过大小限制。',
  invalid_signaling_offer: 'WebRTC SDP offer 格式无效。',
  webxr_not_secure: 'WebXR immersive-vr 需要 HTTPS 或浏览器认可的安全上下文。',
  webxr_unavailable: '当前浏览器没有可用的 WebXR immersive-vr。',
  webxr_not_ready: '只有本标签页 Active 会话与双 RTC 通道全部连接后才能进入 VR。',
  webxr_setup_failed: 'WebXR local-floor 或 WebGL 图层初始化失败。',
  webxr_pose_backpressure: 'Pose DataChannel 已超过 16 KiB 安全背压限制。',
  webxr_pose_too_large: '单个 Pose frame 超过 64 KiB 协议限制。',
  webxr_pose_send_failed: 'Pose frame 发送失败，RTC 与 XR 已关闭。',
  webxr_visibility_lost: '页面或 XR runtime 已不可见。',
  webxr_reference_space_reset: 'local-floor reference space 已被 runtime 重置。',
  rtc_connection_lost: 'WebRTC peer 或必要 DataChannel 已断开。',
  capture_tls_bootstrap_unavailable: 'Core 没有可安全提供给 OpenXR 头显 Capture 的当前 TLS 公开证书链。',
  capture_pairing_limit: '当前身份的一次性配对已达到上限，请等待旧配对过期。',
  capture_device_limit: '当前身份的采集设备已达到上限。',
  capture_not_found: '所选采集设备不存在或已撤销。',
  capture_forbidden: '所选采集设备不属于当前身份。',
  capture_not_ready: 'OpenXR 头显 Capture 尚未连接并进入 xr_standby。',
  capture_protocol_mismatch: 'OpenXR 头显 Capture 或 RTC frame 协议不兼容。',
  capture_assignment_conflict: '采集设备或会话已经有另一个 Capture assignment。',
  capture_attached: '采集设备仍附着会话；请先在 PC 执行 HOLD/Release。',
  capture_loss_pending: 'Capture 丢失后的安全 HOLD 尚未完成。',
  capture_signaling_failed: 'OpenXR 头显 Capture 信令失败；当前会话已 fail-close，必须新建会话。',
  signaling_source_conflict: '当前会话已选择另一种 RTC source；请 HOLD/Release 后新建会话。',
};

const HTTP_STATUS_TEXT = {
  400: '请求格式无效。',
  401: '登录已失效，请重新登录。',
  403: '当前身份无权执行该操作。',
  404: '请求的遥操资源不存在。',
  409: '当前遥操状态与操作冲突。',
  410: '会话已经终止。',
  422: '请求参数未通过校验。',
  500: 'Core 内部处理失败。',
  502: 'Driver 响应无效。',
  503: '遥操服务暂不可用。',
  504: 'Driver 响应超时。',
};

const SESSION_STATE_TEXT = {
  awaiting_confirmation: '等待明确 Live 硬件确认（Driver 尚未调用）',
  preparing: '准备中',
  active: '已占用 / Active',
  paused: '已暂停 / Paused（仍占用）',
  hold: 'Core HOLD / soft_stop（仍占用；姿态不可恢复）',
  released: '已释放',
  expired: '租约已过期',
  faulted: '故障终止',
  recovery_required: 'Core 重启恢复锁（普通写入已锁定）',
};

function canControl() {
  return Boolean(principal && (principal.role === 'operator' || principal.role === 'owner'));
}

function showNotice(message, kind = '') {
  notice.textContent = message;
  notice.className = `notice${kind ? ` ${kind}` : ''}`;
}

function resetConsoleData() {
  closeAllRtcConnections('控制台已重置');
  robots = [];
  sessionViews.clear();
  lastSessionByRobot.clear();
  sessionEvents.clear();
  pendingRobots.clear();
  browserHeartbeatFailures.clear();
  rtcFrameStates.clear();
  xrSessions.clear();
  xrSessionStats.clear();
  actionInFlight = false;
  captureListInFlight = false;
  captureActionInFlight = false;
  captures = [];
  capturePairing = null;
  selectedCaptureId = '';
  selectedCaptureSessionId = '';
  captureStatusMessage = '尚未读取采集设备。';
  captureStatusKind = '';
  renderCapturePanel();
}

function apiError(code, status = 0) {
  const error = new Error(
    ERROR_TEXT[code]
    || HTTP_STATUS_TEXT[status]
    || `请求失败（${code || `HTTP ${status}`}）`,
  );
  error.code = code;
  error.status = status;
  error.cancelled = code === 'request_cancelled';
  return error;
}

async function api(path, options = {}) {
  const generation = requestGeneration;
  const controller = new AbortController();
  const { timeoutMs = REQUEST_TIMEOUT_MS, ...fetchOptions } = options;
  const headers = new Headers(fetchOptions.headers || {});
  headers.set('X-Motus-Teleop-Client', TELEOP_CLIENT_ID);
  let timedOut = false;
  const timeout = window.setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  activeControllers.add(controller);

  let response;
  let data;
  try {
    response = await fetch(path, {
      ...fetchOptions,
      headers,
      signal: controller.signal,
    });
    data = await response.json().catch(() => ({}));
  } catch (error) {
    if (generation !== requestGeneration || (error && error.name === 'AbortError' && !timedOut)) {
      throw apiError('request_cancelled');
    }
    if (timedOut) throw apiError('request_timeout');
    throw apiError('network_error');
  } finally {
    window.clearTimeout(timeout);
    activeControllers.delete(controller);
  }

  if (generation !== requestGeneration) throw apiError('request_cancelled');
  if (timedOut) throw apiError('request_timeout');
  if (!response.ok) {
    const detail = data && data.detail;
    const code = detail && typeof detail === 'object' && typeof detail.code === 'string'
      ? detail.code
      : (typeof detail === 'string' ? detail : `http_${response.status}`);
    throw apiError(code, response.status);
  }
  return data.data;
}

function clearPollingTimers() {
  if (statusTimer !== null) window.clearInterval(statusTimer);
  if (slowTimer !== null) window.clearInterval(slowTimer);
  if (heartbeatTimer !== null) window.clearInterval(heartbeatTimer);
  statusTimer = null;
  slowTimer = null;
  heartbeatTimer = null;
}

function invalidateApiRequests() {
  requestGeneration += 1;
  activeControllers.forEach((controller) => controller.abort());
  activeControllers.clear();
  directoryInFlight = false;
  statusInFlight = false;
  eventsInFlight = false;
  heartbeatInFlight = false;
  captureListInFlight = false;
}

function stopPolling() {
  clearPollingTimers();
  invalidateApiRequests();
  actionInFlight = false;
  refreshButton.disabled = false;
  captureRefreshButton.disabled = false;
}

function showLogin(message = '') {
  stopPolling();
  principal = null;
  resetConsoleData();
  content.classList.add('hidden');
  logoutButton.classList.add('hidden');
  loginPanel.classList.remove('hidden');
  loginError.textContent = message;
  tokenInput.focus();
}

function fact(label, value) {
  const dt = document.createElement('dt');
  dt.textContent = label;
  const dd = document.createElement('dd');
  dd.textContent = value;
  return [dt, dd];
}

function formatSeconds(value) {
  if (value === null || value === undefined || value === '') return '—';
  const seconds = Number(value);
  if (!Number.isFinite(seconds)) return '—';
  return `${Math.max(0, seconds).toFixed(1)} 秒`;
}

function formatMilliseconds(value) {
  if (value === null || value === undefined || value === '') return '—';
  const milliseconds = Number(value);
  if (!Number.isFinite(milliseconds)) return '—';
  return `${Math.max(0, milliseconds).toFixed(0)} ms`;
}

function formatTimestamp(value) {
  const timestamp = Number(value);
  if (!Number.isFinite(timestamp) || timestamp <= 0) return '尚未确认';
  try {
    return new Date(timestamp * 1000).toLocaleTimeString('zh-CN', { hour12: false });
  } catch {
    return '时间无效';
  }
}

function displayValue(value, fallback = '—') {
  if (value === null || value === undefined || value === '') return fallback;
  return String(value);
}

function setCaptureStatus(message, kind = '') {
  captureStatusMessage = message;
  captureStatusKind = kind;
  renderCapturePanel();
}

function captureSessionCandidates() {
  return robots.flatMap((robot) => {
    const sessionId = robot.session?.id;
    const snapshot = sessionView(robot);
    if (!sessionId || !robot.session?.owned_by_client || !snapshot) return [];
    return [{
      ...snapshot,
      id: sessionId,
      owned_by_client: true,
      robot_id: robot.robot_id,
      driver_id: robot.driver_id,
    }];
  });
}

function selectedCapture() {
  return captures.find((capture) => capture.id === selectedCaptureId) || null;
}

function selectedCaptureSession() {
  return captureSessionCandidates().find(
    (session) => session.id === selectedCaptureSessionId,
  ) || null;
}

function replaceSelectOptions(select, options, selectedValue, emptyLabel) {
  const nextOptions = [];
  if (options.length === 0) {
    const empty = document.createElement('option');
    empty.value = '';
    empty.textContent = emptyLabel;
    nextOptions.push(empty);
  } else {
    options.forEach(({ value, label }) => {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = label;
      option.selected = value === selectedValue;
      nextOptions.push(option);
    });
  }
  select.replaceChildren(...nextOptions);
}

function captureAttachStatusText(code) {
  const local = {
    ready: '可手动 Attach；不会自动执行。',
    capture_session_required: '请选择本标签页会话。',
    capture_device_required: '请选择已配对的 OpenXR 头显 Capture。',
    capture_live_confirmation_required: 'Live 仍等待 PC 二次确认；确认前 Attach 禁用。',
    capture_session_not_active: '只有本标签页 owned Active 会话可以 Attach。',
    capture_session_contract_invalid: '会话缺少 mode/profile/capability digest，已 fail-closed。',
  };
  return local[code] || ERROR_TEXT[code] || displayValue(code);
}

function renderCapturePanel() {
  const visible = canControl();
  capturePanel.classList.toggle('hidden', !visible);
  if (!visible) return;
  if (
    capturePairing
    && Number.isFinite(capturePairing.expiresAt)
    && capturePairing.expiresAt <= Date.now() / 1000
  ) {
    capturePairing = null;
    captureStatusMessage = '一次性配对已过期；如需使用请在 PC 重新生成。';
    captureStatusKind = 'error';
  }

  const sessions = captureSessionCandidates();
  if (!captures.some((capture) => capture.id === selectedCaptureId)) {
    selectedCaptureId = captures[0]?.id || '';
  }
  if (!sessions.some((session) => session.id === selectedCaptureSessionId)) {
    selectedCaptureSessionId = sessions[0]?.id || '';
  }

  replaceSelectOptions(
    captureDeviceSelect,
    captures.map((capture) => ({
      value: capture.id,
      label: `${displayValue(capture.label, capture.id)} · ${capture.connected ? 'connected' : 'offline'} · ${displayValue(capture.observed_state)}`,
    })),
    selectedCaptureId,
    '没有已配对的采集设备',
  );
  replaceSelectOptions(
    captureSessionSelect,
    sessions.map((session) => ({
      value: session.id,
      label: `${displayValue(session.robot_id)} · ${displayValue(session.mode).toUpperCase()} · ${displayValue(session.state)} · ${displayValue(session.profile_id)}`,
    })),
    selectedCaptureSessionId,
    '没有本标签页 owned 会话',
  );

  const capture = selectedCapture();
  const session = selectedCaptureSession();
  const assignment = capture?.assignment || null;
  captureDeviceFacts.replaceChildren(
    ...fact('Capture ID', displayValue(capture?.id)),
    ...fact('连接 / presence', capture ? `${capture.connected ? 'connected' : 'offline'} · ${displayValue(capture.observed_state)}` : '—'),
    ...fact('客户端', capture ? `${displayValue(capture.client_kind)} · ${displayValue(capture.app_version)}` : '—'),
    ...fact('协议', capture ? `${displayValue(capture.capture_protocol)} · ${displayValue(capture.frame_protocol)}` : '—'),
    ...fact('Assignment', assignment ? `${displayValue(assignment.state)} · session ${displayValue(assignment.session_id)}` : 'none'),
  );

  let eligibility = captureAttachEligibility(session, capture);
  if (
    eligibility.allowed
    && session
    && (rtcInFlight.has(session.id) || rtcConnectionActive(rtcViews.get(session.id)))
  ) eligibility = { allowed: false, code: 'signaling_source_conflict' };
  const busy = captureActionInFlight || actionInFlight;
  capturePairButton.disabled = busy;
  captureRefreshButton.disabled = busy || captureListInFlight;
  captureLabelInput.disabled = busy;
  captureDeviceSelect.disabled = busy || captures.length === 0;
  captureSessionSelect.disabled = busy || sessions.length === 0;
  captureAttachButton.disabled = busy || !eligibility.allowed;
  captureAttachButton.title = captureAttachStatusText(eligibility.code);
  captureRevokeButton.disabled = busy || !capture || assignment !== null;

  captureBootstrap.classList.toggle('hidden', capturePairing === null);
  captureWssUrl.value = capturePairing?.coreWssUrl || '';
  capturePairingId.value = capturePairing?.pairingId || '';
  capturePairingCode.value = capturePairing?.pairingCode || '';
  captureCaBase64.value = capturePairing?.caCertificateBase64 || '';
  capturePairingExpiry.textContent = capturePairing
    ? `一次性配对过期：${formatTimestamp(capturePairing.expiresAt)}`
    : '—';
  captureStatus.textContent = `${captureStatusMessage} ${captureAttachStatusText(eligibility.code)}`;
  captureStatus.className = `capture-status${captureStatusKind ? ` ${captureStatusKind}` : ''}`;
}

async function loadCaptures(announce = false, allowDuringAction = false) {
  if (
    !canControl()
    || captureListInFlight
    || ((actionInFlight || captureActionInFlight) && !allowDuringAction)
  ) return;
  const generation = requestGeneration;
  captureListInFlight = true;
  renderCapturePanel();
  try {
    const nextCaptures = await api('/api/teleop/captures');
    if (generation !== requestGeneration) return;
    captures = Array.isArray(nextCaptures) ? nextCaptures : [];
    if (
      announce
      || captureStatusMessage === '尚未读取采集设备。'
      || captureStatusKind === 'error'
    ) {
      captureStatusMessage = `已读取 ${captures.length} 个采集设备。`;
      captureStatusKind = announce ? 'success' : '';
    }
  } catch (error) {
    if (error.cancelled || generation !== requestGeneration) return;
    if (error.status === 401) {
      showLogin('登录已失效，请重新输入 token。');
      return;
    }
    captureStatusMessage = `采集设备读取失败：${error.message}`;
    captureStatusKind = 'error';
  } finally {
    if (generation === requestGeneration) {
      captureListInFlight = false;
      renderCapturePanel();
      renderDevices();
    }
  }
}

async function createCapturePairing() {
  if (!canControl() || actionInFlight || captureActionInFlight) return;
  const label = captureLabelInput.value.trim();
  if (!label || label.length > 64) {
    setCaptureStatus('设备标签必须为 1–64 个字符。', 'error');
    return;
  }
  captureActionInFlight = true;
  renderCapturePanel();
  try {
    const pairing = await api('/api/teleop/capture-pairings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ label }),
    });
    capturePairing = captureBootstrapFields(pairing, window.location);
    captureStatusMessage = '一次性配对已生成；页面不会自动连接 OpenXR 头显 Capture。';
    captureStatusKind = 'success';
  } catch (error) {
    if (!error.cancelled) {
      capturePairing = null;
      captureStatusMessage = `生成配对失败：${error.message}`;
      captureStatusKind = 'error';
    }
  } finally {
    captureActionInFlight = false;
    renderCapturePanel();
  }
}

async function attachSelectedCapture() {
  if (!canControl() || actionInFlight || captureActionInFlight) return;
  const capture = selectedCapture();
  const session = selectedCaptureSession();
  const eligibility = captureAttachEligibility(session, capture);
  if (!eligibility.allowed) {
    setCaptureStatus(captureAttachStatusText(eligibility.code), 'error');
    return;
  }
  if (rtcInFlight.has(session.id) || rtcConnectionActive(rtcViews.get(session.id))) {
    setCaptureStatus(ERROR_TEXT.signaling_source_conflict, 'error');
    return;
  }
  captureActionInFlight = true;
  renderCapturePanel();
  try {
    await api(
      `/api/teleop/sessions/${encodeURIComponent(session.id)}/capture-attachment`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          capture_id: capture.id,
          mode: session.mode,
          profile_id: session.profile_id,
          capability_digest: session.capability_digest,
        }),
      },
    );
    captureStatusMessage = 'Capture assignment 已手动签发；等待 OpenXR 头显 Capture 的一次 SDP offer。';
    captureStatusKind = 'success';
    await loadCaptures(false, true);
  } catch (error) {
    if (!error.cancelled) {
      captureStatusMessage = `Attach 失败：${error.message}`;
      captureStatusKind = 'error';
    }
  } finally {
    captureActionInFlight = false;
    renderCapturePanel();
    renderDevices();
  }
}

async function revokeSelectedCapture() {
  if (!canControl() || actionInFlight || captureActionInFlight) return;
  const capture = selectedCapture();
  if (!capture || capture.assignment) return;
  captureActionInFlight = true;
  renderCapturePanel();
  try {
    await api(`/api/teleop/captures/${encodeURIComponent(capture.id)}`, {
      method: 'DELETE',
    });
    capturePairing = null;
    selectedCaptureId = '';
    captureStatusMessage = '设备 enrollment 已撤销；旧 credential 立即失效。';
    captureStatusKind = 'success';
    await loadCaptures(false, true);
  } catch (error) {
    if (!error.cancelled) {
      captureStatusMessage = `撤销失败：${error.message}`;
      captureStatusKind = 'error';
    }
  } finally {
    captureActionInFlight = false;
    renderCapturePanel();
  }
}

async function copyCaptureBootstrapField(fieldName) {
  const value = capturePairing?.[fieldName];
  if (!value || typeof navigator.clipboard?.writeText !== 'function') {
    setCaptureStatus('当前浏览器不能安全写入剪贴板。', 'error');
    return;
  }
  try {
    await navigator.clipboard.writeText(value);
    setCaptureStatus(
      fieldName === 'pairingCode'
        ? '一次性 pairing code 已复制；请勿粘贴到日志或 shell 历史。'
        : '启动字段已复制。',
      'success',
    );
  } catch {
    setCaptureStatus('剪贴板写入失败；页面不会把字段写入 console。', 'error');
  }
}

function webxrSupportText() {
  const origin = window.location.origin;
  if (!webxrSupport.secureContext) return `${origin} · 不可用：当前页面不是安全上下文`;
  if (!webxrSupport.apiAvailable) return `${origin} · 不可用：navigator.xr 缺失`;
  if (webxrSupport.immersiveVr === 'checking') {
    return `${origin} · 检测中：immersive-vr + local-floor`;
  }
  if (webxrSupport.immersiveVr !== 'supported') {
    return `${origin} · 不可用：${webxrSupport.detail}`;
  }
  return `${origin} · 支持 immersive-vr；进入时强制请求 local-floor`;
}

function renderWebxrSupport() {
  if (webxrSupportElement) webxrSupportElement.textContent = webxrSupportText();
}

function probeWebxrSupport() {
  if (!webxrSupport.secureContext) {
    webxrSupport.immersiveVr = 'unsupported';
    webxrSupport.detail = '需要 HTTPS 或安全上下文';
    renderWebxrSupport();
    return;
  }
  if (!webxrSupport.apiAvailable || typeof navigator.xr.isSessionSupported !== 'function') {
    webxrSupport.immersiveVr = 'unsupported';
    webxrSupport.detail = 'navigator.xr/isSessionSupported 不可用';
    renderWebxrSupport();
    return;
  }
  renderWebxrSupport();
  navigator.xr.isSessionSupported('immersive-vr').then((supported) => {
    webxrSupport.immersiveVr = supported === true ? 'supported' : 'unsupported';
    webxrSupport.detail = supported === true
      ? 'immersive-vr 可用；local-floor 将在用户点击时请求'
      : 'immersive-vr 不受支持';
    renderWebxrSupport();
    renderDevices();
  }).catch(() => {
    webxrSupport.immersiveVr = 'unsupported';
    webxrSupport.detail = 'immersive-vr 能力检测失败';
    renderWebxrSupport();
    renderDevices();
  });
}

function frameStateForSession(sessionId) {
  let state = rtcFrameStates.get(sessionId);
  if (!state) {
    state = createRtcFrameState();
    rtcFrameStates.set(sessionId, state);
  }
  return state;
}

function markSessionFrameReleased(sessionId) {
  if (!sessionId) return;
  const released = markRtcFrameDeadmanReleased(frameStateForSession(sessionId));
  rtcFrameStates.set(sessionId, released);
  const stats = xrSessionStats.get(sessionId);
  if (stats) {
    stats.rearmRequired = released.rearmRequired;
    stats.baseTwist = '本地已归零；RTC/XR 不再发送';
  }
}

function formatBaseTwist(value) {
  const linear = value?.linear;
  const angular = value?.angular;
  if (
    !Array.isArray(linear)
    || linear.length !== 3
    || !Array.isArray(angular)
    || angular.length !== 3
    || [...linear, ...angular].some((item) => !Number.isFinite(item))
  ) return 'invalid / fail-closed';
  const fixed = (item) => (Object.is(item, -0) ? 0 : item).toFixed(3);
  return `vx ${fixed(linear[0])} m/s · vy ${fixed(linear[1])} m/s · wz ${fixed(angular[2])} rad/s`;
}

function teleopDescriptor(robot) {
  const descriptor = robot?.teleop;
  return descriptor && typeof descriptor === 'object' ? descriptor : {};
}

function descriptorMode(robot) {
  const mode = teleopDescriptor(robot).mode;
  return mode === 'live' ? 'live' : 'shadow';
}

function descriptorProfile(robot) {
  const descriptor = teleopDescriptor(robot);
  return displayValue(descriptor.profile_id, 'recording');
}

function formatEffectors(capabilities) {
  const effectors = Array.isArray(capabilities?.effectors) ? capabilities.effectors : [];
  const outputs = capabilities?.outputs && typeof capabilities.outputs === 'object'
    ? capabilities.outputs
    : {};
  if (effectors.length === 0) return 'none';
  return effectors.map((name) => {
    const count = outputs[name]?.joint_count;
    return Number.isSafeInteger(count) ? `${name} (${count} joints)` : String(name);
  }).join(' · ');
}

function formatOutputCapabilities(capabilities) {
  const outputs = capabilities?.outputs;
  if (!outputs || typeof outputs !== 'object') return '—';
  return Object.entries(outputs).map(([name, output]) => {
    const state = output?.enabled === true ? 'enabled' : 'disabled';
    const count = Number.isSafeInteger(output?.joint_count)
      ? ` · ${output.joint_count} joints`
      : '';
    return `${name} ${state}${count}`;
  }).join(' · ') || '—';
}

function formatJointVector(value) {
  if (!Array.isArray(value) || value.some((item) => !Number.isFinite(item))) return '—';
  if (value.length === 0) return '[]';
  const preview = value.slice(0, 10).map((item) => item.toFixed(3)).join(', ');
  return `${value.length} joints · [${preview}${value.length > 10 ? ', …' : ''}]`;
}

function formatLatencySummary(value) {
  if (!value || typeof value !== 'object') return '—';
  return [
    `p50 ${formatMilliseconds(value.p50)}`,
    `p95 ${formatMilliseconds(value.p95)}`,
    `p99 ${formatMilliseconds(value.p99)}`,
    `n ${displayValue(value.count, '0')}`,
  ].join(' · ');
}

function percentile(sorted, ratio) {
  if (sorted.length === 0) return null;
  const index = Math.min(sorted.length - 1, Math.ceil(sorted.length * ratio) - 1);
  return sorted[Math.max(0, index)];
}

function summarizeRttSamples(samples) {
  const finite = samples.filter((value) => Number.isFinite(value) && value >= 0).sort((a, b) => a - b);
  if (finite.length === 0) return null;
  const last = samples[samples.length - 1];
  return {
    last: Number.isFinite(last) && last >= 0 ? last : null,
    p50: percentile(finite, 0.5),
    p95: percentile(finite, 0.95),
    p99: percentile(finite, 0.99),
    count: finite.length,
  };
}

function statsForSession(sessionId) {
  let stats = xrSessionStats.get(sessionId);
  if (!stats) {
    stats = {
      state: 'not-entered',
      detail: '尚未进入 immersive-vr',
      tracking: 'head ✕ · left ✕ · right ✕',
      deadman: false,
      baseTwist: '尚未发送；是否存在由 capability binding 决定',
      sent: 0,
      dropped: 0,
      lastSequence: null,
      nextSequence: 0,
      rearmRequired: true,
      leftInput: 'left · input source 尚未出现',
      rightInput: 'right · input source 尚未出现',
      bufferedAmount: 0,
      lastSentAtMs: null,
      lastUiAtMs: 0,
    };
    xrSessionStats.set(sessionId, stats);
  }
  return stats;
}

function updateXrStats(sessionId, patch, render = true) {
  Object.assign(statsForSession(sessionId), patch);
  if (render) renderDevices();
}

function rtcReadyForXr(sessionId) {
  const view = rtcViews.get(sessionId);
  return Boolean(
    view
    && view.pc?.connectionState === 'connected'
    && view.control?.readyState === 'open'
    && view.pose?.readyState === 'open'
    && view.connectionState === 'connected'
    && view.controlState === 'open'
    && view.poseState === 'open',
  );
}

function ownedActiveRobotSession(robotId, sessionId) {
  const robot = currentRobot(robotId);
  const session = robot && sessionView(robot);
  return Boolean(
    robot
    && robot.session?.id === sessionId
    && robot.session?.owned_by_client
    && session?.state === 'active'
    && (session.mode !== 'live' || session.live_confirmed === true),
  );
}

function frameConfigurationForSession(session) {
  if (
    !session
    || !['shadow', 'live'].includes(session.mode)
    || !session.capabilities
    || typeof session.capabilities !== 'object'
  ) return null;
  return { mode: session.mode, capabilities: session.capabilities };
}

function xrSessionContractMatches(record, sessionId) {
  const robot = currentRobot(record.robotId);
  const session = robot && sessionView(robot);
  return Boolean(
    ownedActiveRobotSession(record.robotId, sessionId)
    && session
    && session.mode === record.mode
    && session.profile_id === record.profileId
    && session.capability_digest === record.capabilityDigest,
  );
}

function xrEntryReady(robotId, sessionId) {
  return Boolean(
    webxrSupport.secureContext
    && webxrSupport.apiAvailable
    && webxrSupport.immersiveVr === 'supported'
    && typeof navigator.xr?.requestSession === 'function'
    && ownedActiveRobotSession(robotId, sessionId)
    && rtcReadyForXr(sessionId),
  );
}

function uniqueTrackedPointerGripSources(xrSession) {
  const byHand = { left: [], right: [] };
  try {
    for (const source of xrSession.inputSources) {
      if (
        !source
        || source.targetRayMode !== 'tracked-pointer'
        || !source.gripSpace
        || !Object.hasOwn(byHand, source.handedness)
      ) continue;
      byHand[source.handedness].push(source);
    }
  } catch {
    return { left: null, right: null };
  }
  return {
    left: byHand.left.length === 1 ? byHand.left[0] : null,
    right: byHand.right.length === 1 ? byHand.right[0] : null,
  };
}

function stopXrSession(sessionId, reason) {
  markSessionFrameReleased(sessionId);
  const record = xrSessions.get(sessionId);
  xrSessions.delete(sessionId);
  updateXrStats(sessionId, {
    state: 'ended',
    detail: reason,
    deadman: false,
  }, false);
  if (!record) return;
  record.ending = true;
  if (record.session && record.rafId !== null) {
    try { record.session.cancelAnimationFrame(record.rafId); } catch { /* already ended */ }
  }
  if (record.session) {
    try {
      const ending = record.session.end();
      if (ending && typeof ending.catch === 'function') ending.catch(() => null);
    } catch {
      // The XR runtime already ended; authority is still released locally.
    }
  }
}

function failSafeCloseXr(sessionId, code, detail) {
  const message = ERROR_TEXT[code] || detail;
  closeRtcConnection(sessionId, `${detail}；fail-safe 已关闭 RTC 与 XR`);
  updateXrStats(sessionId, {
    state: 'error',
    detail,
    deadman: false,
  });
  showNotice(`WebXR fail-safe：${message}`, 'error');
}

function rtcConnectionActive(view) {
  return Boolean(
    view
    && view.pc
    && !['closed', 'disconnected', 'failed'].includes(view.connectionState),
  );
}

function updateRtcView(sessionId, peer, patch) {
  const current = rtcViews.get(sessionId);
  if (!current || current.pc !== peer) return;
  Object.assign(current, patch);
  renderDevices();
}

function sendPeerPing(sessionId, peer, control) {
  const view = rtcViews.get(sessionId);
  if (!view || view.pc !== peer || control.readyState !== 'open') return;
  const now = performance.now();
  for (const [requestId, startedAt] of view.pendingPings) {
    if (now - startedAt > RTC_PEER_PING_TIMEOUT_MS) view.pendingPings.delete(requestId);
  }
  peerPingSequence += 1;
  const requestId = `browser-${peerPingSequence}`;
  view.pendingPings.set(requestId, now);
  try {
    control.send(JSON.stringify({ type: 'peer_ping', request_id: requestId }));
  } catch {
    view.pendingPings.delete(requestId);
    throw new Error('teleop-control peer_ping send failed');
  }
}

function startPeerPings(sessionId, peer, control, onFailure) {
  const view = rtcViews.get(sessionId);
  if (!view || view.pc !== peer || view.pingTimer !== null) return;
  const send = () => {
    try {
      sendPeerPing(sessionId, peer, control);
    } catch {
      onFailure('teleop-control peer_ping 发送失败');
    }
  };
  send();
  view.pingTimer = window.setInterval(send, RTC_PEER_PING_MS);
}

function recordPeerPong(sessionId, peer, payload) {
  const view = rtcViews.get(sessionId);
  const requestId = payload?.request_id;
  if (!view || view.pc !== peer || typeof requestId !== 'string') return;
  const startedAt = view.pendingPings.get(requestId);
  if (!Number.isFinite(startedAt)) return;
  view.pendingPings.delete(requestId);
  const elapsed = performance.now() - startedAt;
  if (!Number.isFinite(elapsed) || elapsed < 0 || elapsed > RTC_PEER_PING_TIMEOUT_MS) return;
  view.rttSamples.push(elapsed);
  if (view.rttSamples.length > RTC_RTT_SAMPLE_LIMIT) view.rttSamples.shift();
  view.rttSummary = summarizeRttSamples(view.rttSamples);
}

function closeRtcConnection(sessionId, reason = '已主动关闭', { skipXrStop = false } = {}) {
  markSessionFrameReleased(sessionId);
  if (!skipXrStop) stopXrSession(sessionId, reason);
  const current = rtcViews.get(sessionId);
  if (!current) {
    rtcInFlight.delete(sessionId);
    renderDevices();
    return;
  }
  current.intentionalClose = true;
  if (current.pingTimer !== null && current.pingTimer !== undefined) {
    window.clearInterval(current.pingTimer);
  }
  try { current.control?.close(); } catch { /* already closed */ }
  try { current.pose?.close(); } catch { /* already closed */ }
  try { current.pc?.close(); } catch { /* already closed */ }
  rtcViews.set(sessionId, {
    connectionState: 'closed',
    controlState: 'closed',
    poseState: 'closed',
    detail: reason,
    rttSummary: current.rttSummary || null,
  });
  rtcInFlight.delete(sessionId);
  renderDevices();
}

function closeAllRtcConnections(reason = '已关闭') {
  const sessionIds = new Set([...rtcViews.keys(), ...xrSessions.keys()]);
  sessionIds.forEach((sessionId) => closeRtcConnection(sessionId, reason));
  rtcViews.clear();
  rtcInFlight.clear();
}

function handleXrSessionEnded(sessionId, record) {
  if (xrSessions.get(sessionId) !== record) return;
  xrSessions.delete(sessionId);
  markSessionFrameReleased(sessionId);
  updateXrStats(sessionId, {
    state: 'ended',
    detail: 'XR runtime 已结束；RTC 已同步关闭',
    deadman: false,
  }, false);
  closeRtcConnection(
    sessionId,
    'XR runtime 已结束；不会自动重进 VR 或重连 RTC',
    { skipXrStop: true },
  );
}

function requestNextXrFrame(sessionId, record) {
  if (xrSessions.get(sessionId) !== record || !record.session || record.ending) return;
  try {
    record.rafId = record.session.requestAnimationFrame((timestamp, frame) => {
      onXrFrame(timestamp, frame, sessionId, record);
    });
  } catch {
    failSafeCloseXr(sessionId, 'webxr_setup_failed', 'XR animation frame 无法继续');
  }
}

function poseIsActuallyTracked(pose) {
  if (!pose || typeof pose !== 'object') return null;
  try {
    return pose.emulatedPosition === true ? null : pose;
  } catch {
    return null;
  }
}

function trackingText(tracking) {
  const marker = (tracked) => (tracked ? '✓' : '✕');
  return [
    `head ${marker(tracking.head)}`,
    `left ${marker(tracking.left_controller)}`,
    `right ${marker(tracking.right_controller)}`,
  ].join(' · ');
}

function inputSourceDiagnostic(source, expectedHandedness) {
  if (!source) return `${expectedHandedness} · input source 缺失`;
  try {
    const gamepad = source.gamepad;
    const axesLength = Number.isSafeInteger(gamepad?.axes?.length)
      ? gamepad.axes.length
      : 'invalid';
    const buttonsLength = Number.isSafeInteger(gamepad?.buttons?.length)
      ? gamepad.buttons.length
      : 'invalid';
    const profiles = [];
    const profileCount = Number.isSafeInteger(source.profiles?.length)
      ? Math.min(source.profiles.length, 3)
      : 0;
    for (let index = 0; index < profileCount; index += 1) {
      const profile = source.profiles[index];
      if (typeof profile === 'string') profiles.push(profile.slice(0, 64));
    }
    return [
      `${expectedHandedness}→${displayValue(source.handedness)}`,
      `ray ${displayValue(source.targetRayMode)}`,
      `grip ${source.gripSpace ? 'yes' : 'no'}`,
      `mapping ${displayValue(gamepad?.mapping)}`,
      `axes ${axesLength}`,
      `buttons ${buttonsLength}`,
      `profiles ${profiles.length ? profiles.join(',') : 'none'}`,
    ].join(' · ');
  } catch {
    return `${expectedHandedness} · input source 读取失败`;
  }
}

function isTrackingOrDeadmanSafetyTransition(record, frame) {
  if (record.lastSentDeadman === true && frame.deadman !== true) return true;
  if (!record.lastSentTracking) return false;
  return Object.keys(record.lastSentTracking).some((key) => (
    record.lastSentTracking[key] === true && frame.tracking[key] !== true
  ));
}

function rememberDroppedFrameSafetyState(sessionId, previousState, nextState) {
  rtcFrameStates.set(sessionId, createRtcFrameState({
    nextSequence: previousState.nextSequence,
    clutchSequence: previousState.clutchSequence,
    deadmanActive: previousState.deadmanActive,
    rearmRequired: nextState.rearmRequired,
    lastMonotonicNs: previousState.lastMonotonicNs,
  }));
}

function renderXrFrame(record, viewerPose) {
  const baseLayer = record.session?.renderState?.baseLayer;
  if (!baseLayer || !record.gl) throw new Error('XR render layer missing');
  const gl = record.gl;
  gl.bindFramebuffer(gl.FRAMEBUFFER, baseLayer.framebuffer);
  gl.clearColor(0, 0, 0, 1);
  if (viewerPose) {
    for (const view of viewerPose.views) {
      const viewport = baseLayer.getViewport(view);
      if (viewport) gl.viewport(viewport.x, viewport.y, viewport.width, viewport.height);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    }
  } else {
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  }
}

function onXrFrame(timestamp, frame, sessionId, record) {
  if (xrSessions.get(sessionId) !== record || record.ending) return;
  record.rafId = null;
  if (
    document.visibilityState !== 'visible'
    || record.session.visibilityState !== 'visible'
  ) {
    failSafeCloseXr(sessionId, 'webxr_visibility_lost', '页面或 XR session 已不可见');
    return;
  }
  if (!xrSessionContractMatches(record, sessionId) || !rtcReadyForXr(sessionId)) {
    failSafeCloseXr(
      sessionId,
      'webxr_not_ready',
      'Core ownership/state/session capability 或 RTC 双通道已失效',
    );
    return;
  }

  const frameTimeMs = Number.isFinite(timestamp) ? timestamp : performance.now();
  if (frameTimeMs >= record.nextCoreHeartbeatAtMs && !heartbeatInFlight) {
    record.nextCoreHeartbeatAtMs = frameTimeMs + CORE_HEARTBEAT_MS;
    void renewOwnedSessions(true);
  }

  let viewerPose = null;
  try {
    viewerPose = frame.getViewerPose(record.referenceSpace);
    renderXrFrame(record, viewerPose);
  } catch {
    failSafeCloseXr(sessionId, 'webxr_setup_failed', 'XR viewer pose 或 WebGL frame 读取失败');
    return;
  }

  const sources = uniqueTrackedPointerGripSources(record.session);
  let leftGripPose = null;
  let rightGripPose = null;
  try {
    if (sources.left) leftGripPose = frame.getPose(sources.left.gripSpace, record.referenceSpace);
    if (sources.right) rightGripPose = frame.getPose(sources.right.gripSpace, record.referenceSpace);
  } catch {
    leftGripPose = null;
    rightGripPose = null;
  }

  let built;
  const previousFrameState = frameStateForSession(sessionId);
  try {
    built = buildRtcFrameV1(previousFrameState, {
      monotonicTimeMs: frameTimeMs,
      headPose: poseIsActuallyTracked(viewerPose),
      leftGripPose: poseIsActuallyTracked(leftGripPose),
      rightGripPose: poseIsActuallyTracked(rightGripPose),
      leftInputSource: sources.left,
      rightInputSource: sources.right,
      deadman: isDualSqueezeDeadmanRequested(sources.left, sources.right),
    }, record.frameConfiguration);
  } catch {
    failSafeCloseXr(sessionId, 'webxr_pose_send_failed', 'Pose frame 构建失败');
    return;
  }

  const safetyTransition = isTrackingOrDeadmanSafetyTransition(record, built.frame);
  if (frameTimeMs < record.nextPoseAtMs && !safetyTransition) {
    rememberDroppedFrameSafetyState(sessionId, previousFrameState, built.state);
    const stats = statsForSession(sessionId);
    stats.dropped += 1;
    stats.rearmRequired = built.state.rearmRequired;
    stats.leftInput = inputSourceDiagnostic(sources.left, 'left');
    stats.rightInput = inputSourceDiagnostic(sources.right, 'right');
    if (frameTimeMs - stats.lastUiAtMs >= XR_UI_REFRESH_MS) {
      stats.lastUiAtMs = frameTimeMs;
      renderDevices();
    }
    requestNextXrFrame(sessionId, record);
    return;
  }
  record.nextPoseAtMs = frameTimeMs + XR_SEND_INTERVAL_MS;
  rtcFrameStates.set(sessionId, built.state);

  const poseChannel = rtcViews.get(sessionId)?.pose;
  let payload;
  let payloadBytes;
  try {
    payload = JSON.stringify(built.frame);
    payloadBytes = new TextEncoder().encode(payload).byteLength;
  } catch {
    failSafeCloseXr(sessionId, 'webxr_pose_send_failed', 'Pose frame 序列化失败');
    return;
  }
  if (payloadBytes > POSE_MESSAGE_LIMIT_BYTES) {
    failSafeCloseXr(sessionId, 'webxr_pose_too_large', `Pose frame 为 ${payloadBytes} bytes`);
    return;
  }
  const bufferedAmount = Number(poseChannel?.bufferedAmount);
  if (
    !poseChannel
    || poseChannel.readyState !== 'open'
    || !Number.isFinite(bufferedAmount)
    || bufferedAmount < 0
    || bufferedAmount > POSE_BUFFER_HIGH_WATER_BYTES
    || bufferedAmount + payloadBytes > POSE_BUFFER_HIGH_WATER_BYTES
  ) {
    failSafeCloseXr(sessionId, 'webxr_pose_backpressure', 'Pose DataChannel 未打开或背压超限');
    return;
  }
  try {
    poseChannel.send(payload);
  } catch {
    failSafeCloseXr(sessionId, 'webxr_pose_send_failed', 'Pose DataChannel send() 抛出异常');
    return;
  }
  const bufferedAfterSend = Number(poseChannel.bufferedAmount);
  if (
    !Number.isFinite(bufferedAfterSend)
    || bufferedAfterSend < 0
    || bufferedAfterSend > POSE_BUFFER_HIGH_WATER_BYTES
  ) {
    failSafeCloseXr(sessionId, 'webxr_pose_backpressure', 'Pose send() 后背压超过 16 KiB');
    return;
  }

  const stats = statsForSession(sessionId);
  stats.state = 'active';
  stats.detail = `正在发送 raw local-floor ${record.mode} frames`;
  stats.tracking = trackingText(built.frame.tracking);
  stats.deadman = built.frame.deadman;
  stats.baseTwist = Object.hasOwn(built.frame, 'base_twist')
    ? formatBaseTwist(built.frame.base_twist)
    : 'base capability disabled · field omitted';
  stats.sent += 1;
  stats.lastSequence = built.frame.sequence;
  stats.nextSequence = built.state.nextSequence;
  stats.rearmRequired = built.state.rearmRequired;
  stats.leftInput = inputSourceDiagnostic(sources.left, 'left');
  stats.rightInput = inputSourceDiagnostic(sources.right, 'right');
  stats.bufferedAmount = bufferedAfterSend;
  stats.lastSentAtMs = frameTimeMs;
  record.lastSentDeadman = built.frame.deadman;
  record.lastSentTracking = { ...built.frame.tracking };
  if (frameTimeMs - stats.lastUiAtMs >= XR_UI_REFRESH_MS) {
    stats.lastUiAtMs = frameTimeMs;
    renderDevices();
  }
  requestNextXrFrame(sessionId, record);
}

async function initializeXrSession(sessionId, record, xrSession) {
  if (xrSessions.get(sessionId) !== record) {
    try { await xrSession.end(); } catch { /* request was cancelled while pending */ }
    return;
  }
  record.session = xrSession;
  const onEnd = () => handleXrSessionEnded(sessionId, record);
  const onVisibilityChange = () => {
    if (
      xrSessions.get(sessionId) === record
      && xrSession.visibilityState !== 'visible'
    ) {
      failSafeCloseXr(sessionId, 'webxr_visibility_lost', 'XR session visibility 已离开 visible');
    }
  };
  xrSession.addEventListener('end', onEnd, { once: true });
  xrSession.addEventListener('visibilitychange', onVisibilityChange);

  try {
    if (!xrEntryReady(record.robotId, sessionId)) throw new Error('entry authority changed');
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl', { alpha: false, antialias: false });
    if (!gl || typeof gl.makeXRCompatible !== 'function' || typeof XRWebGLLayer !== 'function') {
      throw new Error('WebGL XR compatibility unavailable');
    }
    await gl.makeXRCompatible();
    const referenceSpace = await xrSession.requestReferenceSpace('local-floor');
    const baseLayer = new XRWebGLLayer(xrSession, gl);
    xrSession.updateRenderState({ baseLayer });
    const onReferenceSpaceReset = () => {
      if (xrSessions.get(sessionId) === record) {
        failSafeCloseXr(
          sessionId,
          'webxr_reference_space_reset',
          'local-floor reference space reset；禁止自动恢复',
        );
      }
    };
    referenceSpace.addEventListener('reset', onReferenceSpaceReset, { once: true });
    if (
      xrSessions.get(sessionId) !== record
      || !xrEntryReady(record.robotId, sessionId)
      || xrSession.visibilityState !== 'visible'
      || document.visibilityState !== 'visible'
    ) throw new Error('entry authority or visibility changed');

    record.canvas = canvas;
    record.gl = gl;
    record.referenceSpace = referenceSpace;
    updateXrStats(sessionId, {
      state: 'active',
      detail: `immersive-vr active；raw local-floor / ${record.mode}`,
      deadman: false,
    });
    requestNextXrFrame(sessionId, record);
  } catch (error) {
    if (xrSessions.get(sessionId) !== record) return;
    const detail = error && typeof error.message === 'string'
      ? error.message
      : 'unknown WebXR setup error';
    failSafeCloseXr(sessionId, 'webxr_setup_failed', `WebXR 初始化失败：${detail}`);
  }
}

function requestImmersiveVr(robotId, sessionId) {
  if (!webxrSupport.secureContext) {
    showNotice(ERROR_TEXT.webxr_not_secure, 'error');
    return;
  }
  if (!webxrSupport.apiAvailable || webxrSupport.immersiveVr !== 'supported') {
    showNotice(ERROR_TEXT.webxr_unavailable, 'error');
    return;
  }
  if (!xrEntryReady(robotId, sessionId) || xrSessions.has(sessionId)) {
    showNotice(ERROR_TEXT.webxr_not_ready, 'error');
    return;
  }
  const robot = currentRobot(robotId);
  const session = robot && sessionView(robot);
  const frameConfiguration = frameConfigurationForSession(session);
  if (!session || !frameConfiguration) {
    showNotice(ERROR_TEXT.webxr_not_ready, 'error');
    return;
  }

  let sessionPromise;
  try {
    sessionPromise = navigator.xr.requestSession(
      'immersive-vr',
      { requiredFeatures: ['local-floor'] },
    );
  } catch (error) {
    showNotice(`WebXR requestSession 失败：${displayValue(error?.message)}`, 'error');
    return;
  }

  const record = {
    robotId,
    mode: session.mode,
    profileId: session.profile_id,
    capabilityDigest: session.capability_digest,
    frameConfiguration,
    session: null,
    referenceSpace: null,
    canvas: null,
    gl: null,
    rafId: null,
    ending: false,
    nextPoseAtMs: Number.NEGATIVE_INFINITY,
    nextCoreHeartbeatAtMs: performance.now() + CORE_HEARTBEAT_MS,
    lastSentDeadman: false,
    lastSentTracking: null,
  };
  xrSessions.set(sessionId, record);
  updateXrStats(sessionId, {
    state: 'requesting',
    detail: '用户已请求 immersive-vr + local-floor',
    deadman: false,
  });
  sessionPromise.then(
    (xrSession) => initializeXrSession(sessionId, record, xrSession),
    (error) => {
      if (xrSessions.get(sessionId) !== record) return;
      const detail = error && typeof error.message === 'string'
        ? error.message
        : 'requestSession rejected';
      failSafeCloseXr(sessionId, 'webxr_setup_failed', `WebXR 请求被拒绝：${detail}`);
    },
  );
}

function waitForIceGatheringComplete(peer) {
  if (peer.iceGatheringState === 'complete') return Promise.resolve();
  return new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      peer.removeEventListener('icegatheringstatechange', onStateChange);
      reject(apiError('rtc_ice_timeout'));
    }, RTC_ICE_GATHER_TIMEOUT_MS);
    function onStateChange() {
      if (peer.iceGatheringState !== 'complete') return;
      window.clearTimeout(timeout);
      peer.removeEventListener('icegatheringstatechange', onStateChange);
      resolve();
    }
    peer.addEventListener('icegatheringstatechange', onStateChange);
    onStateChange();
  });
}

async function connectRtc(robotId, sessionId) {
  const robot = currentRobot(robotId);
  if (
    !robot
    || !sessionId
    || !robot.session?.owned_by_client
    || rtcInFlight.has(sessionId)
    || captureAssignmentForSession(captures, sessionId) !== null
    || typeof RTCPeerConnection !== 'function'
  ) return;

  const session = sessionView(robot);
  if (!session || !ownedActiveRobotSession(robotId, sessionId)) return;
  closeRtcConnection(sessionId, '正在重新连接');
  rtcInFlight.add(sessionId);

  const peer = new RTCPeerConnection();
  const control = peer.createDataChannel('teleop-control', { ordered: true });
  const pose = peer.createDataChannel('teleop-pose', {
    ordered: false,
    maxRetransmits: 0,
  });
  rtcViews.set(sessionId, {
    pc: peer,
    control,
    pose,
    connectionState: peer.connectionState,
    controlState: control.readyState,
    poseState: pose.readyState,
    detail: '正在生成 SDP offer…',
    intentionalClose: false,
    pingTimer: null,
    pendingPings: new Map(),
    rttSamples: [],
    rttSummary: null,
  });
  renderDevices();

  const unexpectedRtcFailure = (detail) => {
    const current = rtcViews.get(sessionId);
    if (!current || current.pc !== peer || current.intentionalClose) return;
    failSafeCloseXr(sessionId, 'rtc_connection_lost', detail);
  };
  const syncPeerState = () => {
    updateRtcView(sessionId, peer, { connectionState: peer.connectionState });
    if (['closed', 'disconnected', 'failed'].includes(peer.connectionState)) {
      unexpectedRtcFailure(`RTCPeerConnection ${peer.connectionState}`);
    }
  };
  peer.addEventListener('connectionstatechange', syncPeerState);
  peer.addEventListener('iceconnectionstatechange', () => {
    updateRtcView(sessionId, peer, {
      detail: `ICE ${peer.iceConnectionState}`,
    });
    if (['closed', 'disconnected', 'failed'].includes(peer.iceConnectionState)) {
      unexpectedRtcFailure(`ICE ${peer.iceConnectionState}`);
    }
  });
  control.addEventListener('open', () => {
    updateRtcView(sessionId, peer, {
      controlState: control.readyState,
      detail: '控制通道已打开；Pose 通道等待 WebXR 用户点击进入。',
    });
    startPeerPings(sessionId, peer, control, unexpectedRtcFailure);
  });
  control.addEventListener('close', () => {
    updateRtcView(sessionId, peer, { controlState: control.readyState });
    unexpectedRtcFailure('teleop-control 已关闭');
  });
  control.addEventListener('error', () => unexpectedRtcFailure('teleop-control 出错'));
  control.addEventListener('message', (event) => {
    try {
      const payload = JSON.parse(String(event.data));
      recordPeerPong(sessionId, peer, payload);
      updateRtcView(sessionId, peer, {
        detail: payload.ok
          ? `Driver RTC 状态：${displayValue(payload.state)}`
          : `Driver RTC 拒绝：${displayValue(payload.error?.code)}`,
      });
    } catch {
      updateRtcView(sessionId, peer, { detail: 'Driver RTC 返回了无效消息。' });
    }
  });
  pose.addEventListener('open', () => {
    updateRtcView(sessionId, peer, { poseState: pose.readyState });
  });
  pose.addEventListener('close', () => {
    updateRtcView(sessionId, peer, { poseState: pose.readyState });
    unexpectedRtcFailure('teleop-pose 已关闭');
  });
  pose.addEventListener('error', () => unexpectedRtcFailure('teleop-pose 出错'));

  try {
    const offer = await peer.createOffer();
    await peer.setLocalDescription(offer);
    updateRtcView(sessionId, peer, { detail: '正在收集 ICE candidates…' });
    await waitForIceGatheringComplete(peer);
    const local = peer.localDescription;
    if (!local || local.type !== 'offer' || !local.sdp) throw apiError('rtc_offer_invalid');
    updateRtcView(sessionId, peer, { detail: 'Core 正在代理一次性 offer…' });
    const answer = await api(
      `/api/teleop/sessions/${encodeURIComponent(sessionId)}/signaling/offer`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ type: 'offer', sdp: local.sdp }),
        timeoutMs: RTC_SIGNAL_TIMEOUT_MS,
      },
    );
    if (rtcViews.get(sessionId)?.pc !== peer) throw apiError('request_cancelled');
    if (!answer || answer.type !== 'answer' || typeof answer.sdp !== 'string') {
      throw apiError('rtc_answer_invalid');
    }
    await peer.setRemoteDescription(answer);
    updateRtcView(sessionId, peer, {
      connectionState: peer.connectionState,
      detail: 'SDP answer 已应用，等待双通道打开…',
    });
    showNotice(`WebRTC ${session.mode} 已完成信令，正在建立数据通道。`, 'success');
  } catch (error) {
    const superseded = rtcViews.get(sessionId)?.pc !== peer;
    if (!superseded && !error.cancelled) {
      showNotice(`WebRTC 连接失败：${error.message}`, 'error');
      closeRtcConnection(sessionId, `连接失败：${error.message}`);
    }
  } finally {
    if (rtcViews.get(sessionId)?.pc === peer) rtcInFlight.delete(sessionId);
    renderDevices();
  }
}

function visibleSessionId(robot) {
  if (robot.session && robot.session.busy) return robot.session.id || null;
  return lastSessionByRobot.get(robot.robot_id) || null;
}

function sessionView(robot) {
  const sessionId = visibleSessionId(robot);
  if (!sessionId) return robot.session || null;
  const record = sessionViews.get(sessionId);
  return record && record.snapshot ? record.snapshot : robot.session;
}

function coreRemaining(robot, session) {
  if (!session) return null;
  const sessionId = visibleSessionId(robot);
  const record = sessionId ? sessionViews.get(sessionId) : null;
  const initial = Number(session.remaining_seconds);
  if (!Number.isFinite(initial)) return null;
  if (!record || !record.receivedAt) return initial;
  return Math.max(0, initial - ((Date.now() - record.receivedAt) / 1000));
}

function rememberSession(snapshot, statusFailure = '') {
  if (!snapshot || !snapshot.id) return;
  const previous = sessionViews.get(snapshot.id) || {};
  sessionViews.set(snapshot.id, {
    ...previous,
    snapshot,
    receivedAt: Date.now(),
    statusFailure,
  });
  if (snapshot.robot_id) lastSessionByRobot.set(snapshot.robot_id, snapshot.id);
  if (
    snapshot.state !== 'active'
    && (
      xrSessions.has(snapshot.id)
      || rtcInFlight.has(snapshot.id)
      || rtcConnectionActive(rtcViews.get(snapshot.id))
    )
  ) {
    closeRtcConnection(
      snapshot.id,
      `Core state=${displayValue(snapshot.state)}；本地已先解除 deadman 并关闭 XR/RTC`,
    );
  }
}

function enforceInteractiveSessionSafety() {
  const sessionIds = new Set([...rtcViews.keys(), ...rtcInFlight, ...xrSessions.keys()]);
  sessionIds.forEach((sessionId) => {
    const view = rtcViews.get(sessionId);
    if (!xrSessions.has(sessionId) && !rtcInFlight.has(sessionId) && !rtcConnectionActive(view)) {
      return;
    }
    const robot = robots.find((candidate) => candidate.session?.id === sessionId);
    const session = robot && sessionView(robot);
    if (!robot || !robot.session?.owned_by_client || session?.state !== 'active') {
      closeRtcConnection(
        sessionId,
        'Core 会话不再是本标签页 owned Active；本地 fail-safe 已关闭',
      );
    }
  });
}

function markStatusFailure(sessionId, message) {
  const previous = sessionViews.get(sessionId);
  if (previous) previous.statusFailure = message;
}

function makeActionButton(label, action, robot, disabled = false, variant = 'secondary') {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = `button ${variant}`;
  button.textContent = label;
  button.dataset.action = action;
  button.disabled = (
    disabled
    || actionInFlight
    || captureActionInFlight
    || pendingRobots.has(robot.id)
  );
  button.addEventListener('click', () => performAction(robot.id, action));
  return button;
}

function makeRtcButton(robot, sessionId, sessionState) {
  const view = rtcViews.get(sessionId);
  const connected = rtcConnectionActive(view);
  const captureAssignment = captureAssignmentForSession(captures, sessionId);
  const button = document.createElement('button');
  button.type = 'button';
  button.className = `button ${connected ? 'warning' : 'secondary'}`;
  button.textContent = connected
    ? '断开 Direct RTC（进入 HOLD）'
    : `Direct WebXR fallback：连接 RTC ${descriptorMode(robot)}`;
  button.disabled = (
    actionInFlight
    || captureActionInFlight
    || pendingRobots.has(robot.id)
    || rtcInFlight.has(sessionId)
    || captureAssignment !== null
    || typeof RTCPeerConnection !== 'function'
    || sessionState !== 'active'
  );
  button.addEventListener('click', () => {
    if (connected) {
      closeRtcConnection(sessionId, '操作员断开；Driver 将进入 HOLD');
      showNotice('WebRTC 已断开；不会自动重连，Driver 将进入 HOLD。', 'error');
    } else {
      connectRtc(robot.id, sessionId);
    }
  });
  return button;
}

function makeXrButton(robot, sessionId) {
  const entered = xrSessions.has(sessionId);
  const button = document.createElement('button');
  button.type = 'button';
  button.className = `button ${entered ? 'warning' : 'primary'}`;
  button.textContent = entered ? '退出 VR（关闭 RTC）' : '进入 Quest VR';
  button.disabled = entered ? false : (
    actionInFlight
    || pendingRobots.has(robot.id)
    || !xrEntryReady(robot.id, sessionId)
  );
  button.addEventListener('click', () => {
    if (xrSessions.has(sessionId)) {
      closeRtcConnection(sessionId, '操作员退出 VR；RTC 已同步关闭且不会自动重连');
      showNotice('已退出 VR 并关闭 RTC；如需继续必须重新手动连接与进入。', 'error');
      return;
    }
    requestImmersiveVr(robot.id, sessionId);
  });
  return button;
}

function appendDriverFacts(facts, session, sessionId) {
  const driver = session && session.driver;
  if (!driver) {
    facts.append(...fact('Driver 状态', '等待 Core 缓存更新'));
  } else {
    const lease = driver.lease || {};
    const leaseState = [
      lease.fresh ? 'fresh' : 'stale',
      `age ${formatMilliseconds(lease.age_ms)}`,
      `timeout ${formatMilliseconds(lease.timeout_ms)}`,
      lease.expired_latched ? 'expired-latched' : 'not-latched',
    ].join(' · ');
    facts.append(
      ...fact('Driver 状态', displayValue(driver.state)),
      ...fact('Driver 原因', displayValue(driver.reason, '无')),
      ...fact('Driver authority', driver.authority_valid ? 'valid' : 'invalid'),
      ...fact('Driver 租约', leaseState),
    );
    const rtc = driver.rtc || {};
    const channels = rtc.channels || {};
    facts.append(
      ...fact('Driver RTC', rtc.connected ? 'connected' : 'disconnected'),
      ...fact(
        'Driver channels',
        `control ${channels['teleop-control'] ? 'open' : 'closed'} · pose ${channels['teleop-pose'] ? 'open' : 'closed'}`,
      ),
    );
    const pose = driver.pose || {};
    facts.append(
      ...fact(
        'Driver pose freshness',
        `${pose.fresh ? 'fresh' : 'stale'} · age ${formatMilliseconds(pose.age_ms)}`,
      ),
      ...fact('Driver pose.latest_sequence', displayValue(pose.latest_sequence, '尚未记录')),
    );
    const dispatch = driver.dispatch || {};
    const decision = displayValue(dispatch.last_decision, '尚无');
    const sequenceLabel = dispatch.kind === 'hardware'
      ? 'Driver hardware published sequence'
      : 'Driver would-apply sequence';
    const outputSequence = dispatch.kind === 'hardware'
      ? dispatch.last_published_sequence
      : dispatch.last_would_apply_sequence;
    facts.append(
      ...fact(
        'Driver dispatch contract',
        `${displayValue(dispatch.contract)} · ${displayValue(dispatch.kind)}`,
      ),
      ...fact('Driver dispatch state', displayValue(dispatch.state)),
      ...fact('Driver dispatch.last_decision', decision),
      ...fact('Driver last_admitted_sequence', displayValue(dispatch.last_admitted_sequence, '尚无')),
      ...fact(sequenceLabel, displayValue(outputSequence, '尚无')),
      ...fact(
        'Safe-stop ACK',
        dispatch.stop_acknowledged === true ? 'acknowledged' : 'NOT ACKNOWLEDGED',
      ),
      ...fact('Driver fault code', displayValue(dispatch.fault_code, '无')),
    );

    const diagnostics = driver.diagnostics;
    if (diagnostics && typeof diagnostics === 'object') {
      const transport = diagnostics.transport || {};
      facts.append(
        ...fact(
          'Driver transport timing',
          `rtc ${formatMilliseconds(transport.rtc_rtt_ms)} · pose age ${formatMilliseconds(transport.pose_age_ms)} · ${displayValue(transport.frame_rate_hz)} Hz`,
        ),
        ...fact(
          'Driver frame counters',
          `rx ${displayValue(transport.frames_received, '0')} · rejected ${displayValue(transport.frames_rejected, '0')} · gaps ${displayValue(transport.sequence_gaps, '0')} · mailbox replace ${displayValue(transport.mailbox_replacements, '0')}`,
        ),
      );
      const latency = diagnostics.latency_ms || {};
      const adapterLatencyLabel = dispatch.kind === 'hardware'
        ? 'Latency adapter publish'
        : 'Latency adapter projection';
      for (const [label, key] of [
        ['Latency receive→admit', 'receive_to_admit'],
        ['Latency mailbox', 'mailbox_wait'],
        ['Latency IK', 'ik'],
        [adapterLatencyLabel, 'adapter_apply'],
        ['Latency next LowState feedback arrival', 'robot_follow'],
      ]) facts.append(...fact(label, formatLatencySummary(latency[key])));
    }

    const output = driver.output;
    if (output && typeof output === 'object') {
      facts.append(
        ...fact('Output evidence profile', displayValue(output.profile_id)),
        ...fact(
          'Output evidence hardware path',
          output.hardware_output === true ? 'true · hardware path' : 'false · recording/shadow path',
        ),
        ...fact('Output evidence state', displayValue(output.state)),
        ...fact('Target joints', formatJointVector(output.target_joint_positions_rad)),
        ...fact('Measured joints', formatJointVector(output.measured_joint_positions_rad)),
        ...fact(
          'Max joint error',
          Number.isFinite(output.max_abs_error_rad)
            ? `${output.max_abs_error_rad.toFixed(4)} rad`
            : '—',
        ),
        ...fact('Arm command weight', displayValue(output.arm_sdk_weight)),
        ...fact('Command age', formatMilliseconds(output.command_age_ms)),
        ...fact('Output fault / HOLD', displayValue(output.fault_reason, '无')),
      );
    }
  }

  const driverHeartbeat = session && session.driver_heartbeat;
  if (driverHeartbeat) {
    facts.append(
      ...fact('Driver HB 最近确认', formatTimestamp(driverHeartbeat.last_confirmed_at)),
      ...fact(
        'Driver HB 失败',
        `${displayValue(driverHeartbeat.state)} · 连续 ${displayValue(driverHeartbeat.consecutive_failures, '0')} 次`,
      ),
    );
  }

  const statusFailure = sessionId && sessionViews.get(sessionId)?.statusFailure;
  const browserFailure = sessionId && browserHeartbeatFailures.get(sessionId);
  if (statusFailure) facts.append(...fact('状态读取失败', statusFailure));
  if (browserFailure) facts.append(...fact('Core 租约续期失败', browserFailure));
}

function appendBrowserRtcFacts(facts, sessionId) {
  if (!sessionId) return;
  facts.append(...fact('Browser WebXR capability', webxrSupportText()));
  facts.append(...fact('Transport route', 'Browser ⇄ RTC DataChannel ⇄ Driver（Core 仅代理信令）'));
  if (typeof RTCPeerConnection !== 'function') {
    facts.append(...fact('Browser WebRTC', '当前浏览器不支持'));
  } else {
    const view = rtcViews.get(sessionId);
    if (!view) {
      facts.append(...fact('Browser WebRTC', '尚未连接'));
    } else {
      facts.append(
        ...fact('Browser WebRTC', displayValue(view.connectionState)),
        ...fact(
          'Browser channels',
          `control ${displayValue(view.controlState)} · pose ${displayValue(view.poseState)}`,
        ),
        ...fact('WebRTC 诊断', displayValue(view.detail)),
        ...fact(
          'Browser peer RTT（同钟）',
          view.rttSummary ? formatLatencySummary(view.rttSummary) : '尚无匹配 peer_ping 回执',
        ),
      );
    }
  }
  const stats = statsForSession(sessionId);
  facts.append(
    ...fact('WebXR session', `${displayValue(stats.state)} · ${displayValue(stats.detail)}`),
    ...fact('WebXR tracking', stats.tracking),
    ...fact('Quest left input', stats.leftInput),
    ...fact('Quest right input', stats.rightInput),
    ...fact('WebXR deadman', stats.deadman ? '双 squeeze 已按下 / true' : 'released / false'),
    ...fact('WebXR optional base_twist', stats.baseTwist),
    ...fact(
      'WebXR re-arm',
      stats.rearmRequired ? '需松开后重握' : '已观察松开，可重新双握',
    ),
    ...fact(
      'Pose frames',
      `sent ${stats.sent} · dropped ${stats.dropped} · last ${displayValue(stats.lastSequence)} · next ${displayValue(stats.nextSequence)}`,
    ),
    ...fact(
      'Pose transport',
      `buffered ${displayValue(stats.bufferedAmount)} B / 16384 B · last send age ${stats.lastSentAtMs === null ? '—' : formatMilliseconds(performance.now() - stats.lastSentAtMs)}`,
    ),
  );
}

function renderSessionFacts(facts, robot, session) {
  const busy = Boolean(robot.session && robot.session.busy);
  const sessionId = visibleSessionId(robot);
  if (!busy) {
    facts.append(...fact('Core 会话', '空闲'));
    if (!session || !TERMINAL_STATES.has(session.state)) return;
    facts.append(...fact('本页最后终态', SESSION_STATE_TEXT[session.state] || displayValue(session.state)));
    appendDriverFacts(facts, session, sessionId);
    appendBrowserRtcFacts(facts, sessionId);
    return;
  }

  if (robot.authority_guard) {
    facts.append(
      ...fact('Core 会话', SESSION_STATE_TEXT.recovery_required),
      ...fact('恢复 Driver', displayValue(robot.authority_guard.driver_id)),
      ...fact('恢复阶段', displayValue(robot.authority_guard.phase, 'recovery_required')),
    );
    return;
  }

  facts.append(
    ...fact('Core 会话', SESSION_STATE_TEXT[session && session.state] || displayValue(session && session.state, '已占用')),
    ...fact('Core 15s 倒计时', formatSeconds(coreRemaining(robot, session))),
    ...fact('Session mode', displayValue(session?.mode)),
    ...fact('Session profile', displayValue(session?.profile_id)),
    ...fact('Session effectors', formatEffectors(session?.capabilities)),
    ...fact(
      'Live hardware confirmation',
      session?.mode === 'live'
        ? (session.live_confirmed === true ? 'confirmed' : 'NOT CONFIRMED · no Driver authority')
        : 'not applicable · Shadow',
    ),
  );
  if (robot.session.principal_id) {
    let ownership = robot.session.principal_id;
    if (robot.session.owned_by_client) ownership = '本标签页';
    else if (robot.session.owned_by_me) ownership = '本人其他标签页';
    facts.append(...fact('会话持有人', ownership));
  }
  appendDriverFacts(facts, session, sessionId);
  appendBrowserRtcFacts(facts, sessionId);
}

function renderEvents(sessionId) {
  const section = document.createElement('section');
  section.className = 'events-panel';
  const heading = document.createElement('h4');
  heading.textContent = '最近会话事件';
  section.appendChild(heading);

  const events = sessionEvents.get(sessionId) || [];
  if (events.length === 0) {
    const empty = document.createElement('p');
    empty.className = 'events-empty';
    empty.textContent = '暂无事件或尚未完成首次读取。';
    section.appendChild(empty);
    return section;
  }

  const list = document.createElement('ol');
  list.className = 'event-list';
  events.slice(0, 12).forEach((event) => {
    const item = document.createElement('li');
    const main = document.createElement('span');
    main.className = 'event-main';
    main.textContent = `${displayValue(event.event_type)} · ${displayValue(event.decision)}`;
    const detail = document.createElement('span');
    detail.className = 'event-detail';
    detail.textContent = `${formatTimestamp(event.created_at)} · ${displayValue(event.reason, '无原因')}`;
    item.append(main, detail);
    list.appendChild(item);
  });
  section.appendChild(list);
  return section;
}

function renderControls(robot, session) {
  if (!canControl()) return null;

  const controls = document.createElement('div');
  controls.className = 'session-controls';
  const busy = Boolean(robot.session && robot.session.busy);
  const sessionId = robot.session && robot.session.id;
  const ownedByClient = Boolean(robot.session && robot.session.owned_by_client);
  const ownerOverride = principal.role === 'owner' && Boolean(sessionId) && !ownedByClient;

  if (robot.authority_guard) {
    if (principal.role === 'owner') {
      controls.appendChild(makeActionButton(
        '验证安全并解除重启锁',
        'reconcile-guard',
        robot,
        false,
        'warning',
      ));
    }
    const hint = document.createElement('p');
    hint.className = 'control-hint busy';
    hint.textContent = principal.role === 'owner'
      ? '只会读取 Driver status，并在必要时调用无旧 fence 的 lifecycle stop；不会恢复旧会话。'
      : '普通写入已锁定；只有 owner 能验证安全停机并解除恢复锁。';
    controls.appendChild(hint);
    return controls;
  }

  if (!busy) {
    controls.appendChild(makeActionButton(
      `Acquire ${descriptorMode(robot).toUpperCase()}`,
      'acquire',
      robot,
      !robot.teleop_ready,
      'primary',
    ));
    if (!robot.teleop_ready) {
      const hint = document.createElement('p');
      hint.className = 'control-hint';
      hint.textContent = '只有 trusted、online 且通用描述协议有效的 Driver 可以 Acquire。';
      controls.appendChild(hint);
    }
    return controls;
  }

  if (!ownedByClient && !ownerOverride) {
    const hint = document.createElement('p');
    hint.className = 'control-hint busy';
    hint.textContent = robot.session.owned_by_me
      ? '会话属于本人的另一个标签页；本标签页不能续租或控制。'
      : '该机器人正由其他操作员占用。';
    controls.appendChild(hint);
    return controls;
  }

  if (ownerOverride) {
    const warning = document.createElement('p');
    warning.className = 'owner-warning';
    warning.textContent = 'Owner 手动 override：自动续租与离页释放仍不会作用于此会话。';
    controls.appendChild(warning);
  }
  const state = session && session.state;
  if (state === 'awaiting_confirmation') {
    const warning = document.createElement('section');
    warning.className = 'live-confirmation';
    const heading = document.createElement('h4');
    heading.textContent = 'LIVE 硬件输出尚未确认';
    const detail = document.createElement('p');
    detail.textContent = [
      `Robot ${displayValue(robot.robot_id)}`,
      `Driver ${displayValue(robot.driver_id)}`,
      `Profile ${displayValue(session.profile_id)}`,
      `Effectors ${formatEffectors(session.capabilities)}`,
      '当前仅内存 reservation；Driver 未 prepare，RTC/heartbeat 禁用',
    ].join(' · ');
    warning.append(heading, detail);
    if (ownedByClient && session.mode === 'live' && session.live_confirmed !== true) {
      warning.appendChild(makeActionButton(
        '我确认：启用 LIVE 硬件输出',
        'confirm-live',
        robot,
        false,
        'danger',
      ));
    }
    controls.append(
      warning,
      makeActionButton('Release', 'release', robot, false, 'warning'),
    );
    return controls;
  }
  if (ownedByClient) {
    const fallback = document.createElement('p');
    fallback.className = 'control-hint';
    fallback.textContent = captureAssignmentForSession(captures, sessionId)
      ? 'Capture 已作为本会话 RTC source；Direct WebXR fallback 禁用且不会静默切换。'
      : 'Direct WebXR fallback 是显式同页模式；必须手动连接 RTC、再手动进入 VR。';
    controls.appendChild(fallback);
    controls.append(
      makeRtcButton(robot, sessionId, state),
      makeXrButton(robot, sessionId),
    );
  }
  controls.append(
    makeActionButton('Pause', 'pause', robot, !['active', 'hold'].includes(state)),
    makeActionButton('HOLD', 'soft-stop', robot, state !== 'active', 'danger'),
    makeActionButton('Release', 'release', robot, false, 'warning'),
  );
  if (state === 'paused') {
    const warning = document.createElement('p');
    warning.className = 'paused-warning';
    warning.textContent = 'Pause 仍占用机器人；此控制台没有 Resume。恢复必须先 Release，再重新 Acquire。';
    controls.appendChild(warning);
  }
  if (state === 'hold') {
    const warning = document.createElement('p');
    warning.className = 'paused-warning';
    warning.textContent = 'Core state=hold / Driver reason=soft_stop 时，Pose 或重连不能恢复控制；必须先 Release，再重新 Acquire。';
    controls.appendChild(warning);
  }
  const driverReason = session?.driver?.reason;
  if (
    state === 'active'
    && RECOVERABLE_DRIVER_HOLD_REASONS.has(driverReason)
  ) {
    const hint = document.createElement('p');
    hint.className = 'control-hint';
    hint.textContent = `Core 仍为 Active，Driver ${driverReason} 可由操作者在需要时手动重连 RTC、重进 VR，先松开再双握以产生更大的 clutch_sequence；不会自动恢复。`;
    controls.appendChild(hint);
  }
  return controls;
}

function renderRobot(robot) {
  const card = document.createElement('article');
  const busy = Boolean(robot.session && robot.session.busy);
  const mode = descriptorMode(robot);
  card.className = `device-card ${mode === 'live' ? 'live-mode' : 'shadow-mode'} ${busy ? 'busy' : (robot.teleop_ready ? 'ready' : 'blocked')}`;

  const header = document.createElement('header');
  const titleWrap = document.createElement('div');
  const title = document.createElement('h3');
  title.className = 'device-name';
  title.textContent = robot.name || robot.id;
  const id = document.createElement('div');
  id.className = 'device-id';
  id.textContent = `Driver: ${robot.driver_id} · Robot: ${robot.robot_id}`;
  titleWrap.append(title, id);

  const badge = document.createElement('span');
  badge.className = `status-badge ${busy ? 'busy' : (robot.teleop_ready ? 'ready' : 'blocked')}`;
  badge.textContent = robot.authority_guard
    ? '重启安全锁定'
    : (busy ? `${mode.toUpperCase()} 会话占用中` : (robot.teleop_ready ? `${mode.toUpperCase()} 就绪` : '不可遥操'));
  header.append(titleWrap, badge);

  const facts = document.createElement('dl');
  facts.className = 'device-facts';
  const descriptor = teleopDescriptor(robot);
  const descriptorCapabilities = descriptor.capabilities || {};
  facts.append(
    ...fact('Driver ID', robot.driver_id),
    ...fact('Robot authority', robot.robot_id),
    ...fact('信任状态', robot.trust_state),
    ...fact('在线状态', robot.online ? 'online' : 'offline'),
    ...fact('遥操声明', robot.teleop_declared ? '已声明' : '未声明'),
    ...fact('Configured mode', mode),
    ...fact('Configured profile', descriptorProfile(robot)),
    ...fact('Configured effectors', formatEffectors(descriptorCapabilities)),
    ...fact('Configured outputs', formatOutputCapabilities(descriptorCapabilities)),
    ...fact(
      'Configured hardware output',
      descriptor.actuation_enabled === true ? 'true · Live confirmation required' : 'false · Shadow recording',
    ),
    ...fact('判定', REASON_TEXT[robot.reason] || robot.reason),
  );
  const session = sessionView(robot);
  renderSessionFacts(facts, robot, session);

  const capabilities = document.createElement('div');
  capabilities.className = 'capability-list';
  (robot.tools || []).forEach((tool) => {
    const chip = document.createElement('span');
    chip.textContent = tool;
    capabilities.appendChild(chip);
  });

  card.append(header, facts, capabilities);
  const controls = renderControls(robot, session);
  if (controls) card.appendChild(controls);
  const sessionId = visibleSessionId(robot);
  if (sessionId && canControl()) card.appendChild(renderEvents(sessionId));
  return card;
}

function renderDevices() {
  const grid = document.getElementById('device-grid');
  const empty = document.getElementById('devices-empty');
  grid.replaceChildren(...robots.map(renderRobot));
  empty.classList.toggle('hidden', robots.length !== 0);
  renderCapturePanel();
}

async function loadDevices(announce = true, allowDuringAction = false) {
  if (directoryInFlight || (actionInFlight && !allowDuringAction)) return;
  const generation = requestGeneration;
  directoryInFlight = true;
  refreshButton.disabled = true;
  try {
    const nextRobots = await api('/api/teleop/robots');
    if (generation !== requestGeneration) return;
    robots = Array.isArray(nextRobots) ? nextRobots : [];
    enforceInteractiveSessionSafety();
    renderDevices();
    if (announce) {
      const permission = canControl()
        ? '可以显式管理本标签页的 Shadow 或 Live 会话。'
        : 'viewer 仅可查看。';
      showNotice(`已发现 ${robots.length} 个 Driver；${permission}`, 'success');
    }
  } catch (error) {
    if (error.cancelled) return;
    if (error.status === 401) {
      showLogin('登录已失效，请重新输入 token。');
      showNotice(HTTP_STATUS_TEXT[401], 'error');
      return;
    }
    if (generation === requestGeneration) showNotice(`设备读取失败：${error.message}`, 'error');
  } finally {
    if (generation === requestGeneration) {
      directoryInFlight = false;
      refreshButton.disabled = actionInFlight;
    }
  }
}

async function pollOneSession(robot, sessionId, generation) {
  try {
    const coreSnapshot = await api(`/api/teleop/sessions/${encodeURIComponent(sessionId)}`);
    if (generation === requestGeneration) rememberSession(coreSnapshot);
  } catch (error) {
    if (error.cancelled || generation !== requestGeneration) return;
    if (error.status === 401) {
      showLogin('登录已失效，请重新输入 token。');
      return;
    }
    markStatusFailure(sessionId, error.message);
  }
}

async function pollSessions(allowDuringAction = false) {
  if (!canControl() || statusInFlight || (actionInFlight && !allowDuringAction)) {
    renderDevices();
    return;
  }
  const generation = requestGeneration;
  statusInFlight = true;
  try {
    const targets = robots.flatMap((robot) => {
      const sessionId = visibleSessionId(robot);
      const session = sessionView(robot);
      if (!sessionId || (!robot.session.busy && session && TERMINAL_STATES.has(session.state))) {
        return [];
      }
      return [{ robot, sessionId }];
    });
    await Promise.all(targets.map(({ robot, sessionId }) => (
      pollOneSession(robot, sessionId, generation)
    )));
    if (generation === requestGeneration) renderDevices();
  } finally {
    if (generation === requestGeneration) statusInFlight = false;
  }
}

async function pollEvents(allowDuringAction = false) {
  if (!canControl() || eventsInFlight || (actionInFlight && !allowDuringAction)) return;
  const generation = requestGeneration;
  eventsInFlight = true;
  try {
    const sessionIds = [...new Set(robots.map(visibleSessionId).filter(Boolean))];
    await Promise.all(sessionIds.map(async (sessionId) => {
      try {
        const events = await api(
          `/api/teleop/sessions/${encodeURIComponent(sessionId)}/events?limit=12`,
        );
        if (generation === requestGeneration) {
          sessionEvents.set(sessionId, Array.isArray(events) ? events : []);
        }
      } catch (error) {
        if (error.cancelled || generation !== requestGeneration) return;
        if (error.status === 401) showLogin('登录已失效，请重新输入 token。');
      }
    }));
    if (generation === requestGeneration) renderDevices();
  } finally {
    if (generation === requestGeneration) eventsInFlight = false;
  }
}

async function renewOwnedSessions(allowDuringAction = false) {
  if (
    !canControl()
    || heartbeatInFlight
    || (actionInFlight && !allowDuringAction)
  ) return;
  const generation = requestGeneration;
  heartbeatInFlight = true;
  const owned = robots.filter((robot) => {
    const session = sessionView(robot);
    return Boolean(
      robot.session
      && robot.session.id
      && robot.session.owned_by_client
      && session
      && ['active', 'paused', 'hold'].includes(session.state),
    );
  });

  try {
    await Promise.all(owned.map(async (robot) => {
      const sessionId = robot.session.id;
      try {
        const snapshot = await api(
          `/api/teleop/sessions/${encodeURIComponent(sessionId)}/heartbeat`,
          { method: 'POST' },
        );
        if (generation !== requestGeneration) return;
        rememberSession(snapshot);
        browserHeartbeatFailures.delete(sessionId);
      } catch (error) {
        if (error.cancelled || generation !== requestGeneration) return;
        browserHeartbeatFailures.set(sessionId, error.message);
        if (error.status === 401) showLogin('登录已失效，请重新输入 token。');
      }
    }));
    if (generation === requestGeneration) renderDevices();
  } finally {
    if (generation === requestGeneration) heartbeatInFlight = false;
  }
}

function currentRobot(robotId) {
  return robots.find((robot) => robot.id === robotId) || null;
}

async function performAction(robotId, action) {
  const robot = currentRobot(robotId);
  if (
    !robot
    || !canControl()
    || actionInFlight
    || captureActionInFlight
    || pendingRobots.has(robotId)
  ) return;
  clearPollingTimers();
  invalidateApiRequests();
  const generation = requestGeneration;
  actionInFlight = true;
  pendingRobots.add(robotId);
  refreshButton.disabled = true;
  renderDevices();

  try {
    let result;
    if (action === 'reconcile-guard') {
      if (principal.role !== 'owner' || !robot.authority_guard) return;
      result = await api(
        `/api/teleop/authority-guards/${encodeURIComponent(robot.robot_id)}/reconcile`,
        { method: 'POST' },
      );
      if (generation !== requestGeneration) return;
      showNotice(
        result.state === 'clear'
          ? 'Driver 已提供安全停机证明；重启恢复锁已解除，旧会话没有恢复。'
          : '恢复锁仍保持。',
        result.state === 'clear' ? 'success' : 'error',
      );
    } else if (action === 'acquire') {
      if (!robot.teleop_ready || (robot.session && robot.session.busy)) return;
      const mode = descriptorMode(robot);
      result = await api('/api/teleop/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ driver_id: robot.driver_id, mode }),
      });
      if (generation !== requestGeneration) return;
      rememberSession(result.session);
      showNotice(
        result.disposition === 'confirmation_required'
          ? 'LIVE reservation 已创建；Driver 尚未调用。请核对机器人、Driver、Profile 与 Effectors 后明确确认硬件输出。'
          : `${mode.toUpperCase()} 会话已获取；Core 15 秒租约开始倒计时。`,
        result.disposition === 'confirmation_required' ? 'error' : 'success',
      );
    } else if (action === 'confirm-live') {
      const sessionId = robot.session && robot.session.id;
      const session = sessionView(robot);
      if (
        !sessionId
        || !robot.session?.owned_by_client
        || session?.id !== sessionId
        || session.mode !== 'live'
        || session.state !== 'awaiting_confirmation'
        || session.live_confirmed === true
        || typeof session.profile_id !== 'string'
      ) return;
      result = await api(
        `/api/teleop/sessions/${encodeURIComponent(sessionId)}/confirm-live`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            confirm_live_actuation: true,
            profile_id: session.profile_id,
          }),
        },
      );
      if (generation !== requestGeneration) return;
      rememberSession(result.session);
      showNotice(
        result.disposition === 'existing'
          ? 'LIVE 确认已存在；没有重复 prepare。RTC 与 VR 仍需手动连接。'
          : 'LIVE 硬件输出已明确确认；Driver 已准备。RTC 与 VR 仍需手动连接。',
        'success',
      );
    } else {
      const sessionId = robot.session && robot.session.id;
      const ownedByClient = Boolean(robot.session && robot.session.owned_by_client);
      const ownerOverride = principal.role === 'owner' && Boolean(sessionId);
      if (!sessionId || (!ownedByClient && !ownerOverride)) return;
      closeRtcConnection(
        sessionId,
        `操作员请求 ${action}；REST 前已本地解除 deadman 并关闭 XR/RTC`,
      );
      const endpoint = action === 'release' ? '' : `/${action}`;
      result = await api(`/api/teleop/sessions/${encodeURIComponent(sessionId)}${endpoint}`, {
        method: action === 'release' ? 'DELETE' : 'POST',
      });
      if (generation !== requestGeneration) return;
      if (action === 'release') {
        rememberSession(result.session);
        browserHeartbeatFailures.delete(sessionId);
        showNotice(
          result.driver_acknowledged
            ? '会话已释放，Driver 已确认。'
            : 'Core 已释放会话；Driver 确认失败，请检查事件。',
          result.driver_acknowledged ? 'success' : 'error',
        );
      } else {
        rememberSession(result);
        showNotice(action === 'pause' ? '会话已 Pause，仍占用机器人。' : '会话已进入 HOLD。', 'success');
      }
    }
    await loadDevices(false, true);
    await pollSessions(true);
    await pollEvents(true);
    await loadCaptures(false, true);
  } catch (error) {
    if (error.cancelled || generation !== requestGeneration) return;
    if (error.status === 401) {
      showLogin('登录已失效，请重新输入 token。');
      return;
    }
    showNotice(`操作失败：${error.message}`, 'error');
    await loadDevices(false, true);
  } finally {
    if (generation === requestGeneration) await renewOwnedSessions(true);
    if (generation === requestGeneration) {
      actionInFlight = false;
      pendingRobots.delete(robotId);
      refreshButton.disabled = false;
      renderDevices();
      startPolling();
    }
  }
}

function ownedClientReleaseTargets() {
  return robots.flatMap((robot) => {
    const session = sessionView(robot);
    if (
      !robot.session
      || !robot.session.id
      || !robot.session.owned_by_client
      || !session
      || !RELEASABLE_SESSION_STATES.has(session.state)
    ) return [];
    return [robot.session.id];
  });
}

function releaseOwnedClientSessions() {
  return ownedClientReleaseTargets().map((sessionId) => {
    try {
      return fetch(`/api/teleop/sessions/${encodeURIComponent(sessionId)}`, {
        method: 'DELETE',
        keepalive: true,
        headers: { 'X-Motus-Teleop-Client': TELEOP_CLIENT_ID },
      }).catch(() => null);
    } catch {
      return Promise.resolve(null);
    }
  });
}

async function waitForBestEffortRelease(requests) {
  if (requests.length === 0) return;
  await new Promise((resolve) => {
    const timeout = window.setTimeout(resolve, LOGOUT_RELEASE_WAIT_MS);
    Promise.allSettled(requests).then(() => {
      window.clearTimeout(timeout);
      resolve();
    });
  });
}

async function refreshSlowState() {
  await Promise.all([
    loadDevices(false),
    loadCaptures(false),
  ]);
  await pollEvents();
}

function startPolling() {
  clearPollingTimers();
  statusTimer = window.setInterval(pollSessions, STATUS_POLL_MS);
  slowTimer = window.setInterval(refreshSlowState, SLOW_POLL_MS);
  heartbeatTimer = window.setInterval(renewOwnedSessions, CORE_HEARTBEAT_MS);
}

async function enterApp() {
  const generation = requestGeneration;
  try {
    const me = await api('/api/teleop/me');
    if (generation !== requestGeneration) return;
    principal = me.principal;
    document.getElementById('principal-id').textContent = me.principal.id;
    document.getElementById('principal-role').textContent = me.principal.role;
    document.getElementById('principal-permission').textContent = me.permissions.control
      ? '可管理本标签页显式 Acquire 的 Shadow 或 Live 会话'
      : 'viewer：只读，不显示会话控制按钮';
    document.getElementById('teleop-client-id').textContent = TELEOP_CLIENT_ID;
    loginPanel.classList.add('hidden');
    content.classList.remove('hidden');
    logoutButton.classList.remove('hidden');
    await Promise.all([
      loadDevices(),
      loadCaptures(false),
    ]);
    if (generation !== requestGeneration || !principal) return;
    await pollSessions();
    if (generation !== requestGeneration || !principal) return;
    await pollEvents();
    if (generation !== requestGeneration || !principal) return;
    startPolling();
  } catch (error) {
    if (error.cancelled) return;
    if (error.status === 401) {
      showLogin('请输入已配置的身份 token。');
      showNotice('遥操控制台需要认证；未配置人类身份时保持关闭。', 'error');
      return;
    }
    showNotice(`身份读取失败：${error.message}`, 'error');
  }
}

loginForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  loginError.textContent = '正在验证…';
  const token = tokenInput.value.trim();
  const status = await getAuthStatus(token);
  if (!status.valid || !status.principal) {
    loginError.textContent = status.authRequired
      ? 'Token 无效。'
      : '服务端尚未配置 owner/operator/viewer token。';
    return;
  }
  setToken(token);
  tokenInput.value = '';
  loginError.textContent = '';
  await enterApp();
});

logoutButton.addEventListener('click', async () => {
  if (logoutButton.disabled) return;
  logoutButton.disabled = true;
  stopPolling();
  closeAllRtcConnections('退出登录；REST release 前已本地 fail-safe 关闭');
  const releaseRequests = releaseOwnedClientSessions();
  await waitForBestEffortRelease(releaseRequests);
  clearToken();
  principal = null;
  resetConsoleData();
  logoutButton.disabled = false;
  showLogin('已退出登录；本标签页会话已请求释放。');
  showNotice('当前未登录。');
});

refreshButton.addEventListener('click', () => loadDevices(true));
captureRefreshButton.addEventListener('click', () => loadCaptures(true));
capturePairButton.addEventListener('click', createCapturePairing);
captureAttachButton.addEventListener('click', attachSelectedCapture);
captureRevokeButton.addEventListener('click', revokeSelectedCapture);
captureDeviceSelect.addEventListener('change', () => {
  selectedCaptureId = captureDeviceSelect.value;
  renderCapturePanel();
  renderDevices();
});
captureSessionSelect.addEventListener('change', () => {
  selectedCaptureSessionId = captureSessionSelect.value;
  renderCapturePanel();
});
document.querySelectorAll('.capture-copy-button').forEach((button) => {
  button.addEventListener('click', () => {
    void copyCaptureBootstrapField(button.dataset.copyField);
  });
});

document.addEventListener('visibilitychange', () => {
  if (document.visibilityState !== 'visible') {
    closeAllRtcConnections('document 已隐藏；不会自动恢复 XR 或 RTC');
  }
});

window.addEventListener('pagehide', () => {
  stopPolling();
  closeAllRtcConnections('pagehide；REST release 前已本地 fail-safe 关闭');
  releaseOwnedClientSessions();
});

window.addEventListener('pageshow', (event) => {
  if (!event.persisted || !principal) return;
  startPolling();
  refreshSlowState().then(pollSessions);
});

probeWebxrSupport();
const initialStatus = await getAuthStatus(getToken());
if (initialStatus.valid && initialStatus.principal) {
  await enterApp();
} else {
  showLogin(initialStatus.authRequired ? '' : '服务端尚未配置人类身份 token。');
  showNotice('登录后可查看可信 Driver；控制台不会自动 Acquire、确认 Live、连接 RTC 或进入 VR。');
}
