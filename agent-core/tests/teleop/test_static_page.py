from __future__ import annotations

import re
from pathlib import Path

WEB = Path(__file__).resolve().parents[2] / 'web'


def _function_source(javascript: str, name: str) -> str:
    match = re.search(
        rf'^(?:async )?function {re.escape(name)}\([^)]*\) \{{.*?^\}}',
        javascript,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match, f'missing JavaScript function: {name}'
    return match.group(0)


def test_teleop_page_is_a_visible_adapter_neutral_session_console():
    html = (WEB / 'teleop.html').read_text(encoding='utf-8')
    javascript = (WEB / 'js' / 'teleop' / 'app.js').read_text(encoding='utf-8')
    webxr = (WEB / 'js' / 'teleop' / 'webxr-frame.js').read_text(encoding='utf-8')
    css = (WEB / 'css' / 'teleop.css').read_text(encoding='utf-8')

    assert "api('/api/teleop/me')" in javascript
    assert "api('/api/teleop/robots')" in javascript
    assert "api('/api/teleop/sessions'," in javascript
    assert '/heartbeat`' in javascript
    assert '/events?limit=12`' in javascript
    assert '/driver-status' not in javascript
    assert "method: 'POST'" in javascript
    assert "method: 'DELETE'" in javascript
    assert 'keepalive: true' in javascript
    assert 'WebSocket' not in javascript
    assert '/call' not in javascript
    assert '/api/mcp' not in javascript

    button_ids = set(re.findall(r'<button[^>]+id="([^"]+)"', html))
    assert button_ids == {
        'capture-attach-button',
        'capture-pair-button',
        'capture-refresh-button',
        'capture-revoke-button',
        'logout-button',
        'refresh-button',
    }
    assert '<form id="login-form">' in html
    assert '通用遥操作控制台' in html
    assert 'Shadow 只记录；Live 必须二次明确确认硬件输出' in html
    assert '第一次 Acquire 只创建短期内存 reservation' in html
    assert '不会调用 Driver、续租或开放 RTC' in html
    assert '必须再次点击醒目的 Live 确认按钮' in html
    assert '等待 Live 确认时不会写这把锁' in html
    assert 'Pause 仍占用机器人' in html
    assert '恢复必须先 Release，再重新 Acquire' in html
    assert 'Core state=hold / Driver reason=soft_stop' in javascript
    for reason in (
            'deadman_released',
            'command_timeout',
            'intent_expired',
        'pose_timeout',
        'rtc_closed',
        'rtc_disconnected',
        'rtc_failed',
        'rtc_not_ready',
        'tracking_lost',
    ):
        assert f"'{reason}'" in javascript
    assert 'RECOVERABLE_DRIVER_HOLD_REASONS.has(driverReason)' in javascript
    assert '先松开再双握以产生更大的 clutch_sequence' in javascript
    assert '每次页面加载都会生成独立 Client ID' in html
    assert '避免复制标签页继承控制权' in html
    assert '<code>local-floor</code>' in html
    assert '能力声明的运动轴全部归入各自 deadzone' in html
    assert '再空挡重握' in html
    assert '不能预装首帧运动' in html
    assert '未启用的 base 或 hands 不会出现在 frame 中' in html
    assert '最终限幅和硬件适配仍由 Driver 负责' in html
    assert '头显只显示诊断黑场' in html
    assert '没有机器人视频，也没有头显内 HUD' in html
    assert '2D 镜像页' in html
    assert '绝不会自动确认 Live、自动恢复、自动重进 VR、自动重连或自动获取会话' in html
    assert '重启后的普通执行器命令仍会被拒绝' in html
    assert '旧会话始终不会恢复' in html
    assert '验证安全并解除重启锁' in javascript
    assert "action === 'reconcile-guard'" in javascript
    assert (
        '/api/teleop/authority-guards/${encodeURIComponent(robot.robot_id)}'
        '/reconcile'
    ) in javascript
    assert '只有 owner 能验证安全停机并解除恢复锁' in javascript
    assert 'id="webxr-support"' in html
    assert 'id="teleop-client-id"' in html
    for reason in (
        'driver_transport_invalid',
        'driver_runtime_not_trusted',
        'driver_runtime_target_mismatch',
    ):
        assert reason in javascript
    assert "driver_epoch_exhausted: 'Driver 防重放 epoch 已耗尽；Core 已拒绝创建新会话。'" in javascript
    assert '.device-card.live-mode' in css
    assert '.live-confirmation' in css
    for vendor_term in ('unitree_go1_high_level', 'Go1', 'unitree'):
        assert vendor_term not in html
        assert vendor_term not in javascript
        assert vendor_term not in webxr
    assert 'dry-run' not in html.lower()
    assert 'dry-run' not in javascript.lower()


def test_teleop_console_locks_browser_lease_and_role_safety_contracts():
    javascript = (WEB / 'js' / 'teleop' / 'app.js').read_text(encoding='utf-8')

    assert 'const STATUS_POLL_MS = 1000;' in javascript
    assert 'const CORE_HEARTBEAT_MS = 5000;' in javascript
    assert 'window.setInterval(pollSessions, STATUS_POLL_MS)' in javascript
    assert 'window.setInterval(renewOwnedSessions, CORE_HEARTBEAT_MS)' in javascript
    assert 'robot.session.owned_by_client' in javascript
    assert "principal.role === 'operator' || principal.role === 'owner'" in javascript
    assert "if (!canControl()) return null;" in javascript
    assert "makeActionButton('Pause'" in javascript
    assert "makeActionButton('HOLD'" in javascript
    assert "makeActionButton('Release'" in javascript
    assert '`Acquire ${descriptorMode(robot).toUpperCase()}`' in javascript
    assert javascript.count("api('/api/teleop/sessions',") == 1
    assert "dataset.action = 'resume'" not in javascript.lower()
    assert '此控制台没有 Resume' in javascript
    assert '不会自动获取新会话' in javascript

    renewal = _function_source(javascript, 'renewOwnedSessions')
    release_targets = _function_source(javascript, 'ownedClientReleaseTargets')
    assert 'owned_by_client' in renewal
    assert 'owned_by_me' not in renewal
    assert 'owned_by_client' in release_targets
    assert 'owned_by_me' not in release_targets
    assert 'RELEASABLE_SESSION_STATES.has(session.state)' in release_targets
    assert "'awaiting_confirmation', 'preparing', 'active', 'paused', 'hold'" in javascript

    action = _function_source(javascript, 'performAction')
    assert 'body: JSON.stringify({ driver_id: robot.driver_id, mode })' in action
    assert "action === 'confirm-live'" in action
    assert '/confirm-live`' in action
    assert 'confirm_live_actuation: true' in action
    assert 'profile_id: session.profile_id' in action
    assert "session.state !== 'awaiting_confirmation'" in action
    assert 'robot.session?.owned_by_client' in action
    assert action.index("action === 'confirm-live'") < action.index('closeRtcConnection(')


def test_teleop_console_uses_per_tab_identity_and_bounded_single_flight_reads():
    javascript = (WEB / 'js' / 'teleop' / 'app.js').read_text(encoding='utf-8')

    assert 'sessionStorage.getItem' not in javascript
    assert 'sessionStorage.setItem(CLIENT_ID_KEY, generated)' in javascript
    assert 'crypto.randomUUID()' in javascript
    assert 'CANONICAL_UUID.test(generated)' in javascript
    assert 'return stored' not in javascript
    assert "headers.set('X-Motus-Teleop-Client', TELEOP_CLIENT_ID)" in javascript
    assert "headers: { 'X-Motus-Teleop-Client': TELEOP_CLIENT_ID }" in javascript

    assert 'const REQUEST_TIMEOUT_MS = 4000;' in javascript
    assert 'new AbortController()' in javascript
    assert 'controller.abort()' in javascript
    assert 'activeControllers.forEach((controller) => controller.abort())' in javascript
    assert 'requestGeneration += 1' in javascript
    assert 'generation !== requestGeneration' in javascript
    assert 'directoryInFlight' in javascript
    assert 'statusInFlight' in javascript
    assert 'eventsInFlight' in javascript
    assert 'heartbeatInFlight' in javascript

    assert 'if (directoryInFlight ||' in _function_source(javascript, 'loadDevices')
    assert 'statusInFlight ||' in _function_source(javascript, 'pollSessions')
    assert 'heartbeatInFlight' in _function_source(javascript, 'renewOwnedSessions')

    status_read = _function_source(javascript, 'pollOneSession')
    assert status_read.count('await api(') == 1
    assert '/heartbeat' not in status_read
    assert '/events' not in status_read
    assert '/driver-status' not in status_read

    action = _function_source(javascript, 'performAction')
    assert action.index('invalidateApiRequests()') < action.index('await api(')
    assert 'actionInFlight' in action

    timeout_ms = int(re.search(r'REQUEST_TIMEOUT_MS = (\d+);', javascript).group(1))
    heartbeat_ms = int(re.search(r'CORE_HEARTBEAT_MS = (\d+);', javascript).group(1))
    assert timeout_ms < heartbeat_ms
    assert (2 * heartbeat_ms) + timeout_ms < 15_000


def test_teleop_console_negotiates_visible_peer_bound_webrtc_without_secrets():
    javascript = (WEB / 'js' / 'teleop' / 'app.js').read_text(encoding='utf-8')
    connect = _function_source(javascript, 'connectRtc')

    assert '`Direct WebXR fallback：连接 RTC ${descriptorMode(robot)}`' in javascript
    assert '断开 Direct RTC（进入 HOLD）' in javascript
    assert '进入 Quest VR' in javascript
    assert "new RTCPeerConnection()" in connect
    assert "createDataChannel('teleop-control', { ordered: true })" in connect
    assert "createDataChannel('teleop-pose', {" in connect
    assert 'ordered: false' in connect
    assert 'maxRetransmits: 0' in connect
    assert connect.index("createDataChannel('teleop-control'") < connect.index('createOffer()')
    assert connect.index("createDataChannel('teleop-pose'") < connect.index('createOffer()')
    assert 'waitForIceGatheringComplete(peer)' in connect
    assert '/signaling/offer`' in connect
    assert "JSON.stringify({ type: 'offer', sdp: local.sdp })" in connect
    assert 'timeoutMs: RTC_SIGNAL_TIMEOUT_MS' in connect
    assert 'peer.setRemoteDescription(answer)' in connect
    ping = _function_source(javascript, 'sendPeerPing')
    pong = _function_source(javascript, 'recordPeerPong')
    assert "type: 'peer_ping'" in ping
    assert 'performance.now()' in ping
    assert 'performance.now() - startedAt' in pong
    assert 'Date.now()' not in ping
    assert 'Date.now()' not in pong
    assert 'startPeerPings(sessionId, peer, control' in connect
    assert 'lease_renewed' not in ping
    assert 'lease_renewed' not in pong
    assert 'poseChannel.send(payload)' in javascript
    assert "robot.session?.owned_by_client" in connect
    active_owner = _function_source(javascript, 'ownedActiveRobotSession')
    assert "session?.state === 'active'" in active_owner
    assert "session.mode !== 'live' || session.live_confirmed === true" in active_owner
    assert "sessionState !== 'active'" in javascript
    assert 'ticket' not in connect
    assert 'fence' not in connect
    assert 'driver_id' not in connect
    assert 'WebSocket' not in javascript
    assert 'ownedActiveRobotSession(robotId, sessionId)' in connect
    assert 'captureAssignmentForSession(captures, sessionId) !== null' in connect


def test_capture_panel_is_manual_pc_owned_and_keeps_direct_webxr_as_fallback():
    html = (WEB / 'teleop.html').read_text(encoding='utf-8')
    javascript = (WEB / 'js' / 'teleop' / 'app.js').read_text(encoding='utf-8')
    capture_module = (
        WEB / 'js' / 'teleop' / 'capture-console.js'
    ).read_text(encoding='utf-8')

    for element_id in (
        'capture-panel',
        'capture-pair-button',
        'capture-refresh-button',
        'capture-device-select',
        'capture-session-select',
        'capture-attach-button',
        'capture-revoke-button',
        'capture-wss-url',
        'capture-pairing-id',
        'capture-pairing-code',
        'capture-ca-base64',
    ):
        assert f'id="{element_id}"' in html

    assert 'PC 标签页始终持有会话与硬件确认权' in html
    assert 'OPENXR CAPTURE' in html
    assert '独立 OpenXR 头显姿态采集设备' in html
    assert 'OpenXR 头显 Capture 只接收当前会话的安全投影' in html
    assert 'value="OpenXR Headset Capture"' in html
    assert 'Pairing code 是短期一次性秘密' in html
    assert '交给 OpenXR 头显 Capture' in html
    assert '不要放入 URL、HTTP header、浏览器 console、日志或 shell 历史' in html
    assert 'Direct WebXR fallback（显式同页模式）仍保留' in html
    assert '再手动进入 Quest VR' in html
    assert '页面不会静默切换' in html
    for quest_only_capture_copy in (
        'QUEST CAPTURE',
        '独立 Quest 姿态采集设备',
        'Quest 只接收当前会话的安全投影',
        'value="Quest 3 Capture"',
        '交给 Quest；',
        '可安全提供给 Quest 的当前 TLS',
        'Quest 尚未连接并进入 xr_standby',
        'Quest Capture 或 RTC frame',
        'Quest 信令失败',
        '已配对的 Quest Capture',
        '自动连接 Quest',
        '等待 Quest 一次 SDP offer',
    ):
        assert quest_only_capture_copy not in html
        assert quest_only_capture_copy not in javascript
    for openxr_capture_copy in (
        '可安全提供给 OpenXR 头显 Capture 的当前 TLS',
        'OpenXR 头显 Capture 尚未连接并进入 xr_standby',
        'OpenXR 头显 Capture 或 RTC frame',
        'OpenXR 头显 Capture 信令失败',
        '已配对的 OpenXR 头显 Capture',
        '不会自动连接 OpenXR 头显 Capture',
        '等待 OpenXR 头显 Capture 的一次 SDP offer',
    ):
        assert openxr_capture_copy in javascript
    assert "api('/api/teleop/capture-pairings'" in javascript
    assert "api('/api/teleop/captures')" in javascript
    assert '/capture-attachment`' in javascript
    assert "method: 'DELETE'" in _function_source(javascript, 'revokeSelectedCapture')
    attach = _function_source(javascript, 'attachSelectedCapture')
    assert 'captureAttachEligibility(session, capture)' in attach
    assert 'capture_id: capture.id' in attach
    assert 'mode: session.mode' in attach
    assert 'profile_id: session.profile_id' in attach
    assert 'capability_digest: session.capability_digest' in attach
    assert 'rtcInFlight.has(session.id)' in attach
    assert 'capturePairing = captureBootstrapFields(pairing, window.location)' in javascript
    assert 'localStorage' not in _function_source(javascript, 'createCapturePairing')
    assert 'sessionStorage' not in _function_source(javascript, 'createCapturePairing')
    assert 'console.' not in _function_source(javascript, 'copyCaptureBootstrapField')
    assert 'wss://${host}/ws/teleop-capture' not in capture_module
    assert 'wss://${host}${path}' in capture_module

    # Read-only listing may refresh; authority-changing operations remain bound
    # only to their explicit click handlers.
    for automatic_source in (
        _function_source(javascript, 'enterApp'),
        _function_source(javascript, 'refreshSlowState'),
    ):
        assert 'createCapturePairing' not in automatic_source
        assert 'attachSelectedCapture' not in automatic_source
        assert 'confirm-live' not in automatic_source
        assert 'connectRtc' not in automatic_source
        assert 'requestImmersiveVr' not in automatic_source


def test_quest_webxr_entry_sampling_and_fail_safe_contracts_are_visible():
    html = (WEB / 'teleop.html').read_text(encoding='utf-8')
    javascript = (WEB / 'js' / 'teleop' / 'app.js').read_text(encoding='utf-8')
    request_vr = _function_source(javascript, 'requestImmersiveVr')
    initialize = _function_source(javascript, 'initializeXrSession')
    on_frame = _function_source(javascript, 'onXrFrame')
    close_rtc = _function_source(javascript, 'closeRtcConnection')
    connect = _function_source(javascript, 'connectRtc')

    assert 'window.isSecureContext === true' in javascript
    assert 'window.location.origin' in javascript
    assert 'Boolean(navigator.xr)' in javascript
    assert "navigator.xr.isSessionSupported('immersive-vr')" in javascript
    assert 'Browser WebXR capability' in javascript
    assert 'WebXR 能力' in html

    # requestSession must remain in the direct click stack: no await can consume
    # transient user activation before the immersive request.
    assert request_vr.startswith('function requestImmersiveVr')
    assert 'await ' not in request_vr
    assert "navigator.xr.requestSession(" in request_vr
    assert "'immersive-vr'" in request_vr
    assert "{ requiredFeatures: ['local-floor'] }" in request_vr
    assert request_vr.index('navigator.xr.requestSession(') < request_vr.index(
        'xrSessions.set(sessionId, record)'
    )
    assert javascript.count('navigator.xr.requestSession(') == 1
    entry_ready = _function_source(javascript, 'xrEntryReady')
    assert 'ownedActiveRobotSession(robotId, sessionId)' in entry_ready
    assert 'rtcReadyForXr(sessionId)' in entry_ready

    assert 'gl.makeXRCompatible()' in initialize
    assert "xrSession.requestReferenceSpace('local-floor')" in initialize
    assert 'new XRWebGLLayer(xrSession, gl)' in initialize
    assert "referenceSpace.addEventListener('reset'" in initialize
    assert 'webxr_reference_space_reset' in initialize
    assert "xrSession.addEventListener('end'" in initialize
    assert "xrSession.addEventListener('visibilitychange'" in initialize

    assert 'frame.getViewerPose(record.referenceSpace)' in on_frame
    assert 'uniqueTrackedPointerGripSources(record.session)' in on_frame
    assert 'frame.getPose(sources.left.gripSpace, record.referenceSpace)' in on_frame
    assert 'frame.getPose(sources.right.gripSpace, record.referenceSpace)' in on_frame
    assert 'poseIsActuallyTracked(viewerPose)' in on_frame
    assert 'pose.emulatedPosition === true ? null : pose' in javascript
    assert 'isDualSqueezeDeadmanRequested(sources.left, sources.right)' in on_frame
    assert 'joystick' not in javascript.lower()

    assert 'const XR_NORMAL_SEND_HZ = 60;' in javascript
    assert 'const POSE_MESSAGE_LIMIT_BYTES = 64 * 1024;' in javascript
    assert 'const POSE_BUFFER_HIGH_WATER_BYTES = 16 * 1024;' in javascript
    assert 'new TextEncoder().encode(payload).byteLength' in on_frame
    assert 'poseChannel?.bufferedAmount' in on_frame
    assert 'poseChannel.bufferedAmount' in on_frame
    assert 'bufferedAmount + payloadBytes > POSE_BUFFER_HIGH_WATER_BYTES' in on_frame
    assert 'poseChannel.send(payload)' in on_frame
    assert 'bufferedamountlow' not in javascript.lower()
    assert '&& !safetyTransition' in on_frame
    assert on_frame.index('isTrackingOrDeadmanSafetyTransition') < on_frame.index(
        'frameTimeMs < record.nextPoseAtMs'
    )

    assert 'markSessionFrameReleased(sessionId)' in close_rtc
    assert 'stopXrSession(sessionId, reason)' in close_rtc
    assert 'rtcFrameStates.delete' not in close_rtc
    assert 'markRtcFrameDeadmanReleased' in javascript
    assert '需松开后重握' in javascript
    assert 'frameStateForSession(sessionId)' in on_frame
    assert 'rtcFrameStates.set(sessionId, built.state)' in on_frame
    assert 'record.frameConfiguration' in on_frame
    assert 'session.mode === record.mode' in _function_source(
        javascript, 'xrSessionContractMatches'
    )
    released = _function_source(javascript, 'markSessionFrameReleased')
    assert "stats.baseTwist = '本地已归零；RTC/XR 不再发送'" in released

    assert "['closed', 'disconnected', 'failed'].includes(peer.connectionState)" in connect
    assert "control.addEventListener('close'" in connect
    assert "pose.addEventListener('close'" in connect
    assert "control.addEventListener('error'" in connect
    assert "pose.addEventListener('error'" in connect
    assert "document.visibilityState !== 'visible'" in on_frame
    assert "record.session.visibilityState !== 'visible'" in on_frame
    assert 'closeRtcConnection(' in _function_source(javascript, 'handleXrSessionEnded')

    assert 'frameTimeMs >= record.nextCoreHeartbeatAtMs && !heartbeatInFlight' in on_frame
    assert 'void renewOwnedSessions(true)' in on_frame
    assert '/heartbeat' not in _function_source(javascript, 'connectRtc')

    for label in (
        'Driver pose.latest_sequence',
        'Driver dispatch.last_decision',
        'Driver hardware published sequence',
        'Driver would-apply sequence',
        'Safe-stop ACK',
        'Driver fault code',
        'Latency receive→admit',
        'Latency mailbox',
        'Latency IK',
        'Latency adapter publish',
        'Latency adapter projection',
        'Latency next LowState feedback arrival',
        'Output evidence profile',
        'Output evidence hardware path',
        'Output evidence state',
        'Configured hardware output',
        'Browser peer RTT（同钟）',
        'WebXR tracking',
        'Quest left input',
        'Quest right input',
        'WebXR deadman',
        'WebXR optional base_twist',
        'Pose frames',
        'Pose transport',
    ):
        assert label in javascript
    assert 'profiles ${profiles.length' in javascript
    assert 'buttons ${buttonsLength}' in javascript


def test_control_and_lifecycle_transitions_quiesce_locally_before_rest():
    javascript = (WEB / 'js' / 'teleop' / 'app.js').read_text(encoding='utf-8')
    action = _function_source(javascript, 'performAction')
    action_rest = 'result = await api(`/api/teleop/sessions/${encodeURIComponent(sessionId)}${endpoint}`'
    assert action.index('closeRtcConnection(') < action.index(action_rest)
    assert 'REST 前已本地解除 deadman 并关闭 XR/RTC' in action

    remember = _function_source(javascript, 'rememberSession')
    assert "snapshot.state !== 'active'" in remember
    assert 'closeRtcConnection(' in remember
    enforce = _function_source(javascript, 'enforceInteractiveSessionSafety')
    assert 'owned_by_client' in enforce
    assert "session?.state !== 'active'" in enforce

    visibility = re.search(
        r"document\.addEventListener\('visibilitychange'.*?^\}\);",
        javascript,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert visibility
    assert "document.visibilityState !== 'visible'" in visibility.group(0)
    assert 'closeAllRtcConnections(' in visibility.group(0)

    # Capability probing and BFCache restoration never enter VR, reconnect RTC,
    # or Acquire a new session.
    probe = _function_source(javascript, 'probeWebxrSupport')
    assert 'requestSession' not in probe
    assert 'confirm-live' not in probe
    assert "api('/api/teleop/sessions'," not in probe
    pageshow = re.search(
        r"window\.addEventListener\('pageshow'.*?^\}\);",
        javascript,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert pageshow
    assert 'requestImmersiveVr' not in pageshow.group(0)
    assert 'connectRtc' not in pageshow.group(0)
    assert 'confirm-live' not in pageshow.group(0)
    assert "api('/api/teleop/sessions'," not in pageshow.group(0)
    assert javascript.count('/confirm-live`') == 1

    load = _function_source(javascript, 'loadDevices')
    assert 'confirm-live' not in load
    assert "api('/api/teleop/sessions'," not in load


def test_logout_and_pagehide_release_only_this_tabs_sessions_before_reset():
    javascript = (WEB / 'js' / 'teleop' / 'app.js').read_text(encoding='utf-8')

    logout = re.search(
        r"logoutButton\.addEventListener\('click', async \(\) => \{.*?^\}\);",
        javascript,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert logout
    logout_source = logout.group(0)
    assert logout_source.index('closeAllRtcConnections(') < logout_source.index(
        'releaseOwnedClientSessions()'
    )
    assert logout_source.index('releaseOwnedClientSessions()') < logout_source.index('clearToken()')
    assert logout_source.index('clearToken()') < logout_source.index('resetConsoleData()')
    assert 'waitForBestEffortRelease(releaseRequests)' in logout_source

    pagehide = re.search(
        r"window\.addEventListener\('pagehide'.*?^\}\);",
        javascript,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert pagehide
    pagehide_source = pagehide.group(0)
    assert 'releaseOwnedClientSessions()' in pagehide_source
    assert pagehide_source.index('closeAllRtcConnections(') < pagehide_source.index(
        'releaseOwnedClientSessions()'
    )
    assert 'keepalive: true' in javascript


def test_teleop_console_retains_last_terminal_snapshot_and_events_in_this_page():
    javascript = (WEB / 'js' / 'teleop' / 'app.js').read_text(encoding='utf-8')

    assert 'const lastSessionByRobot = new Map();' in javascript
    assert 'lastSessionByRobot.set(snapshot.robot_id, snapshot.id)' in javascript
    assert 'lastSessionByRobot.get(robot.robot_id)' in javascript
    assert 'JSON.stringify({ driver_id: robot.driver_id, mode })' in javascript
    assert '`Driver: ${robot.driver_id} · Robot: ${robot.robot_id}`' in javascript
    assert "fact('本页最后终态'" in javascript
    assert 'const sessionEvents = new Map();' in javascript
    assert 'sessionEvents.set(sessionId' in javascript


def test_teleop_console_renders_remote_values_without_html_injection_sinks():
    javascript = (WEB / 'js' / 'teleop' / 'app.js').read_text(encoding='utf-8')

    assert '.textContent =' in javascript
    assert '.innerHTML' not in javascript
    assert '.outerHTML' not in javascript
    assert 'insertAdjacentHTML' not in javascript
    assert 'document.write' not in javascript
    assert 'eval(' not in javascript


def test_main_dashboard_links_to_the_teleop_directory():
    index = (WEB / 'index.html').read_text(encoding='utf-8')
    assert 'href="/teleop.html"' in index


def test_authenticated_websocket_url_is_never_written_to_console():
    dashboard = (WEB / 'js' / 'dashboard.js').read_text(encoding='utf-8')
    console_calls = re.findall(r'console\.[a-z]+\([^\n;]*', dashboard)
    assert all('_wsUrl' not in call and 'wsUrl' not in call for call in console_calls)
    assert "console.log('[preview ws] open')" in dashboard
    assert "console.log('[preview ws] closed', e.code)" in dashboard


def test_dashboard_control_normalizes_tool_descriptors_and_has_no_hidden_replay():
    dashboard = (WEB / 'js' / 'dashboard.js').read_text(encoding='utf-8')

    assert "typeof tool === 'string' ? tool : tool?.name" in dashboard
    assert 'completedTools' in dashboard
    assert "_callNamedTool(skill, startedTool, 'stop')" in dashboard
    assert "if (_globalRunning && skill.online)" not in dashboard
    assert "localStorage.removeItem('motus_global_running')" in dashboard


def test_canvas_project_ui_models_degraded_and_cleans_failed_microphone_start():
    canvas = (WEB / 'js' / 'canvas.js').read_text(encoding='utf-8')

    assert "let _projectState = 'stopped';" in canvas
    assert "_projectState !== 'running'" in canvas
    assert "'停止残留控制'" in canvas
    assert "'状态未知—尝试停止'" in canvas
    assert '_cleanupMicAfterFailedStart' in canvas
    assert "btn.disabled = _projectTransitioning || (!_projectRunning && !_isEditor)" in canvas
    assert "if (!_isEditor) throw new Error('请先获取画布编辑权');" in canvas
    save_call = "await _saveLayout({ strict: true });"
    assert save_call in canvas
    assert canvas.index(save_call) < canvas.index("fetch('/api/config/start-project'")
    assert canvas.index(save_call) < canvas.index("authenticatedWsUrl('/ws/mic')")
    assert 'layout_revision: _layoutRevision' in canvas
    assert 'session_id: _sessionId' in canvas
    assert "sessionStorage.getItem(_CANVAS_SESSION_KEY)" not in canvas
    assert "sessionStorage.setItem(_CANVAS_SESSION_KEY, _sessionId)" in canvas
    assert "globalThis.crypto?.randomUUID?.()" in canvas
    assert "|| ('sess-' + Date.now().toString(36)" in canvas
    assert "localStorage.getItem('canvas_session_id')" not in canvas
    assert "{ type: 'application/json' }" in canvas
    assert 'if (!await _reloadLayout())' in canvas
    assert '_reconcileAmbiguousProjectStart' in canvas
    assert "_applyProjectState(true, 'unknown')" in canvas
    assert "if (detail?.project_state === 'degraded')" in canvas
    assert "typeof snapshot?.transitioning !== 'boolean'" in canvas
    assert "if (!resp.ok || result?.code !== 200)" in canvas
    assert 'result?.data?.revision' in canvas
    assert 'if (strict) throw error;' in canvas


def test_sidebar_keeps_saved_config_visible_when_driver_apply_fails():
    sidebar = (WEB / 'js' / 'sidebar.js').read_text(encoding='utf-8')

    assert 'if (saveResult?.data?.saved)' in sidebar
    assert "statusEl.textContent = '⚠'" in sidebar
    assert '配置已保存到 Core，但 Driver 应用失败' in sidebar
