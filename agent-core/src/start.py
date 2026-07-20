import contextlib
import asyncio
import pathlib
import shutil
import subprocess

import config
import auth
import event
import collector
import scheduler
import topic_subscriber
from channel.manager import manager as channel_manager


def _init_resource_files():
    """如果目标 memory 文件不存在，从 defaults 拷贝（冷启动）。"""
    # 镜像内固定路径（不会被 volume mount 遮盖）
    defaults_dir = pathlib.Path('/opt/defaults/memory')
    if not defaults_dir.exists():
        # 本地开发 fallback
        defaults_dir = pathlib.Path('./resource/memory/defaults')

    memory_dir = pathlib.Path('./resource/memory')
    memory_dir.mkdir(parents=True, exist_ok=True)

    if defaults_dir.exists():
        for f in defaults_dir.iterdir():
            if f.is_file():
                target = memory_dir / f.name
                if not target.exists():
                    shutil.copy(f, target)
                    print(f'[startup] copied default: {f.name}')

    # prompt_memory.md 特殊处理：空则从 init 拷贝
    mem = memory_dir / 'prompt_memory.md'
    init = memory_dir / 'prompt_memory_init.md'
    if init.exists() and (not mem.exists() or not mem.read_text().strip()):
        mem.write_text(init.read_text())
        print('[startup] initialized prompt_memory.md from init template')


def _check_dds():
    """Verify that a ROS2/DDS runtime is available on this host.
    Raises RuntimeError with a human-readable message if not."""
    try:
        import rclpy  # noqa: F401
    except ImportError:
        raise RuntimeError(
            '[PhanthyMotus] DDS 服务不可用：未检测到 rclpy。\n'
            'PhanthyMotus 需要安装在具有 ROS2 DDS 服务的系统上。\n'
            '请先安装 ROS2（例如 ros-humble-desktop 或 ros-jazzy-desktop），'
            '并 source /opt/ros/<版本>/setup.bash 后再启动。'
        )

    # Note: do NOT call rclpy.init()/shutdown() here — it corrupts the
    # global rcl context, causing ros2_bridge.start() to fail later.
    # The import check above is sufficient to verify DDS availability.


def _cleanup_stale_mcps():
    pass  # No-op: services self-register via heartbeat


async def _auto_ping_all_mcps():
    """On startup, ping all registered MCPs to populate tools/topics."""
    await asyncio.sleep(5)  # wait for driver containers to be ready
    import api.mcp_manage as mcp_mgr
    for mcp in mcp_mgr._get_mcp_list():
        mcp_id = mcp.get('id', '')
        if not mcp_id:
            continue
        try:
            await mcp_mgr._do_ping(mcp_id)
            print(f'[startup] auto-ping ok: {mcp_id}')
        except Exception as e:
            print(f'[startup] auto-ping failed: {mcp_id}: {e}')


def _register_core_mcp(silent=False):
    """Register agent-core itself as an MCP with decision_core tool."""
    import api.mcp_manage as mcp_mgr

    CORE_MCP_ID = 'agentcore'
    existing = mcp_mgr._get_mcp_list()
    # Remove stale entry if exists, then re-add fresh
    existing = [m for m in existing if m.get('id') != CORE_MCP_ID]

    existing.append({
        'id': CORE_MCP_ID,
        'name': 'AgentCore',
        'transport': 'internal',
        'url': '',
        'render_hint': '',
        'server_name': 'AgentCore',
        'category': 'controller',
        'online': True,
        'tools': [
            {
                'name': 'decision_core',
                'type': 'controller',
                'description': '决策核心 — 订阅多路输入，思考后发布决策到 /decision_core，通过 tool call 执行动作',
                'inputSchema': {'type': 'object', 'properties': {
                    'action': {'type': 'string', 'enum': ['info', 'config'], 'description': 'Action to perform'},
                }},
                'configSchema': {
                    'type': 'object',
                    'properties': {
                        'llm_url':   {'type': 'string', 'description': 'LLM API URL'},
                        'llm_key':   {'type': 'string', 'description': 'LLM API Key', 'format': 'password'},
                        'llm_model': {'type': 'string', 'description': 'LLM 模型名称'},
                        'trigger_interval_ms': {'type': 'integer', 'description': '采集触发间隔（毫秒）', 'default': 1000},
                    },
                    'required': ['llm_url', 'llm_key']
                },
                'topic_in': [
                    {'format': 'data/json'}
                ],
                'topic_out': [
                    {'topic': '/decision_core', 'format': 'data/json'}
                ],
            },
            {
                'name': 'remote_mic',
                'type': 'sensor',
                'description': '浏览器麦克风 — 通过 WebSocket 采集本地麦克风 PCM-16k 音频流',
                'inputSchema': {'type': 'object', 'properties': {}},
                'configSchema': {
                    'type': 'object',
                    'properties': {
                        'device_id': {
                            'type': 'string',
                            'description': '浏览器音频输入设备',
                            'format': 'audio-input-device',
                            'scope': 'instance',
                        },
                    },
                },
                'topic_out': [{'topic': '/remote_control/mic', 'format': 'audio/pcm-16k'}],
            },
            {
                'name': 'remote_message',
                'type': 'sensor',
                'description': '远程文本消息 — 从浏览器发送文本消息到机器人',
                'inputSchema': {
                    'type': 'object',
                    'properties': {
                        'action': {'type': 'string', 'enum': ['send_message'], 'description': 'Action to perform'},
                        'text': {'type': 'string', 'description': '消息文本'},
                    },
                    'required': ['action', 'text'],
                },
                'topic_out': [{'topic': '/remote_control/message', 'format': 'data/json'}],
            },
            {
                'name': 'channel_request',
                'type': 'sensor',
                'description': 'Channel message input — receive messages from Telegram/Slack and other platforms',
                'inputSchema': {'type': 'object', 'properties': {}},
                'configSchema': {
                    'type': 'object',
                    'properties': {
                        'channel_id': {
                            'type': 'string',
                            'description': 'Select a channel (configure in Channel settings first)',
                            'format': 'channel-select',
                            'scope': 'instance',
                        },
                    },
                },
                'multiInstance': True,
                'topic_out': [{'format': 'data/json'}],
            },
            {
                'name': 'channel_reply',
                'type': 'actuator',
                'description': 'Channel message output — send Agent replies to Telegram/Slack and other platforms',
                'inputSchema': {'type': 'object', 'properties': {}},
                'configSchema': {
                    'type': 'object',
                    'properties': {
                        'channel_id': {
                            'type': 'string',
                            'description': 'Select a channel (configure in Channel settings first)',
                            'format': 'channel-select',
                            'scope': 'instance',
                        },
                    },
                },
                'multiInstance': True,
                'topic_in': [{'format': 'data/json'}],
            }
        ],
        'topic_out': [{'topic': '/decision_core', 'format': 'data/json'}, {'topic': '/remote_control/mic', 'format': 'audio/pcm-16k'}, {'topic': '/remote_control/message', 'format': 'data/json'}],
        'topic_in': [{'format': 'data/json'}],
    })
    mcp_mgr._save_mcp_list(existing)
    if not silent:
        print(f'[startup] registered core MCP: {CORE_MCP_ID}')


async def _heartbeat_core_mcp():
    """Periodically re-register agent-core MCP every 30s."""
    import api.mcp_manage as mcp_mgr
    while True:
        await asyncio.sleep(30)
        try:
            _register_core_mcp(silent=True)
        except Exception as e:
            print(f'[heartbeat] core re-register failed: {e}')


@contextlib.asynccontextmanager
async def lifespan(app):
    # 初始化 access token 认证
    auth.init()

    # 初始化资源文件（从 defaults 拷贝缺失文件）
    _init_resource_files()

    # 检查宿主是否有 ROS2 DDS 服务
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _check_dds)

    # 启动 ROS2 bridge（用于 DDS topic 订阅）
    import ros2_bridge
    _ros2_loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, ros2_bridge.start, _ros2_loop)

    # 注册 AgentCore 自身为 MCP（含 decision_core 工具）
    await loop.run_in_executor(None, _register_core_mcp)

    # 注册 /decision_core output topic 到 inspection
    from api.inspection import register_topic_internal
    await register_topic_internal('/decision_core', 'data/json', 'agentcore')

    # 启动时 ping 所有已注册 MCP，填充 tools/topics
    asyncio.create_task(_auto_ping_all_mcps())

    # 定期刷新 agent-core 自身注册（30s）
    asyncio.create_task(_heartbeat_core_mcp())

    # 启动 DDS topic 订阅（依据 config event.subscribe_topics）
    topics = config.main.get('event', {}).get('subscribe_topics', [])
    topic_subscriber.start(topics, asyncio.get_event_loop())

    # 启动 collector（信息整理器）
    collector.start()

    # 启动 Channel Manager（消息平台适配器）
    await channel_manager.start()

    async with event.llm:
        tasks = [
            asyncio.create_task(event.llm.run_forever()),
            asyncio.create_task(scheduler.run()),
        ]
        try:
            yield
        finally:
            for t in tasks:
                t.cancel()
            await channel_manager.stop()
            await loop.run_in_executor(None, ros2_bridge.stop)


# ========== 网络服务 ==========
import fastapi
import fastapi.staticfiles
import uvicorn

app_api = fastapi.FastAPI()

import api.world
app_api.include_router(api.world.router)

import api.file
app_api.include_router(api.file.router)

import api.logging
app_api.include_router(api.logging.router)

import api.config
app_api.include_router(api.config.router)

import api.mcp_manage
app_api.include_router(api.mcp_manage.router)

import api.drivers
app_api.include_router(api.drivers.router)

import api.registry
app_api.include_router(api.registry.router)

import api.event
app_api.include_router(api.event.router)

import api.system
app_api.include_router(api.system.router)

import api.inspection
app_api.include_router(api.inspection.router)

import api.canvas
app_api.include_router(api.canvas.router)

import api.agent_definition
app_api.include_router(api.agent_definition.router)

import api.skills
app_api.include_router(api.skills.router)

import api.history
app_api.include_router(api.history.router)

import api.network
app_api.include_router(api.network.router)

import api.channel
app_api.include_router(api.channel.router)

app = fastapi.FastAPI(lifespan=lifespan)
app.middleware('http')(auth.auth_middleware)
app.mount('/api', app_api)

# Auth verify endpoint (exempt from middleware, does its own token check)
@app_api.get('/auth/verify')
async def _auth_verify(request: fastapi.Request):
    if not auth.is_enabled():
        return {'valid': True, 'auth_required': False}
    token = auth._extract_token(request)
    if auth.verify(token):
        return {'valid': True, 'auth_required': True}
    return fastapi.responses.JSONResponse(
        status_code=401,
        content={'valid': False, 'auth_required': True}
    )

import api.motus_stream
app.include_router(api.motus_stream.router)

app.include_router(api.inspection.ws_router)

# ── Mic WebSocket endpoint (receive browser PCM and publish to ROS2) ──────────
_mic_pub = None

@app.websocket('/ws/mic')
async def _ws_mic(ws: fastapi.WebSocket):
    """Receive PCM-16k audio from browser and publish to ROS2 topic."""
    global _mic_pub
    await ws.accept()
    try:
        if _mic_pub is None:
            try:
                from audio_msgs.msg import AudioChunk
                import ros2_bridge
                node = ros2_bridge._node_main
                if node:
                    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
                    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                     history=HistoryPolicy.KEEP_LAST, depth=200,
                                     durability=DurabilityPolicy.VOLATILE)
                    _mic_pub = node.create_publisher(AudioChunk, "/remote_control/mic", qos)
            except Exception:
                pass
        while True:
            data = await ws.receive_bytes()
            if _mic_pub:
                from audio_msgs.msg import AudioChunk
                chunk_size = 1024
                offset = 0
                while offset < len(data):
                    chunk = data[offset:offset + chunk_size]
                    offset += chunk_size
                    msg = AudioChunk()
                    msg.format = "pcm_16k_16bit_mono"
                    msg.data = list(chunk)
                    _mic_pub.publish(msg)
    except Exception:
        pass

class _HTTPOnlyStaticFiles(fastapi.staticfiles.StaticFiles):
    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return

        async def send_no_cache(message):
            if message['type'] == 'http.response.start':
                headers = dict(message.get('headers', []))
                headers[b'cache-control'] = b'no-cache, no-store, must-revalidate'
                message = {**message, 'headers': list(headers.items())}
            await send(message)

        await super().__call__(scope, receive, send_no_cache)

app.mount('/', _HTTPOnlyStaticFiles(directory='./web', html=True), name='web')


# ========== SSL 自签名证书 ==========
def _ensure_ssl_certs(cert_dir: str = "./resource/certs") -> tuple[str, str]:
    """自动生成自签名 SSL 证书（如不存在）。首次启动生成，后续复用。"""
    cert_path = pathlib.Path(cert_dir) / "cert.pem"
    key_path = pathlib.Path(cert_dir) / "key.pem"
    if cert_path.exists() and key_path.exists():
        return str(cert_path), str(key_path)
    pathlib.Path(cert_dir).mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:2048",
        "-keyout", str(key_path), "-out", str(cert_path),
        "-days", "3650", "-nodes",
        "-subj", "/CN=phanthy-motus",
    ], check=True, capture_output=True)
    print(f"[ssl] Generated self-signed certificate: {cert_path}")
    return str(cert_path), str(key_path)


# ========== 启动服务 ==========
if __name__ == '__main__':
    cert_file, key_file = _ensure_ssl_certs()
    uvicorn.run(app, host='0.0.0.0', port=15678, ws_ping_interval=None,
                ssl_certfile=cert_file, ssl_keyfile=key_file)
