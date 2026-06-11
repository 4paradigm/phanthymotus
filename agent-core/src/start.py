import contextlib
import asyncio
import pathlib
import shutil

import config
import event
import collector
import scheduler
import topic_subscriber


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

    try:
        import rclpy as _rclpy
        _rclpy.init(args=[])
        _rclpy.shutdown()
    except Exception as e:
        raise RuntimeError(
            f'[PhanthyMotus] DDS 初始化失败：{e}\n'
            'PhanthyMotus 需要安装在具有正常运行的 ROS2 DDS 服务的系统上。\n'
            '请检查 ROS2 环境是否正确 source，以及 DDS 中间件是否正常。'
        ) from e


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
                    {'topic': '', 'format': 'data/json'}
                ],
                'topic_out': [
                    {'topic': '/decision_core', 'format': 'data/json'}
                ],
            }
        ],
        'topic_out': [{'topic': '/decision_core', 'format': 'data/json'}],
        'topic_in': [{'topic': '', 'format': 'data/json'}],
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

app = fastapi.FastAPI(lifespan=lifespan)
app.mount('/api', app_api)

import api.motus_stream
app.include_router(api.motus_stream.router)

app.include_router(api.inspection.ws_router)

class _HTTPOnlyStaticFiles(fastapi.staticfiles.StaticFiles):
    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return

        async def send_no_cache(message):
            if message['type'] == 'http.response.start':
                path = scope.get('path', '')
                if path.endswith('.js') or path.endswith('.css'):
                    headers = dict(message.get('headers', []))
                    headers[b'cache-control'] = b'no-cache, no-store, must-revalidate'
                    message = {**message, 'headers': list(headers.items())}
            await send(message)

        await super().__call__(scope, receive, send_no_cache)

app.mount('/', _HTTPOnlyStaticFiles(directory='./web', html=True), name='web')


# ========== 启动服务 ==========
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=15678)
