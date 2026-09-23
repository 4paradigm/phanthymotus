"""Real HTTP discovery must retain the metadata used to mount pairing UI."""
import ast
import typing
import asyncio
import json
from pathlib import Path
import aiohttp
from aiohttp import web


def test_discovery_keeps_connection_panel():
    source=Path(__file__).parents[1]/'src/api/mcp_manage.py'
    function=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.AsyncFunctionDef) and n.name=='_ping_mcp_http')
    ns={'aiohttp':aiohttp,'asyncio':asyncio,'json':json}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),ns)
    async def run():
        tool={'name':'teleop','type':'processor','x-connection-panel':'teleop-v1','x-teleop-target':{'protocol_version':1,'robot_profile':'tianyi2'},'untrusted-extra':'discard','inputSchema':{'properties':{'action':{'enum':['info','open_pairing']}}}}
        tool['x-motion-control']={'protocol_version':2,'execution_tool':'arm'}
        tool['x-control-target']={'protocol_version':2,'resources':['arm_l','arm_r']}
        async def handle(request):
            method=(await request.json())['method']
            result={'tools':[tool]} if method=='tools/list' else {'content':[{'type':'text','text':'{}'}]} if method=='tools/call' else {}
            return web.json_response({'jsonrpc':'2.0','id':1,'result':result})
        app=web.Application();app.router.add_post('/mcp',handle)
        runner=web.AppRunner(app);await runner.setup()
        server=web.TCPSite(runner,'127.0.0.1',0);await server.start()
        try:
            port=server._server.sockets[0].getsockname()[1]
            result=await ns['_ping_mcp_http'](f'http://127.0.0.1:{port}/mcp')
            assert result['tools'][0]['x-connection-panel']=='teleop-v1'
            assert result['tools'][0]['x-teleop-target']=={'protocol_version':1,'robot_profile':'tianyi2'}
            assert result['tools'][0]['x-motion-control']==tool['x-motion-control']
            assert result['tools'][0]['x-control-target']==tool['x-control-target']
            assert 'untrusted-extra' not in result['tools'][0]
        finally:await runner.cleanup()
    asyncio.run(run())


def test_teleop_lifecycle_does_not_reapply_saved_configuration():
    """Exercise actual HTTP proxy function against a local MCP recorder."""
    from types import SimpleNamespace
    # Framework routing is outside this extracted proxy test.
    fastapi=SimpleNamespace(HTTPException=RuntimeError)
    source=Path(__file__).parents[1]/'src/api/mcp_manage.py'
    fn=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.AsyncFunctionDef) and n.name=='mcp_call_tool')
    fn.decorator_list=[]
    async def run():
        calls=[];reject=[False]
        async def handle(request):
            body=await request.json()
            if body['method']=='tools/call':calls.append(body['params']['arguments'])
            return web.json_response({'jsonrpc':'2.0','id':body['id'],'result':{'isError':reject[0], 'content':[{'type':'text','text':'{"state":"accepted"}'}]}})
        app=web.Application();app.router.add_post('/mcp',handle);runner=web.AppRunner(app);await runner.setup()
        server=web.TCPSite(runner,'127.0.0.1',0);await server.start()
        port=server._server.sockets[0].getsockname()[1]
        target={'id':'local','url':f'http://127.0.0.1:{port}/mcp','tools':[{'name':'teleop'}]}
        ns={'typing':typing,'aiohttp':aiohttp,'asyncio':asyncio,'json':json,'fastapi':fastapi,'MCPCallRequest':SimpleNamespace,
            '_get_mcp_list':lambda:[target], '_teleop_management_headers':lambda *a:{'X-Test':'management'},
            'config':SimpleNamespace(main={'tool_config:local:teleop':{'mode':'shadow'}}),
            'split_config_by_scope':lambda t,c:(c,{}),'missing_required_config':lambda *a:[],
            'plan_config_calls':lambda *a:[({'mode':'shadow'}, {})]}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),ns)
        try:
            for action in ['project_start','project_stop','start','pause','resume','finish','stop','open_pairing']:
                calls.clear();result=await ns['mcp_call_tool']('local',SimpleNamespace(tool='teleop',arguments={'action':action}))
                assert result['code']==200,result
                assert calls==[{'action':action}]
            calls.clear();await ns['mcp_call_tool']('local',SimpleNamespace(tool='teleop',arguments={'action':'config','mode':'live'}))
            assert calls==[{'action':'config','mode':'live'}]
            calls.clear();await ns['mcp_call_tool']('local',SimpleNamespace(tool='other',arguments={'action':'start'}))
            assert calls==[{'action':'config','mode':'shadow'},{'action':'start'}]
            reject[0]=True
            result=await ns['mcp_call_tool']('local',SimpleNamespace(tool='teleop',arguments={'action':'finish'}))
            assert result['code']==400
        finally:await runner.cleanup()
    asyncio.run(run())


def test_teleop_config_persists_only_after_service_ack(monkeypatch):
    from types import SimpleNamespace, ModuleType
    import sys
    source=Path(__file__).parents[1]/'src/api/canvas.py'
    fn=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.AsyncFunctionDef) and n.name=='save_tool_config')
    fn.decorator_list=[];fn.args.defaults=[ast.Constant(None)]
    sent=[];answer={'code':400,'message':'release_before_config'}
    async def call(mid,req):sent.append(req.arguments);return answer
    module=ModuleType('api.mcp_manage');module.mcp_call_tool=call;module.MCPCallRequest=SimpleNamespace
    monkeypatch.setitem(sys.modules,'api',ModuleType('api'));monkeypatch.setitem(sys.modules,'api.mcp_manage',module)
    class HTTPError(Exception):
        def __init__(self,**kwargs):self.detail=kwargs
    store={'row':{'position_scale':.5}}
    ns={'Any':typing.Any,'fastapi':SimpleNamespace(HTTPException=HTTPError,responses=SimpleNamespace(JSONResponse=lambda **kw:kw)),
        'config':SimpleNamespace(main=store),'tool_config_key':lambda *a:'row','apply_tool_config':lambda *a:sent.append('legacy')}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(source),'exec'),ns)
    async def run():
        result=await ns['save_tool_config']('robot','teleop',{'position_scale':.8})
        assert result['status_code']==400 and store['row']=={'position_scale':.5}
        answer.update(code=200,data=[])
        result=await ns['save_tool_config']('robot','teleop',{'mode':'live'})
        assert result['applied'] and store['row']=={'position_scale':.5,'mode':'live'}
        assert sent==[{'action':'config','position_scale':.8},{'action':'config','mode':'live'}]
        try:await ns['save_tool_config']('robot','teleop',{'action':'start'})
        except HTTPError:pass
        else:raise AssertionError('config cannot invoke start')
        await ns['save_tool_config']('robot','ordinary',{'x':1})
        assert sent[-1]=='legacy'
    asyncio.run(run())
