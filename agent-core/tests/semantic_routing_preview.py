"""Isolated browser fixture: real Canvas UI/config APIs; no Core lifespan or ROS.

Run from any cwd: python agent-core/tests/semantic_routing_preview.py
Only listens on 127.0.0.1:18764. Uses disposable DB/identity and a fake API key.
Never starts the agent loop, a device, or a real Jev request.
"""
import ast
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TEMP = tempfile.TemporaryDirectory(prefix='semantic-routing-preview-')
os.chdir(TEMP.name)
os.environ['DB_PATH'] = str(Path(TEMP.name) / 'config.db')
os.environ['TYPESAFE_API_KEY'] = 'fixture-not-a-real-key'
sys.path.insert(0, str(ROOT / 'src'))

import fastapi
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import config
import semantic_routing
from api import canvas, mcp_manage

identity = Path(TEMP.name) / 'resource' / 'memory' / 'identity.md'
identity.parent.mkdir(parents=True)
identity.write_text('你是机器人小范，负责展厅讲解。', encoding='utf-8')

# Execute the production registration function, not a copied schema. Do not
# import start.py: that brings in the production startup surface and side effects.
source = ROOT / 'src' / 'start.py'
tree = ast.parse(source.read_text())
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_register_core_mcp')
namespace = {'semantic_routing': semantic_routing}
exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), 'exec'), namespace)
namespace['_register_core_mcp'](silent=True)

app = fastapi.FastAPI()
app.include_router(canvas.router, prefix='/api')
app.include_router(mcp_manage.router, prefix='/api')


@app.get('/api/config/project-running')
async def project_running():
    return {'running': False}


@app.get('/fixture-schema')
async def schema():
    mcps = mcp_manage._get_mcp_list()
    return next(m for m in mcps if m['id'] == 'agentcore')['tools'][0]['configSchema']


@app.get('/')
async def page():
    html = (ROOT / 'web' / 'index.html').read_text()
    entry = '''<script type="module">
      import {initSidebar, openToolConfigModal} from '/js/sidebar.js';
      import {initCanvas} from '/js/canvas.js';
      document.title = 'Jev isolated Canvas verification';
      initSidebar();
      await initCanvas([]);
      const schema = await (await fetch('/fixture-schema')).json();
      await openToolConfigModal('agentcore', 'decision_core', schema);
    </script>'''
    return HTMLResponse(html.replace('<script type="module" src="/js/app.js?v=2"></script>', entry))


app.mount('/', StaticFiles(directory=ROOT / 'web'), name='static')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=18764)
