#!/usr/bin/env bash
# Optional LOCAL developer setup. Does not install system libraries/services.
set -euo pipefail
TASK_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
cd "$TASK_ROOT"
: "${GOAL_BROWSER_PYTHON:?Point to a new environment containing playwright and Core dependencies}"
export PLAYWRIGHT_BROWSERS_PATH="$TASK_ROOT/.cache/browsers"
"$GOAL_BROWSER_PYTHON" -m playwright install chromium
mkdir -p .cache/native .cache/dashboard-cdn
(
  cd .cache/native
  apt-get download libnspr4 libnss3 libasound2 fonts-noto-cjk
  for deb in ./*.deb; do dpkg-deb --extract "$deb" root; done
)
"$GOAL_BROWSER_PYTHON" - <<'PY'
from pathlib import Path
import hashlib,json,urllib.request
task=Path.cwd().resolve()
(task/'.cache/fontconfig').mkdir(exist_ok=True)
(task/'.cache/fonts.conf').write_text('<fontconfig><include>/etc/fonts/fonts.conf</include><dir>'+str(task/'.cache/native/root/usr/share/fonts')+'</dir><cachedir>'+str(task/'.cache/fontconfig')+'</cachedir></fontconfig>')
root=Path('.cache/dashboard-cdn');manifest={}
for name,url in {'three.module.js':'https://cdn.jsdelivr.net/npm/three@0.164/build/three.module.js',
                 'OrbitControls.js':'https://cdn.jsdelivr.net/npm/three@0.164/examples/jsm/controls/OrbitControls.js'}.items():
    with urllib.request.urlopen(url,timeout=40) as response:data=response.read()
    (root/name).write_bytes(data);manifest[name]={'url':url,'sha256':hashlib.sha256(data).hexdigest()}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2))
PY
printf '%s\n' 'Use absolute paths for LD_LIBRARY_PATH=.cache/native/root/usr/lib/x86_64-linux-gnu, PLAYWRIGHT_BROWSERS_PATH=.cache/browsers, FONTCONFIG_FILE=.cache/fonts.conf and XDG_CACHE_HOME=.cache/user-cache for the E2E command.'
