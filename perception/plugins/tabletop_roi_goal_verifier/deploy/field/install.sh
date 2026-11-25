#!/usr/bin/env bash
# All generated files and dependency caches stay inside a new personal directory.
set -euo pipefail
umask 077
KIT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
DEST=${1:-/home/agilex/goal-verifier-roi-20261114-r1}
PYTHON=${ROI_PYTHON:-/usr/bin/python3.10}
METHOD=${ROI_INSTALL_METHOD:-auto}
fail(){ echo "安装失败：$*" >&2; exit 1; }
[[ "$DEST" = /* ]] || fail "部署目录必须是绝对路径"
[[ ! -e "$DEST" && ! -L "$DEST" ]] || fail "目录已存在，拒绝覆盖：$DEST；请选择新的个人目录"
[[ "$METHOD" == auto || "$METHOD" == venv || "$METHOD" == target ]] || fail "ROI_INSTALL_METHOD 必须是 auto/venv/target"
[[ -f "$KIT/SHA256SUMS" && -f "$KIT/tabletop-roi-goal-verifier.tar.gz" ]] || fail "交付包或 SHA256SUMS 缺失"
(cd "$KIT" && sha256sum --check --strict SHA256SUMS) || fail "交付文件校验失败，未安装"
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,10), "现场包需要 Python 3.10"' || fail "请用 ROI_PYTHON 指定 Ubuntu/ROS 对应的 Python 3.10"
# mkdir itself refuses an existing file, directory or symlink, including races.
mkdir -- "$DEST" || fail "不能创建新目录（父目录必须已存在且属于本人）：$DEST"
DEST=$(cd -- "$DEST" && pwd -P)
export TMPDIR="$DEST/.tmp" PIP_CACHE_DIR="$DEST/.cache/pip" XDG_CACHE_HOME="$DEST/.cache"
export PYTHONUSERBASE="$DEST/.user" PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$DEST/logs" "$DEST/evidence" "$DEST/src" "$DEST/bin" "$DEST/wheels"
trap 'rc=$?; if ((rc)); then echo "安装失败（退出码 $rc）。已生成的文件仅在 $DEST；请查看 $DEST/logs/install.log。重试请选择新的目录。" >&2; fi' EXIT
cp -- "$KIT/"*.whl "$DEST/wheels/"
if [[ -d "$KIT/wheelhouse" ]]; then cp -- "$KIT/wheelhouse/"*.whl "$DEST/wheels/"; fi
cp -- "$KIT/requirements.lock" "$DEST/requirements.lock"
cp -- "$KIT/SHA256SUMS" "$DEST/DELIVERY-SHA256SUMS"
cp -- "$KIT/"*.sh "$KIT/doctor.py" "$KIT/evidence.py" "$KIT/runner.py" "$DEST/bin/"
cp -- "$KIT/config.d435.template.json" "$DEST/config.json"
cp -- "$KIT/dds-local.xml" "$DEST/dds-local.xml"
cp -- "$KIT/ONSITE.md" "$DEST/ONSITE.md"
"$PYTHON" - "$KIT/tabletop-roi-goal-verifier.tar.gz" "$DEST/src" <<'PY'
import sys,tarfile
from pathlib import Path
archive,dest=sys.argv[1:]
with tarfile.open(archive) as tar:
 for m in tar.getmembers():
  p=Path(m.name)
  if p.is_absolute() or '..' in p.parts or not (m.isfile() or m.isdir()):raise SystemExit('Unsafe archive member: '+m.name)
 tar.extractall(dest)
PY
PIP_WHEELS=("$DEST/wheels/"pip-*.whl)
PIP_WHEEL=${PIP_WHEELS[0]}
[[ -f "$PIP_WHEEL" ]] || fail "缺少 pip bootstrap wheel"
SELECTED=target
if [[ "$METHOD" != target ]]; then
  if "$PYTHON" -m venv --without-pip "$DEST/venv" >"$DEST/logs/venv.log" 2>&1 && "$DEST/venv/bin/python" -c 'import sys; assert sys.prefix != sys.base_prefix'; then
    SELECTED=venv
  elif [[ "$METHOD" == venv ]]; then
    fail "venv 不可用；查看 logs/venv.log。可改用 ROI_INSTALL_METHOD=target"
  fi
fi
if [[ "$SELECTED" == venv ]]; then
  PYTHON="$DEST/venv/bin/python"
  INSTALL_ARGS=()
else
  PYTHON=$(command -v "$PYTHON")
  mkdir "$DEST/deps"
  INSTALL_ARGS=(--target "$DEST/deps")
fi
# Bootstrap pip from a verified wheel without ensurepip or system pip installation.
PYTHONPATH="$PIP_WHEEL" "$PYTHON" -m pip --isolated install --disable-pip-version-check --no-cache-dir --no-compile \
  --no-index --find-links "$DEST/wheels" "${INSTALL_ARGS[@]}" -r "$DEST/requirements.lock" >"$DEST/logs/install.log" 2>&1
"$PYTHON" - "$DEST" "$SELECTED" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[1],'runtime.json').write_text(json.dumps(dict(python=sys.executable,method=sys.argv[2]),indent=2))
PY
PYTHONPATH="$DEST/deps:$DEST/src/perception" "$PYTHON" -c 'import cv2,numpy,jsonschema; from plugins.tabletop_roi_goal_verifier import TabletopROIGoalVerifierPlugin; print("ROI runtime import OK")'
echo "安装完成：$DEST（$SELECTED）。先只读核对相机，再填写 config.json 的记录字段；启动：bash $DEST/bin/start.sh"
