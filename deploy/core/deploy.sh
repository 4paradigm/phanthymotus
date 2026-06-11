#!/usr/bin/env bash
# deploy.sh — 从腾讯云拉取 phanthy 镜像，自动列出最新 15 个版本供选择，拉取后直接启动
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_FILE="${SCRIPT_DIR}/.env"
if [ -f "${ENV_FILE}" ]; then
    # shellcheck source=/dev/null
    source "${ENV_FILE}"
fi

: "${REGISTRY:?REGISTRY not set. Copy .env.example to .env and fill in values.}"
: "${IMAGE_NAMESPACE:?IMAGE_NAMESPACE not set}"

echo "============================================"
echo "获取可用版本列表  registry=${REGISTRY}"
echo "============================================"

# IMAGE_NAMESPACE 可能是 "phanthy-motus" 或老值 "phanthy-motus/motus-core"
# 统一取第一段作为 project，image name 固定为 motus-core
IMAGE_PROJECT="${IMAGE_NAMESPACE%%/*}"   # phanthy-motus
IMAGE_NAME="core"
IMAGE_REPO="${IMAGE_PROJECT}/${IMAGE_NAME}"
TAGS_URL="https://${REGISTRY}/v2/${IMAGE_REPO}/tags/list"

echo "查询 tags: ${TAGS_URL}"
# Probe for WWW-Authenticate header (using v2 endpoint)
echo "获取认证方式..."
WWW_AUTH=$(curl -si "${TAGS_URL}" 2>/dev/null | grep -i 'www-authenticate:' | head -1 || true)
echo "  WWW-Authenticate: ${WWW_AUTH:-（无，尝试匿名访问）}"

TAGS=$(DEPLOY_REGISTRY="${REGISTRY}" DEPLOY_IMAGE_REPO="${IMAGE_REPO}" \
       DEPLOY_WWW_AUTH="${WWW_AUTH}" python3 - <<'PYEOF'
import json, re, sys, base64, os

registry   = os.environ["DEPLOY_REGISTRY"]
image_repo = os.environ["DEPLOY_IMAGE_REPO"]
www_auth   = os.environ["DEPLOY_WWW_AUTH"]

def get_basic_creds():
    cfg_path = os.path.expanduser("~/.docker/config.json")
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
        auths = cfg.get("auths", {})
        for key, val in auths.items():
            if registry in key:
                auth_b64 = val.get("auth", "")
                if auth_b64:
                    decoded = base64.b64decode(auth_b64).decode()
                    user, _, pw = decoded.partition(":")
                    print(f"  [creds] docker config 用户: {user}", file=sys.stderr)
                    return user, pw
    except Exception as e:
        print(f"  [creds] 读取 docker config 失败: {e}", file=sys.stderr)
    return None, None

def get_bearer_token(realm, service, scope, user, pw):
    import urllib.request, urllib.parse
    params = urllib.parse.urlencode({"service": service, "scope": scope})
    token_url = f"{realm}?{params}"
    print(f"  [token] 请求: {token_url}", file=sys.stderr)
    req = urllib.request.Request(token_url)
    if user:
        cred = base64.b64encode(f"{user}:{pw}".encode()).decode()
        req.add_header("Authorization", f"Basic {cred}")
    with urllib.request.urlopen(req, timeout=10) as r:
        data = json.load(r)
    return data.get("token") or data.get("access_token")

def fetch_json(url, headers=None):
    import urllib.request
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)

# Build auth headers
bearer_match = re.search(
    r'Bearer realm="([^"]+)",service="([^"]+)"(?:,scope="([^"]+)")?',
    www_auth, re.IGNORECASE
)
headers = {}
if bearer_match:
    realm   = bearer_match.group(1)
    service = bearer_match.group(2)
    scope   = bearer_match.group(3) or f"repository:{image_repo}:pull"
    print(f"  [auth] Bearer scope={scope}", file=sys.stderr)
    user, pw = get_basic_creds()
    token = get_bearer_token(realm, service, scope, user, pw)
    if token:
        print(f"  [auth] token 获取成功", file=sys.stderr)
        headers = {"Authorization": f"Bearer {token}"}
    else:
        print("  [auth] token 获取失败，尝试匿名", file=sys.stderr)
else:
    user, pw = get_basic_creds()
    if user:
        cred = base64.b64encode(f"{user}:{pw}".encode()).decode()
        headers = {"Authorization": f"Basic {cred}"}
        print(f"  [auth] 使用 Basic auth", file=sys.stderr)

# v2 tags/list + per-tag manifest to get 'created' time and platforms
import urllib.request, concurrent.futures
from datetime import datetime, timezone

v2_url = f"https://{registry}/v2/{image_repo}/tags/list"
data = fetch_json(v2_url, headers)
tags = sorted(
    [t for t in data.get("tags", []) if t.startswith("release.")],
    reverse=True
)[:15]

def get_created(tag):
    try:
        # Fetch manifest — accept both manifest list and single manifest
        mf_url = f"https://{registry}/v2/{image_repo}/manifests/{tag}"
        mf_headers = dict(headers)
        mf_headers["Accept"] = (
            "application/vnd.docker.distribution.manifest.list.v2+json,"
            "application/vnd.oci.image.index.v1+json,"
            "application/vnd.docker.distribution.manifest.v2+json,"
            "application/vnd.oci.image.manifest.v1+json"
        )
        req = urllib.request.Request(mf_url, headers=mf_headers)
        with urllib.request.urlopen(req, timeout=10) as r:
            mf = json.load(r)

        # Multi-arch manifest list
        manifests = mf.get("manifests", [])
        platforms = []
        config_digest = ""
        size_bytes = 0

        if manifests:
            # Collect all platforms, skip attestation entries (unknown/unknown)
            for m in manifests:
                p = m.get("platform", {})
                os_ = p.get("os", "")
                arch = p.get("architecture", "")
                if not os_ or not arch or os_ == "unknown" or arch == "unknown":
                    continue
                variant = p.get("variant", "")
                label = f"{os_}/{arch}" + (f"/{variant}" if variant else "")
                platforms.append(label)
            plat_fmt = " ".join(platforms) or "linux/arm64"
            # Pick arm64 manifest for time/size
            arm_mf = next(
                (m for m in manifests if m.get("platform", {}).get("architecture") == "arm64"),
                manifests[0]
            )
            config_digest = arm_mf.get("digest", "")
            digest_url = f"https://{registry}/v2/{image_repo}/manifests/{config_digest}"
            req2 = urllib.request.Request(digest_url, headers={**headers,
                "Accept": "application/vnd.docker.distribution.manifest.v2+json,"
                          "application/vnd.oci.image.manifest.v1+json"
            })
            with urllib.request.urlopen(req2, timeout=10) as r:
                single_mf = json.load(r)
            config_digest = single_mf.get("config", {}).get("digest", "")
            size_bytes = sum(l.get("size", 0) for l in single_mf.get("layers", []))
        else:
            # Single-arch manifest
            config_digest = mf.get("config", {}).get("digest", "")
            size_bytes = sum(l.get("size", 0) for l in mf.get("layers", []))
            plat_fmt = "linux/arm64"

        # Fetch config blob for 'created'
        time_fmt = ""
        if config_digest:
            blob_url = f"https://{registry}/v2/{image_repo}/blobs/{config_digest}"
            req3 = urllib.request.Request(blob_url, headers=headers)
            with urllib.request.urlopen(req3, timeout=10) as r:
                cfg = json.load(r)
            created = cfg.get("created", "")
            try:
                dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                from datetime import timezone, timedelta
                dt_local = dt.astimezone(timezone(timedelta(hours=8)))
                time_fmt = dt_local.strftime("%Y-%m-%d %H:%M")
            except Exception:
                time_fmt = created[:16] if created else ""

        size_fmt = f"{size_bytes / 1024 / 1024:.0f}MB" if size_bytes else ""
        return tag, time_fmt, size_fmt, plat_fmt
    except Exception as e:
        print(f"  [manifest] {tag}: {e}", file=sys.stderr)
        return tag, "", "", ""

print(f"  [v2] 并发获取 {len(tags)} 个 tag 的 manifest...", file=sys.stderr)
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    results = list(pool.map(get_created, tags))
results.sort(key=lambda r: r[1], reverse=True)
for tag, time_fmt, size_fmt, plat_fmt in results:
    print(f"{tag}\t{time_fmt}\t{size_fmt}\t{plat_fmt}")
PYEOF
)

if [ -z "$TAGS" ]; then
    echo ""
    echo "错误：无法从仓库获取版本列表"
    echo "  - 请确认 REGISTRY / IMAGE_NAMESPACE 配置正确"
    echo "  - 请确认已执行过 docker login ${REGISTRY}（凭证缓存在 ~/.docker/config.json）"
    exit 1
fi

echo ""
echo "请选择要拉取的版本（最新在前）："
echo ""
printf "  %-4s  %-32s  %-16s  %-8s  %s\n" "序号" "TAG" "推送时间(+8)" "大小" "平台"
printf "  %-4s  %-32s  %-16s  %-8s  %s\n" "----" "--------------------------------" "----------------" "--------" "------"

# Parse lines into parallel arrays (mapfile not available in bash 3 / macOS default)
TAG_LIST=()
IDX=1
while IFS= read -r line; do
    [ -z "$line" ] && continue
    tag=$(printf '%s' "$line" | cut -f1)
    time=$(printf '%s' "$line" | cut -f2)
    size=$(printf '%s' "$line" | cut -f3)
    plat=$(printf '%s' "$line" | cut -f4)
    TAG_LIST+=("$tag")
    printf "  %-4s  %-32s  %-16s  %-8s  %s\n" "$IDX" "$tag" "$time" "$size" "$plat"
    IDX=$((IDX+1))
done <<< "$TAGS"

echo ""

# Prompt user to enter number
while true; do
    read -rp "请输入编号 [1-${#TAG_LIST[@]}]: " idx
    if [[ "$idx" =~ ^[0-9]+$ ]] && [ "$idx" -ge 1 ] && [ "$idx" -le "${#TAG_LIST[@]}" ]; then
        CHOSEN_TAG="${TAG_LIST[$((idx-1))]}"
        echo ""
        echo "已选择：${CHOSEN_TAG}"
        break
    fi
    echo "无效输入，请重新输入"
done

TAG="${CHOSEN_TAG}"

echo ""
echo "============================================"
echo "拉取镜像  tag=${TAG}"
echo "============================================"

docker pull --platform linux/arm64 "${REGISTRY}/${IMAGE_PROJECT}/core:${TAG}"

ACTIVE_PROFILES=""
if docker pull --platform linux/arm64 "${REGISTRY}/${IMAGE_PROJECT}/hardware:${TAG}" 2>/dev/null; then
    ACTIVE_PROFILES="${ACTIVE_PROFILES},hardware"
else
    echo "  [跳过] hardware:${TAG} 不存在，忽略"
fi
if docker pull --platform linux/arm64 "${REGISTRY}/${IMAGE_PROJECT}/perception:${TAG}" 2>/dev/null; then
    ACTIVE_PROFILES="${ACTIVE_PROFILES},perception"
else
    echo "  [跳过] perception:${TAG} 不存在，忽略"
fi
# Remove leading comma
ACTIVE_PROFILES="${ACTIVE_PROFILES#,}"

echo ""
echo "镜像拉取完成，更新配置并启动服务..."
echo "============================================"

# Write / update TAG and IMAGE_NAMESPACE in .env so docker compose picks up correct values
_upsert_env() {
    local key="$1" val="$2" file="$3"
    if grep -q "^${key}=" "${file}"; then
        sed -i.bak "s|^${key}=.*|${key}=${val}|" "${file}" && rm -f "${file}.bak"
    else
        echo "${key}=${val}" >> "${file}"
    fi
}

if [ -f "${ENV_FILE}" ]; then
    _upsert_env TAG "${TAG}" "${ENV_FILE}"
    _upsert_env IMAGE_NAMESPACE "${IMAGE_PROJECT}" "${ENV_FILE}"
else
    printf "TAG=%s\nIMAGE_NAMESPACE=%s\n" "${TAG}" "${IMAGE_PROJECT}" > "${ENV_FILE}"
fi

export TAG

cd "${SCRIPT_DIR}"

# 确保持久化数据目录和文件存在（Docker 挂载文件时要求目标已存在，否则会创建为目录）
mkdir -p "${SCRIPT_DIR}/data/world"
touch "${SCRIPT_DIR}/data/data.db"
[ -f "${SCRIPT_DIR}/data/prompt_memory.md" ] || echo "" > "${SCRIPT_DIR}/data/prompt_memory.md"

if [ -n "${ACTIVE_PROFILES}" ]; then
    COMPOSE_PROFILES="${ACTIVE_PROFILES}" docker compose up -d
else
    docker compose up -d
fi

echo ""
echo "服务已启动。查看日志: docker compose logs -f"
echo "数据库位置: ${SCRIPT_DIR}/data/data.db（升级/重启后自动保留）"
