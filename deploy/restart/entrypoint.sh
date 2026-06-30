#!/bin/sh
# entrypoint.sh — stop/remove old container, run new image with same config
# Env: CONTAINER_NAME, NEW_IMAGE, NETWORK, ENV_FILE (path on host)
set -e

: "${CONTAINER_NAME:?CONTAINER_NAME required}"
: "${NEW_IMAGE:?NEW_IMAGE required}"

echo "[restart] target: ${CONTAINER_NAME} → ${NEW_IMAGE}"

# Capture run config from old container before stopping
NETWORK_MODE=$(docker inspect "${CONTAINER_NAME}" \
    --format '{{.HostConfig.NetworkMode}}' 2>/dev/null || true)
BINDS=$(docker inspect "${CONTAINER_NAME}" \
    --format '{{range .HostConfig.Binds}}{{.}} {{end}}' 2>/dev/null || true)
PORT_BINDINGS=$(docker inspect "${CONTAINER_NAME}" \
    --format '{{range $p,$b := .HostConfig.PortBindings}}{{$p}}:{{(index $b 0).HostPort}} {{end}}' \
    2>/dev/null || true)
ENV_VARS=$(docker inspect "${CONTAINER_NAME}" \
    --format '{{range .Config.Env}}{{.}} {{end}}' 2>/dev/null || true)
IPC_MODE=$(docker inspect "${CONTAINER_NAME}" \
    --format '{{.HostConfig.IpcMode}}' 2>/dev/null || true)
SHM_SIZE=$(docker inspect "${CONTAINER_NAME}" \
    --format '{{.HostConfig.ShmSize}}' 2>/dev/null || true)
RUNTIME=$(docker inspect "${CONTAINER_NAME}" \
    --format '{{.HostConfig.Runtime}}' 2>/dev/null || true)

echo "[restart] network_mode: ${NETWORK_MODE}"
echo "[restart] binds:        ${BINDS}"
echo "[restart] ports:        ${PORT_BINDINGS}"

# Stop & remove old container
docker stop "${CONTAINER_NAME}" 2>/dev/null || true
docker rm   "${CONTAINER_NAME}" 2>/dev/null || true

# Build docker run args
ARGS="-d --name ${CONTAINER_NAME} --restart unless-stopped"

for bind in ${BINDS}; do
    ARGS="${ARGS} -v ${bind}"
done

# Network: if host mode, use --network host (skip port mappings — not applicable)
if [ "${NETWORK_MODE}" = "host" ]; then
    ARGS="${ARGS} --network host"
else
    for port_pair in ${PORT_BINDINGS}; do
        # format: 8080/tcp:8080
        container_port="${port_pair%%:*}"
        host_port="${port_pair##*:}"
        ARGS="${ARGS} -p ${host_port}:${container_port%/tcp}"
    done
    ARGS="${ARGS} --network ${NETWORK_MODE}"
fi

# IPC mode (required for FastDDS SHM transport)
if [ "${IPC_MODE}" = "host" ]; then
    ARGS="${ARGS} --ipc host"
fi

# SHM size
if [ -n "${SHM_SIZE}" ] && [ "${SHM_SIZE}" != "0" ]; then
    ARGS="${ARGS} --shm-size ${SHM_SIZE}"
fi

# GPU runtime (nvidia)
if [ -n "${RUNTIME}" ] && [ "${RUNTIME}" != "runc" ]; then
    ARGS="${ARGS} --runtime ${RUNTIME}"
fi

for env in ${ENV_VARS}; do
    # Skip build-time metadata baked into the new image
    case "${env}" in IMAGE_TAG=*) continue ;; esac
    ARGS="${ARGS} -e ${env}"
done

# Explicitly set IMAGE_TAG from new image reference to ensure correct version display
NEW_TAG="${NEW_IMAGE##*:}"
ARGS="${ARGS} -e IMAGE_TAG=${NEW_TAG}"

echo "[restart] starting new container..."
# shellcheck disable=SC2086
docker run ${ARGS} "${NEW_IMAGE}"

echo "[restart] done. new container is up."
