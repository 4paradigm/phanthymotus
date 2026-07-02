#!/bin/bash
# 截断指定 Docker 容器的日志文件
# 用法: ./truncate_log.sh <container_name> [container_name2 ...]
# 示例: ./truncate_log.sh embodied-unitree-g1
#       ./truncate_log.sh  (无参数则截断所有容器)

set -e

truncate_container() {
    local name="$1"
    local cid
    cid=$(docker inspect --format='{{.ID}}' "$name" 2>/dev/null)
    if [ -z "$cid" ]; then
        echo "[error] container not found: $name"
        return 1
    fi

    local log_dir="/var/lib/docker/containers/${cid}"
    local count
    count=$(find "$log_dir" -name "*.log" 2>/dev/null | wc -l)

    if [ "$count" -eq 0 ]; then
        echo "[skip] no log files found for: $name"
        return 0
    fi

    find "$log_dir" -name "*.log" -exec truncate -s 0 {} \;
    echo "[done] truncated ${count} log file(s) for: $name (${cid:0:12})"
}

if [ $# -eq 0 ]; then
    echo "Truncating logs for ALL containers..."
    for name in $(docker ps -a --format '{{.Names}}'); do
        truncate_container "$name"
    done
else
    for name in "$@"; do
        truncate_container "$name"
    done
fi
