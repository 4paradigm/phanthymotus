# PR Review Agent image.
#
# Mirrors default to Tencent Cloud, matching the rest of the project's
# Dockerfiles. Override at build time for a host outside the Tencent VPC:
#   docker compose build --build-arg MIRROR_BASE=docker.io \
#       --build-arg PYPI_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple/ \
#       --build-arg APT_MIRROR=mirrors.tuna.tsinghua.edu.cn

ARG MIRROR_BASE=mirror.ccs.tencentyun.com
FROM ${MIRROR_BASE}/library/python:3.12-slim

ARG PYPI_MIRROR=https://mirrors.tencentyun.com/pypi/simple/
ARG APT_MIRROR=mirrors.tencentyun.com
# Docker apt repo. download.docker.com is unreachable from the Tencent VMs, so
# the mirror is the default; override for a host where the upstream works.
ARG DOCKER_REPO=https://mirrors.tencentyun.com/docker-ce/linux/debian

# Point apt at the mirror before the first update, so no request goes to the
# default Debian hosts. Debian 12 ships the DEB822 sources file; the older
# sources.list is rewritten too in case the base image changes.
RUN set -eux; \
    if [ -n "${APT_MIRROR}" ]; then \
        if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
            sed -i "s|deb.debian.org|${APT_MIRROR}|g; s|security.debian.org|${APT_MIRROR}|g" \
                /etc/apt/sources.list.d/debian.sources; \
        fi; \
        if [ -f /etc/apt/sources.list ]; then \
            sed -i "s|deb.debian.org|${APT_MIRROR}|g; s|security.debian.org|${APT_MIRROR}|g" \
                /etc/apt/sources.list; \
        fi; \
    fi

# git + ssh for the bare clones and worktrees; docker CLI + buildx because the
# agent shells out to the repos' build scripts, which invoke `docker buildx`.
#
# The Docker apt repo comes from the Tencent mirror, not download.docker.com:
# that host is unreachable from these VMs (measured 0/3, failing in 0.16s) while
# the mirror answers in 0.08s. This only surfaced when adding openssh-client
# invalidated the cached layer — the build had been passing on cache alone.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        gnupg \
        openssh-client; \
    install -m 0755 -d /etc/apt/keyrings; \
    for attempt in 1 2 3; do \
        curl -fsSL --connect-timeout 15 "${DOCKER_REPO}/gpg" \
            -o /etc/apt/keyrings/docker.asc && break; \
        echo "docker gpg fetch failed (attempt ${attempt})"; sleep 3; \
    done; \
    test -s /etc/apt/keyrings/docker.asc; \
    chmod a+r /etc/apt/keyrings/docker.asc; \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
${DOCKER_REPO} $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
        > /etc/apt/sources.list.d/docker.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        docker-ce-cli \
        docker-buildx-plugin \
        docker-compose-plugin; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY agents/pr_review/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -i "${PYPI_MIRROR}" \
        --trusted-host "$(echo "${PYPI_MIRROR}" | sed -E 's#https?://([^/]+).*#\1#')" \
        -r requirements.txt

COPY agents/ /app/agents/

RUN mkdir -p /data/repos

EXPOSE 25000

CMD ["python", "-m", "agents.pr_review.server"]
