# alpine 3.21 的 docker-cli 是 27.x（API 1.47）。3.19 的 25.x 正好卡在新 Docker 的最低
# API 版本上 —— 再早一点的构建（API 1.43）在 thor_pack 上被直接拒绝，升级卡在 90%。
#
# entrypoint.sh 用 `flock -w` 等 compose 锁。busybox 的 flock 没有 -w，而 util-linux 的
# flock 在 3.21 里是单独的 `flock` 包 —— 漏装时 helper 在拿锁这一步就退出。最后那条
# flock -w 让漏装在构建时就失败，而不是等到某台机器升级时。
FROM mirror.ccs.tencentyun.com/library/alpine:3.21
RUN sed -i 's|https://dl-cdn.alpinelinux.org|https://mirrors.cloud.tencent.com|g' /etc/apk/repositories \
    && apk add --no-cache docker-cli docker-cli-compose python3 py3-yaml flock \
    && flock -w 1 /tmp/.flock-check true
COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
