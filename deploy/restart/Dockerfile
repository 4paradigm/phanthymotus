FROM mirror.ccs.tencentyun.com/library/alpine:3.19
RUN sed -i 's|https://dl-cdn.alpinelinux.org|https://mirrors.cloud.tencent.com|g' /etc/apk/repositories \
    && apk add --no-cache docker-cli
COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
