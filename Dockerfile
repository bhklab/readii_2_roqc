FROM ghcr.io/prefix-dev/pixi:0.50.2 AS build

COPY . /app
WORKDIR /app

RUN pixi install -e default
RUN pixi shell-hook -e default > /shell-hook.sh

FROM ubuntu:24.04 AS production

COPY --from=build /app/.pixi/envs/default /app/.pixi/envs/default
COPY --from=build /shell-hook.sh /shell-hook.sh
WORKDIR /app

RUN chmod +x /shell-hook.sh

ENTRYPOINT ["bash", "--rcfile", "/shell-hook.sh"]