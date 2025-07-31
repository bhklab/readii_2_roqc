FROM ghcr.io/prefix-dev/pixi:latest

WORKDIR /app
COPY . .

# Install required build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install pixi dependencies
RUN pixi install --locked && rm -rf ~/.cache/rattler
EXPOSE 8000
CMD [ "pixi", "shell" , "--no-lockfile-update"]