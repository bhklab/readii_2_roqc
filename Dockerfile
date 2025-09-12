FROM ghcr.io/prefix-dev/pixi:latest

WORKDIR /app
COPY pixi.* pyproject.toml ./
# Install pixi dependencies
RUN pixi install --locked && rm -rf ~/.cache/rattler
COPY . .

# Install required build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 8000
CMD [ "pixi", "shell" , "--frozen", "--no-install"]