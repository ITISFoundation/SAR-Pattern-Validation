#!/usr/bin/env bash
# Host-side wrapper. Spins up itisfoundation/jupyter-math:3.0.5 with the repo
# bind-mounted at the same path the cloud uses, then delegates to
# scripts/run_in_jupyter_math.sh.
#
# Why: a recurring "kernel just died" / silent UI failure on the host venv
# (WSL2) is impossible to disambiguate from a real product bug — Playwright
# only sees the symptom. Running inside the same image production uses
# eliminates the host-vs-container drift dimension. If the bug reproduces
# here, it's a product bug. If it disappears, the host venv is the problem.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${JUPYTER_MATH_IMAGE:-itisfoundation/jupyter-math:3.0.5}"
WORKSPACE_IN_CONTAINER="/home/jovyan/work/workspace"
REPO_IN_CONTAINER="$WORKSPACE_IN_CONTAINER/sar-pattern-validation"
# Forward this port when the user runs `shell` so they can hit Voila from
# their host browser. Override with VOILA_HOST_PORT=... to pin a port;
# otherwise an available port is found automatically starting from 8866.
_find_free_port() {
    local port="${1:-8866}"
    while (echo >/dev/tcp/127.0.0.1/"$port") >/dev/null 2>&1; do
        port=$((port + 1))
    done
    echo "$port"
}
if [ -n "${VOILA_HOST_PORT:-}" ]; then
    # User pinned a port — use it as-is (they own the conflict).
    :
else
    VOILA_HOST_PORT="$(_find_free_port 8866)"
    if [ "$VOILA_HOST_PORT" != "8866" ]; then
        echo "Port 8866 in use — using port $VOILA_HOST_PORT instead."
    fi
fi

# Persistent caches so reruns aren't paying the 200MB+ chromium download,
# the uvx resolve cost, or repeated pip downloads every time.
docker volume create sar-jupyter-math-uv-cache  >/dev/null
docker volume create sar-jupyter-math-pw-cache  >/dev/null
docker volume create sar-jupyter-math-apt-lists >/dev/null
docker volume create sar-jupyter-math-pip-cache >/dev/null

INTERACTIVE_FLAGS=()
if [ -t 0 ] && [ -t 1 ]; then INTERACTIVE_FLAGS=(-it); fi

# --shm-size: headless chromium uses /dev/shm; default 64MB causes flaky
# crashes that look like kernel deaths. Bump to 2g to match Playwright's
# recommendation.
# --entrypoint=bash: jupyter-math's default entrypoint expects oSPARC
# dynamic-sidecar env vars and dies under `set -u` when they're unset. We
# don't need its jupyter-server startup logic — the inner script runs Voila
# directly.
# Forward the voila port only for the `shell` subcommand — `test` runs voila
# bound to 127.0.0.1 inside the container, no host exposure needed.
PORT_FLAGS=()
if [ "${1:-test}" = "shell" ]; then
    PORT_FLAGS=(-p "$VOILA_HOST_PORT:8866")

    # Auto-open the host browser once Voila is reachable on the forwarded port.
    # This runs in the background and is best-effort (no hard failure if no GUI).
    (
        URL="http://localhost:$VOILA_HOST_PORT/"

        for _i in $(seq 1 60); do
            if (echo >"/dev/tcp/127.0.0.1/$VOILA_HOST_PORT") >/dev/null 2>&1; then
                break
            fi
            sleep 1
        done

        if command -v xdg-open >/dev/null 2>&1; then
            xdg-open "$URL" >/dev/null 2>&1 || true
        elif command -v gio >/dev/null 2>&1; then
            gio open "$URL" >/dev/null 2>&1 || true
        elif command -v sensible-browser >/dev/null 2>&1; then
            sensible-browser "$URL" >/dev/null 2>&1 || true
        elif command -v wslview >/dev/null 2>&1; then
            wslview "$URL" >/dev/null 2>&1 || true
        fi
    ) &
fi

exec docker run --rm "${INTERACTIVE_FLAGS[@]}" \
    --name "sar-voila-jm-$$" \
    --user 0 \
    --shm-size=2g \
    --network bridge \
    --entrypoint=bash \
    "${PORT_FLAGS[@]}" \
    -e HOME=/root \
    -e UV_CACHE_DIR=/root/.cache/uv \
    -e PIP_CACHE_DIR=/root/.cache/pip \
    -e PLAYWRIGHT_BROWSERS_PATH=/root/.cache/ms-playwright \
    -e SAR_PATTERN_VALIDATION_BACKEND_MODE=local \
    -e MPLBACKEND=Agg \
    -e VOILA_HOST_PORT="$VOILA_HOST_PORT" \
    -e HOST_UID="$(id -u)" \
    -e HOST_GID="$(id -g)" \
    -v "$REPO_ROOT:$REPO_IN_CONTAINER" \
    -v sar-jupyter-math-uv-cache:/root/.cache/uv \
    -v sar-jupyter-math-pip-cache:/root/.cache/pip \
    -v sar-jupyter-math-pw-cache:/root/.cache/ms-playwright \
    -v sar-jupyter-math-apt-lists:/var/lib/apt/lists \
    -w "$REPO_IN_CONTAINER" \
    "$IMAGE" \
    "$REPO_IN_CONTAINER/scripts/run_in_jupyter_math.sh" "$@"
