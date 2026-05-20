#!/usr/bin/env bash
# Runs INSIDE itisfoundation/jupyter-math:3.0.5.
# Mirrors the cloud workspace layout from osparc_makefile/Makefile:
#   /home/jovyan/work/workspace/sar-pattern-validation/   <- repo (bind-mounted)
#   /home/jovyan/work/workspace/voila.ipynb               <- copy of notebooks/voila.ipynb
#
# **Test harness only.** This script does NOT install anything Voila itself
# needs at runtime. Production parity rule: anything required to run the
# notebook lives inside the notebook (kernel cells), not here. The only things
# this script provisions are testing-layer tools the user never has in
# production:
#   - apt deps for headless Chromium (Playwright)
#   - git-lfs (test fixture data)
#   - pytest / pytest-playwright / pytest-xdist into the kernel venv
#   - the Chromium browser binary
#
# Subcommands: test [extra pytest args] | shell | exec <cmd...>
set -euo pipefail

WORKSPACE=/home/jovyan/work/workspace
REPO="$WORKSPACE/sar-pattern-validation"
KERNEL_VENV=/home/jovyan/.venv
KERNEL_PY="$KERNEL_VENV/bin/python"

# --------------------------------------------------------------------------
# 1. System deps for headless chromium (Playwright) + curl/git-lfs. Re-run
#    every container start because --rm discards the container layer; the
#    volume-cached /var/lib/apt/lists keeps `apt-get update` fast (~5s) and
#    `apt-get install` is idempotent (~5-10s when packages already in cache).
# --------------------------------------------------------------------------
if [ "$(id -u)" -eq 0 ]; then
    echo ">> apt-get install playwright deps + tooling"
    apt-get update -qq
    apt-get install -y --no-install-recommends \
        curl ca-certificates git-lfs procps tree ffmpeg \
        libnss3 libnspr4 libdbus-1-3 libatk1.0-0 libatk-bridge2.0-0 \
        libcups2 libxkbcommon0 libatspi2.0-0 libx11-6 libxcomposite1 \
        libxdamage1 libxext6 libxfixes3 libxrandr2 libgbm1 libasound2 \
        libpangocairo-1.0-0 libpango-1.0-0 libcairo2 fonts-liberation \
        >/dev/null
    git lfs install --force >/dev/null 2>&1 || true
fi

# --------------------------------------------------------------------------
# 2. Testing-layer deps into the kernel venv. Test-harness only — never
#    installed in production. Installing them into /home/jovyan/.venv keeps
#    tests on the same interpreter the production kernel uses.
#    Uses uv when available; falls back to the kernel venv's pip otherwise
#    (production has neither — this is solely for the test-harness).
# --------------------------------------------------------------------------
echo ">> install testing-layer deps into $KERNEL_VENV"
if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$KERNEL_PY" --quiet \
        pytest pytest-playwright pytest-xdist playwright
else
    "$KERNEL_PY" -m pip install --quiet \
        pytest pytest-playwright pytest-xdist playwright
fi

echo ">> playwright install chromium"
"$KERNEL_VENV/bin/playwright" install chromium

# --------------------------------------------------------------------------
# 3. git-lfs pull + stage the notebook into the workspace root (mirrors
#    copy-notebook in osparc_makefile/Makefile).
# --------------------------------------------------------------------------
cd "$REPO"
git lfs pull >/dev/null 2>&1 || true

mkdir -p "$WORKSPACE"
cp "$REPO/notebooks/voila.ipynb" "$WORKSPACE/voila.ipynb"

# --------------------------------------------------------------------------
# 4. Chown bind-mounted files back to host UID/GID on exit so they don't come
#    out root-owned. Only set when HOST_UID/HOST_GID are passed in.
# --------------------------------------------------------------------------
if [ -n "${HOST_UID:-}" ] && [ -n "${HOST_GID:-}" ]; then
    trap 'find "$REPO" -uid 0 -exec chown "$HOST_UID:$HOST_GID" {} + 2>/dev/null || true' EXIT
fi

# --------------------------------------------------------------------------
# 5. Make the kernel venv's bin/ first on PATH so subprocesses (voila,
#    jupyter, uvx) resolve to the production interpreter. Tests that need to
#    import the package do so via PYTHONPATH (no install into the kernel).
# --------------------------------------------------------------------------
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$KERNEL_VENV/bin:$PATH"
export SAR_PATTERN_VALIDATION_PY39_PYTHON="$KERNEL_PY"

# --------------------------------------------------------------------------
# 6. Dispatch.
# --------------------------------------------------------------------------
CMD="${1:-test}"
shift || true
case "$CMD" in
    test)
        echo ">> running e2e suite (extra args: $*)"
        ARTIFACTS_DIR="$REPO/test-artifacts/playwright"
        mkdir -p "$ARTIFACTS_DIR"
        echo ">> playwright artifacts → $ARTIFACTS_DIR"
        export PLAYWRIGHT_ARTIFACTS_DIR="$ARTIFACTS_DIR"
        "$KERNEL_PY" -m pytest -v -s -o "addopts=" --run-e2e -p no:xdist \
            --video on \
            --tracing retain-on-failure \
            --output "$ARTIFACTS_DIR" \
            tests/test_voila_e2e.py "$@"
        ;;
    shell)
        # Spawn voila in the background bound to 0.0.0.0 so the host port
        # mapping reaches it; print the URL the user should open. Voila keeps
        # running while the user pokes at the workspace from bash. Trap kills
        # it on shell exit so we don't leak a zombie.
        VOILA_PORT_INTERNAL=8866
        VOILA_LOG="$WORKSPACE/voila-shell.log"
        echo ">> starting voila on 0.0.0.0:$VOILA_PORT_INTERNAL (log: $VOILA_LOG)"
        cd "$WORKSPACE"
        "$KERNEL_VENV/bin/voila" voila.ipynb \
            --no-browser \
            --Voila.ip=0.0.0.0 \
            "--port=$VOILA_PORT_INTERNAL" \
            >"$VOILA_LOG" 2>&1 &
        VOILA_PID=$!
        trap 'kill $VOILA_PID 2>/dev/null || true' EXIT
        # Wait up to 30s for voila to bind the port before announcing.
        for _i in $(seq 1 30); do
            if ss -ltn 2>/dev/null | grep -q ":$VOILA_PORT_INTERNAL "; then
                break
            fi
            sleep 1
        done
        HOST_PORT="${VOILA_HOST_PORT:-8866}"
        echo ""
        echo "============================================================"
        echo "  voila ready — open in your host browser:"
        echo ""
        echo "      http://localhost:$HOST_PORT/"
        echo ""
        echo "  workspace:  $WORKSPACE"
        echo "  notebook:   $WORKSPACE/voila.ipynb"
        echo "  kernel py:  $KERNEL_PY"
        echo "  voila log:  $VOILA_LOG  (tail -f to watch)"
        echo "  exit shell to stop voila"
        echo "============================================================"
        echo ""
        bash
        ;;
    exec)
        "$@"
        ;;
    *)
        echo "unknown subcommand: $CMD" >&2
        echo "usage: $0 {test|shell|exec <cmd...>}" >&2
        exit 2
        ;;
esac
