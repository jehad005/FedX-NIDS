#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Cleanly free client port if occupied (defaults to 5000)
CLIENT_PORT="${1:-5000}"
fuser -k "${CLIENT_PORT}/tcp" 2>/dev/null || true

if [ -f "$SCRIPT_DIR/venv/bin/python" ]; then
    PYTHON_BIN="$SCRIPT_DIR/venv/bin/python"
else
    PYTHON_BIN="python3"
fi

export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/Client Server:$SCRIPT_DIR/client_server:$PYTHONPATH"
cd "$SCRIPT_DIR/Client Server"
exec "$PYTHON_BIN" client_app.py "$@"
