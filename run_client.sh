#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$SCRIPT_DIR/venv/bin/python" ]; then
    PYTHON_BIN="$SCRIPT_DIR/venv/bin/python"
else
    PYTHON_BIN="python3"
fi

export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/Client Server:$PYTHONPATH"
cd "$SCRIPT_DIR/Client Server"
exec "$PYTHON_BIN" client_app.py "$@"
