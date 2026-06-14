#!/bin/bash
TOOL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if command -v supervisorctl &>/dev/null; then
    sudo supervisorctl stop avatar_studio
else
    if [ -f "$TOOL_DIR/.pid" ]; then
        kill "$(cat $TOOL_DIR/.pid)" 2>/dev/null && echo "Stopped"
        rm -f "$TOOL_DIR/.pid"
    else
        pkill -f "python.*app.py" && echo "Stopped" || echo "Not running"
    fi
fi
