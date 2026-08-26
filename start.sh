#!/bin/bash
TOOL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if command -v supervisorctl &>/dev/null; then
    sudo supervisorctl start avatar_studio
    echo "Started via Supervisor. Status: sudo supervisorctl status avatar_studio"
else
    source "$TOOL_DIR/venv/bin/activate"
    nohup python "$TOOL_DIR/app.py" > "$TOOL_DIR/logs/app.out.log" 2>&1 &
    echo $! > "$TOOL_DIR/.pid"
    echo "Started (PID: $(cat $TOOL_DIR/.pid))"
    echo "Logs: tail -f $TOOL_DIR/logs/app.out.log"
fi
