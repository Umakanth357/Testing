#!/bin/bash
# stop.sh — Stop the Avatar Tool

TOOL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YELLOW='\033[1;33m'; NC='\033[0m'

# Try supervisor first
if command -v supervisorctl &>/dev/null; then
    sudo supervisorctl stop avatar_tool
    echo -e "${YELLOW}[STOP]${NC} avatar_tool stopped via supervisor"
else
    # Kill by PID file
    PID_FILE="$TOOL_DIR/logs/app.pid"
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID"
            rm -f "$PID_FILE"
            echo -e "${YELLOW}[STOP]${NC} Stopped PID $PID"
        else
            echo -e "${YELLOW}[STOP]${NC} Process $PID not running (stale PID file removed)"
            rm -f "$PID_FILE"
        fi
    else
        # Last resort: kill by process name
        pkill -f "python.*app.py" && echo -e "${YELLOW}[STOP]${NC} Stopped app.py process" || echo -e "${YELLOW}[STOP]${NC} No running process found"
    fi
fi
