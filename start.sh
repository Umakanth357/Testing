#!/bin/bash
# start.sh — Start the Avatar Tool
# Uses supervisor if available, falls back to direct python launch

TOOL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'

# Try supervisor first (preferred — auto-restarts on crash)
if command -v supervisorctl &>/dev/null; then
    sudo supervisorctl start avatar_tool
    sleep 2
    STATUS=$(sudo supervisorctl status avatar_tool 2>/dev/null)
    echo -e "${GREEN}[START]${NC} $STATUS"
    echo -e "${GREEN}[START]${NC} Logs: tail -f $TOOL_DIR/logs/app.out.log"
else
    # Fallback: direct launch in background
    echo -e "${GREEN}[START]${NC} Starting Avatar Tool directly (no supervisor)..."
    mkdir -p "$TOOL_DIR/logs"
    source "$TOOL_DIR/venv/bin/activate"
    nohup python "$TOOL_DIR/app.py" \
        > "$TOOL_DIR/logs/app.out.log" \
        2> "$TOOL_DIR/logs/app.err.log" &
    echo $! > "$TOOL_DIR/logs/app.pid"
    sleep 3
    if kill -0 $(cat "$TOOL_DIR/logs/app.pid") 2>/dev/null; then
        echo -e "${GREEN}[START]${NC} Running (PID $(cat $TOOL_DIR/logs/app.pid))"
        echo -e "${GREEN}[START]${NC} Open: ${CYAN}http://$(curl -s ifconfig.me 2>/dev/null || echo 'localhost')${NC}"
    else
        echo "[START] Failed to start — check logs/app.err.log"
        exit 1
    fi
fi
