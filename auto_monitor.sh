#!/bin/bash

# Auto Monitor Script - Continuously monitors and kills stuck processes
# Usage: ./auto_monitor.sh [start|stop|status]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/auto_monitor.pid"
LOG_FILE="$SCRIPT_DIR/auto_monitor.log"

start_monitor() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "Monitor is already running (PID: $(cat "$PID_FILE"))"
        return 1
    fi
    
    echo "Starting auto monitor..."
    nohup python3 "$SCRIPT_DIR/kill_stuck_processes.py" --auto > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    echo "Monitor started (PID: $!)"
    echo "Logs: $LOG_FILE"
}

stop_monitor() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID"
            echo "Monitor stopped (PID: $PID)"
        else
            echo "Monitor was not running"
        fi
        rm -f "$PID_FILE"
    else
        echo "Monitor is not running"
    fi
}

status_monitor() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "Monitor is running (PID: $(cat "$PID_FILE"))"
        echo "Logs: $LOG_FILE"
        if [ -f "$LOG_FILE" ]; then
            echo "Last 10 lines of log:"
            tail -10 "$LOG_FILE"
        fi
    else
        echo "Monitor is not running"
    fi
}

case "${1:-start}" in
    start)
        start_monitor
        ;;
    stop)
        stop_monitor
        ;;
    status)
        status_monitor
        ;;
    restart)
        stop_monitor
        sleep 2
        start_monitor
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac

