#!/usr/bin/env bash
set -e

pkill -INT -f "/install/orchestrator/lib/orchestrator/orchestrator_node" || true
pkill -INT -f "/install/yolo_adapter/lib/yolo_adapter/yolo_node" || true
sleep 1
pkill -9 -f "/install/orchestrator/lib/orchestrator/orchestrator_node" || true
pkill -9 -f "/install/yolo_adapter/lib/yolo_adapter/yolo_node" || true

echo "Stopping robotics stack..."

PID_DIR="/tmp/vlm_stack_pids"

stop_pid () {
  local name="$1"
  local pid_file="${PID_DIR}/${name}.pid"

  if [[ -f "$pid_file" ]]; then
    local pid
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" 2>/dev/null; then
      echo "Stopping $name (PID $pid)..."
      kill "$pid" 2>/dev/null || true
    else
      echo "$name PID file exists but process not running: $pid"
    fi
    rm -f "$pid_file"
  else
    echo "No PID file for $name (skipping)"
  fi
}

# Stop in reverse order
stop_pid "orchestrator"
stop_pid "yolo_adapter"
stop_pid "cam2image"
stop_pid "yolo"
stop_pid "moondream"

sleep 2

# Cleanup any leftover pids directory if empty
rmdir "$PID_DIR" 2>/dev/null || true

echo "Done."
