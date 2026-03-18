#!/usr/bin/env bash
set -uo pipefail

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
      kill -INT "$pid" 2>/dev/null || true

      # Wait a bit for graceful shutdown
      for _ in {1..10}; do
        if ! kill -0 "$pid" 2>/dev/null; then
          break
        fi
        sleep 0.5
      done

      # Force kill if still alive
      if kill -0 "$pid" 2>/dev/null; then
        echo "Force killing $name (PID $pid)..."
        kill -9 "$pid" 2>/dev/null || true
      fi
    else
      echo "$name PID file exists but process not running: $pid"
    fi

    rm -f "$pid_file"
  else
    echo "No PID file for $name (skipping)"
  fi
}

############################################
# Extra cleanup for ROS nodes launched outside PID tracking
# or child processes that may survive
############################################
pkill -INT -f "motion_controller_node" || true
pkill -INT -f "orchestrator_node" || true
pkill -INT -f "yolo_node" || true
pkill -INT -f "cam2image" || true
pkill -INT -f "turtlebot3_bringup" || true
pkill -INT -f "robot.launch.py" || true

sleep 2

############################################
# Stop tracked processes in reverse order
############################################
stop_pid "motion_controller"
stop_pid "orchestrator"
stop_pid "yolo_adapter"
stop_pid "cam2image"
stop_pid "turtlebot3_bringup"
stop_pid "yolo"
stop_pid "moondream"

############################################
# Final hard kill cleanup for stubborn leftovers
############################################
pkill -9 -f "motion_controller_node" || true
pkill -9 -f "orchestrator_node" || true
pkill -9 -f "yolo_node" || true
pkill -9 -f "cam2image" || true
pkill -9 -f "turtlebot3_bringup" || true
pkill -9 -f "robot.launch.py" || true

# Cleanup PID dir if empty
rmdir "$PID_DIR" 2>/dev/null || true

echo "Done."
