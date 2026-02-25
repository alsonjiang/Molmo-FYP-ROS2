#!/usr/bin/env bash
set -euo pipefail

echo "===== VLM ROBOTICS STACK START ====="

REPO_DIR="$HOME/molmo_fyp_ros"
PID_DIR="/tmp/vlm_stack_pids"
LOG_DIR="/tmp"
MOONDREAM_PORT="8001"
YOLO_PORT="9000"

mkdir -p "$PID_DIR"

############################################
# Optional: Lock clocks (set JETSON_CLOCKS=1)
############################################
if [[ "${JETSON_CLOCKS:-0}" == "1" ]]; then
  echo "[0] Locking Jetson clocks..."
  sudo jetson_clocks
else
  echo "[0] Skipping jetson_clocks (set JETSON_CLOCKS=1 to enable)"
fi

############################################
# PyTorch allocator tuning
############################################
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64,garbage_collection_threshold:0.8"

############################################
# Activate venv once
############################################
cd "$REPO_DIR"
source .venv/bin/activate

############################################
# Helper: start a command in background + record PID
############################################
start_bg () {
  local name="$1"; shift
  local logfile="$1"; shift

  echo "Starting $name..."
  nohup "$@" > "$logfile" 2>&1 &
  echo $! > "${PID_DIR}/${name}.pid"
  echo "$name PID: $(cat "${PID_DIR}/${name}.pid")"
}

############################################
# Step 1 — Start Moondream FIRST
############################################
echo "[1] Starting Moondream service..."
start_bg "moondream" "${LOG_DIR}/moondream.log" \
  uvicorn services.moondream_service.app:app \
    --host 0.0.0.0 --port "${MOONDREAM_PORT}"

############################################
# Wait until Moondream responds (with timeout)
############################################
echo "Waiting for Moondream to load model..."
deadline=$((SECONDS+180))  # 3 minutes
until curl -sf "http://127.0.0.1:${MOONDREAM_PORT}/health" > /dev/null; do
  sleep 3
  echo "Moondream still loading..."
  if (( SECONDS > deadline )); then
    echo "ERROR: Moondream failed to start (timeout). Check ${LOG_DIR}/moondream.log"
    exit 1
  fi
done
echo "Moondream is READY."
free -h || true

############################################
# Step 2 — Start YOLO
############################################
echo "[2] Starting YOLO service..."
start_bg "yolo" "${LOG_DIR}/yolo.log" \
  uvicorn services.yolo_service.app:app \
    --host 0.0.0.0 --port "${YOLO_PORT}"

sleep 3
free -h || true

############################################
# Step 3 — ROS env
############################################
echo "[3] Sourcing ROS workspace..."
set +u
source /opt/ros/humble/setup.bash
source "$REPO_DIR/ros_ws/install/setup.bash"
set -u

############################################
# Step 4 — Start Webcam
############################################
echo "[4] Starting webcam..."
# Force topic name so your pipeline is consistent
start_bg "cam2image" "${LOG_DIR}/cam.log" \
  ros2 run image_tools cam2image --ros-args -r image:=/image

sleep 2

############################################
# Step 5 — Start YOLO adapter
############################################
echo "[5] Starting YOLO ROS adapter..."
start_bg "yolo_adapter" "${LOG_DIR}/yolo_adapter.log" \
  ros2 run yolo_adapter yolo_node --ros-args \
    -p image_topic:=/image \
    -p yolo_url:="http://127.0.0.1:${YOLO_PORT}/detect" \
    -p classes:="[0]"

sleep 2

############################################
# Step 6 — Start Orchestrator
############################################
echo "[6] Starting Orchestrator..."
start_bg "orchestrator" "${LOG_DIR}/orchestrator.log" \
  ros2 run orchestrator orchestrator_node --ros-args \
    -p image_topic:=/image \
    -p det_topic:=/yolo/detections_json \
    -p moondream_url:="http://127.0.0.1:${MOONDREAM_PORT}/caption" \
    -p moondream_hz:=1.0
    -p moondream_enabled:=true

echo ""
echo "===== STACK STARTED SUCCESSFULLY ====="
echo ""
echo "Logs:"
echo "  tail -f /tmp/moondream.log"
echo "  tail -f /tmp/yolo.log"
echo "  tail -f /tmp/cam.log"
echo "  tail -f /tmp/yolo_adapter.log"
echo "  tail -f /tmp/orchestrator.log"
echo ""
echo "PIDs saved in: ${PID_DIR}"
