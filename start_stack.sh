#!/usr/bin/env bash
set -e

echo "===== VLM ROBOTICS STACK START ====="

############################################
# Step 0 — Lock clocks
############################################
echo "[0] Locking Jetson clocks..."
sudo jetson_clocks

############################################
# Step 1 — PyTorch allocator tuning
############################################
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64,garbage_collection_threshold:0.8

############################################
# Activate venv once
############################################
cd ~/molmo_fyp_ros
source .venv/bin/activate

############################################
# Step 2 — Start Moondream FIRST
############################################
echo "[1] Starting Moondream service..."

nohup uvicorn services.moondream_service.app:app \
    --host 0.0.0.0 --port 8001 \
    > /tmp/moondream.log 2>&1 &

############################################
# Wait until Moondream responds
############################################
echo "Waiting for Moondream to load model..."

until curl -s http://127.0.0.1:8001/health > /dev/null; do
    sleep 3
    echo "Moondream still loading..."
done

echo "Moondream is READY."

############################################
# VERY IMPORTANT — drop caches
############################################
echo "[2] Dropping filesystem caches..."
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

sleep 5

free -h

############################################
# Step 3 — Start YOLO
############################################
echo "[3] Starting YOLO service..."

nohup uvicorn services.yolo_service.app:app \
    --host 0.0.0.0 --port 9000 \
    > /tmp/yolo.log 2>&1 &

sleep 8
free -h

############################################
# Step 4 — Start Webcam
############################################
echo "[4] Starting webcam..."

source ~/molmo_fyp_ros/ros_ws/install/setup.bash

nohup ros2 run image_tools cam2image \
    > /tmp/cam.log 2>&1 &

sleep 5

############################################
# Step 5 — Start YOLO adapter
############################################
echo "[5] Starting YOLO ROS adapter..."

nohup ros2 run yolo_adapter yolo_node --ros-args \
    -p image_topic:=/image \
    -p yolo_url:=http://127.0.0.1:9000/detect \
    -p classes:="[0]" \
    > /tmp/yolo_adapter.log 2>&1 &

sleep 5

############################################
# Step 6 — Start Orchestrator
############################################
echo "[6] Starting Orchestrator..."

nohup ros2 run orchestrator orchestrator_node --ros-args \
    -p image_topic:=/image \
    -p det_topic:=/yolo/detections_json \
    -p moondream_url:=http://127.0.0.1:8001/caption \
    -p moondream_hz:=1.0 \
    > /tmp/orchestrator.log 2>&1 &

echo ""
echo "===== STACK STARTED SUCCESSFULLY ====="
echo ""
echo "Check logs with:"
echo "tail -f /tmp/moondream.log"
echo "tail -f /tmp/yolo.log"
echo "tail -f /tmp/orchestrator.log"

