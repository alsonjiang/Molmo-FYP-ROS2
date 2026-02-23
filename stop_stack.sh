#!/usr/bin/env bash

echo "Stopping robotics stack..."

pkill -f uvicorn
pkill -f ros2
pkill -f cam2image

sleep 2

echo "Done."
