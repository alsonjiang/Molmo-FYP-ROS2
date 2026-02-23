#!/usr/bin/env bash
set -e
sudo apt update

sudo apt install -y \
  python3-pip python3-venv \
  python3-opencv \
  ros-humble-vision-msgs ros-humble-cv-bridge ros-humble-image-transport \
  libcudss0-cuda-12

sudo ldconfig

