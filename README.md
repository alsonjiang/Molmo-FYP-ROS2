
# Application of Multimodal Models in Robotics Vision Tasks

This project demonstrates a lightweight perception pipeline that combines object detection and vision-language reasoning on a mobile robot. The system runs on a Jetson Orin NX and integrates YOLO-based person detection, Moondream Vision-Language Model (VLM) for caption generation, and ROS2-based robot control on a TurtleBot3 Burger platform.

The goal of this project is to explore the feasibility of deploying multimodal models for robotic perception tasks on embedded hardware.


## ROS2 Nodes

###  Orchestrator Node

The orchestrator node acts as the perception coordinator of the system.

Responsibilities:

Subscribe to camera images `/image`

Subscribe to YOLO detection output `/yolo/detections_json`

Filter and select person detections

Apply stability gating to avoid noisy detections

Crop the detected person region

Send the cropped image to the VLM inference service

Receive and display caption results

Publish debug topics:

`/vlm/caption`

`/vlm/latency_ms`

`/vlm/state`

Render detection boxes and captions on the live image stream

Publish `/image_annotated`

The node uses a background worker thread to perform VLM requests without blocking the main ROS loop.

### Motion Controller Node
The motion controller node implements a minimal perception-driven behaviour controller.

Responsibilities:

Subscribe to YOLO detection output

Filter valid person detections

Publish velocity commands to `/cmd_vel`

Behaviour policy:

No valid person detection → robot rotates slowly

Person detection present → robot stops

If the detection disappears for longer than a timeout window, the robot resumes its rotational search.

## Installation

### Install ROS2 Humble
Follow the official ROS2 installation guide:

https://docs.ros.org/en/humble/Installation.html

### Clone Repository
```bash
git clone https://github.com/alsonjiang/molmo_fyp_ros.git 
```

```bash
cd molmo_fyp_ros
```

### Create Python Environment
```bash
python3 -m venv .venv 
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

### Build ROS2 Workspace
```bash
cd ros_ws
```

```bash
colcon build
```

```bash
source install/setup.bash
```





## Running the System

### Starting the full stack

```bash
  ./start_stack.sh
```
This launches:

-Moondream VLM service

-YOLO detection service

-TurtleBot3 bringup

-Camera node

-YOLO ROS adapter

-Orchestrator node

-Motion controller node

### Stopping the full stack
```bash
./stop_stack.sh
```
This stops all background services and ROS nodes.

### Logs
Runtime logs are stored in /tmp
```bash
tail -f /tmp/orchestrator.log
tail -f /tmp/motion_controller.log
tail -f /tmp/yolo.log
tail -f /tmp/moondream.log
```

To view latency logs, use the following command
```bash
tail ~/molmo_fyp_ros/data/latency/*/latency.jsonl
```
These logs are also saved in /data


