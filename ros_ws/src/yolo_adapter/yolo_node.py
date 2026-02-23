import json
import threading
import time
from typing import Optional, Tuple

import numpy as np
import cv2
import requests

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


def rosimg_to_bgr(msg: Image) -> Optional[np.ndarray]:
    """Convert common ROS2 Image encodings to OpenCV BGR without cv_bridge."""
    h, w = msg.height, msg.width
    if h == 0 or w == 0:
        return None

    enc = (msg.encoding or "").lower()
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if enc in ("bgr8", "rgb8"):
        img = data.reshape((h, w, 3))
        if enc == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    if enc in ("mono8", "8uc1"):
        img = data.reshape((h, w))
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if enc in ("bgra8", "rgba8"):
        img = data.reshape((h, w, 4))
        if enc == "rgba8":
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return None


class YoloHttpAdapter(Node):
    def __init__(self):
        super().__init__("yolo_http_adapter")

        # Params
        self.declare_parameter("image_topic", "/image")
        self.declare_parameter("yolo_url", "http://127.0.0.1:9000/detect")
        self.declare_parameter("rate_hz", 5.0)
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("iou", 0.7)
        self.declare_parameter("classes", "0")     # person=0 ; set "" for all
        self.declare_parameter("timeout_s", 1.0)   # HTTP timeout

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.yolo_url = str(self.get_parameter("yolo_url").value)
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.conf = float(self.get_parameter("conf").value)
        self.iou = float(self.get_parameter("iou").value)
        self.classes = str(self.get_parameter("classes").value)
        self.timeout_s = float(self.get_parameter("timeout_s").value)

        # Pub/Sub
        self.pub = self.create_publisher(String, "yolo/detections_json", 10)
        self.sub = self.create_subscription(Image, self.image_topic, self._on_image, 10)

        # State
        self._lock = threading.Lock()
        self._latest: Optional[Tuple[float, np.ndarray]] = None
        self._busy = threading.Event()

        # Timer-driven inference loop (prevents blocking callbacks)
        period = 1.0 / max(self.rate_hz, 0.1)
        self.timer = self.create_timer(period, self._tick)

        self.get_logger().info(f"Subscribing: {self.image_topic}")
        self.get_logger().info(f"YOLO URL: {self.yolo_url}")
        self.get_logger().info("Publishing: /yolo/detections_json (std_msgs/String JSON)")

    def _on_image(self, msg: Image):
        bgr = rosimg_to_bgr(msg)
        if bgr is None:
            self.get_logger().warn_throttle(2.0, f"Unsupported encoding: {msg.encoding}")
            return

        with self._lock:
            self._latest = (time.time(), bgr)

    def _tick(self):
        if self._busy.is_set():
            return

        with self._lock:
            item = self._latest

        if item is None:
            return

        _, bgr = item
        self._busy.set()
        threading.Thread(target=self._infer, args=(bgr,), daemon=True).start()

    def _infer(self, bgr: np.ndarray):
        try:
            ok, jpg = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                return

            files = {"image": ("frame.jpg", jpg.tobytes(), "image/jpeg")}
            params = {"conf": self.conf, "iou": self.iou, "classes": self.classes}

            r = requests.post(self.yolo_url, files=files, params=params, timeout=self.timeout_s)
            r.raise_for_status()
            payload = r.json()

            self.pub.publish(String(data=json.dumps(payload)))

        except Exception as e:
            self.get_logger().warn_throttle(2.0, f"YOLO request failed: {e}")
        finally:
            self._busy.clear()


def main():
    rclpy.init()
    node = YoloHttpAdapter()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
