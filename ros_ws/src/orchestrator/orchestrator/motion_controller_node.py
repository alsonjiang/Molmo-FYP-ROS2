#!/usr/bin/env python3
import json
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String


@dataclass
class Det:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    cls: Union[int, str] = "person"


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def iou(a: Det, b: Det) -> float:
    xA = max(a.x1, b.x1)
    yA = max(a.y1, b.y1)
    xB = min(a.x2, b.x2)
    yB = min(a.y2, b.y2)
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter <= 0:
        return 0.0
    areaA = max(0, a.x2 - a.x1) * max(0, a.y2 - a.y1)
    areaB = max(0, b.x2 - b.x1) * max(0, b.y2 - b.y1)
    denom = float(areaA + areaB - inter)
    return inter / denom if denom > 0 else 0.0


class MotionController(Node):
    """
    Minimal movement-only controller

    Behaviour:
    - no valid person/face detection -> spin slowly
    - stable valid detection -> stop
    - when target disappears -> resume spin

    This node does not trigger VLM as the orchestrator already handles VLM triggering.
    """

    def __init__(self) -> None:
        super().__init__("motion_controller_node")

        # Topics
        self.declare_parameter("det_topic", "/yolo/detections_json")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")

        # Detection behavior
        self.declare_parameter("min_conf", 0.25)
        self.declare_parameter("choose_best_only", True)

        # Person filtering
        self.declare_parameter("person_class_names", ["person", "face"])
        self.declare_parameter("person_class_ids", [0])

        # Motion behavior
        self.declare_parameter("search_linear_x", 0.0)
        self.declare_parameter("search_angular_z", 0.20)

        # Stability logic
        self.declare_parameter("target_stable_required_s", 0.30)
        self.declare_parameter("target_iou_stable_thresh", 0.6)
        self.declare_parameter("target_timeout_s", 0.75)

        # Control loop
        self.declare_parameter("control_hz", 10.0)

        self.det_topic = str(self.get_parameter("det_topic").value)
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)

        self.min_conf = float(self.get_parameter("min_conf").value)
        self.choose_best_only = bool(self.get_parameter("choose_best_only").value)

        self.person_class_names = {
            str(x).strip().lower() for x in self.get_parameter("person_class_names").value
        }
        self.person_class_ids = {
            int(x) for x in self.get_parameter("person_class_ids").value
        }

        self.search_linear_x = float(self.get_parameter("search_linear_x").value)
        self.search_angular_z = float(self.get_parameter("search_angular_z").value)

        self.target_stable_required_s = float(
            self.get_parameter("target_stable_required_s").value
        )
        self.target_iou_stable_thresh = float(
            self.get_parameter("target_iou_stable_thresh").value
        )
        self.target_timeout_s = float(self.get_parameter("target_timeout_s").value)

        control_hz = float(self.get_parameter("control_hz").value)
        self.control_dt = 1.0 / max(1e-6, control_hz)

        # State
        self.last_dets_raw: Optional[Any] = None
        self.last_detection_msg_time: float = 0.0

        self._stable_det: Optional[Det] = None
        self._stable_since: Optional[float] = None
        self._target_visible: bool = False

        self._last_motion_state: str = "INIT"

        # ROS I/O
        self.sub_det = self.create_subscription(String, self.det_topic, self.on_dets, 10)
        self.pub_cmd = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.timer = self.create_timer(self.control_dt, self.on_timer)

        self.get_logger().info("Motion controller started")
        self.get_logger().info(f"  det_topic        : {self.det_topic}")
        self.get_logger().info(f"  cmd_vel_topic    : {self.cmd_vel_topic}")
        self.get_logger().info(
            f"  search motion    : linear.x={self.search_linear_x:.2f}, angular.z={self.search_angular_z:.2f}"
        )
        self.get_logger().info(
            f"  stable required  : {self.target_stable_required_s:.2f}s"
        )

    def on_dets(self, msg: String) -> None:
        try:
            self.last_dets_raw = json.loads(msg.data)
            self.last_detection_msg_time = time.time()
        except Exception as e:
            self.get_logger().warn(f"Detections JSON parse failed: {e}")

    def parse_person_dets(self, dets_raw: Any, frame_shape_hw: Tuple[int, int]) -> List[Det]:
        """
        Mirrors your orchestrator parsing logic so both nodes behave consistently.
        """
        if dets_raw is None:
            return []

        if isinstance(dets_raw, dict):
            det_list = dets_raw.get("detections") or dets_raw.get("dets") or dets_raw.get("results") or []
        elif isinstance(dets_raw, list):
            det_list = dets_raw
        else:
            det_list = []

        h, w = frame_shape_hw
        out: List[Det] = []

        for d in det_list:
            if not isinstance(d, dict):
                continue

            conf = float(d.get("conf", d.get("confidence", d.get("score", 0.0))))
            if conf < self.min_conf:
                continue

            cls = d.get("class", d.get("cls", d.get("class_id", "person")))

            # Filter to person/face only
            if isinstance(cls, str):
                if cls.strip().lower() not in self.person_class_names:
                    continue
            else:
                try:
                    if int(cls) not in self.person_class_ids:
                        continue
                except Exception:
                    continue

            box = d.get("box") or d.get("bbox") or d.get("xyxy")
            xywh = d.get("xywh")

            x1 = y1 = x2 = y2 = None
            if isinstance(box, list) and len(box) == 4:
                x1, y1, x2, y2 = box
            elif isinstance(xywh, list) and len(xywh) == 4:
                x, y, bw, bh = xywh
                x1, y1, x2, y2 = x, y, x + bw, y + bh

            if x1 is None:
                continue

            x1f, y1f, x2f, y2f = float(x1), float(y1), float(x2), float(y2)

            # normalized coords
            if 0.0 <= x1f <= 1.0 and 0.0 <= x2f <= 1.0 and 0.0 <= y1f <= 1.0 and 0.0 <= y2f <= 1.0:
                x1i, x2i = int(x1f * w), int(x2f * w)
                y1i, y2i = int(y1f * h), int(y2f * h)
            else:
                x1i, y1i, x2i, y2i = int(x1f), int(y1f), int(x2f), int(y2f)

            x1i = clamp(x1i, 0, w - 1)
            y1i = clamp(y1i, 0, h - 1)
            x2i = clamp(x2i, 0, w - 1)
            y2i = clamp(y2i, 0, h - 1)

            if x2i <= x1i or y2i <= y1i:
                continue

            out.append(Det(x1i, y1i, x2i, y2i, conf, cls))

        out.sort(key=lambda dd: dd.conf, reverse=True)
        if self.choose_best_only and out:
            return [out[0]]
        return out

    def _publish_stop(self) -> None:
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.pub_cmd.publish(msg)

    def _publish_search_spin(self) -> None:
        msg = Twist()
        msg.linear.x = self.search_linear_x
        msg.angular.z = self.search_angular_z
        self.pub_cmd.publish(msg)

    def _set_motion_state(self, state: str) -> None:
        if state != self._last_motion_state:
            self._last_motion_state = state
            self.get_logger().info(f"motion_state={state}")

    def _update_stable_target(self, best: Det) -> bool:
        now = time.time()

        if self._stable_det is None:
            self._stable_det = best
            self._stable_since = now
            return False

        if iou(self._stable_det, best) < self.target_iou_stable_thresh:
            self._stable_det = best
            self._stable_since = now
            return False

        return (
            self._stable_since is not None
            and (now - self._stable_since) >= self.target_stable_required_s
        )

    def on_timer(self) -> None:
        # We do not know image size here, but your YOLO boxes are effectively pixel-space
        # in the current setup. Using a large fallback frame avoids clamping issues.
        dets = self.parse_person_dets(self.last_dets_raw, (1080, 1920))

        now = time.time()

        if dets and (now - self.last_detection_msg_time) <= self.target_timeout_s:
            best = dets[0]
            stable_ok = self._update_stable_target(best)
            self._target_visible = True

            if stable_ok:
                self._set_motion_state("TARGET_STABLE_STOP")
                self._publish_stop()
            else:
                self._set_motion_state("TARGET_SEEN_STOP")
                self._publish_stop()

        else:
            self._target_visible = False
            self._stable_det = None
            self._stable_since = None
            self._set_motion_state("SEARCH_SPIN")
            self._publish_search_spin()

    def destroy_node(self):
        try:
            self._publish_stop()
        except Exception:
            pass
        return super().destroy_node()


def main():
    rclpy.init()
    node = MotionController()
    try:
        rclpy.spin(node)
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
