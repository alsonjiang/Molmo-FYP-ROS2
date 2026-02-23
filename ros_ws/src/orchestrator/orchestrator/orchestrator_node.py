#!/usr/bin/env python3
"""
ROS2 Orchestrator Node (YOLO overlay + crop + moondream multipart POST)

moondream service expects:
  POST /caption (multipart/form-data)
    - image: UploadFile (required)
    - prompt: str (optional)
    - max_new_tokens: int (optional)
    - temperature: float (optional)

YOLO adapter publishes:
  /yolo/detections_json (std_msgs/String) JSON payload
"""

import time
import json
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np
import requests
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
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


def bgr_to_jpeg_bytes(img_bgr: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("Failed to JPEG-encode image")
    return buf.tobytes()


def resize_max_width(img_bgr: np.ndarray, max_w: int) -> np.ndarray:
    if max_w <= 0:
        return img_bgr
    h, w = img_bgr.shape[:2]
    if w <= max_w:
        return img_bgr
    new_h = int(h * (max_w / float(w)))
    new_h = max(1, new_h)
    return cv2.resize(img_bgr, (max_w, new_h), interpolation=cv2.INTER_AREA)


class Orchestrator(Node):
    def __init__(self):
        super().__init__("orchestrator")

        # ---------------- Params ----------------
        self.declare_parameter("image_topic", "/image")
        self.declare_parameter("det_topic", "/yolo/detections_json")

        self.declare_parameter("min_conf", 0.25)
        self.declare_parameter("choose_best_only", True)
        self.declare_parameter("crop_pad_px", 0)
        self.declare_parameter("max_crop_width", 640)

        # moondream
        self.declare_parameter("moondream_enabled", True)
        self.declare_parameter("moondream_url", "http://127.0.0.1:8001/caption")
        self.declare_parameter("moondream_prompt", "Describe the person briefly.")
        self.declare_parameter("moondream_hz", 1.0)
        self.declare_parameter("moondream_timeout_s", 12.0)
        self.declare_parameter("moondream_max_new_tokens", 80)
        self.declare_parameter("moondream_temperature", 0.2)

        # Output
        self.declare_parameter("publish_annotated", True)
        self.declare_parameter("show_window", True)
        self.declare_parameter("render_hz", 30.0)

        # ---------------- Read params ----------------
        self.image_topic = str(self.get_parameter("image_topic").value)
        self.det_topic = str(self.get_parameter("det_topic").value)

        self.min_conf = float(self.get_parameter("min_conf").value)
        self.choose_best_only = bool(self.get_parameter("choose_best_only").value)
        self.crop_pad_px = int(self.get_parameter("crop_pad_px").value)
        self.max_crop_width = int(self.get_parameter("max_crop_width").value)

        self.moondream_enabled = bool(self.get_parameter("moondream_enabled").value)
        self.moondream_url = str(self.get_parameter("moondream_url").value)
        self.moondream_prompt = str(self.get_parameter("moondream_prompt").value)
        self.moondream_hz = float(self.get_parameter("moondream_hz").value)
        self.moondream_timeout_s = float(self.get_parameter("moondream_timeout_s").value)
        self.moondream_max_new_tokens = int(self.get_parameter("moondream_max_new_tokens").value)
        self.moondream_temperature = float(self.get_parameter("moondream_temperature").value)

        self.publish_annotated = bool(self.get_parameter("publish_annotated").value)
        self.show_window = bool(self.get_parameter("show_window").value)

        render_hz = float(self.get_parameter("render_hz").value)
        self.render_dt = 1.0 / max(1e-6, render_hz)

        # ---------------- State ----------------
        self.bridge = CvBridge()
        self.last_frame: Optional[np.ndarray] = None
        self.last_dets_raw: Optional[Any] = None
        self.last_dets_time: float = 0.0

        self.last_moondream_call_t: float = 0.0
        self.last_moondream_state: str = "IDLE"
        self.last_moondream_text: str = ""
        self.last_moondream_ms: Optional[float] = None

        # For spam control
        self._prev_logged_state: Optional[str] = None
        self._prev_logged_text: str = ""

        # ---------------- ROS I/O ----------------
        self.sub_img = self.create_subscription(Image, self.image_topic, self.on_image, 10)
        self.sub_det = self.create_subscription(String, self.det_topic, self.on_dets, 10)

        self.pub_annot = None
        if self.publish_annotated:
            self.pub_annot = self.create_publisher(Image, "/image_annotated", 10)

        # VLM debug topics
        self.pub_vlm_caption = self.create_publisher(String, "/vlm/caption", 10)
        self.pub_vlm_latency = self.create_publisher(String, "/vlm/latency_ms", 10)

        self.timer = self.create_timer(self.render_dt, self.on_timer)

        self.get_logger().info("Orchestrator started")
        self.get_logger().info(f"  image_topic: {self.image_topic}")
        self.get_logger().info(f"  det_topic  : {self.det_topic}")
        self.get_logger().info(f"  moondream_url  : {self.moondream_url} (enabled={self.moondream_enabled})")

    # ---------------- Callbacks ----------------
    def on_image(self, msg: Image) -> None:
        try:
            self.last_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge convert failed: {e}")

    def on_dets(self, msg: String) -> None:
        try:
            self.last_dets_raw = json.loads(msg.data)
            self.last_dets_time = time.time()
        except Exception as e:
            self.get_logger().warn(f"Detections JSON parse failed: {e}")

    # ---------------- YOLO JSON parsing ----------------
    def parse_person_dets(self, dets_raw: Any, frame_shape: Tuple[int, int]) -> List[Det]:
        if dets_raw is None:
            return []

        if isinstance(dets_raw, dict):
            det_list = dets_raw.get("detections") or dets_raw.get("dets") or dets_raw.get("results") or []
        elif isinstance(dets_raw, list):
            det_list = dets_raw
        else:
            det_list = []

        h, w = frame_shape
        out: List[Det] = []

        for d in det_list:
            if not isinstance(d, dict):
                continue

            conf = float(d.get("conf", d.get("confidence", d.get("score", 0.0))))
            if conf < self.min_conf:
                continue

            cls = d.get("class", d.get("cls", d.get("class_id", "person")))
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

    # ---------------- Overlay ----------------
    def draw_overlay(self, frame: np.ndarray, dets: List[Det]) -> None:
        for d in dets:
            cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), (0, 255, 0), 2)
            label = f"person {d.conf:.2f}"
            (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            y_text = max(0, d.y1 - th - base - 6)
            cv2.rectangle(frame, (d.x1, y_text), (d.x1 + tw + 10, y_text + th + base + 10), (0, 255, 0), -1)
            cv2.putText(frame, label, (d.x1 + 5, y_text + th + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

        status = f"moondream: {self.last_moondream_state}"
        if self.last_moondream_ms is not None:
            status += f" {self.last_moondream_ms:.0f}ms"
        cv2.putText(frame, status, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        if self.last_moondream_text:
            msg = self.last_moondream_text.strip().replace("\n", " ")
            if len(msg) > 90:
                msg = msg[:87] + "..."
            cv2.putText(frame, msg, (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # ---------------- Crop ----------------
    def crop_from_det(self, frame: np.ndarray, det: Det) -> np.ndarray:
        h, w = frame.shape[:2]
        pad = max(0, self.crop_pad_px)
        x1 = clamp(det.x1 - pad, 0, w - 1)
        y1 = clamp(det.y1 - pad, 0, h - 1)
        x2 = clamp(det.x2 + pad, 0, w - 1)
        y2 = clamp(det.y2 + pad, 0, h - 1)
        if x2 <= x1 or y2 <= y1:
            return frame.copy()
        return frame[y1:y2, x1:x2].copy()

    # ---------------- moondream ----------------
    def should_call_moondream(self) -> bool:
        if not self.moondream_enabled:
            return False
        if self.moondream_hz <= 0:
            return False
        return (time.time() - self.last_moondream_call_t) >= (1.0 / self.moondream_hz)

    def _publish_vlm_debug(self) -> None:
        # Publish caption
        msg = String()
        msg.data = self.last_moondream_text or ""
        self.pub_vlm_caption.publish(msg)

        # Publish latency (as string ms)
        lat = String()
        lat.data = "" if self.last_moondream_ms is None else f"{self.last_moondream_ms:.1f}"
        self.pub_vlm_latency.publish(lat)

    def call_moondream_multipart(self, crop_bgr: np.ndarray) -> None:
        self.last_moondream_call_t = time.time()
        self.last_moondream_state = "PENDING"
        self.last_moondream_text = ""
        self.last_moondream_ms = None
        t0 = time.time()

        try:
            crop_bgr = resize_max_width(crop_bgr, self.max_crop_width)
            jpeg_bytes = bgr_to_jpeg_bytes(crop_bgr, quality=85)

            files = {"image": ("crop.jpg", jpeg_bytes, "image/jpeg")}
            data = {
                "prompt": self.moondream_prompt,
                "max_new_tokens": str(self.moondream_max_new_tokens),
                "temperature": str(self.moondream_temperature),
            }

            r = requests.post(
                self.moondream_url,
                files=files,
                data=data,
                timeout=self.moondream_timeout_s,
            )

            self.last_moondream_ms = (time.time() - t0) * 1000.0

            if r.status_code != 200:
                self.last_moondream_state = f"ERROR({r.status_code})"
                self.last_moondream_text = (r.text or "").strip()[:160]
                self._publish_vlm_debug()
                return

            j = r.json()
            self.last_moondream_text = str(j.get("text", ""))[:500]
            self.last_moondream_state = "OK"

            # Publish + terminal log for successful inference (only when changed)
            self._publish_vlm_debug()

            txt_one_line = self.last_moondream_text.strip().replace("\n", " ")
            if txt_one_line != self._prev_logged_text:
                self.get_logger().info(f"[VLM] {self.last_moondream_ms:.0f} ms | {txt_one_line}")
                self._prev_logged_text = txt_one_line

        except requests.Timeout:
            self.last_moondream_ms = (time.time() - t0) * 1000.0
            self.last_moondream_state = "TIMEOUT"
            self.last_moondream_text = ""
            self._publish_vlm_debug()
        except Exception as e:
            self.last_moondream_ms = (time.time() - t0) * 1000.0
            self.last_moondream_state = "ERROR"
            self.last_moondream_text = str(e)[:160]
            self._publish_vlm_debug()

    # ---------------- Main loop ----------------
    def on_timer(self) -> None:
        if self.last_frame is None:
            return

        frame = self.last_frame.copy()
        h, w = frame.shape[:2]
        dets = self.parse_person_dets(self.last_dets_raw, (h, w))

        if not dets:
            self.last_moondream_state = "SKIP(no_person)"
            self.last_moondream_text = ""
            self.last_moondream_ms = None
        else:
            if self.should_call_moondream():
                crop = self.crop_from_det(self.last_frame, dets[0])
                if crop.shape[0] < 32 or crop.shape[1] < 32:
                    self.last_moondream_state = "SKIP(tiny_crop)"
                    self.last_moondream_text = ""
                    self.last_moondream_ms = None
                else:
                    self.call_moondream_multipart(crop)

        # Log state changes only (avoid spam)
        if self.last_moondream_state != self._prev_logged_state:
            self.get_logger().info(f"[VLM STATE] {self.last_moondream_state}")
            self._prev_logged_state = self.last_moondream_state

        self.draw_overlay(frame, dets)

        if self.publish_annotated and self.pub_annot is not None:
            try:
                msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                self.pub_annot.publish(msg)
            except Exception as e:
                self.get_logger().error(f"publish /image_annotated failed: {e}")

        if self.show_window:
            cv2.imshow("Orchestrator (YOLO overlay + moondream)", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                rclpy.shutdown()


def main():
    rclpy.init()
    node = Orchestrator()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

