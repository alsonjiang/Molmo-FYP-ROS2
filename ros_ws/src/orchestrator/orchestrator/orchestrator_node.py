#!/usr/bin/env python3
import time
import json
import threading
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union, Dict

import cv2
import numpy as np
import requests
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

WIN_NAME = "Orchestrator (YOLO overlay + VLM)"


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
    new_h = max(1, int(h * (max_w / float(w))))
    return cv2.resize(img_bgr, (max_w, new_h), interpolation=cv2.INTER_AREA)


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


def det_key(d: Det, frame_w: int, frame_h: int) -> Tuple:
    cx = (d.x1 + d.x2) // 2
    cy = (d.y1 + d.y2) // 2
    qx = int(cx / max(1, frame_w) * 10)
    qy = int(cy / max(1, frame_h) * 10)
    return (str(d.cls), qx, qy)


class Orchestrator(Node):
    def __init__(self):
        super().__init__("orchestrator_node")

        # ---------------- Params ----------------
        self.declare_parameter("image_topic", "/image")
        self.declare_parameter("det_topic", "/yolo/detections_json")

        self.declare_parameter("min_conf", 0.25)
        self.declare_parameter("choose_best_only", True)
        self.declare_parameter("crop_pad_px", 0)
        self.declare_parameter("max_crop_width", 640)
        self.declare_parameter("jpeg_quality", 85)

        # Preferred params (moondream)
        self.declare_parameter("moondream_enabled", True)
        self.declare_parameter("moondream_url", "http://127.0.0.1:8001/caption")
        self.declare_parameter("moondream_prompt", "Describe the person briefly.")
        self.declare_parameter("moondream_hz", 1.0)
        self.declare_parameter("moondream_timeout_s", 12.0)
        self.declare_parameter("moondream_max_new_tokens", 80)
        self.declare_parameter("moondream_temperature", 0.2)

        # Backward-compat params (molmo_*)
        self.declare_parameter("molmo_enabled", True)
        self.declare_parameter("molmo_url", "http://127.0.0.1:8001/caption")
        self.declare_parameter("molmo_prompt", "Describe the person briefly.")
        self.declare_parameter("molmo_hz", 1.0)
        self.declare_parameter("molmo_timeout_s", 12.0)
        self.declare_parameter("molmo_max_new_tokens", 80)
        self.declare_parameter("molmo_temperature", 0.2)

        # Output
        self.declare_parameter("publish_annotated", True)
        self.declare_parameter("show_window", True)
        self.declare_parameter("render_hz", 30.0)

        # overload guards
        self.declare_parameter("vlm_target_cooldown_s", 8.0)
        self.declare_parameter("vlm_stable_required_s", 0.5)
        self.declare_parameter("vlm_iou_stable_thresh", 0.6)

        # terminal caption printing
        self.declare_parameter("caption_to_terminal", True)
        self.declare_parameter("caption_print_min_interval_s", 0.8)

        # overlay behavior
        self.declare_parameter("overlay_show_pending", True)  # show state= PENDING etc.

        # ---------------- Read params ----------------
        self.image_topic = str(self.get_parameter("image_topic").value)
        self.det_topic = str(self.get_parameter("det_topic").value)

        self.min_conf = float(self.get_parameter("min_conf").value)
        self.choose_best_only = bool(self.get_parameter("choose_best_only").value)
        self.crop_pad_px = int(self.get_parameter("crop_pad_px").value)
        self.max_crop_width = int(self.get_parameter("max_crop_width").value)
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)

        self.publish_annotated = bool(self.get_parameter("publish_annotated").value)
        self.show_window = bool(self.get_parameter("show_window").value)

        render_hz = float(self.get_parameter("render_hz").value)
        self.render_dt = 1.0 / max(1e-6, render_hz)

        self.vlm_target_cooldown_s = float(self.get_parameter("vlm_target_cooldown_s").value)
        self.vlm_stable_required_s = float(self.get_parameter("vlm_stable_required_s").value)
        self.vlm_iou_stable_thresh = float(self.get_parameter("vlm_iou_stable_thresh").value)

        self.caption_to_terminal = bool(self.get_parameter("caption_to_terminal").value)
        self.caption_print_min_interval_s = float(self.get_parameter("caption_print_min_interval_s").value)

        self.overlay_show_pending = bool(self.get_parameter("overlay_show_pending").value)

        # ---------------- VLM param aliasing ----------------
        def p(name: str):
            return self.get_parameter(name).value

        md_url = str(p("moondream_url"))
        mm_url = str(p("molmo_url"))
        md_enabled = bool(p("moondream_enabled"))
        mm_enabled = bool(p("molmo_enabled"))

        self.vlm_enabled = md_enabled or mm_enabled

        md_default = "http://127.0.0.1:8001/caption"
        use_molmo_url = (md_url == md_default and mm_url != md_default)
        self.vlm_url = mm_url if use_molmo_url else md_url

        def choose_str(md_name, md_def, mm_name):
            mdv = str(p(md_name))
            mmv = str(p(mm_name))
            return mmv if (mdv == md_def and mmv != md_def) else mdv

        def choose_float(md_name, md_def, mm_name):
            mdv = float(p(md_name))
            mmv = float(p(mm_name))
            return mmv if (abs(mdv - md_def) < 1e-9 and abs(mmv - md_def) > 1e-9) else mdv

        def choose_int(md_name, md_def, mm_name):
            mdv = int(p(md_name))
            mmv = int(p(mm_name))
            return mmv if (mdv == md_def and mmv != md_def) else mdv

        self.vlm_prompt = choose_str("moondream_prompt", "Describe the person briefly.", "molmo_prompt")
        self.vlm_hz = choose_float("moondream_hz", 1.0, "molmo_hz")
        self.vlm_timeout_s = choose_float("moondream_timeout_s", 12.0, "molmo_timeout_s")
        self.vlm_max_new_tokens = choose_int("moondream_max_new_tokens", 80, "molmo_max_new_tokens")
        self.vlm_temperature = choose_float("moondream_temperature", 0.2, "molmo_temperature")

        # ---------------- State ----------------
        self.bridge = CvBridge()
        self.last_frame: Optional[np.ndarray] = None
        self.last_dets_raw: Optional[Any] = None

        self._vlm_lock = threading.Lock()
        self.last_call_t: float = 0.0
        self.last_state: str = "IDLE"
        self.last_text: str = ""        # debug topic
        self.last_text_ok: str = ""     # overlay text (sticky)
        self.last_ms: Optional[float] = None

        self._shutdown_requested = False

        # Worker thread / queue (latest crop wins)
        self._crop_event = threading.Event()
        self._latest_crop: Optional[np.ndarray] = None
        self._worker_stop = threading.Event()
        self._inflight = False

        # stable-target gating + cooldown map
        self._stable_det: Optional[Det] = None
        self._stable_since: Optional[float] = None
        self._last_caption_by_key: Dict[Tuple, float] = {}

        # terminal print throttling
        self._last_printed_text: str = ""
        self._last_print_t: float = 0.0

        # ---------------- Window Setup (ONCE) ----------------
        if self.show_window:
            cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WIN_NAME, 960, 540)

        # ---------------- ROS I/O ----------------
        self.sub_img = self.create_subscription(Image, self.image_topic, self.on_image, 10)
        self.sub_det = self.create_subscription(String, self.det_topic, self.on_dets, 10)

        self.pub_annot = None
        if self.publish_annotated:
            self.pub_annot = self.create_publisher(Image, "/image_annotated", 10)

        # Debug topics
        self.pub_vlm_caption = self.create_publisher(String, "/vlm/caption", 10)
        self.pub_vlm_latency = self.create_publisher(String, "/vlm/latency_ms", 10)
        self.pub_vlm_state = self.create_publisher(String, "/vlm/state", 10)

        self.timer = self.create_timer(self.render_dt, self.on_timer)

        self._worker = threading.Thread(target=self._vlm_worker, daemon=True)
        self._worker.start()

        rclpy.get_default_context().on_shutdown(self._on_ros_shutdown)

        self.get_logger().info("Orchestrator started")
        self.get_logger().info(f"  image_topic: {self.image_topic}")
        self.get_logger().info(f"  det_topic  : {self.det_topic}")
        self.get_logger().info(f"  vlm_url    : {self.vlm_url} (enabled={self.vlm_enabled})")

    # ---------------- Shutdown ----------------
    def _on_ros_shutdown(self) -> None:
        self._worker_stop.set()
        self._crop_event.set()
        try:
            if self.show_window:
                cv2.destroyAllWindows()
                cv2.waitKey(1)
        except Exception:
            pass

    def destroy_node(self):
        self._worker_stop.set()
        self._crop_event.set()
        try:
            if hasattr(self, "_worker") and self._worker.is_alive():
                self._worker.join(timeout=2.0)
        except Exception:
            pass
        return super().destroy_node()

    # ---------------- Callbacks ----------------
    def on_image(self, msg: Image) -> None:
        try:
            self.last_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge convert failed: {e}")

    def on_dets(self, msg: String) -> None:
        try:
            self.last_dets_raw = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"Detections JSON parse failed: {e}")

    # ---------------- YOLO JSON parsing ----------------
    def parse_person_dets(self, dets_raw: Any, frame_shape_hw: Tuple[int, int]) -> List[Det]:
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

    # ---------------- Overlay helpers ----------------
    def _wrap_lines(self, text: str, max_chars: int = 80, max_lines: int = 4) -> List[str]:
        msg = (text or "").strip().replace("\n", " ")
        if not msg:
            return []
        msg = msg[: max_chars * max_lines]
        lines = [msg[i:i + max_chars] for i in range(0, len(msg), max_chars)]
        return lines[:max_lines]

    def _draw_text_panel(self, frame: np.ndarray, lines: List[str], x: int, y: int) -> None:
        if not lines:
            return
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thickness = 2
        line_h = 20

        # Measure panel size
        widths = []
        for ln in lines:
            (tw, th), base = cv2.getTextSize(ln, font, scale, thickness)
            widths.append(tw)
        panel_w = max(widths) + 16
        panel_h = len(lines) * line_h + 12

        # Background rectangle (solid) to prevent “ghosting/flicker” look
        x2 = min(frame.shape[1] - 1, x + panel_w)
        y2 = min(frame.shape[0] - 1, y + panel_h)
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 0), -1)

        # Draw lines
        yy = y + 22
        for ln in lines:
            cv2.putText(frame, ln, (x + 8, yy), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
            yy += line_h

    def draw_overlay(self, frame: np.ndarray, dets: List[Det]) -> None:
        for d in dets:
            cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), (0, 255, 0), 2)
            label = f"person {d.conf:.2f}"
            (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            y_text = max(0, d.y1 - th - base - 6)
            cv2.rectangle(frame, (d.x1, y_text), (d.x1 + tw + 10, y_text + th + base + 10), (0, 255, 0), -1)
            cv2.putText(frame, label, (d.x1 + 5, y_text + th + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

        with self._vlm_lock:
            state = self.last_state
            ms = self.last_ms
            caption = self.last_text_ok  # sticky caption only (stable)

        # Build panel lines (status always shown; caption shown if we have one)
        status = f"vlm: {state}"
        if ms is not None:
            status += f" {ms:.0f}ms"

        lines = [status]
        if caption:
            lines += self._wrap_lines(caption, max_chars=80, max_lines=4)

        # Optionally hide status during pending (caption stays)
        if not self.overlay_show_pending and state.startswith("PENDING"):
            lines = self._wrap_lines(caption, max_chars=80, max_lines=4)

        self._draw_text_panel(frame, lines, x=8, y=8)

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

    # ---------------- VLM scheduling ----------------
    def should_call_vlm_global(self) -> bool:
        if not self.vlm_enabled or self.vlm_hz <= 0:
            return False
        now = time.time()
        with self._vlm_lock:
            last_t = self.last_call_t
        return (now - last_t) >= (1.0 / self.vlm_hz)

    def _publish_vlm_debug(self, state: str, text: str, ms: Optional[float]) -> None:
        self.pub_vlm_state.publish(String(data=state or ""))
        self.pub_vlm_caption.publish(String(data=text or ""))
        self.pub_vlm_latency.publish(String(data="" if ms is None else f"{ms:.1f}"))

    def enqueue_crop(self, crop_bgr: np.ndarray) -> None:
        self._latest_crop = crop_bgr
        self._crop_event.set()

    def _update_stable_target(self, best: Det) -> Tuple[bool, Tuple]:
        now = time.time()

        if self._stable_det is None:
            self._stable_det = best
            self._stable_since = now
        else:
            if iou(self._stable_det, best) < self.vlm_iou_stable_thresh:
                self._stable_det = best
                self._stable_since = now

        if self.last_frame is None or self._stable_det is None:
            return (False, ("none", 0, 0))

        h, w = self.last_frame.shape[:2]
        key = det_key(self._stable_det, w, h)

        stable_ok = (self._stable_since is not None) and ((now - self._stable_since) >= self.vlm_stable_required_s)
        return stable_ok, key

    def _target_off_cooldown(self, key: Tuple) -> bool:
        now = time.time()
        last = self._last_caption_by_key.get(key, 0.0)
        return (now - last) >= self.vlm_target_cooldown_s

    def _mark_captioned(self, key: Tuple) -> None:
        self._last_caption_by_key[key] = time.time()
        if len(self._last_caption_by_key) > 200:
            cutoff = time.time() - max(30.0, self.vlm_target_cooldown_s * 3.0)
            self._last_caption_by_key = {k: t for k, t in self._last_caption_by_key.items() if t >= cutoff}

    def _maybe_print_caption(self, text: str) -> None:
        if not self.caption_to_terminal:
            return
        t = (text or "").strip()
        if not t:
            return
        now = time.time()
        if t == self._last_printed_text:
            return
        if (now - self._last_print_t) < max(0.0, self.caption_print_min_interval_s):
            return
        self._last_printed_text = t
        self._last_print_t = now
        self.get_logger().info(f"[VLM] {t}")

    def _vlm_worker(self) -> None:
        session = requests.Session()

        while not self._worker_stop.is_set():
            self._crop_event.wait(timeout=0.2)
            if self._worker_stop.is_set():
                break
            if not self._crop_event.is_set():
                continue
            self._crop_event.clear()

            crop = self._latest_crop
            self._latest_crop = None
            if crop is None:
                continue

            if self._inflight:
                continue
            self._inflight = True

            # IMPORTANT: do NOT blank last_text_ok (sticky caption)
            with self._vlm_lock:
                self.last_call_t = time.time()
                self.last_state = "PENDING"
                self.last_text = ""   # debug can be blank
                self.last_ms = None
            self._publish_vlm_debug("PENDING", "", None)

            t0 = time.time()
            try:
                crop = resize_max_width(crop, self.max_crop_width)
                jpeg_bytes = bgr_to_jpeg_bytes(crop, quality=self.jpeg_quality)

                files = {"image": ("crop.jpg", jpeg_bytes, "image/jpeg")}
                data = {
                    "prompt": self.vlm_prompt,
                    "max_new_tokens": str(self.vlm_max_new_tokens),
                    "temperature": str(self.vlm_temperature),
                }

                r = session.post(self.vlm_url, files=files, data=data, timeout=self.vlm_timeout_s)
                ms = (time.time() - t0) * 1000.0

                if r.status_code != 200:
                    text = (r.text or "").strip()[:600]
                    state = f"HTTP_{r.status_code}"
                    with self._vlm_lock:
                        self.last_state = state
                        self.last_text = text
                        self.last_ms = ms
                    self._publish_vlm_debug(state, text, ms)
                    continue

                try:
                    j = r.json()
                except Exception as e:
                    text = f"JSON_PARSE_ERROR: {e} | body={(r.text or '')[:200]}"
                    with self._vlm_lock:
                        self.last_state = "JSON_ERROR"
                        self.last_text = text[:600]
                        self.last_ms = ms
                    self._publish_vlm_debug("JSON_ERROR", text[:600], ms)
                    continue

                text = str(j.get("text", j.get("caption", "")))[:600].strip()

                with self._vlm_lock:
                    self.last_state = "OK"
                    self.last_text = text
                    # CRITICAL: only update sticky caption if non-empty
                    if text:
                        self.last_text_ok = text
                    self.last_ms = ms
                self._publish_vlm_debug("OK", text, ms)

                if text:
                    self._maybe_print_caption(text)

            except requests.Timeout:
                ms = (time.time() - t0) * 1000.0
                with self._vlm_lock:
                    self.last_state = "TIMEOUT"
                    self.last_text = ""
                    self.last_ms = ms
                self._publish_vlm_debug("TIMEOUT", "", ms)

            except Exception as e:
                ms = (time.time() - t0) * 1000.0
                text = str(e)[:600]
                with self._vlm_lock:
                    self.last_state = "ERROR"
                    self.last_text = text
                    self.last_ms = ms
                self._publish_vlm_debug("ERROR", text, ms)

            finally:
                self._inflight = False

    # ---------------- Main loop ----------------
    def on_timer(self) -> None:
        if self._shutdown_requested or self.last_frame is None:
            return

        frame = self.last_frame.copy()
        h, w = frame.shape[:2]
        dets = self.parse_person_dets(self.last_dets_raw, (h, w))

        if dets:
            best = dets[0]
            stable_ok, key = self._update_stable_target(best)

            can_call = (
                stable_ok and
                self.should_call_vlm_global() and
                self._target_off_cooldown(key) and
                (not self._inflight)
            )

            if can_call:
                crop = self.crop_from_det(self.last_frame, best)
                if crop.shape[0] >= 32 and crop.shape[1] >= 32:
                    self._mark_captioned(key)
                    self.enqueue_crop(crop)
                else:
                    with self._vlm_lock:
                        self.last_state = "SKIP_TINY"
                        self.last_text = ""
                        self.last_ms = None
                    self._publish_vlm_debug("SKIP_TINY", "", None)

        else:
            self._stable_det = None
            self._stable_since = None
            with self._vlm_lock:
                if not self.last_state.startswith("PENDING"):
                    self.last_state = "SKIP_NO_PERSON"
                    self.last_text = ""
                    self.last_ms = None

        self.draw_overlay(frame, dets)

        if self.publish_annotated and self.pub_annot is not None:
            try:
                msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                self.pub_annot.publish(msg)
            except Exception as e:
                self.get_logger().error(f"publish /image_annotated failed: {e}")

        if self.show_window:
            cv2.imshow(WIN_NAME, frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                self._shutdown_requested = True
                try:
                    rclpy.shutdown()
                except Exception:
                    pass


def main():
    rclpy.init()
    node = Orchestrator()
    try:
        rclpy.spin(node)
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
