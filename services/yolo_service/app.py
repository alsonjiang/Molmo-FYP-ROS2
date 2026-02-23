import os
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO

# ───────────────────────── Config ─────────────────────────

DEFAULT_WEIGHTS = "/home/jetson/molmo_fyp_ros/weights/yolo26n.pt"
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", DEFAULT_WEIGHTS)

# Default to COCO "person" only. Set YOLO_CLASSES="" to allow all classes.
DEFAULT_CLASSES = os.getenv("YOLO_CLASSES", "0")  # person=0

# ───────────────────────── App ────────────────────────────

app = FastAPI(title="YOLO Service", version="1.0")
model: Optional[YOLO] = None


class Det(BaseModel):
    cls: int
    conf: float
    xyxy: List[float]  # [x1, y1, x2, y2]


class DetectResponse(BaseModel):
    ms: float
    detections: List[Det]


@app.on_event("startup")
def _load():
    global model

    weights_path = Path(YOLO_WEIGHTS).expanduser()
    if not weights_path.exists():
        raise RuntimeError(
            f"YOLO weights not found: '{weights_path}'. "
            f"Set env YOLO_WEIGHTS to an existing .pt file."
        )

    try:
        model = YOLO(str(weights_path))
    except Exception as e:
        raise RuntimeError(f"Failed to load YOLO weights '{weights_path}': {e}")


@app.get("/health")
def health():
    return {
        "ok": True,
        "weights": str(Path(YOLO_WEIGHTS).expanduser()),
        "classes_default": DEFAULT_CLASSES,
    }


@app.post("/detect", response_model=DetectResponse)
async def detect(
    image: UploadFile = File(...),
    conf: float = 0.25,
    iou: float = 0.7,
    classes: Optional[str] = None,  # comma-separated ints
):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    data = await image.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    cls_str = (DEFAULT_CLASSES if classes is None else classes).strip()
    cls_list = None
    if cls_str != "":
        try:
            cls_list = [int(x.strip()) for x in cls_str.split(",")]
        except Exception:
            raise HTTPException(status_code=400, detail="classes must be comma-separated ints, e.g. '0,1,2'")

    t0 = time.time()
    try:
        res = model.predict(img, conf=conf, iou=iou, classes=cls_list, verbose=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"YOLO inference failed: {e}")
    ms = (time.time() - t0) * 1000.0

    out: List[Det] = []
    r0 = res[0]
    if r0.boxes is not None and len(r0.boxes) > 0:
        boxes = r0.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        for b, c, k in zip(xyxy, confs, clss):
            out.append(
                Det(
                    cls=int(k),
                    conf=float(c),
                    xyxy=[float(x) for x in b.tolist()],
                )
            )

    return DetectResponse(ms=ms, detections=out)

