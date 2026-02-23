import os, io, base64, time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig


# ───────────────────────── Paths / Env ─────────────────────────

DEFAULT_MODEL_DIR = "/home/jetson/molmo_fyp_ros/models/MolmoE-1B-0924-NF4"
MODEL_DIR = Path(os.getenv("MOLMO_MODEL_DIR", DEFAULT_MODEL_DIR)).expanduser().resolve()
if not MODEL_DIR.exists():
    raise FileNotFoundError(f"Model folder not found: {MODEL_DIR}")

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# ───────────────────────── Settings ───────────────────────────

MAX_NEW_TOKENS = int(os.getenv("MOLMO_MAX_NEW_TOKENS", "64"))
MAX_SIDE = int(os.getenv("MOLMO_MAX_SIDE", "640"))          # Jetson-friendly default
FORCE_FP32 = os.getenv("MOLMO_FORCE_FP32", "0") == "1"
FORCE_CPU = os.getenv("MOLMO_FORCE_CPU", "0") == "1"

# optional: explicitly choose device ("cuda" / "cpu")
REQ_DEVICE = os.getenv("MOLMO_DEVICE", "").strip().lower()

torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


# ───────────────────────── Load processor/model ─────────────────────────────

processor = AutoProcessor.from_pretrained(
    MODEL_DIR.as_posix(),
    trust_remote_code=True,
    local_files_only=True,
)

cuda_ok = torch.cuda.is_available() and not FORCE_CPU
USE_CUDA = cuda_ok and (REQ_DEVICE != "cpu")
dtype = torch.float16 if (USE_CUDA and not FORCE_FP32) else torch.float32
device_map = "auto" if USE_CUDA else {"": "cpu"}

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR.as_posix(),
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype=dtype,
    device_map=device_map,
    offload_folder=None,
)
model.eval()

try:
    EMBED_DEVICE = model.get_input_embeddings().weight.device
except Exception:
    EMBED_DEVICE = next(model.parameters()).device
MODEL_DTYPE = next(model.parameters()).dtype

tok = getattr(processor, "tokenizer", None)
eos_id = getattr(model.config, "eos_token_id", None) or (getattr(tok, "eos_token_id", None) if tok else None)
pad_id = (getattr(tok, "pad_token_id", None) if tok else None) or eos_id

GENCFG = GenerationConfig(
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=False,
    num_beams=1,
    eos_token_id=eos_id,
    pad_token_id=pad_id,
    use_cache=True,
)
if tok is not None:
    setattr(GENCFG, "stop_strings", "<|endoftext|>")


# ───────────────────────── FastAPI ───────────────────────────

app = FastAPI(title="Molmo Service (Jetson)", version="1.0")


class CaptionIn(BaseModel):
    image_b64: str
    prompt: Optional[str] = "Describe the image."


def _b64_to_pil(b64: str) -> Image.Image:
    try:
        raw = base64.b64decode(b64)
        if not raw:
            raise ValueError("empty payload")
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid image: {e}")


def _resize_if_needed(img: Image.Image) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    if m <= MAX_SIDE:
        return img
    scale = MAX_SIDE / float(m)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return img.resize((nw, nh), Image.Resampling.LANCZOS)


def _build_batch(img: Image.Image, prompt: str) -> Dict[str, Any]:
    """
    Molmo-specific batching: processor.process + explicit batch dims.
    """
    batch = processor.process(images=[img], text=prompt)
    batch = {k: v for k, v in batch.items() if v is not None}

    # Explicit batch dim for these keys (Molmo expects them like this)
    if "images" in batch and isinstance(batch["images"], torch.Tensor):
        batch["images"] = torch.unsqueeze(batch["images"], 0)
    if "image_input_idx" in batch and isinstance(batch["image_input_idx"], torch.Tensor):
        batch["image_input_idx"] = torch.unsqueeze(batch["image_input_idx"], 0)
    if "image_masks" in batch and isinstance(batch["image_masks"], torch.Tensor):
        batch["image_masks"] = torch.unsqueeze(batch["image_masks"], 0)

    # Move tensors to model device
    for k, v in list(batch.items()):
        if isinstance(v, torch.Tensor):
            if v.dim() == 1:
                v = torch.unsqueeze(v, 0)
            batch[k] = v.to(EMBED_DEVICE, non_blocking=True)

    # Avoid broken / None PKVs
    if "past_key_values" in batch:
        pkv = batch["past_key_values"]
        if pkv is None or (isinstance(pkv, (list, tuple)) and any(x is None for x in pkv)):
            batch.pop("past_key_values", None)

    return batch


def _generate(batch: Dict[str, Any]) -> str:
    """
    Prefer Molmo's generate_from_batch if present.
    """
    use_cuda_autocast = (EMBED_DEVICE.type == "cuda") and (MODEL_DTYPE == torch.float16)

    with torch.inference_mode():
        if hasattr(model, "generate_from_batch"):
            with torch.autocast(device_type="cuda", enabled=use_cuda_autocast, dtype=torch.float16):
                out = model.generate_from_batch(batch, GENCFG, tokenizer=tok, use_cache=False)
        else:
            with torch.autocast(device_type="cuda", enabled=use_cuda_autocast, dtype=torch.float16):
                out = model.generate(**batch, generation_config=GENCFG, use_cache=False)

    # decode
    if isinstance(out, dict) and "sequences" in out and isinstance(out["sequences"], torch.Tensor):
        seq = out["sequences"]
        if tok and hasattr(tok, "batch_decode"):
            return tok.batch_decode(seq, skip_special_tokens=True)[0].strip()
        return str(seq.tolist())

    if hasattr(out, "sequences"):
        seq = out.sequences
        if tok and hasattr(tok, "batch_decode") and isinstance(seq, torch.Tensor):
            return tok.batch_decode(seq, skip_special_tokens=True)[0].strip()
        return str(seq)

    if isinstance(out, torch.Tensor):
        if tok and hasattr(tok, "batch_decode"):
            return tok.batch_decode(out, skip_special_tokens=True)[0].strip()
        return str(out.tolist())

    if isinstance(out, str):
        return out.strip()

    return str(out)


def _strip_to_answer(text: str) -> str:
    """
    Keep only assistant answer if model returns chat-like transcript.
    """
    if not text:
        return ""
    t = text.strip()
    lower = t.lower()
    if "assistant:" in lower:
        last = lower.rfind("assistant:")
        t = t[last + len("assistant:"):].strip()
    t = t.splitlines()[0].strip()
    return t


@app.get("/health")
def health():
    return {
        "ok": True,
        "model_dir": str(MODEL_DIR),
        "device": str(EMBED_DEVICE),
        "dtype": str(MODEL_DTYPE),
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_side": MAX_SIDE,
        "force_fp32": FORCE_FP32,
        "force_cpu": FORCE_CPU,
        "has_generate_from_batch": hasattr(model, "generate_from_batch"),
        "cuda_available": torch.cuda.is_available(),
    }


@app.post("/caption")
def caption(inp: CaptionIn):
    t0 = time.time()
    img = _b64_to_pil(inp.image_b64)
    img = _resize_if_needed(img)

    try:
        batch = _build_batch(img, inp.prompt or "Describe the image.")
        raw = _generate(batch)
        answer = _strip_to_answer(raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Molmo inference failed: {e}")

    ms = (time.time() - t0) * 1000.0
    return {"ms": ms, "caption": answer, "raw": raw}

