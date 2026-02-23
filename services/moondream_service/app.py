from fastapi import FastAPI, File, UploadFile, Form
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch
import io

from .tokenizer_patch import apply_moondream_tokenizer_patch

app = FastAPI()

MODEL_ID = "vikhyatk/moondream2"
REVISION = "2024-08-26"
CACHE_DIR = "/home/jetson/models"

DEFAULT_PROMPT = "Describe the image."

apply_moondream_tokenizer_patch()

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    revision=REVISION,
    trust_remote_code=True,
    cache_dir=CACHE_DIR,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    revision=REVISION,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    cache_dir=CACHE_DIR,
).to("cuda").eval()

print("Moondream loaded successfully.")

print([m for m in dir(model) if "encode" in m or "vision" in m or "image" in m])


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/caption")
async def caption(
    image: UploadFile = File(...),
    prompt: str = Form(DEFAULT_PROMPT),
):
    image_bytes = await image.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    prompt = (prompt or "").strip()

    with torch.inference_mode():
        # If caller doesn't provide a prompt, do a plain caption
        if prompt == "":
            answer = model.caption(img, tokenizer=tokenizer)
            prompt_used = DEFAULT_PROMPT
        else:
            # For custom prompts/questions, encode image -> embeddings first
            image_embeds = model.encode_image(img)
            answer = model.answer_question(image_embeds, prompt, tokenizer=tokenizer)
            prompt_used = prompt

    return {"text": answer, "prompt_used": prompt_used}
