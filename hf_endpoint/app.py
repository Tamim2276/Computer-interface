"""FastAPI service for the AirWrite Hugging Face Space."""

import io
import os

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

MODEL_ID = os.environ.get("MODEL_ID", "microsoft/trocr-base-handwritten")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="AirWrite OCR API")
processor = TrOCRProcessor.from_pretrained(MODEL_ID)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID).to(DEVICE)
model.eval()


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "model": MODEL_ID}


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=415, detail="Upload a JPEG or PNG handwriting image.")

    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)
        with torch.inference_mode():
            generated_ids = model.generate(pixel_values, max_new_tokens=64)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip().upper()
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"OCR failed: {error}") from error

    return {"text": text}
