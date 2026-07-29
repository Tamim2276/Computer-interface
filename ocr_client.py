"""Client for the AirWrite Hugging Face OCR endpoint."""

import os

import cv2
import requests


class OCRServiceError(RuntimeError):
    """The remote OCR service could not process a handwriting image."""


def recognize_image(image):
    """Send an OpenCV BGR image to the deployed /ocr endpoint."""
    endpoint_url = os.environ.get("HF_ENDPOINT_URL", "").rstrip("/")
    token = os.environ.get("HF_TOKEN", "")
    if not endpoint_url or not token:
        raise OCRServiceError(
            "Set HF_ENDPOINT_URL and HF_TOKEN before using OCR_BACKEND=huggingface."
        )

    encoded, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not encoded:
        raise OCRServiceError("Could not encode the handwriting image as JPEG.")

    try:
        response = requests.post(
            f"{endpoint_url}/ocr",
            headers={"Authorization": f"Bearer {token}"},
            files={"file": ("canvas.jpg", jpeg.tobytes(), "image/jpeg")},
            timeout=(5, 45),
        )
        response.raise_for_status()
        text = response.json().get("text", "")
    except (requests.RequestException, ValueError) as error:
        raise OCRServiceError(f"Hugging Face OCR request failed: {error}") from error

    return str(text).strip().upper()
