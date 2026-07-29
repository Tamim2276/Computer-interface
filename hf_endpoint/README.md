---
title: AirWrite OCR
emoji: ✍️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# AirWrite OCR API

This Docker Space exposes the TrOCR handwriting model through a FastAPI service.
Hugging Face builds and runs the container remotely.

## API

`POST /ocr` accepts a multipart JPEG or PNG image file named `file` and returns:

```json
{"text": "AIR PEN"}
```

`GET /health` returns the service status. The model is downloaded on first start.
