from base64 import b64encode
from io import BytesIO
import logging
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image, ImageOps, UnidentifiedImageError

from app.config import (
    CORS_ORIGINS,
    MAX_IMAGE_PIXELS,
    SAMPLE_DIR,
    SAMPLE_FILENAMES,
)
from app.schemas import DetectionResponse
from app.services.detector import annotate_image, detect_plates

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="License Plate Detection API",
    version="1.0.0",
    description="YOLOv5 license-plate detection with Tesseract OCR.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "license-plate-detection-api"}


@app.get("/api/samples")
def list_samples():
    return [
        {"filename": filename}
        for filename in SAMPLE_FILENAMES
        if (SAMPLE_DIR / filename).is_file()
    ]


@app.get("/api/samples/{filename}")
def get_sample(filename: str):
    if filename not in SAMPLE_FILENAMES:
        raise HTTPException(status_code=404, detail="Sample image not found.")

    sample_path = SAMPLE_DIR / filename
    if not sample_path.is_file():
        raise HTTPException(status_code=404, detail="Sample image not found.")

    try:
        with Image.open(sample_path) as image:
            image_format = (image.format or "").upper()
    except (OSError, UnidentifiedImageError):
        raise HTTPException(status_code=404, detail="Sample image could not be opened.")

    media_type = "image/png" if image_format == "PNG" else "image/jpeg"
    return Response(
        content=sample_path.read_bytes(),
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.post("/api/detect", response_model=DetectionResponse)
def detect_image(file: UploadFile = File(...)):
    filename = Path(file.filename or "uploaded-image").name
    allowed_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    if Path(filename).suffix.lower() not in allowed_extensions:
        raise HTTPException(
            status_code=415,
            detail="Unsupported image format. Use JPG, JPEG, PNG, or WebP.",
        )

    try:
        image_bytes = file.file.read()
        with Image.open(BytesIO(image_bytes)) as source:
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail="This image is too large. Please use an image with fewer than 20 million pixels.",
                )
            image = ImageOps.exif_transpose(source).convert("RGB")
    except HTTPException:
        raise
    except (OSError, UnidentifiedImageError, ValueError):
        raise HTTPException(status_code=400, detail="The selected image could not be opened.")

    try:
        detections = detect_plates(image)
        annotated = annotate_image(image, detections)
    except Exception as exc:
        logger.exception("Image inference failed for %s", filename)
        raise HTTPException(
            status_code=503,
            detail="Image processing failed. Please try again later.",
        ) from exc

    output = BytesIO()
    annotated.save(output, format="PNG")

    return DetectionResponse(
        filename=filename,
        count=len(detections),
        detections=detections,
        annotated_image=b64encode(output.getvalue()).decode("ascii"),
    )
