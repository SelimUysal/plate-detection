from PIL import Image, ImageDraw, ImageFont
import numpy as np

from app.config import IMAGE_SIZE
from app.services.model import load_model
from app.services.ocr import read_plate_text


def detect_plates(image: Image.Image) -> list[dict]:
    """Run the existing YOLOv5 detection and Tesseract OCR pipeline."""
    model = load_model()
    image_array = np.array(image.convert("RGB"))
    results = model(image_array, size=IMAGE_SIZE)
    raw = results.xyxy[0].detach().cpu().numpy()

    detections = []
    for row in raw:
        x1, y1, x2, y2, confidence, _class_id = row.tolist()
        x1i = max(0, min(image.width, int(x1)))
        y1i = max(0, min(image.height, int(y1)))
        x2i = max(0, min(image.width, int(x2)))
        y2i = max(0, min(image.height, int(y2)))
        if x2i <= x1i or y2i <= y1i:
            continue

        crop = image_array[y1i:y2i, x1i:x2i]
        plate_text, ocr_confidence = read_plate_text(crop)
        detections.append(
            {
                "box": (x1i, y1i, x2i, y2i),
                "confidence": float(confidence),
                "text": plate_text,
                "ocr_confidence": ocr_confidence,
            }
        )
    return detections


def annotate_image(image: Image.Image, detections: list[dict]) -> Image.Image:
    output = image.copy().convert("RGB")
    draw = ImageDraw.Draw(output)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    for item in detections:
        x1, y1, x2, y2 = item["box"]
        label = f"Plate {item['confidence']:.2f}"
        draw.rectangle((x1, y1, x2, y2), outline=(0, 220, 90), width=4)
        left, top, right, bottom = draw.textbbox(
            (x1, max(0, y1 - 25)), label, font=font
        )
        draw.rectangle(
            (left - 3, top - 2, right + 3, bottom + 2), fill=(0, 130, 65)
        )
        draw.text((x1, max(0, y1 - 25)), label, fill="white", font=font)
    return output
