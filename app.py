from __future__ import annotations

from io import BytesIO
from pathlib import Path
import pathlib
import re

import cv2
import numpy as np
import pytesseract
import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "best.pt"
CONFIDENCE_THRESHOLD = 0.25
IMAGE_SIZE = 640
MAX_IMAGE_PIXELS = 20_000_000

st.set_page_config(
    page_title="License Plate Detection",
    page_icon="🚘",
    layout="centered",
)

st.title("🚘 License Plate Detection & Recognition")
st.caption("Upload a vehicle image to detect license plates and read the characters with OCR.")

@st.cache_resource(show_spinner="YOLOv5 model is loading…")
def load_model():
    if not WEIGHTS.exists():
        raise FileNotFoundError("The model file best.pt could not be found in the repository.")

    # The checkpoint was created on Windows and may contain pathlib.WindowsPath
    # objects. Streamlit Cloud runs Linux, where unpickling WindowsPath fails.
    # Map that serialized class to PosixPath while YOLOv5 loads the checkpoint.
    if not hasattr(pathlib, "WindowsPath"):
        pathlib.WindowsPath = pathlib.PosixPath
    else:
        pathlib.WindowsPath = pathlib.PosixPath

    # CPU inference keeps the app compatible with the free hosting tier.
    model = torch.hub.load(
        "ultralytics/yolov5:v7.0",
        "custom",
        path=str(WEIGHTS),
        force_reload=False,
        trust_repo=True,
    )
    model.conf = CONFIDENCE_THRESHOLD
    model.to("cpu")
    model.eval()
    return model


def read_plate_text(crop: np.ndarray) -> tuple[str, float | None]:
    """Read a plate crop with Tesseract; return text and mean OCR confidence."""
    if crop.size == 0:
        return "", None

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    config = (
        "--oem 3 --psm 7 "
        "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )
    data = pytesseract.image_to_data(
        processed,
        config=config,
        output_type=pytesseract.Output.DICT,
    )

    words = []
    confidences = []
    for word, raw_conf in zip(data["text"], data["conf"]):
        cleaned = re.sub(r"[^A-Za-z0-9]", "", word).upper()
        try:
            confidence = float(raw_conf)
        except (TypeError, ValueError):
            confidence = -1
        if cleaned:
            words.append(cleaned)
            if confidence >= 0:
                confidences.append(confidence)

    text = "".join(words)
    mean_confidence = round(sum(confidences) / len(confidences), 1) if confidences else None
    return text, mean_confidence


def annotate_image(
    image: Image.Image,
    detections: list[dict],
) -> Image.Image:
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
        left, top, right, bottom = draw.textbbox((x1, max(0, y1 - 25)), label, font=font)
        draw.rectangle((left - 3, top - 2, right + 3, bottom + 2), fill=(0, 130, 65))
        draw.text((x1, max(0, y1 - 25)), label, fill="white", font=font)
    return output


uploaded_file = st.file_uploader(
    "Choose a vehicle image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Use a clear image where the vehicle and its plate are visible.",
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        if image.width * image.height > MAX_IMAGE_PIXELS:
            st.error("This image is too large. Please upload an image with fewer than 20 million pixels.")
            st.stop()
        st.subheader("Uploaded image")
        st.image(image, use_container_width=True)

        if st.button("Detect license plate", type="primary", use_container_width=True):
            with st.spinner("Detecting plates and running OCR…"):
                model = load_model()
                image_array = np.array(image)
                results = model(image_array, size=IMAGE_SIZE)
                raw = results.xyxy[0].detach().cpu().numpy()

                detections = []
                for row in raw:
                    x1, y1, x2, y2, confidence, class_id = row.tolist()
                    x1i = max(0, min(image.width, int(x1)))
                    y1i = max(0, min(image.height, int(y1)))
                    x2i = max(0, min(image.width, int(x2)))
                    y2i = max(0, min(image.height, int(y2)))
                    if x2i <= x1i or y2i <= y1i:
                        continue

                    crop = image_array[y1i:y2i, x1i:x2i]
                    plate_text, ocr_confidence = read_plate_text(crop)
                    detections.append({
                        "box": (x1i, y1i, x2i, y2i),
                        "confidence": float(confidence),
                        "text": plate_text,
                        "ocr_confidence": ocr_confidence,
                    })

            st.session_state["plate_detections"] = detections
            st.session_state["plate_source_image"] = image

    except Exception as exc:
        st.error("The image could not be processed.")
        st.exception(exc)

if "plate_detections" in st.session_state:
    detections = st.session_state["plate_detections"]
    source_image = st.session_state["plate_source_image"]

    st.divider()
    st.subheader("Detection results")
    st.metric("Detected plates", len(detections))

    if detections:
        annotated = annotate_image(source_image, detections)
        st.image(annotated, caption="Detected license plate regions", use_container_width=True)

        for index, item in enumerate(detections, start=1):
            st.markdown(f"**Plate {index}**")
            st.write(f"Detection confidence: **{item['confidence']:.1%}**")
            if item["text"]:
                st.write(f"Recognized text: **{item['text']}**")
                if item["ocr_confidence"] is not None:
                    st.caption(f"OCR confidence (Tesseract): {item['ocr_confidence']:.1f}/100")
            else:
                st.warning("The plate was detected, but its text could not be read clearly.")

        buffer = BytesIO()
        annotated.save(buffer, format="PNG")
        st.download_button(
            "Download result image",
            data=buffer.getvalue(),
            file_name="license-plate-detection.png",
            mime="image/png",
            use_container_width=True,
        )
    else:
        st.warning("No license plate was detected. Try a sharper image with a larger, clearly visible plate.")

st.divider()
st.caption(
    "Note: OCR output may contain errors. Detection confidence and OCR confidence are different "
    "measurements; always verify the recognized plate manually. Images are processed during the "
    "session and are not intentionally stored by this app."
)
