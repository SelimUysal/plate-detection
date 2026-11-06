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
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "best.pt"
SAMPLE_DIR = ROOT / "assets" / "sample_images"
CONFIDENCE_THRESHOLD = 0.25
IMAGE_SIZE = 640
MAX_IMAGE_PIXELS = 20_000_000

SAMPLES = [
    {"filename": "01_perfect_recognition.jpg"},
    {"filename": "02_detection_correct_ocr_incorrect.jpg"},
    {"filename": "03_detection_and_ocr_failed.jpg"},
]

st.set_page_config(
    page_title="License Plate Detection",
    page_icon="🚘",
    layout="wide",
)

st.markdown(
    """
    <style>
    .result-details {
        text-align: center;
        font-size: 1.08rem;
        line-height: 1.65;
        margin: 0.35rem auto 1rem auto;
    }
    .result-details strong { font-size: 1.15rem; }
    .sample-description {
        text-align: center;
        font-size: 1rem;
        font-weight: 650;
        line-height: 1.35;
        min-height: 52px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .sample-filename {
        text-align: center;
        font-size: 0.9rem;
        font-weight: 500;
        margin-top: 0.45rem;
    }
    /* Consistent, accessible light interface for first-time visitors. */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background: #ffffff !important;
        color: #172033 !important;
    }
    [data-testid="stAppViewContainer"] .main {
        background: #ffffff !important;
    }
    .block-container {
        max-width: 1100px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3, p, label, [data-testid="stMarkdownContainer"] {
        color: #172033;
    }
    h1 {
        text-align: center;
        letter-spacing: -0.035em;
        font-weight: 750;
    }
    [data-testid="stCaptionContainer"] {
        text-align: center;
        color: #526078;
        font-size: 1.02rem;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #e4e9f1 !important;
        border-radius: 14px !important;
        background: #ffffff !important;
    }
    .stButton > button, .stDownloadButton > button {
        border-radius: 10px;
        font-weight: 650;
        min-height: 2.7rem;
        transition: transform 120ms ease, box-shadow 120ms ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(23, 32, 51, 0.10);
    }
    [data-testid="stMetric"] {
        background: #f6f8fc;
        border: 1px solid #e7ecf3;
        border-radius: 12px;
        padding: 0.9rem 1rem;
    }
    [data-testid="stAlert"] {
        border-radius: 12px;
    }
    @media (max-width: 700px) {
        .block-container {
            padding: 1.25rem 1rem 2rem 1rem;
        }
        h1 { font-size: 1.9rem !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("License Plate Detection")
st.caption("Upload a vehicle photo and let the app find and read its license plate.")

st.markdown(
    """
    <div style="max-width:760px;margin:0 auto 1.4rem auto;text-align:center;
                color:#526078;font-size:0.98rem;line-height:1.65;">
        Choose one of the example photos below or upload an image from your device.
        The app will highlight detected license plates and attempt to read their text.
    </div>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner="YOLOv5 model is loading…")
def load_model():
    if not WEIGHTS.exists():
        raise FileNotFoundError("The model file best.pt could not be found in the repository.")

    # The checkpoint was created on Windows; Streamlit Cloud runs Linux.
    pathlib.WindowsPath = pathlib.PosixPath

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
        left, top, right, bottom = draw.textbbox((x1, max(0, y1 - 25)), label, font=font)
        draw.rectangle((left - 3, top - 2, right + 3, bottom + 2), fill=(0, 130, 65))
        draw.text((x1, max(0, y1 - 25)), label, fill="white", font=font)
    return output


def detect_plates(image: Image.Image) -> list[dict]:
    model = load_model()
    image_array = np.array(image.convert("RGB"))
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
    return detections


def select_sample(sample_path: Path, filename: str) -> None:
    # Keep the uploader widget state untouched; Streamlit does not allow
    # assigning to a widget key after that widget has been instantiated.
    st.session_state["selected_sample_bytes"] = sample_path.read_bytes()
    st.session_state["selected_sample_name"] = filename
    st.session_state["active_image_source"] = "sample"
    st.session_state.pop("plate_detections", None)
    st.session_state.pop("plate_source_image", None)
    st.session_state.pop("plate_source_name", None)


st.subheader("Try an example")
sample_columns = st.columns(3, gap="medium")
sample_descriptions = [
    "Detection and OCR both correct",
    "Detection correct, OCR incorrect",
    "Detection and OCR failed",
]

for column, sample, description in zip(sample_columns, SAMPLES, sample_descriptions):
    sample_path = SAMPLE_DIR / sample["filename"]
    with column:
        with st.container(border=True):
            st.markdown(
                f"<div class='sample-description'>{description}</div>",
                unsafe_allow_html=True,
            )
            if sample_path.exists():
                with Image.open(sample_path) as source_image:
                    # Keep every preview frame identical, but show the full photo
                    # smaller inside it so different source aspect ratios do not crop
                    # the subject or make the examples look inconsistent.
                    source_image = ImageOps.exif_transpose(source_image).convert("RGB")
                    contained_image = ImageOps.contain(source_image, (360, 210))
                    thumbnail = Image.new("RGB", (420, 260), (245, 245, 245))
                    paste_x = (420 - contained_image.width) // 2
                    paste_y = (260 - contained_image.height) // 2
                    thumbnail.paste(contained_image, (paste_x, paste_y))
                thumbnail_buffer = BytesIO()
                thumbnail.save(thumbnail_buffer, format="JPEG", quality=88)
                # Preview images are deliberately non-clickable. Use the Try button to select one.
                st.image(thumbnail, use_container_width=True)
                st.markdown(
                    f"<div class='sample-filename'>"
                    f"{Path(sample['filename']).stem}</div>",
                    unsafe_allow_html=True,
                )
                if st.button("Try", key=f"sample_{sample['filename']}", use_container_width=True):
                    select_sample(sample_path, sample["filename"])
            else:
                st.error(f"Missing: {sample['filename']}")

st.divider()
st.subheader("Or upload your own photo")
st.markdown(
    """
    <style>
    .upload-help {
        color: var(--text-color);
        font-size: 0.92rem;
        max-width: 680px;
        margin: -0.35rem auto 0.8rem auto;
        opacity: 0.78;
    }
    div[data-testid="stFileUploader"] {
        width: 100%;
        max-width: 680px;
        margin-left: auto;
        margin-right: auto;
    }
    [data-testid="stFileUploader"] section {
        border: 1.5px dashed var(--primary-color);
        border-radius: 12px;
        padding: 1.15rem 1rem;
        background: rgba(128, 128, 128, 0.045);
    }
    [data-testid="stFileUploader"] section > div {
        gap: 0.35rem;
    }
    [data-testid="stFileUploader"] button {
        border-radius: 8px;
        font-weight: 600;
    }
    </style>
    <div class="upload-help">Supported formats: JPG, JPEG, PNG and WebP. For best results, use a clear photo where the vehicle and plate are visible.</div>
    """,
    unsafe_allow_html=True,
)
def mark_uploaded_image() -> None:
    st.session_state["active_image_source"] = "upload"


uploaded_file = st.file_uploader(
    "Browse image files",
    type=["jpg", "jpeg", "png", "webp"],
    key="user_image_upload",
    label_visibility="collapsed",
    on_change=mark_uploaded_image,
)

selected_bytes = None
selected_name = None

# The uploader's on_change callback switches the active source to "upload".
# Otherwise, a deliberately selected sample remains active even if the uploader
# still holds an older file in its widget state.
if st.session_state.get("active_image_source") == "sample":
    selected_bytes = st.session_state.get("selected_sample_bytes")
    selected_name = st.session_state.get("selected_sample_name", "example-image")
elif uploaded_file is not None:
    selected_bytes = uploaded_file.getvalue()
    selected_name = uploaded_file.name
    st.session_state["active_image_source"] = "upload"
elif "selected_sample_bytes" in st.session_state:
    selected_bytes = st.session_state["selected_sample_bytes"]
    selected_name = st.session_state.get("selected_sample_name", "example-image")

if selected_bytes is not None:
    try:
        selected_preview = Image.open(BytesIO(selected_bytes)).convert("RGB")
        if selected_preview.width * selected_preview.height > MAX_IMAGE_PIXELS:
            st.error("This image is too large. Please choose an image with fewer than 20 million pixels.")
            selected_bytes = None
        else:
            st.subheader("Selected image")
            _, preview_column, _ = st.columns([1, 2, 1])
            with preview_column:
                st.image(selected_preview, use_container_width=True)
    except Exception:
        st.error("The selected image could not be opened.")
        selected_bytes = None

process_now = False
pending_bytes = None
pending_name = "selected-image"
_, detect_column, _ = st.columns([1, 1, 1])
with detect_column:
    _, button_column, _ = st.columns([1, 1, 1])
    with button_column:
        process_clicked = st.button(
            "Detect license plate",
            type="primary",
            use_container_width=False,
            disabled=selected_bytes is None,
        )

if process_clicked:
    pending_bytes = selected_bytes
    pending_name = selected_name or "selected-image"
    process_now = True

if process_now and pending_bytes is not None:
    try:
        image = Image.open(BytesIO(pending_bytes)).convert("RGB")
        if image.width * image.height > MAX_IMAGE_PIXELS:
            st.error("This image is too large. Please upload an image with fewer than 20 million pixels.")
        else:
            with st.status("Processing image… Please wait.", state="running", expanded=False) as process_status:
                detections = detect_plates(image)
                process_status.update(label="Image processing completed.", state="complete", expanded=False)
            st.session_state["plate_detections"] = detections
            st.session_state["plate_source_image"] = image
            st.session_state["plate_source_name"] = pending_name
    except Exception:
        st.error("We could not process this image. Please try another JPG, PNG or WebP photo.")

if "plate_detections" in st.session_state:
    detections = st.session_state["plate_detections"]
    source_image = st.session_state["plate_source_image"]

    st.divider()
    st.subheader("Your results")
    st.markdown(
        f"<div class='result-details'>Photo: <strong>{st.session_state.get('plate_source_name', 'Selected image')}</strong></div>",
        unsafe_allow_html=True,
    )
    _, metric_column, _ = st.columns([1, 2, 1])
    with metric_column:
        st.metric("Detected plates", len(detections))

    if detections:
        annotated = annotate_image(source_image, detections)
        _, result_image_column, _ = st.columns([1, 2, 1])
        with result_image_column:
            st.image(annotated, caption="Detected license plate regions", use_container_width=True)

        for index, item in enumerate(detections, start=1):
            st.markdown(
                f"<div class='result-details'><strong>Plate {index}</strong><br>"
                f"Detection confidence: <strong>{item['confidence']:.1%}</strong></div>",
                unsafe_allow_html=True,
            )
            if item["text"]:
                st.markdown(
                    f"<div class='result-details'>Recognized text: <strong>{item['text']}</strong></div>",
                    unsafe_allow_html=True,
                )
                if item["ocr_confidence"] is not None:
                    st.markdown(
                        f"<div class='result-details'>Text-reading confidence: "
                        f"<strong>{item['ocr_confidence']:.1f}/100</strong></div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.warning("A license plate was found, but we could not read its text clearly. Try a closer, sharper photo.")

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
        st.warning("No license plate was found in this photo. Try a clearer image with the plate fully visible and facing the camera.")

