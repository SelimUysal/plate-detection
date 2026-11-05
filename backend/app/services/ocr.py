import re

import cv2
import numpy as np
import pytesseract


def read_plate_text(crop: np.ndarray) -> tuple[str, float | None]:
    """Read a plate crop using the same preprocessing and Tesseract settings as the legacy app."""
    if crop.size == 0:
        return "", None

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    processed = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    config = (
        "--oem 3 --psm 7 "
        "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )
    data = pytesseract.image_to_data(
        processed,
        config=config,
        output_type=pytesseract.Output.DICT,
    )

    words: list[str] = []
    confidences: list[float] = []
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
    mean_confidence = (
        round(sum(confidences) / len(confidences), 1) if confidences else None
    )
    return text, mean_confidence
