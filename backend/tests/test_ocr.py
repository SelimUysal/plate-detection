import numpy as np

from app.services.ocr import read_plate_text


def test_empty_crop_returns_no_text_without_calling_tesseract():
    text, confidence = read_plate_text(np.empty((0, 0, 3), dtype=np.uint8))
    assert text == ""
    assert confidence is None
