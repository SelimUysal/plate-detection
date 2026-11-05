# Sample images

Add the three demonstration images to this folder using these exact filenames:

1. `01_perfect_recognition.jpg` — the plate is detected and OCR reads the plate correctly.
2. `02_detection_correct_ocr_incorrect.jpg` — the plate is detected, but OCR reads the plate incorrectly.
3. `03_detection_and_ocr_failed.jpg` — the detector fails to identify the plate; OCR therefore has no correct plate crop to read.

Use JPG images for these exact filenames. The Streamlit app displays the examples vertically in the right-hand column and runs each selected image through the live YOLOv5 + Tesseract pipeline. Do not use an image that contains personal or sensitive information unless you have permission to publish it.
