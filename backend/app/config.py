from pathlib import Path
import os

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(REPOSITORY_ROOT / "best.pt")))
SAMPLE_DIR = Path(os.getenv("SAMPLE_DIR", str(REPOSITORY_ROOT / "assets" / "sample_images")))

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "640"))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "20000000"))

SAMPLE_FILENAMES = (
    "01_perfect_recognition.jpg",
    "02_detection_correct_ocr_incorrect.jpg",
    "03_detection_and_ocr_failed.jpg",
)

_raw_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
)
CORS_ORIGINS = [origin.strip() for origin in _raw_origins.split(",") if origin.strip()]
