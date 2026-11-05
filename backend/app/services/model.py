from functools import lru_cache
import pathlib

from app.config import MODEL_PATH, CONFIDENCE_THRESHOLD


@lru_cache(maxsize=1)
def load_model():
    """Load the existing custom YOLOv5 checkpoint once per backend process."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"YOLOv5 weights were not found at: {MODEL_PATH}")

    # The existing checkpoint was created on Windows; the deployment runs Linux.
    pathlib.WindowsPath = pathlib.PosixPath

    import torch

    model = torch.hub.load(
        "ultralytics/yolov5:v7.0",
        "custom",
        path=str(MODEL_PATH),
        force_reload=False,
        trust_repo=True,
    )
    model.conf = CONFIDENCE_THRESHOLD
    model.to("cpu")
    model.eval()
    return model
