from pydantic import BaseModel


class DetectionResult(BaseModel):
    box: tuple[int, int, int, int]
    confidence: float
    text: str
    ocr_confidence: float | None = None


class DetectionResponse(BaseModel):
    filename: str
    count: int
    detections: list[DetectionResult]
    annotated_image: str
