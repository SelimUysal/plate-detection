from fastapi.testclient import TestClient

from app.main import app
from app.config import SAMPLE_FILENAMES

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_sample_list_contains_expected_examples():
    response = client.get("/api/samples")
    assert response.status_code == 200
    returned = {item["filename"] for item in response.json()}
    assert returned == set(SAMPLE_FILENAMES)


def test_all_example_images_can_be_retrieved():
    for filename in SAMPLE_FILENAMES:
        response = client.get(f"/api/samples/{filename}")
        assert response.status_code == 200
        assert response.headers["content-type"] in {"image/jpeg", "image/png"}
        assert len(response.content) > 0


def test_sample_endpoint_rejects_unknown_filename():
    response = client.get("/api/samples/not-a-real-file.jpg")
    assert response.status_code == 404


def test_detect_rejects_unsupported_extension():
    response = client.post(
        "/api/detect",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415


def test_detect_rejects_invalid_image():
    response = client.post(
        "/api/detect",
        files={"file": ("broken.jpg", b"not an image", "image/jpeg")},
    )
    assert response.status_code == 400
