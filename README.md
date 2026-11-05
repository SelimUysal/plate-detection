# License Plate Detection & Recognition

A computer-vision application that detects vehicle license-plate regions with the existing custom-trained YOLOv5 model and attempts to read each plate with Tesseract OCR.

The project is being organized as a separate frontend and backend:

- **Frontend:** React, TypeScript, Vite, HTML, and CSS.
- **Backend:** Python and FastAPI.
- **AI pipeline:** the existing `best.pt` YOLOv5 checkpoint, OpenCV, Pillow, and Tesseract OCR.

The original Streamlit application remains available in `app.py` as a fallback while the new application is tested and deployed separately. The model checkpoint, sample images, and legacy Streamlit configuration are retained.

## Features

- Choose one of the three bundled example images or upload a JPG, JPEG, PNG, or WebP image.
- Preview the selected image before processing.
- Detect one or more license-plate regions with the existing `best.pt` model.
- Draw bounding boxes and detection confidence values on the image.
- Run OCR on each detected plate crop.
- Display recognized text and Tesseract OCR confidence when available.
- Download the annotated result as a PNG image.

## Project structure

```text
plate-detection/
├── app.py                         # Preserved Streamlit fallback
├── best.pt                        # Existing custom YOLOv5 weights
├── assets/sample_images/          # Existing example images
├── backend/
│   ├── app/
│   │   ├── main.py                # FastAPI routes and request handling
│   │   ├── config.py              # Paths and runtime configuration
│   │   ├── schemas.py              # API response models
│   │   └── services/
│   │       ├── model.py            # YOLOv5 model loading
│   │       ├── detector.py         # Detection and annotation
│   │       └── ocr.py              # Tesseract OCR
│   ├── tests/
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── api.ts
│   │   └── styles.css
│   ├── Dockerfile
│   └── package.json
├── docker-compose.yml
├── requirements.txt               # Preserved Streamlit dependencies
└── packages.txt                   # Preserved Streamlit Cloud system packages
```

## Run the React + FastAPI application with Docker

Requirements: Docker Desktop or Docker Engine with the Docker Compose plugin.

From the repository root:

```bash
docker compose up --build
```

Open the frontend at [http://localhost:5173](http://localhost:5173).

The API health endpoint is [http://localhost:8000/health](http://localhost:8000/health), and the interactive API documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).

Stop the services with:

```bash
docker compose down
```

The backend image installs Tesseract OCR and the CPU build of PyTorch. The first model inference may take longer because YOLOv5 and its dependencies may need to initialize.

## Run locally without Docker

### 1. Backend

Python 3.11 is recommended. Install the Tesseract system package first and make sure the `tesseract` command is available on your PATH.

On Debian/Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng
python -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
PYTHONPATH=backend uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

On Windows, install Tesseract OCR separately and ensure its installation directory is on PATH. Then create a virtual environment, install `backend/requirements.txt`, and run the Uvicorn command with `PYTHONPATH=backend` set to the repository's `backend` directory.

### 2. Frontend

In a second terminal:

```bash
cd frontend
npm install
```

The Vite development server proxies `/api` and `/health` requests to the backend on port 8000, so no API URL setting is required for local development. If the frontend and API are hosted on different origins, set `VITE_API_URL` in the frontend build environment to the public FastAPI URL.

Then start Vite:

```bash
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Backend health check |
| GET | `/api/samples` | List available example images |
| GET | `/api/samples/{filename}` | Retrieve an approved example image |
| POST | `/api/detect` | Detect plates and run OCR on an uploaded image |
| GET | `/docs` | Interactive FastAPI documentation |

The detection endpoint accepts multipart form data with a `file` field. It returns the source filename, detection count, bounding boxes, detection confidence, OCR text/confidence, and a base64-encoded annotated PNG.

## Configuration

The backend supports these environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | Repository-root `best.pt` | YOLOv5 checkpoint path |
| `SAMPLE_DIR` | `assets/sample_images` | Example image directory |
| `CORS_ORIGINS` | Local Vite origins | Comma-separated allowed frontend origins |
| `CONFIDENCE_THRESHOLD` | `0.25` | YOLOv5 detection threshold |
| `IMAGE_SIZE` | `640` | YOLOv5 inference size |
| `MAX_IMAGE_PIXELS` | `20000000` | Maximum accepted image pixel count |

For a deployed frontend, set `VITE_API_URL` to the public FastAPI URL at frontend build time, and set `CORS_ORIGINS` on the backend to the exact frontend origin. Do not expose private credentials in frontend environment variables.

## Legacy Streamlit application

The original application has deliberately not been deleted. To run it during the migration:

```bash
pip install -r requirements.txt
streamlit run app.py
```

For Streamlit Community Cloud, the existing `app.py`, root `requirements.txt`, and `packages.txt` remain in place. The currently published Streamlit deployment is not automatically replaced by the React/FastAPI source; the new frontend and API need their own hosting configuration and public API URL before switching the public link.

## Quality checks

GitHub Actions runs backend compilation/tests and a frontend TypeScript production build for pushes to `main` and `feature/**` branches, and for pull requests targeting `main`.

Run the checks locally:

```bash
PYTHONPATH=backend python -m pytest backend/tests -q
python -m compileall -q backend/app
cd frontend
npm install
npm run build
```

## Important limitations

- OCR is best-effort and may misread characters, especially on blurry, angled, distant, reflective, or poorly lit plates.
- Detection confidence is not a guarantee that a detection is correct. OCR confidence is a separate Tesseract measurement.
- The training dataset is small; a few successful examples should not be treated as a reliable real-world accuracy benchmark.
- The application processes one image per request and does not implement a persistent image-storage or plate-recognition history service.
- The inference pipeline currently runs on CPU.

## Dataset attribution

The plate-detection training dataset was exported from Roboflow Universe:

[Plaka Tespiti dataset](https://universe.roboflow.com/selim-chl0i/plaka-tespiti/dataset/1)

The dataset page identifies the dataset as licensed under **CC BY 4.0**. Retain attribution and comply with the dataset license when redistributing or using the dataset.

## Model and dependency licensing

This project uses YOLOv5 through its official repository and Tesseract OCR. YOLOv5 is distributed under the GNU AGPL-3.0 license. Review the applicable license terms and retain required notices when distributing or making this application available over a network. Tesseract OCR is distributed under the Apache-2.0 license.

- [YOLOv5 repository and license](https://github.com/ultralytics/yolov5)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)

## Privacy

Uploaded images are processed in the running backend request and are not written to an application-managed image archive. Hosting providers may have their own operational logs and retention practices; review their terms before publishing the app for public use.
