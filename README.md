# License Plate Detection & Recognition

A small computer-vision web application that detects license-plate regions with a custom-trained YOLOv5 model and attempts to read each detected plate with Tesseract OCR.

**Live website:**  https://plate-detection-selim-uysal.streamlit.app/

## Features

- Upload a vehicle image (JPG, JPEG, PNG, or WebP).
- Detect one or more license-plate regions with the included `best.pt` model.
- Draw bounding boxes and detection confidence values on the image.
- Run OCR on each detected plate crop.
- Display recognized text and OCR confidence when available.
- Download the annotated result as a PNG image.

## Run locally

Python 3.10 or 3.11 is recommended for the pinned PyTorch dependencies.

1. Clone this repository.
2. Install the system Tesseract OCR package and make sure the `tesseract` command is available on your PATH.
3. Create and activate a virtual environment.
4. Install the Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Start the application:

   ```bash
   streamlit run app.py
   ```

The model file `best.pt` must remain in the repository root next to `app.py`.

## Free deployment with Streamlit Community Cloud

1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
2. Create a new app and select this GitHub repository and the `main` branch.
3. Set the main file path to `app.py`.
4. Deploy. Streamlit Cloud uses `requirements.txt` for Python dependencies and `packages.txt` for system packages such as Tesseract.

The app performs inference on CPU. Free hosting resources are limited, so the first launch may take time while dependencies and the model are initialized.

## Important limitations

- OCR is best-effort and may misread characters, especially on blurry, angled, distant, reflective, or poorly lit plates.
- Detection confidence is not a guarantee that a detection is correct. OCR confidence is a separate Tesseract measurement.
- The training dataset is small; do not interpret a few successful examples as a reliable real-world accuracy benchmark.
- The app currently processes one uploaded image per run and does not provide a persistent image-storage service.

## Dataset attribution

The plate-detection training dataset was exported from Roboflow Universe:

[Plaka Tespiti dataset](https://universe.roboflow.com/selim-chl0i/plaka-tespiti/dataset/1)

The dataset page identifies the dataset as licensed under **CC BY 4.0**. Retain attribution and comply with the dataset license when redistributing or using the dataset.

## Model and dependency licensing

This project uses YOLOv5 through its official repository and Tesseract OCR. YOLOv5 is distributed under the GNU AGPL-3.0 license. Review the applicable license terms and retain required notices when distributing or making this application available over a network. Tesseract OCR is distributed under the Apache-2.0 license.

- [YOLOv5 repository and license](https://github.com/ultralytics/yolov5)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)

## Privacy

Uploaded images are processed in the running app session. This project does not include a feature to save user-uploaded images or build a plate-recognition history. Hosting providers may have their own operational logs and retention practices; review their terms before publishing the app for public use.
