import { useEffect, useState, type ChangeEvent } from "react";
import {
  checkApiHealth,
  detectImage,
  getSampleFile,
  getSamples,
  type DetectionResponse,
  type SampleImage,
} from "./api";

const sampleDescriptions: Record<string, string> = {
  "01_perfect_recognition.jpg": "Detection and OCR both correct",
  "02_detection_correct_ocr_incorrect.jpg": "Detection correct, OCR incorrect",
  "03_detection_and_ocr_failed.jpg": "Detection and OCR failed",
};

function App() {
  const [samples, setSamples] = useState<SampleImage[]>([]);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState<DetectionResponse | null>(null);
  const [loadingSamples, setLoadingSamples] = useState(true);
  const [loadingSample, setLoadingSample] = useState("");
  const [detecting, setDetecting] = useState(false);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    Promise.all([getSamples(), checkApiHealth()])
      .then(([items]) => {
        if (!active) return;
        setSamples(items);
        setApiOnline(true);
      })
      .catch(() => {
        if (!active) return;
        setApiOnline(false);
        setError("The processing service is not available. Start the FastAPI backend and refresh this page.");
      })
      .finally(() => {
        if (active) setLoadingSamples(false);
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl("");
      return;
    }
    const objectUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [selectedFile]);

  async function chooseSample(filename: string) {
    setLoadingSample(filename);
    setError("");
    try {
      const file = await getSampleFile(filename);
      setSelectedFile(file);
      setResult(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "The example image could not be loaded.");
    } finally {
      setLoadingSample("");
    }
  }

  function chooseUpload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    setSelectedFile(file);
    setResult(null);
    setError("");
    event.target.value = "";
  }

  async function runDetection() {
    if (!selectedFile) return;
    setDetecting(true);
    setError("");
    try {
      const response = await detectImage(selectedFile);
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "The image could not be processed.");
      setResult(null);
    } finally {
      setDetecting(false);
    }
  }

  return (
    <main className="page-shell">
      <header className="topbar">
        <div className="brand-mark" aria-hidden="true">LP</div>
        <div>
          <p className="eyebrow">COMPUTER VISION</p>
          <h1>License Plate Detection</h1>
        </div>
        <span className={`service-status ${apiOnline === true ? "online" : apiOnline === false ? "offline" : ""}`}>
          <span className="status-dot" />
          {apiOnline === true ? "Service online" : apiOnline === false ? "Service offline" : "Connecting"}
        </span>
      </header>

      <section className="intro">
        <h2>Detect and read license plates</h2>
        <p>Select an example image or upload a vehicle photo to run detection and OCR.</p>
      </section>

      {error && <div className="alert" role="alert">{error}</div>}

      <section className="section-block">
        <div className="section-heading">
          <div>
            <p className="eyebrow">GET STARTED</p>
            <h2>Example images</h2>
          </div>
          <span className="section-hint">Choose an example to preview it</span>
        </div>

        <div className="sample-grid">
          {samples.map((sample) => (
            <article className="sample-card" key={sample.filename}>
              <div className="sample-image-wrap">
                <img
                  src={`http://localhost:8000/api/samples/${encodeURIComponent(sample.filename)}`}
                  alt={sample.filename}
                  loading="lazy"
                  onError={(event) => {
                    const image = event.currentTarget;
                    image.style.visibility = "hidden";
                  }}
                />
              </div>
              <p className="sample-description">
                {sampleDescriptions[sample.filename] || "Example image"}
              </p>
              <p className="sample-filename">{sample.filename.replace(/\.jpg$/i, "")}</p>
              <button
                className="button button-secondary button-full"
                type="button"
                onClick={() => void chooseSample(sample.filename)}
                disabled={Boolean(loadingSample) || apiOnline === false}
              >
                {loadingSample === sample.filename ? "Loading…" : "Try this image"}
              </button>
            </article>
          ))}
          {!loadingSamples && samples.length === 0 && (
            <p className="empty-state">No example images are available.</p>
          )}
          {loadingSamples && <p className="empty-state">Loading example images…</p>}
        </div>
      </section>

      <section className="section-block upload-section">
        <div className="section-heading">
          <div>
            <p className="eyebrow">YOUR IMAGE</p>
            <h2>Upload your own image</h2>
            <p className="section-description">Choose a JPG, PNG, or WebP image from your device.</p>
          </div>
        </div>
        <label className="upload-panel" htmlFor="image-upload">
          <span className="upload-icon" aria-hidden="true">↑</span>
          <span className="upload-title">Choose an image</span>
          <span className="upload-subtitle">Browse files on your device</span>
          <span className="browse-button">Browse files</span>
        </label>
        <input
          id="image-upload"
          className="visually-hidden"
          type="file"
          accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp"
          onChange={chooseUpload}
        />
      </section>

      {selectedFile && (
        <section className="section-block selected-section">
          <div className="section-heading centered-heading">
            <div>
              <p className="eyebrow">READY TO PROCESS</p>
              <h2>Selected image</h2>
            </div>
          </div>
          <div className="preview-card">
            <img src={previewUrl} alt="Selected image preview" />
            <p>{selectedFile.name}</p>
          </div>
          <button
            className="button button-primary detect-button"
            type="button"
            onClick={() => void runDetection()}
            disabled={detecting || apiOnline === false}
          >
            {detecting ? "Detecting plates…" : "Detect"}
          </button>
        </section>
      )}

      {result && (
        <section className="section-block results-section">
          <div className="section-heading">
            <div>
              <p className="eyebrow">ANALYSIS COMPLETE</p>
              <h2>Detection results</h2>
              <p className="section-description">Image: {result.filename}</p>
            </div>
            <div className="result-count">
              <strong>{result.count}</strong>
              <span>Detected plates</span>
            </div>
          </div>

          {result.count > 0 ? (
            <>
              <div className="annotated-image">
                <img
                  src={`data:image/png;base64,${result.annotated_image}`}
                  alt="Image annotated with detected license plate regions"
                />
              </div>
              <div className="detection-list">
                {result.detections.map((detection, index) => (
                  <article className="detection-card" key={`${detection.box.join("-")}-${index}`}>
                    <h3>Plate {index + 1}</h3>
                    <p>Detection confidence: <strong>{(detection.confidence * 100).toFixed(1)}%</strong></p>
                    {detection.text ? (
                      <>
                        <p>Recognized text: <strong>{detection.text}</strong></p>
                        {detection.ocr_confidence !== null && (
                          <p className="muted">OCR confidence (Tesseract): {detection.ocr_confidence.toFixed(1)}/100</p>
                        )}
                      </>
                    ) : (
                      <p className="warning-text">The plate was detected, but its text could not be read clearly.</p>
                    )}
                  </article>
                ))}
              </div>
              <a
                className="button button-secondary download-button"
                href={`data:image/png;base64,${result.annotated_image}`}
                download="license-plate-detection.png"
              >
                Download result image
              </a>
            </>
          ) : (
            <div className="empty-result">
              No license plate was detected. Try a sharper image with a larger, clearly visible plate.
            </div>
          )}
        </section>
      )}

      <footer className="footer">
        <span>License Plate Detection</span>
        <span>YOLOv5 · OpenCV · Tesseract OCR</span>
      </footer>
    </main>
  );
}

export default App;
