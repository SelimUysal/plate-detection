export type SampleImage = {
  filename: string;
};

export type Detection = {
  box: [number, number, number, number];
  confidence: number;
  text: string;
  ocr_confidence: number | null;
};

export type DetectionResponse = {
  filename: string;
  count: number;
  detections: Detection[];
  annotated_image: string;
};

const API_BASE_URL = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/+$/, "");

async function readError(response: Response): Promise<Error> {
  try {
    const payload = await response.json();
    return new Error(payload.detail || `Request failed (${response.status})`);
  } catch {
    return new Error(`Request failed (${response.status})`);
  }
}

export async function checkApiHealth(): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/health`);
  if (!response.ok) throw await readError(response);
}

export async function getSamples(): Promise<SampleImage[]> {
  const response = await fetch(`${API_BASE_URL}/api/samples`);
  if (!response.ok) throw await readError(response);
  return response.json() as Promise<SampleImage[]>;
}

export async function getSampleFile(filename: string): Promise<File> {
  const response = await fetch(
    `${API_BASE_URL}/api/samples/${encodeURIComponent(filename)}`,
  );
  if (!response.ok) throw await readError(response);
  const blob = await response.blob();
  return new File([blob], filename, { type: blob.type || "image/jpeg" });
}

export async function detectImage(file: File): Promise<DetectionResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${API_BASE_URL}/api/detect`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) throw await readError(response);
  return response.json() as Promise<DetectionResponse>;
}
