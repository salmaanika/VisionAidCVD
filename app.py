import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt  # Import matplotlib for bar chart

APP_TITLE = "VISION AID: Color Perception Enhancement System for Color Blind Users"
MODEL_PATH = "best.pt"
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# =============================
# Helpers (non-UML utilities)
# =============================
def is_allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTS


def safe_stem(name: str) -> str:
    stem = Path(name).stem if name else "image"
    stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)
    return stem or "image"


def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def detections_json_bytes(detections: list[dict]) -> bytes:
    return (json.dumps(detections, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def make_zip_bytes(files: list[tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname, data in files:
            zf.writestr(fname, data)
    return buf.getvalue()


def cvd_suffix(cvd_type: str) -> str:
    return "raw" if cvd_type == "None" else cvd_type.lower()


def swatch_image(rgb: tuple[int, int, int], size: int = 70) -> Image.Image:
    return Image.new("RGB", (size, size), rgb)


def dominant_color_from_rgb(raw_rgb: np.ndarray) -> tuple[str, tuple[int, int, int]]:
    avg = raw_rgb.reshape(-1, 3).mean(axis=0)
    r, g, b = [float(x) for x in avg]

    v = (r + g + b) / 3.0
    mx = max(r, g, b)
    mn = min(r, g, b)

    # low chroma -> gray scale
    if (mx - mn) < 18:
        if v < 50:
            name = "Black"
        elif v > 210:
            name = "White"
        else:
            name = "Gray"
        return name, (int(r), int(g), int(b))

    # secondary colors (simple heuristic)
    strong = 160
    not_strong = 190

    if r > strong and g > strong and b < not_strong:
        name = "Yellow"
    elif g > strong and b > strong and r < not_strong:
        name = "Cyan"
    elif r > strong and b > strong and g < not_strong:
        name = "Magenta"
    else:
        if r >= g and r >= b:
            name = "Red"
        elif g >= r and g >= b:
            name = "Green"
        else:
            name = "Blue"

    return name, (int(r), int(g), int(b))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# =============================
# Streamlit-safe caches (IMPORTANT)
# =============================
@st.cache_resource
def load_yolo_model(path: str) -> YOLO:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            "Put best.pt in the repo root (or update MODEL_PATH)."
        )
    return YOLO(path)


@st.cache_data(show_spinner=False)
def yolo_infer_cached(
    model_path: str,
    image_bytes: bytes,
    conf: float,
    iou: float,
) -> tuple[np.ndarray, list[dict]]:
    model = load_yolo_model(model_path)

    frame_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = model.predict(source=frame_pil, conf=conf, iou=iou, verbose=False)

    annotated_rgb = np.array(results[0].plot(pil=True))

    dets: list[dict] = []
    for b in results[0].boxes:
        cls = int(b.cls[0])
        dets.append(
            {
                "box": b.xyxy[0].tolist(),
                "confidence": float(b.conf[0]),
                "class_id": cls,
                "class_name": model.names.get(cls, str(cls)),
            }
        )

    return annotated_rgb.astype(np.uint8), dets


# =============================
# Main Application
# =============================
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    ui = UserInterface()
    color_engine = ColorCorrectionEngine()
    ml = MachineLearningModel(MODEL_PATH)
    feedback = FeedbackModule()
    audio_feedback = AudioFeedbackModule()

    # ---- Sidebar: Configure CVD Type, thresholds (User -> UserInterface)
    with st.sidebar:
        st.header("Input Source")
        source = st.radio("Choose input", ["Upload Image", "Live Camera"], index=0)

        st.header("CVD Type")
        cvd_type = st.selectbox("Select CVD type", ["None", "Protanopia", "Deuteranopia", "Tritanopia"], index=0)
        ui.selectCVDType(cvd_type)

        st.header("CVD Intensity")
        intensity = st.slider("Intensity", 0.0, 1.0, float(ui.cvdIntensity), 0.05)
        ui.setCVDIntensity(float(intensity))

        st.header("Detection Settings")
        conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
        iou_threshold = st.slider("IoU threshold", 0.0, 1.0, 0.45, 0.01)

        st.divider()
        if st.button("Reload model / clear cache", use_container_width=True):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()

    # ---- Capture Frame (UserInterface captures frame)
    image_name = "camera.png"
    image_pil: Image.Image
    image_bytes: bytes

    if source == "Upload Image":
        uploaded = st.file_uploader("Upload image", type=[e[1:] for e in ALLOWED_EXTS])
        if not uploaded:
            st.stop()
        if not is_allowed(uploaded.name):
            st.error(f"Unsupported file. Allowed: {sorted(ALLOWED_EXTS)}")
            st.stop()

        image_name = uploaded.name
        image_bytes = uploaded.getvalue()
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    else:
        cam = st.camera_input("Capture image from camera")
        if not cam:
            st.stop()

        image_name = "camera.png"
        image_bytes = cam.getvalue()
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    base = safe_stem(image_name)

    # raw frame as RGB numpy
    raw_rgb = np.array(image_pil).astype(np.uint8)

    # ---- ML detection (UserInterface -> MachineLearningModel) [CACHED]
    try:
        with st.spinner("Running YOLO inference... (cached when possible)"):
            annotated_rgb, detections = ml.classifyColor(image_bytes, conf_threshold, iou_threshold)
    except Exception as e:
        st.error("YOLO inference failed.")
        st.exception(e)
        st.stop()

    # ---- Feedback text (MachineLearningModel -> FeedbackModule)
    text_label = feedback.generateTextLabel(detections)

    # ---- Bar Chart for Detection Confidence
    # Extract Confidence Data for Bar Chart
    class_names = [item['class_name'] for item in detections]
    confidence = [item['confidence'] for item in detections]

    # Create a bar chart to visualize the confidence of each detection class
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(class_names, confidence, color='skyblue')
    ax.set_xlabel('Class Name')
    ax.set_ylabel('Confidence')
    ax.set_title('Detection Confidence by Class')
    ax.set_xticklabels(class_names, rotation=45, ha='right')

    # Show the bar chart in the Streamlit app
    st.pyplot(fig)

    # ---- Buttons (Filter/Audio) under the middle panel
    col1, col2, col3 = st.columns(3, gap="large")
    # Other parts of the app will remain the same...
