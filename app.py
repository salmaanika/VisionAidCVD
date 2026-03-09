# app.py
# VisionAid - UML-aligned implementation (Class + Sequence Diagram names)
# Pure RGB pipeline: YOLO uses PIL(RGB), color correction uses LMS, no HSV/BGR required.

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
    """
    RGB-only approximate dominant color name.
    Detects: Red, Green, Blue, Yellow, Cyan, Magenta, White, Black, Gray.
    """
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
    """Cache the heavy YOLO model resource. Safe because args are hashable (path string)."""
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
    """
    Cache inference results so Filter/Audio reruns don't re-run YOLO.
    Returns:
      annotated_rgb (np.uint8 HxWx3),
      detections (list[dict]) - JSON-serializable
    """
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
# ColorCorrectionEngine (UML)
# =============================
class ColorCorrectionEngine:
    """
    Applies CVD simulation/correction using LMS missing-cone matrices.
    """

    # RGB -> LMS matrix and inverse
    RGB_TO_LMS = np.array(
        [
            [0.31399, 0.63951, 0.04650],
            [0.15537, 0.75789, 0.08670],
            [0.01775, 0.10944, 0.87257],
        ],
        dtype=np.float32,
    )
    LMS_TO_RGB = np.linalg.inv(RGB_TO_LMS).astype(np.float32)

    LMS_MISSING = {
        "None": np.eye(3, dtype=np.float32),
        "Protanopia": np.array([[0, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
        "Deuteranopia": np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32),
        "Tritanopia": np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0]], dtype=np.float32),
    }

    def applyCorrection(self, data: np.ndarray, cvd_type: str, intensity: float = 1.0) -> np.ndarray:
        """
        UML-like signature:
          applyCorrection(data: RawColorData, type: String, intensity: Double): CorrectedData

        Notes:
        - intensity blends between original and corrected:
            intensity=0 => original
            intensity=1 => fully corrected
        """
        cvd_type = cvd_type if cvd_type in self.LMS_MISSING else "None"

        if cvd_type == "None" or intensity <= 0:
            return data

        corrected = self._apply_cvd(data, cvd_type)

        if intensity >= 1:
            return corrected

        a = float(intensity)
        out = (data.astype(np.float32) * (1 - a) + corrected.astype(np.float32) * a)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _apply_cvd(self, rgb: np.ndarray, cvd_type: str) -> np.ndarray:
        M = self.LMS_MISSING[cvd_type]
        x = rgb.astype(np.float32) / 255.0
        lms = x @ self.RGB_TO_LMS.T
        lms = lms @ M.T
        rgb2 = lms @ self.LMS_TO_RGB.T
        return (np.clip(rgb2, 0, 1) * 255).astype(np.uint8)


# =============================
# MachineLearningModel (UML)
# =============================
class MachineLearningModel:
    """
    Wraps the YOLO model for detection and annotated image generation.
    Uses Streamlit-safe caches (no caching on instance methods).
    """

    def __init__(self, mlModelPath: str):
        self.mlModelPath = mlModelPath
        # Keep a reference (and ensure model exists) but inference is cached separately:
        self.model = load_yolo_model(mlModelPath)

    def classifyColor(self, image_bytes: bytes, conf: float, iou: float) -> tuple[np.ndarray, list[dict]]:
        """
        In your diagram, 'classifyColor' is the ML step.
        Here it returns (annotated_rgb, detections) using cached inference.
        """
        return yolo_infer_cached(self.mlModelPath, image_bytes, conf, iou)


# =============================
# FeedbackModule (UML) - Text
# =============================
class FeedbackModule:
    def generateTextLabel(self, detections: list[dict]) -> str:
        if not detections:
            return "No objects detected."
        counts: dict[str, int] = {}
        for d in detections:
            counts[d["class_name"]] = counts.get(d["class_name"], 0) + 1
        parts = [f"{v} {k}" for k, v in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]
        return "Detected Color: " + ", ".join(parts) + "."


# =============================
# AudioFeedbackModule (UML)
# =============================
class AudioFeedbackModule:
    def generateAudio(self, label: str) -> bytes | None:
        """
        Uses gTTS (requires internet in many environments).
        """
        try:
            from gtts import gTTS  # pip install gTTS

            buf = io.BytesIO()
            gTTS(text=label, lang="en").write_to_fp(buf)
            return buf.getvalue()
        except Exception:
            return None


# =============================
# UserInterface (UML) - Streamlit Controller
# =============================
class UserInterface:
    """
    Manages Streamlit state and rendering.
    """

    def __init__(self):
        st.session_state.setdefault("filterButtonState", False)
        st.session_state.setdefault("playAudioState", False)
        st.session_state.setdefault("cvdType", "None")
        st.session_state.setdefault("cvdIntensity", 1.0)

    @property
    def filterButtonState(self) -> bool:
        return bool(st.session_state.get("filterButtonState", False))

    @property
    def playAudioState(self) -> bool:
        return bool(st.session_state.get("playAudioState", False))

    @property
    def cvdType(self) -> str:
        return str(st.session_state.get("cvdType", "None"))

    @property
    def cvdIntensity(self) -> float:
        return float(st.session_state.get("cvdIntensity", 1.0))

    def selectCVDType(self, cvd_type: str):
        st.session_state["cvdType"] = cvd_type

    def setCVDIntensity(self, intensity: float):
        st.session_state["cvdIntensity"] = float(intensity)

    def toggleFilters(self):
        st.session_state["filterButtonState"] = not self.filterButtonState
        # optional: when filter toggles, stop audio
        st.session_state["playAudioState"] = False

    def toggleAudio(self):
        st.session_state["playAudioState"] = not self.playAudioState

    def displayOutput(self, title: str, image_rgb: np.ndarray, label: str | None = None):
        st.subheader(title)
        st.image(Image.fromarray(image_rgb), use_container_width=True)
        if label:
            st.write(label)


# =============================
# App wiring (matches Sequence Diagram flow)
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

    # ---- Raw dominant color (extra UX)
    color_name, avg_rgb = dominant_color_from_rgb(raw_rgb)
    st.image(swatch_image(avg_rgb), width=80, caption=color_name)
    st.write(f"Average RGB: {avg_rgb}")

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

    # ---- Buttons (Filter/Audio) under the middle panel
    col1, col2, col3 = st.columns(3, gap="large")

    with col2:
        btnA, btnB = st.columns(2)
        with btnA:
            if st.button("Filter", use_container_width=True):
                ui.toggleFilters()
                st.rerun()
        with btnB:
            if st.button("Audio", use_container_width=True):
                ui.toggleAudio()
                st.rerun()

    # ---- Apply ColorCorrectionEngine ONLY when filter is ON
    if ui.filterButtonState and ui.cvdType != "None":
        filtered_original = color_engine.applyCorrection(raw_rgb, ui.cvdType, ui.cvdIntensity)
        filtered_annotated = color_engine.applyCorrection(annotated_rgb, ui.cvdType, ui.cvdIntensity)
    else:
        filtered_original = raw_rgb
        filtered_annotated = annotated_rgb

    # ---- Audio feedback module only when Audio is ON
    audio_mp3 = None
    if ui.playAudioState:
        audio_mp3 = audio_feedback.generateAudio(text_label)

    # ---- Display Output (UserInterface -> User)
    with col1:
        ui.displayOutput("Original (RAW)", raw_rgb)

    with col2:
        ui.displayOutput("Result (Annotated - RAW)", annotated_rgb, label=text_label)

        if ui.playAudioState:
            st.markdown("### Audio")
            if audio_mp3 is None:
                st.warning("Audio not available. Install gTTS and ensure internet access.")
            else:
                st.audio(audio_mp3, format="audio/mp3")

    with col3:
        st.subheader("Filtered Images (CVD)")
        if ui.filterButtonState and ui.cvdType != "None":
            st.image(
                Image.fromarray(filtered_original),
                caption=f"Filtered Original ({ui.cvdType})",
                use_container_width=True,
            )
            st.image(
                Image.fromarray(filtered_annotated),
                caption=f"Filtered Annotated ({ui.cvdType})",
                use_container_width=True,
            )
        elif ui.filterButtonState and ui.cvdType == "None":
            st.info("Filter is ON, but CVD Type is None. Choose a CVD type.")
        else:
            st.info("Press Filter to view CVD images")

    # ---- JSON output
    st.subheader("Detections (JSON)")
    st.json(detections)

    # ---- Downloads (RAW vs FILTERED + ZIP + auto naming)
    st.subheader("Download")

    suffix = cvd_suffix(ui.cvdType)

    raw_img_name = f"{base}_annotated_raw.png"
    raw_json_name = f"{base}_detections_raw.json"

    filtered_img_name = f"{base}_annotated_{suffix}.png"
    filtered_json_name = f"{base}_detections_{suffix}.json"

    # Separate downloads: RAW
    st.markdown("#### Download RAW")
    st.download_button(
        label="Download RAW annotated image (PNG)",
        data=pil_to_bytes(Image.fromarray(annotated_rgb), fmt="PNG"),
        file_name=raw_img_name,
        mime="image/png",
    )
    st.download_button(
        label="Download RAW detections (JSON)",
        data=detections_json_bytes(detections),
        file_name=raw_json_name,
        mime="application/json",
    )

    # Separate downloads: FILTERED
    st.markdown("#### Download FILTERED")
    if ui.cvdType == "None":
        st.info("Choose a CVD type (Protanopia/Deuteranopia/Tritanopia) to enable filtered downloads.")
    else:
        st.download_button(
            label=f"Download FILTERED annotated image (PNG) [{ui.cvdType}]",
            data=pil_to_bytes(Image.fromarray(filtered_annotated), fmt="PNG"),
            file_name=filtered_img_name,
            mime="image/png",
        )
        # NOTE: detections are from RAW inference (same content). Keep name if you want bundling symmetry.
        st.download_button(
            label=f"Download FILTERED detections (JSON) [{ui.cvdType}]",
            data=detections_json_bytes(detections),
            file_name=filtered_json_name,
            mime="application/json",
        )

    # ZIP bundles: image + json together
    st.markdown("#### Download ZIP (Image + JSON)")

    raw_zip = make_zip_bytes(
        [
            (raw_img_name, pil_to_bytes(Image.fromarray(annotated_rgb), fmt="PNG")),
            (raw_json_name, detections_json_bytes(detections)),
        ]
    )
    st.download_button(
        label="Download RAW ZIP",
        data=raw_zip,
        file_name=f"{base}_raw_bundle.zip",
        mime="application/zip",
    )

    if ui.cvdType == "None":
        st.info("Filtered ZIP is disabled because CVD type is None.")
    else:
        filtered_zip = make_zip_bytes(
            [
                (filtered_img_name, pil_to_bytes(Image.fromarray(filtered_annotated), fmt="PNG")),
                (filtered_json_name, detections_json_bytes(detections)),
            ]
        )
        st.download_button(
            label=f"Download FILTERED ZIP ({ui.cvdType})",
            data=filtered_zip,
            file_name=f"{base}_{suffix}_bundle.zip",
            mime="application/zip",
        )


if __name__ == "__main__":
    main()
