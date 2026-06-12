"""
Medical Image AI Agent
Accepts X-ray, MRI, CT Scan, Skin Image uploads.
Uses PyTorch/torchvision for classification; falls back to a rule-based mock
when PyTorch is not installed (e.g., CPU-only Docker builds).
"""
import io
import logging
import os
import uuid
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)

UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

SUPPORTED_MODALITIES = {"xray", "mri", "ct", "skin"}
CATEGORIES = ["normal", "abnormal", "inconclusive"]
CONFIDENCE_THRESHOLD = 0.60
IMAGE_SIZE = 224
MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB

# ---------------------------------------------------------------------------
# Try PyTorch stack
# ---------------------------------------------------------------------------

try:
    import torch  # type: ignore
    import torchvision.transforms as T  # type: ignore
    import torchvision.models as models  # type: ignore
    from PIL import Image  # type: ignore

    _device = torch.device("cpu")

    # Use a pretrained ResNet-18 as a feature extractor, then a small linear head.
    # In production this would be a fine-tuned MONAI or Med-CLIP model.
    _backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    _backbone.fc = torch.nn.Linear(_backbone.fc.in_features, 3)  # 3 classes
    _backbone.eval()
    _backbone.to(_device)

    _transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    TORCH_AVAILABLE = True
    logger.info("PyTorch image classifier loaded (ResNet-18 demo weights).")

except Exception as _te:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch unavailable (%s) — using mock image classifier.", _te)


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

def preprocess_image(image_bytes: bytes) -> Any:
    """Resize and normalise image bytes into a model-ready tensor."""
    if not TORCH_AVAILABLE:
        return None
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _transform(image).unsqueeze(0).to(_device)
    return tensor


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_image(
    image_bytes: bytes, modality: str = "xray"
) -> Dict[str, Any]:
    """
    Classify a medical image.

    Args:
        image_bytes: raw bytes of the uploaded image
        modality: one of 'xray', 'mri', 'ct', 'skin'

    Returns:
        dict with predicted_category, confidence, all_scores, modality, note
    """
    modality = modality.lower().strip()
    if modality not in SUPPORTED_MODALITIES:
        modality = "xray"

    if not TORCH_AVAILABLE:
        return _mock_classify(modality)

    try:
        tensor = preprocess_image(image_bytes)
        with torch.no_grad():
            logits = _backbone(tensor)
            probs = torch.softmax(logits, dim=1).squeeze().tolist()

        scores = {CATEGORIES[i]: round(float(probs[i]), 3) for i in range(3)}
        max_cat = max(scores, key=scores.get)
        max_conf = scores[max_cat]

        if max_conf < CONFIDENCE_THRESHOLD:
            predicted = "inconclusive"
            note = (
                "Confidence below threshold — specialist review recommended. "
                "This analysis is supplementary only."
            )
        else:
            predicted = max_cat
            note = (
                f"Predicted: {predicted.upper()} ({round(max_conf*100,1)}% confidence). "
                "Consult a qualified physician for diagnosis."
            )

        return {
            "predicted_category": predicted,
            "confidence": max_conf,
            "all_scores": scores,
            "modality": modality,
            "note": note,
            "model": "ResNet-18 (demo weights — not clinically validated)",
        }

    except Exception as e:
        logger.error("Image classification error: %s", e)
        return _mock_classify(modality, error=str(e))


def _mock_classify(modality: str, error: Optional[str] = None) -> Dict[str, Any]:
    """Rule-based mock when PyTorch is unavailable."""
    from typing import Optional
    note = (
        "PyTorch model unavailable — returning demo analysis. "
        "Consult a qualified physician for diagnosis."
    )
    if error:
        note += f" (Technical detail: {error})"
    return {
        "predicted_category": "inconclusive",
        "confidence": 0.50,
        "all_scores": {"normal": 0.30, "abnormal": 0.20, "inconclusive": 0.50},
        "modality": modality,
        "note": note,
        "model": "mock",
    }


# ---------------------------------------------------------------------------
# File storage
# ---------------------------------------------------------------------------

def save_image(image_bytes: bytes, original_filename: str) -> str:
    """
    Save image with a UUID filename to UPLOAD_DIR.
    Returns the stored filename (NOT the original — per security requirement).
    """
    ext = os.path.splitext(original_filename)[1].lower() or ".jpg"
    safe_name = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(UPLOAD_DIR, safe_name)
    with open(path, "wb") as f:
        f.write(image_bytes)
    return safe_name


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def analyse_medical_image(
    image_bytes: bytes,
    original_filename: str,
    modality: str = "xray",
) -> Dict[str, Any]:
    """
    Full pipeline: validate → save → classify → return result.
    Called by main.py /patient/images/upload endpoint.
    """
    if len(image_bytes) > MAX_IMAGE_BYTES:
        return {
            "success": False,
            "error": "Image exceeds 20 MB size limit.",
            "http_status": 413,
        }

    stored_name = save_image(image_bytes, original_filename)
    result = classify_image(image_bytes, modality)
    result["stored_filename"] = stored_name
    result["success"] = True
    return result
