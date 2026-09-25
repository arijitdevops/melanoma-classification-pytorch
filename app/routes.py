"""HTTP routes: upload form, prediction result page and the JSON API."""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Any

import torch
from flask import (
    Blueprint,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

from src.config import POSITIVE_CLASS_INDEX, load_config, resolve_path
from src.gradcam import generate_overlay
from src.predict import load_predictor
from src.transforms import build_eval_transform
from src.utils import select_device

LOGGER = logging.getLogger("app.routes")

bp = Blueprint("main", __name__)

#: Lazily-populated model cache: {"model", "class_names", "image_size", "device"}.
_MODEL_CACHE: dict[str, Any] = {}

CHECKPOINT_HINT = (
    "No trained checkpoint was found. Train a model first with "
    "'python -m src.train', or point CHECKPOINT_PATH at an existing .pt file."
)


def _allowed(filename: str) -> bool:
    """Return ``True`` when the filename has a permitted image extension."""
    return Path(filename).suffix.lower() in current_app.config["ALLOWED_EXTENSIONS"]


def get_model() -> dict[str, Any] | None:
    """Return the cached model bundle, loading it on first call.

    Returns:
        A dict with ``model``, ``class_names``, ``image_size`` and ``device``,
        or ``None`` when the checkpoint is missing or unreadable.
    """
    if _MODEL_CACHE.get("model") is not None:
        return _MODEL_CACHE

    checkpoint_path = resolve_path(current_app.config["CHECKPOINT_PATH"])
    try:
        config = load_config(current_app.config.get("CONFIG_PATH"))
        device = select_device(current_app.config.get("DEVICE", "auto"))
        model, class_names, image_size = load_predictor(checkpoint_path, config, device)
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
        LOGGER.warning("Model unavailable: %s", exc)
        return None

    _MODEL_CACHE.update(
        {"model": model, "class_names": class_names, "image_size": image_size, "device": device}
    )
    LOGGER.info("Model loaded from %s", checkpoint_path)
    return _MODEL_CACHE


def reset_model_cache() -> None:
    """Clear the cached model. Used by the tests and after retraining."""
    _MODEL_CACHE.clear()


def _to_data_uri(image: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL image as a base64 ``data:`` URI for inline display."""
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{encoded}"


def _classify(image: Image.Image) -> tuple[dict[str, Any], str | None]:
    """Run inference plus Grad-CAM on one image.

    Returns:
        ``(payload, error)``. ``error`` is a human-readable string when the
        model is unavailable or inference failed.
    """
    bundle = get_model()
    if bundle is None:
        return {}, CHECKPOINT_HINT

    model = bundle["model"]
    class_names = bundle["class_names"]
    device = bundle["device"]
    image_size = bundle["image_size"]
    threshold = float(current_app.config.get("DECISION_THRESHOLD", 0.5))

    try:
        tensor = build_eval_transform(image_size)(image.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            probabilities = torch.softmax(model(tensor).float(), dim=1)[0].cpu().numpy()
    except (RuntimeError, ValueError) as exc:
        LOGGER.exception("Inference failed")
        return {}, f"Inference failed: {exc}"

    malignant_probability = float(probabilities[POSITIVE_CLASS_INDEX])
    predicted_index = (
        POSITIVE_CLASS_INDEX if malignant_probability >= threshold else 1 - POSITIVE_CLASS_INDEX
    )

    payload: dict[str, Any] = {
        "predicted_class": class_names[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "threshold": threshold,
        "probabilities": {
            name: float(probabilities[index]) for index, name in enumerate(class_names)
        },
        "malignant_probability": malignant_probability,
    }
    return payload, None


@bp.route("/", methods=["GET"])
def index():
    """Render the upload form."""
    return render_template(
        "index.html",
        model_ready=get_model() is not None,
        checkpoint_hint=CHECKPOINT_HINT,
    )


@bp.route("/predict", methods=["POST"])
def predict():
    """Handle a form upload and render the result page."""
    uploaded = request.files.get("image")
    if uploaded is None or not uploaded.filename:
        flash("Please choose an image file.", "warning")
        return redirect(url_for("main.index"))

    filename = secure_filename(uploaded.filename)
    if not _allowed(filename):
        flash("Unsupported file type. Use JPG, PNG, BMP or WebP.", "danger")
        return redirect(url_for("main.index"))

    try:
        image = Image.open(uploaded.stream).convert("RGB")
    except (OSError, UnidentifiedImageError):
        flash("That file could not be read as an image.", "danger")
        return redirect(url_for("main.index"))

    payload, error = _classify(image)
    if error:
        flash(error, "danger")
        return redirect(url_for("main.index"))

    bundle = get_model()
    overlay_uri = None
    if bundle is not None:
        try:
            overlay, _target, _probabilities = generate_overlay(
                bundle["model"], image, bundle["image_size"], bundle["device"],
                class_index=POSITIVE_CLASS_INDEX,
            )
            overlay_uri = _to_data_uri(overlay)
        except (RuntimeError, ValueError, OSError) as exc:
            LOGGER.warning("Grad-CAM unavailable for this image: %s", exc)

    return render_template(
        "result.html",
        filename=filename,
        original_uri=_to_data_uri(image, fmt="JPEG"),
        overlay_uri=overlay_uri,
        result=payload,
    )


@bp.route("/api/predict", methods=["POST"])
def api_predict():
    """Classify an uploaded image and return JSON.

    Request: ``multipart/form-data`` with an ``image`` field.
    Response: ``{"predicted_class", "confidence", "probabilities", ...}``.
    """
    uploaded = request.files.get("image")
    if uploaded is None or not uploaded.filename:
        return jsonify({"error": "Missing 'image' file field."}), 400
    if not _allowed(secure_filename(uploaded.filename)):
        return jsonify({"error": "Unsupported file type."}), 415

    try:
        image = Image.open(uploaded.stream).convert("RGB")
    except (OSError, UnidentifiedImageError):
        return jsonify({"error": "File could not be decoded as an image."}), 400

    payload, error = _classify(image)
    if error:
        status = 503 if error == CHECKPOINT_HINT else 500
        return jsonify({"error": error}), status

    payload["filename"] = secure_filename(uploaded.filename)
    payload["disclaimer"] = (
        "Research demo only. This output is not a diagnosis and must not be used "
        "for clinical decisions."
    )
    return jsonify(payload), 200


@bp.route("/api/health", methods=["GET"])
def api_health():
    """Report service and model-loading status."""
    bundle = get_model()
    return (
        jsonify(
            {
                "status": "ok",
                "model_loaded": bundle is not None,
                "checkpoint_path": str(resolve_path(current_app.config["CHECKPOINT_PATH"])),
                "classes": bundle["class_names"] if bundle else None,
                "device": str(bundle["device"]) if bundle else None,
            }
        ),
        200,
    )
