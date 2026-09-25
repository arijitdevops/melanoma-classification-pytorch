"""Tests for the Flask demo, including behaviour with no checkpoint present."""

from __future__ import annotations

import io

from PIL import Image


def _jpeg_bytes(size=(64, 64)) -> io.BytesIO:
    """Return an in-memory JPEG suitable for an upload field."""
    buffer = io.BytesIO()
    Image.new("RGB", size, color=(200, 140, 130)).save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer


def test_health_endpoint_reports_missing_model(client):
    """/api/health answers 200 even when no checkpoint exists."""
    response = client.get("/api/health")
    assert response.status_code == 200

    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is False
    assert payload["classes"] is None
    assert payload["checkpoint_path"].endswith("__does_not_exist__.pt")


def test_index_renders_with_checkpoint_warning(client):
    """The landing page loads and tells the user to train a model first."""
    response = client.get("/")
    assert response.status_code == 200

    body = response.get_data(as_text=True)
    assert "No model loaded" in body
    assert "python -m src.train" in body
    assert "Medical disclaimer" in body


def test_api_predict_returns_503_without_checkpoint(client):
    """The JSON API reports 'service unavailable' rather than a 500 traceback."""
    response = client.post(
        "/api/predict",
        data={"image": (_jpeg_bytes(), "lesion.jpg")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 503
    assert "checkpoint" in response.get_json()["error"].lower()


def test_api_predict_requires_an_image_field(client):
    """A request with no file field is a 400."""
    response = client.post("/api/predict", data={}, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "image" in response.get_json()["error"].lower()


def test_api_predict_rejects_unsupported_extensions(client):
    """A non-image extension is refused with 415 before any decoding happens."""
    response = client.post(
        "/api/predict",
        data={"image": (io.BytesIO(b"not an image"), "notes.txt")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 415


def test_api_predict_rejects_undecodable_image(client):
    """A file with an image extension but corrupt bytes is a 400."""
    response = client.post(
        "/api/predict",
        data={"image": (io.BytesIO(b"\x00\x01\x02 not a jpeg"), "broken.jpg")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400


def test_form_upload_redirects_when_model_is_missing(client):
    """The HTML form flashes an error and redirects instead of erroring out."""
    response = client.post(
        "/predict",
        data={"image": (_jpeg_bytes(), "lesion.jpg")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert "No trained checkpoint was found" in response.get_data(as_text=True)


def test_missing_file_redirects_to_index(client):
    """Submitting the form with no file returns to the upload page."""
    response = client.post(
        "/predict", data={}, content_type="multipart/form-data", follow_redirects=True
    )
    assert response.status_code == 200
    assert "Please choose an image file." in response.get_data(as_text=True)
