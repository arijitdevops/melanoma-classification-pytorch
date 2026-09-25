"""Flask application factory for the melanoma classification demo.

The app loads a checkpoint lazily on first use. When no checkpoint is present
the UI and the JSON API both report that clearly instead of crashing, so the
demo can be started before any model has been trained.

Research demo only. Not a medical device.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from flask import Flask, jsonify, request

from src.utils import setup_logging

LOGGER = logging.getLogger("app")

#: Upload size ceiling; dermoscopic JPEGs are far smaller than this.
MAX_CONTENT_LENGTH = 8 * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def create_app(config_overrides: dict[str, Any] | None = None) -> Flask:
    """Build and configure the Flask application.

    Args:
        config_overrides: Values merged into ``app.config`` after the defaults,
            used mainly by the test suite.

    Returns:
        The configured :class:`~flask.Flask` application.
    """
    setup_logging()
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=os.getenv("SECRET_KEY", "change-me-in-production"),
        MAX_CONTENT_LENGTH=MAX_CONTENT_LENGTH,
        ALLOWED_EXTENSIONS=ALLOWED_EXTENSIONS,
        CHECKPOINT_PATH=os.getenv("CHECKPOINT_PATH", "checkpoints/best.pt"),
        CONFIG_PATH=os.getenv("CONFIG_PATH"),
        DEVICE=os.getenv("DEVICE", "auto"),
        DECISION_THRESHOLD=float(os.getenv("DECISION_THRESHOLD", "0.5")),
    )
    app.json.sort_keys = False  # keep response keys in insertion order
    if config_overrides:
        app.config.update(config_overrides)

    from .routes import bp as main_blueprint

    app.register_blueprint(main_blueprint)

    @app.errorhandler(413)
    def too_large(_error):
        """Return a friendly message when an upload exceeds MAX_CONTENT_LENGTH."""
        message = f"File too large. The limit is {MAX_CONTENT_LENGTH // (1024 * 1024)} MB."
        if request.path.startswith("/api/"):
            return jsonify({"error": message}), 413
        return message, 413

    LOGGER.info("Flask app created (checkpoint: %s)", app.config["CHECKPOINT_PATH"])
    return app


__all__ = ["create_app", "ALLOWED_EXTENSIONS", "MAX_CONTENT_LENGTH"]
