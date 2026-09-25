"""Development entry point for the Flask demo.

Usage::

    python run.py
    python run.py --port 8080 --host 0.0.0.0

For anything beyond local development, serve the factory with a WSGI server::

    waitress-serve --port 8000 --call app:create_app        # Windows friendly
    gunicorn "app:create_app()" --bind 0.0.0.0:8000         # Linux / macOS
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from app import create_app

LOGGER = logging.getLogger("run")


def build_arg_parser() -> argparse.ArgumentParser:
    """Define the command-line interface for the development server."""
    parser = argparse.ArgumentParser(
        prog="python run.py",
        description="Run the melanoma classification demo (development server).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default=os.getenv("FLASK_HOST", "127.0.0.1"), help="Bind address.")
    parser.add_argument(
        "--port", type=int, default=int(os.getenv("FLASK_PORT", "5000")), help="Bind port."
    )
    parser.add_argument("--debug", action="store_true", help="Enable the reloader and debugger.")
    return parser


def main() -> int:
    """Start the development server. Returns a process exit code."""
    args = build_arg_parser().parse_args()
    debug = args.debug or os.getenv("FLASK_ENV", "production").lower() == "development"

    app = create_app()
    LOGGER.info("Starting development server on http://%s:%d", args.host, args.port)
    try:
        app.run(host=args.host, port=args.port, debug=debug)
    except OSError as exc:
        LOGGER.error("Could not start the server: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
