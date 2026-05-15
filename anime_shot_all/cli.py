"""Command-line entrypoint."""

from __future__ import annotations

import argparse

from .gui import build_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the Anime Shot All Gradio GUI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    app = build_app()
    app.launch(server_name=args.host, server_port=args.port, share=args.share)
