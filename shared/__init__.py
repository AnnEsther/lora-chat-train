"""shared/__init__.py — Shared logging configuration for all components."""

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """Call once at application startup to set up structured log format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
        stream=sys.stdout,
        force=True,
    )
