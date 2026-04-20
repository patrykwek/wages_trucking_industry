from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def _configure() -> None:
    """Configure the package logger once, reading HAULAGE_LOG_LEVEL."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    level_name = os.environ.get("HAULAGE_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s :: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    root = logging.getLogger("haulage")
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under the `haulage` root."""
    _configure()
    if not name.startswith("haulage"):
        name = f"haulage.{name}"
    return logging.getLogger(name)
