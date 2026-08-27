#!/usr/bin/env python3
"""
logs.py - one place that configures root logging, called from entry points only.

Every stage in this project is two things at once: a CLI, and a function
orchestrate imports and calls. That is the whole reason this module exists.

``logging.basicConfig`` at module level runs on **import**, not on run, so
importing a stage to call its function configured root logging for the whole
process as a side effect — including under pytest, where a test that imports
orchestrate gave the entire session a stderr handler at INFO. It is also a no-op
after the first call, so with ten modules doing it the one that won was decided
by import order. They were byte-identical, which is the only reason nothing
visibly broke; changing one would have silently done nothing.

So: modules take a logger and never configure one.

    logger = logging.getLogger(__name__)

Entry points — and only entry points — configure the root:

    def main():
        setup_logging()
"""

from __future__ import annotations

import logging

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


def setup_logging(verbose: bool = False) -> None:
    """Configure root logging for this process. Call from ``main()``, never at import.

    ``force=True`` because a library on the import path may already have called
    ``basicConfig`` — whisperx and its dependencies do — and without it this call
    would be the silent no-op the arrangement above was suffering from. The
    entry point is the one caller entitled to have the last word.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=LOG_FORMAT,
        force=True,
    )
