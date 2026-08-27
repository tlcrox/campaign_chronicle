"""cc_stages — the pipeline's individual stages.

Each module here is one step of the pipeline and is BOTH:
  * imported and composed by scripts/orchestrate.py, and
  * runnable on its own, either as a script
    (``python scripts/cc_stages/merge_scenes.py --session-dir ...``)
    or as a module (``python -m cc_stages.merge_scenes --session-dir ...``).

They stay outside src/pipeline deliberately: pipeline/* is library code, these
are the executable steps. The ``cc_`` prefix keeps the installed top-level name
from colliding — plain ``stages``, ``steps``, ``phases`` and ``tools`` are all
taken (or takeable) on PyPI, and this package ships in the wheel.
"""
