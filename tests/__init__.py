"""Test package.

These __init__.py files exist so `python -m unittest discover` works — unittest
will not recurse into non-package directories, and without them it silently
reports "Ran 0 tests ... OK", which looks like success.

A HAZARD TO KNOW ABOUT
----------------------
The test tree mirrors the source tree (tests/unit/pipeline/...), so
tests/unit/pipeline/__init__.py makes `tests/unit/pipeline` a *regular* package.
If anything ever puts `tests/unit` on sys.path, that package SHADOWS the real
`pipeline`, and every `pipeline.*` import in the suite dies with a confusing
"No module named 'pipeline.config'".

So: keep these __init__.py files, and do NOT add sys.path manipulation to any
test module. `pipeline` and `cc_stages` come from the installed project (see
pyproject.toml); no test needs to touch sys.path. The two rules are only safe
together.
"""
