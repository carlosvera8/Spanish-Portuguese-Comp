"""Small helpers shared by the pipeline scripts."""

import sys


def use_utf8_stdout() -> None:
    """Windows consoles default to cp1252, which cannot print Portuguese text."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass
