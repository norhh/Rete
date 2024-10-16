"""Compatibility launcher for the canonical Rete/Trident synthesizer.

The original archive duplicated a stale, broken copy here. Keep one algorithm
implementation in Rete-Trident/main/synthesis.py so the two entry points agree.
"""

from pathlib import Path
import runpy
import sys


if __name__ == "__main__":
    canonical = Path(__file__).resolve().parents[1] / "Rete-Trident/main/synthesis.py"
    sys.path.insert(0, str(canonical.parent))
    runpy.run_path(str(canonical), run_name="__main__")
