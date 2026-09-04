"""
Makes `import stego` work when running pytest straight from a clone,
without needing `pip install -e .` first.

pytest imports this file automatically before collecting any tests.
"""

import pathlib
import sys

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
