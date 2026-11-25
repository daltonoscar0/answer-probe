"""Make the test helpers in `tests/` importable as plain modules."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
