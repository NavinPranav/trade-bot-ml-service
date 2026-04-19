import sys
from pathlib import Path

_dir = str(Path(__file__).parent)
if _dir not in sys.path:
    sys.path.insert(0, _dir)
