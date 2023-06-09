import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  

# if os.path.join(str(ROOT), "PaddleOCR") not in sys.path:
#     sys.path.append(os.path.join(str(ROOT), "PaddleOCR"))