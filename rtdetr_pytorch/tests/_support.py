"""TEST-ONLY placeholders for optional dependencies irrelevant to model tests.

No production command imports this module. It does not provide usable datasets.
"""

import sys
import types
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def prepare_imports():
    try:
        import pycocotools
    except ImportError:
        package = types.ModuleType('pycocotools')
        for name in ('mask', 'coco', 'cocoeval'):
            child = types.ModuleType(f'pycocotools.{name}')
            setattr(package, name, child)
            sys.modules[f'pycocotools.{name}'] = child
        package.coco.COCO = object
        package.cocoeval.COCOeval = object
        sys.modules['pycocotools'] = package
    try:
        import transformers
    except ImportError:
        package = types.ModuleType('transformers')
        package.RegNetModel = object
        sys.modules['transformers'] = package
