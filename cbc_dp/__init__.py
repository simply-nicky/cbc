"""
Convergent crystallogrphy data processing package.

Compatible with Python 3.X.
"""
from .data_process import Frame, open_scan, Scan1D, Scan2D, ScanST, CorrectedData
from .feat_detect import HoughLineDetector, LineSegmentDetector, NMS, ExperimentSettings
from .feat_detect import FrameStreaks, ScanStreaks, CircPropagator, SquarePropagator
from .feat_detect import RecVectors, RefinedRecVectors
from . import utils, Qt
