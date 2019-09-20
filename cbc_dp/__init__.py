"""
Convergent crystallogrphy data processing package.

Compatible with Python 3.X.
"""
from .wrapper import Frame, OpenScan, Scan1D, Scan2D, ScanST, CorrectedData
from .data import HoughLineDetector, LineSegmentDetector
from .data import FrameStreaks, ScanStreaks, ReciprocalPeaks, CircPropagator, SquarePropagator, NMS
from . import utils, Qt
