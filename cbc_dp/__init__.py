"""
Convergent crystallogrphy data processing package.

Compatible with Python 3.X.
"""
from . import utils, Qt, batch, index
from .data_process import Frame, open_scan, Scan1D, Scan2D, ScanST, CorrectedData, NormalizedData
from .feat_detect import HoughLineDetector, LineSegmentDetector, FrameSetup, ScanSetup
from .grouper import Grouper, TiltGroups
from .model import BallLattice, RecLattice, CircModel, RectModel
from .indexer import FCBI, RCBI, ScanCBI, FrameStreaks, ScanStreaks, RecVectors
