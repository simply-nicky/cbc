"""
Convergent crystallogrphy data processing package.

Compatible with Python 3.X.
"""
from . import utils, Qt
from .data_process import Frame, open_scan, Scan1D, Scan2D, ScanST, CorrectedData
from .feat_detect import HoughLineDetector, LineSegmentDetector, FrameSetup, ScanSetup
from .feat_detect import FrameStreaks, ScanStreaks, RecVectors
from .grouper import Grouper, TiltGroups
from .model import BallLattice, CircModel, RectModel
from .indexer import FCBI, RCBI, IndexSolution
