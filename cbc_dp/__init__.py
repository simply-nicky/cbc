"""
Convergent crystallogrphy data processing package.

Compatible with Python 3.X.
"""
from .data_process import Frame, open_scan, Scan1D, Scan2D, ScanST, CorrectedData
from .feat_detect import HoughLineDetector, LineSegmentDetector, ExperimentSettings
from .feat_detect import FrameStreaks, ScanStreaks, RecVectors
from .model import CircModel, SquareModel, RecLattice, QIndexTF, QIndexStreaksTF, OrthQIndexTF
from .model import qindex_point, orthqindex_point, ExpSetTF, IndexingSolution
from .grouper import Grouper, TiltGroups
from . import utils, Qt
