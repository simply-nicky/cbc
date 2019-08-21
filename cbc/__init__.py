"""
Convergent beam crystallography package.

Compatible with Python 2.X and 3.X.
"""
from __future__ import absolute_import

from .wrapper import Setup, Diff, DiffRes, Detector
from .beam import GausBeam, BesselBeam, CircBeam, RectBeam
from .lattice import Cell, CubicLattice, BallLattice
from . import utils