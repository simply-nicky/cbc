"""
Convergent beam crystallography package.

Compatible with Python 2.X and 3.X.
"""
from __future__ import absolute_import

from .wrapper import Setup, Diff, DiffSA, DiffYar, DiffRes, Detector
from .beam import GausBeam, BesselBeam, CircLens, RectLens
from .lattice import Cell, CubicLattice, BallLattice, rec_basis
from . import utils
