"""
Utility package for convergent beam crystallography project.

Compatible with Python 2.X and 3.X.
"""
from __future__ import absolute_import

from .utilfuncs import j0, jit_integrand, quad_complex
from .utilfuncs import rotation_matrix
from .utilfuncs import search_rec, AxesSeq
from .utilfuncs import make_filename, make_dirs, PAR_PATH, RES_PATH
from .utilfuncs import get_logpath, NullHandler
from .utilfuncs import CORES_NUM, DiffWorker, phase_inc
from . import asf, pdb
