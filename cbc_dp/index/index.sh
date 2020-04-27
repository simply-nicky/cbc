#!/bin/bash

source /home/nivanov/conda-setup.sh
conda activate python3

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=6
export MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=6

python -m cbc_dp.index $*
