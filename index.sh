#!/bin/bash

source /home/nivanov/conda-setup.sh
conda activate python3

python -m cbc_dp.index $*
