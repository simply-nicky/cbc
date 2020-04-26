#!/bin/bash

source /home/nivanov/conda-setup.sh
conda activate python3

python cbc_dp/index/combine.py $*