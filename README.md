# Convergent gaussian beam crystallography project
Simulation for convergent gaussian beam diffration on a crystalline sample written in Python. Code is compatible with Python 2.X and 3.X.

For more information regarding the theory behind this see the article written by Prof. Henry Chapman:
https://e-reports-ext.llnl.gov/pdf/314988.pdf

## compilation

It's a package called cbc, with which you can program the simulation. See usage examples: [go to cbc structure](#cbc-structure)

Required dependencies:

- NumPy
- SciPy
- matplotlib
- h5py

## cbc structure

The library itself consists of two modules and an utility package:

- functions.py - a module with all functions for convergent gaussian beam crystallography simulation
- wrapper.py - class wrapper for settting up and running diffraction simulations as well as plotting and writing into HDF5 file diffraction results
- utils - utility package:
    - utilfuncs.py - utility functions for convergent beam diffraction project

Also there are couple of usage examples:

- diff-run.py - conducting diffraction simulation and saving results to HDF5 file
- diff-slides.py - calculating and saving diffraction pattern series
- read.py - reading and ploting diffraction results

## upcoming things to do

- consider more complex samples, protein crystalls