# Convergent beam crystallography project
Simulation for convergent beam diffration on a crystalline sample written in Python. Code is compatible with Python 2.X and 3.X.

For more information regarding the theory behind this see [the article written by Prof. Henry Chapman](https://e-reports-ext.llnl.gov/pdf/314988.pdf).

## compilation

It's a package called cbc, with which you can conduct a diffraction simulation. See usage examples: [go to cbc package structure](#cbc-package-structure)

Required dependencies:

- NumPy
- Numba
- SciPy
- matplotlib
- h5py

## features

The package can perform convergent beam diffraction simulation of crystalline samples based on first Born approximation theory.

Available incoming beam models:

- Gaussian beam
- Bessel beam
- Lens beam with rectangular or circular aperture

Samples could be composed of different compound unit cells deffined by array of atom coordinates within the unit cell and corresponding B-factors. These data can be imported via PDB format file. Unit cells are arranged in a rectangular grid.

## cbc package structure

The library itself consists of two modules and an utility package:

- beam.py - a module with incoming beam classes
- lattice.py - a module with sample lattice classes
- wrapper.py - simulation calculation wrapper module
- utils - utility package:
    - utilfuncs.py - utility functions for convergent beam diffraction project
    - asf - atomic scattering factor calculation package:
        - load.py - a module, that loads atomic scattering factor fit coefficients
    - pdb - PDB data import package
        - readpdb.py - a module, that imports molecular structure .pdb files

Also there is a couple of usage examples:

- diff-run.py - conducting diffraction simulation and saving results to HDF5 file
- diff-slides.py - calculating and saving diffraction pattern series
- read.py - reading and ploting diffraction results
- test.py - testing diffraction simulation

## upcoming things to do

- consider different crystall space groups
