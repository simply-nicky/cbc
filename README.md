# Convergent beam crystallography project
This project consists of two packages - convergent beam simulation program ([cbc](#1.-convergent-beam-simulation)) and experimental data processing package ([cbc_dp](#2.-convergent-crystallography-data-processing))

## 1. Convergent beam simulation
Simulation for convergent beam diffration on a crystalline sample written in Python. Code is compatible with Python 2.X and 3.X.

For more information regarding the theory behind this see [the article written by Prof. Henry Chapman](https://e-reports-ext.llnl.gov/pdf/314988.pdf).

### Compilation

It's a package called cbc, with which you can conduct a diffraction simulation. See usage examples: [go to cbc package structure](#cbc-package-structure)

Required dependencies:

- NumPy
- Numba
- SciPy
- matplotlib
- h5py

### Features

The package can perform convergent beam diffraction simulation of crystalline samples based on first Born approximation theory.

Available incoming beam models:

- Gaussian beam
- Bessel beam
- Lens beam with rectangular or circular aperture

Samples could be composed of different compound unit cells deffined by array of atom coordinates within the unit cell and corresponding B-factors. These data can be imported via PDB format file. Unit cells are arranged in a rectangular grid.

### cbc package structure

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

- read.py - reading and ploting diffraction results
- diff_run.py - diffraction simulation example

### Upcoming things to do

- consider different crystall space groups

## 2. Convergent crystallography data processing
Experimental convergent diffraction data processing package. Data is acquired from PETRA P06 beamtime on June 13, 2019.

### Data correction:

    - Flatfield correction
    - Median filtering
    - Non maximum supression

### Line detection

    - Progressive Probabilistic Hough line transform
    - Line Segment Detector (LSD)

More information about Progressive Hough transform is written in [this article](https://ieeexplore.ieee.org/document/786993) and about Line Segment Detector in [this article](https://www.ipol.im/pub/art/2012/gjmr-lsd/).

### Indexing:

    - Rotational diffraction pattern indexing
    - Convergent beam indexer