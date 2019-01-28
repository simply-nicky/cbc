"""
File: read.py (Python 2.X and 3.X)

An example of reading and ploting diffraction results
"""

import cbc, h5py
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

f = h5py.File("results/diff_24-01-2019_06-03.hdf5", 'r')
data = []
results = f[list(f)[1]]
for key in results.keys():
    data.append(results[key][:])
f.close()
plt.contourf(data[1], data[2], np.abs(data[0]))
plt.savefig('results/diff_res.eps', format='eps')
plt.show()