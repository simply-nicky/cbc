"""
File: read.py (Python 2.X and 3.X)

An example of reading and ploting diffraction results.
Can be used as a script in terminal.
"""

import cbc, h5py, os
import numpy as np
import matplotlib.pyplot as plt

def read_diff(filename):
    path = os.path.splitext(filename)[0]
    f = h5py.File(filename, 'r')
    data = []
    results = f[list(f)[1]]
    for key in results.keys():
        data.append(results[key][:])
    f.close()
    plt.contourf(data[1], data[2], np.abs(data[0]))
    plt.savefig(path + '.eps', format='eps')
    plt.show()

if __name__ == "__main__":
    import sys, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='the path to the diffraction results HDF5 file')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    
    if args.verbose:
        print('Reading the file %s' % args.path)
        read_diff(args.path)
        print('Done!')
    else:
        read_diff(args.path)