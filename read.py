"""
File: read.py (Python 2.X and 3.X)

An example of reading and ploting diffraction results.
Can be used as a script in terminal.
"""

import cbc, h5py, os
import numpy as np
import matplotlib.pyplot as plt

def read(filename):
    """
    Read HDF5 data file and return it's name and data.

    filename - name of the HDF5 file
    """
    if not os.path.isfile(os.path.abspath(filename)): 
        raise  ValueError("the file doesn't exist")
    f = h5py.File(filename, 'r')
    data = []
    results = f[list(f)[1]]
    for key in results.keys():
        data.append(results[key][:])
    f.close()
    return filename, data

def savefig(reslist):
    """
    Safe figures as an eps vector graphics files from list of (filename, data) tuples.

    reslist - list of results data
    """
    fig, ax = plt.subplots()
    for filename, data in reslist:
        ax.clear()
        path = os.path.splitext(filename)[0]    
        ax.contourf(data[1], data[2], np.abs(data[0]))
        ax.set_title(filename)
        fig.canvas.draw()
        fig.savefig(path + '.eps', format='eps')
    
def show(reslist):
    """
    Show figures in one window, you can navigate between different figures by right/left arrow keys.

    reslist - list of (filename, data) tuples
    """
    axes = cbc.utils.AxesSeq(len(reslist))
    for ax, res in zip(axes, reslist):
        filename, data = res
        ax.contourf(data[1], data[2], np.abs(data[0]))
        ax.set_title(filename)
    axes.show()

def read_all(path):
    """
    Read all HDF5 data files in all subfolders and files in given path.

    path - a path to search for results
    """
    paths = cbc.utils.search_rec(path, 'hdf5')
    paths.sort()
    return [read(path) for path in paths]

if __name__ == "__main__":
    import sys, argparse
    parser = argparse.ArgumentParser(description='Read diffraction results HDF5 files')
    parser.add_argument('path', type=str, help='path to the folder or a file to read')
    parser.add_argument('action', type=str, choices=['show', 'save'], help='choose between show the diffraction pattern or save the figure in a file')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    if args.all:
        results = cbc.utils.verbose_call(args.verbose, read_all, args.path)
        cbc.utils.verbose_call(args.verbose, show, results) if args.action == 'show' else cbc.utils.verbose_call(args.verbose, savefig, results)
    else:
        results = []
        results.append(cbc.utils.verbose_call(args.verbose, read, args.path))
        cbc.utils.verbose_call(args.verbose, show, results) if args.action == 'show' else cbc.utils.verbose_call(args.verbose, savefig, results)
