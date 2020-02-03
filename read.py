"""
File: read.py (Python 2.X and 3.X)

An example of reading and ploting diffraction results.
Can be used as a script in terminal.
"""
import os
import argparse
from timeit import default_timer as timer
import h5py
import numpy as np
import matplotlib.pyplot as plt
import cbc

def read(filename):
    """
    Read HDF5 data file and return it's name and data.

    filename - name of the HDF5 file
    """
    if not os.path.isfile(os.path.abspath(filename)): 
        raise  ValueError("the file doesn't exist")
    _file = h5py.File(filename, 'r')
    data = []
    data_dict = _file[list(_file)[1]]
    for key in data_dict.keys():
        data.append(data_dict[key][:])
    _file.close()
    return filename, data

def save_fig(res_list):
    """
    Safe figures as an eps vector graphics files from list of (filename, data) tuples.

    res_list - list of results data
    """
    fig, ax = plt.subplots()
    for filename, data in res_list:
        ax.clear()
        path = os.path.splitext(filename)[0]
        ints = np.abs(data[0])   
        ax.imshow(data[1], data[2], ints, cmap='viridis', vmin=ints.min(), vmax=ints.max())
        ax.set_title(filename)
        fig.canvas.draw()
        fig.savefig(path + '.eps', format='eps')

def show(res_list):
    """
    Show figures in one window, you can navigate between different figures by right/left arrow keys.

    res_list - list of (filename, data) tuples
    """
    axes = cbc.utils.AxesSeq(res_list)
    axes.show()

def read_all(path):
    """
    Read all HDF5 data files in all subfolders and files in given path.

    path - a path to search for results
    """
    paths = cbc.utils.search_rec(path, 'hdf5')
    paths.sort()
    return [read(path) for path in paths]

def verbose_return(verbosity, func, *args):
    """
    Call function func with given arguments args verbose if v is True and silent if V is False
    and return function result.

    verbosity - verbosity flag
    func - a function to call
    args - tuple of arguments for func

    Returns results of function func.
    """
    if verbosity:
        print('Parsing argument(s):', *args, sep='\n')
        start = timer()
        res = func(*args)
        print('%s is done\nEstimated time: %f' % (func.__name__, (timer() - start)))
        return res
    else:
        return func(*args)

def verbose_call(verbosity, func, *args):
    """
    Call function func with given arguments args verbose if v is True and silent if V is False.

    verbosity - verbosity flag
    func - a function to call
    args - tuple of arguments for func

    Returns results of function func.
    """
    if verbosity:
        print('Parsing argument(s):', *args, sep='\n')
        start = timer()
        res = func(*args)
        print('%s is done\nEstimated time: %f' % (func.__name__, (timer() - start)))
    else:
        func(*args)

def main():
    parser = argparse.ArgumentParser(description='Read diffraction results HDF5 files')
    parser.add_argument('path', type=str, help='path to the folder or a file to read')
    parser.add_argument('action',
                        type=str,
                        choices=['show', 'save'],
                        help='choose between show the diffraction pattern or save the figure in a file')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    if args.all:
        results = verbose_return(args.verbose, read_all, args.path)
        if args.action == 'show':
            verbose_call(args.verbose, show, results)
        else:
            verbose_call(args.verbose, save_fig, results)
    else:
        results = []
        results.append(verbose_return(args.verbose, read, args.path))
        if args.action == 'show':
            verbose_call(args.verbose, show, results)
        else:
            verbose_call(args.verbose, save_fig, results)

if __name__ == "__main__":
    main()
