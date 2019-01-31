"""
File: utilfuncs.py (Python 2.X and 3.X)

Utility functions for convergent beam diffraction project.
"""
import os
from timeit import default_timer as timer
import matplotlib.pyplot as plt

def verbose_call(v, func, *args):
    """
    Call function func with given arguments args verbose if v is True and silent if V is False.

    v - verbosity flag
    func - a function to call
    args - tuple of arguments for func

    Returns results of function func.
    """
    if v:
        print('Parsing argument(s):\n' + '\n'.join((str(arg) for arg in args)))
        start = timer()
        res = func(*args)
        print('%s is done\nEstimated time: %f' % (func.__name__, (timer() - start)))
        return res
    else:
        return func(*args)

def search_rec(path, ext='hdf5', filelist=None):
    """
    Search recursively in sub folders of given path for files with extension ext.

    path - a path to search
    ext - file extension to search

    Returns list of paths.
    """
    if not os.path.isdir(path): 
        raise  ValueError("the path is invalid")
    if filelist == None:
        filelist = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith(ext):
                filelist.append(os.path.join(root, f))
    return filelist

class AxesSeq(object):
    """
    Creates a series of axes in a figure where only one is displayed at any given time. Which plot is displayed is controlled by the arrow keys.
    """
    def __init__(self, size):
        self.fig = plt.figure()
        self.axes = [self.fig.add_subplot(1,1,1, label=i, visible=False) for i in range(size)]
        self.index = 0
        self.fig.canvas.mpl_connect('key_press_event', self.on_keypress)

    def __iter__(self):
        return iter(self.axes)

    def on_keypress(self, event):
        if event.key == 'right':
            self.next_plot()
        elif event.key == 'left':
            self.prev_plot()
        else:
            return
        self.fig.canvas.draw()

    def next_plot(self):
        if self.index < len(self.axes) - 1:
            self.axes[self.index].set_visible(False)
            self.axes[self.index+1].set_visible(True)
            self.index += 1

    def prev_plot(self):
        if self.index > 0:
            self.axes[self.index].set_visible(False)
            self.axes[self.index-1].set_visible(True)
            self.index -= 1

    def show(self):
        self.axes[self.index].set_visible(True)
        plt.show()