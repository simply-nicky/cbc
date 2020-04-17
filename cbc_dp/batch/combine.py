"""
combine.py - python script to combine HDF5 files into one
"""
import os
import argparse
import h5py
import numpy as np

def make_path(path, i=0):
    """
    Return a nonexistant path to write a file
    """
    filename, ext = os.path.splitext(path)
    new_path = filename + "_{:d}".format(i) + ext
    if not os.path.isfile(path):
        return path
    elif os.path.isfile(new_path):
        return make_path(path, i + 1)
    else:
        return new_path

class HDF_Data(object):
    """
    Extract HDF5 file data recursively with the structure preserved

    hdf_file - h5py File object
    """
    def __init__(self, hdf_file):
        self.data = {}
        hdf_file.visititems(self._append_data)

    def _append_data(self, path, node):
        if isinstance(node, h5py.Dataset):
            self.data[path] = node[...]

    def keys(self):
        """
        Return names
        """
        return self.data.keys()

    def items(self):
        """
        Get (name, item) pairs
        """
        return self.data.items()

    def get(self, name):
        """
        Retrieve an item
        """
        return self.data.get(name)

def write_data(files, out_path):
    """
    Write a file with all data combined

    files - list of data files to combine
    out_path - output file name
    """
    print("Reading data files: {}".format(files))
    data_list = []
    for path in files:
        with h5py.File(path, 'r') as data_file:
            data_list.append(HDF_Data(data_file))
    print("Writing data to the file: {}".format(out_path))
    print("Datasets' names:\n{:s}".format('\n'.join(list(data_list[0].keys()))))
    out_path = make_path(out_path)
    with h5py.File(out_path, 'w') as out_file:
        for key in data_list[0].keys():
            if key.startswith('data'):
                dataset = np.concatenate([data.get(key) for data in data_list])
                out_file.create_dataset(key, data=dataset)
            elif key.startswith('config'):
                out_file.create_dataset(key, data=data_list[0].get(key))
            else:
                err_txt = "HDF5 dataset name {:s} doesn't start with 'data' or 'config'".format(key)
                raise RuntimeError(err_txt)
    print("Deleting old files...")
    # for filename in files:
    #     os.remove(filename)
    print('Done!')

def main():
    """
    Main function to combine the data
    """
    parser = argparse.ArgumentParser(description='Combine data')
    parser.add_argument('files', nargs='+', type=str, help='Data files to combine')
    parser.add_argument('out_path', type=str, help='Output file name')
    args = parser.parse_args()

    write_data(files=args.files, out_path=args.out_path)

if __name__ == "__main__":
    main()
