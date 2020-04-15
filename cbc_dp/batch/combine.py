"""
combine.py - python script to combine HDF5 files into one
"""
import argparse
import h5py
import numpy as np

class HDF_Data(object):
    def __init__(self, hdf_file):
        self.data = {}
        hdf_file.visititems(self._append_data)

    def _append_data(self, path, node):
        if isinstance(node, h5py.Dataset):
            self.data[path] = node[:]

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

def write_data(filenames, out_filename):
    """
    Write a file with all data combined

    filenames - list of data files to combine
    out_filename - output file name
    """
    print("Reading data files: {}".format(filenames))
    data_list = []
    for filename in filenames:
        with h5py.File(filename, 'r') as data_file:
            data_list.append(HDF_Data(data_file))
    print("Writing data to the file: {}".format(out_filename))
    with h5py.File(out_filename, 'w') as out_file:
        for key in data_list[0].keys():
            dataset = np.concatenate([data[key] for data in data_list])
            out_file.create_dataset(key, data=dataset)
    print("Done!")

def main():
    """
    Main function to combine the data
    """
    parser = argparse.ArgumentParser(description='Combine data')
    parser.add_argument('filenames', nargs='+', type=str, help='Data files to combine')
    parser.add_argument('out_filename', type=str, help='Output file name')
    args = parser.parse_args()

    write_data(filenames=args.filenames, out_filename=args.out_filename)

if __name__ == "__main__":
    main()