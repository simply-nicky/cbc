"""
File: converter.py

Converting atomic scattering data module.
"""
import numpy as np
import os

def asf_converter(filename):
    assert os.path.isfile(os.path.abspath(filename)), "the file doesn't exist"
    title = []
    coeffs = []
    for line in open(filename):
        parts = line.split()
        try:
            coeffs.append([float(parts[0]), float(parts[1])])
        except:
            title.append(line)
            continue
    name, file_ext = os.path.splitext(filename)
    new_file = open(name + '_new' + file_ext, 'w')
    new_file.writelines(title)
    new_file.write('\n'.join(('%E\t%E' % tuple(coeff) for coeff in coeffs)))

if __name__ == "__main__":
    import sys, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="a path to the asf file")
    parser.add_argument("-v", "--verbosity", action="store_true")
    
    if args.