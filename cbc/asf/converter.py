#!/usr/bin/env python
"""
File: converter.py

Converting atomic scattering data module.
"""
import numpy as np
import os

def asf_converter(filename):
    if not os.path.isfile(os.path.abspath(filename)): 
        raise  ValueError("the file doesn't exist")
    title = []
    coeffs = []
    for line in open(filename):
        parts = line.split()
        try:
            coeffs.append([float(parts[0]), float(parts[1])])
        except:
            title.append(line)
            continue
    name, ext = os.path.splitext(filename)
    new_file = open(name + '_new' + ext, 'w')
    new_file.writelines(title)
    new_file.write('\n'.join(('%E\t%E' % tuple(coeff) for coeff in coeffs)))
    new_file.close()

def asf_fit_converter(filename):
    if not os.path.isfile(os.path.abspath(filename)):
        raise ValueError("the file doesn't exist")
    title = []
    for line in open(filename):
        parts = line.split()
        try:
            coeffs = [float(part) for part in parts]
        except:
            title.append(line)
            continue
    name, ext = os.path.splitext(filename)
    new_file = open(name + '_new' + ext, 'w')
    new_file.writelines(title)
    new_file.write('\t'.join(('%E' % coeff for coeff in coeffs)))
    new_file.close()

def verbose_call(v, func, *args):
    if v:
        print('Reading file(s) ' + ' '.join(args))
        func(*args)
        print('Done!')
    else:
        func(*args)

if __name__ == "__main__":
    import sys, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("type", type=str, choices=['asf', 'asf_fit'], help="choose between given converters")
    parser.add_argument("path", type=str, help="the path to the asf file")
    parser.add_argument("-v", "--verbosity", action="store_true")
    args = parser.parse_args()
    if args.type == 'asf':
        verbose_call(args.verbosity, asf_converter, args.path)
    elif args.type == 'asf_fit':
        verbose_call(args.verbosity, asf_fit_converter, args.path)
    else:
        raise ValueError('wrong type argument')