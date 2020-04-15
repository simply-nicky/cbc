"""
run_index.py - script to run indexing refinement
"""
import os

os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6

from timeit import default_timer as timer
import argparse
import numpy as np
import pygmo
import h5py
import configparser
from ..utils import OUT_PATH, FILENAME
from ..feat_detect import ScanSetup
from ..indexer import ScanStreaks, ScanCBI

def import_rb(rb_file):
    """
    Import reciprocal lattice basis vectors from an ini file

    rb_file - path to a file
    """
    config = configparser.ConfigParser()
    config.read(rb_file)
    ax = config.getfloat('rec_basis', 'ax')
    ay = config.getfloat('rec_basis', 'ay')
    az = config.getfloat('rec_basis', 'az')
    bx = config.getfloat('rec_basis', 'bx')
    by = config.getfloat('rec_basis', 'by')
    bz = config.getfloat('rec_basis', 'bz')
    cx = config.getfloat('rec_basis', 'cx')
    cy = config.getfloat('rec_basis', 'cy')
    cz = config.getfloat('rec_basis', 'cz')
    return np.array([[ax, ay, az], [bx, by, bz], [cx, cy, cz]])

def open_scan(scan_num, exp_set):
    """
    Open experimentally measured scan saved in exp_results folder

    scan_num - scan number
    exp_set - ScanStreaks class object
    """
    data_path = os.path.join(os.path.dirname(__file__),
                             OUT_PATH['scan'].format(scan_num),
                             FILENAME['scan'].format('streaks', scan_num, 'h5'))
    print("Looking for data file: {}".format(data_path))

    if not os.path.exists(data_path):
        raise ValueError("Data doesn't exist at the following path: {}".format(data_path))

    print("Opening the data file...")
    with h5py.File(data_path, 'r') as data_file:
        det_scan = ScanStreaks(raw_lines=data_file['streaks/lines'][:],
                               exp_set=exp_set,
                               frame_idxs=data_file['streaks/frame_idxs'][:])
    print("{:d} streaks detected in total".format(det_scan.size))
    return det_scan

def write_data(index_sol, index_pts, out_path):
    """
    Write refinement solutions to an hdf5 file

    index_sol - an array of refinement solutions
    index_pts - an array of refinement points
    """
    out_file = h5py.File(os.path.join('exp_results', out_path), 'w')
    out_file.create_dataset('data/index_sol', data=index_sol)
    out_file.create_dataset('data/index_pts', data=index_pts)
    print("The refined solutions have been saved, file: {}".format(out_file.filename))
    out_file.close()

def rot_index(scan, rec_basis, n_isl, pop_size, gen_num, pos_tol, rb_tol, ang_tol):
    """
    Conduct indexing solution framewise rotational refinement

    scan - ScanStreaks class object
    rec_basis - reciprocal lattice basis vectors matrix
    n_isl - number of islands for every frame
    pop_size - population size
    gen_num - maximal number of generations
    [pos_tol, rb_tol, ang_tol] - refinement tolerances
    """
    scan_size = scan.frames.size

    print("Setting up the indexing solution refinement...")
    archi = scan.rot_index(rec_basis=rec_basis, n_isl=n_isl, pop_size=pop_size,
                           gen_num=gen_num, pos_tol=pos_tol, rb_tol=rb_tol,
                           ang_tol=ang_tol)
    print("Starting indexing solution refinement...")
    start = timer()
    archi.evolve()
    archi.wait()
    print("The refinement has been completed, elapsed time: {:f}s".format(timer() - start))
    index_sol = np.array(archi.get_champions_x()).reshape((n_isl, scan_size, -1), order='F')
    index_pts = np.array(archi.get_champions_f()).reshape((n_isl, scan_size), order='F')
    return index_sol, index_pts

def full_index(scan, rec_basis, n_isl, pop_size, gen_num, pos_tol, rb_tol, ang_tol):
    """
    Conduct indexing solution framewise full refinement

    scan - ScanStreaks class object
    rec_basis - reciprocal lattice basis vectors matrix
    n_isl - number of islands for every frame
    pop_size - population size
    gen_num - maximal number of generations
    [pos_tol, rb_tol, ang_tol] - refinement tolerances
    """
    scan_size = scan.frames.size

    print("Setting up the indexing solution refinement...")
    archi = scan.full_index(rec_basis=rec_basis, n_isl=n_isl, pop_size=pop_size,
                            gen_num=gen_num, pos_tol=pos_tol, rb_tol=rb_tol,
                            ang_tol=ang_tol)
    print("Starting indexing solution refinement")
    start = timer()
    archi.evolve()
    archi.wait()
    print("The refinement has been completed, elapsed time: {:f}s".format(timer() - start))
    index_sol = np.array(archi.get_champions_x()).reshape((n_isl, scan_size, -1), order='F')
    index_pts = np.array(archi.get_champions_f()).reshape((n_isl, scan_size), order='F')
    return index_sol, index_pts

def scan_index(scan, rec_basis, n_isl, pop_size, gen_num, pos_tol, rb_tol, ang_tol):
    """
    Conduct indexing solution whole scan refinement

    scan - ScanStreaks class object
    rec_basis - reciprocal lattice basis vectors matrix
    n_isl - number of islands for every frame
    pop_size - population size
    gen_num - maximal number of generations
    [pos_tol, rb_tol, ang_tol] - refinement tolerances
    """
    print("Setting up the indexing solution refinement...")
    prob = ScanCBI(streaks=scan[10::10], rec_basis=rec_basis,
                   tol=(pos_tol, rb_tol, ang_tol), pen_coeff=1.)
    pops = [pygmo.population(prob, size=pop_size, b=pygmo.mp_bfe()) for _ in range(n_isl)]
    archi = pygmo.archipelago()
    algo = pygmo.moead(gen=gen_num)
    for pop in pops:
        archi.push_back(algo=algo, pop=pop)
    print("Starting indexing solution refinement")
    start = timer()
    archi.evolve()
    archi.wait()
    print("The refinement has been completed, elapsed time: {:f}s".format(timer() - start))
    ev_pops = [island.get_population() for island in archi]
    index_sol = np.stack([pop.get_x() for pop in ev_pops])
    index_pts = np.stack([pop.get_f() for pop in ev_pops])
    return index_sol, index_pts

def main():
    """
    Main fucntion to run indexing refinement
    """
    parser = argparse.ArgumentParser(description='Run CBC indexing refinement')
    parser.add_argument('geom_file', type=str, help='Path to a geometry ini file')
    parser.add_argument('rb_file', type=str, help='Path to a reciprocal lattice basis vectors ini file')
    parser.add_argument('mode', type=str, choices=['rot', 'full', 'scan'],
                        help='Choose between rotation and full indexing refinement')
    parser.add_argument('out_path', type=str, help='Output file path')
    parser.add_argument('--scan_num', type=int, default=135, help='Scan number')
    parser.add_argument('--pop_size', type=int, default=50,
                        help='Population size of the refinement islands')
    parser.add_argument('--n_isl', type=int, default=16, help='Number of islands for one frame')
    parser.add_argument('--gen_num', type=int, default=3000,
                        help='Maximum generations number of the refinement algorithm')
    parser.add_argument('--pos_tol', type=float, nargs=3, default=[0.02, 0.02, 0.075],
                        help='Relative sample position tolerance')
    parser.add_argument('--rb_tol', type=float, default=0.01,
                        help='Lattice basis vectors length tolerance')
    parser.add_argument('--ang_tol', type=float, default=0.1, help='Rotation anlges tolerance')

    args = parser.parse_args()
    rec_basis = import_rb(args.rb_file)
    scan = open_scan(scan_num=args.scan_num, exp_set=ScanSetup.import_ini(args.geom_file))
    if args.mode == 'rot':
        index_sol, index_pts = rot_index(scan=scan, pop_size=args.pop_size,
                                         n_isl=args.n_isl, rec_basis=rec_basis,
                                         gen_num=args.gen_num, pos_tol=args.pos_tol,
                                         rb_tol=args.rb_tol, ang_tol=args.ang_tol)
    elif args.mode == 'full':
        index_sol, index_pts = full_index(scan=scan, pop_size=args.pop_size,
                                          n_isl=args.n_isl, rec_basis=rec_basis,
                                          gen_num=args.gen_num, pos_tol=args.pos_tol,
                                          rb_tol=args.rb_tol, ang_tol=args.ang_tol)
    else:
        index_sol, index_pts = scan_index(scan=scan, pop_size=args.pop_size,
                                          n_isl=args.n_isl, rec_basis=rec_basis,
                                          gen_num=args.gen_num, pos_tol=args.pos_tol,
                                          rb_tol=args.rb_tol, ang_tol=args.ang_tol)
    write_data(index_sol, index_pts, args.out_path)
    