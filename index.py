"""
index.py - indexing refinement script
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
import configparser
import h5py
import cbc_dp

class ConfigParser():
    """
    Class parser for ini files
    config_file - path to a config file
    exp_geom_file - path to a experimental geometry config file
    """
    def __init__(self, config_file, exp_geom_file):
        self.config = configparser.ConfigParser()
        self.config.read([os.path.abspath(config_file),
                          os.path.abspath(exp_geom_file)])
        self._init_exp_set()
        self._init_param()

    def _init_exp_set(self):
        pix_size = self.config.getfloat('exp_geom', 'pix_size')
        smp_pos = np.array([float(coord) for coord in self.config.get('exp_geom', 'sample_pos').split()])
        z_f = self.config.getfloat('exp_geom', 'focus_z')
        pupil = np.array([int(bound) for bound in self.config.get('exp_geom', 'pupil_bounds').split()]).reshape((2, 2))
        beam_pos = np.array([int(coord) for coord in self.config.get('exp_geom', 'beam_pos').split()])
        axis = np.array([float(coord) for coord in self.config.get('exp_geom', 'rot_axis').split()])
        th_min = self.config.getfloat('exp_geom', 'theta_min')
        th_max = self.config.getfloat('exp_geom', 'theta_max')
        self.exp_set = cbc_dp.ScanSetup(pix_size=pix_size, smp_pos=smp_pos, z_f=z_f,
                                        pupil=pupil, beam_pos=beam_pos, axis=axis,
                                        thetas=np.radians(np.arange(th_min, th_max)))

    def _init_param(self):
        self.mode = self.config.get('config', 'mode')
        self.out_path = self.config.get('config', 'out_path')
        self.scan_num = self.config.getint('config', 'scan_num')
        self.pop_size = self.config.getint('config', 'pop_size')
        self.n_isl = self.config.getint('config', 'n_islands')
        self.gen_num = self.config.getint('config', 'gen_number')
        self.pos_tol = np.array([float(tol) for tol in self.config.get('config', 'pos_tol').split()])
        self.size_tol = self.config.getfloat('config', 'size_tol')
        self.ang_tol = self.config.getfloat('config', 'ang_tol')
        self.rec_basis = np.array([float(coord) for coord in self.config.get('config', 'rec_basis').split()]).reshape((3, 3))

def open_scan(scan_num, exp_set):
    """
    Open experimentally measured scan saved in exp_results folder
    """
    data_path = os.path.join(os.path.dirname(__file__),
                             "exp_results/scan_{0:05d}".format(scan_num),
                             cbc_dp.utils.FILENAME['scan'].format('streaks', scan_num, 'h5'))
    print("Looking for data file: {}".format(data_path))

    if not os.path.exists(data_path):
        raise ValueError("Data doesn't exist at the following path: {}".format(data_path))

    print("Opening the data file...")
    with h5py.File(data_path, 'r') as data_file:
        det_scan = cbc_dp.ScanStreaks(raw_lines=data_file['streaks/lines'][:],
                                      exp_set=exp_set,
                                      frame_idxs=data_file['streaks/frame_idxs'][:])
    print("{:d} streaks detected in total".format(det_scan.size))
    return det_scan

def write_data(index_sol, index_f, out_path):
    """
    Write refinement solutions to an hdf5 file
    """
    out_file = h5py.File(os.path.join('exp_results', out_path), 'w')
    out_file.create_dataset('data/index_sol', data=index_sol)
    out_file.create_dataset('data/index_f', data=index_f)
    print("The refined solutions have been saved, file: {}".format(out_file.filename))
    out_file.close()

def rot_index(scan, rec_basis, n_isl, pop_size, gen_num, pos_tol, size_tol, ang_tol):
    """
    Conduct indexing solution framewise rotational refinement
    """
    scan_size = scan.frames.size

    print("Setting up the indexing solution refinement...")
    archi = scan.rot_index_refine(rec_basis=rec_basis, n_isl=n_isl, pop_size=pop_size,
                                  gen_num=gen_num, pos_tol=pos_tol, size_tol=size_tol,
                                  ang_tol=ang_tol)
    print("Starting indexing solution refinement")
    start = timer()
    archi.evolve()
    archi.wait()
    print("The refinement has been completed, elapsed time: {:f}s".format(timer() - start))
    index_sol = np.array(archi.get_champions_x()).reshape((n_isl, scan_size, -1), order='F')
    index_f = np.array(archi.get_champions_f()).reshape((n_isl, scan_size), order='F')
    return index_sol, index_f

def full_index(scan, rec_basis, n_isl, pop_size, gen_num, pos_tol, size_tol, ang_tol):
    """
    Conduct indexing solution framewise full refinement
    """
    scan_size = scan.frames.size

    print("Setting up the indexing solution refinement...")
    archi = scan.full_index_refine(rec_basis=rec_basis, n_isl=n_isl, pop_size=pop_size,
                                   gen_num=gen_num, pos_tol=pos_tol, size_tol=size_tol,
                                   ang_tol=ang_tol)
    print("Starting indexing solution refinement")
    start = timer()
    archi.evolve()
    archi.wait()
    print("The refinement has been completed, elapsed time: {:f}s".format(timer() - start))
    index_sol = np.array(archi.get_champions_x()).reshape((n_isl, scan_size, -1), order='F')
    index_f = np.array(archi.get_champions_f()).reshape((n_isl, scan_size), order='F')
    return index_sol, index_f

def scan_index(scan, rec_basis, n_isl, pop_size, gen_num, pos_tol, size_tol, ang_tol):
    """
    Conduct indexing solution whole scan refinement
    """
    print("Setting up the indexing solution refinement...")
    prob = cbc_dp.ScanCBI(streaks=scan[10::10], rec_basis=rec_basis,
                          tol=(pos_tol, size_tol, ang_tol), pen_coeff=1.)
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
    index_f = np.stack([pop.get_f() for pop in ev_pops])
    return index_sol, index_f

def run_index(config):
    """
    Run indexing solution refinement
    """
    scan = open_scan(scan_num=config.scan_num, exp_set=config.exp_set)
    if config.mode == 'rot':
        index_sol, index_f = rot_index(scan=scan, pop_size=config.pop_size,
                                       n_isl=config.n_isl, rec_basis=config.rec_basis,
                                       gen_num=config.gen_num, pos_tol=config.pos_tol,
                                       size_tol=config.size_tol, ang_tol=config.ang_tol)
    elif config.mode == 'full':
        index_sol, index_f = full_index(scan=scan, pop_size=config.pop_size,
                                        n_isl=config.n_isl, rec_basis=config.rec_basis,
                                        gen_num=config.gen_num, pos_tol=config.pos_tol,
                                        size_tol=config.size_tol, ang_tol=config.ang_tol)
    else:
        index_sol, index_f = scan_index(scan=scan, pop_size=config.pop_size,
                                        n_isl=config.n_isl, rec_basis=config.rec_basis,
                                        gen_num=config.gen_num, pos_tol=config.pos_tol,
                                        size_tol=config.size_tol, ang_tol=config.ang_tol)
    write_data(index_sol, index_f, config.out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Indexing refinement script')
    parser.add_argument('config_file', type=str, help='path to an ini config file')
    parser.add_argument('exp_geom_file', type=str, help='path to a experimental geometry config file')
    args = parser.parse_args()
    run_index(ConfigParser(args.config_file, args.exp_geom_file))
    