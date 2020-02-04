"""
test.py - cbc and cbc_dp packages testing script
"""
import os
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6

from timeit import default_timer as timer
import argparse
import pickle
import numpy as np
import pygmo
import h5py
import cbc_dp

PIX_SIZE = 75 * 1e-3 #mm
WL = 7.293188082141599e-08 #mm
ROT_AX = np.array([0, 1, 0])

B12_PREFIX = 'b12_2'
B12_NUM = 135
B12_DET_POS = np.array([115.3, 129.5, 107.9]) #mm
B12_EXP = cbc_dp.ScanSetup(rot_axis=ROT_AX,
                           pix_size=PIX_SIZE,
                           det_pos=B12_DET_POS)
B12_MASK = np.load('cbc_dp/utils/b12_mask.npy')
B12_PUPIL = 0.9 * np.radians([0.74, 1.6])

def save_exp_data(prefix, scan_num, mask=None, good_frames=None):
    scan = cbc_dp.open_scan(prefix, scan_num, good_frames)
    scan.save_corrected(mask)

def get_exp_data(prefix, scan_num, mask=None, good_frames=None):
    scan = cbc_dp.open_scan(prefix, scan_num, good_frames)
    return scan.corrected_data(mask)

def detect_scan(strks_data, exp_set, scale=0.6, sigma_scale=0.4, log_eps=0):
    lsd = cbc_dp.LineSegmentDetector(scale=scale, sigma_scale=sigma_scale, log_eps=log_eps)
    return lsd.det_scan(strks_data, exp_set)

def get_refine(det_scan, exp_set, rec_basis, num_ap,
               tol=(0.05, 0.12), n_isl=20, pop_size=50, gen_num=2000):
    archi = pygmo.archipelago()
    for idx, frame in enumerate(det_scan):
        frame_basis = rec_basis.dot(exp_set.rotation_matrix(-np.radians(idx)).T)
        full_tf = cbc_dp.FCBI(lines=frame.raw_lines,
                              num_ap=num_ap,
                              exp_set=exp_set,
                              rec_basis=frame_basis,
                              tol=tol)
        prob = pygmo.problem(full_tf)
        populations = [pygmo.population(size=pop_size, prob=prob, b=pygmo.mp_bfe()) for _ in range(n_isl)]
        for pop in populations:
            archi.push_back(algo=pygmo.de(gen_num), pop=pop)
    return archi

def main(prefix, scan_num, exp_set, num_ap, mask, tol, n_isl):
    data_path = os.path.join(os.path.dirname(__file__),
                             "exp_results/scan_{0:05d}".format(scan_num),
                             cbc_dp.utils.FILENAME['scan'].format('corrected', scan_num, 'h5'))
    scan_path = os.path.join(os.path.dirname(__file__),
                             "exp_results/b12_scan.p")
    print("Looking for scan file: {}".format(scan_path))
    if not os.path.exists(scan_path):
        print("Scan file doesn't exist, looking for data file: {}".format(scan_path))
        if not os.path.exists(data_path):
            print('Data file has not been found, generating the file...')
            save_exp_data(prefix, scan_num, mask)
        print("Opening the data file...")
        data_file = h5py.File(data_path, 'r')
        strks_data = data_file['corrected_data/streaks_data'][:]
        print("Detecting diffraction streaks...")
        det_scan = detect_scan(strks_data, exp_set)
    else:
        det_scan = pickle.load(open(scan_path, 'rb'))
    print("{:d} streaks detected in total".format(det_scan.size))
    print("Projecting streaks to reciprocal space...")
    scan_qs = det_scan.kout_ref(theta=np.radians(np.arange(len(det_scan.shapes))))
    rec_basis = scan_qs.index()
    print("The Diffraction data successfully autoindexed, reciprocal basis:\n{:}".format(rec_basis))
    print("Setting up the indexing solution refinement...")
    archi = get_refine(det_scan, exp_set, rec_basis, num_ap, tol, n_isl)
    print("Starting indexing solution refinement")
    start = timer()
    archi.evolve()
    archi.wait()
    print("The refinement has been completed, elapsed time: {:f}s".format(timer() - start))
    index_sol = archi.get_champions_x()
    index_f = archi.get_champions_f()
    out_file = h5py.File('exp_results/index_sol.h5', 'w')
    out_file.create_dataset('data/index_sol', data=index_sol)
    out_file.create_dataset('data/index_f', data=index_f)
    print("The refined solutions have been saved, file: {}".format(out_file.filename))
    out_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Index b12 diffraction data')
    parser.add_argument('n_isl', type=int, help='Number of islands for one frame')
    parser.add_argument('--tol', type=float, nargs=2, default=[0.05, 0.12], help='Refinement tolerance: det_pos, rec_basis')
    args = parser.parse_args()

    main(B12_PREFIX, B12_NUM, B12_EXP, B12_PUPIL, B12_MASK, args.tol, args.n_isl)
