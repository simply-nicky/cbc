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
import numpy as np
import h5py
import cbc_dp

PIX_SIZE = 75 * 1e-3 #mm
WL = 7.293188082141599e-08 #mm
ROT_AX = np.array([0, 1, 0])

B12_PREFIX = 'b12_2'
B12_NUM = 135
B12_DET_POS = np.array([115.27911592, 128.57856985, 100.41825068]) #mm
B12_EXP = cbc_dp.ScanSetup(rot_axis=ROT_AX,
                           pix_size=PIX_SIZE,
                           det_pos=B12_DET_POS)
B12_PUPIL = np.radians([0.65, 1.05])
REC_BASIS = np.array([[0.00906475, -0.04583905, -0.00082416],
                      [0.03227241, 0.00576972, 0.00194436],
                      [0.0016247, 0.00172209, -0.02941539]])

def main(out_path, prefix, scan_num, rec_basis, exp_set, num_ap, n_isl, pop_size, gen_num, pos_tol, size_tol, ang_tol):
    data_path = os.path.join(os.path.dirname(__file__),
                             "exp_results/scan_{0:05d}".format(scan_num),
                             cbc_dp.utils.FILENAME['scan'].format('streaks', scan_num, 'h5'))
    print("Looking for data file: {}".format(data_path))

    if not os.path.exists(data_path):
        raise ValueError("Data doesn't exist at the following path: {}".format(data_path))

    print("Opening the data file...")
    data_file = h5py.File(data_path, 'r')
    det_scan = cbc_dp.ScanStreaks(raw_lines=data_file['streaks/lines'][:],
                                  exp_set=exp_set,
                                  frame_idxs=data_file['streaks/frame_idxs'][:])
    frame_num = len(np.unique(det_scan.frame_idxs))
    print("{:d} streaks detected in total".format(det_scan.size))

    print("Setting up the indexing solution refinement...")
    archi = det_scan.rot_index_refine(theta=np.radians(np.arange(det_scan.size)), rec_basis=rec_basis,
                                      num_ap=num_ap, n_isl=n_isl, pop_size=pop_size, gen_num=gen_num,
                                      pos_tol=pos_tol, size_tol=size_tol, ang_tol=ang_tol)
    print("Starting indexing solution refinement")
    start = timer()
    archi.evolve()
    archi.wait()
    print("The refinement has been completed, elapsed time: {:f}s".format(timer() - start))
    index_sol = np.array(archi.get_champions_x()).reshape((n_isl, frame_num, -1), order='F')
    index_f = np.array(archi.get_champions_f()).reshape((n_isl, frame_num), order='F')
    out_file = h5py.File(os.path.join('exp_results', out_path + '.h5'), 'w')
    out_file.create_dataset('data/index_sol', data=index_sol)
    out_file.create_dataset('data/index_f', data=index_f)
    print("The refined solutions have been saved, file: {}".format(out_file.filename))
    out_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Index b12 diffraction data')
    parser.add_argument('out_path', type=str, help='Output file path')
    parser.add_argument('--n_isl', type=int, default=20, help='Number of islands for one frame')
    parser.add_argument('--pos_tol', type=float, nargs=3, default=[0.007, 0.014, 0.06], help='Relative sample position tolerance')
    parser.add_argument('--size_tol', type=float, default=0.03, help='Lattice basis vectors length tolerance')
    parser.add_argument('--ang_tol', type=float, default=1.5, help='Rotation anlges tolerance')
    parser.add_argument('--gen_num', type=int, default=3000, help='Maximum generations number of the refinement algorithm')
    parser.add_argument('--pop_size', type=int, default=50, help='Population size of the refinement islands')
    args = parser.parse_args()

    main(out_path=args.out_path, prefix=B12_PREFIX, scan_num=B12_NUM, exp_set=B12_EXP,
         num_ap=B12_PUPIL, rec_basis=REC_BASIS, n_isl=args.n_isl, pop_size=args.pop_size,
         gen_num=args.gen_num, pos_tol=args.pos_tol, size_tol=args.size_tol,
         ang_tol=args.ang_tol)
