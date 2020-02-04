"""
test.py - cbc and cbc_dp packages testing script
"""
import os
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
               tol=(0.1, 0.1), arch_size=20, pop_size=36, gen_num=2000):
    archs = []
    for idx, frame in enumerate(det_scan):
        frame_basis = rec_basis.dot(exp_set.rotation_matrix(-np.radians(idx)).T)
        full_tf = cbc_dp.FCBI(lines=frame.raw_lines,
                              num_ap=num_ap,
                              exp_set=exp_set,
                              rec_basis=frame_basis,
                              tol=tol)
        prob = pygmo.problem(full_tf)
        arch = pygmo.archipelago(arch_size, algo=pygmo.de(gen_num), prob=prob, pop_size=pop_size)
        archs.append(arch)
    return archs

def main(prefix, scan_num, exp_set, num_ap, mask):
    out_path = os.path.join(os.path.dirname(__file__),
                            cbc_dp.utils.OUT_PATH['scan'].format(scan_num),
                            cbc_dp.utils.FILENAME['scan'].format('corrected', scan_num, 'h5'))
    if not os.path.exists(out_path):
        save_exp_data(prefix, scan_num, mask)
    scan_file = h5py.File(out_path, 'r')
    strks_data = scan_file['corrected_data/streaks_data'][:]
    det_scan = detect_scan(strks_data, exp_set)
    scan_qs = det_scan.kout_ref(theta=np.radians(np.arange(strks_data.shape[0])))
    rec_basis = scan_qs.index()
    archs = get_refine(det_scan, exp_set, rec_basis, num_ap)
    for arch in archs:
        arch.evolve()
    for arch in archs:
        arch.wait()
    index_sol = np.stack([arch.get_champion_x() for arch in archs], axis=-1)
    np.save('exp_results/index_solutions.npy', index_sol)

if __name__ == "__main__":
    main(B12_PREFIX, B12_NUM, B12_EXP, B12_PUPIL, B12_MASK)
