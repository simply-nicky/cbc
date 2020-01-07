from timeit import default_timer as timer
import numpy as np
from scipy import constants
import cbc
import cbc_dp

WL = constants.h * constants.c / 17000 / constants.e * 1e3
NUM_AP = np.sqrt(2.5e5**2 + 3e5**2) * WL
FOCUS = 1.
PIX_SIZE = 75 * 1e-3
DET_NX, DET_NY, DET_DIST = 2000, 2000, 100
REC_BASIS = 1.4 * np.array([[2e5, 0, 0],
                            [0, 2.5e5, 0],
                            [0, 0, 3e5]])
REC_BASIS = REC_BASIS.dot(cbc_dp.utils.rotation_matrix(np.array([1, 0, 0]), np.radians(20)).T)
REC_BASIS = REC_BASIS.dot(cbc_dp.utils.rotation_matrix(np.array([0, 1, 0]), np.radians(7)).T)
LAT_R = 4e-5

def main(rec_basis=REC_BASIS, lat_r=LAT_R, num_ap=NUM_AP, focus=FOCUS, wavelength=WL,
         det_dist=DET_DIST, det_nx=DET_NX, det_ny=DET_NY, pix_size=PIX_SIZE):
    basis = cbc.rec_basis(rec_basis)
    delta_z = lat_r / np.arctan(num_ap) / 2
    detector = cbc.Detector(det_dist=det_dist, det_nx=det_nx, det_ny=det_ny, pix_size=pix_size)
    beam = cbc.CircLens(focus=focus, aperture=2 * focus * num_ap, wavelength=wavelength)
    lattice = cbc.BallLattice(basis_a=basis[0], basis_b=basis[1], basis_c=basis[2], lat_r=lat_r)
    diff = cbc.DiffYar(beam=beam, detector=detector, lattice=lattice)
    diff.move_lat([0, 0, delta_z])
    start = timer()
    diffres = diff.calculate().pool()
    diffres.write()
    print('Estimated time: %fs' % (timer() - start))

if __name__ == "__main__":
    main()
