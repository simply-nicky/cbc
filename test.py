import numpy as np
import cbc_dp

or_mat = np.array([[-0.00955958, 0.04569071, 0.00127158],
                   [-0.03160975, -0.00589558, -0.00016407],
                   [-0.00153314, -0.00164234, 0.02641596]])
center = np.zeros(3)

rec_lat = cbc_dp.RecLattice(or_mat=or_mat, center=center)
rec_lat.vectors(0.1)