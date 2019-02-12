import os, numpy as np

path = os.path.dirname(__file__)
asf_henke = np.load(os.path.join(path, 'asf_henke.npy')).item()
asf_waskif = np.load(os.path.join(path, 'asf_waskif.npy')).item()