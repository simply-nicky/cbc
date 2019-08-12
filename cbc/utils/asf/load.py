import os, numpy as np

path = os.path.dirname(__file__)
asf_henke = np.load(os.path.join(path, 'asf_henke.npy'), encoding='latin1', allow_pickle=True).item()
asf_waskif = np.load(os.path.join(path, 'asf_waskif.npy'), encoding='latin1', allow_pickle=True).item()