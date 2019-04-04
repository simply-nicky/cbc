import numpy as np, os

def importpdb(filename):
    at = "ATOM"
    dir = os.path.dirname(__file__)
    xs, ys, zs, bs, els = [], [], [], [], []
    for line in open(os.path.join(dir, filename)):
        if line.startswith(at):
            xs.append(float(line[31:38]))
            ys.append(float(line[39:46]))
            zs.append(float(line[47:54]))
            bs.append(float(line[61:66]))
            els.append(line[77:78])
    return np.array(xs) * 1e-7, np.array(ys) * 1e-7, np.array(zs) * 1e-7, np.array(bs), els

def savepdb(filename):
    at = "ATOM"
    dir = os.path.dirname(__file__)
    xs, ys, zs, bs, els = [], [], [], [], []
    for line in open(os.path.join(dir, filename)):
        if line.startswith(at):
            xs.append(float(line[31:38]))
            ys.append(float(line[39:46]))
            zs.append(float(line[47:54]))
            bs.append(float(line[61:66]))
            els.append(line[77:78])
    res = np.array([xs * 1e-7, ys * 1e-7, zs * 1e-7, bs, els])
    np.save(os.path.splitext(filename)[0], res)

if __name__ == "__main__":
    xs, ys, zs, bs, els = importpdb('4et8.pdb')
    print(els)