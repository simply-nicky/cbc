import numpy as np, os

def readpdb(filename):
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
    return xs, ys, zs, bs, els

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
    res = np.array([xs, ys, zs, bs, els])
    np.save(os.path.splitext(filename)[0], res)

if __name__ == "__main__":
    xs, ys, zs, bs, els = readpdb('4et8.pdb')
    print(els)