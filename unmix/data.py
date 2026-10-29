"""Dataset loader: Samson and Jasper Ridge with physical (global) scaling and checksum checks.

Pixel order: the .mat files are MATLAB column-major; images are (rows, cols) with order="F".
Returned arrays: Y (L, N) data, M (L, R) reference endmembers, A (R, N) reference abundances.
"""
import hashlib
from pathlib import Path

import numpy as np
import scipy.io as sio

DATA = Path(__file__).resolve().parent.parent / "data"

SHA256 = {
    "jasperRidge2_R198.mat": "0e4118a6452f6044978a8ca3762fb0f791115467904936d463c4e111e56e682e",
    "gt_jasper_end4.mat": "92f5697b43705802b904fd13ba99b6ce65a3d203682864abc3fbec922beec374",
    "samson_1.mat": "1ebacaf7cd32bfc31c0ee3fd56c63a7f29a434893a8b45705fb59cd8a0c8beb6",
    "gt_samson_end3.mat": "ce768182ce5292390180f827ff5769112cf3d299f4e64ab43e950a310b7ec9fa",
}


def _load(name):
    path = DATA / name
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256[name]:
        raise ValueError(f"{name}: checksum {digest} does not match the expected file")
    return sio.loadmat(path)


def _labels(cood):
    return [str(c[0][0]).split("-", 1)[1].lower() for c in cood]


def load(name):
    """Return dict with Y, M, A, shape (rows, cols), labels."""
    if name == "jasper":
        d, g = _load("jasperRidge2_R198.mat"), _load("gt_jasper_end4.mat")
        Y = d["Y"].astype(np.float64) / float(d["maxValue"][0, 0])     # one global factor
        shape = (int(d["nRow"][0, 0]), int(d["nCol"][0, 0]))
    elif name == "samson":
        d, g = _load("samson_1.mat"), _load("gt_samson_end3.mat")
        Y = d["V"].astype(np.float64)                                   # already globally scaled to [0, 1]
        shape = (int(d["nRow"][0, 0]), int(d["nCol"][0, 0]))
    else:
        raise ValueError(name)
    M, A = g["M"].astype(np.float64), g["A"].astype(np.float64)
    assert Y.shape == (M.shape[0], A.shape[1]) and A.shape[1] == shape[0] * shape[1]
    return dict(name=name, Y=Y, M=M, A=A, shape=shape, labels=_labels(g["cood"]))


def to_image(X, shape):
    """(C, N) -> (C, rows, cols) in MATLAB order."""
    return X.reshape(X.shape[0], *shape, order="F")


def from_image(X):
    """(C, rows, cols) -> (C, N) in MATLAB order."""
    return X.reshape(X.shape[0], -1, order="F")


def grid_edges(shape, eight=False):
    """Undirected 4- (or 8-) neighbour pixel edges in MATLAB pixel order, shape (E, 2)."""
    idx = np.arange(shape[0] * shape[1]).reshape(shape, order="F")
    pairs = [(idx[1:, :], idx[:-1, :]), (idx[:, 1:], idx[:, :-1])]
    if eight:
        pairs += [(idx[1:, 1:], idx[:-1, :-1]), (idx[1:, :-1], idx[:-1, 1:])]
    return np.concatenate([np.stack([a.ravel(), b.ravel()], 1) for a, b in pairs])
