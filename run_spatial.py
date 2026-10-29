"""Spatial unmixing under noise, known endmembers (quick, single seed).

Every method gets the same endmembers: the reference endmembers, peak-normalized (the reference
convention). Noise: white Gaussian, sigma^2 = mean(Y^2) / 10^(SNR/10), fixed seed. The target is always
the clean-data reference abundances, so spatial pooling can help only by removing noise.

Usage: python run_spatial.py [--snr inf 30 20 10] [--seed 0] [--epochs 300]
Writes results/spatial_noise.csv.
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

from unmix.classical import fcls, fcls_tv, nnls_norm
from unmix.compunmix import fit, grid_neighbours
from unmix.data import from_image, load, to_image
from unmix.metrics import evaluate

OUT = Path(__file__).resolve().parent / "results"


def add_noise(Y, snr, rng):
    if np.isinf(snr):
        return Y.copy()
    return Y + rng.normal(0, np.sqrt((Y ** 2).mean() / 10 ** (snr / 10)), Y.shape)


def edge_aware(A, Y, shape):
    """Non-learned analogue of the message passing: average each pixel with its 8 neighbours, weights
    exp(-SAD / median neighbour SAD), self weight 1, then renormalize."""
    nb, ok = grid_neighbours(shape)
    Yn = Y / np.linalg.norm(Y, axis=0)
    s = np.arccos(np.clip((Yn.T[:, None, :] * Yn.T[nb]).sum(-1), -1, 1))
    w = np.exp(-s / np.median(s[ok])) * ok
    out = (A + (w[None] * A[:, nb]).sum(-1)) / (1 + w.sum(1))
    return out / out.sum(0, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr", nargs="+", type=float, default=[np.inf, 30, 20, 10])
    ap.add_argument("--datasets", nargs="+", default=["samson", "jasper"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=300)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    path = OUT / "spatial_noise.csv"
    rows = []

    def record(d, snr, method, A, t, extra=None):
        r = evaluate(d["Y"], d["A"], A, permute=False)
        row = dict(dataset=d["name"], snr=snr, method=method, seed=a.seed, seconds=round(t, 1),
                   **{k: float(v) for k, v in r.items()}, **(extra or {}))
        rows.append(row)
        print(f"{d['name']:6s} SNR {snr:>4} {method:30s} aRMSE {row['aRMSE']:.4f} {extra or ''}", flush=True)
        with open(path, "w", newline="") as f:
            keys = sorted({k for x in rows for k in x}, key=lambda k: (k not in ("dataset", "snr", "method", "seed"), k))
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(rows)

    for name in a.datasets:
        d = load(name)
        E = d["M"] / d["M"].max(0)
        for snr in a.snr:
            Yn = add_noise(d["Y"], snr, np.random.default_rng(a.seed))
            t = time.time(); A0 = nnls_norm(Yn, E); t0 = time.time() - t
            record(d, snr, "FCLS (per pixel)", fcls(Yn, E), 0)
            record(d, snr, "NNLS-norm (per pixel)", A0, t0)
            Ys = from_image(uniform_filter(to_image(Yn, d["shape"]), size=(1, 3, 3)))
            record(d, snr, "NNLS-norm on 3x3-mean Y", nnls_norm(Ys, E), 0)
            A3 = from_image(uniform_filter(to_image(A0, d["shape"]), size=(1, 3, 3)))
            record(d, snr, "NNLS-norm + 3x3 mean of A", A3 / A3.sum(0), 0)
            record(d, snr, "NNLS-norm + edge-aware smoothing", edge_aware(A0, Yn, d["shape"]), 0)
            for lam in [0.01, 0.1, 0.5]:
                record(d, snr, f"FCLS-TV lam={lam:g}", fcls_tv(Yn, E, d["shape"], lam), 0)
            for geo, label in [("fisher-rao", "CompUnmix B (Fisher-Rao)"), ("aitchison", "CompUnmix C (Aitchison)")]:
                t = time.time()
                A, gates, loss = fit(Yn, E, d["shape"], geo, epochs=a.epochs, seed=a.seed)
                record(d, snr, label, A, time.time() - t,
                       {"gates": " ".join(f"{g:.3f}" for g in gates), "heldout_SAD": round(loss, 5)})
    print("wrote", path)


if __name__ == "__main__":
    main()
