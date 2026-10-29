"""Step 1 benchmark: classical baselines and deep autoencoders on Samson and Jasper Ridge.

Usage: python run_benchmark.py --stage classical|deep [--datasets samson jasper] [--seeds 10]
Writes one row per (dataset, method, setting, seed, abundance variant) to results/<stage>.csv.

Abundance variants for blind methods:
  own      the method's abundances as produced
  aligned  the same solution expressed with peak-normalized endmembers (a_k * max(e_k), renormalized).
           The reference abundances of both datasets use this convention (see FINDINGS.md); it uses no
           ground-truth information, only the dataset's convention for endmember scale.
"""
import argparse
import csv
import itertools
import time
from pathlib import Path

import numpy as np

from unmix.data import load
from unmix.classical import fcls, fcls_tv, l12_nmf, nnls_norm, vca
from unmix.metrics import evaluate

OUT = Path(__file__).resolve().parent / "results"


def align(A, M):
    """Re-express abundances for peak-normalized endmembers. M (L, R) or per-pixel (N, L, R)."""
    peak = M.max(0)[:, None] if M.ndim == 2 else M.max(1).T
    W = A * peak
    return W / W.sum(0, keepdims=True)


def rows_for(d, method, setting, seed, A, M_est=None, Y_hat=None, t=0.0, permute=True, variants=True):
    out = []
    if Y_hat is None and M_est is not None and M_est.ndim == 2:
        Y_hat = M_est @ A                          # the solution's own reconstruction, for every variant
    todo = [("own", A)] + ([("aligned", align(A, M_est))] if variants and M_est is not None else [])
    for var, Av in todo:
        r = evaluate(d["Y"], d["A"], Av, d["M"], M_est if M_est is None or M_est.ndim == 2 else M_est.mean(0),
                     d["labels"], permute=permute, Y_hat=Y_hat)
        out.append(dict(dataset=d["name"], method=method, setting=setting, seed=seed, variant=var,
                        seconds=round(t, 2), **{k: float(v) for k, v in r.items()}))
    return out


def classical(d, seeds):
    Y, M, R = d["Y"], d["M"], d["A"].shape[0]
    Mp = M / M.max(0)
    rows = []
    rows += rows_for(d, "uniform", "-", 0, np.full_like(d["A"], 1 / R), permute=False)
    rows += rows_for(d, "FCLS[ref M]", "-", 0, fcls(Y, M), M, permute=False, variants=False)
    rows += rows_for(d, "NNLS-norm[ref M]", "-", 0, nnls_norm(Y, Mp), Mp, permute=False, variants=False)
    for lam in [1e-2, 1e-1, 5e-1]:
        t = time.time()
        rows += rows_for(d, "FCLS-TV[ref M]", f"lam={lam:g}", 0, fcls_tv(Y, M, d["shape"], lam), M,
                         t=time.time() - t, permute=False, variants=False)
    for s in range(seeds):
        Me = vca(Y, R, np.random.default_rng(s))[0]
        Me = np.maximum(Me, 1e-6)
        rows += rows_for(d, "VCA+FCLS", "-", s, fcls(Y, Me), Me)
        Mn = Me / Me.max(0)
        rows += rows_for(d, "VCA+NNLS-norm", "-", s, nnls_norm(Y, Mn), Mn, variants=False)
        for lam in [1e-2, 1e-1, 5e-1]:
            rows += rows_for(d, "VCA+FCLS-TV", f"lam={lam:g}", s, fcls_tv(Y, Me, d["shape"], lam), Me)
        A0 = fcls(Y, Me)
        for lam in [0.0, 0.01, 0.1]:
            t = time.time()
            Mn_, An_ = l12_nmf(Y, Me, A0, lam, iters=1000)
            rows += rows_for(d, "L1/2-NMF", f"lam={lam:g}", s, An_, Mn_, t=time.time() - t)
    return rows


def deep(d, seeds, which, on_result=None):
    import torch
    from unmix import deep as D
    torch.set_num_threads(4)
    Y, R, L = d["Y"], d["A"].shape[0], d["Y"].shape[0]
    rows = []

    def run(method, setting, seed, build, loss_fn, epochs, lr, wd=0.0, labels=None, adj=None, init="vca",
            clip_grad=None):
        torch.manual_seed(seed)
        Me = np.maximum(vca(Y, R, np.random.default_rng(seed))[0], 1e-6) if init == "vca" else d["M"].copy()
        ctx = D.context(Y, d["shape"], labels, adj)
        model = build(Me)
        t = time.time()
        A, Yh = D.train(model, ctx, loss_fn, epochs, lr, wd, seed, clip_grad=clip_grad)
        if hasattr(model, "dec"):
            Mest = model.dec.E.detach().numpy().astype(np.float64)
        else:
            Mest = model.endmembers().detach().numpy().astype(np.float64)[labels]   # (N, L, R)
        r = rows_for(d, method, setting, seed, A, Mest, Yh, time.time() - t)
        print(f"{d['name']:6s} {method:18s} {setting:24s} seed {seed}: "
              + " | ".join(f"{x['variant']} aRMSE {x['aRMSE']:.4f} mSAD {x.get('mSAD', np.nan):.4f}" for x in r),
              flush=True)
        if on_result is not None:
            on_result(r)
        return r

    n_sp = 200
    for seed in range(seeds):
        labels = D.superpixels(Y, d["shape"], n_sp)
        _, Oh = D.assignment(labels)
        adj_knn = D.knn_adjacency((Oh.T @ Y.T), 8)
        if "pixelae" in which:
            rows += run("PixelAE", "SAD,lr=1e-3,ep=1000", seed, lambda Me: D.PixelAE(L, R, Me),
                        lambda A, Yh, m: D.sad_loss(torch.tensor(Y.T, dtype=torch.float32), Yh), 1000, 1e-3)
        Yt = torch.tensor(Y.T, dtype=torch.float32)
        paper_loss = lambda A, Yh, m: D.sad_loss(Yt, Yh) + 0.3 * 0.5 * ((Yh - Yt) ** 2).sum(1).mean() + 0.05 * D.l12(A)
        variants = {"gatcnn": ("GAT-CNN-HU", {}), "gatcnn-nogat": ("GAT-CNN-HU w/o GAT", {"use_gat": False}),
                    "gatcnn-noaacm": ("GAT-CNN-HU w/o AACM", {"aacm": False})}
        for key, (name, kw) in variants.items():
            if key not in which:
                continue
            rows += run(name, "paper,lr=1e-2,ep=500", seed,
                        lambda Me, kw=kw: D.GATCNN(L, R, Me, n_sp, **kw), paper_loss, 500, 1e-2,
                        labels=labels, adj=adj_knn)
        if "mgalite" in which:
            lab50 = D.superpixels(Y, d["shape"], 50)
            _, Oh50 = D.assignment(lab50)
            S = lab50.max() + 1
            maxlen = np.bincount(lab50).max()
            groups = -np.ones((S, maxlen), dtype=np.int64)
            for s_ in range(S):
                idx = np.where(lab50 == s_)[0]
                groups[s_, :len(idx)] = idx
            adj_sp = D.spatial_adjacency(lab50, d["shape"])

            def mga_loss(A, Yh, m, lam=0.01, beta=0.001, mu=10.0):
                g = m.groups
                nuc = sum(torch.linalg.matrix_norm(A[g[s_][g[s_] >= 0]], ord="nuc") for s_ in range(g.shape[0]))
                var = ((m.dE - m.dE.mean(0, keepdim=True)) ** 2).sum(1).mean()
                return D.sad_loss(Yt, Yh) + lam * D.l12(A) + beta * nuc / A.shape[0] + mu * var
            rows += run("MGA-lite", "lam=.01,beta=1e-3,lr=3e-3,ep=400", seed,
                        lambda Me: D.MGALite(L, R, Me, S, torch.tensor(groups)), mga_loss, 400, 3e-3, 1e-5,
                        labels=lab50, adj=adj_sp)
        # AE-SU (Ampas et al., GRSL 2025), notebook settings and variants; "ref" = initialized from the
        # reference endmembers (the letter initializes from field-measured endmembers)
        aesu_runs = {
            "aesu":        ("AE-SU", "notebook,lr=1e-5,ep=200", "ref", 1e-5, {}),
            "aesu-vca":    ("AE-SU", "notebook,lr=1e-5,ep=200,VCA init", "vca", 1e-5, {}),
            "aesu-lr":     ("AE-SU", "lr=1e-3,ep=200", "ref", 1e-3, {}),
            "aesu-linear": ("AE-SU w/o refine", "lr=1e-3,ep=200", "ref", 1e-3, {"refine": False}),
        }
        for key, (name, setting, init, lr, kw) in aesu_runs.items():
            if key not in which:
                continue
            rows += run(name, setting, seed, lambda Me, kw=kw: D.AESU(L, R, Me, **kw),
                        lambda A, Yh, m: D.aesu_loss(Yt, Yh, A, 0.01), 200, lr, init=init, clip_grad=1.0)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["classical", "deep"], required=True)
    ap.add_argument("--datasets", nargs="+", default=["samson", "jasper"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--models", nargs="+", default=["pixelae", "gatcnn", "gatcnn-nogat", "gatcnn-noaacm", "mgalite"])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{a.stage}{a.tag}.csv"
    rows = []

    def save(new=()):
        rows.extend(new)
        keys = sorted({k for r in rows for k in r},
                      key=lambda k: (k not in ("dataset", "method", "setting", "seed", "variant"), k))
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    for name in a.datasets:
        d = load(name)
        if a.stage == "classical":
            save(classical(d, a.seeds))
        else:
            deep(d, a.seeds, a.models, on_result=save)      # saved after every run
    print("wrote", path)
