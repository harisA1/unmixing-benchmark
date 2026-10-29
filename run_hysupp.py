"""Run HySUPP models (Rasti et al. 2023, github.com/inria-thoth/HySUPP) on our data and protocol.

The models are called directly (bypassing HySUPP's hydra/MLXP launcher) with their default
config/model/*.yaml parameters, on our loader's data (global scaling only, no projection, no L2
normalization: HySUPP's defaults). Scored with our metrics: Hungarian matching on SAD, abundances
reported as produced ("own") and in the peak-normalized endmember convention ("aligned").

HySUPP reshapes pixel vectors in C order; our vectors are MATLAB (F) order, so its spatial models see
the transposed image. Both scenes are square, so neighbourhoods are unchanged.

Usage: python run_hysupp.py [--models CNNAEU MiSiCNet ...] [--datasets samson jasper] [--seeds 1]
Needs HySUPP cloned at $HYSUPP_DIR (default /home/user/inria-thoth/hysupp).
Writes results/hysupp.csv.
"""
import argparse
import csv
import importlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from run_benchmark import rows_for
from unmix.data import load

HYSUPP = Path(os.environ.get("HYSUPP_DIR", "/home/user/inria-thoth/hysupp"))
OUT = Path(__file__).resolve().parent / "results"

BLIND = {"EDAA": "EDAA", "AA": "ArchetypalAnalysis", "CNNAEU": "CNNAEU", "MiSiCNet": "MiSiCNet",
         "PGMSU": "PGMSU", "MSNet": "MSNet", "ADMMNet": "ADMMNet"}
SUPERVISED = {"UnDIP": "UnDIP"}


def build(name):
    cfg = yaml.safe_load(open(HYSUPP / "config" / "model" / f"{name}.yaml"))
    for k, v in cfg.items():            # PyYAML reads "1e-3" as a string; hydra/OmegaConf reads a float
        if isinstance(v, str):
            try:
                cfg[k] = float(v)
            except ValueError:
                pass
    target = cfg.pop("name")
    mod, cls = target.rsplit(".", 1)
    return getattr(importlib.import_module(mod), cls)(**cfg), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(BLIND) + list(SUPERVISED))
    ap.add_argument("--datasets", nargs="+", default=["samson", "jasper"])
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    sys.path.insert(0, str(HYSUPP))
    OUT.mkdir(exist_ok=True)
    path = OUT / f"hysupp{a.tag}.csv"
    rows = []
    for ds in a.datasets:
        d = load(ds)
        Y, R = d["Y"], d["A"].shape[0]
        H, W = d["shape"]
        for name in a.models:
            for seed in range(a.seeds):
                np.random.seed(seed)
                import torch
                torch.manual_seed(seed)
                model, cfg = build(name)
                t = time.time()
                try:
                    if name in SUPERVISED:      # known endmembers: reference, peak-normalized (as run_spatial.py)
                        Mest = d["M"] / d["M"].max(0)
                        A = np.asarray(model.compute_abundances(Y, Mest, H=H, W=W, seed=seed))
                        r = rows_for(d, f"HySUPP {name} [ref M]", "default", seed, A, Mest, permute=False,
                                     t=time.time() - t, variants=False)
                    else:
                        E, A = model.compute_endmembers_and_abundances(Y, R, H=H, W=W, seed=seed)
                        E, A = np.asarray(E, dtype=np.float64), np.asarray(A, dtype=np.float64)
                        r = rows_for(d, f"HySUPP {name}", "default", seed, A, np.maximum(E, 1e-12),
                                     t=time.time() - t)
                except Exception as e:                # report and continue with the next model
                    print(f"{ds} {name} seed {seed}: FAILED {type(e).__name__}: {e}", flush=True)
                    continue
                rows += r
                print(f"{ds:6s} {name:10s} seed {seed} ({time.time() - t:6.0f}s): "
                      + " | ".join(f"{x['variant']} aRMSE {x['aRMSE']:.4f} mSAD {x.get('mSAD', float('nan')):.4f}"
                                   for x in r), flush=True)
                keys = sorted({k for x in rows for k in x},
                              key=lambda k: (k not in ("dataset", "method", "setting", "seed", "variant"), k))
                with open(path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=keys)
                    w.writeheader(); w.writerows(rows)
    print("wrote", path)


if __name__ == "__main__":
    main()
