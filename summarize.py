"""Summarize results/*.csv: mean ± sd over seeds per (dataset, method, setting, variant), plus the
"oracle" (best mean aRMSE) and "honest" (best mean reconstruction SAD, no ground truth) setting choice."""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

COLS = ["aRMSE", "mSAD", "mAAD", "recon_SAD"]


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            rows += list(csv.DictReader(f))
    return rows


def main(paths):
    rows = load(paths)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r["method"], r["variant"], r["setting"])].append(r)
    stats = {}
    for k, rs in groups.items():
        stats[k] = {c: [float(r[c]) for r in rs if r.get(c) not in (None, "")] for c in COLS}
    fmt = lambda v: f"{np.mean(v):.4f} ± {np.std(v):.4f}" if v else "—"
    for ds in sorted({k[0] for k in stats}):
        print(f"\n### {ds}\n")
        print("| method | variant | setting | n | aRMSE | mSAD | mAAD | recon SAD |")
        print("|---|---|---|---|---|---|---|---|")
        for k in sorted(k for k in stats if k[0] == ds):
            s = stats[k]
            print(f"| {k[1]} | {k[2]} | {k[3]} | {len(s['aRMSE'])} | " + " | ".join(fmt(s[c]) for c in COLS) + " |")
        print("\nSelection over settings (oracle = best mean aRMSE, honest = best mean recon SAD):\n")
        print("| method | variant | oracle setting | oracle aRMSE | honest setting | honest aRMSE |")
        print("|---|---|---|---|---|---|")
        for m, v in sorted({(k[1], k[2]) for k in stats if k[0] == ds}):
            ks = [k for k in stats if k[0] == ds and k[1] == m and k[2] == v]
            o = min(ks, key=lambda k: np.mean(stats[k]["aRMSE"]))
            h = min(ks, key=lambda k: np.mean(stats[k]["recon_SAD"]) if stats[k]["recon_SAD"] else np.inf)
            print(f"| {m} | {v} | {o[3]} | {fmt(stats[o]['aRMSE'])} | {h[3]} | {fmt(stats[h]['aRMSE'])} |")


if __name__ == "__main__":
    main(sys.argv[1:] or sorted(Path("results").glob("*.csv")))
