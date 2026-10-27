"""How were the reference abundances made? Compare them with simple unmixings of the data using the
reference endmembers. Prints aRMSE (mean over materials of per-material RMSE) for each recipe.

Result (27 Oct 2026): for both Samson and Jasper Ridge the reference abundances are, up to ~0.004
aRMSE, NNLS against the peak-normalized reference endmembers followed by division by the sum. FCLS
(the usual "physical" definition, sum-to-one inside the solver) is far from the reference.
"""
import numpy as np

from unmix.classical import fcls, nnls_norm
from unmix.data import load
from unmix.metrics import evaluate

for name in ["samson", "jasper"]:
    d = load(name)
    Y, M, A = d["Y"], d["M"], d["A"]
    peak = M / M.max(0)
    recipes = {
        "FCLS, M as given": fcls(Y, M),
        "FCLS, peak-normalized M": fcls(Y, peak),
        "NNLS + normalize, M as given": nnls_norm(Y, M),
        "NNLS + normalize, L2-normalized M": nnls_norm(Y, M / np.linalg.norm(M, axis=0)),
        "NNLS + normalize, peak-normalized M": nnls_norm(Y, peak),
    }
    print(f"\n{name}: reference M column peaks {np.round(M.max(0), 3)}, reference A zeros {np.mean(A == 0):.0%}")
    for k, Ak in recipes.items():
        r = evaluate(Y, A, Ak, permute=False)
        print(f"  {k:38s} aRMSE {r['aRMSE']:.4f}  mAAD {r['mAAD']:.4f}  same zero pattern {np.mean((Ak < 1e-9) == (A < 1e-9)):.3f}")
