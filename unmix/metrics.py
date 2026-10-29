"""Unmixing metrics. Conventions follow the two Dong et al. (2025) papers:

- aRMSE: mean over endmembers of the per-endmember abundance RMSE (GAT-CNN-HU eq. 29); the
  per-material values in the papers' tables are the per-endmember RMSEs.
- RMSE_all: sqrt of the mean squared error over all abundance entries (the INSPECTION.md number).
- mSAD: mean spectral angle (radians) between matched endmembers.
- mAAD: mean angle (radians) between matched abundance maps (one vector of N pixels per material).
Blind estimates are matched to the reference by the Hungarian algorithm on SAD (or on abundance RMSE
when no endmembers are estimated by the method).
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def _angle(a, b, axis=0):
    c = (a * b).sum(axis) / (np.linalg.norm(a, axis=axis) * np.linalg.norm(b, axis=axis) + 1e-12)
    return np.arccos(np.clip(c, -1, 1))


def sad_matrix(M_ref, M_est):
    """(R_ref, R_est) spectral angles."""
    return np.array([[_angle(M_ref[:, i], M_est[:, j]) for j in range(M_est.shape[1])]
                     for i in range(M_ref.shape[1])])


def match(M_ref, A_ref, M_est=None, A_est=None):
    """Permutation p so that estimate p[k] corresponds to reference k."""
    if M_est is not None:
        cost = sad_matrix(M_ref, M_est)
    else:
        cost = np.array([[np.sqrt(((A_ref[i] - A_est[j]) ** 2).mean()) for j in range(A_est.shape[0])]
                         for i in range(A_ref.shape[0])])
    _, p = linear_sum_assignment(cost)
    return p


def evaluate(Y, A_ref, A_est, M_ref=None, M_est=None, labels=None, permute=True, Y_hat=None):
    """Return a flat dict of metrics. A_*: (R, N); M_*: (L, R). Y_hat defaults to M_est @ A_est."""
    if permute:
        p = match(M_ref, A_ref, M_est, A_est) if M_est is not None else match(M_ref, A_ref, None, A_est)
        A_est = A_est[p]
        M_est = M_est[:, p] if M_est is not None else None
    R = A_ref.shape[0]
    labels = labels or [str(k) for k in range(R)]
    rmse_k = np.sqrt(((A_ref - A_est) ** 2).mean(1))
    aad_k = np.array([_angle(A_ref[k], A_est[k]) for k in range(R)])
    out = {"aRMSE": rmse_k.mean(), "RMSE_all": np.sqrt(((A_ref - A_est) ** 2).mean()), "mAAD": aad_k.mean()}
    out.update({f"RMSE_{labels[k]}": rmse_k[k] for k in range(R)})
    if M_est is not None:
        sad_k = np.array([_angle(M_ref[:, k], M_est[:, k]) for k in range(R)])
        out["mSAD"] = sad_k.mean()
        out.update({f"SAD_{labels[k]}": sad_k[k] for k in range(R)})
    if Y_hat is None and M_est is not None:
        Y_hat = M_est @ A_est
    if Y_hat is not None:
        out["recon_RMSE"] = np.sqrt(((Y - Y_hat) ** 2).mean())
        out["recon_SAD"] = _angle(Y, Y_hat, axis=0).mean()
    return out
