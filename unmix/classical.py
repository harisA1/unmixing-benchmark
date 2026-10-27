"""Classical unmixing baselines (numpy only).

- fcls: exact fully constrained least squares (non-negativity + sum-to-one inside the solver), by
  enumerating supports; exact and fast for the small numbers of endmembers used here (R <= 6).
- vca: Vertex Component Analysis (Nascimento & Bioucas-Dias 2005).
- fcls_tv: SUnSAL-TV (Iordache, Bioucas-Dias & Plaza 2012) with the endmember matrix as library and the
  sum-to-one constraint. With ASC the l1 term of SUnSAL is constant, so only the anisotropic TV weight
  matters. Solved by ADMM; the A-step is exact (eigenbasis of M'M x DCT basis of the Neumann Laplacian).
- l12_nmf: L1/2-sparsity-constrained NMF (Qian et al. 2011), multiplicative updates with the
  sum-to-one row trick, initialized by VCA + FCLS.
"""
import itertools

import numpy as np
from scipy.fft import dctn, idctn


# ------------------------------------------------------------------ FCLS
def project_simplex(X):
    """Euclidean projection of each column of X (R, N) onto the probability simplex."""
    R = X.shape[0]
    U = -np.sort(-X, axis=0)
    css = np.cumsum(U, axis=0) - 1
    ind = np.arange(1, R + 1)[:, None]
    cond = U - css / ind > 0
    rho = R - 1 - np.argmax(cond[::-1], axis=0)
    theta = css[rho, np.arange(X.shape[1])] / (rho + 1)
    return np.maximum(X - theta, 0)


def fcls(Y, M):
    """argmin ||y - M a||^2 s.t. a >= 0, 1'a = 1, for every column of Y. Returns A (R, N)."""
    L, R = M.shape
    N = Y.shape[1]
    best = np.full(N, np.inf)
    A = np.zeros((R, N))
    for k in range(1, R + 1):
        for S in itertools.combinations(range(R), k):
            S = list(S)
            Ms = M[:, S]
            # equality-constrained LS via KKT system [[Ms'Ms, 1], [1', 0]]
            K = np.zeros((k + 1, k + 1))
            K[:k, :k] = Ms.T @ Ms
            K[:k, k] = K[k, :k] = 1
            rhs = np.vstack([Ms.T @ Y, np.ones((1, N))])
            sol = np.linalg.lstsq(K, rhs, rcond=None)[0][:k]
            ok = (sol >= -1e-12).all(0)
            err = ((Y - Ms @ sol) ** 2).sum(0)
            upd = ok & (err < best - 1e-15)
            if upd.any():
                best[upd] = err[upd]
                A[:, upd] = 0
                A[np.ix_(S, np.where(upd)[0])] = np.maximum(sol[:, upd], 0)
    return A / A.sum(0, keepdims=True)


# ------------------------------------------------------------------ VCA
def vca(Y, R, rng):
    """Vertex Component Analysis. Returns endmembers (L, R) and their pixel indices."""
    L, N = Y.shape
    snr_th = 15 + 10 * np.log10(R)
    ym = Y.mean(1, keepdims=True)
    Yc = Y - ym
    Ud = np.linalg.svd(Yc @ Yc.T / N)[0][:, :R]
    x_p = Ud.T @ Yc
    P_y = (Y ** 2).sum() / N
    P_x = (x_p ** 2).sum() / N + (ym ** 2).sum()
    snr = 10 * np.log10(max(P_x - R / L * P_y, 1e-12) / max(P_y - P_x, 1e-12))
    if snr < snr_th:                                   # projective projection onto the affine subspace
        d = R - 1
        Ud = Ud[:, :d]
        Yp = Ud @ x_p[:d] + ym
        x = x_p[:d]
        c = np.sqrt((x ** 2).sum(0).max())
        Yproj = np.vstack([x, c * np.ones((1, N))])
    else:                                              # projection onto the R-dim subspace
        Ud = np.linalg.svd(Y @ Y.T / N)[0][:, :R]
        x_p = Ud.T @ Y
        Yp = Ud @ x_p
        u = x_p.mean(1, keepdims=True)
        Yproj = x_p / (u * x_p).sum(0, keepdims=True)
    idx = np.zeros(R, dtype=int)
    Aux = np.zeros((R, R))
    Aux[-1, 0] = 1
    for i in range(R):
        w = rng.standard_normal((R, 1))
        f = w - Aux @ np.linalg.pinv(Aux) @ w
        f /= np.linalg.norm(f)
        v = f.T @ Yproj
        idx[i] = np.argmax(np.abs(v))
        Aux[:, i] = Yproj[:, idx[i]]
    return Yp[:, idx], idx


# ------------------------------------------------------------------ SUnSAL-TV (with ASC)
def _diff(X):
    """Forward differences with Neumann boundary on (R, rows, cols): returns (dr, dc)."""
    dr = np.zeros_like(X); dr[:, :-1] = X[:, 1:] - X[:, :-1]
    dc = np.zeros_like(X); dc[:, :, :-1] = X[:, :, 1:] - X[:, :, :-1]
    return dr, dc


def _diff_T(dr, dc):
    out = np.zeros_like(dr)
    out[:, 1:] += dr[:, :-1]; out[:, :-1] -= dr[:, :-1]
    out[:, :, 1:] += dc[:, :, :-1]; out[:, :, :-1] -= dc[:, :, :-1]
    return out


def fcls_tv(Y, M, shape, lam_tv, mu=None, iters=500, A0=None):
    """min 0.5||Y - M A||_F^2 + lam_tv * TV_aniso(A)  s.t. A in simplex (per pixel). Returns A (R, N)."""
    R, N = M.shape[1], Y.shape[1]
    rows, cols = shape
    toimg = lambda X: X.reshape(R, rows, cols, order="F")
    tovec = lambda X: X.reshape(R, -1, order="F")
    MtM, MtY = M.T @ M, toimg(M.T @ Y)
    mu = mu if mu is not None else 0.05 * np.trace(MtM) / R + 10 * lam_tv
    s, U = np.linalg.eigh(MtM)
    lap = (2 - 2 * np.cos(np.pi * np.arange(rows) / rows))[:, None] + \
          (2 - 2 * np.cos(np.pi * np.arange(cols) / cols))[None, :]
    denom = s[:, None, None] + mu + mu * lap[None]
    A = toimg(fcls(Y, M) if A0 is None else A0)
    V1, (V2r, V2c) = A.copy(), _diff(A)
    D1, D2r, D2c = np.zeros_like(A), np.zeros_like(A), np.zeros_like(A)
    thr = lam_tv / mu
    for _ in range(iters):
        rhs = MtY + mu * (V1 - D1) + mu * _diff_T(V2r - D2r, V2c - D2c)
        t = np.einsum("kr,rij->kij", U.T, rhs)                      # eigenbasis of M'M
        t = idctn(dctn(t, type=2, axes=(1, 2), norm="ortho") / denom, type=2, axes=(1, 2), norm="ortho")
        A = np.einsum("rk,kij->rij", U, t)
        V1 = toimg(project_simplex(tovec(A + D1)))
        dr, dc = _diff(A)
        V2r = np.sign(dr + D2r) * np.maximum(np.abs(dr + D2r) - thr, 0)
        V2c = np.sign(dc + D2c) * np.maximum(np.abs(dc + D2c) - thr, 0)
        D1 += A - V1; D2r += dr - V2r; D2c += dc - V2c
    return tovec(V1)


# ------------------------------------------------------------------ L1/2-NMF
def l12_nmf(Y, M0, A0, lam, iters=2000, delta=15.0):
    """Qian et al. (2011): min 0.5||Y - MA||^2 + lam ||A||_{1/2}, M, A >= 0, ASC via augmented row delta."""
    Yb = np.vstack([Y, delta * np.ones((1, Y.shape[1]))])
    M, A = M0.copy(), np.maximum(A0, 1e-4)
    eps = 1e-9
    for _ in range(iters):
        Mb = np.vstack([M, delta * np.ones((1, M.shape[1]))])
        A *= (Mb.T @ Yb) / (Mb.T @ Mb @ A + 0.5 * lam * A ** -0.5 + eps)
        M *= (Y @ A.T) / (M @ A @ A.T + eps)
    return M, A


# ------------------------------------------------------------------ scale-invariant abundances
def nnls_norm(Y, M):
    """Non-negative least squares per pixel, then each abundance vector divided by its sum.

    With M peak-normalized (M / M.max(0)) this reproduces the reference abundances shipped with Samson
    and Jasper Ridge (aRMSE 0.002 and 0.004 from the reference endmembers), i.e. the reference is a
    scale-invariant unmixing: each pixel has a free brightness factor, and the abundance scale is set by
    the convention that every endmember peaks at 1.
    """
    from scipy.optimize import nnls
    A = np.stack([nnls(M, Y[:, j])[0] for j in range(Y.shape[1])], 1)
    s = A.sum(0, keepdims=True)
    return np.where(s > 0, A / np.where(s > 0, s, 1), 1.0 / M.shape[1])
