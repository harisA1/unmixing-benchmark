"""CompMP: compositional message passing, as a standalone starting point.

Copied from github.com/harisA1/comp-gcn-reproduction (branch compmp: compgcn/compmp.py, compgcn/e3.py,
compgcn/ilr.py), so this file has no dependency on that repository. See that repository's PLAN.md and
SUMMARY.md for what was learned with it.

Contents
- CompLayer / CompMP: node states are C compositions of D parts stored as clr vectors (n, C, D).
  Aggregation A @ clr (row-normalized A = Aitchison barycenter), linear maps ("off": one weight per channel
  pair, "per"/"shared": plus a D x D map, "full": one D x D zero-sum map per channel pair = as expressive
  as a dense GCN layer on ilr coordinates), optional skip (self) connection, scalar covariate input,
  nonlinearity "naive" (ReLU on ilr, basis-dependent), "norm" (norm-gated) or "part" (per-part, centered;
  S_D-equivariant), heads for scalar or compositional outputs, residual hidden layers.
- SphereLayer / SphereGNN: Fisher-Rao variant on the square-root sphere u = sqrt(p); allows exact zeros.
- loss_fn: squared Aitchison, KL, count-weighted multinomial, Hellinger.

Lessons from the CompMP project (most relevant first)
- Without a skip connection every GNN lost to plain linear regression on county vote shares: averaging
  with neighbours dilutes the node's own signal. Use skip=True.
- The "stack" scalar head overfits small training sets; use head="prop" for scalar targets.
- Linear (Aitchison) parts are exactly a GCN on ilr coordinates; only nonlinearities, outputs and losses
  can differ from an ordinary GNN on transformed data.

Adapting it to spectral unmixing (read before using)
- Aitchison / clr geometry needs strictly positive parts. Reference abundances on Jasper Ridge are 43%
  exact zeros, and linear mixing combines abundances arithmetically, not geometrically. Prefer the
  Fisher-Rao variant (SphereGNN, zeros allowed) or Euclidean simplex outputs (sparsemax / ReLU + normalize)
  for abundances; if clr is used, zeros must be replaced and that choice reported.
- Unmixing inputs are spectra (not compositions): feed them as scalar covariates (n_scalar = bands) and
  let the network output the abundance composition (out="comp"), followed by a linear decoder y = E a.
- Graph: 4- or 8-neighbour pixel grid (MATLAB column-major pixel order, reshape with order="F").
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


# ------------------------------------------------------------------ log-ratio helpers
def helmert_basis(D: int) -> np.ndarray:
    """Orthonormal basis of the zero-sum hyperplane in R^D, shape (D-1, D).

    Row i is clr(xi_i) for the i-th element of an orthonormal basis of the
    simplex (the contrast matrix Psi in the paper, Definition 3).
    """
    Psi = np.zeros((D - 1, D))
    for i in range(1, D):
        Psi[i - 1, :i] = 1.0 / i
        Psi[i - 1, i] = -1.0
        Psi[i - 1] *= np.sqrt(i / (i + 1.0))
    return Psi


def random_basis(D: int, rng: np.random.Generator) -> np.ndarray:
    """A random orthonormal basis of the zero-sum hyperplane (Helmert rotated)."""
    Q, _ = np.linalg.qr(rng.standard_normal((D - 1, D - 1)))
    return Q @ helmert_basis(D)


def clr(x: np.ndarray) -> np.ndarray:
    lx = np.log(x)
    return lx - lx.mean(-1, keepdims=True)


# ------------------------------------------------------------------ CompMP
def center(z):
    """Apply P = I - 11'/D on the last axis."""
    return z - z.mean(-1, keepdim=True)


def clr_blocks(X, D):
    """(n, p*D) raw compositions with p equal blocks of D parts -> (n, p, D) clr."""
    assert len(set(D)) == 1, "CompMP needs blocks of equal size"
    n, d = X.shape[0], D[0]
    return clr(X.reshape(n, len(D), d))


def adjacency(edges, n, mode="row"):
    """Dense normalized adjacency with self-loops (receivers are rows), undirected.

    mode="row": D^-1 (A+I), rows sum to 1, so A_hat @ clr is a true Aitchison barycenter.
    mode="sym": D^-1/2 (A+I) D^-1/2, as in GCN.
    """
    A = np.zeros((n, n), dtype=np.float32)
    A[edges[:, 0], edges[:, 1]] = 1
    A[edges[:, 1], edges[:, 0]] = 1
    np.fill_diagonal(A, 1)
    deg = A.sum(1)
    if mode == "row":
        A = A / deg[:, None]
    elif mode == "sym":
        s = 1 / np.sqrt(deg)
        A = A * s[:, None] * s[None, :]
    else:
        raise ValueError(mode)
    return torch.tensor(A)


def comp_dropout(h, p, training):
    """Drop whole compositions (set to the neutral element, clr = 0), rescale the rest."""
    if not training or p == 0:
        return h
    keep = (torch.rand(h.shape[0], h.shape[1], 1) > p).to(h.dtype)
    return h * keep / (1 - p)


class CompLayer(nn.Module):
    def __init__(self, p_in, p_out, D, nonlin=None, within="off", basis=None, skip=False):
        super().__init__()
        self.within = within
        # skip: the node's own state through a separate full zero-sum map (no aggregation), so the
        # node's own composition is not diluted by averaging with its neighbours
        self.Ws = nn.Parameter((torch.rand(p_out, p_in, D, D) * 2 - 1) * np.sqrt(6.0 / ((p_in + p_out) * D))) \
            if skip else None
        if within == "full":
            # every input -> output channel pair gets its own D x D map, output projected back to the
            # zero-sum hyperplane: the general zero-sum-preserving linear map (as expressive as a dense
            # linear layer on ilr coordinates)
            bound = np.sqrt(6.0 / ((p_in + p_out) * D))
            self.W = nn.Parameter((torch.rand(p_out, p_in, D, D) * 2 - 1) * bound)
        else:
            self.alpha = nn.Parameter(torch.empty(p_out, p_in))
            nn.init.xavier_uniform_(self.alpha)
        if within == "shared":
            self.M = nn.Parameter(torch.eye(D))
        elif within == "per":
            self.M = nn.Parameter(torch.eye(D).repeat(p_out, 1, 1))
        elif within not in ("off", "full"):
            raise ValueError(within)
        self.bias = nn.Parameter(torch.zeros(p_out, D))
        self.nonlin = nonlin
        if nonlin == "naive":
            Psi = helmert_basis(D) if basis is None else basis
            self.register_buffer("Psi", torch.tensor(Psi, dtype=torch.float32))   # (D-1, D)
        elif nonlin == "norm":
            # output norm rho(r) = softplus(a(r - c)) - softplus(-a c): rho(0) = 0, smooth, ~ a(r - c) for large r
            self.a_raw = nn.Parameter(torch.full((p_out,), 0.5413))   # softplus^-1(1)
            self.c = nn.Parameter(torch.zeros(p_out))
        elif nonlin not in (None, "part"):
            raise ValueError(nonlin)
        self._ah = None

    def forward(self, h, A, cache=False, extra=None):
        """extra: (n, p_out, D) added before bias and nonlinearity (e.g. projected scalar covariates)."""
        n = h.shape[0]
        if cache:                       # first layer: the input is fixed, compute A @ clr once
            if self._ah is None:
                self._ah = (A @ h.reshape(n, -1)).reshape(h.shape)
            ah = self._ah
        else:
            ah = (A @ h.reshape(n, -1)).reshape(h.shape)
        if self.within == "full":
            z = center(torch.einsum("kjde,nje->nkd", self.W, ah))
        else:
            z = torch.einsum("kj,njd->nkd", self.alpha, ah)
        if self.within == "shared":
            z = center(z @ self.M.T)
        elif self.within == "per":
            z = center(torch.einsum("kde,nke->nkd", self.M, z))
        if self.Ws is not None:
            z = z + center(torch.einsum("kjde,nje->nkd", self.Ws, h))
        if extra is not None:
            z = z + center(extra)
        z = z + center(self.bias)
        return self.act(z)

    def act(self, z):
        if self.nonlin == "naive":
            return F.relu(z @ self.Psi.T) @ self.Psi
        if self.nonlin == "norm":
            r = z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            a, c = F.softplus(self.a_raw)[:, None], self.c[:, None]
            rho = F.softplus(a * (r - c)) - F.softplus(-a * c)
            return z * rho / r
        if self.nonlin == "part":
            return center(F.leaky_relu(z, 0.1))
        return z


class CompMP(nn.Module):
    """CompLayer (p -> hidden, nonlinearity) -> composition dropout -> CompLayer (hidden -> q, linear) -> readout.

    out="scalar": q = hidden; the readout is a sum of Aitchison inner products of each output
      composition with learned compositions, i.e. a Linear layer on the flattened clr state.
    out="comp":   q = number of target compositions; the output is the clr of the prediction
      (prediction = softmax of it).
    layers > 2 inserts hidden -> hidden CompLayers with the nonlinearity, each with a residual
    connection in clr (h + layer(h), i.e. perturbation), against over-smoothing towards the
    global Aitchison barycenter.
    head (scalar targets only): "stack" = CompLayer (hidden -> hidden) then Linear readout (used in E1);
      "prop" = Aitchison inner products with learned compositions, then propagation, like GCN's second
      layer (A H W). "stack" stacks two linear maps and overfits small training sets much faster.
    """
    def __init__(self, p, D, hidden, n_out, dropout, nonlin="part", within="off", basis=None, out="scalar",
                 layers=2, head="stack", n_scalar=0, skip=False):
        super().__init__()
        self.l1 = CompLayer(p, hidden, D, nonlin, within, basis, skip)
        # scalar covariates enter the first layer Aitchison-linearly: A s -> powering of learned directions
        self.scalar_in = nn.Linear(n_scalar, hidden * D, bias=False) if n_scalar else None
        self.scalar_self = nn.Linear(n_scalar, hidden * D, bias=False) if (n_scalar and skip) else None
        self._as = None
        self.mid = nn.ModuleList([CompLayer(hidden, hidden, D, nonlin, within, basis) for _ in range(layers - 2)])
        q = hidden if out == "scalar" else n_out
        self.head = head if out == "scalar" else "stack"
        self.l2 = CompLayer(hidden, q, D, None, within, skip=skip) if self.head == "stack" else None
        self.readout = nn.Linear(q * D, n_out) if out == "scalar" else None
        if self.readout is not None:
            nn.init.xavier_uniform_(self.readout.weight); nn.init.zeros_(self.readout.bias)
        self.dp = dropout

    def forward(self, h, A, s=None):
        extra = None
        if self.scalar_in is not None:
            if self._as is None:
                self._as = A @ s
            extra = self.scalar_in(self._as)
            if self.scalar_self is not None:
                extra = extra + self.scalar_self(s)
            extra = extra.reshape(h.shape[0], -1, h.shape[-1])
        h = self.l1(h, A, cache=True, extra=extra)
        for layer in self.mid:
            h = h + layer(comp_dropout(h, self.dp, self.training), A)
        h = comp_dropout(h, self.dp, self.training)
        if self.head == "prop":
            return A @ self.readout(h.reshape(h.shape[0], -1))
        h = self.l2(h, A)
        if self.readout is None:
            return h                                  # (n, q, D) clr of the predicted compositions
        return self.readout(h.reshape(h.shape[0], -1))


def aitchison_loss(pred_clr, target_clr):
    """Mean squared Aitchison distance; inputs (n, q, D) clr."""
    return ((pred_clr - target_clr) ** 2).sum(-1).mean()


# ------------------------------------------------------------------ Fisher-Rao (square-root sphere) variant
class SphereLayer(nn.Module):
    """States are unit vectors u = sqrt(p) in the positive orthant of the sphere (zeros allowed).
    Aggregation: weighted chordal mean A u (normalized by the nonlinearity below). Then a linear map per
    channel pair, scalar covariates, bias, ReLU per part and projection back onto the sphere. The
    projection makes the layer nonlinear in every coordinate system, so it is not a GCN on any fixed
    transform of the shares. ReLU can produce exact zero parts."""
    def __init__(self, c_in, c_out, D, n_scalar=0):
        super().__init__()
        bound = np.sqrt(6.0 / ((c_in + c_out) * D))
        self.W = nn.Parameter((torch.rand(c_out, c_in, D, D) * 2 - 1) * bound)
        self.Ws = nn.Parameter((torch.rand(c_out, c_in, D, D) * 2 - 1) * bound)      # skip: own state
        self.V = nn.Linear(n_scalar, c_out * D, bias=False) if n_scalar else None
        self.Vs = nn.Linear(n_scalar, c_out * D, bias=False) if n_scalar else None
        self.b = nn.Parameter(torch.full((c_out, D), 1 / np.sqrt(D)))   # start inside the orthant
        self._au, self._as = None, None

    def forward(self, u, A, s=None, cache=False):
        n = u.shape[0]
        if cache and self._au is not None:
            au = self._au
        else:
            au = (A @ u.reshape(n, -1)).reshape(u.shape)
            if cache:
                self._au = au
        z = torch.einsum("kjde,nje->nkd", self.W, au) + torch.einsum("kjde,nje->nkd", self.Ws, u) + self.b
        if self.V is not None:
            if self._as is None:
                self._as = A @ s
            z = z + (self.V(self._as) + self.Vs(s)).reshape(z.shape)
        z = F.relu(z)
        return z / (z.norm(dim=-1, keepdim=True) + EPS)


class SphereGNN(nn.Module):
    def __init__(self, D, hidden, dropout, n_scalar):
        super().__init__()
        self.l1 = SphereLayer(1, hidden, D, n_scalar)
        self.l2 = SphereLayer(hidden, 1, D)
        self.dp, self.D = dropout, D

    def forward(self, u, A, s):
        h = self.l1(u, A, s, cache=True)
        if self.training and self.dp > 0:     # drop whole channels: replace by the barycenter direction
            keep = (torch.rand(h.shape[0], h.shape[1], 1) > self.dp).to(h.dtype)
            h = keep * h + (1 - keep) / np.sqrt(self.D)
        p = self.l2(h, A)[:, 0] ** 2
        return torch.log((p + EPS) / (1 + self.D * EPS))


# ------------------------------------------------------------------ losses on log-probabilities lp (n, D)
def loss_fn(name, lp, shares, counts):
    if name == "aitchison":
        return ((center(lp) - center(torch.log(shares))) ** 2).sum(1).mean()
    if name == "kl":
        return (shares * (torch.log(shares) - lp)).sum(1).mean()
    if name == "multinomial":          # count-weighted: per-vote negative log-likelihood
        return -(counts * lp).sum() / counts.sum()
    if name == "hellinger":
        return ((torch.exp(lp / 2) - torch.sqrt(shares)) ** 2).sum(1).mean()
    raise ValueError(name)
