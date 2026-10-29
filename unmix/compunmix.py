"""CompUnmix: unrolled unmixing with compositional message passing (CompMP-inspired), known endmembers.

State per pixel: abundance a_i in the simplex and brightness s_i >= 0, with y_i ~ s_i E a_i (the free
per-pixel brightness of the reference abundances, see FINDINGS.md).

  a0 = NNLS-normalize(y_i, E)                  per-pixel start (the skip connection)
  for each layer:
      a <- MP(a)                               neighbours refine the composition
      b = s a;  b <- relu(b - eta E'(E b - y)) data consistency (a few projected-gradient steps)
      s = sum(b);  a = b / s                   back to the simplex; exact zeros are possible

Message passing geometries (`geometry`):
  "fisher-rao"  B. u = sqrt(a) on the unit sphere (zeros allowed). Messages: attention-weighted chordal
                mean of neighbours' u; update u <- normalize((1 - g_i) u_i + g_i relu(W m_i + c)).
                SphereLayer of model.py with a gated skip.
  "aitchison"   C. h = clr(a_eps) with zeros replaced, a_eps = (a + eps) / (1 + R eps), eps = 1e-3.
                Messages: attention-weighted Aitchison barycenter (mean of clr); update
                h <- (1 - g_i) h_i + g_i P(W m_i + c), P = centering (stays in clr space); a = softmax(h).
                The following data step restores exact zeros.

Neighbour attention on the 8-neighbour pixel grid: alpha_ij ~ exp(-kappa SAD(y_i, y_j)), learned kappa
per layer. Per-pixel gate g_i = sigmoid(w . f_i + c) from features f_i (pixel's own reconstruction
angle, its mean spectral angle to neighbours, log brightness): g_i says how much the pixel listens to
its neighbours.

Training is unsupervised with a band split: each step a random half of the bands is used to unmix,
and the loss is the spectral angle on the other half. Noise independent across bands cannot be
fitted on the held-out half, so pooling neighbours is rewarded only when it removes noise. With the
plain reconstruction loss the per-pixel solution is already optimal and message passing would be
switched off.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

AIT_EPS = 1e-3


def grid_neighbours(shape):
    """(N, 8) neighbour indices in MATLAB pixel order and validity mask."""
    rows, cols = shape
    idx = np.arange(rows * cols).reshape(shape, order="F")
    nb, ok = [], []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            r = np.arange(rows)[:, None] + dr
            c = np.arange(cols)[None, :] + dc
            valid = (r >= 0) & (r < rows) & (c >= 0) & (c < cols)
            j = idx[np.clip(r, 0, rows - 1), np.clip(c, 0, cols - 1)]
            nb.append(j.ravel(order="F")); ok.append(valid.ravel(order="F"))
    return np.stack(nb, 1), np.stack(ok, 1)


def nnls_pg(Y, E, iters=500, B0=None):
    """Batched NNLS by accelerated projected gradient. Y (N, L), E (L, R) -> B (N, R) >= 0."""
    EtE, EtY = E.T @ E, Y @ E
    step = 1.0 / torch.linalg.eigvalsh(EtE).max()
    B = torch.zeros(Y.shape[0], E.shape[1], dtype=Y.dtype) if B0 is None else B0.clone()
    Z, t = B.clone(), 1.0
    for _ in range(iters):
        Bn = F.relu(Z - step * (Z @ EtE - EtY))
        tn = (1 + (1 + 4 * t * t) ** 0.5) / 2
        Z = Bn + (t - 1) / tn * (Bn - B)
        B, t = Bn, tn
    return B


def split(B):
    s = B.sum(1, keepdim=True)
    R = B.shape[1]
    return torch.where(s > 0, B / s.clamp_min(1e-12), torch.full_like(B, 1.0 / R)), s


def sad(Y, Yh):
    c = (Y * Yh).sum(1) / (Y.norm(dim=1) * Yh.norm(dim=1) + 1e-12)
    return torch.acos(c.clamp(-1 + 1e-7, 1 - 1e-7))


class MPLayer(nn.Module):
    def __init__(self, R, geometry, n_feat=3):
        super().__init__()
        self.geometry = geometry
        self.W = nn.Parameter(torch.eye(R))
        self.c = nn.Parameter(torch.zeros(R))
        self.log_kappa = nn.Parameter(torch.tensor(0.0))
        self.gate = nn.Linear(n_feat, 1)
        nn.init.zeros_(self.gate.weight); nn.init.constant_(self.gate.bias, -2.0)   # g ~ 0.12 at start
        self.log_eta = nn.Parameter(torch.tensor(0.0))                               # data-step scale

    def forward(self, A, nb, ok, sad_nb, feat):
        logits = -torch.exp(self.log_kappa) * sad_nb
        alpha = torch.softmax(logits.masked_fill(~ok, float("-inf")), 1)             # (N, 8)
        g = torch.sigmoid(self.gate(feat))                                             # (N, 1)
        if self.geometry == "fisher-rao":
            U = A.clamp_min(0).sqrt()
            M = (alpha[..., None] * U[nb]).sum(1)                                      # chordal mean
            Z = F.relu(M @ self.W.T + self.c)
            U = (1 - g) * U + g * Z
            U = U / (U.norm(dim=1, keepdim=True) + 1e-12)
            return U ** 2, g
        if self.geometry == "aitchison":
            R = A.shape[1]
            L = torch.log((A + AIT_EPS) / (1 + R * AIT_EPS))
            H = L - L.mean(1, keepdim=True)                                            # clr
            M = (alpha[..., None] * H[nb]).sum(1)                                      # Aitchison barycenter
            Z = M @ self.W.T + self.c
            Z = Z - Z.mean(1, keepdim=True)
            H = (1 - g) * H + g * Z
            return torch.softmax(H, 1), g
        raise ValueError(self.geometry)


class CompUnmix(nn.Module):
    def __init__(self, E, shape, geometry, layers=3, data_steps=5):
        super().__init__()
        self.register_buffer("E", torch.as_tensor(E, dtype=torch.float32))
        nb, ok = grid_neighbours(shape)
        self.register_buffer("nb", torch.tensor(nb)); self.register_buffer("ok", torch.tensor(ok))
        self.layers = nn.ModuleList(MPLayer(self.E.shape[1], geometry) for _ in range(layers))
        self.data_steps = data_steps

    def prepare(self, Y):
        """Fixed per-image quantities: neighbour spectral angles and their mean (feature)."""
        Yt = torch.as_tensor(Y.T, dtype=torch.float32)
        Yn = Yt / (Yt.norm(dim=1, keepdim=True) + 1e-12)
        cos = (Yn[:, None, :] * Yn[self.nb]).sum(-1).clamp(-1, 1)
        self.sad_nb = torch.acos(cos) * self.ok
        self.sad_nb_mean = (self.sad_nb.sum(1) / self.ok.sum(1)).unsqueeze(1)
        return Yt

    def forward(self, Yt, bands=None, B0=None, init_iters=500):
        E = self.E if bands is None else self.E[bands]
        Yv = Yt if bands is None else Yt[:, bands]
        with torch.no_grad():
            B = nnls_pg(Yv, E, init_iters, B0)                  # per-pixel start, not differentiated
        A, s = split(B)
        EtE, EtY = E.T @ E, Yv @ E
        step = 1.0 / torch.linalg.eigvalsh(EtE).max()
        gates = []
        for layer in self.layers:
            feat = torch.cat([sad(Yv, (s * A) @ E.T)[:, None], self.sad_nb_mean, torch.log(s + 1e-6)], 1)
            feat = (feat - feat.mean(0)) / (feat.std(0) + 1e-6)
            A, g = layer(A, self.nb, self.ok, self.sad_nb, feat)
            gates.append(g)
            B = s * A
            eta = step * torch.exp(layer.log_eta)
            for _ in range(self.data_steps):
                B = F.relu(B - eta * (B @ EtE - EtY))
            A, s = split(B)
        return A, s, gates


def fit(Y, E, shape, geometry, epochs=300, lr=1e-2, seed=0, layers=3, log_every=0):
    """Unsupervised band-split training on one image; returns abundances (R, N) and mean gates."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = CompUnmix(E, shape, geometry, layers)
    Yt = model.prepare(Y)
    L = Yt.shape[1]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        perm = rng.permutation(L)
        vis, held = torch.tensor(np.sort(perm[: L // 2])), torch.tensor(np.sort(perm[L // 2:]))
        A, s, _ = model(Yt, vis, init_iters=300)
        loss = sad(Yt[:, held], (s * A) @ model.E[held].T).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if log_every and (ep % log_every == 0 or ep == epochs - 1):
            print(f"  ep {ep:4d} held-out SAD {loss.item():.5f}")
    with torch.no_grad():
        A, s, gates = model(Yt, None, init_iters=2000)
    return A.numpy().T.astype(np.float64), [float(g.mean()) for g in gates], float(loss.item())
