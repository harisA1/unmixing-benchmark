"""Deep blind-unmixing autoencoders (PyTorch, CPU-friendly, full-image batches).

- PixelAE: per-pixel dense autoencoder in the style of DAEU (Palsson et al. 2018): dense encoder, BN,
  softmax, non-negative linear decoder initialized by VCA, SAD reconstruction loss.
- GATCNN: re-implementation of GAT-CNN-HU (Dong et al., IEEE JSTARS 18, 2025, 12865-12880) from the paper
  text: spectral reduction (2x [BN, 1x1 conv, LeakyReLU] -> 128), global branch (superpixel graph, kNN
  edges on superpixel means, 2 GAT layers 128->64->64 with 2 averaged heads, broadcast back to pixels),
  local branch (2x [3x3 conv 64, BN, (CBAM), ReLU] with a 1x1 skip from the reduced features), sum
  fusion, [dropout, Linear 32, ReLU, Linear R, BN], adaptive abundance constraint module (AACM), linear
  bias-free decoder clipped to be non-negative. Loss 1*SAD + 0.3*MSE + 0.05*L1/2, Adam lr 1e-2.
  Flags give the paper's ablations: use_gat=False ("w/o GCN"), use_cnn=False, aacm=False (softmax).
  Not specified in the paper and chosen here: superpixels from SLIC on the input's first PCA components
  (the paper segments the learned features), k = 8 graph neighbours, dropout 0.1, VCA decoder init,
  epochs (see run scripts).
- MGALite: the core ideas of MGA-Net (Dong et al., IJRS 46, 2025, 3709-3735) without the graph U-Net:
  pixel spectrum concatenated with superpixel-graph features (GAT over spatially adjacent superpixels),
  3-layer encoder with intra-superpixel self-attention, BN, softmax; one endmember matrix per superpixel
  (spectral variability) tied to a shared one by a quadratic penalty; loss SAD + lam * L1/2 +
  beta * sum of per-superpixel nuclear norms of the abundance matrix.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.segmentation import slic

from .data import to_image, from_image


# ------------------------------------------------------------------ superpixels and graphs
def superpixels(Y, shape, n_target, compactness=10.0, n_pc=3):
    """SLIC on the first PCA components (scaled to [0, 1]); returns labels (N,) in MATLAB pixel order.

    skimage's connectivity step merges segments, so n_segments is increased until at least n_target
    superpixels remain (the papers' "number of superpixels" is taken as the actual count)."""
    Yc = Y - Y.mean(1, keepdims=True)
    U = np.linalg.svd(Yc @ Yc.T)[0][:, :n_pc]
    P = U.T @ Yc
    P = (P - P.min(1, keepdims=True)) / (np.ptp(P, 1, keepdims=True) + 1e-12)
    img = np.moveaxis(to_image(P, shape), 0, -1)
    n = n_target
    while True:
        lab = slic(img, n_segments=n, compactness=compactness, channel_axis=-1, start_label=0,
                   enforce_connectivity=True, min_size_factor=0.2)
        if lab.max() + 1 >= n_target or n > 20 * n_target:
            break
        n = int(n * 1.15) + 1
    lab = np.unique(lab, return_inverse=True)[1].reshape(lab.shape)   # consecutive labels
    return from_image(lab[None])[0]


def assignment(labels):
    """O (N, S) one-hot pixel -> superpixel, and column-normalized O_hat."""
    S = labels.max() + 1
    O = np.zeros((labels.size, S), dtype=np.float32)
    O[np.arange(labels.size), labels] = 1
    return O, O / O.sum(0, keepdims=True)


def knn_adjacency(V, k):
    """Symmetric kNN adjacency (with self-loops) on node features V (S, C)."""
    d = ((V[:, None] - V[None]) ** 2).sum(-1)
    nn_idx = np.argsort(d, 1)[:, 1:k + 1]
    A = np.zeros_like(d, dtype=bool)
    A[np.repeat(np.arange(len(V)), k), nn_idx.ravel()] = True
    A = A | A.T
    np.fill_diagonal(A, True)
    return A


def spatial_adjacency(labels, shape):
    """Superpixels touching each other (4-neighbourhood), with self-loops."""
    img = labels.reshape(shape, order="F")
    S = labels.max() + 1
    A = np.eye(S, dtype=bool)
    for a, b in [(img[1:, :], img[:-1, :]), (img[:, 1:], img[:, :-1])]:
        A[a.ravel(), b.ravel()] = True
        A[b.ravel(), a.ravel()] = True
    return A


# ------------------------------------------------------------------ building blocks
class GATLayer(nn.Module):
    """Dense multi-head GAT (Velickovic et al. 2018), heads averaged, as in GAT-CNN-HU eq. (13)."""
    def __init__(self, c_in, c_out, heads=2):
        super().__init__()
        self.W = nn.Linear(c_in, c_out * heads, bias=False)
        self.a_src = nn.Parameter(torch.randn(heads, c_out) * 0.1)
        self.a_dst = nn.Parameter(torch.randn(heads, c_out) * 0.1)
        self.h, self.c = heads, c_out

    def forward(self, x, mask):
        h = self.W(x).view(-1, self.h, self.c)                                # (S, H, C)
        e = F.leaky_relu((h * self.a_dst).sum(-1)[:, None] + (h * self.a_src).sum(-1)[None], 0.2)  # (S, S, H)
        e = e.masked_fill(~mask[..., None], float("-inf"))
        att = torch.softmax(e, dim=1)
        return F.elu(torch.einsum("ijh,jhc->ihc", att, h).mean(1))


class CBAM(nn.Module):
    def __init__(self, c, r=8, k=7):
        super().__init__()
        self.mlp = nn.Sequential(nn.Conv2d(c, c // r, 1), nn.ReLU(), nn.Conv2d(c // r, c, 1))
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2)

    def forward(self, x):
        x = x * torch.sigmoid(self.mlp(x.mean((2, 3), keepdim=True)) + self.mlp(x.amax((2, 3), keepdim=True)))
        s = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)
        return x * torch.sigmoid(self.conv(s))


class AACM(nn.Module):
    """Adaptive abundance constraint: a = norm(relu(h - tanh-bounded learned threshold(h)))."""
    def __init__(self, R):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(R, R), nn.BatchNorm1d(R), nn.LeakyReLU(0.1), nn.Linear(R, R), nn.Tanh())

    def forward(self, h):
        z = F.relu(h - self.net(h))
        return z / (z.sum(1, keepdim=True) + 1e-8)


class Decoder(nn.Module):
    """Bias-free linear decoder; weights are the endmembers (L, R), kept non-negative by clipping."""
    def __init__(self, M0):
        super().__init__()
        self.E = nn.Parameter(torch.tensor(M0, dtype=torch.float32))

    def forward(self, A):                      # A (N, R) -> (N, L)
        return A @ self.E.T

    def clip(self):
        with torch.no_grad():
            self.E.clamp_(min=1e-6)


# ------------------------------------------------------------------ models
class PixelAE(nn.Module):
    def __init__(self, L, R, M0):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(L, 9 * R), nn.LeakyReLU(0.1), nn.Linear(9 * R, 6 * R), nn.LeakyReLU(0.1),
                                 nn.Linear(6 * R, 3 * R), nn.LeakyReLU(0.1), nn.Linear(3 * R, R), nn.BatchNorm1d(R))
        self.dec = Decoder(M0)

    def forward(self, ctx):
        A = torch.softmax(self.enc(ctx["Yt"]), 1)
        return A, self.dec(A)


class GATCNN(nn.Module):
    def __init__(self, L, R, M0, n_sp_mask, use_gat=True, use_cnn=True, use_cbam=True, aacm=True, dropout=0.1):
        super().__init__()
        self.reduce = nn.Sequential(nn.BatchNorm2d(L), nn.Conv2d(L, 128, 1), nn.LeakyReLU(0.1),
                                    nn.BatchNorm2d(128), nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1))
        self.use_gat, self.use_cnn = use_gat, use_cnn
        if use_gat:
            self.gat1, self.gat2 = GATLayer(128, 64), GATLayer(64, 64)
        if use_cnn:
            self.c1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
            self.c2 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64))
            self.cbam = CBAM(64) if use_cbam else nn.Identity()
            self.skip = nn.Conv2d(128, 64, 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, R), nn.BatchNorm1d(R))
        self.aacm = AACM(R) if aacm else None
        self.dec = Decoder(M0)

    def forward(self, ctx):
        X = ctx["Ximg"]                                        # (1, L, rows, cols)
        Fm = self.reduce(X)
        Fp = Fm.flatten(2)[0].T                                # (N, 128), row-major pixel order of the image tensor
        H = 0
        if self.use_gat:
            V = ctx["Ohat_T_rm"] @ Fp                          # Img2Graph (row-major pixel order)
            V = self.gat2(self.gat1(V, ctx["mask"]), ctx["mask"])
            H = H + ctx["O_rm"] @ V                            # Graph2Img
        if self.use_cnn:
            Hl = F.relu(self.cbam(self.c2(self.c1(Fm))) + self.skip(Fm))
            H = H + Hl.flatten(2)[0].T
        h = self.head(H)
        A = self.aacm(h) if self.aacm is not None else torch.softmax(h, 1)
        A = A[ctx["perm"]]                                     # back to MATLAB pixel order
        return A, self.dec(A)


class MGALite(nn.Module):
    def __init__(self, L, R, M0, n_sp, groups, hidden=64):
        super().__init__()
        self.gat1, self.gat2 = GATLayer(L, hidden), GATLayer(hidden, hidden)
        self.l1 = nn.Linear(L + hidden, 4 * hidden)
        self.q, self.k, self.v = (nn.Linear(4 * hidden, 4 * hidden, bias=False) for _ in range(3))
        self.l2, self.l3 = nn.Linear(4 * hidden, hidden), nn.Linear(hidden, R)
        self.bn = nn.BatchNorm1d(R)
        self.groups = groups                                   # (S, maxlen) pixel indices, -1 padded
        self.E = nn.Parameter(torch.tensor(M0, dtype=torch.float32))                   # shared (L, R)
        self.dE = nn.Parameter(torch.zeros(n_sp, *M0.shape))                           # per superpixel

    def endmembers(self):
        return F.relu(self.E[None] + self.dE) + 1e-6           # (S, L, R)

    def forward(self, ctx):
        Y = ctx["Yt"]
        V = self.gat2(self.gat1(ctx["Ohat_T"] @ Y, ctx["mask"]), ctx["mask"])
        h = F.leaky_relu(self.l1(torch.cat([Y, ctx["O"] @ V], 1)), 0.1)
        g = self.groups
        hg = h[g.clamp(min=0)]                                 # (S, maxlen, C)
        valid = g >= 0
        att = (self.q(hg) @ self.k(hg).transpose(1, 2)) / np.sqrt(hg.shape[-1])
        att = torch.softmax(att.masked_fill(~valid[:, None, :], float("-inf")), -1)
        hg = hg + att @ self.v(hg)
        h = torch.zeros_like(h).index_put_((g[valid],), hg[valid])
        h = F.leaky_relu(self.l2(h), 0.1)
        A = torch.softmax(self.bn(self.l3(h)), 1)
        Es = self.endmembers()[ctx["labels"]]                  # (N, L, R)
        return A, torch.einsum("nlr,nr->nl", Es, A)


class AESU(nn.Module):
    """AE-SU (Ampas et al., GRSL 2025), copied from AESU_Unmix_Embed_ZAZ_1.ipynb.

    Spatial encoder (reflect-pad 3x3 conv, BN, ReLU, dilated 3x3 conv, BN, ReLU), spectral encoder (1-D
    convs over wavelength per pixel, average-pooled), 1x1 fusion, softmax abundances, endmember matrix
    initialized from given spectra and frozen for the first `freeze_epochs`, then a convolutional
    refine layer applied to the reconstruction (refine=False drops it, leaving a strictly linear
    decoder as described in the letter). The notebook does not clip endmembers; neither does this.
    """
    def __init__(self, L, R, M0, embed_dim=64, hid=32, refine=True, freeze_epochs=50):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(L, hid, 3), nn.BatchNorm2d(hid), nn.ReLU(),
            nn.Conv2d(hid, embed_dim, 3, dilation=2, padding=2), nn.BatchNorm2d(embed_dim), nn.ReLU())
        self.spectral = nn.Sequential(
            nn.Conv1d(1, hid, 5, padding=2), nn.BatchNorm1d(hid), nn.ReLU(),
            nn.Conv1d(hid, embed_dim, 5, padding=2), nn.BatchNorm1d(embed_dim), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.fusion = nn.Sequential(nn.Conv2d(2 * embed_dim, embed_dim, 1), nn.BatchNorm2d(embed_dim), nn.ReLU())
        self.logits = nn.Conv2d(embed_dim, R, 1)
        self.dec = Decoder(M0)
        self.dec.clip = lambda: None
        self.refine = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(L, hid, 3), nn.ReLU(),
                                    nn.ReflectionPad2d(1), nn.Conv2d(hid, L, 3)) if refine else None
        self.freeze_epochs, self.embed_dim = freeze_epochs, embed_dim

    def on_epoch(self, ep):
        self.dec.E.requires_grad_(ep >= self.freeze_epochs)

    def forward(self, ctx):
        x = ctx["Ximg"]
        _, L, H, W = x.shape
        sp = self.spatial(x)
        se = self.spectral(x.permute(0, 2, 3, 1).reshape(-1, 1, L)).squeeze(-1)
        se = se.reshape(1, H, W, self.embed_dim).permute(0, 3, 1, 2)
        Aimg = torch.softmax(self.logits(self.fusion(torch.cat([sp, se], 1))), 1)        # (1, R, H, W)
        rec = torch.einsum("bkhw,lk->blhw", Aimg, self.dec.E)
        if self.refine is not None:
            rec = self.refine(rec)
        perm = ctx["perm"]
        return Aimg.flatten(2)[0].T[perm], rec.flatten(2)[0].T[perm]


def aesu_loss(Y, Yh, A, lam=0.01):
    """Notebook loss: MSE (mean over bands and pixels) + SAD + lam * mean(-a log a)."""
    return ((Yh - Y) ** 2).mean() + sad_loss(Y, Yh) + lam * (-(A * torch.log(A + 1e-8)).mean())


# ------------------------------------------------------------------ losses and training
def sad_loss(Y, Yh):
    c = (Y * Yh).sum(1) / (Y.norm(dim=1) * Yh.norm(dim=1) + 1e-8)
    return torch.acos(c.clamp(-1 + 1e-6, 1 - 1e-6)).mean()


def l12(A):
    return (A + 1e-8).sqrt().sum(1).mean()


def context(Y, shape, labels=None, adj=None):
    """Tensors shared by the models. Y (L, N) MATLAB order. Image tensors use row-major (torch) order."""
    L, N = Y.shape
    img = to_image(Y, shape)                                               # (L, rows, cols)
    rm = np.arange(N).reshape(shape, order="F").ravel()                    # row-major position -> MATLAB index
    perm = np.argsort(rm)                                                  # MATLAB index -> row-major position
    ctx = {"Yt": torch.tensor(Y.T, dtype=torch.float32),
           "Ximg": torch.tensor(img[None], dtype=torch.float32),
           "perm": torch.tensor(perm)}
    if labels is not None:
        O, Oh = assignment(labels)
        lab_rm = labels[rm]                                                # labels in row-major order
        O_rm, Oh_rm = assignment(lab_rm)
        ctx.update(O=torch.tensor(O), Ohat_T=torch.tensor(Oh.T), labels=torch.tensor(labels),
                   mask=torch.tensor(adj))
        ctx["O_rm"], ctx["Ohat_T_rm"] = torch.tensor(O_rm), torch.tensor(Oh_rm.T)
    return ctx


def train(model, ctx, loss_fn, epochs, lr, weight_decay=0.0, seed=0, log_every=0, clip_grad=None):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for ep in range(epochs):
        if hasattr(model, "on_epoch"):
            model.on_epoch(ep)
        model.train()
        opt.zero_grad()
        A, Yh = model(ctx)
        loss = loss_fn(A, Yh, model)
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        opt.step()
        if hasattr(model, "dec"):
            model.dec.clip()
        if log_every and (ep % log_every == 0 or ep == epochs - 1):
            print(f"  ep {ep:4d} loss {loss.item():.5f}")
    model.eval()
    with torch.no_grad():
        A, Yh = model(ctx)
    return A.numpy().T.astype(np.float64), Yh.numpy().T.astype(np.float64)
