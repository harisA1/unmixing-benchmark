# Step 1 findings: baselines and paper reproductions on Samson and Jasper Ridge

Date: 27 October 2026. Code: `unmix/` (loader, metrics, classical and deep methods), `run_benchmark.py`,
`summarize.py`, `check_reference.py`. Raw per-seed results: `results/*.csv`.

## 1. Is the data enough?

Yes, for both scenes, now that `samson_1.mat` has been added.

| file | contents | checks |
|---|---|---|
| `jasperRidge2_R198.mat` | `Y` 198 × 10,000 (uint16), `maxValue` 5000 | scaled by one global factor, `Y / 5000` |
| `gt_jasper_end4.mat` | `M` 198 × 4, `A` 4 × 10,000 (tree, water, dirt, road) | sum to one, 43% exact zeros |
| `samson_1.mat` | `V` 156 × 9,025 (float, already in [0, 1]), 95 × 95 | standard Zhu crop |
| `gt_samson_end3.mat` | `M` 156 × 3, `A` 3 × 9,025 (rock/soil, tree, water) | sum to one, 26% exact zeros |

Pixel order is MATLAB column-major for all four files. On Samson, pure reference pixels match the
reference endmembers to within 0.3–2.7°. The data loader checks SHA-256 checksums. Data files live in
the git-ignored `data/` folder.

The papers also use a synthetic dataset and Urban. Those are not here, and they aren't needed for
Samson and Jasper.

## 2. The reference abundances are a scale-invariant NNLS, not FCLS

This is the most important finding for interpreting any number on these two datasets
(`python check_reference.py`).

| recipe, using the reference endmembers | Samson aRMSE | Jasper aRMSE |
|---|---|---|
| FCLS (sum-to-one inside the solver), M as given | 0.410 | 0.085 |
| NNLS, then divide by the sum, M as given | 0.002 | 0.048 |
| NNLS, then divide by the sum, **each endmember scaled to peak at 1** | **0.002** | **0.004** |

On both datasets the reference `A` is `NNLS(Y, M / max(M))` normalized to sum to one; the exact-zero
pattern agrees on 99% of the entries. Consequences:

1. **Every pixel has a free brightness factor** (shading, illumination). FCLS does not allow one, so FCLS
   with the *true* endmembers scores 0.41 on Samson, worse than uniform abundances (0.37).
2. **The abundance scale is a convention.** It is set by scaling every endmember to peak at 1. A blind
   method that estimates endmembers with different relative scales gets proportionally different
   abundances, even when its endmember shapes are perfect. Rescaling an estimate to the convention
   ("aligned" below: `a_k * max(e_k)`, renormalized) uses no ground-truth information.
3. **Methods trained with an SAD loss are favoured.** SAD is scale-invariant, as the reference is. Both
   papers train with SAD (MGA-Net with SAD only).
4. Spatial regularization cannot help against this reference, which is purely pixel-wise. FCLS-TV with
   the true endmembers only gets worse as the TV weight grows (Jasper 0.084 → 0.145).

This is a property of the published benchmark files, so it affects every paper that uses them. It is
not specific to the two papers here. Neither paper states its abundance or endmember normalization.

**The rows that use reference endmembers are oracles, not methods.** "NNLS-normalize with the reference
M" reaches 0.002 / 0.004 because it repeats how the reference was computed. It is a reverse-engineering
check and must not be compared with blind methods. The reference endmembers themselves are (near-)pure
image pixels: Samson rock and tree are exact pixels, water is 0.02 rad from the nearest pixel, Jasper
road is an exact pixel and the others are within 0.06 rad. VCA does not find them (best of 10 seeds:
0.02–0.10 rad on Samson, 0.18–0.53 rad on Jasper).

**Blind scores depend on the endmember-scale convention, so report raw and normalized side by side.**
Peak normalization is learned from the reference files, so using it is mild leakage. VCA endmembers,
10 seeds:

| abundances from VCA endmembers | Samson aRMSE | Jasper aRMSE |
|---|---|---|
| FCLS, raw VCA | 0.261 | 0.239 |
| NNLS-normalize, raw VCA (no ground-truth knowledge) | 0.128 | 0.157 |
| NNLS-normalize, L2-normalized VCA (generic default) | 0.061 | 0.183 |
| NNLS-normalize, peak-normalized VCA (reference convention) | 0.050 | 0.157 |

The free per-pixel brightness factor (FCLS → NNLS-normalize, same raw endmembers) is the
legitimate part of the gain. The further gain on Samson comes from the convention, mostly through the
very dark water endmember. Protocol from here: fix the convention before running, and report every
blind method raw and normalized.

INSPECTION.md reported 0.077 for FCLS with the reference `M` on Jasper. With exact FCLS and the
per-material aRMSE used in the papers it is 0.0845 (RMSE over all entries 0.0851). The older number
came from the notebook's pseudo-inverse + clip + renormalize.

## 3. What the papers report and how they evaluate

Both papers are by the same group (Dong, Zhang, Meng, Jiao; Xidian).

| | MGA-Net (IJRS 2025) | GAT-CNN-HU (JSTARS 2025) |
|---|---|---|
| core | superpixel graph attention U-Net (global) + intra-superpixel attention AE (local); one endmember set per superpixel | SLIC superpixel graph + 2-layer GAT (global) ‖ 2-layer CNN + CBAM (local), sum fusion, adaptive soft-threshold abundance constraint (AACM) |
| loss | SAD + λ‖A‖½ + β‖A‖* per superpixel | 1·SAD + 0.3·MSE + 0.05·L½ |
| settings given | Samson/Jasper: 50 superpixels, m = 30, λ = 5e3, β = 1e-2, 400 epochs, lr 3e-2, wd 1e-3 | 200 superpixels, Adam lr 1e-2; epochs, dropout, k of kNN, decoder init not given |
| seeds | "each algorithm 10 times, average" (synthetic) | not stated |
| hyperparameter selection | grid search with the ground truth (Figs. 8–9) | "grid search" (Fig. 5: number of superpixels against ground-truth metrics) |
| code | promised at github.com/donghua898/MGA-Net; not found by web search | none given |

Published numbers (aRMSE / mSAD in radians):

| | Samson aRMSE | Samson mSAD | Jasper aRMSE | Jasper mSAD |
|---|---|---|---|---|
| MGA-Net | 0.0384 | 0.0228 | 0.0436 | 0.0335 |
| GAT-CNN-HU | 0.0353 | 0.0355 | 0.0784 | 0.0559 |
| VCA+FCLS as reported in MGA-Net | 0.1382 | 0.0710 | 0.2121 | 0.2975 |
| VCA+FCLS as reported in GAT-CNN-HU | 0.0783 | 0.0779 | 0.1521 | 0.1526 |
| TANet as reported in both papers | 0.2554 | 0.0738 | 0.1662 | 0.0837 / 0.0937 |

The same group reports VCA+FCLS on the same data with aRMSE 0.078 in one paper and 0.138 in the other.
Baseline numbers taken from these tables are not reliable. TANet's per-material numbers are identical
in both papers except one Jasper SAD (tree 0.0451 against 0.0851, one digit apart). They were probably
copied rather than rerun.

## 4. AE-SU (Ampas et al., GRSL 2025): quick test, seed 0 only

`unmix/deep.py: AESU` is copied from `AESU_Unmix_Embed_ZAZ_1.ipynb`: spatial and spectral conv encoders,
softmax abundances, endmembers frozen for 50 epochs, a convolutional refine layer on the reconstruction,
loss MSE + SAD + 0.01·entropy, gradient clipping at 1, 200 epochs, one Adam step per epoch.
"Reference init" starts the endmembers at the reference `M`, the analogue of the letter's
field-measured endmembers. One seed only; these numbers set direction and are not final.

| AE-SU variant | Samson aRMSE | Samson recon SAD | Jasper aRMSE | Jasper recon SAD |
|---|---|---|---|---|
| notebook settings (lr 1e-5), reference init | 0.485 | 0.640 | 0.382 | 0.749 |
| notebook settings (lr 1e-5), VCA init | 0.448 | 0.576 | 0.326 | 0.701 |
| lr 1e-3, refine layer | 0.427 | 0.068 | 0.344 | 0.097 |
| lr 1e-3, **no refine layer** (linear decoder) | **0.194** | 0.040 | **0.085** | 0.064 |
| *uniform abundances* | *0.375* | | *0.342* | |
| *FCLS, reference M (oracle)* | *0.410* | | *0.085* | |
| *VCA + NNLS-normalize, raw scale (blind, 10 seeds)* | *0.128* | | *0.157* | |

Abundances are the model's own output ("aligned" differs by at most 0.05 and is in the CSVs).

1. **The notebook settings do not train.** With lr 1e-5 and 200 steps the reconstruction is still poor
   (SAD 0.58–0.75 rad), and the abundances are no better than uniform.
2. **The refine layer makes the abundances unidentifiable.** At lr 1e-3 with the refine layer the
   reconstruction is good (SAD 0.07–0.10) but the abundances stay at uniform level: the convolution on
   E·A does the reconstruction instead of the abundances. The letter describes a strictly linear
   decoder; the notebook's decoder is not linear.
3. **Without the refine layer AE-SU works.** On Jasper it matches FCLS with the same (reference)
   endmembers, 0.085, and beats the blind VCA pipeline. On Samson it beats FCLS (0.194 vs 0.410, mostly
   because SAD is brightness-invariant) but not NNLS-normalize with the same endmembers (0.002, oracle).
4. **Training moves the endmembers away from the truth.** Starting at mSAD 0, the no-refine variant
   ends at 0.12 (Samson) and 0.18 (Jasper). Keeping the endmembers fixed, or training them more
   slowly, is the next thing to test. That is also the clean version of the letter's comparison:
   same known endmembers, AE-SU against FCLS and NNLS-normalize.
