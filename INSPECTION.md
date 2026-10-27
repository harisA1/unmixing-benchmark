# Data inspection (27 October 2026)

First look at the files provided for the unmixing benchmark. Data files are not in this repository;
keep them in a git-ignored `data/` folder.

## Files

| file | contents |
|---|---|
| `jasperRidge2_R198.mat` | `Y` (198 bands × 10,000 pixels, uint16, max 5437), `SlectBands` (the 198 kept of 224), `nRow = nCol = 100`, `maxValue = 5000` |
| `jasperRidge2_F224_2.mat` | the same scene with all 224 bands. Checked: `R198.Y == F224.Y[SlectBands - 1]` exactly |
| `gt_jasper_end4.mat` | `M` (198 × 4 endmembers, 0–0.63), `A` (4 × 10,000 abundances), labels tree / water / dirt / road |
| `gt_samson_end3.mat` | `M` (156 × 3), `A` (3 × 9,025), labels rock / tree / water |
| `bare_soil_unmixing.ipynb` | exploratory notebook: least-squares unmixing on Jasper, noise tests, bare-soil maps, spectral indices |

**Missing: the Samson image itself** (usually `samson_1.mat`, key `V`, 156 × 9,025). Only its ground truth
was provided.

## Checks

- Jasper ground-truth abundances are exactly sum-to-one, and **43% of the entries are exactly zero**
  (most materials are absent from most pixels).
- `M @ A` reconstructs `Y / 5000` with a relative residual of about 17%. That's expected: the "ground truth"
  is itself an unmixing result and the scaling is approximate. It means the reference abundances are a
  reference, not truth. State this in any paper.
- Pixel order: the `.mat` files come from MATLAB (column-major). Reshaping with `order='F'` gives the
  MATLAB orientation; C order gives the transpose. The metrics don't depend on it, but plots and any spatial
  model must use one convention consistently.

## Issues in the notebook (worth fixing before benchmarking)

1. **Per-band min–max normalization breaks the linear mixing model.** It rescales each band of `Y`
   differently but not the endmembers `M`, so `Y ≈ M A` no longer holds. Measured on Jasper with the
   ground-truth `M`:

   | preprocessing | abundance RMSE |
   |---|---|
   | `Y / 5000` (physical scale) | **0.077** |
   | per-band min–max (as in the notebook) | 0.256 |
   | uniform 1/4 (trivial baseline) | 0.350 |

   Use one global scale factor for `Y` (e.g. `maxValue`), or normalize `Y` and `M` identically.
2. **Pseudo-inverse + clip + renormalize is not FCLS.** It's a fast approximation; proper fully
   constrained least squares (non-negativity and sum-to-one inside the solver) should be the classical
   baseline.
3. **`add_snr(Y, 0.01)` adds noise at 0.01 dB** (noise power ≈ signal power), not 10 dB or "1%". Check
   whether that was intended. The noise also has no fixed random seed, so results aren't reproducible.
4. **R² on binary maps** (thresholded abundances) isn't very meaningful; RMSE on abundances plus
   precision / recall / F1 on the maps is cleaner.

## Lessons carried over from the COMP-GCN project

- Tune classical baselines properly (FCLS, SUnSAL-TV); many deep-unmixing comparisons don't.
- Never select hyperparameters with the ground truth. Report "oracle" (ground-truth-tuned) and "honest"
  (reconstruction-error-tuned) results side by side.
- Use 10 seeds and report the spread; autoencoder unmixing is very initialization-sensitive.
- Report several metrics (abundance RMSE, SAD for endmembers, reconstruction error, per-material errors);
  rankings can change between them.
- Linear mixing means abundances combine **arithmetically**, so spatial smoothing should average
  arithmetically. Aitchison / log-ratio geometry doesn't fit the physics and can't handle the many zeros.
- A pixel's own spectrum carries most of the information; spatial terms should refine it, not replace it.

## Suggested first prompt for the next session

> Set up this repo as a fair benchmark for hyperspectral unmixing on Samson and Jasper Ridge. Read
> INSPECTION.md first. Data files (uploaded) go in a git-ignored `data/` folder with a loader that
> verifies shapes and checksums and uses physical scaling (Y / maxValue), not per-band normalization.
> Step 1, then stop and report: implement and evaluate (a) uniform-abundance and ground-truth-endmember
> FCLS baselines, (b) SUnSAL-TV, and (c) two popular deep blind-unmixing autoencoders with public code
> (suggest which, with citations I can check). Protocol: the same small hyperparameter grid for every
> method; two selection regimes reported side by side, "oracle" (best abundance RMSE against ground truth)
> and "honest" (best reconstruction error, no ground truth); 10 seeds; metrics: abundance RMSE overall
> and per material, endmember SAD (blind methods), reconstruction RMSE. First reproduce each method's
> published numbers approximately and report any large mismatch before running the full comparison.
> Commit to a branch, not main.
