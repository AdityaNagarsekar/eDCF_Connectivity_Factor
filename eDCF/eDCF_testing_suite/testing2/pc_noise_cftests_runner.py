#!/usr/bin/env python3
"""
pc_noise_cftests_runner.py

Generates point-cloud datasets using skdim.datasets.BenchmarkManifolds at
noise levels [0.01, 0.1, 0.2, 0.3, 0.4] and evaluates a scalar noise estimator
(based on local PCA). Saves results to CSV and prints a compact table.

This file is self-contained and does NOT depend on your custom pipeline. It uses only:
- numpy
- scikit-learn (NearestNeighbors)
- scikit-dimension (skdim) for dataset generation

Usage:
    python pc_noise_cftests_runner.py --n 4000 --k 50 --seed 42 --normalize \
        --planarity_q 0.25 --var_target 0.98 --out results_noise.csv

Notes:
- If --normalize is set, σ̂ is reported as a fraction of the bounding-box diagonal
  (handy to interpret "percent" noise). If you want raw units, omit --normalize.
- The estimator automatically handles ambient dimension d>3 by inferring a local
  intrinsic dimension m per patch via explained variance target (--var_target),
  and averaging the residual (d - m) eigenvalues to estimate noise variance.
"""

import argparse
import numpy as np
from sklearn.neighbors import NearestNeighbors
from skdim.datasets import BenchmarkManifolds
from tqdm import tqdm
import csv
import sys


def estimate_noise_sigma(points, ks=(30, 50, 80), normalize=True, planarity_q=0.25, var_target=0.98):
    """
    Estimate additive (roughly isotropic) noise σ from a point cloud (N,d) using local PCA.
    - ks: neighborhood sizes to try; results are pooled for robustness (multi-scale).
    - normalize: if True, scale to unit bounding-box diagonal so σ̂ is scale-invariant.
    - planarity_q: keep the flattest q-quantile neighborhoods (based on residual variance ratio).
    - var_target: explained variance target (0<var_target<1). Locally choose the smallest m such that
                  the largest-m eigenvalues explain at least var_target of the total variance.
                  The noise variance for a patch is then the mean of the remaining d-m eigenvalues;
                  if m=d, we fall back to the smallest eigenvalue.
    Returns a single float σ̂.
    Time: ~O(N log N + N * max(ks)) ; Memory: O(N) to O(N*max(ks)).
    """
    P = np.asarray(points, dtype=np.float64)
    N, d = P.shape
    if N < 8 or d < 2:
        return 0.0

    # Optional: normalize to unit bounding-box diagonal for scale-invariance
    if normalize:
        mn, mx = P.min(axis=0), P.max(axis=0)
        diag = np.linalg.norm(mx - mn)
        if diag > 0:
            P = (P - mn) / diag

    # Build KD-tree once at the maximum k we will need
    kmax = int(min(max(ks), N - 1))
    kmax = max(3, kmax)
    nn = NearestNeighbors(n_neighbors=kmax + 1, algorithm="auto").fit(P)

    # Accumulate per-patch noise-variance estimates and a flatness score
    noise_vars = []
    flat_scores = []

    for k in ks:
        k = int(max(3, min(k, N - 1)))
        idx = nn.kneighbors(P, n_neighbors=k + 1, return_distance=False)[:, 1:]  # (N,k)
        X = P[idx]                                   # (N,k,d)
        Xc = X - X.mean(axis=1, keepdims=True)       # center each patch
        C = Xc.transpose(0, 2, 1) @ Xc / float(k)    # (N,d,d) covariance per patch

        # Eigenvalues ascending per patch
        w = np.linalg.eigh(C)[0]                     # (N,d) ascending
        # Work in descending for explained-variance logic
        w_desc = w[:, ::-1]                          # (N,d) descending
        w_sum = w_desc.sum(axis=1, keepdims=True)    # (N,1)

        # Avoid divide-by-zero in pathological degenerate patches
        w_sum_safe = np.where(w_sum <= 0, 1.0, w_sum)

        # Compute cumulative explained variance ratio along descending eigenvalues
        cums = np.cumsum(w_desc, axis=1) / w_sum_safe  # (N,d)

        # Smallest m with >= var_target explained variance
        # m_idx is in [0, d-1]; m = m_idx+1
        m_idx = (cums >= var_target).argmax(axis=1)
        m = m_idx + 1
        # Number of residual eigenvalues (treated as noise-only space)
        r = w.shape[1] - m

        # For each patch, compute residual variance mean; if r==0, fallback to smallest eigenvalue
        # Note: residual eigenvalues are the r smallest in DESC order => indices [-r:] in ASC order
        # In ASC order w, the residual set is w[:, :r] (the r smallest)
        if np.any(r > 0):
            # For patches with r>0:
            # Select w_small = w[:, :r], but r varies per row; we handle via loop for clarity.
            for i in range(N):
                ri = int(r[i]) if np.ndim(r) else int(r)
                if ri > 0:
                    nv = float(np.mean(w[i, :ri]))
                    noise_vars.append(nv)
                    # Residual variance ratio as flatness score
                    flat = float((w[i, :ri].sum()) / (w[i, :].sum() + 1e-12))
                    flat_scores.append(flat)
                else:
                    # Fallback
                    nv = float(w[i, 0])
                    noise_vars.append(nv)
                    flat = float(w[i, 0] / (w[i, :].sum() + 1e-12))
                    flat_scores.append(flat)
        else:
            # All patches have r==0 (intrinsic dimension ~ ambient); fallback to smallest eigenvalue
            nv = w[:, 0]
            noise_vars.extend([float(x) for x in nv])
            flat = w[:, 0] / (w.sum(axis=1) + 1e-12)
            flat_scores.extend([float(x) for x in flat])

    noise_vars = np.asarray(noise_vars)
    flat_scores = np.asarray(flat_scores)

    # Keep the flattest patches by quantile on residual ratio
    thr = np.quantile(flat_scores, planarity_q)
    kept = noise_vars[flat_scores <= thr]
    if kept.size == 0:
        kept = noise_vars

    # Return σ̂ as sqrt of a robust aggregate of variance estimates
    return float(np.sqrt(np.median(kept)))


def run(noise_levels, n=4000, k=50, seed=42, normalize=True, planarity_q=0.25, var_target=0.98, out_csv="results_noise.csv"):
    bm = BenchmarkManifolds(random_state=seed)
    headers = ["manifold", "N", "ambient_dim", "noise_param", "sigma_hat", "abs_error", "rel_error"]

    rows = []
    print(f"\nGenerating manifolds with N={n} at noise levels {noise_levels}")
    for nz in noise_levels:
        # skdim: name='all' returns a dict of {str: np.ndarray}
        manifolds = bm.generate(name='all', n=n, noise=float(nz))
        for name, data in tqdm(manifolds.items(), desc=f"noise={nz:.3f}"):
            try:
                sigma_hat = estimate_noise_sigma(
                    data,
                    ks=(min(30, data.shape[0]-1), min(k, data.shape[0]-1), min(80, data.shape[0]-1)),
                    normalize=normalize,
                    planarity_q=planarity_q,
                    var_target=var_target
                )
                # If normalize=True and skdim's noise parameter is a scale-like σ, then a reasonable
                # proxy for "truth" is the nz we passed. If you want raw units, set normalize=False.
                true_sigma = float(nz) if normalize else float(nz)
                abs_err = abs(sigma_hat - true_sigma)
                rel_err = abs_err / (true_sigma + 1e-12) if true_sigma > 0 else np.nan

                rows.append([name, data.shape[0], data.shape[1], true_sigma, sigma_hat, abs_err, rel_err])
            except Exception as e:
                rows.append([name, data.shape[0], data.shape[1], nz, "ERROR", "NA", "NA"])
                print(f"[warn] {name} failed: {e}", file=sys.stderr)

    # Save CSV
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    # Print a compact preview
    print(f"\nSaved results to {out_csv}\n")
    # Show a few lines per noise level
    by_noise = {}
    for r in rows:
        by_noise.setdefault(r[3], []).append(r)
    for nz, rs in by_noise.items():
        print(f"--- noise={nz} ---")
        for r in rs[:5]:
            name, N, d, nzv, sh, ae, re = r
            print(f"{name:<22} N={N:<6} d={d:<3} σ̂={sh:<.4f}  |  abs={ae:<.4f}  rel={re if isinstance(re, float) else re}")
        print("")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4000, help="points per manifold")
    parser.add_argument("--k", type=int, default=50, help="k-NN size")
    parser.add_argument("--seed", type=int, default=42, help="random seed for skdim BenchmarkManifolds")
    parser.add_argument("--normalize", action="store_true", help="normalize to unit bounding-box diagonal")
    parser.add_argument("--planarity_q", type=float, default=0.25, help="quantile for flattest patches [0..1]")
    parser.add_argument("--var_target", type=float, default=0.98, help="explained variance target to infer local intrinsic dimension")
    parser.add_argument("--out", type=str, default="results_noise.csv", help="output CSV file path")
    args = parser.parse_args()

    noise_levels = [0.01, 0.1, 0.2, 0.3, 0.4]
    run(
        noise_levels=noise_levels,
        n=args.n,
        k=args.k,
        seed=args.seed,
        normalize=args.normalize,
        planarity_q=args.planarity_q,
        var_target=args.var_target,
        out_csv=args.out,
    )


if __name__ == "__main__":
    main()
