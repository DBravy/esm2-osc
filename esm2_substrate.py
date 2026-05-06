"""
ESM-2 Substrate Characterization
================================

Computes the input-invariant oscillatory substrate of ESM-2's residual
stream from a corpus of masked protein positions.

For each (protein, masked_position) input, the residual stream is captured
at the masked position across all sublayer LayerNorm inputs (2 per block:
pre-attention LN and pre-MLP LN). After trimming K sublayers from each
end, the script computes:

  Per-unit phase-space diagnostics ("is this a clean oscillator?"):
    r_per_unit       median Pearson(x, dx) across inputs.
    A_norm_per_unit  median signed shoelace area across inputs.
    R_per_unit       median PCA aspect across inputs (frequency-invariant).
    dtheta_per_unit  circular mean of Hilbert-derived quadrature offset.

  Per-(unit, sublayer) substrate observables ("what is the carrier doing?"):
    amp_mean(L, D)        <|z|> across inputs.
    phase_resultant(L, D) |<exp(i*phase)>| across inputs (1 = locked).

  Pairwise coupling ("who locks with whom?"):
    C(D, D) complex: C[u,v] = <exp(i*(theta_u - theta_v))> over (n, l).
    |C| is phase consistency; arg(C) is canonical phase offset.
    Spectral cluster on |C|. Hub centrality from dominant eigenvector
    of |C|.

  Tier-2 channel variance maps (free byproduct of the same forward passes):
    amp_var(L, D)         std(|z|) across inputs.
    phase_circ_var(L, D)  1 - phase_resultant.
    freq_var(L-1, D)      std(unwrapped phase diff) across inputs.
    harm2_amp_var(D)      std(|FFT(x)[2]|) across inputs.

Outputs (in --output-dir):
  substrate.npz                       All arrays for downstream pipeline.
  substrate_summary.json              Per-checkpoint stats, top hubs, cluster sizes.
  phase_portraits.png                 9-panel grid of representative units.
  diagnostics.png                     Histograms of r, A_norm, R_PCA, dtheta.
  substrate_amplitude_profile.png     Heatmap <|z(l)|>, sorted by hub centrality.
  channel_variance_maps.png           Four Tier-2 maps in a 2x2 grid.
  coupling_matrix.png                 |C| reordered by Fiedler vector.
  laplacian_spectrum.png              Eigenvalue spectrum of L = D - |C|.
  fiedler_and_top_eigenvectors.png    Top-k low Laplacian eigenvectors.

Usage:
  # Sanity check on ubiquitin alone.
  python esm2_substrate.py --single-protein --n-samples 76

  # Diverse corpus (recommended).
  python esm2_substrate.py --fasta proteins.fasta --n-samples 1000

  # All knobs:
  python esm2_substrate.py \\
      --fasta proteins.fasta \\
      --n-samples 1000 --trim-sublayers 2 --n-clusters 4 \\
      --top-k-eigvecs 10 --max-len 512 \\
      --output-dir esm2_substrate_v1 \\
      --hf-home /Volumes/ORICO/huggingface_cache
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.signal import hilbert as scipy_hilbert
from scipy.cluster.vq import kmeans2


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "facebook/esm2_t33_650M_UR50D"
SEED = 0

UBIQUITIN_SEQ = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
)

CANONICAL_AAS = set("ACDEFGHIKLMNPQRSTVWY")


def get_device(override=None):
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# FASTA loading and input sampling
# ---------------------------------------------------------------------------

def parse_fasta(path):
    """Yield (name, sequence) from a FASTA file."""
    name, chunks = None, []
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks)
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line.upper())
    if name is not None:
        yield name, "".join(chunks)


def load_sequences(args):
    """Build a list of (name, sequence) pairs from CLI args, filtering for
    canonical-AA-only sequences within the length window."""
    if args.single_protein:
        seqs = [("ubiquitin", UBIQUITIN_SEQ)]
    elif args.protein_sequence:
        seqs = [("user_seq", args.protein_sequence.upper())]
    elif args.fasta:
        seqs = list(parse_fasta(args.fasta))
        if not seqs:
            raise ValueError(f"No sequences found in {args.fasta}")
    else:
        raise ValueError(
            "Must provide one of --fasta, --single-protein, --protein-sequence."
        )

    kept, dropped_aa, dropped_len = [], 0, 0
    for name, seq in seqs:
        if len(seq) < args.min_len or len(seq) > args.max_len:
            dropped_len += 1
            continue
        if any(c not in CANONICAL_AAS for c in seq):
            dropped_aa += 1
            continue
        kept.append((name, seq))

    print(f"  Loaded {len(seqs)} sequences; kept {len(kept)} after filtering "
          f"(dropped {dropped_len} by length, {dropped_aa} by non-canonical AAs).")
    if not kept:
        raise ValueError("No sequences passed filters.")
    return kept


def sample_inputs(sequences, n_samples, rng):
    """Sample N (seq_idx, position) pairs uniformly across all valid positions."""
    pool = []
    for si, (_, seq) in enumerate(sequences):
        for pi in range(len(seq)):
            pool.append((si, pi))
    if not pool:
        raise ValueError("No valid positions available.")
    if n_samples >= len(pool):
        print(f"  n_samples={n_samples} >= total positions={len(pool)}; "
              f"using all {len(pool)}.")
        return pool
    idx = rng.choice(len(pool), n_samples, replace=False)
    return [pool[i] for i in idx]


# ---------------------------------------------------------------------------
# ESM-2 hook setup and stream collection
# ---------------------------------------------------------------------------

def get_esm2_hook_targets(model):
    """For ESM-2 (pre-LN), per layer the relevant LN modules are:
      - layer.attention.LayerNorm   (pre-attention)
      - layer.LayerNorm              (pre-MLP)
    Order: layer 0 attn LN, layer 0 mlp LN, layer 1 attn LN, ...
    """
    if not hasattr(model, "esm") or not hasattr(model.esm, "encoder"):
        raise RuntimeError(
            f"Could not locate ESM encoder in {type(model).__name__}; "
            "expected `model.esm.encoder.layer[i]`."
        )
    targets = []
    for block in model.esm.encoder.layer:
        if not hasattr(block, "attention") or not hasattr(block.attention, "LayerNorm"):
            raise RuntimeError(
                "Block missing attention.LayerNorm; ESM-2 model layout has changed."
            )
        if not hasattr(block, "LayerNorm"):
            raise RuntimeError(
                "Block missing top-level LayerNorm; ESM-2 model layout has changed."
            )
        targets.append(block.attention.LayerNorm)
        targets.append(block.LayerNorm)
    return targets


def collect_streams(model, tokenizer, sequences, samples, device, log_every=64):
    """For each (seq_idx, pos) input, mask the position, forward-pass, and
    capture the residual stream at the masked position across all
    sublayer LayerNorm inputs.

    Returns array of shape (N, n_sub, d_model), float32.
    """
    model.eval()
    targets = get_esm2_hook_targets(model)
    streams = []
    n_total = len(samples)

    for i, (si, pos) in enumerate(samples):
        seq = sequences[si][1]
        enc = tokenizer(seq, return_tensors="pt")
        ids = enc["input_ids"].to(device)
        attn_mask = enc.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)
        # ESM-2 prepends CLS so masked-position index is pos + 1.
        tidx = pos + 1
        if tidx >= ids.shape[1] - 1:
            # Should not happen given length filter, but guard anyway.
            continue
        ids[0, tidx] = tokenizer.mask_token_id

        captures = []
        hooks = [
            t.register_forward_hook(
                lambda m, inp, out, c=captures: c.append(inp[0].detach())
            )
            for t in targets
        ]
        try:
            with torch.no_grad():
                if attn_mask is not None:
                    model(input_ids=ids, attention_mask=attn_mask)
                else:
                    model(input_ids=ids)
        finally:
            for h in hooks:
                h.remove()

        if len(captures) != len(targets):
            raise RuntimeError(
                f"Captured {len(captures)} streams, expected {len(targets)} "
                "(hook firing inconsistent)."
            )

        # Each capture has shape (1, seq_len_with_special, d_model).
        # Slice out the masked-token position.
        snap = torch.stack(
            [c[0, tidx, :].float() for c in captures], dim=0
        ).cpu().numpy()  # (n_sub, d_model)
        streams.append(snap)

        if (i + 1) % log_every == 0 or (i + 1) == n_total:
            print(f"    [{i + 1}/{n_total}] seq={sequences[si][0]} pos={pos}")

    return np.stack(streams, axis=0)  # (N, n_sub, d_model)


# ---------------------------------------------------------------------------
# Phase-space diagnostics (same conventions as phase_space_llm.py)
# ---------------------------------------------------------------------------

def pair_x_dx(x):
    """x: (N, L, D). Returns (x_pair, dx) each (N, L-1, D)."""
    dx = np.diff(x, axis=1)
    x_pair = x[:, :-1, :]
    return x_pair, dx


def per_traj_pearson(x, dx):
    """Pearson(x, dx) along L, per (input, unit). (N, L-1, D) -> (N, D)."""
    x_c = x - x.mean(axis=1, keepdims=True)
    dx_c = dx - dx.mean(axis=1, keepdims=True)
    num = (x_c * dx_c).sum(axis=1)
    den = np.sqrt((x_c ** 2).sum(axis=1) * (dx_c ** 2).sum(axis=1)) + 1e-30
    return num / den


def per_traj_signed_area(x, dx):
    """Shoelace signed area of (x_l, dx_l), per (input, unit)."""
    if x.shape[1] < 2:
        return np.zeros((x.shape[0], x.shape[2]))
    a = x[:, :-1, :] * dx[:, 1:, :]
    b = x[:, 1:, :] * dx[:, :-1, :]
    return 0.5 * (a - b).sum(axis=1)


def per_traj_pca_aspect(x, dx):
    """PCA aspect of (x, dx) after rescaling dx so var(dx_rescaled)=var(x).
    Frequency-invariant; equals (1 - |r|) / (1 + |r|)."""
    x_c = x - x.mean(axis=1, keepdims=True)
    dx_c = dx - dx.mean(axis=1, keepdims=True)
    var_x = (x_c ** 2).mean(axis=1)
    var_dx = (dx_c ** 2).mean(axis=1)
    cov_xy = (x_c * dx_c).mean(axis=1)
    s = np.sqrt(var_x / (var_dx + 1e-30))
    cov_rescaled = s * cov_xy
    lam_max = var_x + np.abs(cov_rescaled)
    lam_min = np.maximum(var_x - np.abs(cov_rescaled), 0.0)
    return lam_min / (lam_max + 1e-30)


def per_traj_quadrature_phase(x, dx):
    """Mean phase offset arg(Hilbert(dx)) - arg(Hilbert(x)), per (input, unit)."""
    z_x = scipy_hilbert(x, axis=1)
    z_dx = scipy_hilbert(dx, axis=1)
    ratio = z_dx / (z_x + 1e-30)
    return np.angle(ratio.mean(axis=1))


def collect_per_unit_diagnostics(streams):
    """Apply the four phase-space diagnostics, aggregate across inputs.
    Returns the per-unit summary plus the raw (x_pair, dx) arrays.
    Note: streams are assumed already centered/trimmed."""
    x_pair, dx = pair_x_dx(streams)
    r = per_traj_pearson(x_pair, dx)
    A = per_traj_signed_area(x_pair, dx)
    R = per_traj_pca_aspect(x_pair, dx)
    dtheta = per_traj_quadrature_phase(x_pair, dx)

    std_x = x_pair.std(axis=1)
    std_dx = dx.std(axis=1)
    L_pair = x_pair.shape[1]
    A_norm = A / ((std_x * std_dx * L_pair) + 1e-30)

    r_per_unit = np.median(r, axis=0)
    A_norm_per_unit = np.median(A_norm, axis=0)
    R_per_unit = np.median(R, axis=0)
    dtheta_complex = np.exp(1j * dtheta)
    dtheta_resultant = np.abs(dtheta_complex.mean(axis=0))
    dtheta_per_unit = np.angle(dtheta_complex.mean(axis=0))

    return {
        "r_per_unit": r_per_unit.astype(np.float32),
        "A_norm_per_unit": A_norm_per_unit.astype(np.float32),
        "R_per_unit": R_per_unit.astype(np.float32),
        "dtheta_per_unit": dtheta_per_unit.astype(np.float32),
        "dtheta_resultant_per_unit": dtheta_resultant.astype(np.float32),
        "x_pair": x_pair,
        "dx": dx,
    }


def classify_units(per_unit, r_thresh=0.30, pca_thresh=0.40, dtheta_thresh=np.pi / 4):
    r_abs = np.abs(per_unit["r_per_unit"])
    R = per_unit["R_per_unit"]
    dt_dev = np.abs(np.abs(per_unit["dtheta_per_unit"]) - np.pi / 2)
    is_osc = (r_abs < r_thresh) & (R > pca_thresh) & (dt_dev < dtheta_thresh)
    is_degenerate = R < pca_thresh / 2
    is_other = ~is_osc & ~is_degenerate
    return is_osc, is_degenerate, is_other


# ---------------------------------------------------------------------------
# Substrate observables and Tier-2 channel-variance maps
# ---------------------------------------------------------------------------

def compute_analytic_signal(x):
    """Hilbert transform along the L axis. x: (N, L, D) real -> (N, L, D) complex."""
    return scipy_hilbert(x, axis=1)


def compute_substrate_and_variance(streams):
    """One-pass computation of:
      - substrate amplitude profile <|z|>(L, D)
      - phase resultant length |<exp(i*phase)>|(L, D)
      - amplitude variance std(|z|)(L, D)
      - phase circular variance 1 - resultant_length(L, D)
      - frequency variance std(unwrapped diff)(L-1, D)
      - second-harmonic amplitude per (input, unit), and its variance.
      - phases (N, L, D) for the coupling-matrix step.

    Returns dict + (phases array kept separately because it's large).
    """
    x = streams.astype(np.float32)  # (N, L, D)
    z = compute_analytic_signal(x)  # (N, L, D) complex
    amp = np.abs(z)
    phase = np.angle(z)

    amp_mean = amp.mean(axis=0)
    amp_var = amp.std(axis=0)

    phase_complex = np.exp(1j * phase)
    phase_resultant = np.abs(phase_complex.mean(axis=0))
    phase_circ_var = 1.0 - phase_resultant

    # Instantaneous angular velocity from unwrapped phase along the L axis.
    phase_unwr = np.unwrap(phase, axis=1)
    omega = np.diff(phase_unwr, axis=1)  # (N, L-1, D)
    freq_mean = omega.mean(axis=0)
    freq_var = omega.std(axis=0)

    # Harmonic content via FFT along the L axis. Bin 1 is the fundamental
    # (one cycle across the depth window after trimming); bin 2 is the
    # second harmonic.
    X = np.fft.rfft(x, axis=1)  # (N, L_fft, D)
    if X.shape[1] >= 3:
        harm2_amp = np.abs(X[:, 2, :])  # (N, D)
    else:
        harm2_amp = np.zeros((x.shape[0], x.shape[2]), dtype=np.float32)
    harm2_amp_mean = harm2_amp.mean(axis=0)
    harm2_amp_var = harm2_amp.std(axis=0)

    return {
        "amp_mean":          amp_mean.astype(np.float32),
        "amp_var":           amp_var.astype(np.float32),
        "phase_resultant":   phase_resultant.astype(np.float32),
        "phase_circ_var":    phase_circ_var.astype(np.float32),
        "freq_mean":         freq_mean.astype(np.float32),
        "freq_var":          freq_var.astype(np.float32),
        "harm2_amp_mean":    harm2_amp_mean.astype(np.float32),
        "harm2_amp_var":     harm2_amp_var.astype(np.float32),
        "phase":             phase,  # (N, L, D), float32
    }


# ---------------------------------------------------------------------------
# Coupling matrix C[u, v] = <exp(i*(theta_u - theta_v))> over (n, l)
# ---------------------------------------------------------------------------

def compute_coupling_matrix(phase, chunk_size=64):
    """Memory-bounded accumulation of the complex coupling matrix.

    phase: (N, L, D) real (radians).
    Returns: C (D, D) complex64; |C|, arg(C) derived downstream.
    Hermitian by construction; explicitly symmetrized to suppress drift.
    """
    N, L, D = phase.shape
    accum = np.zeros((D, D), dtype=np.complex128)
    total = N * L
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk = phase[start:end].astype(np.float64)              # (m, L, D)
        z = np.exp(1j * chunk).reshape(-1, D)                    # (m*L, D)
        # accum[u, v] += sum_k z[k, u] * conj(z[k, v]).
        accum += z.T @ z.conj()
    C = accum / total
    C = 0.5 * (C + C.conj().T)
    return C.astype(np.complex64)


# ---------------------------------------------------------------------------
# Spectral analysis of |C|
# ---------------------------------------------------------------------------

def laplacian_spectral(coupling_abs, k=10):
    """Eigen-decompose L = D_diag - |C| (combinatorial Laplacian).
    Returns ascending eigvals/eigvecs plus the Fiedler (eigvecs[:, 1])
    and a top-k low-end slice."""
    D = coupling_abs.shape[0]
    A = coupling_abs.copy().astype(np.float64)
    np.fill_diagonal(A, 0.0)
    A = 0.5 * (A + A.T)
    deg = A.sum(axis=1)
    L = np.diag(deg) - A
    w, V = np.linalg.eigh(L)
    return {
        "eigvals":         w.astype(np.float32),
        "eigvecs":         V.astype(np.float32),
        "fiedler_val":     float(w[1]),
        "fiedler_vec":     V[:, 1].astype(np.float32),
        "top_k_eigvals":   w[1:k + 1].astype(np.float32),
        "top_k_eigvecs":   V[:, 1:k + 1].astype(np.float32),
        "degree":          deg.astype(np.float32),
    }


def hub_centrality(coupling_abs):
    """Eigenvector centrality: dominant eigenvector of |C|.
    Made non-negative (Perron-Frobenius applies because |C| is non-negative)
    and L1-normalized."""
    A = coupling_abs.copy().astype(np.float64)
    np.fill_diagonal(A, 0.0)
    A = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(A)
    v = V[:, -1]
    if v.sum() < 0:
        v = -v
    v = np.maximum(v, 0)
    s = v.sum()
    if s > 0:
        v = v / s
    return v.astype(np.float32)


def cluster_units_spectral(top_k_eigvecs, n_clusters, rng_seed=0):
    """Ng-Jordan-Weiss spectral clustering on the row-normalized top-k
    Laplacian eigenvectors."""
    rows = top_k_eigvecs.copy().astype(np.float64)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    rows = rows / np.maximum(norms, 1e-12)
    np.random.seed(rng_seed)
    centroids, labels = kmeans2(rows, n_clusters, minit="++", seed=rng_seed)
    return labels.astype(np.int32), centroids.astype(np.float32)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _draw_cov_ellipse(ax, x_pts, y_pts, n_sigma=2.0, color="red",
                      alpha=0.7, lw=1.5):
    if len(x_pts) < 3:
        return
    cov = np.cov(x_pts, y_pts)
    eigvals, eigvecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
    semi_a = n_sigma * np.sqrt(max(eigvals[1], 0.0))
    semi_b = n_sigma * np.sqrt(max(eigvals[0], 0.0))
    e = Ellipse(
        xy=(x_pts.mean(), y_pts.mean()),
        width=2 * semi_a, height=2 * semi_b, angle=angle,
        fill=False, edgecolor=color, alpha=alpha, linewidth=lw,
    )
    ax.add_patch(e)


def plot_phase_portraits(per_unit, x_pair, dx, out_path):
    A = np.abs(per_unit["A_norm_per_unit"])
    order = np.argsort(A)
    D = len(order)
    if D < 9:
        chosen = list(order[-min(D, 9):])
        chosen_labels = [f"u{u}" for u in chosen]
    else:
        chosen = [
            order[-1], order[-2], order[-3],
            order[D // 2 - 1], order[D // 2], order[D // 2 + 1],
            order[0], order[1], order[2],
        ]
        chosen_labels = (
            ["high |A|"] * 3 + ["median |A|"] * 3 + ["low |A|"] * 3
        )

    N, Lm1, _ = x_pair.shape
    layer_norm = np.arange(Lm1) / max(Lm1 - 1, 1)

    fig, axes = plt.subplots(3, 3, figsize=(15, 14))
    for ax, u, lbl in zip(axes.flatten(), chosen, chosen_labels):
        xs = x_pair[:, :, u].flatten()
        dys = dx[:, :, u].flatten()
        cs = np.tile(layer_norm, N)
        ax.scatter(xs, dys, c=cs, cmap="viridis",
                   s=3, alpha=0.12, edgecolors="none")
        ax.plot(x_pair[0, :, u], dx[0, :, u], "-",
                color="black", linewidth=0.8, alpha=0.8)
        ax.scatter(x_pair[0, 0, u], dx[0, 0, u], marker="o", s=40,
                   facecolor="white", edgecolor="black",
                   linewidth=1.0, zorder=5)
        _draw_cov_ellipse(ax, xs, dys, n_sigma=2.0)

        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
        ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)
        ax.set_xlabel(r"$x_\ell$")
        ax.set_ylabel(r"$\Delta x_\ell$")
        ax.set_title(
            f"unit {u} ({lbl})\n"
            f"r={per_unit['r_per_unit'][u]:+.2f}, "
            f"$A_{{\\mathrm{{norm}}}}$={per_unit['A_norm_per_unit'][u]:+.3f}, "
            f"$R$={per_unit['R_per_unit'][u]:.2f}, "
            f"$\\Delta\\theta$={per_unit['dtheta_per_unit'][u]:+.2f}",
            fontsize=9,
        )

    fig.suptitle(
        "Substrate phase portraits $(x_\\ell, \\Delta x_\\ell)$\n"
        "color = sublayer fraction (dark = early, bright = late); "
        "white circle = trajectory start; red = 2$\\sigma$ cov ellipse",
        fontsize=12, y=0.998,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_diagnostics(per_unit, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.hist(per_unit["r_per_unit"], bins=40, range=(-1, 1),
            color="C0", alpha=0.85)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0,
               label="oscillator ideal (r=0)")
    ax.set_xlabel(r"Pearson $r$ between $x$ and $\Delta x$")
    ax.set_ylabel("number of units")
    ax.set_title("(a) correlation: 0 = oscillator, $\\pm 1$ = degenerate")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    A = per_unit["A_norm_per_unit"]
    bound = max(np.abs(A).max(), 1e-6)
    ax.hist(A, bins=40, range=(-bound, bound), color="C2", alpha=0.85)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"normalized signed area $A_{\mathrm{norm}}$")
    ax.set_ylabel("number of units")
    ax.set_title("(b) signed area: sign = direction; magnitude = strength")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.hist(per_unit["R_per_unit"], bins=40, range=(0, 1),
            color="C3", alpha=0.85)
    ax.set_xlabel(r"$R_{\mathrm{PCA}}$ (rescaled aspect)")
    ax.set_ylabel("number of units")
    ax.set_title("(c) PCA aspect: 1 = circle, 0 = line")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.hist(per_unit["dtheta_per_unit"], bins=40,
            range=(-np.pi, np.pi), color="C4", alpha=0.85)
    for v in (-np.pi / 2, np.pi / 2):
        ax.axvline(v, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"$\Delta\theta$ (rad)")
    ax.set_ylabel("number of units")
    ax.set_title("(d) Hilbert quadrature: $\\pm\\pi/2$ = ideal cosine osc")
    ax.grid(alpha=0.3)

    fig.suptitle("Per-unit substrate diagnostics", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_substrate_amplitude_profile(amp_mean, hub_central, out_path,
                                      top_units=None):
    """Heatmap of <|z(l)|> with units sorted by hub centrality (descending),
    sublayers on the x axis."""
    L, D = amp_mean.shape
    order = np.argsort(hub_central)[::-1]
    if top_units is not None and top_units < D:
        order = order[:top_units]
    M = amp_mean[:, order].T  # (D_kept, L), units on y, sublayers on x

    fig, ax = plt.subplots(figsize=(max(8, L * 0.18), max(6, len(order) * 0.012)))
    im = ax.imshow(M, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_xlabel("sublayer (post-trim index, 0 = first kept)")
    ax.set_ylabel(f"unit (sorted by hub centrality, top {len(order)})")
    ax.set_title(
        r"Substrate amplitude profile $\langle |z_u(\ell)| \rangle$ "
        "(units descending by hub centrality)"
    )
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_channel_variance_maps(sub, out_path, top_units=None):
    """2x2 grid: amp_var, phase_circ_var, freq_var, harm2_amp_var.
    Units sorted by hub centrality (passed externally as sub['_hub_central']).
    Sublayers on x axis, units on y axis."""
    hub = sub["_hub_central"]
    order = np.argsort(hub)[::-1]
    if top_units is not None and top_units < len(order):
        order = order[:top_units]

    panels = [
        ("amp_var",        sub["amp_var"],        "amplitude std($|z|$)"),
        ("phase_circ_var", sub["phase_circ_var"], "phase circular var (1 - resultant)"),
        ("freq_var",       sub["freq_var"],       "frequency std (rad/sublayer)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, (key, M, title) in zip(axes.flatten()[:3], panels):
        if M.ndim != 2:
            continue
        # M shape (L, D) (or (L-1, D) for freq_var)
        Msub = M[:, order].T
        im = ax.imshow(Msub, aspect="auto", cmap="viridis",
                       interpolation="nearest")
        ax.set_xlabel("sublayer")
        ax.set_ylabel(f"unit (top {len(order)} by hub centrality)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)

    # Fourth panel: per-unit harm2 variance as a 1D bar/scatter.
    ax = axes.flatten()[3]
    h2_var = sub["harm2_amp_var"]
    h2_mean = sub["harm2_amp_mean"]
    ax.scatter(h2_mean[order], h2_var[order],
               s=8, alpha=0.6, edgecolors="none")
    ax.set_xlabel(r"second-harmonic mean amplitude $\langle |X[2]_u| \rangle$")
    ax.set_ylabel(r"second-harmonic std across inputs")
    ax.set_title("harmonic-channel signal vs substrate")
    ax.grid(alpha=0.3)

    fig.suptitle("Tier-2 channel variance maps", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_coupling_matrix(coupling_abs, fiedler_vec, cluster_labels, out_path):
    """|C| reordered by Fiedler vector, with cluster boundaries marked."""
    order = np.argsort(fiedler_vec)
    C_ord = coupling_abs[order][:, order]
    labels_ord = cluster_labels[order]

    fig, ax = plt.subplots(figsize=(8, 7))
    vmax = float(np.percentile(np.abs(C_ord), 99))
    im = ax.imshow(C_ord, cmap="viridis", vmin=0, vmax=vmax,
                   aspect="equal", interpolation="nearest")
    # Cluster boundaries: where the sorted label changes.
    boundaries = np.where(np.diff(labels_ord) != 0)[0] + 0.5
    for b in boundaries:
        ax.axhline(b, color="white", linewidth=0.5, alpha=0.6)
        ax.axvline(b, color="white", linewidth=0.5, alpha=0.6)
    ax.set_title(
        r"$|C_{uv}|$ reordered by Fiedler vector"
        f" ({len(np.unique(cluster_labels))} clusters)"
    )
    ax.set_xlabel("unit (Fiedler-ordered)")
    ax.set_ylabel("unit (Fiedler-ordered)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_laplacian_spectrum(eigvals, out_path, max_show=50):
    fig, ax = plt.subplots(figsize=(10, 5))
    show = min(max_show, len(eigvals))
    ax.plot(np.arange(show), eigvals[:show], "o-", markersize=4)
    ax.set_xlabel("eigenvalue index (ascending)")
    ax.set_ylabel(r"$\lambda$")
    ax.set_title(
        f"Laplacian spectrum (lowest {show} of {len(eigvals)})\n"
        "spectral gap suggests cluster count"
    )
    ax.grid(alpha=0.3)
    # Mark biggest gap among the first 30
    head = eigvals[:min(30, len(eigvals))]
    if len(head) > 2:
        gaps = np.diff(head)
        k = int(np.argmax(gaps[1:])) + 1  # skip the gap from the trivial
        ax.axvline(k + 0.5, color="red", linestyle="--", alpha=0.6,
                   label=f"largest gap at k={k+1}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_top_eigenvectors(top_k_eigvecs, fiedler_vec, cluster_labels, out_path):
    """Heatmap of top-k Laplacian eigenvectors, units ordered by Fiedler."""
    order = np.argsort(fiedler_vec)
    M = top_k_eigvecs[order, :].T  # (k, D), reorder rows of orig (D, k)
    fig, ax = plt.subplots(figsize=(14, 0.45 * M.shape[0] + 1))
    vmax = float(np.abs(M).max())
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels([f"v_{i+2}" for i in range(M.shape[0])])
    ax.set_xlabel("unit (Fiedler-ordered)")
    ax.set_title(
        f"Top {M.shape[0]} non-trivial Laplacian eigenvectors "
        "(rows; v_2 = Fiedler)"
    )
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def build_summary(args, sequences, samples, n_sub_used, D, per_unit, sub,
                  spec, hub_central, cluster_labels):
    is_osc, is_deg, is_other = classify_units(
        per_unit,
        r_thresh=args.r_threshold,
        pca_thresh=args.pca_threshold,
        dtheta_thresh=args.dtheta_threshold,
    )
    cluster_sizes = {
        int(c): int((cluster_labels == c).sum())
        for c in np.unique(cluster_labels)
    }
    top_hubs = np.argsort(hub_central)[::-1][:20].tolist()

    return {
        "model": args.model,
        "device": args.device or get_device(),
        "n_inputs": int(len(samples)),
        "n_sequences_in_corpus": int(len(sequences)),
        "n_sublayers_used_after_trim": int(n_sub_used),
        "trim_sublayers": int(args.trim_sublayers),
        "d_model": int(D),
        "thresholds": {
            "r_threshold": float(args.r_threshold),
            "pca_threshold": float(args.pca_threshold),
            "dtheta_threshold_radians": float(args.dtheta_threshold),
        },
        "phase_space_classification": {
            "n_oscillator": int(is_osc.sum()),
            "n_degenerate": int(is_deg.sum()),
            "n_other": int(is_other.sum()),
            "fraction_oscillator": float(is_osc.sum() / D),
            "fraction_degenerate": float(is_deg.sum() / D),
        },
        "per_unit_aggregates": {
            "r": {
                "abs_mean": float(np.abs(per_unit["r_per_unit"]).mean()),
                "abs_median": float(np.median(np.abs(per_unit["r_per_unit"]))),
            },
            "A_norm": {
                "abs_mean": float(np.abs(per_unit["A_norm_per_unit"]).mean()),
                "median": float(np.median(per_unit["A_norm_per_unit"])),
            },
            "R_PCA": {
                "mean": float(per_unit["R_per_unit"].mean()),
                "median": float(np.median(per_unit["R_per_unit"])),
            },
            "dtheta": {
                "deviation_from_pi_over_2": {
                    "mean": float(np.abs(np.abs(per_unit["dtheta_per_unit"]) - np.pi / 2).mean()),
                    "median": float(np.median(np.abs(np.abs(per_unit["dtheta_per_unit"]) - np.pi / 2))),
                },
                "resultant_length_mean": float(per_unit["dtheta_resultant_per_unit"].mean()),
            },
        },
        "coupling_spectrum": {
            "fiedler_lambda": float(spec["fiedler_val"]),
            "top_k_eigvals": [float(v) for v in spec["top_k_eigvals"].tolist()],
            "degree_mean": float(spec["degree"].mean()),
            "degree_std": float(spec["degree"].std()),
            "degree_min": float(spec["degree"].min()),
            "degree_max": float(spec["degree"].max()),
        },
        "clusters": {
            "n_clusters": int(args.n_clusters),
            "sizes_by_label": cluster_sizes,
        },
        "hub_centrality": {
            "top_20_unit_indices": [int(i) for i in top_hubs],
            "top_20_centrality": [float(hub_central[i]) for i in top_hubs],
            "centrality_min": float(hub_central.min()),
            "centrality_max": float(hub_central.max()),
        },
        "tier2_summary": {
            "amp_var_mean":         float(sub["amp_var"].mean()),
            "phase_circ_var_mean":  float(sub["phase_circ_var"].mean()),
            "freq_var_mean":        float(sub["freq_var"].mean()),
            "harm2_amp_var_mean":   float(sub["harm2_amp_var"].mean()),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def slugify(name):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # Model
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"HuggingFace model id (default: {DEFAULT_MODEL}).")
    parser.add_argument("--hf-home", default=None,
                        help="Override HF cache directory. If not set, uses "
                             "HF_HOME env var or HuggingFace default.")
    parser.add_argument("--device", default=None,
                        help="Override device (cuda / mps / cpu). "
                             "Default: auto-detect.")

    # Inputs
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--fasta", default=None,
                     help="Path to FASTA file with proteins.")
    grp.add_argument("--single-protein", action="store_true",
                     help="Use the bundled ubiquitin sequence as a single-protein "
                          "input distribution. Sanity-check mode only.")
    grp.add_argument("--protein-sequence", default=None,
                     help="Provide a single sequence directly on the command line.")
    parser.add_argument("--n-samples", type=int, default=256,
                        help="Number of (protein, position) inputs to sample. "
                             "Default: 256.")
    parser.add_argument("--min-len", type=int, default=20)
    parser.add_argument("--max-len", type=int, default=512)

    # Phase-space analysis
    parser.add_argument("--trim-sublayers", type=int, default=2,
                        help="Drop K sublayers from each end of the captured "
                             "stream. Default: 2.")
    parser.add_argument("--r-threshold", type=float, default=0.30)
    parser.add_argument("--pca-threshold", type=float, default=0.40)
    parser.add_argument("--dtheta-threshold", type=float,
                        default=float(np.pi / 4))

    # Coupling and clustering
    parser.add_argument("--coupling-chunk-size", type=int, default=64,
                        help="Inputs per chunk in coupling-matrix accumulation. "
                             "Smaller = lower peak RAM. Default: 64.")
    parser.add_argument("--n-clusters", type=int, default=4,
                        help="Number of clusters in spectral cluster step. "
                             "Default: 4.")
    parser.add_argument("--top-k-eigvecs", type=int, default=10,
                        help="Number of low Laplacian eigenvectors to retain. "
                             "Default: 10.")
    parser.add_argument("--top-units-in-plots", type=int, default=200,
                        help="Limit unit-axis of heatmaps to top-N by hub "
                             "centrality. Default: 200.")

    # Output
    parser.add_argument("--output-dir", default=None,
                        help="Output directory. Default: "
                             "esm2_substrate_<model_slug>/")

    # Misc
    parser.add_argument("--seed", type=int, default=SEED)

    args = parser.parse_args()

    # Resolve HF cache
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
    elif "HF_HOME" not in os.environ:
        # Fallback: try the user's SSD path that other scripts in this project
        # use, but do not require it.
        ssd_default = "/Volumes/ORICO/huggingface_cache"
        if os.path.isdir(os.path.dirname(ssd_default)):
            os.environ["HF_HOME"] = ssd_default

    # Resolve device
    device = get_device(args.device)
    args.device = device

    # Output dir
    out_dir = (Path(args.output_dir) if args.output_dir
               else Path(f"esm2_substrate_{slugify(args.model)}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    print(f"=== ESM-2 Substrate Characterization ===")
    print(f"  model:           {args.model}")
    print(f"  device:          {device}")
    print(f"  output:          {out_dir.resolve()}")
    print(f"  n_samples:       {args.n_samples}")
    print(f"  trim_sublayers:  {args.trim_sublayers}")
    print(f"  n_clusters:      {args.n_clusters}")
    print(f"  top_k_eigvecs:   {args.top_k_eigvecs}")

    # Load sequences
    print("\n[1/6] Loading sequences ...")
    sequences = load_sequences(args)
    samples = sample_inputs(sequences, args.n_samples, rng)
    print(f"  Sampled {len(samples)} (protein, position) inputs.")

    # Load model
    print(f"\n[2/6] Loading {args.model} ...")
    from transformers import EsmTokenizer, EsmForMaskedLM
    tokenizer = EsmTokenizer.from_pretrained(args.model)
    model = EsmForMaskedLM.from_pretrained(
        args.model, attn_implementation="eager",
    ).to(device).to(torch.float32)
    model.eval()
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    print(f"  n_layers={n_layers}, d_model={d_model}")
    print(f"  expected n_sub captured = {2 * n_layers} "
          f"(after trim={args.trim_sublayers}: {2 * n_layers - 2 * args.trim_sublayers})")

    # Capture streams
    print(f"\n[3/6] Capturing residual streams at masked positions ...")
    streams = collect_streams(model, tokenizer, sequences, samples, device)
    print(f"  raw streams shape (N, n_sub, d_model) = {streams.shape}")

    # Free GPU memory before heavy numpy work
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # Center along sublayer axis, then trim
    streams = streams - streams.mean(axis=1, keepdims=True)
    if args.trim_sublayers > 0:
        if streams.shape[1] <= 2 * args.trim_sublayers + 2:
            raise ValueError(
                f"Cannot trim {args.trim_sublayers} from each end of "
                f"{streams.shape[1]} sublayers."
            )
        streams = streams[:, args.trim_sublayers:streams.shape[1] - args.trim_sublayers, :]
    N, L_used, D = streams.shape
    print(f"  post-trim, post-centering: (N, L_used, D) = {streams.shape}")

    # Phase-space diagnostics
    print(f"\n[4/6] Per-unit phase-space diagnostics ...")
    per_unit = collect_per_unit_diagnostics(streams)

    # Substrate observables and Tier-2 channel variance
    print(f"\n[5/6] Substrate observables and channel-variance maps ...")
    sub = compute_substrate_and_variance(streams)

    # Coupling matrix (uses sub['phase'] which has shape (N, L_used, D))
    print(f"  Computing coupling matrix C (chunk_size={args.coupling_chunk_size}) ...")
    coupling = compute_coupling_matrix(
        sub["phase"], chunk_size=args.coupling_chunk_size,
    )
    coupling_abs = np.abs(coupling).astype(np.float32)
    coupling_phase = np.angle(coupling).astype(np.float32)
    np.fill_diagonal(coupling_abs, 0.0)

    # Spectral analysis
    print(f"  Laplacian spectral analysis ...")
    spec = laplacian_spectral(coupling_abs, k=args.top_k_eigvecs)
    hub_central = hub_centrality(coupling_abs)
    cluster_labels, _ = cluster_units_spectral(
        spec["top_k_eigvecs"], n_clusters=args.n_clusters,
        rng_seed=args.seed,
    )
    sub["_hub_central"] = hub_central  # used by plot helpers

    # Plots
    print(f"\n[6/6] Generating plots ...")
    plot_phase_portraits(per_unit, per_unit["x_pair"], per_unit["dx"],
                         out_dir / "phase_portraits.png")
    print("  phase_portraits.png")
    plot_diagnostics(per_unit, out_dir / "diagnostics.png")
    print("  diagnostics.png")
    plot_substrate_amplitude_profile(
        sub["amp_mean"], hub_central,
        out_dir / "substrate_amplitude_profile.png",
        top_units=args.top_units_in_plots,
    )
    print("  substrate_amplitude_profile.png")
    plot_channel_variance_maps(
        sub, out_dir / "channel_variance_maps.png",
        top_units=args.top_units_in_plots,
    )
    print("  channel_variance_maps.png")
    plot_coupling_matrix(
        coupling_abs, spec["fiedler_vec"], cluster_labels,
        out_dir / "coupling_matrix.png",
    )
    print("  coupling_matrix.png")
    plot_laplacian_spectrum(
        spec["eigvals"], out_dir / "laplacian_spectrum.png",
    )
    print("  laplacian_spectrum.png")
    plot_top_eigenvectors(
        spec["top_k_eigvecs"], spec["fiedler_vec"], cluster_labels,
        out_dir / "fiedler_and_top_eigenvectors.png",
    )
    print("  fiedler_and_top_eigenvectors.png")

    # Save everything to NPZ
    print(f"\nSaving substrate.npz ...")
    np.savez(
        out_dir / "substrate.npz",
        # Metadata
        n_inputs=np.int32(N),
        n_sublayers_used=np.int32(L_used),
        d_model=np.int32(D),
        trim_sublayers=np.int32(args.trim_sublayers),
        # Per-unit diagnostics
        r_per_unit=per_unit["r_per_unit"],
        A_norm_per_unit=per_unit["A_norm_per_unit"],
        R_per_unit=per_unit["R_per_unit"],
        dtheta_per_unit=per_unit["dtheta_per_unit"],
        dtheta_resultant_per_unit=per_unit["dtheta_resultant_per_unit"],
        # Per-(unit, sublayer) substrate observables
        amp_mean=sub["amp_mean"],
        phase_resultant=sub["phase_resultant"],
        # Tier-2 channel variance maps
        amp_var=sub["amp_var"],
        phase_circ_var=sub["phase_circ_var"],
        freq_mean=sub["freq_mean"],
        freq_var=sub["freq_var"],
        harm2_amp_mean=sub["harm2_amp_mean"],
        harm2_amp_var=sub["harm2_amp_var"],
        # Coupling matrix
        coupling_abs=coupling_abs,
        coupling_phase=coupling_phase,
        # Spectral analysis
        laplacian_eigvals=spec["eigvals"],
        laplacian_top_k_eigvecs=spec["top_k_eigvecs"],
        fiedler_vec=spec["fiedler_vec"],
        coupling_degree=spec["degree"],
        hub_centrality=hub_central,
        cluster_labels=cluster_labels,
    )

    # Save summary JSON
    summary = build_summary(
        args, sequences, samples, L_used, D,
        per_unit, sub, spec, hub_central, cluster_labels,
    )
    with open(out_dir / "substrate_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("  substrate_summary.json")

    # Console table
    print("\n--- Phase-space classification ---")
    is_osc, is_deg, is_other = classify_units(
        per_unit,
        r_thresh=args.r_threshold,
        pca_thresh=args.pca_threshold,
        dtheta_thresh=args.dtheta_threshold,
    )
    print(f"  D = {D}")
    print(f"  oscillator units: {int(is_osc.sum())} ({100 * is_osc.sum() / D:.1f}%)")
    print(f"  degenerate units: {int(is_deg.sum())} ({100 * is_deg.sum() / D:.1f}%)")
    print(f"  other units:      {int(is_other.sum())}")
    print(f"  <|r|>             = {np.abs(per_unit['r_per_unit']).mean():.3f}")
    print(f"  <R_PCA>           = {per_unit['R_per_unit'].mean():.3f}")
    print(f"  <|dtheta|-pi/2|>  = "
          f"{np.abs(np.abs(per_unit['dtheta_per_unit']) - np.pi / 2).mean():.3f}")
    print(f"\n--- Coupling structure ---")
    print(f"  fiedler lambda_2     = {spec['fiedler_val']:.4f}")
    print(f"  mean degree (sum|C|) = {spec['degree'].mean():.2f}")
    print(f"  cluster sizes        = {summary['clusters']['sizes_by_label']}")
    print(f"  top 5 hub units      = {summary['hub_centrality']['top_20_unit_indices'][:5]}")

    print(f"\nDone. Outputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
