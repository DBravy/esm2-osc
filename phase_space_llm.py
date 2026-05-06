"""
Phase-space oscillator test on a trained language model.

Adapted from phase_space_test.py (toy transformer) to work with a
HuggingFace causal LM, using the same data-collection pattern as the
reference random_v2.py: hooks on the input-LN and post-attention-LN of
each transformer block, last-token snapshots, wikitext-2 validation as
the input distribution.

Streams are captured as (n_samples, n_sub, d_model) where n_sub equals
2 * n_layers (one snapshot before attention, one before MLP, for each
block). Delta x between consecutive snapshots is exactly one sublayer
output, matching the toy convention.

The four diagnostics, computed per (input, unit) along the depth axis l:

  r       Pearson(x_l, Delta x_l). Near 0 = oscillator. +/-1 = degenerate.
  A_norm  Signed shoelace area of (x, Delta x), normalized by
          std(x) * std(Delta x) * L. Sign = direction (negative is
          forward-time clockwise). Magnitude = rotational strength.
  R_PCA   Smaller / larger eigenvalue of the 2x2 covariance of
          (x, Delta x), AFTER rescaling Delta x so var(rescaled Delta x)
          = var(x). Frequency-invariant: a clean oscillator at any omega
          gives 1; a line gives 0. Equals (1 - |r|) / (1 + |r|) by
          construction; reported because the [0, 1] scale is more
          interpretable than the [-1, 1] correlation.
  dtheta  Mean phase offset arg(Hilbert(Delta x)) - arg(Hilbert(x))
          per (input, unit), aggregated by circular mean across inputs.
          +pi/2 = forward-rotating cosine oscillator.

Aggregation: each diagnostic is computed per (input, unit), then taken
median (or circular mean for dtheta) across inputs to give one value
per unit. d_model can be large (e.g. 960 for SmolLM2-360M) so the
plots use small markers and low alpha.

A unit is classified "oscillator" if all of:
  |r|     < r_threshold              (default 0.30)
  R_PCA   > pca_threshold             (default 0.40, consistent under
                                       the rescaling identity)
  ||dtheta| - pi/2| < dtheta_threshold (default pi/4)

Inputs:
  --model: HF model id, default HuggingFaceTB/SmolLM2-360M.
  --trim-sublayers: drop this many sublayers from each end of the
                    captured stream (default 2).

Outputs in phase_space_llm_<model>/:
  phase_portraits_trained.png         9-panel grid of trained units
  phase_portraits_random_init.png     same for random init
  diagnostics_trained.png             histograms of r, A, R_PCA, dtheta
  diagnostics_random_init.png         same for random init
  scatter_trained.png                 |r| vs R_PCA, with reference curve
  scatter_random_init.png             same for random init
  phase_space_comparison.png          stacked-bar oscillator/degenerate/other
  R_PCA_overlay.png                   R_PCA distributions, both checkpoints
  dtheta_overlay.png                  dtheta distributions, both checkpoints
  phase_space_summary.json            full numerical summary

Usage:
  python phase_space_llm.py --model HuggingFaceTB/SmolLM2-360M
  python phase_space_llm.py --model gpt2 --n-samples 256 --seq-len 128
  python phase_space_llm.py --model HuggingFaceTB/SmolLM2-360M \\
      --trim-sublayers 4 --r-threshold 0.25
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.signal import hilbert as scipy_hilbert
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset


DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0


# ---------------------------------------------------------------------------
# Stream collection (mirroring the reference script)
# ---------------------------------------------------------------------------

def get_hook_targets(model):
    """Locate the LayerNorm modules in each transformer block."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        blocks = model.transformer.h
        attn_ln, mlp_ln = "ln_1", "ln_2"
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        blocks = model.model.layers
        attn_ln, mlp_ln = "input_layernorm", "post_attention_layernorm"
    else:
        raise RuntimeError("Could not locate transformer blocks.")
    targets = []
    for block in blocks:
        targets.append(getattr(block, attn_ln))
        targets.append(getattr(block, mlp_ln))
    return targets


def collect_streams(model, tokenizer, texts, seq_len):
    """Capture residual stream snapshots at the last token across all
    sublayers (input-LN and post-attention-LN inputs).

    Returns array of shape (n_samples, n_sub, d_model)."""
    model.eval()
    targets = get_hook_targets(model)
    out = []
    for text in texts:
        enc = tokenizer(text, return_tensors="pt",
                        max_length=seq_len, truncation=True)
        ids = enc["input_ids"].to(DEVICE)
        if ids.shape[1] < 8:
            continue
        captures = []
        hooks = [
            t.register_forward_hook(
                lambda m, i, o, c=captures: c.append(i[0].detach())
            )
            for t in targets
        ]
        try:
            with torch.no_grad():
                model(ids)
        finally:
            for h in hooks:
                h.remove()
        last = torch.stack([c[0, -1, :].float() for c in captures], dim=0)
        out.append(last.cpu().numpy())
    return np.stack(out, axis=0)


# ---------------------------------------------------------------------------
# Phase-space diagnostics (from phase_space_test.py)
# ---------------------------------------------------------------------------

def pair_x_dx(x):
    """Return (x_pair, dx) shape (N, L-1, D) each. dx[:, l, :] = x[:, l+1, :]
    - x[:, l, :]."""
    dx = np.diff(x, axis=1)
    x_pair = x[:, :-1, :]
    return x_pair, dx


def per_traj_pearson(x, dx):
    """Pearson(x, dx) over l, per (input, unit). Shape (N, L-1, D) -> (N, D)."""
    x_c = x - x.mean(axis=1, keepdims=True)
    dx_c = dx - dx.mean(axis=1, keepdims=True)
    num = (x_c * dx_c).sum(axis=1)
    den = np.sqrt((x_c ** 2).sum(axis=1) * (dx_c ** 2).sum(axis=1)) + 1e-30
    return num / den


def per_traj_signed_area(x, dx):
    """Shoelace signed area of (x_l, dx_l) trajectory, per (input, unit).
    Forward oscillation gives clockwise motion -> negative signed area."""
    if x.shape[1] < 2:
        return np.zeros((x.shape[0], x.shape[2]))
    a = x[:, :-1, :] * dx[:, 1:, :]
    b = x[:, 1:, :] * dx[:, :-1, :]
    return 0.5 * (a - b).sum(axis=1)


def per_traj_pca_aspect(x, dx):
    """PCA aspect ratio AFTER rescaling dx so var(rescaled dx) = var(x).
    Frequency-invariant. Equals (1 - |r|) / (1 + |r|)."""
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
    """Mean phase offset arg(Hilbert(dx)) - arg(Hilbert(x)) over l, per
    (input, unit). For a clean cosine oscillator this is +pi/2."""
    z_x = scipy_hilbert(x, axis=1)
    z_dx = scipy_hilbert(dx, axis=1)
    ratio = z_dx / (z_x + 1e-30)
    return np.angle(ratio.mean(axis=1))


def collect_metrics(streams, trim=0):
    """Center the streams along the sublayer axis, optionally trim
    sublayers from each end, then compute all diagnostics."""
    if trim > 0:
        if streams.shape[1] <= 2 * trim + 2:
            raise ValueError(
                f"Cannot trim {trim} from each end of {streams.shape[1]} sublayers."
            )
        streams = streams[:, trim:streams.shape[1] - trim, :]
    x = streams - streams.mean(axis=1, keepdims=True)

    x_pair, dx = pair_x_dx(x)

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
        "shape": x.shape,
        "x_pair": x_pair,
        "dx": dx,
        "r": r,
        "A": A,
        "A_norm": A_norm,
        "R": R,
        "dtheta": dtheta,
        "r_per_unit": r_per_unit,
        "A_norm_per_unit": A_norm_per_unit,
        "R_per_unit": R_per_unit,
        "dtheta_per_unit": dtheta_per_unit,
        "dtheta_resultant_per_unit": dtheta_resultant,
    }


def classify_units(m, r_thresh=0.3, pca_thresh=0.4, dtheta_thresh=np.pi / 4):
    r_abs = np.abs(m["r_per_unit"])
    R = m["R_per_unit"]
    dt_dev = np.abs(np.abs(m["dtheta_per_unit"]) - np.pi / 2)
    is_osc = (r_abs < r_thresh) & (R > pca_thresh) & (dt_dev < dtheta_thresh)
    is_degenerate = R < pca_thresh / 2
    is_other = ~is_osc & ~is_degenerate
    return is_osc, is_degenerate, is_other


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _draw_cov_ellipse(ax, x_pts, y_pts, n_sigma=2.0,
                      color="red", alpha=0.7, lw=1.5):
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


def plot_phase_portraits(m, label, out_path):
    """9-panel grid: 3 highest, 3 median, 3 lowest |A_norm|."""
    A = np.abs(m["A_norm_per_unit"])
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
            ["high |A|"] * 3
            + ["median |A|"] * 3
            + ["low |A|"] * 3
        )

    x_pair = m["x_pair"]
    dx = m["dx"]
    N, Lm1, _ = x_pair.shape
    layer_norm = np.arange(Lm1) / max(Lm1 - 1, 1)

    fig, axes = plt.subplots(3, 3, figsize=(15, 14))
    for ax, u, lbl in zip(axes.flatten(), chosen, chosen_labels):
        xs = x_pair[:, :, u].flatten()
        dys = dx[:, :, u].flatten()
        cs = np.tile(layer_norm, N)
        ax.scatter(xs, dys, c=cs, cmap="viridis",
                   s=3, alpha=0.12, edgecolors="none")
        ax.plot(x_pair[0, :, u], dx[0, :, u],
                "-", color="black", linewidth=0.8, alpha=0.8)
        ax.scatter(x_pair[0, 0, u], dx[0, 0, u],
                   marker="o", s=40, facecolor="white",
                   edgecolor="black", linewidth=1.0, zorder=5)
        _draw_cov_ellipse(ax, xs, dys, n_sigma=2.0)

        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
        ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)
        ax.set_xlabel(r"$x_\ell$")
        ax.set_ylabel(r"$\Delta x_\ell$")
        ax.set_title(
            f"unit {u} ({lbl})\n"
            f"r={m['r_per_unit'][u]:+.2f}, "
            f"$A_{{\\mathrm{{norm}}}}$={m['A_norm_per_unit'][u]:+.3f}, "
            f"$R$={m['R_per_unit'][u]:.2f}, "
            f"$\\Delta\\theta$={m['dtheta_per_unit'][u]:+.2f}",
            fontsize=9,
        )

    fig.suptitle(
        f"Phase portrait $(x_\\ell, \\Delta x_\\ell)$: {label}\n"
        "color = sublayer fraction (dark = early, bright = late); "
        "white circle = trajectory start; red = 2$\\sigma$ cov ellipse",
        fontsize=12, y=0.998,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_diagnostics(m, label, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.hist(m["r_per_unit"], bins=40, range=(-1, 1), color="C0", alpha=0.85)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0,
               label="oscillator ideal (r=0)")
    ax.set_xlabel(r"Pearson correlation $r$ between $x$ and $\Delta x$")
    ax.set_ylabel("number of units")
    ax.set_title("(a) correlation: 0 = oscillator, $\\pm 1$ = degenerate")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    A = m["A_norm_per_unit"]
    bound = max(np.abs(A).max(), 1e-6)
    ax.hist(A, bins=40, range=(-bound, bound), color="C2", alpha=0.85)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"normalized signed area $A_{\mathrm{norm}}$")
    ax.set_ylabel("number of units")
    ax.set_title("(b) signed area: $<0$ = forward osc")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.hist(m["R_per_unit"], bins=40, range=(0, 1), color="C3", alpha=0.85)
    ax.axvline(0.4, color="black", linestyle=":", linewidth=1.0,
               label="default threshold 0.4")
    ax.set_xlabel(r"$R_{\mathrm{PCA}}$ (rescaled)")
    ax.set_ylabel("number of units")
    ax.set_title("(c) circularity: 1 = circular, 0 = line")
    ax.set_xlim(0, 1)
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    dt = m["dtheta_per_unit"]
    ax.hist(dt, bins=48, range=(-np.pi, np.pi), color="C4", alpha=0.85)
    ax.axvline(np.pi / 2, color="black", linestyle="--", linewidth=1.0,
               label=r"$+\pi/2$ (forward osc)")
    ax.axvline(-np.pi / 2, color="black", linestyle=":", linewidth=1.0,
               label=r"$-\pi/2$ (reverse osc)")
    ax.axvline(0, color="red", linestyle=":", linewidth=0.8)
    ax.set_xlabel(r"$\Delta\theta$")
    ax.set_ylabel("number of units")
    ax.set_title(r"(d) quadrature phase: $\pm\pi/2$ = oscillator")
    ax.set_xlim(-np.pi, np.pi)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(f"Phase-space diagnostics: {label}", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(m, label, out_path,
                 r_thresh=0.3, pca_thresh=0.4, dtheta_thresh=np.pi / 4):
    is_osc, is_deg, is_other = classify_units(
        m, r_thresh=r_thresh, pca_thresh=pca_thresh, dtheta_thresh=dtheta_thresh,
    )
    r_abs = np.abs(m["r_per_unit"])
    R = m["R_per_unit"]

    fig, ax = plt.subplots(figsize=(9, 8))

    rr = np.linspace(0, 1, 200)
    ax.plot(rr, (1 - rr) / (1 + rr), color="black",
            linestyle="-", alpha=0.4, linewidth=1.2,
            label=r"$(1-|r|)/(1+|r|)$ (rescaled identity)")

    # Small markers: D may be ~960
    if is_other.any():
        ax.scatter(r_abs[is_other], R[is_other], s=15, alpha=0.5,
                   color="gray", edgecolors="none",
                   label=f"other (n={is_other.sum()})")
    if is_deg.any():
        ax.scatter(r_abs[is_deg], R[is_deg], s=15, alpha=0.6,
                   color="C3", edgecolors="none",
                   label=f"degenerate (n={is_deg.sum()})")
    if is_osc.any():
        ax.scatter(r_abs[is_osc], R[is_osc], s=20, alpha=0.7,
                   color="C2", edgecolors="none",
                   label=f"oscillator (n={is_osc.sum()})")

    ax.axhline(pca_thresh, color="C2", linestyle=":", alpha=0.6)
    ax.axvline(r_thresh, color="C2", linestyle=":", alpha=0.6)
    ax.set_xlabel(r"$|r|$")
    ax.set_ylabel(r"$R_{\mathrm{PCA}}$ (rescaled)")
    ax.set_title(f"Phase-space classification: {label}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(metrics_trained, metrics_init, out_path,
                    r_thresh=0.3, pca_thresh=0.4, dtheta_thresh=np.pi / 4):
    fig, ax = plt.subplots(figsize=(9, 6))
    labels = ["trained", "random_init"]
    osc_frac, deg_frac, oth_frac = [], [], []
    for m in [metrics_trained, metrics_init]:
        o, d, ot = classify_units(
            m, r_thresh=r_thresh, pca_thresh=pca_thresh,
            dtheta_thresh=dtheta_thresh,
        )
        D = len(o)
        osc_frac.append(o.sum() / D)
        deg_frac.append(d.sum() / D)
        oth_frac.append(ot.sum() / D)

    x = np.arange(2)
    width = 0.6
    osc_arr = np.array(osc_frac)
    deg_arr = np.array(deg_frac)
    oth_arr = np.array(oth_frac)
    ax.bar(x, osc_arr, width, label="oscillator", color="C2")
    ax.bar(x, deg_arr, width, bottom=osc_arr, label="degenerate", color="C3")
    ax.bar(x, oth_arr, width, bottom=osc_arr + deg_arr,
           label="other", color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("fraction of units")
    ax.set_title("Phase-space classification")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_overlay_R(metrics_trained, metrics_init, out_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for m, label, style in [(metrics_trained, "trained", "-"),
                              (metrics_init, "random_init", "--")]:
        ax.hist(m["R_per_unit"], bins=60, range=(0, 1), histtype="step",
                density=True, linewidth=1.5, label=label, linestyle=style)
    ax.axvline(0.4, color="black", linestyle=":", alpha=0.5,
               label="threshold 0.4")
    ax.set_xlabel(r"$R_{\mathrm{PCA}}$ (rescaled)")
    ax.set_ylabel("density")
    ax.set_title(r"$R_{\mathrm{PCA}}$ distribution: trained vs random init")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_overlay_dtheta(metrics_trained, metrics_init, out_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for m, label, style in [(metrics_trained, "trained", "-"),
                              (metrics_init, "random_init", "--")]:
        ax.hist(m["dtheta_per_unit"], bins=72, range=(-np.pi, np.pi),
                histtype="step", density=True, linewidth=1.5,
                label=label, linestyle=style)
    ax.axvline(np.pi / 2, color="black", linestyle="--", alpha=0.5,
               label=r"$+\pi/2$")
    ax.axvline(-np.pi / 2, color="gray", linestyle="--", alpha=0.5,
               label=r"$-\pi/2$")
    ax.set_xlabel(r"$\Delta\theta$")
    ax.set_ylabel("density")
    ax.set_title(r"$\Delta\theta$ distribution: trained vs random init")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def build_summary(metrics_trained, metrics_init,
                  r_thresh, pca_thresh, dtheta_thresh,
                  model_name, n_sub_used):
    summary = {
        "model": model_name,
        "n_sublayers_used_after_trim": int(n_sub_used),
        "thresholds": {
            "r_threshold": r_thresh,
            "pca_threshold": pca_thresh,
            "dtheta_threshold_radians": dtheta_thresh,
        },
        "per_checkpoint": {},
    }
    for label, m in [("trained", metrics_trained),
                     ("random_init", metrics_init)]:
        is_osc, is_deg, is_other = classify_units(
            m, r_thresh=r_thresh, pca_thresh=pca_thresh,
            dtheta_thresh=dtheta_thresh,
        )
        D = m["shape"][2]
        summary["per_checkpoint"][label] = {
            "shape": {"N": int(m["shape"][0]),
                      "L": int(m["shape"][1]),
                      "D": int(D)},
            "n_oscillator": int(is_osc.sum()),
            "n_degenerate": int(is_deg.sum()),
            "n_other": int(is_other.sum()),
            "fraction_oscillator": float(is_osc.sum() / D),
            "fraction_degenerate": float(is_deg.sum() / D),
            "r_per_unit": {
                "mean": float(m["r_per_unit"].mean()),
                "median": float(np.median(m["r_per_unit"])),
                "abs_mean": float(np.abs(m["r_per_unit"]).mean()),
                "abs_median": float(np.median(np.abs(m["r_per_unit"]))),
            },
            "A_norm_per_unit": {
                "mean": float(m["A_norm_per_unit"].mean()),
                "abs_mean": float(np.abs(m["A_norm_per_unit"]).mean()),
                "median": float(np.median(m["A_norm_per_unit"])),
            },
            "R_PCA_per_unit": {
                "mean": float(m["R_per_unit"].mean()),
                "median": float(np.median(m["R_per_unit"])),
            },
            "dtheta_per_unit": {
                "circular_mean": float(m["dtheta_per_unit"].mean()),
                "abs_mean": float(np.abs(m["dtheta_per_unit"]).mean()),
                "deviation_from_pi_over_2_radians": {
                    "mean": float(np.abs(np.abs(m["dtheta_per_unit"]) - np.pi / 2).mean()),
                    "median": float(np.median(np.abs(np.abs(m["dtheta_per_unit"]) - np.pi / 2))),
                },
                "resultant_length": {
                    "mean": float(m["dtheta_resultant_per_unit"].mean()),
                    "median": float(np.median(m["dtheta_resultant_per_unit"])),
                },
            },
        }
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def slugify(name):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="HuggingFace model id.")
    parser.add_argument("--n-samples", type=int, default=256,
                        help="Number of texts from wikitext-2 validation.")
    parser.add_argument("--seq-len", type=int, default=128,
                        help="Max tokens per text.")
    parser.add_argument("--trim-sublayers", type=int, default=2,
                        help="Drop this many sublayers from each end.")
    parser.add_argument("--r-threshold", type=float, default=0.30)
    parser.add_argument("--pca-threshold", type=float, default=0.40)
    parser.add_argument("--dtheta-threshold", type=float,
                        default=float(np.pi / 4))
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = (Path(args.out_dir) if args.out_dir
               else Path(f"phase_space_llm_{slugify(args.model)}"))
    out_dir.mkdir(exist_ok=True, parents=True)

    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    print("Loading wikitext-2 ...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    cands = [x["text"] for x in ds if 200 < len(x["text"]) < 1500]
    rng.shuffle(cands)
    texts = cands[:args.n_samples]

    print(f"Loading tokenizer for {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Trained ----
    print(f"\n=== TRAINED ({args.model}) ===")
    trained_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32,
    ).to(DEVICE)
    print("  collecting streams ...")
    streams_trained = collect_streams(
        trained_model, tokenizer, texts, args.seq_len,
    )
    print(f"  shape (N, n_sub, d_model) = {streams_trained.shape}")
    del trained_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Random init ----
    print(f"\n=== RANDOM INIT ===")
    config = AutoConfig.from_pretrained(args.model)
    init_model = AutoModelForCausalLM.from_config(config).to(DEVICE).float()
    print("  collecting streams ...")
    streams_init = collect_streams(
        init_model, tokenizer, texts, args.seq_len,
    )
    print(f"  shape (N, n_sub, d_model) = {streams_init.shape}")
    del init_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Diagnostics ----
    print(f"\nComputing phase-space diagnostics (trim={args.trim_sublayers}) ...")
    metrics_trained = collect_metrics(
        streams_trained, trim=args.trim_sublayers,
    )
    metrics_init = collect_metrics(
        streams_init, trim=args.trim_sublayers,
    )
    L_used = metrics_trained["shape"][1]
    print(f"  effective L (after trim): {L_used}")

    # ---- Plots ----
    print("\nMaking plots ...")
    plot_phase_portraits(metrics_trained, "trained",
                         out_dir / "phase_portraits_trained.png")
    plot_phase_portraits(metrics_init, "random_init",
                         out_dir / "phase_portraits_random_init.png")
    plot_diagnostics(metrics_trained, "trained",
                     out_dir / "diagnostics_trained.png")
    plot_diagnostics(metrics_init, "random_init",
                     out_dir / "diagnostics_random_init.png")
    plot_scatter(metrics_trained, "trained",
                 out_dir / "scatter_trained.png",
                 args.r_threshold, args.pca_threshold, args.dtheta_threshold)
    plot_scatter(metrics_init, "random_init",
                 out_dir / "scatter_random_init.png",
                 args.r_threshold, args.pca_threshold, args.dtheta_threshold)
    plot_comparison(metrics_trained, metrics_init,
                    out_dir / "phase_space_comparison.png",
                    args.r_threshold, args.pca_threshold, args.dtheta_threshold)
    plot_overlay_R(metrics_trained, metrics_init,
                   out_dir / "R_PCA_overlay.png")
    plot_overlay_dtheta(metrics_trained, metrics_init,
                        out_dir / "dtheta_overlay.png")

    # ---- Summary ----
    summary = build_summary(
        metrics_trained, metrics_init,
        args.r_threshold, args.pca_threshold, args.dtheta_threshold,
        args.model, L_used,
    )
    with open(out_dir / "phase_space_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Console table ----
    print("\n--- Phase-space classification ---")
    print(f"{'checkpoint':<14} {'D':>5} {'osc':>5} {'deg':>5} {'other':>6} "
          f"{'osc%':>7} {'<|r|>':>8} {'<R>':>8} {'<dtheta_dev>':>14}")
    for label in ("trained", "random_init"):
        s = summary["per_checkpoint"][label]
        print(
            f"{label:<14} {s['shape']['D']:>5} "
            f"{s['n_oscillator']:>5} {s['n_degenerate']:>5} "
            f"{s['n_other']:>6} "
            f"{100 * s['fraction_oscillator']:>6.1f}% "
            f"{s['r_per_unit']['abs_mean']:>8.3f} "
            f"{s['R_PCA_per_unit']['mean']:>8.3f} "
            f"{s['dtheta_per_unit']['deviation_from_pi_over_2_radians']['mean']:>14.3f}"
        )

    print(f"\nOutputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
