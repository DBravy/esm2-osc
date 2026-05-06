"""
ESM-2 Phase 1: per-position oscillatory analysis (patched)
============================================================

Patched relative to the first version:
  - Dropped Fiedler |cos| and top-k eigenspace overlap from the
    substrate-comparison block. They compared per-trajectory R-matrix
    eigenstructure to the substrate's phase-coupling C-matrix
    eigenstructure, which are essentially orthogonal quantities. They
    came out near zero at every position and carried no signal.
  - Kept the |R| vs |C| Frobenius cosine (the only well-defined
    substrate-comparison scalar).
  - Added a substrate-amplitude validity mask: cells with substrate
    amp_mean below a quantile floor are excluded from any ranking that
    uses amp_modulation. This kills the boundary contamination that
    caused top units to peak at sublayers 0, 1, 59, 60.
  - Added a hub signature per position: for the top-K substrate hubs
    (in centrality order), the position's max amp_mod and argmax
    sublayer. Saved as fixed-length vectors to the npz so cross-position
    comparisons (Phase 2) work in a uniform basis.
  - Overview plot panels updated: spectral entropy out (uninformative
    last run), Fiedler/eigenspace cos out (broken), in their place
    mean_hub_max_amp_mod, max_hub_max_amp_mod, and validity-mask coverage.
  - New plot: hub signature heatmap (n_hubs, n_positions).

Per-position outputs (in --output-dir):
  pos_{i}_oscillatory.json    Lightweight summary with depth profiles,
                              top-N units (filtered by validity mask),
                              hub signature, scalars.

Protein-level outputs:
  {protein}_oscillatory.npz                    Stacked channel arrays,
                                               hub signatures, validity
                                               mask, per-position scalars.
  {protein}_oscillatory_overview.png           Six-panel cross-position plot.
  {protein}_hub_signatures.png                 (n_hubs, n_positions) heatmap.

Usage:
  python esm2_phase1_oscillatory.py \\
      --substrate /path/to/substrate.npz --single-protein \\
      --output-dir esm2_phase1_ubiquitin

  # Tune the validity floor or hub signature length:
  python esm2_phase1_oscillatory.py --substrate substrate.npz \\
      --single-protein --output-dir out \\
      --substrate-amp-floor-quantile 0.15 --n-hubs-in-signature 30
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import hilbert as scipy_hilbert


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
# Substrate loading
# ---------------------------------------------------------------------------

REQUIRED_SUBSTRATE_KEYS = (
    "amp_mean",
    "freq_mean",
    "harm2_amp_mean",
    "coupling_abs",
    "fiedler_vec",
    "laplacian_top_k_eigvecs",
    "hub_centrality",
    "trim_sublayers",
    "n_sublayers_used",
    "d_model",
)


def load_substrate(path):
    sub = dict(np.load(path))
    missing = [k for k in REQUIRED_SUBSTRATE_KEYS if k not in sub]
    if missing:
        raise ValueError(
            f"Substrate file {path} is missing required keys: {missing}.\n"
            "Re-run esm2_substrate.py to regenerate."
        )
    return sub


# ---------------------------------------------------------------------------
# ESM-2 hooks
# ---------------------------------------------------------------------------

def get_esm2_hook_targets(model):
    if not hasattr(model, "esm") or not hasattr(model.esm, "encoder"):
        raise RuntimeError(
            f"Could not locate ESM encoder in {type(model).__name__}."
        )
    targets = []
    for block in model.esm.encoder.layer:
        if not hasattr(block, "attention") or not hasattr(block.attention, "LayerNorm"):
            raise RuntimeError("Block missing attention.LayerNorm.")
        if not hasattr(block, "LayerNorm"):
            raise RuntimeError("Block missing top-level LayerNorm.")
        targets.append(block.attention.LayerNorm)
        targets.append(block.LayerNorm)
    return targets


def collect_streams_at_position(model, tokenizer, sequence, mask_pos, device):
    model.eval()
    targets = get_esm2_hook_targets(model)
    enc = tokenizer(sequence, return_tensors="pt")
    ids = enc["input_ids"].to(device)
    attn_mask = enc.get("attention_mask")
    if attn_mask is not None:
        attn_mask = attn_mask.to(device)
    tidx = mask_pos + 1
    if tidx >= ids.shape[1] - 1:
        raise ValueError(f"mask_pos={mask_pos} out of range for seq of len "
                         f"{ids.shape[1] - 2}")
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
        raise RuntimeError("Hook firing inconsistent.")
    snap = torch.stack(
        [c[0, tidx, :].float() for c in captures], dim=0
    ).cpu().numpy()
    return snap


# ---------------------------------------------------------------------------
# Per-position analysis
# ---------------------------------------------------------------------------

def analyze_position(streams_one_pos, substrate, valid_mask, hub_indices,
                     k_eigvecs=10):
    """streams_one_pos: (L, D), centered and trimmed.
    valid_mask: (L, D) bool. True where substrate amp_mean is above the floor;
        rankings on amp_modulation use this mask to suppress boundary noise.
    hub_indices: (n_hubs,) int array, substrate hub units in centrality order.
    """
    L, D = streams_one_pos.shape

    # Hilbert analytic signal along depth.
    z = scipy_hilbert(streams_one_pos, axis=0)
    amp = np.abs(z).astype(np.float32)
    phase = np.angle(z).astype(np.float32)

    phase_unwr = np.unwrap(phase.astype(np.float64), axis=0)
    freq = np.diff(phase_unwr, axis=0).astype(np.float32)

    X = np.fft.rfft(streams_one_pos.astype(np.float32), axis=0)
    if X.shape[0] >= 3:
        harm2_amp = np.abs(X[2]).astype(np.float32)
    else:
        harm2_amp = np.zeros(D, dtype=np.float32)

    sub_amp_mean = substrate["amp_mean"].astype(np.float32)
    sub_freq_mean = substrate["freq_mean"].astype(np.float32)
    sub_harm2_amp_mean = substrate["harm2_amp_mean"].astype(np.float32)

    if sub_amp_mean.shape != amp.shape:
        raise ValueError(
            f"Shape mismatch: substrate amp_mean {sub_amp_mean.shape} "
            f"vs position amp {amp.shape}."
        )

    amp_modulation = (amp / np.maximum(sub_amp_mean, 1e-6)).astype(np.float32)
    freq_deviation = (freq - sub_freq_mean).astype(np.float32)
    harm2_deviation = (harm2_amp - sub_harm2_amp_mean).astype(np.float32)

    # Hub signature: max amp_mod over valid sublayers, per substrate hub.
    masked_amp_mod = np.where(valid_mask, amp_modulation, -np.inf)
    n_hubs = len(hub_indices)
    hub_max_amp_mod = np.full(n_hubs, np.nan, dtype=np.float32)
    hub_argmax_sublayer = np.full(n_hubs, -1, dtype=np.int32)
    for i, u in enumerate(hub_indices):
        col = masked_amp_mod[:, u]
        if np.isfinite(col).any():
            argmax = int(np.argmax(col))
            hub_max_amp_mod[i] = float(col[argmax])
            hub_argmax_sublayer[i] = argmax
    mean_hub_max_amp_mod = float(np.nanmean(hub_max_amp_mod))
    max_hub_max_amp_mod = float(np.nanmax(hub_max_amp_mod))

    # LM-style depth-axis cross-correlation.
    norms = np.linalg.norm(streams_one_pos, axis=0)
    norms = np.maximum(norms, 1e-12)
    R = (streams_one_pos.T @ streams_one_pos) / np.outer(norms, norms)
    R = R.astype(np.float64)
    affinity = np.abs(R)
    np.fill_diagonal(affinity, 0.0)

    deg = affinity.sum(axis=1)
    laplacian = np.diag(deg) - affinity
    eigvals, eigvecs = np.linalg.eigh(laplacian)

    fiedler_val = float(eigvals[1])
    fiedler_vec = eigvecs[:, 1].astype(np.float32)
    top_k_eigvals = eigvals[1:k_eigvecs + 1].astype(np.float32)
    top_k_eigvecs = eigvecs[:, 1:k_eigvecs + 1].astype(np.float32)

    nontriv = np.maximum(eigvals[1:], 0.0)
    s_nt = nontriv.sum()
    if s_nt > 0:
        ne = nontriv / s_nt
        spectral_entropy = float(-np.sum(ne * np.log(ne + 1e-30)))
    else:
        spectral_entropy = 0.0
    mean_off_diag_R = float(affinity.sum() / (D * (D - 1)))

    sv = np.linalg.svd(streams_one_pos, compute_uv=False)
    sv2 = sv ** 2
    s2 = sv2.sum()
    if s2 > 0:
        ne2 = sv2 / s2
        effective_rank = float(np.exp(-np.sum(ne2 * np.log(ne2 + 1e-30))))
    else:
        effective_rank = 0.0

    # Substrate comparison: keep ONLY the well-defined Frobenius cos.
    sub_C_abs = substrate["coupling_abs"].astype(np.float32)
    aff_flat = affinity.astype(np.float32).ravel()
    sub_flat = sub_C_abs.ravel()
    aff_norm = np.linalg.norm(aff_flat) + 1e-30
    sub_norm = np.linalg.norm(sub_flat) + 1e-30
    corr_frobenius_cos = float(np.dot(aff_flat, sub_flat)
                               / (aff_norm * sub_norm))

    return {
        "amp": amp,
        "phase": phase,
        "freq": freq,
        "harm2_amp": harm2_amp,
        "amp_modulation": amp_modulation,
        "freq_deviation": freq_deviation,
        "harm2_deviation": harm2_deviation,
        "fiedler_vec": fiedler_vec,
        "top_k_eigvecs": top_k_eigvecs,
        "top_k_eigvals": top_k_eigvals,
        "hub_max_amp_mod": hub_max_amp_mod,
        "hub_argmax_sublayer": hub_argmax_sublayer,
        "fiedler_val": fiedler_val,
        "spectral_entropy": spectral_entropy,
        "mean_off_diag_R": mean_off_diag_R,
        "effective_rank": effective_rank,
        "corr_frobenius_cos_substrate": corr_frobenius_cos,
        "mean_hub_max_amp_mod": mean_hub_max_amp_mod,
        "max_hub_max_amp_mod": max_hub_max_amp_mod,
    }


# ---------------------------------------------------------------------------
# Per-position summary builder (JSON)
# ---------------------------------------------------------------------------

def build_position_summary(pos, aa, result, substrate, substrate_path,
                           valid_mask, hub_indices, valid_floor,
                           top_n_units=20):
    amp_mod = result["amp_modulation"]
    phase = result["phase"]
    freq = result["freq"]
    freq_dev = result["freq_deviation"]
    harm2_dev = result["harm2_deviation"]

    L, D = amp_mod.shape

    valid_amp_mod = np.where(valid_mask, amp_mod, np.nan)
    depth_profiles = {
        "amp_modulation_mean": [
            float(amp_mod[ell].mean()) for ell in range(L)
        ],
        "amp_modulation_mean_valid_only": [
            float(np.nanmean(valid_amp_mod[ell]))
            if valid_mask[ell].any() else float("nan")
            for ell in range(L)
        ],
        "phase_consistency_across_units": [
            float(np.abs(np.exp(1j * phase[ell]).mean())) for ell in range(L)
        ],
        "freq_mean": [float(v) for v in freq.mean(axis=1)],
        "freq_deviation_abs_mean": [
            float(v) for v in np.abs(freq_dev).mean(axis=1)
        ],
        "valid_unit_count_per_sublayer": [
            int(valid_mask[ell].sum()) for ell in range(L)
        ],
    }

    # Top-N rankings on amp_modulation (with validity mask applied).
    masked_amp_mod = np.where(valid_mask, amp_mod, -np.inf)
    amp_mod_max = masked_amp_mod.max(axis=0)
    amp_mod_argmax = masked_amp_mod.argmax(axis=0)
    valid_per_unit = valid_mask.any(axis=0)
    rank_basis = np.where(valid_per_unit, amp_mod_max, -np.inf)
    top_amp_mod_idx = np.argsort(rank_basis)[::-1][:top_n_units]
    top_amp_mod_idx = [int(u) for u in top_amp_mod_idx
                       if rank_basis[u] > -np.inf]

    # Top-N by signed extreme deviation of amp_mod from 1.0 (valid cells).
    amp_mod_dev = amp_mod - 1.0
    masked_dev = np.where(valid_mask, amp_mod_dev, 0.0)
    abs_dev = np.abs(masked_dev)
    arg_dev = abs_dev.argmax(axis=0)
    signed_extreme = masked_dev[arg_dev, np.arange(D)]
    signed_extreme = np.where(valid_per_unit, signed_extreme, 0.0)
    top_dev_idx = np.argsort(np.abs(signed_extreme))[::-1][:top_n_units]
    top_dev_idx = [int(u) for u in top_dev_idx]

    # Top-N by harm2 deviation magnitude.
    top_harm2_idx = np.argsort(np.abs(harm2_dev))[::-1][:top_n_units]

    hub_max = result["hub_max_amp_mod"]
    hub_argmax = result["hub_argmax_sublayer"]
    sub_centrality = substrate["hub_centrality"]

    summary = {
        "position": int(pos),
        "sequence_aa": aa,
        "substrate_file": str(substrate_path),
        "n_sublayers": int(L),
        "n_units": int(D),
        "validity_floor": {
            "quantile": float(valid_floor["quantile"]),
            "absolute_value": float(valid_floor["absolute_value"]),
            "n_valid_cells": int(valid_mask.sum()),
            "n_total_cells": int(L * D),
            "fraction_valid": float(valid_mask.mean()),
        },

        "depth_profiles": depth_profiles,

        "top_units_by_amp_modulation_max": [
            {
                "unit": u,
                "max_amp_mod": float(amp_mod_max[u]),
                "argmax_sublayer": int(amp_mod_argmax[u]),
                "substrate_amp_at_argmax": float(
                    substrate["amp_mean"][amp_mod_argmax[u], u]
                ),
            }
            for u in top_amp_mod_idx
        ],

        "top_units_by_amp_modulation_signed_dev": [
            {
                "unit": u,
                "signed_extreme_dev": float(signed_extreme[u]),
                "argmax_sublayer": int(arg_dev[u]),
            }
            for u in top_dev_idx
        ],

        "top_units_by_harm2_deviation": [
            {
                "unit": int(u),
                "harm2_deviation": float(harm2_dev[u]),
                "harm2_amp_position": float(result["harm2_amp"][u]),
                "harm2_amp_substrate":
                    float(substrate["harm2_amp_mean"][u]),
            }
            for u in top_harm2_idx
        ],

        "hub_signature": {
            "hub_indices": [int(u) for u in hub_indices],
            "hub_substrate_centrality": [
                float(sub_centrality[u]) for u in hub_indices
            ],
            "max_amp_mod": [
                float(v) if np.isfinite(v) else None for v in hub_max
            ],
            "argmax_sublayer": [int(v) for v in hub_argmax],
            "mean_max_amp_mod": float(result["mean_hub_max_amp_mod"]),
            "max_max_amp_mod": float(result["max_hub_max_amp_mod"]),
        },

        "spectral": {
            "fiedler_val":      float(result["fiedler_val"]),
            "top_k_eigvals":    [float(v) for v in result["top_k_eigvals"]],
            "spectral_entropy": float(result["spectral_entropy"]),
            "effective_rank":   float(result["effective_rank"]),
            "mean_off_diag_R":  float(result["mean_off_diag_R"]),
        },

        "substrate_comparison": {
            "corr_frobenius_cos": float(result["corr_frobenius_cos_substrate"]),
            # Note: previous fiedler_cos / eigenspace_overlap with substrate
            # were dropped because they compared incompatible quantities
            # (per-trajectory amplitude correlation R vs across-input phase
            # consistency C). They were near zero at every position.
        },
    }
    return summary


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_overview(positions, sequence, summaries, out_path):
    pos_arr = np.array(positions)
    aa_arr = [sequence[p] for p in positions]

    eff_rank = np.array([s["spectral"]["effective_rank"] for s in summaries])
    fro_cos = np.array([s["substrate_comparison"]["corr_frobenius_cos"]
                        for s in summaries])
    mean_amp_mod = np.array([
        np.nanmean(s["depth_profiles"]["amp_modulation_mean_valid_only"])
        for s in summaries
    ])
    mean_hub = np.array([s["hub_signature"]["mean_max_amp_mod"]
                         for s in summaries])
    max_hub = np.array([s["hub_signature"]["max_max_amp_mod"]
                        for s in summaries])
    valid_frac = np.array([s["validity_floor"]["fraction_valid"]
                           for s in summaries])

    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)

    def _common(ax, title, ylab):
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylab, fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xticks(pos_arr)
        ax.set_xticklabels([f"{p}\n{a}" for p, a in zip(pos_arr, aa_arr)],
                           fontsize=6, rotation=0)

    ax = axes[0, 0]
    ax.plot(pos_arr, eff_rank, "o-", color="C0", markersize=3)
    _common(ax, "(a) effective rank of position trajectory",
            r"$\exp(-\sum p_i \log p_i)$")

    ax = axes[0, 1]
    ax.plot(pos_arr, fro_cos, "o-", color="C4", markersize=3)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    _common(ax, "(b) |R| vs |C| Frobenius cosine",
            r"$\langle |R|, |C| \rangle_F$")
    ax.set_ylim(0, 1.05)

    ax = axes[1, 0]
    ax.plot(pos_arr, mean_amp_mod, "o-", color="C5", markersize=3)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    _common(ax, "(c) <amp_modulation> across valid (sublayer, unit)",
            r"$\langle |z|/\langle|z|\rangle_{\mathrm{sub}}\rangle$")

    ax = axes[1, 1]
    ax.plot(pos_arr, mean_hub, "o-", color="C2", markersize=3)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    _common(ax, "(d) mean hub max amp_mod (per-position)",
            "mean of hub-signature")

    ax = axes[2, 0]
    ax.plot(pos_arr, max_hub, "o-", color="C3", markersize=3)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    _common(ax, "(e) max hub max amp_mod (per-position)",
            "max of hub-signature")
    ax.set_xlabel("position\nresidue", fontsize=9)

    ax = axes[2, 1]
    ax.plot(pos_arr, valid_frac, "o-", color="C6", markersize=3)
    _common(ax, "(f) substrate-validity-mask coverage",
            "fraction of (sublayer, unit) cells used")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("position\nresidue", fontsize=9)

    fig.suptitle(
        "Phase 1 oscillatory diagnostics across positions", fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_hub_signatures(positions, sequence, hub_signatures, hub_indices,
                        substrate_centrality, out_path):
    pos_arr = np.array(positions)
    aa_arr = [sequence[p] for p in positions]
    M = np.array(hub_signatures).T  # (n_hubs, n_positions)
    n_hubs, n_pos = M.shape

    width = max(12, n_pos * 0.16)
    height = max(6, n_hubs * 0.22)
    fig, ax = plt.subplots(figsize=(width, height))

    finite_M = M[np.isfinite(M)]
    if finite_M.size > 0:
        vmax = float(np.nanpercentile(finite_M, 99))
    else:
        vmax = 2.0
    im = ax.imshow(M, aspect="auto", cmap="magma", vmin=0, vmax=vmax,
                   interpolation="nearest")
    ax.set_xticks(range(n_pos))
    ax.set_xticklabels([f"{p}\n{a}" for p, a in zip(pos_arr, aa_arr)],
                       fontsize=6, rotation=0)
    ax.set_yticks(range(n_hubs))
    ax.set_yticklabels(
        [f"u{int(u)} (c={substrate_centrality[u]:.2e})"
         for u in hub_indices],
        fontsize=7,
    )
    ax.set_xlabel("position / residue")
    ax.set_ylabel("substrate hub (descending centrality)")
    ax.set_title(
        f"Hub signature: max amp_modulation per (hub, position). "
        f"top-{n_hubs} substrate hubs."
    )
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="max amp_mod")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sequence loading
# ---------------------------------------------------------------------------

def parse_fasta(path):
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


def resolve_sequence(args):
    if args.single_protein:
        return "ubiquitin", UBIQUITIN_SEQ
    if args.protein_sequence:
        seq = args.protein_sequence.upper()
        if any(c not in CANONICAL_AAS for c in seq):
            raise ValueError("--protein-sequence contains non-canonical AAs.")
        return args.protein_name or "user_seq", seq
    if args.fasta:
        if not args.protein_name:
            raise ValueError("--fasta requires --protein-name.")
        for name, seq in parse_fasta(args.fasta):
            if name == args.protein_name:
                if any(c not in CANONICAL_AAS for c in seq):
                    raise ValueError(f"Sequence '{name}' has non-canonical AAs.")
                return name, seq
        raise ValueError(f"--protein-name '{args.protein_name}' not found.")
    raise ValueError(
        "Must provide one of --single-protein, --protein-sequence, "
        "or --fasta + --protein-name."
    )


# ---------------------------------------------------------------------------
# Optional merge into existing pos_{i}.json
# ---------------------------------------------------------------------------

def merge_into_existing(pos_json_path, oscillatory_summary):
    if not pos_json_path.exists():
        return False
    with open(pos_json_path) as f:
        data = json.load(f)
    data["oscillatory"] = oscillatory_summary
    with open(pos_json_path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def slugify(name):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--substrate", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--hf-home", default=None)
    parser.add_argument("--device", default=None)

    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--single-protein", action="store_true")
    grp.add_argument("--protein-sequence", default=None)
    grp.add_argument("--fasta", default=None)
    parser.add_argument("--protein-name", default=None)

    parser.add_argument("--positions", nargs="+", type=int, default=None)

    parser.add_argument("--top-k-eigvecs", type=int, default=10)
    parser.add_argument("--top-n-units", type=int, default=20)

    # Patched args:
    parser.add_argument(
        "--substrate-amp-floor-quantile", type=float, default=0.10,
        help="Cells with substrate.amp_mean below this quantile across all "
             "(L, D) cells are excluded from amp_modulation rankings. "
             "Default: 0.10 (drop bottom 10%% of cells).",
    )
    parser.add_argument(
        "--n-hubs-in-signature", type=int, default=20,
        help="Number of substrate hubs (top-K by centrality) to use in the "
             "fixed-length per-position hub signature. Default: 20.",
    )

    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--merge-into-existing", action="store_true")
    parser.add_argument("--no-npz", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)

    args = parser.parse_args()

    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
    elif "HF_HOME" not in os.environ:
        ssd_default = "/Volumes/ORICO/huggingface_cache"
        if os.path.isdir(os.path.dirname(ssd_default)):
            os.environ["HF_HOME"] = ssd_default

    device = get_device(args.device)
    args.device = device

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    protein_name, sequence = resolve_sequence(args)
    seq_len = len(sequence)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.positions:
        positions = sorted(set(args.positions))
        bad = [p for p in positions if p < 0 or p >= seq_len]
        if bad:
            raise ValueError(f"Positions out of range: {bad}")
    else:
        positions = list(range(seq_len))

    print(f"=== ESM-2 Phase 1 oscillatory analysis (patched) ===")
    print(f"  substrate:           {args.substrate}")
    print(f"  model:               {args.model}")
    print(f"  device:              {device}")
    print(f"  protein:             {protein_name} (len={seq_len})")
    print(f"  positions:           {len(positions)}")
    print(f"  output:              {out_dir.resolve()}")
    print(f"  amp floor quantile:  {args.substrate_amp_floor_quantile}")
    print(f"  n hubs in signature: {args.n_hubs_in_signature}")
    print(f"  merge existing:      {args.merge_into_existing}")

    print(f"\n[1/5] Loading substrate ...")
    substrate = load_substrate(args.substrate)
    L_sub = int(substrate["n_sublayers_used"])
    D_sub = int(substrate["d_model"])
    trim_sub = int(substrate["trim_sublayers"])

    sub_amp_mean = substrate["amp_mean"]
    floor = float(np.quantile(sub_amp_mean, args.substrate_amp_floor_quantile))
    valid_mask = (sub_amp_mean >= floor)
    valid_floor = {
        "quantile": float(args.substrate_amp_floor_quantile),
        "absolute_value": floor,
    }
    print(f"  L={L_sub}, D={D_sub}, trim={trim_sub}")
    print(f"  validity floor: amp_mean >= {floor:.4f} "
          f"(q={args.substrate_amp_floor_quantile}); "
          f"{valid_mask.sum()}/{valid_mask.size} cells valid "
          f"({100*valid_mask.mean():.1f}%)")

    centrality = substrate["hub_centrality"]
    n_hubs = int(args.n_hubs_in_signature)
    if n_hubs > len(centrality):
        n_hubs = len(centrality)
    hub_indices = np.argsort(centrality)[::-1][:n_hubs].astype(np.int32)
    print(f"  hub signature: top {n_hubs} units, "
          f"centrality range [{centrality[hub_indices[-1]]:.2e}, "
          f"{centrality[hub_indices[0]]:.2e}]")

    print(f"\n[2/5] Loading {args.model} ...")
    from transformers import EsmTokenizer, EsmForMaskedLM
    tokenizer = EsmTokenizer.from_pretrained(args.model)
    model = EsmForMaskedLM.from_pretrained(
        args.model, attn_implementation="eager",
    ).to(device).to(torch.float32)
    model.eval()
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    if d_model != D_sub:
        raise ValueError(
            f"Model d_model={d_model} != substrate d_model={D_sub}."
        )
    print(f"  n_layers={n_layers}, d_model={d_model}")

    substrate_path_str = str(Path(args.substrate).resolve())

    npz_amp = []
    npz_phase = []
    npz_freq = []
    npz_harm2 = []
    npz_amp_mod = []
    npz_freq_dev = []
    npz_harm2_dev = []
    npz_fiedler = []
    npz_top_k = []
    npz_top_k_eigvals = []
    npz_hub_max_amp_mod = []
    npz_hub_argmax = []
    scalar_keys = (
        "fiedler_val", "spectral_entropy", "mean_off_diag_R",
        "effective_rank", "corr_frobenius_cos_substrate",
        "mean_hub_max_amp_mod", "max_hub_max_amp_mod",
    )
    scalars = {k: [] for k in scalar_keys}

    summaries = []
    failures = []

    print(f"\n[3/5] Processing {len(positions)} positions ...")
    t_start = time.time()
    for i, pos in enumerate(positions):
        aa = sequence[pos]
        try:
            t0 = time.time()
            raw = collect_streams_at_position(
                model, tokenizer, sequence, pos, device,
            )
            raw = raw - raw.mean(axis=0, keepdims=True)
            if trim_sub > 0:
                if raw.shape[0] <= 2 * trim_sub + 2:
                    raise ValueError(
                        f"Cannot trim {trim_sub} from each end of "
                        f"{raw.shape[0]} sublayers."
                    )
                raw = raw[trim_sub:raw.shape[0] - trim_sub]
            if raw.shape[0] != L_sub:
                raise ValueError(
                    f"Post-trim L={raw.shape[0]} != substrate L={L_sub}."
                )

            result = analyze_position(
                raw, substrate, valid_mask, hub_indices,
                k_eigvecs=args.top_k_eigvecs,
            )
            summary = build_position_summary(
                pos, aa, result, substrate, substrate_path_str,
                valid_mask, hub_indices, valid_floor,
                top_n_units=args.top_n_units,
            )
            summaries.append(summary)

            out_json = out_dir / f"pos_{pos}_oscillatory.json"
            with open(out_json, "w") as f:
                json.dump(summary, f, indent=1)

            if args.merge_into_existing:
                existing = out_dir / f"pos_{pos}.json"
                merged = merge_into_existing(existing, summary)
                merge_note = " (merged)" if merged else " (no existing)"
            else:
                merge_note = ""

            if not args.no_npz:
                npz_amp.append(result["amp"])
                npz_phase.append(result["phase"])
                npz_freq.append(result["freq"])
                npz_harm2.append(result["harm2_amp"])
                npz_amp_mod.append(result["amp_modulation"])
                npz_freq_dev.append(result["freq_deviation"])
                npz_harm2_dev.append(result["harm2_deviation"])
                npz_fiedler.append(result["fiedler_vec"])
                npz_top_k.append(result["top_k_eigvecs"])
                npz_top_k_eigvals.append(result["top_k_eigvals"])
                npz_hub_max_amp_mod.append(result["hub_max_amp_mod"])
                npz_hub_argmax.append(result["hub_argmax_sublayer"])
                for k in scalar_keys:
                    scalars[k].append(result[k])

            elapsed = time.time() - t0
            print(f"  [{i + 1}/{len(positions)}] pos={pos} ({aa}): "
                  f"eff_rank={result['effective_rank']:.2f}, "
                  f"frob_cos={result['corr_frobenius_cos_substrate']:.3f}, "
                  f"mean_hub={result['mean_hub_max_amp_mod']:.2f}, "
                  f"{elapsed:.1f}s{merge_note}")

        except Exception as e:
            print(f"  [{i + 1}/{len(positions)}] pos={pos} ({aa}) FAILED: {e}")
            failures.append((pos, str(e)))

    total_elapsed = time.time() - t_start
    n_ok = max(1, len(positions) - len(failures))
    print(f"  Total: {total_elapsed:.1f}s ({total_elapsed/n_ok:.1f}s/position).")
    if failures:
        print(f"  Failures: {len(failures)} positions.")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    if not args.no_npz and len(npz_amp) > 0:
        print(f"\n[4/5] Writing protein-level npz ...")
        npz_path = out_dir / f"{slugify(protein_name)}_oscillatory.npz"
        success_positions = [s["position"] for s in summaries]
        np.savez(
            npz_path,
            positions=np.array(success_positions, dtype=np.int32),
            sequence=np.array(list(sequence), dtype="U1"),
            n_sublayers=np.int32(L_sub),
            d_model=np.int32(D_sub),
            trim_sublayers=np.int32(trim_sub),
            valid_mask=valid_mask.astype(bool),
            valid_amp_floor=np.float32(floor),
            valid_amp_floor_quantile=np.float32(args.substrate_amp_floor_quantile),
            hub_indices=hub_indices,
            hub_centralities=np.array(
                [centrality[u] for u in hub_indices], dtype=np.float32
            ),
            amp=np.stack(npz_amp, axis=0).astype(np.float32),
            phase=np.stack(npz_phase, axis=0).astype(np.float32),
            freq=np.stack(npz_freq, axis=0).astype(np.float32),
            harm2_amp=np.stack(npz_harm2, axis=0).astype(np.float32),
            amp_modulation=np.stack(npz_amp_mod, axis=0).astype(np.float32),
            freq_deviation=np.stack(npz_freq_dev, axis=0).astype(np.float32),
            harm2_deviation=np.stack(npz_harm2_dev, axis=0).astype(np.float32),
            fiedler_vecs=np.stack(npz_fiedler, axis=0).astype(np.float32),
            top_k_eigvecs=np.stack(npz_top_k, axis=0).astype(np.float32),
            top_k_eigvals=np.stack(npz_top_k_eigvals, axis=0).astype(np.float32),
            hub_max_amp_mod=np.stack(npz_hub_max_amp_mod, axis=0).astype(np.float32),
            hub_argmax_sublayer=np.stack(npz_hub_argmax, axis=0).astype(np.int32),
            **{k: np.array(v, dtype=np.float32) for k, v in scalars.items()},
        )
        size_mb = npz_path.stat().st_size / (1024 * 1024)
        print(f"  {npz_path.name}: {size_mb:.1f} MB")
    else:
        print(f"\n[4/5] Skipping protein-level npz.")

    if not args.no_plot and len(summaries) > 0:
        print(f"\n[5/5] Plots ...")
        ok_positions = [s["position"] for s in summaries]

        ov_path = out_dir / f"{slugify(protein_name)}_oscillatory_overview.png"
        plot_overview(ok_positions, sequence, summaries, ov_path)
        print(f"  {ov_path.name}")

        hub_path = out_dir / f"{slugify(protein_name)}_hub_signatures.png"
        hub_signatures = [s["hub_signature"]["max_amp_mod"]
                          for s in summaries]
        hub_signatures_arr = np.array(
            [[(v if v is not None else np.nan) for v in sig]
             for sig in hub_signatures],
            dtype=np.float32,
        )
        plot_hub_signatures(
            ok_positions, sequence, hub_signatures_arr,
            hub_indices, centrality, hub_path,
        )
        print(f"  {hub_path.name}")

    if summaries:
        eff_rank = np.array([s["spectral"]["effective_rank"] for s in summaries])
        fro_cos = np.array([s["substrate_comparison"]["corr_frobenius_cos"]
                            for s in summaries])
        mean_hub = np.array([s["hub_signature"]["mean_max_amp_mod"]
                             for s in summaries])
        max_hub = np.array([s["hub_signature"]["max_max_amp_mod"]
                            for s in summaries])
        print(f"\n--- Per-position summary across {len(summaries)} positions ---")
        print(f"  effective rank:     mean={eff_rank.mean():.2f} "
              f"std={eff_rank.std():.2f} "
              f"range=[{eff_rank.min():.2f}, {eff_rank.max():.2f}]")
        print(f"  |R|-|C| frob cos:   mean={fro_cos.mean():.3f} "
              f"std={fro_cos.std():.3f} "
              f"range=[{fro_cos.min():.3f}, {fro_cos.max():.3f}]")
        print(f"  mean hub amp_mod:   mean={mean_hub.mean():.2f} "
              f"std={mean_hub.std():.2f} "
              f"range=[{mean_hub.min():.2f}, {mean_hub.max():.2f}]")
        print(f"  max hub amp_mod:    mean={max_hub.mean():.2f} "
              f"std={max_hub.std():.2f} "
              f"range=[{max_hub.min():.2f}, {max_hub.max():.2f}]")

    print(f"\nDone. Outputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
