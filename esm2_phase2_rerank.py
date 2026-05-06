"""
Phase 2 patch: rerank channel-property MI cells from existing npz files.
=========================================================================

The original Phase 2 ranked cells by z-score = (raw - null_mean) / null_std.
For cells where the property carries no signal, sklearn's k-NN MI estimator
returns near-zero values clipped at 0, giving null_std ~ 1e-5. Dividing by
that produces astronomical z-scores for any cell with even slight raw MI,
biased toward properties whose nulls are narrowest (categorical, near-binary).

This patch reranks using three more robust metrics, computed offline from
the saved raw_mi and null_mi arrays. No MI recomputation needed.

Metrics:
  excess_mi      raw_mi - null_mi.mean(axis=0). Cleanest measure of "MI
                 above the typical permutation baseline." Default ranking.
  empirical_p    fraction of permutations with null_mi >= raw_mi. With
                 n_perm=10 takes values in {0/10, ..., 10/10}; coarse but
                 unambiguous.
  z_floored      (raw - null_mean) / max(null_std, floor) where floor is
                 the per-(channel, property) median null_std. Preserves
                 z-score-style ranking but caps the tiny-null-std blowup.

Inputs:
  --phase2-dir         directory containing mi_<channel>.npz and (optionally)
                       mi_summary.json from the original Phase 2 run.

Outputs (in --phase2-dir, side by side with the originals):
  mi_summary_v2.json
  mi_global_top_cells_v2.png

Usage:
  python esm2_phase2_rerank.py --phase2-dir esm2_phase2_ubiquitin

  # Rank by z_floored instead:
  python esm2_phase2_rerank.py --phase2-dir esm2_phase2_ubiquitin \\
      --rank-by z_floored

  # Custom null_std floor (overrides per-(channel, property) median):
  python esm2_phase2_rerank.py --phase2-dir esm2_phase2_ubiquitin \\
      --null-std-floor 1e-3
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Discovery and loading
# ---------------------------------------------------------------------------

CHANNEL_FILE_PATTERN = re.compile(r"^mi_(.+)\.npz$")


def discover_channel_npzs(phase2_dir):
    """Find all mi_<channel>.npz files (excluding properties.npz)."""
    out = {}
    for p in Path(phase2_dir).iterdir():
        if not p.is_file():
            continue
        m = CHANNEL_FILE_PATTERN.match(p.name)
        if not m:
            continue
        channel = m.group(1)
        if channel == "summary":
            continue
        if p.name == "properties.npz":
            continue
        out[channel] = p
    if not out:
        raise FileNotFoundError(
            f"No mi_<channel>.npz files in {phase2_dir}"
        )
    return out


def load_channel_data(npz_path):
    """Returns dict {property_name: {raw_mi, null_mi}} from one channel npz."""
    d = dict(np.load(npz_path))
    if "property_names" not in d:
        raise ValueError(
            f"{npz_path} has no 'property_names' key; was it written by the "
            "patched Phase 2 script?"
        )
    prop_names = [str(p) for p in d["property_names"].tolist()]
    out = {}
    for name in prop_names:
        # The Phase 2 script slugifies names for npz keys but here the
        # property_names array preserves the original. Both should map
        # to the same key string under the slugify function in Phase 2.
        # Be tolerant: try the name verbatim first, then a slugified
        # version.
        slug = _slugify(name)
        raw_key = f"raw_{slug}"
        null_key = f"null_{slug}"
        if raw_key not in d or null_key not in d:
            raise KeyError(
                f"{npz_path}: missing keys {raw_key!r} or {null_key!r}; "
                f"available: {sorted(k for k in d if k.startswith(('raw_', 'null_')))}"
            )
        out[name] = {
            "raw_mi": d[raw_key],
            "null_mi": d[null_key],
        }
    return out


def _slugify(name):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


# ---------------------------------------------------------------------------
# Per-cell metric computation
# ---------------------------------------------------------------------------

def compute_metrics(raw_mi, null_mi, null_std_floor=None):
    """raw_mi: (*shape).
    null_mi: (n_perm, *shape).
    Returns dict with same shapes:
      raw_mi
      null_mean
      null_std
      excess_mi      raw - null_mean
      empirical_p    fraction of null >= raw
      z_floored      (raw - null_mean) / max(null_std, floor)
    Floor: if null_std_floor is None, uses median(null_std) across the array.
    """
    null_mean = null_mi.mean(axis=0)
    null_std = null_mi.std(axis=0)

    excess = raw_mi - null_mean

    # Empirical p-value
    n_perm = null_mi.shape[0]
    ge = (null_mi >= raw_mi[None, ...]).sum(axis=0)
    empirical_p = ge.astype(np.float32) / n_perm

    # z_floored
    if null_std_floor is None:
        non_zero = null_std[null_std > 0]
        if non_zero.size > 0:
            floor = float(np.median(non_zero))
        else:
            floor = 1e-3
    else:
        floor = float(null_std_floor)

    safe_std = np.maximum(null_std, floor)
    z_floored = excess / safe_std

    return {
        "raw_mi":      raw_mi.astype(np.float32),
        "null_mean":   null_mean.astype(np.float32),
        "null_std":    null_std.astype(np.float32),
        "excess_mi":   excess.astype(np.float32),
        "empirical_p": empirical_p.astype(np.float32),
        "z_floored":   z_floored.astype(np.float32),
        "null_std_floor_used": float(floor),
    }


# ---------------------------------------------------------------------------
# Top-cell extraction
# ---------------------------------------------------------------------------

def top_cells_2d(metrics, top_k, rank_by, channel, prop):
    """For (L, D) channel. Returns list of dicts sorted by chosen metric."""
    L, D = metrics["raw_mi"].shape
    score = metrics[rank_by].ravel()
    order = np.argsort(score)[::-1][:top_k]
    cells = []
    for idx in order:
        ell, u = divmod(int(idx), D)
        cells.append({
            "channel": channel,
            "property": prop,
            "sublayer": int(ell),
            "unit": int(u),
            "raw_mi": float(metrics["raw_mi"][ell, u]),
            "null_mean": float(metrics["null_mean"][ell, u]),
            "null_std": float(metrics["null_std"][ell, u]),
            "excess_mi": float(metrics["excess_mi"][ell, u]),
            "empirical_p": float(metrics["empirical_p"][ell, u]),
            "z_floored": float(metrics["z_floored"][ell, u]),
        })
    return cells


def top_cells_1d(metrics, top_k, rank_by, channel, prop, index_label="unit"):
    """For (D,) or (n_hubs,) channel."""
    score = metrics[rank_by]
    order = np.argsort(score)[::-1][:top_k]
    cells = []
    for idx in order:
        cells.append({
            "channel": channel,
            "property": prop,
            index_label: int(idx),
            "raw_mi": float(metrics["raw_mi"][idx]),
            "null_mean": float(metrics["null_mean"][idx]),
            "null_std": float(metrics["null_std"][idx]),
            "excess_mi": float(metrics["excess_mi"][idx]),
            "empirical_p": float(metrics["empirical_p"][idx]),
            "z_floored": float(metrics["z_floored"][idx]),
        })
    return cells


# ---------------------------------------------------------------------------
# Plot: global top cells
# ---------------------------------------------------------------------------

def plot_global_top_cells(global_top, rank_by, out_path, top_k_show=30):
    if not global_top:
        return
    entries = global_top[:top_k_show][::-1]
    labels = []
    score_vals = []
    raw_vals = []
    excess_vals = []
    for e in entries:
        ch = e["channel"]
        prop = e["property"]
        if ch in ("amp_modulation", "freq_deviation"):
            cell_id = f"L={e['sublayer']:>2},u={e['unit']:>4}"
        elif ch == "hub_max_amp_mod":
            hub_idx = e.get("hub_idx", e.get("unit", "?"))
            cell_id = f"hub#{hub_idx:>2}"
        else:
            cell_id = f"u={e.get('unit', '?'):>4}"
        labels.append(f"{ch:18s} | {prop:22s} | {cell_id}")
        score_vals.append(e[rank_by])
        raw_vals.append(e["raw_mi"])
        excess_vals.append(e["excess_mi"])

    fig, ax = plt.subplots(figsize=(13, 0.32 * len(entries) + 1.5))
    y_pos = np.arange(len(entries))
    ax.barh(y_pos, score_vals, color="C2", alpha=0.85)
    for y, raw, exc in zip(y_pos, raw_vals, excess_vals):
        ax.text(score_vals[y] * 1.01, y,
                f"raw={raw:.3f}, excess={exc:.3f}",
                va="center", fontsize=7, color="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7, family="monospace")
    ax.set_xlabel(f"{rank_by} (ranking metric)")
    ax.set_title(f"Global top {len(entries)} (channel, property, cell) "
                 f"by {rank_by}  [v2 patched ranking]")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--phase2-dir", required=True,
                        help="Phase 2 output directory containing "
                             "mi_<channel>.npz files.")
    parser.add_argument(
        "--rank-by", default="excess_mi",
        choices=["excess_mi", "z_floored", "raw_mi"],
        help="Metric used to rank cells. Default: excess_mi.",
    )
    parser.add_argument(
        "--null-std-floor", type=float, default=None,
        help="Floor for null_std in z_floored calculation. If unset, uses "
             "the per-(channel, property) median null_std.",
    )
    parser.add_argument(
        "--top-k-cells", type=int, default=20,
        help="Top-K cells reported per (channel, property). Default: 20.",
    )
    parser.add_argument(
        "--top-k-global", type=int, default=80,
        help="Total cells in global_top_cells. Default: 80.",
    )
    parser.add_argument(
        "--top-k-show", type=int, default=30,
        help="Top cells in the global plot. Default: 30.",
    )

    args = parser.parse_args()
    phase2_dir = Path(args.phase2_dir)

    print(f"=== Phase 2 patch: rerank MI cells ===")
    print(f"  phase2_dir: {phase2_dir.resolve()}")
    print(f"  rank_by:    {args.rank_by}")

    # --- Discover channel files ---
    channel_files = discover_channel_npzs(phase2_dir)
    print(f"\n[1/3] Found {len(channel_files)} channel files: "
          f"{sorted(channel_files.keys())}")

    # --- Load original summary if present (for cross-reference) ---
    orig_summary_path = phase2_dir / "mi_summary.json"
    orig_summary = None
    if orig_summary_path.exists():
        with open(orig_summary_path) as f:
            orig_summary = json.load(f)
        print(f"  Original summary loaded ({orig_summary['n_positions']} pos, "
              f"{orig_summary['n_permutations']} perms).")

    # --- Compute metrics per (channel, property), build new summary ---
    print(f"\n[2/3] Computing metrics per (channel, property) ...")
    new_summary = {
        "ranking_metric": args.rank_by,
        "n_permutations": None,
        "channels": list(channel_files.keys()),
        "properties": [],
        "null_std_floor_per_channel_property": {},
        "top_cells_by_channel_property": {},
        "global_top_cells": [],
    }
    if orig_summary:
        new_summary["n_positions"] = orig_summary.get("n_positions")
        new_summary["n_sublayers"] = orig_summary.get("n_sublayers")
        new_summary["d_model"] = orig_summary.get("d_model")
        new_summary["n_hubs"] = orig_summary.get("n_hubs")

    global_pool = []
    all_props = set()

    for ch_name, ch_path in channel_files.items():
        print(f"\n  Channel: {ch_name}")
        ch_data = load_channel_data(ch_path)
        new_summary["top_cells_by_channel_property"][ch_name] = {}
        new_summary["null_std_floor_per_channel_property"][ch_name] = {}
        for prop_name, arrs in ch_data.items():
            all_props.add(prop_name)
            raw_mi = arrs["raw_mi"]
            null_mi = arrs["null_mi"]
            n_perm = null_mi.shape[0]
            if new_summary["n_permutations"] is None:
                new_summary["n_permutations"] = int(n_perm)

            metrics = compute_metrics(
                raw_mi, null_mi,
                null_std_floor=args.null_std_floor,
            )
            new_summary["null_std_floor_per_channel_property"][ch_name][prop_name] = (
                float(metrics["null_std_floor_used"])
            )

            # Per-cell top-K, ranking by chosen metric.
            if raw_mi.ndim == 2:
                cells = top_cells_2d(metrics, args.top_k_cells,
                                     args.rank_by, ch_name, prop_name)
            else:
                idx_label = ("hub_idx" if ch_name == "hub_max_amp_mod"
                             else "unit")
                cells = top_cells_1d(metrics, args.top_k_cells,
                                     args.rank_by, ch_name, prop_name,
                                     index_label=idx_label)
                # For hubs, also add the substrate-unit index if accessible.
            new_summary["top_cells_by_channel_property"][ch_name][prop_name] = cells

            # Pool for global ranking.
            global_pool.extend(cells)

            # Console print: top 3 cells.
            print(f"    {prop_name:25s}: floor={metrics['null_std_floor_used']:.2e}, "
                  f"max_raw={raw_mi.max():.3f}, "
                  f"max_excess={metrics['excess_mi'].max():.3f}")

    new_summary["properties"] = sorted(all_props)

    # --- Global top cells ---
    global_pool.sort(key=lambda e: e[args.rank_by], reverse=True)
    new_summary["global_top_cells"] = global_pool[:args.top_k_global]

    # --- Save new summary JSON ---
    print(f"\n[3/3] Writing outputs ...")
    out_summary = phase2_dir / "mi_summary_v2.json"
    with open(out_summary, "w") as f:
        json.dump(new_summary, f, indent=1, default=lambda x: float(x))
    print(f"  {out_summary.name}")

    # --- Updated global plot ---
    out_plot = phase2_dir / "mi_global_top_cells_v2.png"
    plot_global_top_cells(
        new_summary["global_top_cells"], args.rank_by, out_plot,
        top_k_show=args.top_k_show,
    )
    print(f"  {out_plot.name}")

    # --- Console summary ---
    print(f"\n--- Top {min(20, args.top_k_global)} global cells "
          f"by {args.rank_by} ---")
    for entry in new_summary["global_top_cells"][:20]:
        ch = entry["channel"]
        prop = entry["property"]
        if ch in ("amp_modulation", "freq_deviation"):
            cell_id = f"L={entry['sublayer']:>2},u={entry['unit']:>4}"
        elif ch == "hub_max_amp_mod":
            hub_idx = entry.get("hub_idx", "?")
            cell_id = f"hub#{hub_idx}"
        else:
            cell_id = f"u={entry.get('unit', '?'):>4}"
        print(f"  {ch:18s} | {prop:25s} | {cell_id} | "
              f"raw={entry['raw_mi']:.3f}, "
              f"excess={entry['excess_mi']:.3f}, "
              f"emp_p={entry['empirical_p']:.2f}, "
              f"z_floored={entry['z_floored']:.2f}")

    print(f"\nDone. New summary in {out_summary.resolve()}")


if __name__ == "__main__":
    main()
