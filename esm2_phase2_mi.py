"""
ESM-2 Phase 2: channel-property mutual information
====================================================

Consumes the per-protein oscillatory.npz from Phase 1 and a substrate.npz
from the substrate-characterization stage. For each oscillatory channel
and each input property, computes mutual information between the channel's
per-(unit, sublayer) value across positions and the property value across
positions.

Channels analyzed:
  amp_modulation     (n_pos, L, D)    masked by substrate validity mask.
  freq_deviation     (n_pos, L-1, D)  masked by neighbor-conjunction mask.
  harm2_deviation    (n_pos, D)       no per-cell mask.
  hub_max_amp_mod    (n_pos, n_hubs)  no per-cell mask (hubs are valid by
                                       construction).

Properties:
  Auto from sequence:
    aa_identity      (categorical, 20-way)
    hydrophobicity   (continuous, Kyte-Doolittle)
    charge           (continuous, net at pH 7)
    volume           (continuous, side-chain volume in A^3)
  Optional from --load-properties-from-pos-jsons:
    n_structural_contacts
    n_coevolving_partners
  Optional from --properties-json:
    arbitrary {name: {kind, values}} dict

Per channel-property MI:
  raw_mi      cell-shaped MI between channel cell values and property.
  null_mi     (n_perm, cell-shaped) null distribution from property
              permutation.
  z_score     (raw_mi - null_mean) / null_std, computed per cell in summary.

Outputs (in --output-dir):
  mi_amp_modulation.npz, mi_freq_deviation.npz,
  mi_harm2_deviation.npz, mi_hub_max_amp_mod.npz
                                  Each contains raw_<prop> and null_<prop>
                                  arrays plus property metadata.
  mi_summary.json                 Top cells per (channel, property), global
                                  top cells, property metadata.
  properties.npz                  All property arrays (n_pos,) per name.
  mi_amp_modulation_heatmaps.png  Per-property MI heatmap grid.
  mi_freq_deviation_heatmaps.png  Same for freq_deviation.
  mi_1d_channels_bars.png         Bar charts for harm2 and hub channels.
  mi_global_top_cells.png         Global top-K cells across everything.

Usage:
  python esm2_phase2_mi.py \
      --phase1-npz /ubiquitin/ubiquitin_oscillatory.npz \
      --substrate /esm2_substrate_facebook_esm2_t33_650M_UR50D/substrate.npz \
      --load-properties-from-pos-jsons ubiquitin_old \
      --output-dir esm2_phase2_ubiquitin

  # With existing per-position JSONs from generate_viz_data.py:
  python esm2_phase2_mi.py --phase1-npz ubiquitin_oscillatory.npz \
      --substrate substrate.npz --output-dir out \
      --load-properties-from-pos-jsons /Volumes/ORICO/esm2_viz_data/ubiquitin

  # With a custom properties JSON:
  python esm2_phase2_mi.py --phase1-npz ubiquitin_oscillatory.npz \
      --substrate substrate.npz --output-dir out \
      --properties-json my_properties.json --n-permutations 20

Properties JSON format:
  {
    "conservation": {"kind": "continuous", "values": [0.85, 0.34, ...]},
    "is_helix":     {"kind": "categorical", "values": [0, 0, 1, 1, 0, ...]},
    ...
  }
  Length of values must equal the number of analyzed positions.
"""

import argparse
import json
import os
import re
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_selection import (
    mutual_info_classif, mutual_info_regression,
)

# Suppress sklearn's frequent UserWarnings about constant features etc.
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# ---------------------------------------------------------------------------
# AA constants
# ---------------------------------------------------------------------------

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INT = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

# Kyte-Doolittle hydrophobicity
HYDRO_KD = {
    "A":  1.8, "C":  2.5, "D": -3.5, "E": -3.5, "F":  2.8,
    "G": -0.4, "H": -3.2, "I":  4.5, "K": -3.9, "L":  3.8,
    "M":  1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V":  4.2, "W": -0.9, "Y": -1.3,
}

# Net charge at pH 7. Histidine partially protonated (~10%).
CHARGE = {
    "D": -1.0, "E": -1.0, "K":  1.0, "R":  1.0, "H":  0.1,
    "A": 0.0, "C": 0.0, "F": 0.0, "G": 0.0, "I": 0.0, "L": 0.0, "M": 0.0,
    "N": 0.0, "P": 0.0, "Q": 0.0, "S": 0.0, "T": 0.0, "V": 0.0, "W": 0.0,
    "Y": 0.0,
}

# Side-chain volume (Zamyatnin 1972), Angstrom^3
VOLUME = {
    "A":  88.6, "C": 108.5, "D": 111.1, "E": 138.4, "F": 189.9,
    "G":  60.1, "H": 153.2, "I": 166.7, "K": 168.6, "L": 166.7,
    "M": 162.9, "N": 114.1, "P": 112.7, "Q": 143.8, "R": 173.4,
    "S":  89.0, "T": 116.1, "V": 140.0, "W": 227.8, "Y": 193.6,
}


# ---------------------------------------------------------------------------
# Property building
# ---------------------------------------------------------------------------

def build_auto_properties(sequence, positions):
    """Per-position AA-derived scalars and categorical."""
    aa = [sequence[p] for p in positions]
    return {
        "aa_identity": (
            "categorical",
            np.array([AA_TO_INT[a] for a in aa], dtype=np.int32),
        ),
        "hydrophobicity": (
            "continuous",
            np.array([HYDRO_KD[a] for a in aa], dtype=np.float32),
        ),
        "charge": (
            "continuous",
            np.array([CHARGE[a] for a in aa], dtype=np.float32),
        ),
        "volume": (
            "continuous",
            np.array([VOLUME[a] for a in aa], dtype=np.float32),
        ),
    }


def load_user_properties(path, n_pos):
    """Load arbitrary properties from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    out = {}
    for name, entry in data.items():
        kind = entry.get("kind", "continuous")
        values = entry["values"]
        if len(values) != n_pos:
            raise ValueError(
                f"Property '{name}' has {len(values)} values, expected {n_pos}."
            )
        if kind == "categorical":
            arr = np.asarray(values, dtype=np.int32)
        else:
            arr = np.asarray(values, dtype=np.float32)
        out[name] = (kind, arr)
    return out


def load_properties_from_pos_jsons(pos_dir, positions):
    """Extract scalar properties from existing pos_{i}.json files written
    by generate_viz_data.py (annotations.structural_contacts and
    annotations.coevolving_positions)."""
    pos_dir = Path(pos_dir)
    contacts = []
    coevolving = []
    found_any = False
    for p in positions:
        fp = pos_dir / f"pos_{p}.json"
        if fp.exists():
            found_any = True
            with open(fp) as f:
                d = json.load(f)
            ann = d.get("annotations", {})
            contacts.append(len(ann.get("structural_contacts", [])))
            coevolving.append(len(ann.get("coevolving_positions", [])))
        else:
            contacts.append(0)
            coevolving.append(0)
    if not found_any:
        print(f"  WARNING: no pos_*.json files found in {pos_dir}; "
              "skipping JSON-derived properties.")
        return {}
    out = {}
    if any(c > 0 for c in contacts):
        out["n_structural_contacts"] = (
            "continuous", np.array(contacts, dtype=np.float32),
        )
    if any(c > 0 for c in coevolving):
        out["n_coevolving_partners"] = (
            "continuous", np.array(coevolving, dtype=np.float32),
        )
    return out


# ---------------------------------------------------------------------------
# MI computation
# ---------------------------------------------------------------------------

def _mi_call(X, y, kind, n_neighbors, random_state):
    if kind == "categorical":
        return mutual_info_classif(
            X, y, n_neighbors=n_neighbors, random_state=random_state,
        )
    return mutual_info_regression(
        X, y, n_neighbors=n_neighbors, random_state=random_state,
    )


def compute_mi_with_null(channel_data, property_values, kind,
                         valid_mask=None, n_perm=10, n_neighbors=3, seed=0):
    """channel_data: (n_pos, *feature_shape).
    property_values: (n_pos,).
    valid_mask: optional (feature_shape,) bool. Cells outside the mask are
        not passed to the MI estimator; their MI is returned as 0.
    Returns dict with 'raw_mi' (feature_shape) and 'null_mi'
        (n_perm, feature_shape)."""
    n_pos = channel_data.shape[0]
    feature_shape = channel_data.shape[1:]
    X_full = channel_data.reshape(n_pos, -1).astype(np.float64)
    n_features_total = X_full.shape[1]

    if valid_mask is None:
        valid_idx = np.arange(n_features_total)
        X = X_full
    else:
        valid_idx = np.where(valid_mask.ravel())[0]
        X = X_full[:, valid_idx]
    n_valid = X.shape[1]

    # Real MI
    mi_valid = _mi_call(X, property_values, kind, n_neighbors, seed)

    # Null MI from permutations
    rng = np.random.default_rng(seed)
    null_mi_valid = np.zeros((n_perm, n_valid), dtype=np.float32)
    for i in range(n_perm):
        perm = rng.permutation(n_pos)
        null_mi_valid[i] = _mi_call(
            X, property_values[perm], kind, n_neighbors, seed + i + 1,
        )

    # Expand back to full shape
    raw_mi = np.zeros(n_features_total, dtype=np.float32)
    raw_mi[valid_idx] = mi_valid.astype(np.float32)
    raw_mi = raw_mi.reshape(feature_shape)

    null_mi = np.zeros((n_perm, n_features_total), dtype=np.float32)
    null_mi[:, valid_idx] = null_mi_valid
    null_mi = null_mi.reshape((n_perm,) + feature_shape)

    return {"raw_mi": raw_mi, "null_mi": null_mi}


def per_cell_z_scores(raw_mi, null_mi):
    """Per-cell z-score = (raw - null_mean) / null_std.
    Cells with null_std == 0 (e.g. masked-out cells) get z = 0.
    raw_mi: (*feature_shape).
    null_mi: (n_perm, *feature_shape)."""
    null_mean = null_mi.mean(axis=0)
    null_std = null_mi.std(axis=0)
    safe_std = np.where(null_std > 1e-12, null_std, 1.0)
    z = (raw_mi - null_mean) / safe_std
    z = np.where(null_std > 1e-12, z, 0.0)
    return z, null_mean, null_std


# ---------------------------------------------------------------------------
# Top-cells extraction
# ---------------------------------------------------------------------------

def top_cells_2d(raw_mi, null_mi, top_k):
    """For (L, D) channel. Returns list of dicts sorted by z-score desc."""
    z, null_mean, null_std = per_cell_z_scores(raw_mi, null_mi)
    L, D = raw_mi.shape
    flat_z = z.ravel()
    order = np.argsort(flat_z)[::-1][:top_k]
    cells = []
    for idx in order:
        ell, u = divmod(int(idx), D)
        cells.append({
            "sublayer": int(ell),
            "unit": int(u),
            "raw_mi": float(raw_mi[ell, u]),
            "null_mi_mean": float(null_mean[ell, u]),
            "null_mi_std": float(null_std[ell, u]),
            "z_score": float(z[ell, u]),
        })
    return cells


def top_cells_1d(raw_mi, null_mi, top_k, index_label="unit"):
    """For (D,) or (n_hubs,) channel. Returns list of dicts."""
    z, null_mean, null_std = per_cell_z_scores(raw_mi, null_mi)
    order = np.argsort(z)[::-1][:top_k]
    cells = []
    for idx in order:
        cells.append({
            index_label: int(idx),
            "raw_mi": float(raw_mi[idx]),
            "null_mi_mean": float(null_mean[idx]),
            "null_mi_std": float(null_std[idx]),
            "z_score": float(z[idx]),
        })
    return cells


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_mi_heatmaps_2d(channel_name, mi_results, properties, hub_centrality,
                        top_units_by_centrality, out_path):
    """Grid of MI heatmaps for a (L, D) channel, one per property.
    Units sorted by centrality (descending) on y axis (top-N only).
    Sublayers on x axis."""
    prop_names = list(mi_results.keys())
    n_props = len(prop_names)
    if n_props == 0:
        return

    n_units_show = len(top_units_by_centrality)
    fig, axes = plt.subplots(
        n_props, 1, figsize=(14, 2.5 * n_props),
        squeeze=False,
    )
    for ax, prop in zip(axes[:, 0], prop_names):
        raw_mi = mi_results[prop]["raw_mi"]
        # raw_mi: (L, D) for amp_modulation, (L-1, D) for freq_deviation
        L, D = raw_mi.shape
        M = raw_mi[:, top_units_by_centrality].T  # (n_units_show, L)
        finite = M[np.isfinite(M)]
        vmax = float(np.nanpercentile(finite, 99)) if finite.size else 1.0
        if vmax <= 0:
            vmax = 1.0
        im = ax.imshow(M, aspect="auto", cmap="viridis",
                       vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_xlabel("sublayer", fontsize=8)
        ax.set_ylabel("unit (descending centrality)", fontsize=8)
        ax.set_title(f"{channel_name} -> {prop}", fontsize=10)
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="raw MI")

    fig.suptitle(
        f"MI heatmaps: {channel_name} channel  vs each property\n"
        f"top-{n_units_show} units by hub centrality on y axis",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_mi_bars_1d(channel_results, channel_shapes, out_path,
                    top_k_show=15):
    """Bar charts of top units per property, for the 1D channels
    (harm2_deviation, hub_max_amp_mod)."""
    channel_names = [c for c in channel_results.keys()
                     if c in ("harm2_deviation", "hub_max_amp_mod")]
    if not channel_names:
        return

    rows = []
    for ch in channel_names:
        for prop in channel_results[ch].keys():
            rows.append((ch, prop))
    if not rows:
        return

    n_rows = len(rows)
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(12, 2.2 * n_rows), squeeze=False,
    )
    for ax, (ch, prop) in zip(axes[:, 0], rows):
        raw_mi = channel_results[ch][prop]["raw_mi"]
        null_mi = channel_results[ch][prop]["null_mi"]
        z, null_mean, null_std = per_cell_z_scores(raw_mi, null_mi)
        order = np.argsort(z)[::-1][:top_k_show]
        labels = [str(int(i)) for i in order]
        x_axis = np.arange(len(order))
        ax.bar(x_axis - 0.18, raw_mi[order], width=0.36,
               color="C0", label="raw MI")
        ax.bar(x_axis + 0.18, null_mean[order], width=0.36,
               yerr=null_std[order], color="C7",
               label="null mean +/- std")
        ax.set_xticks(x_axis)
        ax.set_xticklabels(labels, fontsize=7, rotation=0)
        idx_label = "hub idx" if ch == "hub_max_amp_mod" else "unit"
        ax.set_xlabel(idx_label, fontsize=8)
        ax.set_ylabel("MI", fontsize=8)
        ax.set_title(f"{ch} -> {prop}: top-{top_k_show} by z-score",
                     fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("MI on 1D channels (harm2_deviation and hub_max_amp_mod)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_global_top_cells(global_top, out_path, top_k_show=30):
    """Horizontal bar chart of top-K (channel, property, cell) entries
    by z-score."""
    if not global_top:
        return
    entries = global_top[:top_k_show][::-1]  # bottom of plot = highest z
    labels = []
    z_vals = []
    raw_vals = []
    for e in entries:
        ch = e["channel"]
        prop = e["property"]
        if ch in ("amp_modulation", "freq_deviation"):
            cell_id = f"L={e['sublayer']},u={e['unit']}"
        elif ch == "hub_max_amp_mod":
            cell_id = f"hub#{e['hub_idx']}(u{e['unit']})"
        else:
            cell_id = f"u={e['unit']}"
        labels.append(f"{ch} | {prop} | {cell_id}")
        z_vals.append(e["z_score"])
        raw_vals.append(e["raw_mi"])

    fig, ax = plt.subplots(figsize=(11, 0.32 * len(entries) + 1.2))
    y_pos = np.arange(len(entries))
    ax.barh(y_pos, z_vals, color="C2", alpha=0.85)
    for y, raw in zip(y_pos, raw_vals):
        ax.text(0.05, y, f"raw={raw:.3f}",
                va="center", fontsize=7, color="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("z-score (raw - null_mean) / null_std")
    ax.set_title(f"Global top {len(entries)} (channel, property, cell) "
                 "entries by z-score")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slugify(s):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)


def safe_npz_key(name):
    """Convert a property name into a valid npz key fragment."""
    return slugify(name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--phase1-npz", required=True,
                        help="protein-level oscillatory.npz from Phase 1.")
    parser.add_argument("--substrate", required=True,
                        help="substrate.npz with the validity mask info.")

    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--protein-name", default=None,
                        help="Override protein name for output filenames.")

    # Properties
    parser.add_argument("--no-auto-properties", action="store_true",
                        help="Skip the auto AA-derived properties.")
    parser.add_argument("--properties-json", default=None,
                        help="Optional JSON file with extra properties: "
                             "{name: {kind: 'categorical'|'continuous', "
                             "values: [...]}}.")
    parser.add_argument("--load-properties-from-pos-jsons", default=None,
                        help="Path to a directory with pos_{i}.json files "
                             "(from generate_viz_data.py). Extracts "
                             "n_structural_contacts and n_coevolving_partners.")

    # MI estimator
    parser.add_argument("--n-neighbors", type=int, default=3,
                        help="k for the k-NN MI estimator. Default: 3.")
    parser.add_argument("--n-permutations", type=int, default=10,
                        help="Number of permutations for null MI. "
                             "Default: 10. Use 0 to skip.")
    parser.add_argument("--top-k-cells", type=int, default=20,
                        help="Top-K cells reported per (channel, property).")
    parser.add_argument("--top-units-in-plots", type=int, default=200,
                        help="Number of top-by-centrality units shown on the "
                             "y axis of (L, D) heatmaps.")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== ESM-2 Phase 2: channel-property MI ===")
    print(f"  phase1_npz:   {args.phase1_npz}")
    print(f"  substrate:    {args.substrate}")
    print(f"  output_dir:   {out_dir.resolve()}")

    # --- Load Phase 1 data ---
    print(f"\n[1/5] Loading Phase 1 npz ...")
    p1 = dict(np.load(args.phase1_npz))
    positions = p1["positions"].tolist()
    sequence = "".join(p1["sequence"].tolist())
    n_pos = len(positions)
    L = int(p1["n_sublayers"])
    D = int(p1["d_model"])
    print(f"  n_pos={n_pos}, L={L}, D={D}")

    protein_name = args.protein_name
    if protein_name is None:
        # Infer from filename
        protein_name = Path(args.phase1_npz).stem.replace("_oscillatory", "")
    print(f"  protein name (for output files): {protein_name}")

    # --- Load substrate (need valid_mask if not in phase1) ---
    print(f"\n[2/5] Loading substrate (for validity mask) ...")
    sub = dict(np.load(args.substrate))
    if "valid_mask" in p1:
        valid_mask = p1["valid_mask"].astype(bool)
    elif "amp_mean" in sub:
        # Fallback: rebuild from substrate using a 0.10 quantile floor
        amp_mean = sub["amp_mean"]
        floor = float(np.quantile(amp_mean, 0.10))
        valid_mask = (amp_mean >= floor)
        print(f"  WARNING: phase1 npz has no valid_mask; rebuilt from "
              f"substrate at q=0.10 (floor={floor:.3f}).")
    else:
        valid_mask = np.ones((L, D), dtype=bool)
        print(f"  WARNING: no validity info found; using all-true mask.")
    print(f"  valid_mask: {valid_mask.sum()}/{valid_mask.size} cells "
          f"({100*valid_mask.mean():.1f}%)")

    # Derived freq mask: a freq bin is valid only if both endpoints are.
    if valid_mask.shape[0] > 1:
        freq_valid_mask = valid_mask[:-1] & valid_mask[1:]
    else:
        freq_valid_mask = valid_mask

    # Hub indices (saved by Phase 1)
    if "hub_indices" in p1:
        hub_indices = p1["hub_indices"]
        n_hubs = len(hub_indices)
    else:
        hub_indices = None
        n_hubs = p1.get("hub_max_amp_mod", np.zeros((1, 0))).shape[1]

    # --- Build properties ---
    print(f"\n[3/5] Building properties ...")
    properties = {}
    if not args.no_auto_properties:
        properties.update(build_auto_properties(sequence, positions))
        print(f"  auto-properties: {list(build_auto_properties(sequence, positions).keys())}")
    if args.load_properties_from_pos_jsons:
        loaded = load_properties_from_pos_jsons(
            args.load_properties_from_pos_jsons, positions,
        )
        properties.update(loaded)
        print(f"  pos_jsons properties: {list(loaded.keys())}")
    if args.properties_json:
        loaded = load_user_properties(args.properties_json, n_pos)
        properties.update(loaded)
        print(f"  user-json properties: {list(loaded.keys())}")

    if not properties:
        raise ValueError("No properties available. Pass at least one source.")

    print(f"\n  Total properties: {len(properties)}")
    for name, (kind, vals) in properties.items():
        if kind == "categorical":
            n_unique = int(np.unique(vals).size)
            print(f"    {name:25s} ({kind}, n_unique={n_unique})")
        else:
            print(f"    {name:25s} ({kind}, "
                  f"range=[{vals.min():.2f}, {vals.max():.2f}])")

    # Save property arrays for transparency / reproducibility
    prop_npz_path = out_dir / "properties.npz"
    np.savez(
        prop_npz_path,
        positions=np.array(positions, dtype=np.int32),
        sequence=np.array(list(sequence), dtype="U1"),
        **{
            f"{safe_npz_key(name)}__values": vals
            for name, (kind, vals) in properties.items()
        },
        property_names=np.array(list(properties.keys()), dtype="U64"),
        property_kinds=np.array(
            [kind for (kind, _) in properties.values()], dtype="U16",
        ),
    )
    print(f"  Saved {prop_npz_path.name}")

    # --- Define channels and their masks ---
    channel_specs = [
        ("amp_modulation",  p1["amp_modulation"],  valid_mask),
        ("freq_deviation",  p1["freq_deviation"],  freq_valid_mask),
        ("harm2_deviation", p1["harm2_deviation"], None),
        ("hub_max_amp_mod", p1["hub_max_amp_mod"], None),
    ]

    # Sanity check: check shapes
    for ch_name, ch_data, ch_mask in channel_specs:
        if ch_data.shape[0] != n_pos:
            raise ValueError(
                f"Channel {ch_name} has {ch_data.shape[0]} positions, "
                f"expected {n_pos}."
            )
        if ch_mask is not None and ch_mask.shape != ch_data.shape[1:]:
            raise ValueError(
                f"Channel {ch_name}: mask shape {ch_mask.shape} != "
                f"data feature shape {ch_data.shape[1:]}."
            )

    # --- Compute MI for each (channel, property) ---
    print(f"\n[4/5] Computing MI ...")
    n_perm = max(0, int(args.n_permutations))

    # Replace NaNs in any channel array (could appear in hub_max_amp_mod when
    # a hub has no valid sublayers). Replace with 0 so MI estimator is happy.
    def _clean(x):
        if np.isnan(x).any():
            return np.where(np.isnan(x), 0.0, x)
        return x

    channel_results = {}
    for ch_name, ch_data, ch_mask in channel_specs:
        print(f"\n  Channel: {ch_name} (shape per position {ch_data.shape[1:]})")
        ch_data_clean = _clean(ch_data)
        channel_results[ch_name] = {}
        for prop_name, (kind, vals) in properties.items():
            t0 = time.time()
            res = compute_mi_with_null(
                ch_data_clean, vals, kind,
                valid_mask=ch_mask,
                n_perm=n_perm,
                n_neighbors=args.n_neighbors,
                seed=args.seed,
            )
            elapsed = time.time() - t0
            channel_results[ch_name][prop_name] = res
            print(f"    {prop_name:25s} ({kind}): "
                  f"max_mi={res['raw_mi'].max():.3f}, "
                  f"mean_null={res['null_mi'].mean():.3f}, "
                  f"{elapsed:.1f}s")

    # --- Save per-channel npz ---
    print(f"\n[5/5] Saving outputs ...")
    for ch_name, prop_dict in channel_results.items():
        npz_dict = {}
        for prop_name, res in prop_dict.items():
            key = safe_npz_key(prop_name)
            npz_dict[f"raw_{key}"] = res["raw_mi"]
            npz_dict[f"null_{key}"] = res["null_mi"]
        npz_dict["property_names"] = np.array(
            list(prop_dict.keys()), dtype="U64",
        )
        out_path = out_dir / f"mi_{ch_name}.npz"
        np.savez(out_path, **npz_dict)
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"  {out_path.name}: {size_mb:.2f} MB")

    # --- Build JSON summary ---
    print(f"\n  Building summary JSON ...")
    summary = {
        "n_positions": int(n_pos),
        "n_sublayers": int(L),
        "d_model": int(D),
        "n_hubs": int(n_hubs),
        "n_neighbors": int(args.n_neighbors),
        "n_permutations": int(n_perm),
        "channels": [c for (c, _, _) in channel_specs],
        "properties": {
            name: {
                "kind": kind,
                "values_range": (
                    [int(vals.min()), int(vals.max())]
                    if kind == "categorical"
                    else [float(vals.min()), float(vals.max())]
                ),
            }
            for name, (kind, vals) in properties.items()
        },
        "top_cells_by_channel_property": {},
        "global_top_cells": [],
    }

    global_pool = []
    for ch_name, prop_dict in channel_results.items():
        summary["top_cells_by_channel_property"][ch_name] = {}
        for prop_name, res in prop_dict.items():
            raw = res["raw_mi"]
            null = res["null_mi"]
            if raw.ndim == 2:
                cells = top_cells_2d(raw, null, args.top_k_cells)
                for c in cells:
                    global_pool.append({
                        "channel": ch_name, "property": prop_name,
                        "sublayer": c["sublayer"], "unit": c["unit"],
                        "raw_mi": c["raw_mi"], "z_score": c["z_score"],
                        "null_mi_mean": c["null_mi_mean"],
                        "null_mi_std": c["null_mi_std"],
                    })
            else:
                idx_label = ("hub_idx" if ch_name == "hub_max_amp_mod"
                             else "unit")
                cells = top_cells_1d(raw, null, args.top_k_cells,
                                     index_label=idx_label)
                for c in cells:
                    entry = {
                        "channel": ch_name, "property": prop_name,
                        "raw_mi": c["raw_mi"], "z_score": c["z_score"],
                        "null_mi_mean": c["null_mi_mean"],
                        "null_mi_std": c["null_mi_std"],
                    }
                    if idx_label == "hub_idx":
                        hub_idx = c[idx_label]
                        entry["hub_idx"] = hub_idx
                        if hub_indices is not None and hub_idx < n_hubs:
                            entry["unit"] = int(hub_indices[hub_idx])
                    else:
                        entry["unit"] = c[idx_label]
                    global_pool.append(entry)
            summary["top_cells_by_channel_property"][ch_name][prop_name] = cells

    # Global top cells by z-score
    global_pool.sort(key=lambda e: e["z_score"], reverse=True)
    summary["global_top_cells"] = global_pool[:args.top_k_cells * 4]

    # Write summary JSON
    summary_path = out_dir / "mi_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=1, default=lambda x: float(x))
    print(f"  {summary_path.name}")

    # --- Plots ---
    centrality = sub.get("hub_centrality", None)
    if centrality is None:
        # Fallback: use coupling_degree from substrate
        centrality = sub.get("coupling_degree", np.zeros(D))
    top_units_by_centrality = (
        np.argsort(centrality)[::-1][:args.top_units_in_plots]
    )

    # Two heatmap panels: amp_modulation, freq_deviation
    if "amp_modulation" in channel_results:
        ov_path = out_dir / f"mi_amp_modulation_heatmaps.png"
        plot_mi_heatmaps_2d(
            "amp_modulation",
            channel_results["amp_modulation"],
            properties, centrality,
            top_units_by_centrality,
            ov_path,
        )
        print(f"  {ov_path.name}")
    if "freq_deviation" in channel_results:
        ov_path = out_dir / f"mi_freq_deviation_heatmaps.png"
        plot_mi_heatmaps_2d(
            "freq_deviation",
            channel_results["freq_deviation"],
            properties, centrality,
            top_units_by_centrality,
            ov_path,
        )
        print(f"  {ov_path.name}")

    # 1D channel bars
    bars_path = out_dir / f"mi_1d_channels_bars.png"
    plot_mi_bars_1d(
        channel_results,
        {ch: ch_data.shape[1:] for ch, ch_data, _ in channel_specs},
        bars_path,
    )
    print(f"  {bars_path.name}")

    # Global top cells
    global_path = out_dir / f"mi_global_top_cells.png"
    plot_global_top_cells(summary["global_top_cells"], global_path,
                          top_k_show=30)
    print(f"  {global_path.name}")

    # --- Console summary ---
    print(f"\n--- Top global signals (z-score) ---")
    for entry in summary["global_top_cells"][:15]:
        ch = entry["channel"]
        prop = entry["property"]
        if ch in ("amp_modulation", "freq_deviation"):
            cell_id = f"L={entry['sublayer']:2d},u={entry['unit']:4d}"
        elif ch == "hub_max_amp_mod":
            cell_id = (f"hub#{entry.get('hub_idx', '?'):2d}"
                       f"(u{entry.get('unit', '?'):4d})")
        else:
            cell_id = f"u={entry.get('unit', '?'):4d}"
        print(f"  {ch:18s} | {prop:25s} | {cell_id} | "
              f"raw={entry['raw_mi']:.3f}, z={entry['z_score']:.2f}")

    print(f"\nDone. Outputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
