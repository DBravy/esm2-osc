"""
ESM-2 Phase 2 (phase channel): per-unit and pairwise phase MI
==============================================================

Phase 2 originally analyzed amp_modulation, freq_deviation, harm2_deviation,
and hub_max_amp_mod. The phase channel was deliberately deferred because
phase doesn't have a clean per-cell substrate-relative scalar: the
framework's claim is that phase carries relational content, which is
intrinsically pairwise.

This script computes two phase-channel observables:

  phase_dev_per_unit  (n_pos, L, D)
      For each cell, 1 - cos(theta_pos - theta_circular_mean), where
      theta_circular_mean is the cross-position circular mean of phase at
      that cell. Bounded in [0, 2]: 0 at-mean, 2 antipodal.

  phase_coh_pairwise  (n_pos, L, N_pairs)
      For the top-N pairs by substrate |C_uv|, the per-position pair-phase
      coherence with the substrate canonical offset:
          cos((theta_u - theta_v) - arg(C_uv)).
      Bounded in [-1, 1]: +1 at canonical, -1 antipodal.

Each channel is then run through the same MI machinery as Phase 2 and
saved as an mi_<channel>.npz that the existing esm2_phase2_rerank.py
picks up automatically.

Outputs (in --output-dir):
  mi_phase_dev_per_unit.npz       Compatible with rerank script.
                                  Includes 'circular_mean_phase' for
                                  downstream interpretation.
  mi_phase_coh_pairwise.npz       Compatible with rerank script.
                                  Includes 'pair_units' (N_pairs, 2),
                                  'pair_substrate_coupling' and
                                  'pair_canonical_offset' so the
                                  rerank's "unit" axis can be mapped
                                  back to (u, v) pairs.

Usage:
  # Run alongside existing Phase 2 output (same dir):
  python esm2_phase2_phase_channel.py \
      --phase1-npz ubiquitin/ubiquitin_oscillatory.npz \
      --substrate esm2_substrate_facebook_esm2_t33_650M_UR50D/substrate.npz \
      --load-properties-from-pos-jsons ubiquitin_old \
      --output-dir esm2_phase2_ubiquitin

  # Then rerun rerank to merge into a unified summary:
  python esm2_phase2_rerank.py --phase2-dir esm2_phase2_ubiquitin

  # Restrict to per-unit only (faster):
  python esm2_phase2_phase_channel.py ... --no-pairwise

  # Larger pair set:
  python esm2_phase2_phase_channel.py ... --n-top-pairs 5000
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

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# ---------------------------------------------------------------------------
# AA constants (duplicated from phase2_mi.py for self-containment)
# ---------------------------------------------------------------------------

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INT = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

HYDRO_KD = {
    "A":  1.8, "C":  2.5, "D": -3.5, "E": -3.5, "F":  2.8,
    "G": -0.4, "H": -3.2, "I":  4.5, "K": -3.9, "L":  3.8,
    "M":  1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V":  4.2, "W": -0.9, "Y": -1.3,
}
CHARGE = {
    "D": -1.0, "E": -1.0, "K":  1.0, "R":  1.0, "H":  0.1,
    "A": 0.0, "C": 0.0, "F": 0.0, "G": 0.0, "I": 0.0, "L": 0.0, "M": 0.0,
    "N": 0.0, "P": 0.0, "Q": 0.0, "S": 0.0, "T": 0.0, "V": 0.0, "W": 0.0,
    "Y": 0.0,
}
VOLUME = {
    "A":  88.6, "C": 108.5, "D": 111.1, "E": 138.4, "F": 189.9,
    "G":  60.1, "H": 153.2, "I": 166.7, "K": 168.6, "L": 166.7,
    "M": 162.9, "N": 114.1, "P": 112.7, "Q": 143.8, "R": 173.4,
    "S":  89.0, "T": 116.1, "V": 140.0, "W": 227.8, "Y": 193.6,
}


# ---------------------------------------------------------------------------
# Property building (duplicated from phase2_mi.py)
# ---------------------------------------------------------------------------

def build_auto_properties(sequence, positions):
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
        arr = (np.asarray(values, dtype=np.int32) if kind == "categorical"
               else np.asarray(values, dtype=np.float32))
        out[name] = (kind, arr)
    return out


def load_properties_from_pos_jsons(pos_dir, positions):
    pos_dir = Path(pos_dir)
    contacts, coevolving = [], []
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
# Phase observables
# ---------------------------------------------------------------------------

def compute_per_unit_phase_deviation(phase):
    """phase: (n_pos, L, D). Returns (n_pos, L, D) with values in [0, 2]
    and also (L, D) circular mean phase as the substrate reference."""
    z = np.exp(1j * phase.astype(np.float64))                # (n_pos, L, D)
    mean_z = z.mean(axis=0)                                  # (L, D)
    theta_mean = np.angle(mean_z).astype(np.float32)         # (L, D)
    # Deviation: 1 - cos(theta_pos - theta_mean).
    delta = phase - theta_mean[None, :, :]
    phase_dev = (1.0 - np.cos(delta)).astype(np.float32)
    return phase_dev, theta_mean


def select_top_pairs(coupling_abs, n_top_pairs):
    """coupling_abs: (D, D) symmetric, zero diagonal. Returns indices
    (us, vs) of top-N pairs by |C|, upper triangle only (u < v)."""
    D = coupling_abs.shape[0]
    iu, iv = np.triu_indices(D, k=1)
    abs_vals = coupling_abs[iu, iv]
    if n_top_pairs >= len(abs_vals):
        order = np.argsort(abs_vals)[::-1]
        return iu[order], iv[order], abs_vals[order]
    order = np.argpartition(abs_vals, -n_top_pairs)[-n_top_pairs:]
    sub_abs = abs_vals[order]
    sort = np.argsort(sub_abs)[::-1]
    order = order[sort]
    return iu[order], iv[order], abs_vals[order]


def compute_pairwise_phase_coherence(phase, us, vs, canonical_offsets):
    """phase: (n_pos, L, D).
    us, vs: (N_pairs,) integer indices.
    canonical_offsets: (N_pairs,) substrate arg(C_uv) values.
    Returns (n_pos, L, N_pairs) values in [-1, 1]."""
    phase_u = phase[:, :, us].astype(np.float64)             # (n_pos, L, N)
    phase_v = phase[:, :, vs].astype(np.float64)
    delta = phase_u - phase_v - canonical_offsets[None, None, :]
    return np.cos(delta).astype(np.float32)


# ---------------------------------------------------------------------------
# MI machinery (duplicated from phase2_mi.py)
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
                         valid_mask=None, n_perm=10, n_neighbors=3,
                         seed=0):
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

    mi_valid = _mi_call(X, property_values, kind, n_neighbors, seed)

    rng = np.random.default_rng(seed)
    null_mi_valid = np.zeros((n_perm, n_valid), dtype=np.float32)
    for i in range(n_perm):
        perm = rng.permutation(n_pos)
        null_mi_valid[i] = _mi_call(
            X, property_values[perm], kind, n_neighbors, seed + i + 1,
        )

    raw_mi = np.zeros(n_features_total, dtype=np.float32)
    raw_mi[valid_idx] = mi_valid.astype(np.float32)
    raw_mi = raw_mi.reshape(feature_shape)

    null_mi = np.zeros((n_perm, n_features_total), dtype=np.float32)
    null_mi[:, valid_idx] = null_mi_valid
    null_mi = null_mi.reshape((n_perm,) + feature_shape)
    return {"raw_mi": raw_mi, "null_mi": null_mi}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slugify(name):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--phase1-npz", required=True)
    parser.add_argument("--substrate", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--no-auto-properties", action="store_true")
    parser.add_argument("--properties-json", default=None)
    parser.add_argument("--load-properties-from-pos-jsons", default=None)

    parser.add_argument("--n-neighbors", type=int, default=3)
    parser.add_argument("--n-permutations", type=int, default=10)

    # Channel selection
    parser.add_argument("--no-per-unit", action="store_true",
                        help="Skip the per-unit phase deviation channel.")
    parser.add_argument("--no-pairwise", action="store_true",
                        help="Skip the pairwise phase coherence channel.")
    parser.add_argument("--n-top-pairs", type=int, default=2000,
                        help="Number of top substrate pairs by |C_uv| to "
                             "include in the pairwise channel. Default: 2000.")

    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    if args.no_per_unit and args.no_pairwise:
        raise ValueError("Both --no-per-unit and --no-pairwise; nothing to do.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Phase 2 phase-channel MI ===")
    print(f"  phase1_npz:   {args.phase1_npz}")
    print(f"  substrate:    {args.substrate}")
    print(f"  output_dir:   {out_dir.resolve()}")
    print(f"  n_top_pairs:  {args.n_top_pairs}")

    # --- Load Phase 1 data ---
    print(f"\n[1/5] Loading Phase 1 npz ...")
    p1 = dict(np.load(args.phase1_npz))
    if "phase" not in p1:
        raise ValueError(
            f"{args.phase1_npz} has no 'phase' key. Re-run Phase 1 to get it."
        )
    phase = p1["phase"]                                   # (n_pos, L, D)
    positions = p1["positions"].tolist()
    sequence = "".join(p1["sequence"].tolist())
    n_pos = len(positions)
    L = int(p1["n_sublayers"])
    D = int(p1["d_model"])
    print(f"  phase shape: {phase.shape}, n_pos={n_pos}, L={L}, D={D}")

    # Validity mask (for per-unit phase, masked the same way as amp_modulation)
    if "valid_mask" in p1:
        valid_mask = p1["valid_mask"].astype(bool)
    else:
        valid_mask = np.ones((L, D), dtype=bool)
        print(f"  WARNING: no valid_mask in phase1 npz; using all-true.")

    # --- Load substrate ---
    print(f"\n[2/5] Loading substrate ...")
    sub = dict(np.load(args.substrate))
    if "coupling_phase" not in sub or "coupling_abs" not in sub:
        raise ValueError(
            f"{args.substrate} missing coupling_phase or coupling_abs."
        )
    coupling_abs = sub["coupling_abs"].astype(np.float32)
    coupling_phase = sub["coupling_phase"].astype(np.float32)
    if coupling_abs.shape != (D, D):
        raise ValueError(
            f"coupling_abs shape {coupling_abs.shape} != (D, D) = ({D}, {D})."
        )
    np.fill_diagonal(coupling_abs, 0.0)

    # --- Build properties ---
    print(f"\n[3/5] Building properties ...")
    properties = {}
    if not args.no_auto_properties:
        properties.update(build_auto_properties(sequence, positions))
    if args.load_properties_from_pos_jsons:
        properties.update(load_properties_from_pos_jsons(
            args.load_properties_from_pos_jsons, positions,
        ))
    if args.properties_json:
        properties.update(load_user_properties(args.properties_json, n_pos))
    if not properties:
        raise ValueError("No properties available.")
    print(f"  properties: {list(properties.keys())}")
    for name, (kind, vals) in properties.items():
        if kind == "categorical":
            n_unique = int(np.unique(vals).size)
            print(f"    {name:25s} ({kind}, n_unique={n_unique})")
        else:
            print(f"    {name:25s} ({kind}, "
                  f"range=[{vals.min():.2f}, {vals.max():.2f}])")

    n_perm = max(0, int(args.n_permutations))

    # --- Per-unit phase channel ---
    if not args.no_per_unit:
        print(f"\n[4a/5] Computing phase_dev_per_unit channel ...")
        t0 = time.time()
        phase_dev, theta_mean = compute_per_unit_phase_deviation(phase)
        print(f"  observable shape: {phase_dev.shape}, "
              f"computed in {time.time()-t0:.1f}s")
        print(f"  range: [{phase_dev.min():.3f}, {phase_dev.max():.3f}], "
              f"mean: {phase_dev.mean():.3f}")

        npz_dict = {}
        for prop_name, (kind, vals) in properties.items():
            t0 = time.time()
            res = compute_mi_with_null(
                phase_dev, vals, kind,
                valid_mask=valid_mask,
                n_perm=n_perm,
                n_neighbors=args.n_neighbors,
                seed=args.seed,
            )
            elapsed = time.time() - t0
            key = slugify(prop_name)
            npz_dict[f"raw_{key}"] = res["raw_mi"]
            npz_dict[f"null_{key}"] = res["null_mi"]
            print(f"    {prop_name:25s} ({kind}): "
                  f"max_mi={res['raw_mi'].max():.3f}, "
                  f"max_excess={res['raw_mi'].max() - res['null_mi'].mean():.3f}, "
                  f"{elapsed:.1f}s")
        npz_dict["property_names"] = np.array(
            list(properties.keys()), dtype="U64",
        )
        npz_dict["circular_mean_phase"] = theta_mean
        out_path = out_dir / "mi_phase_dev_per_unit.npz"
        np.savez(out_path, **npz_dict)
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"  -> {out_path.name} ({size_mb:.2f} MB)")

    # --- Pairwise phase channel ---
    if not args.no_pairwise:
        print(f"\n[4b/5] Computing phase_coh_pairwise channel ...")
        n_top = min(args.n_top_pairs, D * (D - 1) // 2)
        us, vs, sub_abs_pair = select_top_pairs(coupling_abs, n_top)
        canonical_offsets = coupling_phase[us, vs].astype(np.float64)
        print(f"  selected top {len(us)} pairs by |C|")
        print(f"  pair |C| range: "
              f"[{sub_abs_pair.min():.3f}, {sub_abs_pair.max():.3f}]")

        t0 = time.time()
        phase_coh = compute_pairwise_phase_coherence(
            phase, us, vs, canonical_offsets,
        )
        print(f"  observable shape: {phase_coh.shape}, "
              f"computed in {time.time()-t0:.1f}s")
        print(f"  range: [{phase_coh.min():.3f}, {phase_coh.max():.3f}], "
              f"mean: {phase_coh.mean():.3f}")

        npz_dict = {}
        for prop_name, (kind, vals) in properties.items():
            t0 = time.time()
            res = compute_mi_with_null(
                phase_coh, vals, kind,
                valid_mask=None,  # pairs pre-filtered by |C|
                n_perm=n_perm,
                n_neighbors=args.n_neighbors,
                seed=args.seed,
            )
            elapsed = time.time() - t0
            key = slugify(prop_name)
            npz_dict[f"raw_{key}"] = res["raw_mi"]
            npz_dict[f"null_{key}"] = res["null_mi"]
            print(f"    {prop_name:25s} ({kind}): "
                  f"max_mi={res['raw_mi'].max():.3f}, "
                  f"max_excess={res['raw_mi'].max() - res['null_mi'].mean():.3f}, "
                  f"{elapsed:.1f}s")
        npz_dict["property_names"] = np.array(
            list(properties.keys()), dtype="U64",
        )
        # IMPORTANT: pair metadata so rerank's "unit" axis can be mapped
        # back to (u, v) pairs via pair_units[unit_idx] = [u, v].
        npz_dict["pair_units"] = np.stack([us, vs], axis=1).astype(np.int32)
        npz_dict["pair_substrate_coupling"] = sub_abs_pair.astype(np.float32)
        npz_dict["pair_canonical_offset"] = canonical_offsets.astype(np.float32)
        out_path = out_dir / "mi_phase_coh_pairwise.npz"
        np.savez(out_path, **npz_dict)
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"  -> {out_path.name} ({size_mb:.2f} MB)")

    print(f"\n[5/5] Done. To merge into a unified summary, rerun:")
    print(f"  python esm2_phase2_rerank.py --phase2-dir {out_dir}")


if __name__ == "__main__":
    main()
