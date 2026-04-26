"""
run_nextgen.py — Next Generation NMM experiments on HCP-DK68 connectome.

Six experiments map the methodological replacement of Wilson–Cowan with
Next Gen NMM (Forrester 2024) on real human connectivity (ENIGMA Toolbox,
n=207 HCP subjects). Only BOLD-FC is empirically validated; |Z|, PLV,
and gap-junction effects are NG-only capability demonstrations.

  scale            — Distance-scale calibration (multi-seed sweep, bootstrap CI)
  topology         — NG on Conti's 4 toy graphs (bridge to WC testbed)
  gap              — Gap-junction effect on static + dynamic FC (paired t)
  ng_capabilities  — |Z_E|(t) demo (NG-only observable WC cannot produce)
  scfc             — Structure–function: SC vs sim BOLD FC vs empirical FC
  delays           — Constant vs distance-based vs gamma-velocity delays

Conventions:
  - Adaptive transient discard: max(30 s, 0.2 × duration).
  - BOLD-FC experiments use 500 s default; 120 s default for diagnostic.
  - All r vs empirical use ENIGMA pipeline alignment (max(0,r) → arctanh)
    matching Royer 2022 / ENIGMA Toolbox docs.
  - Distance scale s = 1.5 adopted (within working band, see `scale`).
  - Multi-seed for hypothesis-testing experiments (gap N=5, scfc N=3,
    delays N=3, topology N=5, scale N=3); single-seed + bootstrap CI for
    the calibration scan.
  - Stochastic delay modes (matrix_gamma_velocity, gamma, heterogeneous_velocity)
    use ctx.args.seed (default 42); deterministic 'matrix' / 'constant'
    modes give identical output regardless of seed.
  - Each experiment saves a `.npz` of raw arrays + PNG figure(s) into
    experiments/Results/ and experiments/Plots/ respectively.

Statistical infrastructure:
  - bootstrap_r_ci      — 95% CI on Pearson r (1000 resamples of edges)
  - permutation_null_p  — null p value (1000 permutations of empirical labels)
  - paired_seed_stats   — paired t-test across multi-seed conditions
  - _steiger_z_correlated_r — Williams-Steiger Z for two correlated r's
  - enigma_align        — ENIGMA pipeline (max(0)→arctanh) on simulated FC

Usage examples:
  python3 experiments/run_nextgen.py --exp scale --save
  python3 experiments/run_nextgen.py --exp gap --save
  python3 experiments/run_nextgen.py --exp ng_capabilities --save
  python3 experiments/run_nextgen.py --exp all --save
  python3 experiments/run_nextgen.py --exp delays --delay-modes constant,matrix

Author: Ningqian — PBM Group Project 2026 (Next Gen NMM track)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from next_gen_network import NextGenNetwork

# ====================================================================
# Constants & paths
# ====================================================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(THIS_DIR)
DATA_DIR = os.path.join(REPO_DIR, 'data')
PLOTS_DIR = os.path.join(REPO_DIR, 'experiments', 'Plots')
RESULTS_DIR = os.path.join(REPO_DIR, 'experiments', 'Results')

OPTIMAL_DISTANCE_SCALE = 1.5            # empirically calibrated (exp scale)
DEFAULT_K_EXT = 0.2                      # Forrester p.7
DEFAULT_KAPPA_V_EE = 0.01                # Forrester p.7
DEFAULT_KAPPA_V_II = 0.025               # Forrester p.7
DEFAULT_DURATION = 120.0                 # seconds (s) — fast/diagnostic
BOLD_FC_DURATION = 500.0                 # paper-grade for BOLD FC fit (Forrester / CLAUDE.md)
SCALE_DURATION   = BOLD_FC_DURATION      # back-compat alias

# Multi-seed counts for hypothesis-testing experiments. Sensitivity sweeps
# (scale) stay single-seed by design — see CLAUDE.md "Statistical principles".
GAP_N_SEEDS      = 5     # exp_gap binary {off, on} × N seeds, paired t
DELAY_N_SEEDS    = 3     # exp_delays {constant, matrix} × N seeds, paired t
SCFC_N_SEEDS     = 3     # exp_scfc multi-seed averaging
TOPOLOGY_N_SEEDS = 5     # exp_topology 4 toy graphs × N seeds, ANOVA
SCALE_N_SEEDS    = 3     # exp_scale 6 scales × N seeds, mean ± SE per scale
TOPOLOGY_N_NODES = 16    # Conti & Van Gorder used N=16 for path/cycle/complete/lattice
TOPOLOGY_DURATION = 60.0 # toy graphs are small → fast; 60 s plenty for stable |Z|/PLV
DT = 0.001                               # 1 ms output step


# ====================================================================
# Plot style — module-wide. Figures are saved as separate PNGs without
# titles or in-figure annotations; the user authors captions in the
# report. Each figure plots one quantity (no mixed dimensions).
# ====================================================================
plt.rcParams.update({
    'font.size':        11,
    'axes.labelsize':   12,
    'axes.titlesize':   12,
    'xtick.labelsize':  10,
    'ytick.labelsize':  10,
    'legend.fontsize':  12,
    'legend.frameon':   True,
    'legend.framealpha': 0.8,
    'figure.dpi':       100,
    'savefig.dpi':      150,
    'savefig.bbox':     'tight',
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'grid.linewidth':   0.5,
})

# Three-colour palette: red/blue contrast + neutral grey
COLOR_PRIMARY   = '#1f77b4'    # blue  (main series, line/scatter)
COLOR_SECONDARY = '#d62728'    # red   (second series, contrast)
COLOR_GREY      = '#888888'    # neutral (reference lines, chance levels)


def sweep_palette(n: int):
    """`n` evenly-spaced colours from matplotlib `viridis` for a sweep series.
    viridis is perceptually uniform and standard for ordinal sweep variables."""
    cmap = plt.cm.viridis
    if n <= 1:
        return [cmap(0.5)]
    return [cmap(i / (n - 1)) for i in range(n)]


# FC matrix colormap: diverging RdBu_r is the standard for fMRI BOLD FC
# (Pearson correlations span ±1). Forrester Fig 6/8-10 use a sequential
# (viridis-style) colormap because their FC is GIM/MIM (≥ 0). For
# BOLD FC we follow the fMRI convention.
FC_CMAP = 'RdBu_r'


# ====================================================================
# Common infrastructure
# ====================================================================
@dataclass
class Context:
    """Shared experiment context (data + args)."""
    sc:        np.ndarray           # (N, N)  SC weights
    emp_fc:    Optional[np.ndarray] # (N, N)  empirical FC (None if missing)
    labels:    np.ndarray           # (N,)    DK68 region names
    distance:  np.ndarray           # (N, N)  Euclidean centroid distance (mm)
    args:      argparse.Namespace
    plots_dir: str
    results_dir: str


class HCPNetwork:
    """Wrapper that NextGenNetwork accepts (needs .N, .A)."""
    def __init__(self, sc: np.ndarray):
        self.A = sc
        self.N = sc.shape[0]


def load_context(args) -> Context:
    """Load HCP data, build Context. Errors clearly if data is missing."""
    sc_path  = os.path.join(DATA_DIR, 'hcp_sc_68.npy')
    fc_path  = os.path.join(DATA_DIR, 'hcp_fc_68.npy')
    lab_path = os.path.join(DATA_DIR, 'hcp_labels_68.npy')
    dist_path= os.path.join(DATA_DIR, 'hcp_dist_68.npy')

    missing = [p for p in (sc_path, dist_path) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing required data files:\n  " + "\n  ".join(missing)
            + "\nRun: python3 data/prepare_data.py")

    sc      = np.load(sc_path)
    distance= np.load(dist_path)
    labels  = (np.load(lab_path, allow_pickle=True)
               if os.path.exists(lab_path)
               else np.array([f'Region {i}' for i in range(sc.shape[0])]))
    emp_fc  = np.load(fc_path) if os.path.exists(fc_path) else None
    if emp_fc is None:
        warnings.warn("Empirical FC not found; FC-fit metrics will be NaN.")

    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    return Context(
        sc=sc, emp_fc=emp_fc, labels=labels, distance=distance,
        args=args, plots_dir=PLOTS_DIR, results_dir=RESULTS_DIR,
    )


def fc_corr(simulated: np.ndarray, empirical: Optional[np.ndarray]) -> float:
    """Pearson correlation between upper triangles of two FC matrices.
    Returns NaN if input has no variance, NaNs, or is None.
    """
    if empirical is None or simulated is None:
        return float('nan')
    mask = np.triu_indices(simulated.shape[0], k=1)
    s, e = simulated[mask], empirical[mask]
    if (np.std(s) == 0 or np.std(e) == 0
            or np.any(np.isnan(s)) or np.any(np.isnan(e))):
        return float('nan')
    return float(np.corrcoef(s, e)[0, 1])


def enigma_align(FC: np.ndarray) -> np.ndarray:
    """ENIGMA HCP per-subject preprocessing applied to a simulated FC matrix.

    Per Royer et al. 2022 / ENIGMA Toolbox docs:
      (i)   start from Pearson r ∈ [-1, 1]
      (ii)  clip negatives to 0
      (iii) Fisher-z transform: z = arctanh(r)
    Group-averaging across subjects (step iv) is left to the caller (e.g.
    multi-seed). For comparison against ENIGMA's group-averaged z-space FC
    (range observed [0, 1.43] on the DK68 atlas).
    """
    FC = np.maximum(FC, 0.0)
    FC = np.arctanh(np.clip(FC, 0.0, 1.0 - 1e-9))
    np.fill_diagonal(FC, 0.0)
    return FC


def r_vs_emp(BOLD_FC_raw: np.ndarray,
             emp_fc: Optional[np.ndarray]) -> Tuple[float, float]:
    """Return (r_aligned, r_raw): aligned r is the headline number for any
    comparison against ENIGMA's z-space empirical FC; raw is kept for
    diagnostic / sanity. Headline = aligned.
    """
    return fc_corr(enigma_align(BOLD_FC_raw), emp_fc), fc_corr(BOLD_FC_raw, emp_fc)


# ----------------------------------------------------------------------
# Statistical helpers
# ----------------------------------------------------------------------
def bootstrap_r_ci(simulated: np.ndarray, empirical: Optional[np.ndarray],
                   n_boot: int = 1000, alpha: float = 0.05,
                   seed: int = 42) -> Tuple[float, float, float]:
    """Bootstrap 95% CI on Pearson r between two FC matrices.

    Resamples upper-triangle edges with replacement. Note: this gives the
    sampling SE of r given THIS realisation of simulated FC. Does NOT
    account for between-seed variability (use multi-seed for that).

    Returns (r_point, ci_lo, ci_hi).
    """
    if empirical is None or simulated is None:
        return float('nan'), float('nan'), float('nan')
    triu = np.triu_indices(simulated.shape[0], k=1)
    s, e = simulated[triu], empirical[triu]
    n = len(s)
    if n < 4 or np.std(s) == 0 or np.std(e) == 0:
        return float('nan'), float('nan'), float('nan')
    r_point = float(np.corrcoef(s, e)[0, 1])
    rng = np.random.default_rng(seed)
    rs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        ss, ee = s[idx], e[idx]
        if np.std(ss) > 0 and np.std(ee) > 0:
            rs[i] = np.corrcoef(ss, ee)[0, 1]
        else:
            rs[i] = np.nan
    rs = rs[np.isfinite(rs)]
    if len(rs) < 10:
        return r_point, float('nan'), float('nan')
    lo = float(np.percentile(rs, 100 * alpha / 2))
    hi = float(np.percentile(rs, 100 * (1 - alpha / 2)))
    return r_point, lo, hi


def permutation_null_p(simulated: np.ndarray, empirical: Optional[np.ndarray],
                       n_perm: int = 1000, seed: int = 42) -> Tuple[float, float]:
    """Permutation test against the null that simulated and empirical FC
    are unrelated. Permutes node labels of empirical (preserving its
    upper-triangle marginal distribution).

    Returns (r_observed, p_one_sided): p = fraction of |r_perm| >= |r_obs|.
    """
    if empirical is None or simulated is None:
        return float('nan'), float('nan')
    N = simulated.shape[0]
    triu = np.triu_indices(N, k=1)
    s = simulated[triu]
    if np.std(s) == 0:
        return float('nan'), float('nan')
    r_obs = float(np.corrcoef(s, empirical[triu])[0, 1])
    rng = np.random.default_rng(seed)
    n_extreme = 0
    for _ in range(n_perm):
        perm = rng.permutation(N)
        emp_perm = empirical[np.ix_(perm, perm)]
        e = emp_perm[triu]
        if np.std(e) > 0:
            r_p = np.corrcoef(s, e)[0, 1]
            if abs(r_p) >= abs(r_obs):
                n_extreme += 1
    p = (n_extreme + 1) / (n_perm + 1)            # add-one for safety
    return r_obs, float(p)


def paired_seed_stats(rs_a: List[float], rs_b: List[float]) -> Dict[str, float]:
    """Paired t-test on per-seed r values from two model conditions.

    Returns mean_a, mean_b, mean_diff, se_diff, t_stat, p_two_sided, n.
    Pairs match by seed index (rs_a[i] and rs_b[i] used the same seed).
    """
    a = np.array(rs_a, dtype=float)
    b = np.array(rs_b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    n = len(a)
    out = {'n': float(n), 'mean_a': float('nan'), 'mean_b': float('nan'),
           'mean_diff': float('nan'), 'se_diff': float('nan'),
           't': float('nan'), 'p': float('nan')}
    if n < 2:
        return out
    diffs = b - a
    out['mean_a'] = float(a.mean())
    out['mean_b'] = float(b.mean())
    out['mean_diff'] = float(diffs.mean())
    out['se_diff'] = float(diffs.std(ddof=1) / np.sqrt(n))
    if out['se_diff'] > 0:
        from scipy.stats import t as t_dist
        out['t'] = out['mean_diff'] / out['se_diff']
        out['p'] = float(2 * (1 - t_dist.cdf(abs(out['t']), df=n - 1)))
    return out


def adaptive_t_start(duration: float) -> float:
    """At least 30 s, otherwise 20% of duration."""
    return max(30.0, duration * 0.2)


def make_params(distance: np.ndarray,
                distance_scale: float = OPTIMAL_DISTANCE_SCALE,
                k_ext: float = DEFAULT_K_EXT,
                kv_scale: float = 1.0,
                eta_E: Optional[float] = None,
                delay_mode: str = 'matrix',
                seed: int = 42,
                **extra) -> dict:
    """Build NextGenNetwork param dict with sensible defaults."""
    params = {
        'k_ext': k_ext,
        'delay_mode': delay_mode,
        'distance_matrix': distance * distance_scale,
        'kappa_v_EE': DEFAULT_KAPPA_V_EE * kv_scale,
        'kappa_v_II': DEFAULT_KAPPA_V_II * kv_scale,
        'seed': seed,
    }
    if eta_E is not None:
        params['eta_E'] = eta_E
    params.update(extra)
    return params


def run_model(ctx: Context, params: dict, duration: float = DEFAULT_DURATION,
              verbose: bool = True) -> NextGenNetwork:
    """Build & integrate NextGenNetwork; returns the integrated model."""
    model = NextGenNetwork(HCPNetwork(ctx.sc), params=params)
    if verbose:
        print(f"    Dim={model.n_vars}, edges={model.M}, "
              f"delay=[{model.delays.min()*1000:.1f}, "
              f"{model.delays.max()*1000:.1f}] ms")
        print(f"    Integrating ({duration} s)...", flush=True)
    t0 = time.time()
    model.run(duration=duration, dt=DT)
    if verbose:
        print(f"    Done ({time.time() - t0:.1f} s).")
    return model


def parse_float_list(s: str) -> List[float]:
    return [float(x) for x in s.split(',') if x.strip()]


def parse_str_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(',') if x.strip()]


def save_data(ctx: Context, exp_name: str, **arrays):
    """Save raw arrays as a single .npz (one per experiment)."""
    if not ctx.args.save:
        return
    npz_path = os.path.join(ctx.results_dir, f'{exp_name}.npz')
    saveable = {k: np.asarray(v) for k, v in arrays.items()
                if isinstance(v, (np.ndarray, list, float, int))}
    np.savez(npz_path, **saveable)
    print(f"  Saved data: {npz_path}")


def save_fig(ctx: Context, name: str, fig):
    """Save a single-panel figure as `<name>.png` (no title, no annotation
    embedded — captions are authored in the report)."""
    if not ctx.args.save:
        plt.close(fig); return
    png_path = os.path.join(ctx.plots_dir, f'{name}.png')
    fig.savefig(png_path)                         # rcParams handles dpi/bbox
    print(f"  Saved figure: {png_path}")
    plt.close(fig)


def plot_fc_matrix(fc: np.ndarray, vmax: float, hemisphere_split: int = 34,
                   cbar_label: str = 'FC', signed: bool = True,
                   axis_label: str = 'Region (DK68 index)'):
    """Single-panel matrix heatmap (FC / SC / PLV / etc.). Returns (fig, ax).

    Convention (matches Forrester 2024 Fig 8-10):
      - diverging colormap (RdBu_r) centred at 0 for SIGNED quantities (FC)
      - sequential colormap (viridis) on [0, vmax] for non-negative quantities
        (SC, PLV — both ≥ 0 by construction)
      - thin black line separating L (idx 0..33) and R (idx 34..67) for HCP
      - X/Y labelled per `axis_label` (default DK68; pass 'Node index' for
        toy topologies); colorbar takes a quantity-specific label
    """
    fig, ax = plt.subplots(figsize=(5.5, 5))
    if signed:
        im = ax.imshow(fc, cmap=FC_CMAP, vmin=-vmax, vmax=vmax,
                       interpolation='nearest')
    else:
        im = ax.imshow(fc, cmap='viridis', vmin=0.0, vmax=vmax,
                       interpolation='nearest')
    if hemisphere_split is not None:
        ax.axhline(hemisphere_split - 0.5, color='black', lw=0.6)
        ax.axvline(hemisphere_split - 0.5, color='black', lw=0.6)
    ax.set_xlabel(axis_label)
    ax.set_ylabel(axis_label)
    ax.grid(False)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=cbar_label)
    return fig, ax




# ====================================================================
# Experiments
# ====================================================================

# Scale sweep
def exp_scale(ctx: Context):
    """Distance scale sweep — find optimal s for BOLD FC fit.

    Sweep s ∈ args.scales.
    Long simulation (default 500 s) is required for stable BOLD FC.
    """
    print("\n" + "=" * 70)
    print("  EXP 1: Distance scale sweep (multi-seed)")
    print("=" * 70)
    scales = ctx.args.scales or [1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
    duration = ctx.args.duration if ctx.args.duration_set_by_user else SCALE_DURATION
    t_start = adaptive_t_start(duration)
    n_seeds = SCALE_N_SEEDS
    seeds = list(range(42, 42 + n_seeds))
    # Use stochastic delay (matrix_gamma_velocity, mean-matched to v=12 m/s)
    # so different seeds produce statistically independent realisations.
    # Pure 'matrix' mode is deterministic — multi-seed would be wasted.

    per_seed_FCs = {}    # (s, seed) → BOLD_FC matrix
    rows = []            # one per (s, seed)
    for s in scales:
        for seed in seeds:
            print(f"\n  scale = {s}, seed = {seed}:")
            params = make_params(ctx.distance, distance_scale=s, seed=seed,
                                 delay_mode='matrix_gamma_velocity',
                                 conduction_velocity=15.0)
            m = run_model(ctx, params, duration=duration)
            Z_E, _ = m.compute_Z()
            half = len(m.time_array) // 2
            mean_Z = float(np.abs(Z_E[:, half:]).mean())
            BOLD_FC = m.compute_BOLD_FC(t_start=t_start)
            BOLD_FC_z = enigma_align(BOLD_FC)
            r_aligned = fc_corr(BOLD_FC_z, ctx.emp_fc)
            r_raw = fc_corr(BOLD_FC, ctx.emp_fc)
            print(f"    mean |Z_E|={mean_Z:.4f}  r_aligned={r_aligned:+.4f}  "
                  f"r_raw={r_raw:+.4f}")
            rows.append({'scale': s, 'seed': seed, 'mean_Z': mean_Z,
                         'r_bold': r_aligned, 'r_bold_raw': r_raw,
                         'delay_ms_lo': float(m.delays.min() * 1000),
                         'delay_ms_hi': float(m.delays.max() * 1000)})
            per_seed_FCs[(s, seed)] = BOLD_FC

    # --- Per-scale summary: mean ± SE across seeds, plus single-seed bootstrap CI
    # on the seed-averaged z-space FC matrix (more conservative).
    summary = []
    BOLD_FCs_avg = []
    for s in scales:
        rs = [r['r_bold'] for r in rows if r['scale'] == s]
        zs = [r['mean_Z'] for r in rows if r['scale'] == s]
        # seed-averaged FC (z-space) per scale
        avg_FC_z = np.mean(np.stack([enigma_align(per_seed_FCs[(s, sd)].copy())
                                      for sd in seeds]), axis=0)
        BOLD_FCs_avg.append(avg_FC_z)
        r_avg, lo, hi = bootstrap_r_ci(avg_FC_z, ctx.emp_fc, n_boot=1000)
        r_arr = np.array(rs)
        summary.append({
            'scale': s,
            'r_mean': float(r_arr.mean()),
            'r_se':   float(r_arr.std(ddof=1) / np.sqrt(len(r_arr))) if len(r_arr) > 1 else float('nan'),
            'r_avg_aligned': r_avg,
            'r_avg_lo': lo, 'r_avg_hi': hi,
            'mean_Z': float(np.mean(zs)),
        })

    print("\n  Summary (n = %d seeds):" % n_seeds)
    print(f"  {'s':>5}  {'r mean±SE':>15}  {'r(avg-FC)':>10}  {'95% CI':>20}  {'|Z|':>7}")
    for sm in summary:
        print(f"  {sm['scale']:>5.2f}  "
              f"{sm['r_mean']:+.4f}±{sm['r_se']:.4f}  "
              f"{sm['r_avg_aligned']:>+10.4f}  "
              f"[{sm['r_avg_lo']:+.3f},{sm['r_avg_hi']:+.3f}]  "
              f"{sm['mean_Z']:>7.4f}")
    best_idx = int(np.argmax([sm['r_mean'] for sm in summary]))
    print(f"\n  >>> BEST scale (by mean r): s = {summary[best_idx]['scale']}  "
          f"(mean r = {summary[best_idx]['r_mean']:+.4f}±{summary[best_idx]['r_se']:.4f})")

    s_arr = np.array([r['scale'] for r in rows])
    seed_arr = np.array([r['seed'] for r in rows])
    r_arr = np.array([r['r_bold'] for r in rows])

    # --- Save raw data ---
    save_data(ctx, 'exp_scale',
              scales=s_arr, seeds=seed_arr,
              r_bold=r_arr,
              r_bold_raw=np.array([r['r_bold_raw'] for r in rows]),
              mean_Z=np.array([r['mean_Z'] for r in rows]),
              # per-scale summary
              summary_scales=np.array([sm['scale'] for sm in summary]),
              r_mean=np.array([sm['r_mean'] for sm in summary]),
              r_se=np.array([sm['r_se'] for sm in summary]),
              r_avg_aligned=np.array([sm['r_avg_aligned'] for sm in summary]),
              r_avg_lo=np.array([sm['r_avg_lo'] for sm in summary]),
              r_avg_hi=np.array([sm['r_avg_hi'] for sm in summary]),
              # seed-averaged FC matrices per scale
              BOLD_FCs_avg=np.stack(BOLD_FCs_avg))

    # No figures — exp_scale is text-only in the report. The methods section
    # quotes "r ranged X to Y across s ∈ [1.2, 1.7]; bootstrap CIs overlap;
    # we adopt s = 1.5 (Forrester default) within the working band". Visual
    # FC vs empirical comparison is in exp_scfc (3-panel SC/BOLD/empirical).
    # All numerical results (per-s r, CIs, mean |Z|, full BOLD FC matrices)
    # are saved in exp_scale.npz for reference.


def exp_topology(ctx: Context):
    """Next Gen on Conti & Van Gorder's 4 idealized topologies (N=16).

    Bridge to Dean's WC track: same testbed (path / cycle / complete /
    lattice), different model. Tests the original-plan claim that
    "topology affects population coherence". Uses uniform inter-node
    distance (no DTI here — toy graphs have no anatomical embedding) so
    the only effect is graph topology.

    Statistical design: 4 topologies × N=TOPOLOGY_N_SEEDS seeds × N=16
    nodes × TOPOLOGY_DURATION s. Per-seed metrics: mean |Z|, mean PLV.
    One-way ANOVA across topologies for each metric.
    """
    print("\n" + "=" * 70)
    print("  EXP TOPOLOGY: Next Gen on path/cycle/complete/lattice (N=16)")
    print("=" * 70)
    from network import Network         # src/network.py (Dean's WC topology gen)

    seeds = list(range(42, 42 + TOPOLOGY_N_SEEDS))
    topologies = ['line', 'ring', 'full', 'lattice']         # path/cycle/complete/lattice
    pretty_names = {'line': 'path', 'ring': 'cycle',
                    'full': 'complete', 'lattice': 'lattice'}

    # Heterogeneous random per-edge distances on toy graphs.
    # NOTE: Conti & Van Gorder use a single constant delay on toy graphs.
    # Repeating that here gives identical results for ALL topologies
    # (Next Gen self-locks regardless of topology when delay is uniform —
    # the same over-lock pathology demonstrated in exp_delays). To expose
    # the genuine topology effect on |Z|/PLV we use heterogeneous random
    # distances drawn from a HCP-realistic uniform [20, 140] mm range.
    N = TOPOLOGY_N_NODES
    rng_dist = np.random.default_rng(0)
    toy_dist = rng_dist.uniform(20.0, 140.0, size=(N, N))
    toy_dist = (toy_dist + toy_dist.T) / 2.0      # symmetrize
    np.fill_diagonal(toy_dist, 0.0)

    rows = []
    PLV_per_topology = {}
    SC_per_topology  = {}
    for topo in topologies:
        cfg = {'N': N, 'topology': topo, 'p': 0.125 if topo == 'ring' else None}
        net = Network({k: v for k, v in cfg.items() if v is not None})
        SC_per_topology[topo] = net.A.copy()
        for seed in seeds:
            print(f"\n  topology={pretty_names[topo]}, seed={seed}:")
            # matrix_gamma_velocity injects per-seed stochasticity (Atay-Hutt
            # truncated Gamma per-edge velocities), so multi-seed actually
            # produces independent realisations. Without stochastic delays
            # the deterministic DDE gives identical output for every seed.
            params = make_params(toy_dist, seed=seed,
                                 delay_mode='matrix_gamma_velocity',
                                 conduction_velocity=15.0)  # mean-matched
            ctx_local = Context(sc=net.A, emp_fc=None, labels=None,
                                distance=toy_dist, args=ctx.args,
                                plots_dir=ctx.plots_dir, results_dir=ctx.results_dir)
            m = run_model(ctx_local, params, duration=TOPOLOGY_DURATION,
                          verbose=False)
            Z_E, _ = m.compute_Z()
            half = len(m.time_array) // 2
            mean_Z = float(np.abs(Z_E[:, half:]).mean())
            t_start = adaptive_t_start(TOPOLOGY_DURATION)
            PLV = m.compute_PLV(t_start=t_start)
            mean_PLV = float(PLV[np.triu_indices(N, k=1)].mean())
            print(f"    |Z|={mean_Z:.4f}  PLV={mean_PLV:.4f}")
            rows.append({'topology': topo, 'seed': seed,
                         'mean_Z': mean_Z, 'mean_PLV': mean_PLV})
            if seed == seeds[-1]:
                PLV_per_topology[topo] = PLV

    # --- ANOVA across topologies ---
    from scipy.stats import f_oneway
    z_groups   = [[r['mean_Z']  for r in rows if r['topology']==t] for t in topologies]
    plv_groups = [[r['mean_PLV'] for r in rows if r['topology']==t] for t in topologies]
    F_z, p_z     = f_oneway(*z_groups)
    F_plv, p_plv = f_oneway(*plv_groups)

    print(f"\n  --- One-way ANOVA across topologies (n={len(seeds)} seeds each) ---")
    print(f"  mean |Z|: F = {F_z:.2f}, p = {p_z:.4f}")
    print(f"  mean PLV: F = {F_plv:.2f}, p = {p_plv:.4f}")
    print(f"  Per-topology means:")
    for t in topologies:
        z_arr = np.array([r['mean_Z'] for r in rows if r['topology']==t])
        plv_arr = np.array([r['mean_PLV'] for r in rows if r['topology']==t])
        print(f"    {pretty_names[t]:<10} |Z|={z_arr.mean():.4f}±{z_arr.std(ddof=1)/np.sqrt(len(z_arr)):.4f}   "
              f"PLV={plv_arr.mean():.4f}±{plv_arr.std(ddof=1)/np.sqrt(len(plv_arr)):.4f}")

    save_data(ctx, 'exp_topology',
              topologies=np.array([r['topology'] for r in rows]),
              seeds=np.array([r['seed'] for r in rows]),
              mean_Z=np.array([r['mean_Z'] for r in rows]),
              mean_PLV=np.array([r['mean_PLV'] for r in rows]),
              F_Z=F_z, p_Z=p_z, F_PLV=F_plv, p_PLV=p_plv,
              **{f'PLV_{t}': PLV_per_topology[t] for t in topologies},
              **{f'SC_{t}':  SC_per_topology[t]  for t in topologies})

    # --- Plots ---
    # Only the PLV-by-topology dot plot enters the report. |Z| was insensitive
    # to topology (range ~0.94-0.95), reported as one sentence in text.
    # Per-topology PLV matrices and adjacency heatmaps were dropped — toy graph
    # shapes are common knowledge; pairwise PLV info is summarised by the dot
    # plot's mean ± SE.
    pretty_x = [pretty_names[t] for t in topologies]
    positions = np.arange(len(topologies))
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, t in enumerate(topologies):
        plv_vals = [r['mean_PLV'] for r in rows if r['topology']==t]
        ax.scatter([i] * len(plv_vals), plv_vals,
                   color=COLOR_SECONDARY, alpha=0.6, s=40)
        ax.errorbar(i, np.mean(plv_vals),
                    yerr=np.std(plv_vals, ddof=1)/np.sqrt(len(plv_vals)),
                    fmt='_', color='black', markersize=20, capsize=5, lw=2)
    ax.set_xticks(positions); ax.set_xticklabels(pretty_x)
    ax.set_ylabel('Mean PLV')
    save_fig(ctx, 'exp_topology__PLV_by_topology', fig)


def exp_gap(ctx: Context):
    """Gap-junction effect — binary {off, on} × N seeds with paired t-test.

    Forrester paper's main GJ claim is "removing GJ degrades model fit".
    Binary comparison + multi-seed gives clean paired statistics for that
    one well-defined claim. (Earlier 7-point sweep was single-seed and
    statistically incapable of distinguishing intermediate κ_v values;
    see audit notes in CLAUDE.md.)
    """
    print("\n" + "=" * 70)
    print("  EXP 2: Gap-junction effect (binary, multi-seed)")
    print("=" * 70)
    duration = ctx.args.duration if ctx.args.duration_set_by_user else BOLD_FC_DURATION
    t_start = adaptive_t_start(duration)
    seeds = list(range(42, 42 + GAP_N_SEEDS))   # 5 seeds default
    conditions = [('off', 0.0), ('on', 1.0)]

    rows = []           # one per (condition, seed)
    BOLD_FCs = {}       # last-seed FC matrix per condition (for plots)
    PLVs = {}
    for cond_name, kv in conditions:
        for seed in seeds:
            print(f"\n  cond={cond_name} (κ_v={kv}× default), seed={seed}:")
            params = make_params(ctx.distance, kv_scale=kv, seed=seed)
            m = run_model(ctx, params, duration=duration)
            Z_E, _ = m.compute_Z()
            half = len(m.time_array) // 2
            mean_Z = float(np.abs(Z_E[:, half:]).mean())
            PLV = m.compute_PLV(t_start=t_start)
            BOLD_FC = m.compute_BOLD_FC(t_start=t_start)
            mean_PLV = float(PLV[np.triu_indices(m.N, k=1)].mean())
            r_aligned, r_raw = r_vs_emp(BOLD_FC, ctx.emp_fc)
            # Dynamic FC (sliding-window PLV) — merged from former exp_dynamic_fc.
            # 10 s window, 90% overlap (matches Forrester Fig 7); for 500 s sim
            # gives 491 windows (effective n_eff << 491 due to overlap).
            dPLV, _ = m.compute_dynamic_PLV(window_sec=10.0, overlap=0.9,
                                             t_start=t_start)
            triu_idx = np.triu_indices(m.N, k=1)
            edge_ts = dPLV[:, triu_idx[0], triu_idx[1]]
            dPLV_sigma = float(edge_ts.std(axis=0).mean())  # mean per-edge temporal SD
            print(f"    |Z|={mean_Z:.4f}  PLV={mean_PLV:.3f}  "
                  f"r_aligned={r_aligned:.4f}  dPLV σ={dPLV_sigma:.4f}")
            rows.append({'cond': cond_name, 'kv_scale': kv, 'seed': seed,
                         'mean_Z': mean_Z, 'mean_PLV': mean_PLV,
                         'r_bold': r_aligned, 'r_bold_raw': r_raw,
                         'dPLV_sigma': dPLV_sigma})
            BOLD_FCs[(cond_name, seed)] = BOLD_FC
            PLVs[(cond_name, seed)] = PLV

    # --- Paired statistics — 4 metrics, Bonferroni α/4 = 0.0125 ---
    def by_cond(field):
        return ([r[field] for r in rows if r['cond'] == 'off'],
                [r[field] for r in rows if r['cond'] == 'on'])
    rs_off, rs_on   = by_cond('r_bold')
    z_off,  z_on    = by_cond('mean_Z')
    plv_off, plv_on = by_cond('mean_PLV')
    dplv_off, dplv_on = by_cond('dPLV_sigma')
    stat_r   = paired_seed_stats(rs_off,  rs_on)
    stat_z   = paired_seed_stats(z_off,   z_on)
    stat_plv = paired_seed_stats(plv_off, plv_on)
    stat_dplv = paired_seed_stats(dplv_off, dplv_on)

    print(f"\n  --- Paired t-test (n={len(seeds)} seeds, Bonferroni α = 0.0125) ---")
    for name, st, off, on in [
            ('r_aligned (BOLD,emp)', stat_r,   rs_off,   rs_on),
            ('mean |Z|            ', stat_z,   z_off,    z_on),
            ('mean PLV            ', stat_plv, plv_off,  plv_on),
            ('dPLV σ (sliding PLV) ', stat_dplv, dplv_off,  dplv_on),
    ]:
        print(f"  {name}: off={st['mean_a']:+.5f}  on={st['mean_b']:+.5f}  "
              f"Δ={st['mean_diff']:+.5f}±{st['se_diff']:.5f}  "
              f"t={st['t']:+.2f}  p={st['p']:.4f}  "
              f"{'**' if st['p'] < 0.0125 else ' '}")

    save_data(ctx, 'exp_gap',
              conds=np.array([r['cond'] for r in rows]),
              seeds=np.array([r['seed'] for r in rows]),
              kv_scales=np.array([r['kv_scale'] for r in rows]),
              r_bold=np.array([r['r_bold'] for r in rows]),
              r_bold_raw=np.array([r['r_bold_raw'] for r in rows]),
              mean_Z=np.array([r['mean_Z'] for r in rows]),
              mean_PLV=np.array([r['mean_PLV'] for r in rows]),
              dPLV_sigma=np.array([r['dPLV_sigma'] for r in rows]),
              # paired stats (4 metrics)
              t_r=stat_r['t'],     p_r=stat_r['p'],     diff_r=stat_r['mean_diff'],     se_r=stat_r['se_diff'],
              t_Z=stat_z['t'],     p_Z=stat_z['p'],     diff_Z=stat_z['mean_diff'],     se_Z=stat_z['se_diff'],
              t_PLV=stat_plv['t'], p_PLV=stat_plv['p'], diff_PLV=stat_plv['mean_diff'], se_PLV=stat_plv['se_diff'],
              t_dPLV=stat_dplv['t'], p_dPLV=stat_dplv['p'], diff_dPLV=stat_dplv['mean_diff'], se_dPLV=stat_dplv['se_diff'],
              # last-seed FC for plotting
              BOLD_FC_off=BOLD_FCs[('off', seeds[-1])],
              BOLD_FC_on=BOLD_FCs[('on',  seeds[-1])])

    # ============== Plots ==============
    # r vs emp metric → text-only in report (paired t/p saved in npz).

    # (1) Static vs dynamic synchrony — 2-panel side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    # Panel A: static synchrony (mean PLV across full window)
    se_plv_off = np.std(plv_off, ddof=1) / np.sqrt(len(plv_off))
    se_plv_on  = np.std(plv_on,  ddof=1) / np.sqrt(len(plv_on))
    for seed in seeds:
        p_o = [r['mean_PLV'] for r in rows if r['cond']=='off' and r['seed']==seed][0]
        p_n = [r['mean_PLV'] for r in rows if r['cond']=='on'  and r['seed']==seed][0]
        axes[0].plot([0, 1], [p_o, p_n], 'o-', color=COLOR_GREY, lw=0.8, ms=4, alpha=0.5)
    axes[0].errorbar([0, 1], [stat_plv['mean_a'], stat_plv['mean_b']],
                     yerr=[se_plv_off, se_plv_on], fmt='s-',
                     color=COLOR_PRIMARY, lw=2, ms=10, capsize=5)
    axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(['GJ off', 'GJ on'])
    axes[0].set_ylabel('Mean PLV  (static, across full window)')
    # Panel B: dynamic variability (dPLV σ)
    se_dplv_off = np.std(dplv_off, ddof=1) / np.sqrt(len(dplv_off))
    se_dplv_on  = np.std(dplv_on,  ddof=1) / np.sqrt(len(dplv_on))
    for seed in seeds:
        d_o = [r['dPLV_sigma'] for r in rows if r['cond']=='off' and r['seed']==seed][0]
        d_n = [r['dPLV_sigma'] for r in rows if r['cond']=='on'  and r['seed']==seed][0]
        axes[1].plot([0, 1], [d_o, d_n], 'o-', color=COLOR_GREY, lw=0.8, ms=4, alpha=0.5)
    axes[1].errorbar([0, 1], [stat_dplv['mean_a'], stat_dplv['mean_b']],
                     yerr=[se_dplv_off, se_dplv_on], fmt='s-',
                     color=COLOR_SECONDARY, lw=2, ms=10, capsize=5)
    axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(['GJ off', 'GJ on'])
    axes[1].set_ylabel(r'dPLV $\sigma$  (dynamic, sliding-window SD)')
    fig.tight_layout()
    save_fig(ctx, 'exp_gap__paired_sync', fig)
    # |Z| dropped — insensitive to GJ at our N (Δ < SE), reported in text only.

    # (2) BOLD FC heatmap pair off vs on (2-panel, aligned z-space, same vmax)
    BOLD_off = BOLD_FCs[('off', seeds[-1])]
    BOLD_on  = BOLD_FCs[('on',  seeds[-1])]
    BOLD_off_z = enigma_align(BOLD_off.copy())
    BOLD_on_z  = enigma_align(BOLD_on.copy())
    abs_max = max(np.nanmax(BOLD_off_z), np.nanmax(BOLD_on_z))
    if ctx.emp_fc is not None:
        abs_max = max(abs_max, float(np.nanmax(ctx.emp_fc)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax_i, (FC, name) in zip(axes, [(BOLD_off_z, 'GJ off (κ_v=0)'),
                                        (BOLD_on_z,  'GJ on (κ_v=default)')]):
        im = ax_i.imshow(FC, cmap='viridis', vmin=0, vmax=abs_max,
                         interpolation='nearest')
        ax_i.axhline(33.5, color='black', lw=0.5)
        ax_i.axvline(33.5, color='black', lw=0.5)
        ax_i.set_xlabel(f'Region (DK68 index) — {name}')
        ax_i.set_ylabel('Region (DK68 index)')
    fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04,
                 label='BOLD FC (Fisher-z, aligned)')
    save_fig(ctx, 'exp_gap__fc_pair', fig)


def exp_ng_capabilities(ctx: Context):
    """NG-only capability demonstration on HCP-68 (renamed from
    exp_wc_vs_nextgen).

    Demonstrates an observable that Wilson–Cowan fundamentally cannot
    produce: within-population synchrony |Z_E|(t). The Pearson-FC horse-race
    against WC was dropped: (i) WC at default Conti params saturates on the
    HCP connectome (uninformative); (ii) comparing two model FCs at neural
    rate against fMRI BOLD empirical FC is apples-to-oranges in timescale;
    (iii) NG has many more parameters than WC, so beating WC on FC fit is
    not a meaningful claim.

    Net narrative: replacing WC with NG buys us |Z|, GJ effects, and BOLD
    synthesis — observables that map onto MEG / EEG / fMRI experiments. The
    Z_band figure visualises one such observable.
    """
    print("\n" + "=" * 70)
    print("  EXP 3: Next Gen capability demonstration (|Z_E|)")
    print("=" * 70)
    duration = ctx.args.duration if ctx.args.duration_set_by_user else BOLD_FC_DURATION

    print("\n  Next Gen (HCP-68, scale=1.5, default params):")
    params = make_params(ctx.distance)
    ngm = run_model(ctx, params, duration=duration)
    R_E_ng = ngm.get_all_RE()
    Z_E_ng, _ = ngm.compute_Z()
    print(f"    R_E range: [{R_E_ng.min():.2f}, {R_E_ng.max():.2f}]   "
          f"mean |Z_E|: {np.abs(Z_E_ng).mean():.3f}")

    Z_abs = np.abs(Z_E_ng)
    save_data(ctx, 'exp_ng_capabilities',
              R_E_ng=R_E_ng[:, ::100],
              Z_E_ng_abs=Z_abs[:, ::100])

    # Single figure: |Z_E| distribution band across 68 nodes vs time
    t = ngm.time_array
    Z_med = np.median(Z_abs, axis=0)
    Z_p05 = np.percentile(Z_abs,  5, axis=0)
    Z_p95 = np.percentile(Z_abs, 95, axis=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(t, Z_p05, Z_p95, color=COLOR_PRIMARY, alpha=0.25,
                    label='5–95 percentile across 68 nodes')
    ax.plot(t, Z_med, color=COLOR_PRIMARY, lw=1.5, label='median across nodes')
    ax.set_xlabel('Time (s)'); ax.set_ylabel(r'$|Z_E|$')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right', frameon=False, fontsize=9)
    save_fig(ctx, 'exp_ng_capabilities__Z_band', fig)


def _steiger_z_correlated_r(FC_a: np.ndarray, FC_b: np.ndarray,
                             emp: np.ndarray) -> Tuple[float, float]:
    """Steiger (1980) Z test for two correlated correlations sharing one
    sample (the empirical FC).  Tests H0: r(a, emp) = r(b, emp).
    Returns (z, p_two_sided).  Uses upper-triangle off-diagonals.
    """
    triu = np.triu_indices(emp.shape[0], k=1)
    a, b, e = FC_a[triu], FC_b[triu], emp[triu]
    n = len(e)
    if n < 4:
        return float('nan'), float('nan')
    r12 = float(np.corrcoef(a, e)[0, 1])    # a vs emp
    r13 = float(np.corrcoef(b, e)[0, 1])    # b vs emp
    r23 = float(np.corrcoef(a, b)[0, 1])    # a vs b (the correlation between the two)
    # Steiger 1980 t-statistic (Williams variant for one sample)
    rm  = (r12 + r13) / 2.0
    det = (1.0 - r12*r12 - r13*r13 - r23*r23 + 2.0*r12*r13*r23)
    if det <= 0 or n <= 3:
        return float('nan'), float('nan')
    t_stat = (r12 - r13) * np.sqrt((n - 1) * (1 + r23)
                                   / (2 * ((n - 1) / (n - 3)) * det
                                      + rm*rm * (1 - r23)**3))
    from scipy.stats import t as t_dist
    p = 2.0 * (1.0 - t_dist.cdf(abs(t_stat), df=n - 3))
    return float(t_stat), float(p)


def exp_scfc(ctx: Context):
    """SC vs simulated BOLD FC vs empirical FC, multi-seed.

    Reports per-seed mean ± SE for each r, plus permutation-null p-value
    against the null that simulated and empirical are unrelated. Also
    reports r restricted to connected edges (SC > 0) to avoid the inflation
    from SC's 70% structural zeros.
    """
    print("\n" + "=" * 70)
    print("  EXP 5: Structure–function analysis (multi-seed)")
    print("=" * 70)
    duration = ctx.args.duration if ctx.args.duration_set_by_user else BOLD_FC_DURATION
    t_start = adaptive_t_start(duration)
    seeds = list(range(42, 42 + SCFC_N_SEEDS))

    rs_bold_emp = []; rs_bold_raw = []; rs_plv_emp = []; rs_sc_bold = []
    rs_bold_emp_conn = []
    BOLD_FC_z_avg = None; PLV_avg = None     # z-space averages
    SC = None
    triu = None
    sc_pos_mask = None
    for seed in seeds:
        print(f"\n  seed = {seed}:")
        params = make_params(ctx.distance, seed=seed)
        m = run_model(ctx, params, duration=duration)
        if SC is None:
            SC = m.W
            triu = np.triu_indices(SC.shape[0], k=1)
            sc_pos_mask = (SC > 0)[triu]
        PLV     = m.compute_PLV(t_start=t_start)
        BOLD_FC = m.compute_BOLD_FC(t_start=t_start)
        BOLD_FC_z = enigma_align(BOLD_FC)
        PLV_z     = enigma_align(PLV)
        # accumulate z-space averages
        BOLD_FC_z_avg = BOLD_FC_z if BOLD_FC_z_avg is None else BOLD_FC_z_avg + BOLD_FC_z
        PLV_avg       = PLV_z     if PLV_avg       is None else PLV_avg       + PLV_z
        # per-seed r
        r_be   = fc_corr(BOLD_FC_z, ctx.emp_fc)
        r_br   = fc_corr(BOLD_FC,   ctx.emp_fc)
        r_pe   = fc_corr(PLV_z,     ctx.emp_fc)
        r_sb   = fc_corr(SC,        BOLD_FC_z)
        if ctx.emp_fc is not None and sc_pos_mask.sum() >= 3:
            boldz_od = BOLD_FC_z[triu][sc_pos_mask]
            emp_od   = ctx.emp_fc[triu][sc_pos_mask]
            r_be_conn = float(np.corrcoef(boldz_od, emp_od)[0, 1])
        else:
            r_be_conn = float('nan')
        rs_bold_emp.append(r_be); rs_bold_raw.append(r_br); rs_plv_emp.append(r_pe)
        rs_sc_bold.append(r_sb); rs_bold_emp_conn.append(r_be_conn)
        print(f"    r_aligned(BOLD,emp)={r_be:+.4f}  raw={r_br:+.4f}  "
              f"PLV-emp={r_pe:+.4f}  SC-BOLD={r_sb:+.4f}  conn-only={r_be_conn:+.4f}")
    BOLD_FC_z_avg /= len(seeds)
    PLV_avg       /= len(seeds)

    # SC vs empirical (deterministic, no seed dependence)
    r_sc_emp = fc_corr(SC, ctx.emp_fc)
    if ctx.emp_fc is not None and sc_pos_mask.sum() >= 3:
        sc_od    = SC[triu][sc_pos_mask]
        emp_od_c = ctx.emp_fc[triu][sc_pos_mask]
        r_sc_emp_conn = float(np.corrcoef(sc_od, emp_od_c)[0, 1])
    else:
        r_sc_emp_conn = float('nan')

    def m_se(xs):
        a = np.array(xs)
        a = a[np.isfinite(a)]
        if len(a) < 2:
            return float(a.mean()) if len(a) else float('nan'), float('nan')
        return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a)))

    be_m, be_s = m_se(rs_bold_emp)
    br_m, br_s = m_se(rs_bold_raw)
    pe_m, pe_s = m_se(rs_plv_emp)
    sb_m, sb_s = m_se(rs_sc_bold)
    bec_m, bec_s = m_se(rs_bold_emp_conn)

    # Permutation null on the seed-averaged BOLD_FC_z
    _, p_perm_be = permutation_null_p(BOLD_FC_z_avg, ctx.emp_fc, n_perm=500)

    print(f"\n  --- Multi-seed mean ± SE (n={len(seeds)} seeds) ---")
    print(f"  All edges (n=2278):")
    print(f"    r(SC, emp FC)               = {r_sc_emp:.4f}     [deterministic]")
    print(f"    r(BOLD aligned, emp FC)     = {be_m:+.4f} ± {be_s:.4f}   "
          f"perm-null p = {p_perm_be:.3f}  [headline]")
    print(f"    r(BOLD raw,     emp FC)     = {br_m:+.4f} ± {br_s:.4f}   [pre-alignment]")
    print(f"    r(PLV aligned,  emp FC)     = {pe_m:+.4f} ± {pe_s:.4f}   [neural-timescale]")
    print(f"    r(SC, BOLD aligned)         = {sb_m:+.4f} ± {sb_s:.4f}")
    print(f"  Connected edges only (SC>0, n={int(sc_pos_mask.sum())}):")
    print(f"    r(SC, emp FC)               = {r_sc_emp_conn:.4f}     [SC has signal here?]")
    print(f"    r(BOLD aligned, emp FC)     = {bec_m:+.4f} ± {bec_s:.4f}   [model added value]")

    # (save_data deferred until after Steiger Z + bootstrap CI computed below.)
    # Use seed-averaged z-space matrices for all subsequent plots.
    BOLD_FC = BOLD_FC_z_avg
    PLV     = PLV_avg
    r_bold_emp = be_m; r_bold_raw = br_m; r_plv_emp = pe_m

    # ============== Plots: 3-panel SC vs BOLD vs empirical FC ==============
    # All on same vmax, sequential viridis. r_bars dropped — text reports
    # r(SC,emp), r(BOLD,emp), Steiger Z, perm-null p (saved in npz).
    if ctx.emp_fc is not None:
        # Bootstrap CI + Steiger Z + perm-null computed once for npz
        r_sc_pt, sc_lo, sc_hi = bootstrap_r_ci(SC, ctx.emp_fc, n_boot=1000)
        r_be_pt, be_lo, be_hi = bootstrap_r_ci(BOLD_FC, ctx.emp_fc, n_boot=1000)
        r_pe_pt, pe_lo, pe_hi = bootstrap_r_ci(PLV,     ctx.emp_fc, n_boot=1000)
        z_sb, p_sb = _steiger_z_correlated_r(SC, BOLD_FC, ctx.emp_fc)
        _, p_perm  = permutation_null_p(BOLD_FC, ctx.emp_fc, n_perm=500)

        # r(SC, seed-averaged PLV) — directly answers "does physical wiring
        # predict which regions oscillate together?" PLV is the neural-rate
        # phase-locking metric (arg(Z_E) coherence), SC is the wiring.
        # Positive r ⇒ anatomically connected pairs are more strongly
        # phase-locked. Computed on z-space averaged PLV (same as PLV_avg).
        r_sc_plv = fc_corr(SC, PLV_avg)
        print(f"\n  Structure–phase-locking: r(SC, PLV) = {r_sc_plv:+.4f}")
        print("    [direct answer to 'does physical wiring predict which "
              "regions oscillate together?']")

        # 3-panel SC / BOLD / empirical with shared colorbar
        # (Per-matrix vmax differ slightly: SC ≤ 1, BOLD/emp in z-space.
        # Use per-panel vmax to make patterns visible.)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        panels = [
            (SC,         'SC weight\n(row-normalised)', COLOR_GREY),
            (BOLD_FC,    'Simulated BOLD FC\n(Fisher-z, aligned)', COLOR_PRIMARY),
            (ctx.emp_fc, 'Empirical FC\n(Fisher-z, ENIGMA)', COLOR_SECONDARY),
        ]
        for ax_i, (M, title, _) in zip(axes, panels):
            v = float(np.nanmax(M))
            im = ax_i.imshow(M, cmap='viridis', vmin=0, vmax=v,
                             interpolation='nearest')
            ax_i.axhline(33.5, color='black', lw=0.5)
            ax_i.axvline(33.5, color='black', lw=0.5)
            ax_i.set_xlabel(f'Region (DK68 index)\n{title}')
            ax_i.set_ylabel('Region (DK68 index)' if ax_i is axes[0] else '')
            fig.colorbar(im, ax=ax_i, fraction=0.046, pad=0.02)
        save_fig(ctx, 'exp_scfc__sc_bold_emp', fig)

        # Save test results
        save_data(ctx, 'exp_scfc',
                  SC=SC, BOLD_FC_z_avg=BOLD_FC_z_avg, PLV_z_avg=PLV_avg,
                  seeds=np.array(seeds),
                  rs_bold_emp=np.array(rs_bold_emp),
                  rs_bold_raw=np.array(rs_bold_raw),
                  rs_plv_emp=np.array(rs_plv_emp),
                  rs_sc_bold=np.array(rs_sc_bold),
                  rs_bold_emp_conn=np.array(rs_bold_emp_conn),
                  r_sc_emp=r_sc_emp, r_sc_emp_conn=r_sc_emp_conn,
                  n_connected_edges=int(sc_pos_mask.sum()),
                  p_perm_bold_emp=p_perm,
                  z_steiger_sc_vs_bold=z_sb, p_steiger_sc_vs_bold=p_sb,
                  ci_sc=(sc_lo, sc_hi), ci_bold=(be_lo, be_hi), ci_plv=(pe_lo, pe_hi),
                  r_sc_plv=r_sc_plv)


def exp_delays(ctx: Context):
    """Delay-mode binary comparison: constant vs matrix (heterogeneous), N seeds.

    The well-defined claim is "constant-mean delay over-locks the network
    (PLV → 1) and degrades FC; heterogeneous per-edge matrix delay
    restores differential phase relationships." Binary + multi-seed
    paired stats. Atay-Hutt Gamma-velocity is implemented but excluded
    from this primary comparison because at single-seed it was
    statistically indistinguishable from matrix on r vs emp (drop avoids
    over-claiming on a confounded comparison).
    """
    print("\n" + "=" * 70)
    print("  EXP 8: Delay-mode comparison (constant vs matrix, multi-seed)")
    print("=" * 70)
    duration = ctx.args.duration if ctx.args.duration_set_by_user else BOLD_FC_DURATION
    t_start = adaptive_t_start(duration)
    seeds = list(range(42, 42 + DELAY_N_SEEDS))

    mean_delay_s = (ctx.distance[ctx.distance > 0].mean()
                    * OPTIMAL_DISTANCE_SCALE / 12_000.0)

    rows = []
    BOLD_FCs = {}; PLV_mats = {}; RE_zooms = {}; delays_seen = {}
    modes = ctx.args.delay_modes or ['constant', 'matrix', 'matrix_gamma_velocity']
    for mode in modes:
        for seed in seeds:
            print(f"\n  mode={mode}, seed={seed}:")
            if mode == 'constant':
                params = make_params(ctx.distance, delay_mode='constant', seed=seed)
                params['delay'] = mean_delay_s
            elif mode == 'matrix':
                params = make_params(ctx.distance, delay_mode='matrix', seed=seed)
            elif mode == 'matrix_gamma_velocity':
                params = make_params(ctx.distance, delay_mode='matrix_gamma_velocity',
                                     seed=seed, conduction_velocity=15.0)
            else:
                print(f"    Unknown delay_mode {mode}, skipping.")
                continue

            m = run_model(ctx, params, duration=duration)
            Z_E, _ = m.compute_Z()
            half = len(m.time_array) // 2
            mean_Z = float(np.abs(Z_E[:, half:]).mean())
            BOLD_FC = m.compute_BOLD_FC(t_start=t_start)
            PLV = m.compute_PLV(t_start=t_start)
            mean_PLV = float(PLV[np.triu_indices(m.N, k=1)].mean())
            r, r_raw = r_vs_emp(BOLD_FC, ctx.emp_fc)
            delay_lo = float(m.delays.min() * 1000)
            delay_hi = float(m.delays.max() * 1000)
            delay_std = float(m.delays.std() * 1000)
            print(f"    delays [{delay_lo:.1f}, {delay_hi:.1f}] ms (σ={delay_std:.2f})")
            print(f"    |Z|={mean_Z:.4f}  PLV={mean_PLV:.3f}  "
                  f"r_aligned={r:.4f}  (r_raw={r_raw:.4f})")
            rows.append({'mode': mode, 'seed': seed, 'mean_Z': mean_Z,
                         'mean_PLV': mean_PLV, 'r_bold': r, 'r_bold_raw': r_raw,
                         'delay_lo': delay_lo, 'delay_hi': delay_hi,
                         'delay_std': delay_std})
            if seed == seeds[-1]:
                BOLD_FCs[mode] = BOLD_FC
                PLV_mats[mode] = PLV
                delays_seen[mode] = m.delays.copy()
                # Save 1 s zoom of R_E from middle of run, all 68 nodes
                t_arr = m.time_array
                half_idx = len(t_arr) // 2
                zoom_mask = (t_arr >= t_arr[half_idx]) & (t_arr <= t_arr[half_idx] + 1.0)
                RE_zooms[mode] = m.get_all_RE()[:, zoom_mask]

    # --- One-way ANOVA across the 3 modes for each metric (PLV is primary).
    # Delay distribution stats are reported as text:
    #   constant: 9.7 ms uniform
    #   matrix:   D/v with v=12 m/s, range [1.2, 16.5] ms
    #   matrix_gamma_velocity: D/v with v_i ~ truncated Gamma E[v]=15 m/s,
    #                          mean delay matched to matrix.
    from scipy.stats import f_oneway
    modes_seen = list(dict.fromkeys(r['mode'] for r in rows))
    def by_mode(field, mode):
        return [r[field] for r in rows if r['mode'] == mode]
    def me_se(xs):
        a = np.array(xs); a = a[np.isfinite(a)]
        if len(a) < 2:
            return float(a.mean()) if len(a) else float('nan'), float('nan')
        return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a)))

    # Per-mode summary
    summary = {}
    for m_name in modes_seen:
        summary[m_name] = {
            'r':   me_se(by_mode('r_bold',   m_name)),
            'PLV': me_se(by_mode('mean_PLV', m_name)),
            'Z':   me_se(by_mode('mean_Z',   m_name)),
        }
    # ANOVA (across modes)
    F_r,   p_r   = f_oneway(*[by_mode('r_bold',   m) for m in modes_seen])
    F_PLV, p_PLV = f_oneway(*[by_mode('mean_PLV', m) for m in modes_seen])
    F_Z,   p_Z   = f_oneway(*[by_mode('mean_Z',   m) for m in modes_seen])

    print(f"\n  --- 3-mode comparison (one-way ANOVA, n={len(seeds)} seeds) ---")
    print(f"  Per-mode mean ± SE:")
    for m_name in modes_seen:
        s = summary[m_name]
        print(f"    {m_name:<25} r={s['r'][0]:+.4f}±{s['r'][1]:.4f}   "
              f"PLV={s['PLV'][0]:.4f}±{s['PLV'][1]:.4f}   "
              f"|Z|={s['Z'][0]:.4f}±{s['Z'][1]:.4f}")
    print(f"  ANOVA: r       F={F_r:.2f}  p={p_r:.4f}")
    print(f"         PLV     F={F_PLV:.2f}  p={p_PLV:.4f}")
    print(f"         |Z|     F={F_Z:.2f}  p={p_Z:.4f}")

    save_data(ctx, 'exp_delays',
              modes=np.array([r['mode'] for r in rows]),
              seeds=np.array([r['seed'] for r in rows]),
              mean_Z=np.array([r['mean_Z'] for r in rows]),
              mean_PLV=np.array([r['mean_PLV'] for r in rows]),
              r_bold=np.array([r['r_bold'] for r in rows]),
              r_bold_raw=np.array([r['r_bold_raw'] for r in rows]),
              F_r=F_r, p_r=p_r, F_PLV=F_PLV, p_PLV=p_PLV, F_Z=F_Z, p_Z=p_Z,
              **{f'BOLD_FC_{m}': BOLD_FCs[m] for m in BOLD_FCs},
              **{f'PLV_mat_{m}': PLV_mats[m] for m in PLV_mats},
              **{f'delays_{m}': delays_seen[m] for m in delays_seen})

    # ============== Plots ==============
    # (1) PRIMARY: 3-mode metric grouped bar (r, PLV, |Z|) ± SE, ANOVA F/p
    metric_keys = ['r', 'PLV', 'Z']
    metric_labels = ['r aligned\n(BOLD,emp)', 'mean PLV', 'mean |Z|']
    F_p_per_metric = [(F_r, p_r), (F_PLV, p_PLV), (F_Z, p_Z)]
    n_modes = len(modes_seen)
    palette = sweep_palette(n_modes)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(metric_keys))
    w = 0.8 / n_modes
    for j, m_name in enumerate(modes_seen):
        means = [summary[m_name][k][0] for k in metric_keys]
        ses   = [summary[m_name][k][1] for k in metric_keys]
        offset = (j - (n_modes - 1) / 2) * w
        ax.bar(x + offset, means, w, yerr=ses,
               label=m_name, color=palette[j], capsize=3)
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(metric_labels)
    ax.set_ylabel('Metric value')
    ax.legend(loc='best', frameon=False, fontsize=9)
    save_fig(ctx, 'exp_delays__metrics_by_mode', fig)

    # (2) PLV matrices for all 3 modes — single 3-panel figure, same vmax
    plv_max = max(np.nanmax(p) for p in PLV_mats.values())
    n_modes_p = len(PLV_mats)
    fig, axes = plt.subplots(1, n_modes_p, figsize=(5 * n_modes_p, 5))
    if n_modes_p == 1:
        axes = [axes]
    for ax_i, mode in zip(axes, PLV_mats):
        im = ax_i.imshow(PLV_mats[mode], cmap='viridis', vmin=0, vmax=plv_max,
                         interpolation='nearest')
        ax_i.axhline(33.5, color='black', lw=0.5)
        ax_i.axvline(33.5, color='black', lw=0.5)
        ax_i.set_xlabel(f'Region (DK68 index)\ndelay mode = {mode}')
        ax_i.set_ylabel('Region (DK68 index)' if ax_i is axes[0] else '')
    fig.colorbar(im, ax=axes, fraction=0.046, pad=0.02, label='PLV')
    save_fig(ctx, 'exp_delays__plv_by_mode', fig)
    # (Delay-distribution histogram dropped — reported as text in caption.)
    # (R_E heatmaps, BOLD-FC matrices, paired dot plots dropped — text-only.)


# ====================================================================
# Dispatch
# ====================================================================
EXPERIMENTS: Dict[str, Callable[[Context], None]] = {
    'scale':           exp_scale,
    'topology':        exp_topology,
    'gap':             exp_gap,
    'ng_capabilities': exp_ng_capabilities,
    'scfc':            exp_scfc,
    'delays':          exp_delays,
}
# Backward-compat alias for old CLI invocations
EXPERIMENTS['wc_vs_nextgen'] = exp_ng_capabilities
# 6 experiments. Recommended ordering per ANALYSIS_REPORT:
#   1. scale       — Methods/calibration
#   2. topology    — bridge to Conti & Van Gorder testbed
#   3. wc_vs_nextgen — direct WC vs NG on connectome
#   4. gap         — GJ effect (Next Gen-only)  + dPLV σ (merged)
#   5. scfc        — structure-function relationship
#   6. delays      — delay heterogeneity necessity
ALL_EXPERIMENTS = list(EXPERIMENTS.keys())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Next Gen NMM experiments on HCP connectome.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--exp', nargs='+', default=['scale'],
                   help=("One or more of: " + ', '.join(EXPERIMENTS) + ", or 'all'"))
    p.add_argument('--duration', type=float, default=DEFAULT_DURATION,
                   help='Simulation duration (s). MUST be > 30 for BOLD FC. '
                        '`scale` defaults to 500 s if not overridden here.')
    p.add_argument('--seed', type=int, default=42, help='Random seed.')
    p.add_argument('--save', action='store_true',
                   help='Save .png figures and .npz raw results.')
    # Per-experiment sweep grids (None → use experiment-specific default)
    p.add_argument('--scales',     type=parse_float_list, default=None,
                   help='Comma list for `scale` (e.g. 1.2,1.4,1.5,1.6).')
    p.add_argument('--kv-scales',  type=parse_float_list, default=None,
                   help='Comma list for `gap` (× Forrester default κ_v).')
    p.add_argument('--k-ext',      type=parse_float_list, default=None,
                   help='Comma list for `coupling`.')
    p.add_argument('--etas',       type=parse_float_list, default=None,
                   help='Comma list for `synchrony` (η_E values).')
    p.add_argument('--delay-modes', type=parse_str_list, default=None,
                   help='Comma list for `delays` '
                        '(constant,matrix,matrix_gamma_velocity).')

    args = p.parse_args()
    args.duration_set_by_user = (
        '--duration' in sys.argv or '-duration' in sys.argv)
    return args


def main():
    args = parse_args()

    # Expand aliases
    requested = []
    for e in args.exp:
        if e == 'all':
            requested.extend(ALL_EXPERIMENTS)
        elif e in EXPERIMENTS:
            requested.append(e)
        else:
            print(f"  WARN: unknown experiment '{e}'; valid: "
                  f"{list(EXPERIMENTS.keys())} or 'all'")
    # de-dup, preserve order
    seen = set(); experiments = [x for x in requested
                                  if not (x in seen or seen.add(x))]
    if not experiments:
        print("  Nothing to run."); return

    np.random.seed(args.seed)
    ctx = load_context(args)
    print(f"\nRunning {len(experiments)} experiment(s): {experiments}")
    print(f"  duration={args.duration} s, seed={args.seed}, save={args.save}")
    print(f"  data dir: {DATA_DIR}")

    t_total = time.time()
    for name in experiments:
        t0 = time.time()
        EXPERIMENTS[name](ctx)
        print(f"\n  [{name}] elapsed: {time.time() - t0:.1f} s")

    print(f"\nAll done in {time.time() - t_total:.1f} s.")


if __name__ == '__main__':
    main()
