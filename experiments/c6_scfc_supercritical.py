"""c6_scfc_supercritical.py — SC vs sim BOLD FC vs empirical FC reality check.

At main-line operating point (η_E=-2.15, BEST_S, BEST_KV_SCALE,
BEST_DELAY_MODE) compare three matrices:
  (a) row-normalised SC (HCP-DK68)
  (b) simulated BOLD FC (Balloon-Windkessel from R_E)
  (c) empirical HCP rs-fMRI FC (ENIGMA-aligned)

BEST_DELAY_MODE is whichever C5 mode achieved highest r(BOLD, emp). If
gamma wins, this script automatically averages over GAMMA_SEEDS for the
sim BOLD FC (mean ± SE on r); for deterministic modes a single seed.

Statistical reporting:
  r(sim BOLD, emp)            with bootstrap CI (or seed mean ± SE for gamma)
  r(SC, emp)                  baseline
  Steiger Z + p               (correlated r's, share emp)
  partial r(sim, emp | SC)    with bootstrap CI
  r(sim, SC)                  redundancy with structure

Figure: 3-panel matrix grid (SC | sim BOLD FC | empirical FC) with
overlaid r values and Steiger Z annotation.
"""
import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
from next_gen_network import NextGenNetwork

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
    'font.size': 11, 'axes.labelsize': 11, 'axes.titlesize': 11,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
})

ETA_E              = -2.15
BEST_S             = 1.4      # from C2 (best_s, r=+0.157)
BEST_KV_SCALE      = 5.0      # from C4 (r=+0.216)
BEST_DELAY_MODE    = 'gamma'      # default for super-critical Gamma vm=12 main line; orchestrator overrides from C5
VM_FIXED           = 12.0  # main-line vm (Forrester representative)
DEFAULT_KV_EE      = 0.01
DEFAULT_KV_II      = 0.025

GAMMA_SEEDS        = [42]   # SINGLE SEED (multi-seed averaging skipped per user request)

DURATION_S = 500.0
DT         = 0.001
SEED       = 42
N_BOOT     = 1000

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RES_DIR  = os.path.join(os.path.dirname(__file__), 'Results')
PLT_DIR  = os.path.join(os.path.dirname(__file__), 'Plots')
NPZ      = os.path.join(RES_DIR, 'c6_scfc_supercritical.npz')
PNG      = os.path.join(PLT_DIR, 'c6_scfc_supercritical.png')

SC   = np.load(os.path.join(DATA_DIR, 'hcp_sc_68.npy'))
EMP  = np.load(os.path.join(DATA_DIR, 'hcp_fc_68.npy'))
DIST = np.load(os.path.join(DATA_DIR, 'hcp_dist_68.npy'))


class HCPNet:
    def __init__(self, A):
        self.A = A; self.N = A.shape[0]
        self.edges = [(i, j) for i in range(A.shape[0])
                      for j in range(A.shape[0]) if A[i, j] > 0 and i != j]


def enigma_align(FC):
    fc = FC.copy(); np.fill_diagonal(fc, 0)
    fc = np.maximum(0, fc)
    return np.arctanh(np.clip(fc, 0, 0.999))


def fc_corr(A, B):
    iu = np.triu_indices(A.shape[0], k=1)
    return float(np.corrcoef(A[iu], B[iu])[0, 1])


def bootstrap_r_ci(A, B, n_boot=N_BOOT, seed=42):
    iu = np.triu_indices(A.shape[0], k=1)
    a, b = A[iu], B[iu]
    n = len(a); rng = np.random.default_rng(seed)
    rs = np.empty(n_boot)
    for i in range(n_boot):
        ix = rng.choice(n, n, replace=True)
        rs[i] = np.corrcoef(a[ix], b[ix])[0, 1]
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def partial_r(X, Y, Z):
    """Partial correlation r(X, Y | Z), all 1-D arrays."""
    rxy = np.corrcoef(X, Y)[0, 1]
    rxz = np.corrcoef(X, Z)[0, 1]
    ryz = np.corrcoef(Y, Z)[0, 1]
    denom = np.sqrt((1 - rxz**2) * (1 - ryz**2))
    return float((rxy - rxz * ryz) / denom) if denom > 0 else float('nan')


def bootstrap_partial_r_ci(X, Y, Z, n_boot=N_BOOT, seed=42):
    n = len(X); rng = np.random.default_rng(seed)
    rs = np.empty(n_boot)
    for i in range(n_boot):
        ix = rng.choice(n, n, replace=True)
        rs[i] = partial_r(X[ix], Y[ix], Z[ix])
    return (float(np.nanpercentile(rs, 2.5)),
            float(np.nanpercentile(rs, 97.5)))


def steiger_z(r12, r13, r23, n):
    """Steiger Z for two correlated correlations sharing variable 1.
    Tests H0: r12 == r13. Returns (Z, two-sided p)."""
    z12 = np.arctanh(r12); z13 = np.arctanh(r13)
    rm = (r12 + r13) / 2.0
    f = (1 - r23) / (2 * (1 - rm**2))
    h = (1 - f * rm**2) / (1 - rm**2)
    se = np.sqrt(2 * (1 - r23) * h / (n - 3))
    Z = (z12 - z13) / se
    p = 2 * (1 - stats.norm.cdf(abs(Z)))
    return float(Z), float(p)


def make_delay_kwargs(mode, best_s):
    """Build NG params for the chosen best delay mode from C5."""
    if mode == 'constant':
        matrix_D = (DIST * best_s) / VM_FIXED
        mask = SC > 0
        const_delay_s = float(matrix_D[mask].mean())
        return dict(delay_mode='constant', delay=const_delay_s,
                    distance_matrix=np.zeros_like(DIST))
    elif mode == 'matrix':
        return dict(delay_mode='matrix',
                    distance_matrix=DIST * best_s,
                    conduction_velocity=VM_FIXED)
    elif mode == 'gamma':
        return dict(delay_mode='matrix_gamma_velocity',
                    distance_matrix=DIST * best_s,
                    conduction_velocity=VM_FIXED,
                    velocity_gamma_shape=4.5,
                    velocity_truncate_low=1.0,
                    velocity_truncate_high=20.0)
    raise ValueError(f"Unknown delay mode: {mode}")


def run_one(seed, delay_kwargs):
    p = dict(eta_E=ETA_E, k_ext=0.2,
             kappa_v_EE=DEFAULT_KV_EE * float(BEST_KV_SCALE),
             kappa_v_II=DEFAULT_KV_II * float(BEST_KV_SCALE),
             seed=seed, **delay_kwargs)
    m = NextGenNetwork(HCPNet(SC), params=p)
    t0 = time.time()
    m.run(duration=DURATION_S, dt=DT)
    print(f"    integ {time.time()-t0:.1f}s (seed={seed})", flush=True)
    t_start = max(30.0, DURATION_S * 0.2)
    BOLD_FC = m.compute_BOLD_FC(t_start=t_start)
    BOLD_FC_z = enigma_align(BOLD_FC)
    PLV = m.compute_PLV(t_start=t_start)
    return BOLD_FC_z, PLV


def main():
    os.makedirs(RES_DIR, exist_ok=True)
    os.makedirs(PLT_DIR, exist_ok=True)
    print(f"=== c6_scfc_supercritical η_E={ETA_E}, s={BEST_S}, "
          f"κv_scale={BEST_KV_SCALE}, delay={BEST_DELAY_MODE} ===", flush=True)
    print(f"start: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    delay_kw = make_delay_kwargs(BEST_DELAY_MODE, BEST_S)

    # Run sim(s)
    if BEST_DELAY_MODE == 'gamma':
        print(f"  gamma → averaging over {len(GAMMA_SEEDS)} seeds", flush=True)
        BOLD_seeds = []
        PLV_seeds = []
        r_seeds = []
        for sd in GAMMA_SEEDS:
            BFC_z, PLV = run_one(sd, delay_kw)
            BOLD_seeds.append(BFC_z)
            PLV_seeds.append(PLV)
            r_seeds.append(fc_corr(BFC_z, EMP))
            print(f"    r(BOLD,emp) seed {sd} = {r_seeds[-1]:+.4f}", flush=True)
        BOLD_FC_z = np.mean(BOLD_seeds, axis=0)
        PLV       = np.mean(PLV_seeds,  axis=0)
        r_seeds   = np.array(r_seeds)
        n_seeds   = len(r_seeds)
        r_mean    = float(r_seeds.mean())
        r_se      = float(r_seeds.std(ddof=1) / np.sqrt(n_seeds))
        print(f"  → r mean ± SE = {r_mean:+.4f} ± {r_se:.4f}", flush=True)
    else:
        BOLD_FC_z, PLV = run_one(SEED, delay_kw)
        r_seeds = np.array([fc_corr(BOLD_FC_z, EMP)])
        n_seeds = 1
        r_mean  = float(r_seeds[0])
        r_se    = 0.0

    iu = np.triu_indices(SC.shape[0], k=1)
    sc_v   = SC[iu]
    bold_v = BOLD_FC_z[iu]
    emp_v  = EMP[iu]
    n_pair = len(sc_v)

    r_bold_emp = float(np.corrcoef(bold_v, emp_v)[0, 1])
    r_sc_emp   = float(np.corrcoef(sc_v,   emp_v)[0, 1])
    r_sc_bold  = float(np.corrcoef(sc_v,   bold_v)[0, 1])
    r_plv_emp  = float(np.corrcoef(PLV[iu], emp_v)[0, 1])

    bold_lo, bold_hi = bootstrap_r_ci(BOLD_FC_z, EMP)
    sc_lo,   sc_hi   = bootstrap_r_ci(SC,        EMP)
    Z_st, p_st = steiger_z(r_sc_emp, r_bold_emp, r_sc_bold, n_pair)

    r_part = partial_r(bold_v, emp_v, sc_v)
    p_lo, p_hi = bootstrap_partial_r_ci(bold_v, emp_v, sc_v)

    print(f"\n  r(SC,   emp)   = {r_sc_emp:+.4f} [{sc_lo:+.4f}, {sc_hi:+.4f}]",
          flush=True)
    print(f"  r(BOLD, emp)   = {r_bold_emp:+.4f} [{bold_lo:+.4f}, {bold_hi:+.4f}]",
          flush=True)
    if n_seeds > 1:
        print(f"  r(BOLD, emp)   per-seed = {[f'{r:+.3f}' for r in r_seeds]}  "
              f"mean ± SE = {r_mean:+.4f} ± {r_se:.4f}", flush=True)
    print(f"  r(SC,   BOLD)  = {r_sc_bold:+.4f}", flush=True)
    print(f"  r(PLV,  emp)   = {r_plv_emp:+.4f}  (neural ~10 Hz vs BOLD ~0.05 Hz)",
          flush=True)
    print(f"  Steiger Z (SC vs BOLD against emp) = {Z_st:+.3f}, p = {p_st:.3e}",
          flush=True)
    print(f"  partial r(BOLD, emp | SC) = {r_part:+.4f} "
          f"[{p_lo:+.4f}, {p_hi:+.4f}]", flush=True)

    np.savez(NPZ, eta_E=ETA_E, best_s=BEST_S, best_kv_scale=BEST_KV_SCALE,
             best_delay_mode=BEST_DELAY_MODE, n_seeds=n_seeds,
             duration_s=DURATION_S, seed=SEED, gamma_seeds=np.array(GAMMA_SEEDS),
             SC=SC, BOLD_FC_z=BOLD_FC_z, EMP=EMP, PLV=PLV,
             r_seeds=r_seeds, r_mean_seeds=r_mean, r_se_seeds=r_se,
             r_bold_emp=r_bold_emp, r_sc_emp=r_sc_emp,
             r_sc_bold=r_sc_bold, r_plv_emp=r_plv_emp,
             r_bold_emp_lo=bold_lo, r_bold_emp_hi=bold_hi,
             r_sc_emp_lo=sc_lo, r_sc_emp_hi=sc_hi,
             steiger_Z=Z_st, steiger_p=p_st,
             partial_r=r_part, partial_r_lo=p_lo, partial_r_hi=p_hi)

    # 3-panel order: SC | Empirical | Simulated.
    # Each panel has its own colorbar so structure within each is legible
    # (empirical Fisher-z range is much narrower than simulated).
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panels = [
        (SC,        'Structural connectivity\n(ENIGMA HCP, raw edge weights)',
         float(np.max(SC))),
        (EMP,       'Empirical BOLD FC\n(ENIGMA HCP rs-fMRI, Fisher-z)',
         float(np.max(EMP))),
        (BOLD_FC_z, 'Simulated BOLD FC\n(NG model, ENIGMA-aligned Fisher-z)',
         float(np.max(BOLD_FC_z))),
    ]
    for ax, (M, title, vmx) in zip(axes, panels):
        im = ax.imshow(M, cmap='viridis', vmin=0, vmax=vmx,
                       interpolation='nearest')
        ax.axhline(33.5, color='black', lw=0.5)
        ax.axvline(33.5, color='black', lw=0.5)
        ax.set_title(title)
        ax.set_xlabel('Region (DK68 index)')
        ax.set_ylabel('Region (DK68 index)' if ax is axes[0] else '')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    fig.savefig(PNG)
    plt.close(fig)
    print(f"\nSaved: {PNG}", flush=True)


if __name__ == '__main__':
    main()
