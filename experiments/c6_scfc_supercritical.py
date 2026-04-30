"""c6_scfc_supercritical.py — Structural baseline for the
"Model versus structural baseline" paragraph of the report.

Computes the headline reference number r(SC, empirical FC) = 0.403 on
the HCP DK68 dataset, plus two supplementary statistics that the report
text does not currently quote but that are useful when discussing the
gap between r(SC, empirical) and r(simulated, empirical):

  - Steiger Z (correlated r's, two share emp): tests whether SC and
    simulated BOLD FC predict empirical FC equally well.
  - Partial r(simulated BOLD, empirical | SC): the simulated BOLD FC's
    explanatory power beyond what SC already provides.

For the simulated BOLD FC at the operating point (eta_E = -2.15,
s = 1.4, v_m = 12 m/s, kappa_v at the argmax = 5x defaults,
gamma-distributed velocity), the script first tries to read it from the
existing kappa_v sweep (Results/c3_kappa_v_sweep_HCP.npz, BOLD_FC_z at
kappa_v_scale = 5). If that entry is missing, it falls back to running
a fresh single-seed simulation.

Outputs:
  Results/c6_scfc_supercritical.npz   r(SC,emp), r(BOLD,emp), Steiger Z
                                       and p, partial r and CI.
"""
import os, sys, time
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
from next_gen_network import NextGenNetwork

# ---- Operating point (matches c3_kappa_v_sweep_HCP.py) ----
ETA_E              = -2.15
BEST_S             = 1.4
BEST_KV_SCALE      = 5.0      # argmax of the kappa_v sweep
VM_FIXED           = 12.0
DEFAULT_KV_EE      = 0.01
DEFAULT_KV_II      = 0.025
DURATION_S         = 500.0
DT                 = 0.001
SEED               = 42
N_BOOT             = 1000
T_START            = max(30.0, DURATION_S * 0.2)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RES_DIR  = os.path.join(os.path.dirname(__file__), 'Results')
KV_NPZ   = os.path.join(RES_DIR, 'c3_kappa_v_sweep_HCP.npz')
NPZ      = os.path.join(RES_DIR, 'c6_scfc_supercritical.npz')

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


def load_simulated_FC_from_kv_sweep():
    """Reuse Results/c3_kappa_v_sweep_HCP.npz, BOLD_FC_z at kappa_v_scale=5
    if available."""
    if not os.path.exists(KV_NPZ): return None
    d = np.load(KV_NPZ, allow_pickle=True)
    key = f'BOLD_FC_z_kv{BEST_KV_SCALE:.2f}'
    if key not in d.files: return None
    print(f"reusing simulated FC from {KV_NPZ} (key={key})", flush=True)
    return np.array(d[key])


def run_simulation():
    """Fallback single-seed simulation at the operating point."""
    print(f"running fresh simulation at operating point "
          f"(kappa_v_scale={BEST_KV_SCALE}, gamma-velocity)", flush=True)
    p = dict(eta_E=ETA_E, k_ext=0.2,
             delay_mode='matrix_gamma_velocity',
             distance_matrix=DIST * BEST_S,
             kappa_v_EE=DEFAULT_KV_EE * BEST_KV_SCALE,
             kappa_v_II=DEFAULT_KV_II * BEST_KV_SCALE,
             conduction_velocity=VM_FIXED,
             velocity_gamma_shape=4.5,
             velocity_truncate_low=1.0,
             velocity_truncate_high=20.0,
             seed=SEED)
    m = NextGenNetwork(HCPNet(SC), params=p)
    t0 = time.time()
    m.run(duration=DURATION_S, dt=DT)
    print(f"  integ {time.time()-t0:.1f}s", flush=True)
    BOLD_FC = m.compute_BOLD_FC(t_start=T_START)
    return enigma_align(BOLD_FC)


def main():
    os.makedirs(RES_DIR, exist_ok=True)
    print(f"=== c6_scfc_supercritical eta_E={ETA_E}, s={BEST_S}, "
          f"kappa_v_scale={BEST_KV_SCALE} ===", flush=True)

    BOLD_FC_z = load_simulated_FC_from_kv_sweep()
    if BOLD_FC_z is None:
        BOLD_FC_z = run_simulation()

    iu = np.triu_indices(SC.shape[0], k=1)
    sc_v   = SC[iu]
    bold_v = BOLD_FC_z[iu]
    emp_v  = EMP[iu]
    n_pair = len(sc_v)

    r_sc_emp   = float(np.corrcoef(sc_v,   emp_v)[0, 1])
    r_bold_emp = float(np.corrcoef(bold_v, emp_v)[0, 1])
    r_sc_bold  = float(np.corrcoef(sc_v,   bold_v)[0, 1])

    sc_lo,   sc_hi   = bootstrap_r_ci(SC,        EMP)
    bold_lo, bold_hi = bootstrap_r_ci(BOLD_FC_z, EMP)
    Z_st, p_st = steiger_z(r_sc_emp, r_bold_emp, r_sc_bold, n_pair)

    r_part = partial_r(bold_v, emp_v, sc_v)
    p_lo, p_hi = bootstrap_partial_r_ci(bold_v, emp_v, sc_v)

    print(f"\n  r(SC,   empirical) = {r_sc_emp:+.4f} "
          f"[{sc_lo:+.4f}, {sc_hi:+.4f}]", flush=True)
    print(f"  r(BOLD, empirical) = {r_bold_emp:+.4f} "
          f"[{bold_lo:+.4f}, {bold_hi:+.4f}]", flush=True)
    print(f"  r(SC,   BOLD)      = {r_sc_bold:+.4f}", flush=True)
    print(f"  Steiger Z (SC vs BOLD against empirical) "
          f"= {Z_st:+.3f}, p = {p_st:.3e}", flush=True)
    print(f"  partial r(BOLD, empirical | SC) = {r_part:+.4f} "
          f"[{p_lo:+.4f}, {p_hi:+.4f}]", flush=True)

    np.savez(NPZ,
             eta_E=ETA_E, best_s=BEST_S, best_kv_scale=BEST_KV_SCALE,
             vm_fixed=VM_FIXED, duration_s=DURATION_S, seed=SEED,
             SC=SC, EMP=EMP, BOLD_FC_z=BOLD_FC_z,
             r_sc_emp=r_sc_emp, r_sc_emp_lo=sc_lo, r_sc_emp_hi=sc_hi,
             r_bold_emp=r_bold_emp, r_bold_emp_lo=bold_lo,
             r_bold_emp_hi=bold_hi, r_sc_bold=r_sc_bold,
             steiger_Z=Z_st, steiger_p=p_st,
             partial_r=r_part, partial_r_lo=p_lo, partial_r_hi=p_hi)
    print(f"\nSaved: {NPZ}", flush=True)


if __name__ == '__main__':
    main()
