"""c3_kappa_v_sweep_HCP.py — Gap-junction effect on BOLD-FC pattern
(report Section "Gap-junction effect").

Sweeps a multiplicative scale c on the Forrester gap-junction defaults
kappa_v_EE(c) = 0.01*c, kappa_v_II(c) = 0.025*c, holding the calibrated
operating point fixed (eta_E = -2.15, s = 1.4, v_m = 12 m/s, gamma-
distributed velocity). Each run is 500 s + 100 s transient discard at
seed 42, with ENIGMA-aligned Fisher-z preprocessing on the simulated
BOLD FC.

The script is idempotent: if Results/c3_kappa_v_sweep_HCP.npz already
contains entries for some kappa_v-scale values, those are reused and
only the missing ones are simulated.

Outputs:
  Results/c3_kappa_v_sweep_HCP.npz   per-kappa_v r / CI / mean|Z| / BOLD_FC_z
  Plots/sc_fc_emp_sim.png            4-panel 2x2 (headline figure):
                                     SC, empirical FC,
                                     simulated FC at the kappa_v argmax,
                                     simulated FC without gap junctions.
"""
import os, sys, time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
from next_gen_network import NextGenNetwork

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
    'font.size': 11, 'axes.labelsize': 11, 'axes.titlesize': 11,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# ---- Operating point (matches report Methods) ----
ETA_E              = -2.15           # super-critical, just past Hopf
BEST_S             = 1.4             # calibrated distance scale
VM_FIXED           = 12.0            # gamma-velocity mode
KV_SCALE_GRID      = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
DEFAULT_KV_EE      = 0.01            # Forrester default
DEFAULT_KV_II      = 0.025           # Forrester default
GAMMA_P            = 4.5             # gamma shape (Atay-Hutt)
GAMMA_VL           = 1.0
GAMMA_VH           = 20.0
DURATION_S         = 500.0
DT                 = 0.001
SEED               = 42
N_BOOT             = 1000
T_START            = max(30.0, DURATION_S * 0.2)   # 100 s transient

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RES_DIR  = os.path.join(os.path.dirname(__file__), 'Results')
PLT_DIR  = os.path.join(os.path.dirname(__file__), 'Plots')
NPZ      = os.path.join(RES_DIR, 'c3_kappa_v_sweep_HCP.npz')
PNG      = os.path.join(PLT_DIR, 'sc_fc_emp_sim.png')

SC   = np.load(os.path.join(DATA_DIR, 'hcp_sc_68.npy'))
EMP  = np.load(os.path.join(DATA_DIR, 'hcp_fc_68.npy'))
DIST = np.load(os.path.join(DATA_DIR, 'hcp_dist_68.npy'))


class HCPNet:
    def __init__(self, A):
        self.A = A; self.N = A.shape[0]
        self.edges = [(i, j) for i in range(A.shape[0])
                      for j in range(A.shape[0]) if A[i, j] > 0 and i != j]


def enigma_align(FC):
    """ENIGMA-style alignment on simulated FC for fair comparison
    against the empirical Fisher-z FC."""
    fc = FC.copy(); np.fill_diagonal(fc, 0)
    fc = np.maximum(0, fc)
    return np.arctanh(np.clip(fc, 0, 0.999))


def fc_corr(A, B):
    iu = np.triu_indices(A.shape[0], k=1)
    return float(np.corrcoef(A[iu], B[iu])[0, 1])


def bootstrap_ci(sim, emp, n_boot=N_BOOT, seed=42):
    iu = np.triu_indices(sim.shape[0], k=1)
    a, b = sim[iu], emp[iu]
    n = len(a); rng = np.random.default_rng(seed)
    rs = np.empty(n_boot)
    for i in range(n_boot):
        ix = rng.choice(n, n, replace=True)
        rs[i] = np.corrcoef(a[ix], b[ix])[0, 1]
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def run_one(kv_scale):
    print(f"\n=== kappa_v_scale = {kv_scale:.2f} === "
          f"[{time.strftime('%H:%M:%S')}]", flush=True)
    p = dict(eta_E=ETA_E, k_ext=0.2,
             delay_mode='matrix_gamma_velocity',
             distance_matrix=DIST * BEST_S,
             kappa_v_EE=DEFAULT_KV_EE * float(kv_scale),
             kappa_v_II=DEFAULT_KV_II * float(kv_scale),
             conduction_velocity=VM_FIXED,
             velocity_gamma_shape=GAMMA_P,
             velocity_truncate_low=GAMMA_VL,
             velocity_truncate_high=GAMMA_VH,
             seed=SEED)
    m = NextGenNetwork(HCPNet(SC), params=p)
    t0 = time.time()
    m.run(duration=DURATION_S, dt=DT)
    Z_E, _ = m.compute_Z()
    half = Z_E.shape[1] // 2
    mean_Z = float(np.abs(Z_E[:, half:]).mean())
    BOLD_FC = m.compute_BOLD_FC(t_start=T_START)
    BOLD_FC_z = enigma_align(BOLD_FC)
    r_a    = fc_corr(BOLD_FC_z, EMP)
    lo, hi = bootstrap_ci(BOLD_FC_z, EMP)
    print(f"  integ {time.time()-t0:.1f}s  |Z|={mean_Z:.4f}  "
          f"r={r_a:+.4f} [{lo:+.4f},{hi:+.4f}]", flush=True)
    return BOLD_FC_z, mean_Z, r_a, lo, hi


def load_existing(npz_path):
    if not os.path.exists(npz_path): return {}
    d = np.load(npz_path, allow_pickle=True)
    if 'kv_scale_done' not in d.files: return {}
    out = {}
    for i, kv in enumerate(d['kv_scale_done']):
        kv = float(kv)
        out[kv] = dict(FC_z=d[f'BOLD_FC_z_kv{kv:.2f}'],
                       mean_Z=float(d['mean_Z'][i]),
                       r_a=float(d['r_aligned'][i]),
                       lo=float(d['r_lo'][i]),
                       hi=float(d['r_hi'][i]))
    return out


def save_inc(done, results):
    out = dict(kv_scale_done=np.array(done, dtype=float),
               kv_scale_grid=KV_SCALE_GRID,
               eta_E=ETA_E, best_s=BEST_S, vm_fixed=VM_FIXED,
               kv_EE_default=DEFAULT_KV_EE, kv_II_default=DEFAULT_KV_II,
               duration_s=DURATION_S, seed=SEED,
               mean_Z=np.array([results[k]['mean_Z'] for k in done]),
               r_aligned=np.array([results[k]['r_a'] for k in done]),
               r_lo=np.array([results[k]['lo']      for k in done]),
               r_hi=np.array([results[k]['hi']      for k in done]))
    for k in done:
        out[f'BOLD_FC_z_kv{k:.2f}'] = results[k]['FC_z']
    np.savez(NPZ, **out)


def make_headline_figure(results):
    """4-panel 2x2: SC | empirical FC | sim FC at argmax kappa_v |
    sim FC at kappa_v=0. Per-panel vmax."""
    if 0.0 not in results:
        print("[fig] missing kappa_v=0; skip headline figure", flush=True)
        return
    rs = {k: v['r_a'] for k, v in results.items()}
    kv_best = float(max(rs, key=rs.get))
    if kv_best == 0.0:
        print("[fig] argmax is at kappa_v=0; skip headline figure", flush=True)
        return

    iu = np.triu_indices(SC.shape[0], k=1)
    r_sc_emp = float(np.corrcoef(SC[iu], EMP[iu])[0, 1])
    fc_best = results[kv_best]['FC_z']
    r_best  = results[kv_best]['r_a']
    fc_off  = results[0.0]['FC_z']
    r_off   = results[0.0]['r_a']

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    ax = axes.flatten()
    panels = [
        (SC,
         'Structural connectivity\n(ENIGMA HCP, raw edge weights)',
         float(np.max(SC))),
        (EMP,
         f'Empirical FC (HCP rs-fMRI, Fisher-z)\n'
         rf'r(SC, empirical) = {r_sc_emp:.3f}',
         float(np.max(EMP))),
        (fc_best,
         rf'Simulated FC with gap junction (Fisher-z)'
         + '\n'
         + rf'($\kappa_v$_scale={kv_best:.2f}): r={r_best:+.3f}',
         float(np.nanmax(fc_best))),
        (fc_off,
         rf'Simulated FC without gap junction (Fisher-z)'
         + '\n'
         + rf'($\kappa_v$=0): r={r_off:+.3f}',
         float(np.nanmax(fc_off))),
    ]
    for a, (M, title, vmx) in zip(ax, panels):
        im = a.imshow(M, cmap='viridis', vmin=0, vmax=vmx,
                      interpolation='nearest')
        a.axhline(33.5, color='black', lw=0.5)
        a.axvline(33.5, color='black', lw=0.5)
        a.set_title(title, fontsize=11)
        a.set_xlabel('Region (DK68 index)')
        a.set_ylabel('Region (DK68 index)')
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)
    fig.tight_layout()
    fig.savefig(PNG)
    plt.close(fig)
    print(f"[fig] saved {PNG} (argmax kappa_v_scale={kv_best})", flush=True)


def main():
    os.makedirs(RES_DIR, exist_ok=True)
    os.makedirs(PLT_DIR, exist_ok=True)
    print(f"=== c3_kappa_v_sweep_HCP eta_E={ETA_E}, s={BEST_S}, "
          f"vm={VM_FIXED} ===", flush=True)
    print(f"kappa_v_scale grid: {KV_SCALE_GRID.tolist()}", flush=True)

    results = load_existing(NPZ)
    if results:
        print(f"resuming from {NPZ}: already have "
              f"{sorted(results.keys())}", flush=True)

    t0_total = time.time()
    for kv in KV_SCALE_GRID:
        kv = float(kv)
        if kv in results:
            print(f"\n--- kappa_v_scale={kv} already done "
                  f"(r={results[kv]['r_a']:+.4f}); skip", flush=True)
            continue
        try:
            FC_z, mZ, r_a, lo, hi = run_one(kv)
            results[kv] = dict(FC_z=FC_z, mean_Z=mZ, r_a=r_a, lo=lo, hi=hi)
            save_inc(sorted(results.keys()), results)
        except Exception as e:
            print(f"  !! kappa_v_scale={kv} FAILED: {e}", flush=True)

    make_headline_figure(results)

    print(f"\n=== r vs kappa_v_scale summary ===", flush=True)
    for kv in KV_SCALE_GRID:
        kv = float(kv)
        if kv in results:
            r = results[kv]
            print(f"  kv_scale={kv:4.2f}  r={r['r_a']:+.4f} "
                  f"[{r['lo']:+.4f}, {r['hi']:+.4f}]", flush=True)
    print(f"\n=== DONE in {(time.time()-t0_total)/60:.1f} min ===",
          flush=True)


if __name__ == '__main__':
    main()
