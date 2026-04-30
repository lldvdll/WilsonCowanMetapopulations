"""c3_gamma_vm_sweep.py — Conduction-velocity sensitivity (report Section
"Conduction-velocity sensitivity").

Sweeps the mode v_m of the gamma-distributed velocity model
(Atay-Hutt 2006) over {6, 8, 10, 12, 15} m/s at the calibrated
operating point (eta_E = -2.15, s = 1.4, kappa_v at Forrester defaults).
Each run uses 500 s simulation + 100 s transient discard, single seed
(42), and ENIGMA-aligned Fisher-z preprocessing on the simulated BOLD
FC for direct comparison against the empirical HCP rs-fMRI FC.

The script is idempotent: if Results/c3_gamma_vm_sweep.npz already
contains entries for some v_m, those are reused and only the missing
ones are simulated.

Outputs:
  Results/c3_gamma_vm_sweep.npz   per-v_m r / CI / mean|Z| / BOLD_FC_z
  Plots/gamma_vm_FC.png           4-panel 2x2 (headline figure):
                                  empirical FC + simulated FC at
                                  v_m in {8, 12, 15}.
"""
import os, sys, time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
from next_gen_network import NextGenNetwork

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# ---- Operating point (matches report Methods) ----
ETA_E      = -2.15            # super-critical, just past Hopf eta_E* ~= -2.25
S_FIXED    = 1.4              # calibrated distance scale
VM_GRID    = [6.0, 8.0, 10.0, 12.0, 15.0]
K_EXT      = 0.2              # Forrester default
KAPPA_V_EE = 0.01             # Forrester default
KAPPA_V_II = 0.025            # Forrester default
GAMMA_P    = 4.5              # gamma shape (Atay-Hutt)
GAMMA_VL   = 1.0              # truncation low
GAMMA_VH   = 20.0             # truncation high
DURATION_S = 500.0
DT         = 0.001
SEED       = 42
N_BOOT     = 1000
T_START    = max(30.0, DURATION_S * 0.2)   # 100 s transient discard

# Panels shown in the headline 4-panel figure
FIG_VM = [8.0, 12.0, 15.0]

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RES_DIR  = os.path.join(os.path.dirname(__file__), 'Results')
PLT_DIR  = os.path.join(os.path.dirname(__file__), 'Plots')
NPZ      = os.path.join(RES_DIR, 'c3_gamma_vm_sweep.npz')
PNG      = os.path.join(PLT_DIR, 'gamma_vm_FC.png')

SC   = np.load(os.path.join(DATA_DIR, 'hcp_sc_68.npy'))
EMP  = np.load(os.path.join(DATA_DIR, 'hcp_fc_68.npy'))
DIST = np.load(os.path.join(DATA_DIR, 'hcp_dist_68.npy'))


class HCPNet:
    def __init__(self, A):
        self.A = A; self.N = A.shape[0]
        self.edges = [(i, j) for i in range(A.shape[0])
                      for j in range(A.shape[0]) if A[i, j] > 0 and i != j]


def enigma_align(FC):
    """ENIGMA-style alignment for fair comparison against empirical FC.
    Half-wave rectify and Fisher-z, matching the empirical preprocessing."""
    fc = FC.copy(); np.fill_diagonal(fc, 0)
    fc = np.maximum(0, fc)
    return np.arctanh(np.clip(fc, 0, 0.999))


def fc_corr(A, B):
    iu = np.triu_indices(A.shape[0], k=1)
    return float(np.corrcoef(A[iu], B[iu])[0, 1])


def bootstrap_ci(sim, emp, n_boot=N_BOOT, seed=42):
    """95% bootstrap CI of r(sim, emp) over the 2278 upper-triangle edges."""
    iu = np.triu_indices(sim.shape[0], k=1)
    a, b = sim[iu], emp[iu]
    n = len(a); rng = np.random.default_rng(seed)
    rs = np.empty(n_boot)
    for i in range(n_boot):
        ix = rng.choice(n, n, replace=True)
        rs[i] = np.corrcoef(a[ix], b[ix])[0, 1]
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def run_one(vm):
    print(f"\n=== vm={vm} m/s, eta={ETA_E}, s={S_FIXED}, gamma-velocity === "
          f"[{time.strftime('%H:%M:%S')}]", flush=True)
    p = dict(eta_E=ETA_E, k_ext=K_EXT,
             delay_mode='matrix_gamma_velocity',
             distance_matrix=DIST * S_FIXED,
             kappa_v_EE=KAPPA_V_EE, kappa_v_II=KAPPA_V_II,
             conduction_velocity=vm,
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
    r_a = fc_corr(BOLD_FC_z, EMP)
    lo, hi = bootstrap_ci(BOLD_FC_z, EMP)
    print(f"  integ {time.time()-t0:.1f}s  |Z|={mean_Z:.4f}  "
          f"r={r_a:+.4f} [{lo:+.4f},{hi:+.4f}]", flush=True)
    return BOLD_FC_z, mean_Z, r_a, lo, hi


def load_existing(npz_path):
    """Load already-completed sweep entries so we can skip them on re-run."""
    if not os.path.exists(npz_path): return {}
    d = np.load(npz_path, allow_pickle=True)
    if 'vm_done' not in d.files: return {}
    out = {}
    for i, vm in enumerate(d['vm_done']):
        vm = float(vm)
        out[vm] = dict(FC_z=d[f'BOLD_FC_z_vm{vm:.0f}'],
                       mean_Z=float(d['mean_Z'][i]),
                       r_a=float(d['r_aligned'][i]),
                       lo=float(d['r_lo'][i]),
                       hi=float(d['r_hi'][i]))
    return out


def save_inc(done, results):
    out = dict(vm_grid=np.array(VM_GRID), eta_E=ETA_E, s_fixed=S_FIXED,
               duration_s=DURATION_S, seed=SEED,
               vm_done=np.array(done, dtype=float),
               r_aligned=np.array([results[v]['r_a']  for v in done]),
               r_lo=np.array([results[v]['lo']        for v in done]),
               r_hi=np.array([results[v]['hi']        for v in done]),
               mean_Z=np.array([results[v]['mean_Z']  for v in done]))
    for v in done:
        out[f'BOLD_FC_z_vm{v:.0f}'] = results[v]['FC_z']
    np.savez(NPZ, **out)


def make_headline_figure(results):
    """4-panel 2x2 figure: empirical FC + simulated FC at v_m in {8,12,15}.
    Per-panel vmax (Fisher-z magnitudes are not comparable across panels)."""
    if not all(v in results for v in FIG_VM):
        missing = [v for v in FIG_VM if v not in results]
        print(f"[fig] missing panels {missing}; skip headline figure", flush=True)
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    ax = axes.flatten()
    panels = [(EMP, 'Empirical FC\n(HCP rs-fMRI, Fisher-z)',
               float(np.max(EMP)))]
    for vm in FIG_VM:
        M = results[vm]['FC_z']
        r = results[vm]['r_a']
        title = (rf'Simulated FC, $\gamma$ $v_m$={vm:.0f} m/s (Fisher-z)'
                 + '\n'
                 + rf'r = {r:+.3f}')
        panels.append((M, title, float(np.nanmax(M))))
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
    print(f"[fig] saved {PNG}", flush=True)


def main():
    os.makedirs(RES_DIR, exist_ok=True)
    os.makedirs(PLT_DIR, exist_ok=True)
    print(f"=== c3_gamma_vm_sweep eta_E={ETA_E}, s={S_FIXED}, "
          f"gamma-velocity ===", flush=True)
    print(f"vm grid: {VM_GRID}", flush=True)

    results = load_existing(NPZ)
    if results:
        print(f"resuming from {NPZ}: already have "
              f"{sorted(results.keys())}", flush=True)

    t0_total = time.time()
    for vm in VM_GRID:
        if vm in results:
            print(f"\n--- vm={vm} m/s already done "
                  f"(r={results[vm]['r_a']:+.4f}); skip", flush=True)
            continue
        try:
            FC_z, mZ, r, lo, hi = run_one(float(vm))
            results[float(vm)] = dict(FC_z=FC_z, mean_Z=mZ,
                                      r_a=r, lo=lo, hi=hi)
            done = sorted(results.keys())
            save_inc(done, results)
        except Exception as e:
            print(f"  !! vm={vm} FAILED: {e}", flush=True)

    make_headline_figure(results)

    print(f"\n=== r vs vm summary ===", flush=True)
    for vm in VM_GRID:
        if vm in results:
            r = results[vm]
            print(f"  vm={vm:5.1f} m/s  r={r['r_a']:+.4f} "
                  f"[{r['lo']:+.4f}, {r['hi']:+.4f}]", flush=True)
    print(f"\n=== DONE in {(time.time()-t0_total)/60:.1f} min ===",
          flush=True)


if __name__ == '__main__':
    main()
