"""c3_kappa_v_sweep_HCP.py — Gap-junction κ_v sweep on HCP at η_E=-2.15.

Sweeps κ_v scale (multiplier on Forrester κ_v_EE=0.01 and κ_v_II=0.025
defaults) to identify the operating κ_v* that maximises r(BOLD, empirical FC).

Sweep: κ_v_scale ∈ {0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0} × HCP-DK68
       × η_E = -2.15 (super-critical) × s = BEST_S (from C2)
       × Atay-Hutt Gamma delays (vm=12, p=4.5, vl=1, vh=20)

Delay choice: deterministic matrix (matches C2 calibration; gamma vs matrix
contrast is the dedicated subject of C5 — keeping C3/C4/C6 deterministic
matches Forrester p.8 preferred Gamma).

Outputs:
  Results/c3_kappa_v_sweep_HCP.npz   per-κv r/PLV/|Z|/dPLV σ/BOLD-FC
  Plots/c3_kappa_v_sweep_HCP.png     2-panel: r vs κ_v + dPLV σ vs κ_v
  Plots/c3_kappa_v_FC_compare.png    side-by-side FC matrix: best κ_v vs κ_v=0
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

ETA_E              = -2.15
BEST_S             = 1.4      # from C2 (best_s, r=+0.157)
KV_SCALE_GRID      = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0])
VM_FIXED           = 12.0  # main-line vm (Forrester representative)
DEFAULT_KV_EE      = 0.01     # Forrester
DEFAULT_KV_II      = 0.025    # Forrester
DURATION_S         = 500.0
DT                 = 0.001
SEED               = 42
N_BOOT             = 1000

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RES_DIR  = os.path.join(os.path.dirname(__file__), 'Results')
PLT_DIR  = os.path.join(os.path.dirname(__file__), 'Plots')
NPZ      = os.path.join(RES_DIR, 'c3_kappa_v_sweep_HCP.npz')
PNG_R    = os.path.join(PLT_DIR, 'c3_kappa_v_sweep_HCP.png')
PNG_FC   = os.path.join(PLT_DIR, 'c3_kappa_v_FC_compare.png')

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
    print(f"\n=== κ_v_scale = {kv_scale:.2f} === [{time.strftime('%H:%M:%S')}]",
          flush=True)
    p = dict(eta_E=ETA_E, k_ext=0.2,
             delay_mode='matrix_gamma_velocity',
             distance_matrix=DIST * BEST_S,
             kappa_v_EE=DEFAULT_KV_EE * float(kv_scale),
             kappa_v_II=DEFAULT_KV_II * float(kv_scale),
             conduction_velocity=VM_FIXED,
             velocity_gamma_shape=4.5,
             velocity_truncate_low=1.0,
             velocity_truncate_high=20.0,
             seed=SEED)
    m = NextGenNetwork(HCPNet(SC), params=p)
    t0 = time.time()
    m.run(duration=DURATION_S, dt=DT)
    t_start = max(30.0, DURATION_S * 0.2)

    Z_E, _ = m.compute_Z()
    half = Z_E.shape[1] // 2
    mean_Z = float(np.abs(Z_E[:, half:]).mean())
    PLV = m.compute_PLV(t_start=t_start)
    iu = np.triu_indices(SC.shape[0], k=1)
    mean_PLV = float(PLV[iu].mean())

    # dPLV σ via sliding-window PLV temporal SD
    dPLV, _ = m.compute_dynamic_PLV(window_sec=10.0, overlap=0.9, t_start=t_start)
    edge_ts = dPLV[:, iu[0], iu[1]]
    dPLV_sigma = float(edge_ts.std(axis=0).mean())

    BOLD_FC = m.compute_BOLD_FC(t_start=t_start)
    BOLD_FC_z = enigma_align(BOLD_FC)
    r_aligned = fc_corr(BOLD_FC_z, EMP)
    r_raw     = fc_corr(BOLD_FC,   EMP)
    lo, hi    = bootstrap_ci(BOLD_FC_z, EMP)
    print(f"  integ {time.time()-t0:.1f}s  |Z|={mean_Z:.4f}  PLV={mean_PLV:.4f}  "
          f"dPLVσ={dPLV_sigma:.4f}  r={r_aligned:+.4f} [{lo:+.4f},{hi:+.4f}]",
          flush=True)
    return BOLD_FC_z, mean_Z, mean_PLV, dPLV_sigma, r_aligned, r_raw, lo, hi


def save_inc(done, results):
    out = dict(kv_scale_done=np.array(done, dtype=float),
               kv_scale_grid=KV_SCALE_GRID,
               eta_E=ETA_E, best_s=BEST_S,
               kv_EE_default=DEFAULT_KV_EE, kv_II_default=DEFAULT_KV_II,
               duration_s=DURATION_S, seed=SEED,
               mean_Z=np.array([results[k]['mean_Z']    for k in done]),
               mean_PLV=np.array([results[k]['mean_PLV'] for k in done]),
               dPLV_sigma=np.array([results[k]['dPLV']  for k in done]),
               r_aligned=np.array([results[k]['r_a']    for k in done]),
               r_raw=np.array([results[k]['r_r']        for k in done]),
               r_lo=np.array([results[k]['lo']          for k in done]),
               r_hi=np.array([results[k]['hi']          for k in done]))
    for k in done:
        out[f'BOLD_FC_z_kv{k:.2f}'] = results[k]['FC_z']
    np.savez(NPZ, **out)


def make_plots(done, results):
    if not done: return
    k = np.array(done)
    r  = np.array([results[v]['r_a'] for v in done])
    lo = np.array([results[v]['lo']  for v in done])
    hi = np.array([results[v]['hi']  for v in done])
    dplv = np.array([results[v]['dPLV'] for v in done])
    iu = np.triu_indices(SC.shape[0], k=1)
    r_sc_emp = float(np.corrcoef(SC[iu], EMP[iu])[0, 1])

    # 2-panel: r vs κ_v + dPLV σ vs κ_v
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    ax.fill_between(k, lo, hi, color='#1f77b4', alpha=0.20, label='95% bootstrap CI')
    ax.plot(k, r, 'o-', color='#1f77b4', lw=2, ms=7,
            label='r(simulated BOLD FC, empirical)')
    ax.axhline(r_sc_emp, color='#d62728', lw=1.3, ls='--',
               label=f'r(SC, empirical) = {r_sc_emp:.3f}')
    if len(r):
        i_best = int(np.nanargmax(r))
        ax.plot(k[i_best], r[i_best], 'o', mfc='none', mec='#d62728',
                ms=14, mew=2.0, label=f'best κ_v_scale = {k[i_best]:.2f}')
    ax.set_xlabel(r'$\kappa_v$ scale (× Forrester defaults)')
    ax.set_ylabel('r(simulated BOLD FC, empirical FC)')
    ax.set_title(rf'(a) BOLD-FC fit vs $\kappa_v$ at $\eta_E$ = {ETA_E}')
    ax.legend(loc='best', fontsize=9)

    ax = axes[1]
    ax.plot(k, dplv, 'o-', color='#2ca02c', lw=2, ms=7)
    ax.set_xlabel(r'$\kappa_v$ scale (× Forrester defaults)')
    ax.set_ylabel(r'dPLV $\sigma$ (sliding-window temporal SD)')
    ax.set_title(r'(b) Dynamic FC stability (lower = more stable)')
    fig.tight_layout()
    fig.savefig(PNG_R)
    plt.close(fig)

    # Side-by-side FC matrix: best κ_v vs κ_v=0
    if 0.0 in results and len(done) > 1:
        i_best = int(np.nanargmax([results[v]['r_a'] for v in done]))
        kv_best = done[i_best]
        fc_off = results[0.0]['FC_z']
        fc_best = results[kv_best]['FC_z']
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, M, title in zip(axes,
                                 [fc_off, fc_best, EMP],
                                 [rf'Simulated FC without gap junction ($\kappa_v$=0): r={results[0.0]["r_a"]:+.3f}',
                                  rf'Simulated FC with gap junction ($\kappa_v$_scale={kv_best:.2f}): r={results[kv_best]["r_a"]:+.3f}',
                                  'Empirical FC (HCP rs-fMRI)']):
            im = ax.imshow(M, cmap='viridis', vmin=0, vmax=float(np.nanmax(M)),
                           interpolation='nearest')
            ax.axhline(33.5, color='black', lw=0.5)
            ax.axvline(33.5, color='black', lw=0.5)
            ax.set_title(title)
            ax.set_xlabel('Region (DK68 index)')
            ax.set_ylabel('Region (DK68 index)' if ax is axes[0] else '')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        fig.tight_layout()
        fig.savefig(PNG_FC)
        plt.close(fig)


def main():
    os.makedirs(RES_DIR, exist_ok=True)
    os.makedirs(PLT_DIR, exist_ok=True)
    print(f"=== c3_kappa_v_sweep_HCP η_E={ETA_E}, s={BEST_S} ===", flush=True)
    print(f"start: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"κ_v_scale grid: {KV_SCALE_GRID.tolist()}", flush=True)

    results = {}
    done = []
    t0_total = time.time()
    for k in KV_SCALE_GRID:
        try:
            FC_z, mZ, mP, dplv, r_a, r_r, lo, hi = run_one(float(k))
            results[float(k)] = dict(FC_z=FC_z, mean_Z=mZ, mean_PLV=mP,
                                     dPLV=dplv, r_a=r_a, r_r=r_r, lo=lo, hi=hi)
            done.append(float(k))
            save_inc(done, results); make_plots(done, results)
        except Exception as e:
            print(f"  !!! κ_v_scale={k} FAILED: {e}", flush=True)
    print(f"\n=== DONE: {len(done)}/{len(KV_SCALE_GRID)} sims in "
          f"{(time.time()-t0_total)/60:.1f} min ===", flush=True)


if __name__ == '__main__':
    main()
