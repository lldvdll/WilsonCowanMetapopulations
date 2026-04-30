"""c3_gamma_vm_sweep.py — Gamma-distributed velocity vm sweep at s=1.4.

Companion to the deterministic-velocity C3 sweep: tests whether the
shape of the BOLD-FC fit landscape changes when v is drawn from a
truncated γ-distribution rather than fixed deterministically.

Sweep: vm ∈ {6, 8, 10, 12, 15} m/s × η_E = -2.15 × s = 1.4 ×
       γ-velocity (Atay-Hutt p=4.5, vl=1, vh=20) × seed 42.
5 sims × 500 s ≈ 1.5 h wall.
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

ETA_E      = -2.15
S_FIXED    = 1.4
VM_GRID    = [6.0, 8.0, 10.0, 12.0, 15.0]
DURATION_S = 500.0
DT         = 0.001
SEED       = 42
N_BOOT     = 1000

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RES_DIR  = os.path.join(os.path.dirname(__file__), 'Results')
PLT_DIR  = os.path.join(os.path.dirname(__file__), 'Plots')
NPZ      = os.path.join(RES_DIR, 'c3_gamma_vm_sweep.npz')
PNG      = os.path.join(PLT_DIR, 'c3_gamma_vm_sweep.png')
PNG_FC   = os.path.join(PLT_DIR, 'c3_gamma_vm_FC_compare.png')
PNG_CMP  = os.path.join(PLT_DIR, 'c3_v_sweep_det_vs_gamma.png')

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


def run_one(vm):
    print(f"\n=== vm={vm} m/s, η={ETA_E}, s={S_FIXED}, γ-velocity === "
          f"[{time.strftime('%H:%M:%S')}]", flush=True)
    p = dict(eta_E=ETA_E, k_ext=0.2,
             delay_mode='matrix_gamma_velocity',
             distance_matrix=DIST * S_FIXED,
             kappa_v_EE=0.01, kappa_v_II=0.025,
             conduction_velocity=vm,
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
    BOLD_FC = m.compute_BOLD_FC(t_start=t_start)
    BOLD_FC_z = enigma_align(BOLD_FC)
    r_a = fc_corr(BOLD_FC_z, EMP)
    lo, hi = bootstrap_ci(BOLD_FC_z, EMP)
    print(f"  integ {time.time()-t0:.1f}s  |Z|={mean_Z:.4f}  PLV={mean_PLV:.4f}  "
          f"r={r_a:+.4f} [{lo:+.4f},{hi:+.4f}]", flush=True)
    return BOLD_FC_z, mean_Z, mean_PLV, r_a, lo, hi


def save_inc(done, results):
    out = dict(vm_grid=np.array(VM_GRID), eta_E=ETA_E, s_fixed=S_FIXED,
               duration_s=DURATION_S, seed=SEED,
               vm_done=np.array(done, dtype=float),
               r_aligned=np.array([results[v]['r_a']  for v in done]),
               r_lo=np.array([results[v]['lo']        for v in done]),
               r_hi=np.array([results[v]['hi']        for v in done]),
               mean_Z=np.array([results[v]['mean_Z']  for v in done]),
               mean_PLV=np.array([results[v]['mean_PLV'] for v in done]))
    for v in done:
        out[f'BOLD_FC_z_vm{v:.0f}'] = results[v]['FC_z']
    np.savez(NPZ, **out)


def make_plots(done, results):
    if not done: return
    iu = np.triu_indices(SC.shape[0], k=1)
    r_sc_emp = float(np.corrcoef(SC[iu], EMP[iu])[0, 1])
    vs = np.array(done)
    rs = np.array([results[v]['r_a'] for v in done])
    los = np.array([results[v]['lo'] for v in done])
    his = np.array([results[v]['hi'] for v in done])

    # γ-only plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(vs, los, his, color='#2ca02c', alpha=0.20,
                    label='95% bootstrap CI')
    ax.plot(vs, rs, 'o-', color='#2ca02c', lw=2, ms=8,
            label='r(simulated BOLD FC, empirical)')
    ax.axhline(r_sc_emp, color='#d62728', lw=1.3, ls='--',
               label=f'r(SC, empirical) = {r_sc_emp:.3f}')
    ax.axhline(0, color='grey', lw=0.5, ls=':')
    if len(rs):
        i_best = int(np.nanargmax(rs))
        ax.plot(vs[i_best], rs[i_best], 'o', mfc='none', mec='#d62728',
                ms=14, mew=2.0, label=f'best $v_m$ = {vs[i_best]:.0f} m/s')
    ax.set_xlabel(r'γ-distribution velocity mode $v_m$ (m/s)')
    ax.set_ylabel('r(simulated BOLD FC, empirical FC)')
    ax.set_title(rf'BOLD-FC fit vs $v_m$ (γ-velocity) at $\eta_E$={ETA_E}, '
                 rf'$s$={S_FIXED} (volume centroids)')
    ax.legend(loc='best', frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(PNG)
    plt.close(fig)

    # FC matrices comparison (vm=6 and vm=12)
    if all(v in results for v in [6.0, 12.0]):
        fc_vmax = float(max(np.max(results[6.0]['FC_z']),
                            np.max(results[12.0]['FC_z']),
                            np.max(EMP)))
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        panels = [
            (EMP, 'Empirical BOLD FC\n(ENIGMA HCP rs-fMRI, Fisher-z)'),
            (results[6.0]['FC_z'],
             f'Simulated BOLD FC, γ $v_m$=6 m/s\nr = {results[6.0]["r_a"]:+.3f}'),
            (results[12.0]['FC_z'],
             f'Simulated BOLD FC, γ $v_m$=12 m/s\nr = {results[12.0]["r_a"]:+.3f}'),
        ]
        for ax, (M, title) in zip(axes, panels):
            im = ax.imshow(M, cmap='viridis', vmin=0, vmax=fc_vmax,
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

    # Combined plot: deterministic vs γ
    det_npz_path = os.path.join(RES_DIR, 'c2_v_sweep_supercritical.npz')
    if os.path.exists(det_npz_path):
        d_det = np.load(det_npz_path)
        v_det = d_det['v_done']; r_det = d_det['r_aligned']
        lo_det = d_det['r_lo']; hi_det = d_det['r_hi']

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.fill_between(v_det, lo_det, hi_det, color='#1f77b4', alpha=0.20)
        ax.plot(v_det, r_det, 'o-', color='#1f77b4', lw=2, ms=8,
                label='deterministic v')
        ax.fill_between(vs, los, his, color='#2ca02c', alpha=0.20)
        ax.plot(vs, rs, 'o-', color='#2ca02c', lw=2, ms=8,
                label=r'γ-distributed velocity (mode $v_m$)')
        ax.axhline(r_sc_emp, color='#d62728', lw=1.3, ls='--',
                   label=f'r(SC, empirical) = {r_sc_emp:.3f}')
        ax.axhline(0, color='grey', lw=0.5, ls=':')
        ax.set_xlabel(r'Conduction velocity (m/s)')
        ax.set_ylabel('r(simulated BOLD FC, empirical FC)')
        ax.set_title(rf'Deterministic vs γ velocity at $\eta_E$={ETA_E}, '
                     rf'$s$={S_FIXED}, volume centroids')
        ax.legend(loc='best', frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(PNG_CMP)
        plt.close(fig)


def main():
    os.makedirs(RES_DIR, exist_ok=True)
    os.makedirs(PLT_DIR, exist_ok=True)
    print(f"=== c3_gamma_vm_sweep η={ETA_E}, s={S_FIXED}, γ-velocity ===",
          flush=True)
    print(f"start: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"vm grid: {VM_GRID}\n", flush=True)
    t0_total = time.time()

    results = {}
    done = []
    for vm in VM_GRID:
        try:
            FC_z, mZ, mP, r, lo, hi = run_one(float(vm))
            results[float(vm)] = dict(FC_z=FC_z, mean_Z=mZ, mean_PLV=mP,
                                     r_a=r, lo=lo, hi=hi)
            done.append(float(vm))
            save_inc(done, results); make_plots(done, results)
        except Exception as e:
            print(f"  !! vm={vm} FAILED: {e}", flush=True)

    print(f"\n=== r vs vm table ===", flush=True)
    for vm in VM_GRID:
        if vm in results:
            print(f"  vm={vm:5.1f} m/s  r={results[vm]['r_a']:+.4f} "
                  f"[{results[vm]['lo']:+.4f},{results[vm]['hi']:+.4f}]",
                  flush=True)
    print(f"\n=== DONE in {(time.time()-t0_total)/60:.1f} min ===", flush=True)


if __name__ == '__main__':
    main()
