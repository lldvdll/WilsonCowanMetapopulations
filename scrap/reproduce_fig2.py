"""
Reproduce Figure 2 from Conti & Van Gorder (2018) using model.py.

2-node Wilson-Cowan model with paper's stated parameters:
  c_ee = c_ei = 1, c_ie = c_ii = -1
  P = Q = 0.5, k = 1, beta = 10, alpha = 0.6, T_e = T_i = 1
  W = [[0,1],[1,0]], rho = 10
  IC: E(0) = I(0) = 0.5

Panel (a): tau1=1, tau2=1.4
Panel (b): tau1=4, tau2=40
Panel (c): chaotic regime - c_ee=c_ii=-6, c_ei=c_ie=2.5,
           P=Q=0.2, tau1=1, tau2=1.4, k=11
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib.pyplot as plt
from model import Model
from network import Network


def run_panel(params, duration=100, dt=0.01):
    """Run a 2-node simulation using Model class."""
    net = Network({"topology": "full", "N": 2})
    m = Model(net, params)
    m.set_time_grid(duration=duration, dt=dt)
    m.run()
    return m.time_array, m.trajectories


def plot_panel(ax, time_array, traj, title_label):
    """Plot E1, I1, E2, I2 matching paper's style."""
    ax.plot(time_array, traj[0, 0, :], 'b-',  linewidth=1.2, label=r'$E_1(t)$')
    ax.plot(time_array, traj[1, 0, :], 'k-',  linewidth=1.2, label=r'$I_1(t)$')
    ax.plot(time_array, traj[0, 1, :], 'b--', linewidth=1.2, label=r'$E_2(t)$')
    ax.plot(time_array, traj[1, 1, :], 'k--', linewidth=1.2, label=r'$I_2(t)$')
    ax.set_xlabel('t', fontsize=13)
    ax.set_xlim(0, time_array[-1])
    ax.set_ylim(0, 1)
    ax.legend(loc='center right', fontsize=10)
    ax.set_title(title_label, fontsize=14, y=-0.15)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel (a): tau1=1, tau2=1.4
    print("Panel (a): tau1=1, tau2=1.4 ...")
    params_a = {
        "c_ee": 1, "c_ei": 1, "c_ie": -1, "c_ii": -1,
        "tau_1": 1.0, "tau_2": 1.4, "rho": 10.0,
        "beta": 10.0, "alpha": 0.6, "T_e": 1.0, "T_i": 1.0,
        "P": 0.5, "Q": 0.5, "k": 1.0,
    }
    ta, tra = run_panel(params_a)
    plot_panel(axes[0], ta, tra, '(a)')

    # Panel (b): tau1=4, tau2=40
    print("Panel (b): tau1=4, tau2=40 ...")
    params_b = {**params_a, "tau_1": 4.0, "tau_2": 40.0}
    tb, trb = run_panel(params_b)
    plot_panel(axes[1], tb, trb, '(b)')

    # Panel (c): chaotic regime
    print("Panel (c): chaotic regime ...")
    params_c = {
        "c_ee": -6, "c_ei": 2.5, "c_ie": 2.5, "c_ii": -6,
        "tau_1": 1.0, "tau_2": 1.4, "rho": 10.0,
        "beta": 10.0, "alpha": 0.6, "T_e": 1.0, "T_i": 1.0,
        "P": 0.2, "Q": 0.2, "k": 11.0,
    }
    tc, trc = run_panel(params_c)
    plot_panel(axes[2], tc, trc, '(c)')

    plt.suptitle('Figure 2: 2-node Wilson-Cowan model (using model.py)\n'
                 'Paper parameters: $c_{ee}=c_{ei}=1,\\ c_{ie}=c_{ii}=-1$',
                 fontsize=13)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'fig2_reproduction.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved to fig2_reproduction.png")

    # Print final values
    print(f"\nPanel (a) final: E1={tra[0,0,-1]:.4f}, I1={tra[1,0,-1]:.4f}, "
          f"E2={tra[0,1,-1]:.4f}, I2={tra[1,1,-1]:.4f}")
    print(f"Panel (b) final: E1={trb[0,0,-1]:.4f}, I1={trb[1,0,-1]:.4f}, "
          f"E2={trb[0,1,-1]:.4f}, I2={trb[1,1,-1]:.4f}")
    print(f"Panel (c) final: E1={trc[0,0,-1]:.4f}, I1={trc[1,0,-1]:.4f}, "
          f"E2={trc[0,1,-1]:.4f}, I2={trc[1,1,-1]:.4f}")


if __name__ == "__main__":
    main()
