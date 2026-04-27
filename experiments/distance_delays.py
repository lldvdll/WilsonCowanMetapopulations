"""
Distance-based delay experiment.
rho_nj = d_nj / v where d_nj is shortest-path hop count.

Two velocities are compared:
    v=1: delay equals hop count (slow, large delays)
    v=6: faster conduction, smaller delays

Four topologies: line, full, ring, lattice

Analysis:
    - Trajectory plots
    - Mean delay matrix per topology and velocity
    - Synchrony index: mean over time of variance across nodes
    - Dominant frequency: peak of FFT of excitatory trajectories
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("/Users/kavya/Documents/GitHub/WilsonCowanMetapopulations")

from src.metapopulation import Metapopulation

TOPOLOGIES = ['line', 'full', 'ring', 'lattice']
VELOCITIES = [0.5, 0.3]
CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "distance_delays.yaml")


def run(topology, v):
    model = Metapopulation()
    model.load_config(CONFIG)
    model.create_network(params={'topology': topology})
    model.create_delay_matrix(mode='distance', v=v, target_mean_rho=10.0)
    model.initialise_model()
    model.run_simulation()
    return model


def synchrony_index(trajectories):
    """
    Mean over time of variance across nodes.
    Low value = high synchrony.
    Computed on excitatory population only.
    """
    E = trajectories[0]  # (N, T)
    return float(np.mean(np.var(E, axis=0)))


def dominant_frequency(trajectories, dt):
    """
    Mean dominant frequency across nodes (Hz).
    Computed on excitatory population via FFT.
    """
    E = trajectories[0]  # (N, T)
    freqs = np.fft.rfftfreq(E.shape[1], d=dt)
    fft_mean = np.mean(np.abs(np.fft.rfft(E, axis=1)), axis=0)
    return float(freqs[np.argmax(fft_mean)])


def mean_delay(delay_matrix):
    N = delay_matrix.shape[0]
    mask = ~np.eye(N, dtype=bool)
    return float(np.mean(delay_matrix[mask]))


# Run all combinations and collect results
results = {}
for topology in TOPOLOGIES:
    for v in VELOCITIES:
        print(f"Running {topology}, v={v}...")
        m = run(topology, v)
        D = m.model.params['rho']
        results[(topology, v)] = {
            'time': m.model.time_array,
            'trajectories': m.model.trajectories,
            'mean_delay': mean_delay(D),
            'synchrony': synchrony_index(m.model.trajectories),
            'dom_freq': dominant_frequency(m.model.trajectories, m.model.dt)
        }

# Trajectory plots
fig, axes = plt.subplots(len(TOPOLOGIES), len(VELOCITIES), figsize=(12, 10))
fig.suptitle("Distance-based delays: trajectories")

for i, topology in enumerate(TOPOLOGIES):
    for j, v in enumerate(VELOCITIES):
        r = results[(topology, v)]
        ax = axes[i, j]
        ax.plot(r['time'], r['trajectories'][0].T, color='blue', alpha=0.5, linewidth=0.8)
        ax.plot(r['time'], r['trajectories'][1].T, color='black', alpha=0.5, linewidth=0.8)
        ax.set_title(f"{topology} | v={v} | mean rho={r['mean_delay']:.2f}")
        ax.set_ylim(0, 1)
        if i == len(TOPOLOGIES) - 1:
            ax.set_xlabel("Time (ms)")
        if j == 0:
            ax.set_ylabel("Activity")

plt.tight_layout()
plt.show()

# Analysis summary plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Distance-based delays: analysis")

x = np.arange(len(TOPOLOGIES))
width = 0.35

for k, v in enumerate(VELOCITIES):
    sync = [results[(t, v)]['synchrony'] for t in TOPOLOGIES]
    freq = [results[(t, v)]['dom_freq'] for t in TOPOLOGIES]
    axes[0].bar(x + k*width, sync, width, label=f"v={v}")
    axes[1].bar(x + k*width, freq, width, label=f"v={v}")

axes[0].set_xticks(x + width/2)
axes[0].set_xticklabels(TOPOLOGIES)
axes[0].set_ylabel("Synchrony index (lower = more synchronised)")
axes[0].set_title("Synchrony")
axes[0].legend()

axes[1].set_xticks(x + width/2)
axes[1].set_xticklabels(TOPOLOGIES)
axes[1].set_ylabel("Dominant frequency")
axes[1].set_title("Dominant frequency")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "distance_delays_analysis.png"), dpi=800)
plt.show()

# Print summary table
print("\nSummary:")
print(f"{'Topology':<10} {'v':<5} {'Mean delay':<12} {'Synchrony':<12} {'Dom freq':<10}")
print("-" * 50)
for topology in TOPOLOGIES:
    for v in VELOCITIES:
        r = results[(topology, v)]
        print(f"{topology:<10} {v:<5} {r['mean_delay']:<12.3f} {r['synchrony']:<12.5f} {r['dom_freq']:<10.3f}")