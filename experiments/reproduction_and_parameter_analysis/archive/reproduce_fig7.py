"""
Reproduce parameter sweep from Figure 7 of Conti & Van Gorder (2018).
Maps the presence of limit cycles across varying delays.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import sys

# Ensure we can import from src
script_path = os.path.abspath(__file__)
src_path = os.path.dirname(os.path.dirname(script_path))
sys.path.insert(0, src_path)

from src.metapopulation import Metapopulation

def check_for_oscillations(trajectory, tail_length=300):
    """
    Analyzes the end of the simulation to see if it is oscillating.
    Returns the peak-to-peak amplitude.
    """
    # Grab the excitatory trajectory of Node 0 over the last 'tail_length' steps
    node_0_E = trajectory[0, 0, -tail_length:]
    amplitude = np.max(node_0_E) - np.min(node_0_E)
    return amplitude

# 1. Setup Orchestrator
model = Metapopulation()
config_path = os.path.join(os.path.dirname(script_path), "reproduce_fig7.yaml")
model.load_config(config_path)

# Test on the cycle graph
model.create_network()

# 2. Define the Parameter Grid (Fig 7 sweeps tau from 0.1-20, rho from 0.1-40)
grid_size = 8  # 15x15 grid = 225 simulations. Increase for higher res.
tau_values = np.linspace(0.1, 20.0, grid_size)
rho_values = np.linspace(0.1, 40.0, grid_size)

# Container for results
oscillation_map = np.zeros((grid_size, grid_size))

print(f"Starting 2D parameter sweep ({grid_size * grid_size} runs)...")
start_time = time.time()

# 3. Execute the Sweep
for i, rho_val in enumerate(rho_values):
    for j, tau_val in enumerate(tau_values):
        
        # Inject the new delays. (Fig 7 sets tau_1 = tau_2 = tau)
        model.initialise_model(params={
            'tau_1': tau_val, 
            'tau_2': tau_val, 
            'rho': rho_val
        })
        
        # Run the simulation (Ensure jitcdde simplify=False is on in model.py!)
        model.run_simulation()
        
        # Check for limit cycles
        amp = check_for_oscillations(model.model.trajectories)
        oscillation_map[i, j] = amp
        
        print(f"Tested rho={rho_val:.1f}, tau={tau_val:.1f} | Amplitude: {amp:.3f}")

print(f"Sweep complete in {(time.time() - start_time):.1f} seconds.")

# 4. Plot the Limit Cycle Heatmap
plt.figure(figsize=(10, 8))
# extent=[x_min, x_max, y_min, y_max]
plt.imshow(oscillation_map, origin='lower', aspect='auto', cmap='plasma',
           extent=[tau_values[0], tau_values[-1], rho_values[0], rho_values[-1]])

plt.colorbar(label='Oscillation Amplitude (0 = Steady State)')
plt.xlabel(r'Intra-node delay ($\tau$)')
plt.ylabel(r'Inter-node delay ($\rho$)')
plt.title('Figure 7: Limit Cycle Search Space (Cycle Graph)')

# Add contour line to strictly separate steady states (amp < 0.05) from limit cycles
X, Y = np.meshgrid(tau_values, rho_values)
plt.contour(X, Y, oscillation_map, levels=[0.05], colors='white', linestyles='dashed')

plt.tight_layout()
plt.show()