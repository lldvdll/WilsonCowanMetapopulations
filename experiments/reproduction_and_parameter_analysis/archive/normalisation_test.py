"""
Comprehensive Normalization Test (Zero-Delay Saturation)
"""

import os
from src.metapopulation import Metapopulation
import matplotlib.pyplot as plt

# 1. Setup Orchestrator
model = Metapopulation()
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "normalisation_test.yaml")
model.load_config(config_path)

print("==================================================")
print("TEST A: NORMALIZATION ON")
print("==================================================")
# Force normalization ON
model.create_network(params={'normalise': True}) 
model.initialise_model()
model.run_simulation(duration=20, dt=0.1)

# Plot Test A
print("Plotting Test A... Close the plot window to run Test B.")
plt.figure(figsize=(8, 5))
for i, c in enumerate(model.model.components):
    t = model.model.time_array
    y = model.model.trajectories[i]
    colour = 'blue' if c == 'Excitatory' else 'black'
    plt.plot(t, y.T, color=colour, alpha=0.5, linewidth=1)
    plt.plot(0, 0, color=colour, label=c) # Legend item      
plt.title("Test A: Normalization ON (Balanced Steady State)")
plt.legend()
plt.ylim(-0.1, 1.1)
plt.show() 


print("\n==================================================")
print("TEST B: NORMALIZATION OFF")
print("==================================================")
# Force normalization OFF
model.create_network(params={'normalise': False}) 
model.initialise_model()
model.run_simulation(duration=20, dt=0.1)

# Plot Test B
print("Plotting Test B...")
plt.figure(figsize=(8, 5))
for i, c in enumerate(model.model.components):
    t = model.model.time_array
    y = model.model.trajectories[i]
    colour = 'blue' if c == 'Excitatory' else 'black'
    plt.plot(t, y.T, color=colour, alpha=0.5, linewidth=1)
    plt.plot(0, 0, color=colour, label=c) # Legend item      
plt.title("Test B: Normalization OFF (Ceiling Saturation!)")
plt.legend()
plt.ylim(-0.1, 1.1)
plt.show()