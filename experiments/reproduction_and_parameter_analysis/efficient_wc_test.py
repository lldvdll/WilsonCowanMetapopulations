"""
Test wilson_cowan_efficient method, a vectorised python native implementation

Not sure what's happened to the jitcdde method, but it seems pretty broken. 

"""

import os
import time
from src.metapopulation import Metapopulation
        
# Example run
model = Metapopulation()
model.load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), "efficient_wc_test.yaml"))
model.create_network()
# model.network.plot_adjacency_matrix()

# Normal
model.initialise_model({"mode": "wilson_cowan"})
model.run_simulation(timeit=True)
model.plot_trajectories()

# Efficient WC
model.initialise_model({"mode": "wilson_cowan_efficient"})
model.run_simulation(timeit=True)
model.plot_trajectories()