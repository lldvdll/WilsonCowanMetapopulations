"""
Reproduce  figure 2 from Conti & Van Gorder (2018) using metapopulation.py.

Fig. 2. Numerical solutions for the 2-node Wilson–Cowan model. 
Resulting dynamics are shown for 
    (a) τn,1 = 1, τn,2 = 1.4, 
    (b) τn,1 = 4, τn,2 = 40. 
Other parameter values are cee,n = cei,n = 1, cie,n = cii,n = −1, kn = 1, and Pn = Qn = 0.5, for n = 1, 2, while ρ1,2 = 10. 
    In (c) we instead show a solution curve corresponding to the chaotic parameter regime cee,n = cii,n = −6, cei,n = cie,n = 2.5, Pn = Qn = 0.2, τn,1 = 1, τn,2 = 1.4, k1 = k2 = 11. 
The blue solid curves correspond to the activities of the excitatory subpopulations (E), while the black dashed curves correspond to the activities of the inhibitory subpopulations (I)

"""

import os
from src.metapopulation import Metapopulation
        
# Example run
model = Metapopulation()
model.load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), "reproduce_fig2.yaml"))
model.create_network()
# model.network.plot_adjacency_matrix()

# Figure 2 (a)
model.initialise_model()
model.run_simulation(duration=100, dt=0.1)
model.plot_trajectories()

# Figure 2 (b)
model.initialise_model(params={'tau_1': 4, 'tau_2': 40})
model.run_simulation(duration=100, dt=0.1)
model.plot_trajectories()

# Figure 2 (c)
model.initialise_model(params={'c_ee': -6, 'c_ii': -6, 'c_ei': 2.5, 'c_ie': 2.5, 'P': 0.2, 'Q': 0.2, 'tau_1': 1, 'tau_2': 1.4, 'k': 11})
model.run_simulation(duration=100, dt=0.1)
model.plot_trajectories()