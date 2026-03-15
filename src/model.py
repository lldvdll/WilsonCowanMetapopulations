"""
Excitation Inhibition Models
    - Wilson-Cowan model
    
Add a model by creating a function. It should follow this structure:
    - Arguments: 
    - Returns:
"""

import numpy as np

class Model:
    
    def __init__(self, network, params=None):
        self.network = network
        self.params = params
        self.time_array = None  # Time grid
        self.trajectories = None  # Reulsting simulated trajectories 2xNxT
        self.process_mode()
        
    def process_mode(self):
        self.mode = self.params['mode']
        if self.mode == 'wilson_cowan':
            self.C = 2  # Number of model components
            self.components = ['Excitatory', 'Inhibitory']
        
    def set_time_grid(self, duration=1000, dt=0.1):
        self.time_array = np.arange(0, duration, dt)
    
    def set_initial_conditions(self, N):
        self.trajectories = np.arange(self.C, N)
        
    def run(self):
        # TODO: Implement the DDE solver and call wilson_cowan
        
        # Dummy simulation just generates random sine waves
        N = self.network.N 
        C = self.C
        T = len(self.time_array)
        freqs = np.random.uniform(0.001, 0.01, (C, N, 1))
        phases = np.random.uniform(0, 2 * np.pi, (C, N, 1))
        self.trajectories = np.sin(2 * np.pi * freqs * self.time_array + phases)
    
    def wilson_cowan(self, state, params=None):
        # TODO: Implement the wilson_cowan model
        return None